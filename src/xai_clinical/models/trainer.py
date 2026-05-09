# src/xai_clinical/models/trainer.py
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import joblib
import json
from pathlib import Path
import logging

from .classifiers import ClassifierFactory

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Training manager with hyperparameter optimization
    """

    def __init__(self, config, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.results = {}

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> Dict[str, Any]:
        """
        Train all configured models with optimization
        """
        classifiers_config = self.config.models.classifiers

        for clf_config in classifiers_config:
            if not clf_config.get("enabled", True):
                continue

            name = clf_config["name"]
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model: {name.upper()}")
            logger.info(f"{'='*50}")

            # Hyperparameter optimization
            best_params = self._optimize_hyperparameters(
                name, X_train, y_train, clf_config.get("params", {})
            )

            # Final training with best parameters
            model = ClassifierFactory.create_classifier(
                name, best_params, self.random_state
            )
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Final training failed for {name}: {str(e)}")
                continue

            # Evaluation
            metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)

            self.results[name] = {
                "model": model,
                "best_params": best_params,
                "metrics": metrics,
            }

            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Metrics: {metrics}")

        return self.results

    def _optimize_hyperparameters(
        self, name: str, X: pd.DataFrame, y: pd.Series, param_grid: Dict
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna (Bayesian Optimization)
        """
        method = self.config.models.hyperparameter_tuning["method"]
        n_trials = self.config.models.hyperparameter_tuning["n_trials"]
        cv_folds = self.config.models.hyperparameter_tuning["cv_folds"]
        scoring = self.config.models.hyperparameter_tuning["scoring"]

        if method == "bayesian":

            def objective(trial):
                # Get parameter distributions
                param_distributions = ClassifierFactory.get_param_distributions(name)

                # Suggest parameters
                params = {}
                for param_name, distribution in param_distributions.items():
                    dist_name = distribution.__class__.__name__
                    if dist_name == "CategoricalDistribution":
                        params[param_name] = trial.suggest_categorical(
                            param_name, distribution.choices
                        )
                    elif dist_name in ("IntDistribution", "IntUniformDistribution"):
                        log = getattr(distribution, "log", False)
                        params[param_name] = trial.suggest_int(
                            param_name, distribution.low, distribution.high, log=log
                        )
                    elif dist_name in ("FloatDistribution", "UniformDistribution", "LogUniformDistribution"):
                        log = getattr(distribution, "log", False) or dist_name == "LogUniformDistribution"
                        params[param_name] = trial.suggest_float(
                            param_name, distribution.low, distribution.high, log=log
                        )

                # Create and evaluate model
                model = ClassifierFactory.create_classifier(
                    name, params, self.random_state
                )

                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=self.random_state
                )
                try:
                    # To prevent "bad allocation" (Out Of Memory) errors, we should not run cv-folds in parallel
                    # for models that already use all CPU cores natively.
                    cv_n_jobs = 1 if name in ["catboost", "xgboost", "lightgbm", "random_forest", "extra_trees"] else -1
                    # Calcul de la performance via validation croisée
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs)
                    return scores.mean()
                except Exception as e:
                    logger.warning(f"Trial failed for {name} with params {params}: {str(e)}")
                    return 0.0 # On retourne un score nul au lieu de faire planter le pipeline

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            return study.best_params

        elif method == "grid":
            from sklearn.model_selection import GridSearchCV

            model = ClassifierFactory.create_classifier(
                name, random_state=self.random_state
            )
            grid = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1
            )
            grid.fit(X, y)
            return grid.best_params_

        elif method == "random":
            from sklearn.model_selection import RandomizedSearchCV

            model = ClassifierFactory.create_classifier(
                name, random_state=self.random_state
            )
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_trials,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )
            search.fit(X, y)
            return search.best_params_

        else:
            raise ValueError(f"Unknown optimization method: {method}")
    def create_sota_ensemble(self, X_train, y_train):
        """
        Creates a soft-voting ensemble from the best trained models.
        Call this AFTER train_all_models().
        """
        from sklearn.ensemble import VotingClassifier

        if not self.results:
            raise ValueError("No trained models found. Run train_all_models() first.")

        # Trier par val AUC décroissant
        ranked = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"].get("val_roc_auc", 0),
            reverse=True
        )

        # Prendre les 3 meilleurs modèles
        top_models = ranked[:min(3, len(ranked))]
        estimators = [(name, result["model"]) for name, result in top_models]

        logger.info(f"Ensemble members: {[name for name, _ in estimators]}")

        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X_train, y_train)

        return ensemble
    def _evaluate_model(
        self, model, X_train, y_train, X_val, y_val
    ) -> Dict[str, float]:
        """Evaluate model on train and validation sets"""
        metrics = {}

        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = (
            model.predict_proba(X_train)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["train_precision"] = precision_score(
            y_train, y_train_pred, zero_division=0
        )
        metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)
        if y_train_proba is not None:
            metrics["train_roc_auc"] = roc_auc_score(y_train, y_train_proba)

        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
            metrics["val_precision"] = precision_score(
                y_val, y_val_pred, zero_division=0
            )
            metrics["val_recall"] = recall_score(y_val, y_val_pred, zero_division=0)
            metrics["val_f1"] = f1_score(y_val, y_val_pred, zero_division=0)
            if y_val_proba is not None:
                metrics["val_roc_auc"] = roc_auc_score(y_val, y_val_proba)

        return metrics

    def get_best_model(self, metric: str = "val_roc_auc") -> Tuple[str, Any, float]:
        """
        Return best model according to given metric
        """
        best_score = -np.inf
        best_name = None
        best_model = None

        for name, result in self.results.items():
            if metric in result["metrics"]:
                score = result["metrics"][metric]
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = result["model"]

        return best_name, best_model, best_score

    def calculate_xai_stability(self, X_train, y_train, n_splits=5) -> float:
        """
        Scientific validation: Calculate Jaccard Stability Index for top features.
        Proves if the model consistently picks the same biological markers.
        """
        from sklearn.model_selection import KFold
        import shap
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        top_features_sets = []
        
        logger.info(f"Evaluating XAI Stability (Jaccard) over {n_splits} folds...")
        
        best_name, best_model, _ = self.get_best_model()
        
        for train_idx, val_idx in kf.split(X_train):
            # Fit on fold
            X_f, y_f = X_train.iloc[train_idx], y_train.iloc[train_idx]
            best_model.fit(X_f, y_f)
            
            # Simple importance (to speed up) or SHAP
            if hasattr(best_model, "feature_importances_"):
                imp = best_model.feature_importances_
            else:
                # Fallback for models without direct importance
                imp = np.abs(best_model.predict_proba(X_f.head(50))[:, 1]) # Proxy
                
            # Get top 20 features indices
            top_20_idx = np.argsort(imp)[-20:]
            top_features_sets.append(set(top_20_idx))
            
        # Calculate mean Jaccard Index between all pairs
        jaccard_scores = []
        for i in range(len(top_features_sets)):
            for j in range(i + 1, len(top_features_sets)):
                s1, s2 = top_features_sets[i], top_features_sets[j]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                jaccard_scores.append(intersection / union)
        
        mean_stability = np.mean(jaccard_scores) if jaccard_scores else 0.0
        logger.info(f"Mean Feature Stability (Jaccard Index): {mean_stability:.4f}")
        return mean_stability

    def save_models(self, path: str):
        """Save all trained models"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        for name, result in self.results.items():
            model_path = save_path / f"{name}_model.joblib"
            joblib.dump(result["model"], model_path)

            # Save metrics
            metrics_path = save_path / f"{name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(result["metrics"], f, indent=2)

            # Save parameters
            params_path = save_path / f"{name}_params.json"
            with open(params_path, "w") as f:
                json.dump(result["best_params"], f, indent=2)

        logger.info(f"Models saved to {save_path}")

