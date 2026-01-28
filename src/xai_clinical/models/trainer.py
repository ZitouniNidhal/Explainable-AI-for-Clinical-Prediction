# src/xai_clinical/models/trainer.py
"""
Model training and optimization
"""
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train all configured models with optimization
        """
        classifiers_config = self.config.models.classifiers
        
        for clf_config in classifiers_config:
            if not clf_config.get('enabled', True):
                continue
                
            name = clf_config['name']
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model: {name.upper()}")
            logger.info(f"{'='*50}")
            
            # Hyperparameter optimization
            best_params = self._optimize_hyperparameters(
                name, X_train, y_train, clf_config['params']
            )
            
            # Final training with best parameters
            model = ClassifierFactory.create_classifier(name, best_params, self.random_state)
            model.fit(X_train, y_train)
            
            # Evaluation
            metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
            
            self.results[name] = {
                'model': model,
                'best_params': best_params,
                'metrics': metrics
            }
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Metrics: {metrics}")
            
        return self.results
    
    def _optimize_hyperparameters(self, name: str, X: pd.DataFrame, y: pd.Series,
                                 param_grid: Dict) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna (Bayesian Optimization)
        """
        method = self.config.models.hyperparameter_tuning['method']
        n_trials = self.config.models.hyperparameter_tuning['n_trials']
        cv_folds = self.config.models.hyperparameter_tuning['cv_folds']
        scoring = self.config.models.hyperparameter_tuning['scoring']
        
        if method == "bayesian":
            def objective(trial):
                # Get parameter distributions
                param_distributions = ClassifierFactory.get_param_distributions(name)
                
                # Suggest parameters
                params = {}
                for param_name, distribution in param_distributions.items():
                    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                        params[param_name] = trial.suggest_categorical(param_name, distribution.choices)
                    elif isinstance(distribution, optuna.distributions.IntUniformDistribution):
                        params[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high)
                    elif isinstance(distribution, optuna.distributions.LogUniformDistribution):
                        params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high, log=True)
                    elif isinstance(distribution, optuna.distributions.UniformDistribution):
                        params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high)
                
                # Create and evaluate model
                model = ClassifierFactory.create_classifier(name, params, self.random_state)
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                
                return scores.mean()
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            return study.best_params
            
        elif method == "grid":
            from sklearn.model_selection import GridSearchCV
            model = ClassifierFactory.create_classifier(name, random_state=self.random_state)
            grid = GridSearchCV(model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
            grid.fit(X, y)
            return grid.best_params_
            
        elif method == "random":
            from sklearn.model_selection import RandomizedSearchCV
            model = ClassifierFactory.create_classifier(name, random_state=self.random_state)
            search = RandomizedSearchCV(model, param_grid, n_iter=n_trials, cv=cv_folds,scoring=scoring, n_jobs=-1, random_state=self.random_state)
            search.fit(X, y)
            return search.best_params_
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _evaluate_model(self, model, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        """Evaluate model on train and validation sets"""
        metrics = {}
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_precision'] = precision_score(y_train, y_train_pred, zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics['train_f1'] = f1_score(y_train, y_train_pred, zero_division=0)
        if y_train_proba is not None:
            metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            metrics['val_precision'] = precision_score(y_val, y_val_pred, zero_division=0)
            metrics['val_recall'] = recall_score(y_val, y_val_pred, zero_division=0)
            metrics['val_f1'] = f1_score(y_val, y_val_pred, zero_division=0)
            if y_val_proba is not None:
                metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_proba)
        
        return metrics
    
    def get_best_model(self, metric: str = 'val_roc_auc') -> Tuple[str, Any]:
        """
        Return best model according to given metric
        """
        best_score = -np.inf
        best_name = None
        best_model = None
        
        for name, result in self.results.items():
            if metric in result['metrics']:
                score = result['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = result['model']
                    
        return best_name, best_model, best_score
    
    def save_models(self, path: str):
        """Save all trained models"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = save_path / f"{name}_model.joblib"
            joblib.dump(result['model'], model_path)
            
            # Save metrics
            metrics_path = save_path / f"{name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(result['metrics'], f, indent=2)
                
            # Save parameters
            params_path = save_path / f"{name}_params.json"
            with open(params_path, 'w') as f:
                json.dump(result['best_params'], f, indent=2)
                
        logger.info(f"Models saved to {save_path}")