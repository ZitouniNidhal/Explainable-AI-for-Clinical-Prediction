# src/xai_clinical/models/classifiers.py
"""
Factory and classifier definitions
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
class ClassifierFactory:
    """Factory for creating classification models"""

    @staticmethod
    def create_classifier(
        name: str, params: Dict[str, Any] = None, random_state: int = 42
    ):
        """
        Create a classifier with specified parameters

        Args:
            name: Classifier name ('logistic_regression', 'random_forest', 'xgboost', 'lightgbm')
            params: Hyperparameter dictionary
            random_state: Random seed
        """
        params = params or {}

        if name == "logistic_regression":
            default_params = {
                "random_state": random_state,
                "max_iter": 1000,
                "class_weight": "balanced",
            }
            default_params.update(params)
            return LogisticRegression(**default_params)

        elif name == "random_forest":
            default_params = {
                "random_state": random_state,
                "n_jobs": -1,
                "class_weight": "balanced",
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)

        elif name == 'xgboost':
            # CORRECTION: Ajouter eval_metric et use_label_encoder
            return xgb.XGBClassifier(
                **params,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            default_params.update(params)
            return XGBClassifier(**default_params)

        elif name == "lightgbm":
            default_params = {"random_state": random_state, "n_jobs": -1, "verbose": -1}
            default_params.update(params)
            return LGBMClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported classifier: {name}")

    @staticmethod
    def get_param_distributions(name: str) -> Dict[str, Any]:
        """Return parameter distributions for optimization"""
        if name == "logistic_regression":
            return {
                "C": optuna.distributions.LogUniformDistribution(1e-3, 1e2),
                "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2"]),
                "solver": optuna.distributions.CategoricalDistribution(
                    ["liblinear", "saga"]
                ),
            }
        elif name == "random_forest":
            return {
                "n_estimators": optuna.distributions.IntUniformDistribution(50, 300),
                "max_depth": optuna.distributions.IntUniformDistribution(3, 20),
                "min_samples_split": optuna.distributions.IntUniformDistribution(2, 20),
                "min_samples_leaf": optuna.distributions.IntUniformDistribution(1, 10),
            }
        elif name == "xgboost":
            return {
                "n_estimators": optuna.distributions.IntUniformDistribution(50, 300),
                "max_depth": optuna.distributions.IntUniformDistribution(3, 10),
                "learning_rate": optuna.distributions.LogUniformDistribution(1e-3, 0.5),
                "subsample": optuna.distributions.UniformDistribution(0.6, 1.0),
                "colsample_bytree": optuna.distributions.UniformDistribution(0.6, 1.0),
                "reg_alpha": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
                "reg_lambda": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
            }
        elif name == "lightgbm":
            return {
                "n_estimators": optuna.distributions.IntUniformDistribution(50, 300),
                "num_leaves": optuna.distributions.IntUniformDistribution(20, 100),
                "learning_rate": optuna.distributions.LogUniformDistribution(1e-3, 0.5),
                "feature_fraction": optuna.distributions.UniformDistribution(0.6, 1.0),
                "bagging_fraction": optuna.distributions.UniformDistribution(0.6, 1.0),
                "bagging_freq": optuna.distributions.IntUniformDistribution(1, 10),
                "reg_alpha": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
                "reg_lambda": optuna.distributions.LogUniformDistribution(1e-8, 10.0),
            }
        else:
            raise ValueError(f"Unsupported classifier: {name}")
