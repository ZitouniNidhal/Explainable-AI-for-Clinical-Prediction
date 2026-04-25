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
            default_params = {
                "random_state": random_state,
                "eval_metric": 'logloss',
                "use_label_encoder": False
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)

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
                "C": FloatDistribution(1e-3, 1e2, log=True),
                "penalty": CategoricalDistribution(["l1", "l2"]),
                "solver": CategoricalDistribution(["liblinear", "saga"]),
            }
        elif name == "random_forest":
            return {
                "n_estimators": IntDistribution(50, 300),
                "max_depth": IntDistribution(3, 20),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 10),
            }
        elif name == "xgboost":
            return {
                "n_estimators": IntDistribution(50, 300),
                "max_depth": IntDistribution(3, 10),
                "learning_rate": FloatDistribution(1e-3, 0.5, log=True),
                "subsample": FloatDistribution(0.6, 1.0),
                "colsample_bytree": FloatDistribution(0.6, 1.0),
                "reg_alpha": FloatDistribution(1e-8, 10.0, log=True),
                "reg_lambda": FloatDistribution(1e-8, 10.0, log=True),
            }
        elif name == "lightgbm":
            return {
                "n_estimators": IntDistribution(50, 300),
                "num_leaves": IntDistribution(20, 100),
                "learning_rate": FloatDistribution(1e-3, 0.5, log=True),
                "feature_fraction": FloatDistribution(0.6, 1.0),
                "bagging_fraction": FloatDistribution(0.6, 1.0),
                "bagging_freq": IntDistribution(1, 10),
                "reg_alpha": FloatDistribution(1e-8, 10.0, log=True),
                "reg_lambda": FloatDistribution(1e-8, 10.0, log=True),
            }
        else:
            raise ValueError(f"Unsupported classifier: {name}")
