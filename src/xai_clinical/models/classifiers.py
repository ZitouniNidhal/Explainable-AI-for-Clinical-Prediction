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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier



class ClassifierFactory:
    """Factory for creating classification models"""

    SUPPORTED_MODELS = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
        "extra_trees",
        "catboost",
        "ada_boost",
        "knn",
        "gradient_boosting",
        "mlp",
        "naive_bayes",
        "decision_tree",
        "sgd",
        "ridge",
        "qda",
        "bagging",
        "hist_gradient_boosting",
    ]


    @staticmethod
    def create_classifier(
        name: str, params: Dict[str, Any] = None, random_state: int = 42, pos_label_weight: float = 1.0
    ):
        """
        Create a classifier with specified parameters

        Args:
            name: Classifier name. Supported values:
                  'logistic_regression', 'random_forest', 'xgboost', 'lightgbm',
                  'svm', 'extra_trees', 'catboost'
            params: Hyperparameter dictionary (overrides defaults)
            random_state: Random seed
            pos_label_weight: Weight for positive class (ratio of negative/positive)
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

        elif name == "xgboost":
            default_params = {
                "random_state": random_state,
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "scale_pos_weight": pos_label_weight, 
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)

        elif name == "lightgbm":
            default_params = {
                "random_state": random_state, 
                "n_jobs": -1, 
                "verbose": -1,
                "scale_pos_weight": pos_label_weight
            }
            default_params.update(params)
            return LGBMClassifier(**default_params)

        elif name == "svm":
            from sklearn.svm import SVC
            default_params = {
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,   # needed for predict_proba / SHAP KernelExplainer
                "random_state": random_state,
            }
            default_params.update(params)
            return SVC(**default_params)

        elif name == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            default_params = {
                "random_state": random_state,
                "n_jobs": -1,
                "class_weight": "balanced",
                "n_estimators": 200,
            }
            default_params.update(params)
            return ExtraTreesClassifier(**default_params)

        elif name == "catboost":
            from catboost import CatBoostClassifier
            default_params = {
                "random_seed": random_state,
                "verbose": 0,
                "scale_pos_weight": pos_label_weight, 
                "eval_metric": "AUC",
                "thread_count": -1,                 
            }
            default_params.update(params)
            return CatBoostClassifier(**default_params)

        elif name == "ada_boost":
            default_params = {
                "random_state": random_state,
                "n_estimators": 100,
            }
            default_params.update(params)
            return AdaBoostClassifier(**default_params)

        elif name == "knn":
            default_params = {
                "n_neighbors": 5,
                "n_jobs": -1,
            }
            default_params.update(params)
            return KNeighborsClassifier(**default_params)

        elif name == "gradient_boosting":
            default_params = {
                "random_state": random_state,
                "n_estimators": 100,
                "learning_rate": 0.1,
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)

        elif name == "mlp":
            default_params = {
                "random_state": random_state,
                "max_iter": 500,
                "hidden_layer_sizes": (100, 50),
            }
            default_params.update(params)
            return MLPClassifier(**default_params)

        elif name == "naive_bayes":
            return GaussianNB(**params)

        elif name == "decision_tree":
            default_params = {"random_state": random_state, "class_weight": "balanced"}
            default_params.update(params)
            return DecisionTreeClassifier(**default_params)

        elif name == "sgd":
            default_params = {"random_state": random_state, "loss": "log_loss", "class_weight": "balanced"}
            default_params.update(params)
            return SGDClassifier(**default_params)

        elif name == "ridge":
            default_params = {"random_state": random_state, "class_weight": "balanced"}
            default_params.update(params)
            return RidgeClassifier(**default_params)

        elif name == "qda":
            return QuadraticDiscriminantAnalysis(**params)

        elif name == "bagging":
            default_params = {"random_state": random_state, "n_jobs": -1}
            default_params.update(params)
            return BaggingClassifier(**default_params)

        elif name == "hist_gradient_boosting":
            default_params = {"random_state": random_state}
            default_params.update(params)
            return HistGradientBoostingClassifier(**default_params)

        else:
            raise ValueError(
                f"Unsupported classifier: '{name}'. "
                f"Supported: {ClassifierFactory.SUPPORTED_MODELS}"
            )

    @staticmethod
    def get_param_distributions(name: str) -> Dict[str, Any]:
        """Return Optuna parameter distributions for hyperparameter optimisation"""
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
        elif name == "svm":
            return {
                "C": FloatDistribution(1e-2, 1e2, log=True),
                "gamma": CategoricalDistribution(["scale", "auto"]),
                "kernel": CategoricalDistribution(["rbf", "poly", "sigmoid"]),
            }
        elif name == "extra_trees":
            return {
                "n_estimators": IntDistribution(50, 400),
                "max_depth": IntDistribution(3, 25),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 10),
                "max_features": CategoricalDistribution(["sqrt", "log2"]),
            }
        elif name == "catboost":
            return {
                "iterations": IntDistribution(50, 300),          # Reduced max iterations from 500 to 300
                "depth": IntDistribution(3, 7),                  # Reduced max depth from 10 to 7 (8+ is very slow)
                "learning_rate": FloatDistribution(1e-3, 0.5, log=True),
                "l2_leaf_reg": FloatDistribution(1e-3, 10.0, log=True),
                "bagging_temperature": FloatDistribution(0.0, 1.0),
            }
        elif name == "ada_boost":
            return {
                "n_estimators": IntDistribution(50, 200),
                "learning_rate": FloatDistribution(1e-2, 1.0, log=True),
            }
        elif name == "knn":
            return {
                "n_neighbors": IntDistribution(3, 30),
                "weights": CategoricalDistribution(["uniform", "distance"]),
                "metric": CategoricalDistribution(["euclidean", "manhattan"]),
            }
        elif name == "gradient_boosting":
            return {
                "n_estimators": IntDistribution(50, 300),
                "learning_rate": FloatDistribution(1e-3, 0.5, log=True),
                "max_depth": IntDistribution(3, 10),
                "min_samples_split": IntDistribution(2, 20),
            }
        elif name == "mlp":
            return {
                "hidden_layer_sizes": CategoricalDistribution([(50,), (100,), (100, 50), (50, 25)]),
                "activation": CategoricalDistribution(["relu", "tanh"]),
                "alpha": FloatDistribution(1e-5, 1e-1, log=True),
                "learning_rate_init": FloatDistribution(1e-4, 1e-2, log=True),
            }

        elif name == "naive_bayes":
            return {}  # No hyperparameters to tune for GaussianNB usually

        elif name == "decision_tree":
            return {
                "max_depth": IntDistribution(3, 30),
                "min_samples_split": IntDistribution(2, 20),
                "min_samples_leaf": IntDistribution(1, 10),
                "criterion": CategoricalDistribution(["gini", "entropy"]),
            }

        elif name == "sgd":
            return {
                "alpha": FloatDistribution(1e-6, 1e-1, log=True),
                "penalty": CategoricalDistribution(["l1", "l2", "elasticnet"]),
                "max_iter": IntDistribution(500, 2000),
            }

        elif name == "ridge":
            return {
                "alpha": FloatDistribution(1e-3, 1e2, log=True),
            }

        elif name == "qda":
            return {
                "reg_param": FloatDistribution(1e-2, 1.0, log=True),
            }

        elif name == "bagging":
            return {
                "n_estimators": IntDistribution(10, 100),
                "max_samples": FloatDistribution(0.5, 1.0),
                "max_features": FloatDistribution(0.5, 1.0),
            }

        elif name == "hist_gradient_boosting":
            return {
                "max_iter": IntDistribution(50, 300),
                "max_depth": IntDistribution(3, 15),
                "learning_rate": FloatDistribution(1e-3, 0.5, log=True),
                "l2_regularization": FloatDistribution(1e-8, 10.0, log=True),
            }

        else:
            raise ValueError(
                f"Unsupported classifier: '{name}'. "
                f"Supported: {ClassifierFactory.SUPPORTED_MODELS}"
            )
