# src/xai_clinical/explainability/lime_explainer.py
"""
LIME Explainer for comparison with SHAP
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Union
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path
import logging


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) wrapper
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        class_names: List[str] = None,
        mode: str = "classification",
    ):
        """
        Initialize LIME explainer

        Args:
            model: Trained model
            X_train: Training data to define distributions
            feature_names: Feature names
            class_names: Class names
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.feature_names = feature_names or X_train.columns.tolist()
        self.class_names = class_names or ["Class 0", "Class 1"]
        self.mode = mode

        # Detect categorical features
        self.categorical_features = []
        self.categorical_names = {}

        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype == "object" or X_train[col].nunique() < 10:
                self.categorical_features.append(i)
                self.categorical_names[i] = list(X_train[col].unique())

        # Create explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            categorical_names=self.categorical_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain_instance(
        self,
        X_instance: Union[pd.Series, np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> dict:
        """
        Explain individual instance

        Returns:
            Dictionary with feature weights and prediction
        """
        if isinstance(X_instance, pd.Series):
            X_array = X_instance.values.reshape(1, -1)[0]
        else:
            X_array = (
                X_instance.reshape(1, -1)[0]
                if len(X_instance.shape) > 1
                else X_instance
            )

        # Generate explanation
        explanation = self.explainer.explain_instance(
            X_array,
            (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            ),
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extract results
        features_weights = explanation.as_list()
        features_map = explanation.as_map()

        return {
            "explanation": explanation,
            "features_weights": features_weights,
            "prediction_proba": (
                explanation.predict_proba
                if hasattr(explanation, "predict_proba")
                else None
            ),
            "local_pred": explanation.local_pred,
            "intercept": explanation.intercept,
        }

    def plot_explanation(
        self,
        X_instance: Union[pd.Series, np.ndarray],
        num_features: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Display LIME explanation plot
        """
        explanation = self.explain_instance(X_instance, num_features)

        fig = explanation["explanation"].as_pyplot_figure()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return explanation

    def compare_with_shap(
        self,
        X_instance: Union[pd.Series, np.ndarray],
        shap_explainer,
        num_features: int = 10,
    ) -> pd.DataFrame:
        """
        Compare LIME and SHAP explanations for an instance
        """
        # LIME explanation
        lime_exp = self.explain_instance(X_instance, num_features)
        lime_features = {
            feat.split("=")[0].strip(): weight
            for feat, weight in lime_exp["features_weights"]
        }

        # SHAP explanation
        X_df = pd.DataFrame(
            [X_instance.values] if isinstance(X_instance, pd.Series) else [X_instance],
            columns=self.feature_names,
        )
        shap_exp = shap_explainer.explain_local(X_df, instance_idx=0)
        shap_features = dict(zip(shap_exp["feature_names"], shap_exp["shap_values"]))

        # Comparison
        comparison = []
        all_features = set(lime_features.keys()) | set(shap_features.keys())

        for feat in all_features:
            comparison.append(
                {
                    "feature": feat,
                    "lime_weight": lime_features.get(feat, 0),
                    "shap_value": shap_features.get(feat, 0),
                    "abs_lime": abs(lime_features.get(feat, 0)),
                    "abs_shap": abs(shap_features.get(feat, 0)),
                }
            )

        return pd.DataFrame(comparison).sort_values("abs_shap", ascending=False)
