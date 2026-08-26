# src/xai_clinical/explainability/shap_explainer.py
"""
SHAP Explainer for prediction interpretation
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) wrapper adapted for clinical models
    """

    def __init__(
        self,
        model: Any,
        X_background: pd.DataFrame,
        explainer_type: str = "auto",
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize SHAP explainer

        Args:
            model: Trained model (sklearn, xgboost, lightgbm)
            X_background: Background data for SHAP
            explainer_type: Explainer type ('tree', 'kernel', 'deep', 'auto')
            feature_names: Feature names
        """
        self.model = model
        self.feature_names = feature_names or X_background.columns.tolist()
        self.explainer_type = explainer_type
        self.explainer = None
        self.expected_value = None

        self._create_explainer(X_background)

    def _create_explainer(self, X_background: pd.DataFrame):
        """Create appropriate explainer based on model type"""
        try:
            # Auto-detect model type
            model_type = type(self.model).__name__

            if self.explainer_type == "auto":
                if (
                    "XGB" in model_type
                    or "LGBM" in model_type
                    or "RandomForest" in model_type
                ):
                    self.explainer_type = "tree"
                else:
                    self.explainer_type = "kernel"

            if self.explainer_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "kernel":
                # Sample for faster kernel explainer
                background_sample = shap.sample(X_background, 100)
                
                # FIX: Ensure pure numpy output and fixed shape
                def predict_fn(x):
                    # Convert to numpy and ensure 2D
                    return np.array(self.model.predict_proba(x))
                
                self.explainer = shap.KernelExplainer(
                    predict_fn, background_sample
                )
            else:
                raise ValueError(f"Unsupported explainer type: {self.explainer_type}")

            self.expected_value = self.explainer.expected_value
            if isinstance(self.expected_value, list):
                self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
            if hasattr(self.expected_value, '__len__') and len(self.expected_value) > 1:
                self.expected_value = float(self.expected_value[1])
            else:
                self.expected_value = float(np.asarray(self.expected_value).flatten()[0])

        except Exception as e:
            logger.error(f"Error creating explainer: {e}")
            raise

    def explain_local(self, X: pd.DataFrame, instance_idx: int = 0) -> dict:
        """
        Explain individual prediction

        Returns:
            Dictionary with SHAP values, feature values, and base value
        """
        if isinstance(X, pd.DataFrame):
            X_input = X.iloc[[instance_idx]]
        else:
            X_input = X[instance_idx:instance_idx+1]

        # Use helper for consistent 1D values
        instance_shap = self._get_1d_shap_values(X_input)

        # Create result
        explanation = {
            "shap_values": instance_shap,
            "feature_values": X.iloc[instance_idx].values if isinstance(X, pd.DataFrame) else X[instance_idx],
            "base_value": self.expected_value[1] if isinstance(self.expected_value, (list, np.ndarray)) else self.expected_value,
            "prediction": (self.expected_value[1] if isinstance(self.expected_value, (list, np.ndarray)) else self.expected_value) + np.sum(instance_shap),
            "feature_names": self.feature_names,
        }

        return explanation

    def explain_global(self, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
        """
        Calculate global feature importance

        Returns:
            DataFrame with mean feature importance
        """
        # Use a safe number of samples for KernelExplainer
        kwargs = {}
        if self.explainer_type == "kernel":
            kwargs["nsamples"] = 100 # Fast and safe
            
        shap_values = self.explainer.shap_values(
            X.values if isinstance(X, pd.DataFrame) else X,
            **kwargs
        )

        # Handle different SHAP output formats (List, Explanation object, Multi-class array)
        # 1. Explanation object
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values

        # 2. List of arrays (Binary classification often returns [class0_values, class1_values])
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]

        # 3. Multi-dimensional array (Samples, Features, Classes)
        if len(values.shape) == 3:
            # Assume binary/multi-class, take the positive/relevant class (index 1)
            values = values[:, :, 1]

        # Mean absolute importance
        importance = np.abs(values).mean(axis=0)

        # Ensure importance is 1D
        if len(importance.shape) > 1:
            importance = importance.flatten()

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_importance": importance}
        ).sort_values("shap_importance", ascending=False)

        return importance_df.head(max_display)

    def _get_1d_shap_values(self, X_input) -> np.ndarray:
        """Helper to get 1D SHAP values for a single instance or mean across instances."""
        # Force conversion to array if it's an Explanation object
        # Use a safe number of samples for KernelExplainer
        kwargs = {}
        if self.explainer_type == "kernel":
            kwargs["nsamples"] = 100
            
        shap_out = self.explainer.shap_values(X_input, **kwargs)
        
        # Handle different SHAP output formats
        if isinstance(shap_out, list):
            # Binary classification usually [class0, class1]
            values = shap_out[1] if len(shap_out) > 1 else shap_out[0]
        elif hasattr(shap_out, 'values'):
            values = shap_out.values
        else:
            values = shap_out

        # Reduce dimensions
        # 3D: (Samples, Features, Classes) -> (Samples, Features)
        if len(values.shape) == 3:
            values = values[:, :, 1]
            
        # If we passed a single instance, it might still have a sample dimension: (1, Features) -> (Features,)
        if len(values.shape) == 2 and values.shape[0] == 1:
            values = values[0]
            
        return values

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        max_display: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Create waterfall plot for specific instance
        """
        instance_x = X.iloc[[instance_idx]]
        instance_values = self._get_1d_shap_values(instance_x)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_values,
                base_values=self.expected_value[1] if isinstance(self.expected_value, (list, np.ndarray)) else self.expected_value,
                data=X.iloc[instance_idx].values,
                feature_names=self.feature_names,
            ),
            max_display=max_display,
            show=False
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_type: str = "dot",
        save_path: Optional[str] = None,
    ):
        """
        Create global summary plot
        """
        # Handle multi-class
        values = self._get_1d_shap_values(X.values)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            values,
            X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Create dependence plot to analyze feature effect
        """
        # Handle multi-class
        values = self._get_1d_shap_values(X.values)

        feature_idx = (
            self.feature_names.index(feature)
            if isinstance(feature, str)
            else feature
        )

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            values,
            X.values,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def get_force_plot_html(self, X: pd.DataFrame, instance_idx: int = 0) -> str:
        """
        Generate interactive force plot in HTML
        """
        shap_values = self.explainer.shap_values(X.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        force_plot = shap.force_plot(
            self.expected_value,
            shap_values[instance_idx],
            X.iloc[instance_idx].values,
            feature_names=self.feature_names,
            matplotlib=False,
        )

        return shap.save_html("force_plot.html", force_plot)
