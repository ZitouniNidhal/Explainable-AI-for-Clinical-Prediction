"""
Visualization module for XAI Clinical Prediction.
"""

from .plots import (
    ClinicalPlotter,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_feature_importance,
    plot_shap_summary,
    plot_waterfall,
    plot_stability_analysis,
    plot_correlation_heatmap,
    plot_metrics_report,
)

__all__ = [
    "ClinicalPlotter",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "plot_feature_importance",
    "plot_shap_summary",
    "plot_waterfall",
    "plot_stability_analysis",
    "plot_correlation_heatmap",
    "plot_metrics_report",
]
