# src/xai_clinical/visualization/plots.py
"""
Visualization utilities for XAI clinical predictions.

Provides a unified `ClinicalPlotter` class and standalone functions for
model evaluation, explainability (SHAP), and stability analysis in a
clinical context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

CLINICAL_PALETTE = {
    "primary": "#1A6B8A",
    "secondary": "#E07B39",
    "positive": "#2E8B57",
    "negative": "#C0392B",
    "neutral": "#7F8C8D",
    "grid": "#ECF0F1",
    "text": "#2C3E50",
    "bg": "#FAFBFC",
}

_FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": CLINICAL_PALETTE["text"]}
_FONT_LABEL = {"fontsize": 11, "color": CLINICAL_PALETTE["text"]}
_FONT_TICK = {"labelsize": 9, "colors": CLINICAL_PALETTE["text"]}
_SPINE_COLOR = "#D5D8DC"


def _apply_clinical_style(ax: plt.Axes) -> None:
    """Apply a consistent clinical aesthetic to a matplotlib Axes."""
    ax.set_facecolor(CLINICAL_PALETTE["bg"])
    ax.grid(
        True, linestyle="--", linewidth=0.6, color=CLINICAL_PALETTE["grid"], zorder=0
    )
    ax.tick_params(**_FONT_TICK)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
        spine.set_linewidth(0.8)


def _save_and_show(
    fig: Figure,
    save_path: Optional[str],
    dpi: int = 300,
) -> None:
    """Save figure to disk (if requested) and display it."""
    if save_path:
        fig.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        logger.info("Figure saved → %s", save_path)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# ClinicalPlotter — unified interface
# ---------------------------------------------------------------------------


class ClinicalPlotter:
    """
    Centralised plotting interface for XAI clinical model evaluation.

    Parameters
    ----------
    output_dir:
        Default directory for saving figures.  Individual calls may
        override via their ``save_path`` argument.
    dpi:
        Resolution used when saving figures.
    """

    def __init__(self, output_dir: Optional[str] = None, dpi: int = 300) -> None:
        self.output_dir = output_dir
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: Optional[str]) -> Optional[str]:
        """Resolve a save path, prepending ``output_dir`` if needed."""
        if filename is None:
            return None
        if self.output_dir and not filename.startswith("/"):
            import os

            return os.path.join(self.output_dir, filename)
        return filename

    # ------------------------------------------------------------------
    # Public API — thin wrappers around module-level functions
    # ------------------------------------------------------------------

    def model_comparison(
        self, results: Dict[str, Dict], save_path: Optional[str] = None
    ) -> None:
        plot_model_comparison(results, self._resolve_path(save_path), dpi=self.dpi)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        plot_confusion_matrix(
            y_true, y_pred, class_names, self._resolve_path(save_path), dpi=self.dpi
        )

    def roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        plot_roc_curves(y_true, y_proba, self._resolve_path(save_path), dpi=self.dpi)

    def precision_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        plot_precision_recall_curve(
            y_true, y_proba, self._resolve_path(save_path), dpi=self.dpi
        )

    def calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        plot_calibration_curve(
            y_true, y_proba, n_bins, self._resolve_path(save_path), dpi=self.dpi
        )

    def feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        plot_feature_importance(
            importance_df, title, top_n, self._resolve_path(save_path), dpi=self.dpi
        )

    def shap_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        plot_shap_summary(
            shap_values, X, feature_names, self._resolve_path(save_path), dpi=self.dpi
        )

    def waterfall(
        self,
        explanation: Dict[str, Any],
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        plot_waterfall(
            explanation, feature_names, self._resolve_path(save_path), dpi=self.dpi
        )

    def stability(
        self,
        stability_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        plot_stability_analysis(
            stability_results, self._resolve_path(save_path), dpi=self.dpi
        )

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Feature Correlation",
        save_path: Optional[str] = None,
    ) -> None:
        plot_correlation_heatmap(df, title, self._resolve_path(save_path), dpi=self.dpi)

    def metrics_report(
        self,
        metrics: Dict[str, float],
        title: str = "Model Metrics",
        save_path: Optional[str] = None,
    ) -> None:
        plot_metrics_report(metrics, title, self._resolve_path(save_path), dpi=self.dpi)


# ---------------------------------------------------------------------------
# Module-level plot functions
# ---------------------------------------------------------------------------


def plot_model_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Bar-chart comparison of multiple models across four key metrics.

    Parameters
    ----------
    results:
        ``{model_name: {"metrics": {metric_key: value}}}`` mapping.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    if not results:
        logger.warning("plot_model_comparison: empty results dict — skipping.")
        return

    models = list(results.keys())
    metrics = ["train_roc_auc", "val_roc_auc", "train_f1", "val_f1"]
    titles = ["Train ROC-AUC", "Validation ROC-AUC", "Train F1", "Validation F1"]
    colors = [
        CLINICAL_PALETTE["primary"],
        CLINICAL_PALETTE["secondary"],
        CLINICAL_PALETTE["positive"],
        CLINICAL_PALETTE["neutral"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=CLINICAL_PALETTE["bg"])
    fig.suptitle("Model Performance Comparison", **_FONT_TITLE, fontsize=15, y=1.01)

    for ax, metric, title, color in zip(axes.ravel(), metrics, titles, colors):
        values = [results[m]["metrics"].get(metric, 0.0) for m in models]
        bars = ax.bar(
            models,
            values,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        _apply_clinical_style(ax)
        ax.set_title(title, **_FONT_TITLE)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        # Value annotations
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=CLINICAL_PALETTE["text"],
                fontweight="bold",
            )

    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Annotated confusion matrix with per-cell percentage labels.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    class_names:
        Display names for each class.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    class_names = class_names or ["No Complication", "Complication"]
    cm = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
    )
    cm_norm = cm.div(cm.sum(axis=1), axis=0)  # row-normalised for colour scale

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=CLINICAL_PALETTE["bg"])
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor=_SPINE_COLOR,
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Row proportion"},
    )

    # Overlay raw counts + percentages in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm.iloc[i, j])
            pct = cm_norm.iloc[i, j]
            text_color = "white" if pct > 0.55 else CLINICAL_PALETTE["text"]
            ax.text(
                j + 0.5,
                i + 0.45,
                f"{count}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                j + 0.5,
                i + 0.62,
                f"({pct:.1%})",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_title("Confusion Matrix", **_FONT_TITLE, pad=12)
    ax.set_ylabel("Actual", **_FONT_LABEL)
    ax.set_xlabel("Predicted", **_FONT_LABEL)
    ax.tick_params(**_FONT_TICK)
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    ROC curve with shaded AUC area and 95 % confidence band (bootstrap).

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_proba:
        Predicted positive-class probabilities.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=CLINICAL_PALETTE["bg"])
    _apply_clinical_style(ax)

    ax.fill_between(fpr, tpr, alpha=0.15, color=CLINICAL_PALETTE["primary"])
    ax.plot(
        fpr,
        tpr,
        color=CLINICAL_PALETTE["primary"],
        lw=2.5,
        label=f"ROC curve  (AUC = {roc_auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=CLINICAL_PALETTE["neutral"],
        lw=1.5,
        label="Random classifier",
    )

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    opt_idx = int(np.argmax(j_scores))
    ax.scatter(
        fpr[opt_idx],
        tpr[opt_idx],
        color=CLINICAL_PALETTE["secondary"],
        zorder=5,
        s=80,
        label=f"Optimal threshold  (J = {j_scores[opt_idx]:.3f})",
    )

    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate", **_FONT_LABEL)
    ax.set_ylabel("True Positive Rate", **_FONT_LABEL)
    ax.set_title("Receiver Operating Characteristic", **_FONT_TITLE, pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Precision-Recall curve with average precision annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_proba:
        Predicted positive-class probabilities.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=CLINICAL_PALETTE["bg"])
    _apply_clinical_style(ax)

    ax.fill_between(recall, precision, alpha=0.15, color=CLINICAL_PALETTE["positive"])
    ax.plot(
        recall,
        precision,
        color=CLINICAL_PALETTE["positive"],
        lw=2.5,
        label=f"PR curve  (AP = {ap:.3f})",
    )
    ax.axhline(
        baseline,
        color=CLINICAL_PALETTE["neutral"],
        linestyle="--",
        lw=1.5,
        label=f"Random baseline  ({baseline:.3f})",
    )

    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("Recall (Sensitivity)", **_FONT_LABEL)
    ax.set_ylabel("Precision (PPV)", **_FONT_LABEL)
    ax.set_title("Precision-Recall Curve", **_FONT_TITLE, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Calibration (reliability) diagram with histogram of predicted probabilities.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_proba:
        Predicted positive-class probabilities.
    n_bins:
        Number of calibration bins.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    from sklearn.calibration import calibration_curve

    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, (ax_cal, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        facecolor=CLINICAL_PALETTE["bg"],
    )

    # --- calibration plot ---
    _apply_clinical_style(ax_cal)
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax_cal.plot(
        mean_pred,
        fraction_pos,
        "o-",
        color=CLINICAL_PALETTE["primary"],
        lw=2,
        ms=7,
        label="Model calibration",
    )
    ax_cal.fill_between(
        mean_pred,
        mean_pred,
        fraction_pos,
        alpha=0.12,
        color=CLINICAL_PALETTE["secondary"],
        label="Calibration gap",
    )
    ax_cal.set_ylabel("Fraction of Positives", **_FONT_LABEL)
    ax_cal.set_title("Calibration Plot", **_FONT_TITLE, pad=12)
    ax_cal.legend(fontsize=9, framealpha=0.9)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xticklabels([])

    # --- histogram ---
    _apply_clinical_style(ax_hist)
    ax_hist.hist(
        y_proba,
        bins=n_bins,
        color=CLINICAL_PALETTE["primary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_hist.set_xlabel("Mean Predicted Probability", **_FONT_LABEL)
    ax_hist.set_ylabel("Count", **_FONT_LABEL)
    ax_hist.set_xlim(0, 1)

    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importance_df:
        DataFrame with columns ``feature`` and ``importance``.
    title:
        Figure title.
    top_n:
        Number of top features to display.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    if (
        "feature" not in importance_df.columns
        or "importance" not in importance_df.columns
    ):
        raise ValueError(
            "importance_df must contain 'feature' and 'importance' columns."
        )

    df = (
        importance_df.sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)  # flip for horizontal bars
    )

    cmap = plt.cm.get_cmap("Blues", len(df) + 4)
    colors = [cmap(i + 4) for i in range(len(df))]

    fig, ax = plt.subplots(
        figsize=(10, max(5, top_n * 0.45)), facecolor=CLINICAL_PALETTE["bg"]
    )
    _apply_clinical_style(ax)

    bars = ax.barh(
        df["feature"], df["importance"], color=colors, edgecolor="white", linewidth=0.6
    )
    ax.set_xlabel("Importance Score", **_FONT_LABEL)
    ax.set_title(title, **_FONT_TITLE, pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Value annotations
    for bar, val in zip(bars, df["importance"]):
        ax.text(
            val + df["importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
            color=CLINICAL_PALETTE["text"],
        )

    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values:
        SHAP value matrix (n_samples × n_features).
    X:
        Feature matrix aligned with ``shap_values``.
    feature_names:
        Display names for each feature column.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(feature_names) * 0.45)),
        facecolor=CLINICAL_PALETTE["bg"],
    )
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor(CLINICAL_PALETTE["bg"])
    plt.title("SHAP Feature Impact", **_FONT_TITLE)
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_waterfall(
    explanation: Dict[str, Any],
    feature_names: List[str],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    SHAP waterfall plot for a single patient prediction.

    Parameters
    ----------
    explanation:
        Dict with keys ``shap_values``, ``base_value``, and ``feature_values``.
    feature_names:
        Display names for each feature.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    required = {"shap_values", "base_value", "feature_values"}
    missing = required - set(explanation)
    if missing:
        raise KeyError(f"explanation dict is missing keys: {missing}")

    shap_exp = shap.Explanation(
        values=np.asarray(explanation["shap_values"]),
        base_values=float(explanation["base_value"]),
        data=np.asarray(explanation["feature_values"]),
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(
        figsize=(10, max(5, len(feature_names) * 0.4)), facecolor=CLINICAL_PALETTE["bg"]
    )
    shap.waterfall_plot(shap_exp, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor(CLINICAL_PALETTE["bg"])
    plt.title("Individual Prediction Explanation", **_FONT_TITLE)
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_stability_analysis(
    stability_results: Dict[str, Any],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Line plots of Jaccard index and Spearman correlation across noise levels.

    Parameters
    ----------
    stability_results:
        Dict keyed by perturbation type, each containing per-noise-level dicts
        with keys ``jaccard_mean``, ``jaccard_std``, ``spearman_mean``,
        ``spearman_std``.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    gaussian = stability_results.get("gaussian_noise", {})
    if not gaussian:
        logger.warning(
            "plot_stability_analysis: 'gaussian_noise' key not found — skipping."
        )
        return

    noise_levels = []
    jaccard_means = []
    jaccard_stds = []
    spearman_means = []
    spearman_stds = []

    for key, metrics in gaussian.items():
        try:
            noise_levels.append(float(key.split("_")[-1]))
        except ValueError:
            noise_levels.append(float(key))
        jaccard_means.append(metrics["jaccard_mean"])
        jaccard_stds.append(metrics["jaccard_std"])
        spearman_means.append(metrics["spearman_mean"])
        spearman_stds.append(metrics["spearman_std"])

    # Sort by noise level
    order = np.argsort(noise_levels)
    noise_levels = np.array(noise_levels)[order]
    jaccard_means = np.array(jaccard_means)[order]
    jaccard_stds = np.array(jaccard_stds)[order]
    spearman_means = np.array(spearman_means)[order]
    spearman_stds = np.array(spearman_stds)[order]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor=CLINICAL_PALETTE["bg"]
    )

    for ax, means, stds, ylabel, title, color in [
        (
            ax1,
            jaccard_means,
            jaccard_stds,
            "Jaccard Index",
            "Top-K Feature Overlap vs Noise",
            CLINICAL_PALETTE["primary"],
        ),
        (
            ax2,
            spearman_means,
            spearman_stds,
            "Spearman ρ",
            "Rank Correlation vs Noise",
            CLINICAL_PALETTE["secondary"],
        ),
    ]:
        _apply_clinical_style(ax)
        ax.fill_between(
            noise_levels,
            means - stds,
            means + stds,
            alpha=0.18,
            color=color,
            label="±1 SD",
        )
        ax.plot(
            noise_levels, means, "o-", color=color, lw=2, ms=6, label="Mean", zorder=3
        )
        ax.set_xlabel("Noise Level (σ)", **_FONT_LABEL)
        ax.set_ylabel(ylabel, **_FONT_LABEL)
        ax.set_title(title, **_FONT_TITLE, pad=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Triangular Pearson correlation heatmap.

    Parameters
    ----------
    df:
        DataFrame of numeric features.
    title:
        Figure title.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(
        figsize=(max(8, len(df.columns) * 0.7), max(7, len(df.columns) * 0.65)),
        facecolor=CLINICAL_PALETTE["bg"],
    )
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        linecolor=_SPINE_COLOR,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title(title, **_FONT_TITLE, pad=12)
    ax.tick_params(**_FONT_TICK, axis="both")
    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)


def plot_metrics_report(
    metrics: Dict[str, float],
    title: str = "Model Metrics",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Horizontal gauge-style bar chart summarising all scalar metrics.

    Parameters
    ----------
    metrics:
        ``{metric_name: value}`` mapping (values in [0, 1]).
    title:
        Figure title.
    save_path:
        Optional file path to save the figure.
    dpi:
        Figure resolution when saving.
    """
    if not metrics:
        logger.warning("plot_metrics_report: empty metrics dict — skipping.")
        return

    names = list(metrics.keys())
    values = [float(v) for v in metrics.values()]

    # Colour-code by threshold
    bar_colors = [
        (
            CLINICAL_PALETTE["positive"]
            if v >= 0.8
            else (
                CLINICAL_PALETTE["secondary"]
                if v >= 0.6
                else CLINICAL_PALETTE["negative"]
            )
        )
        for v in values
    ]

    fig, ax = plt.subplots(
        figsize=(9, max(4, len(names) * 0.5)), facecolor=CLINICAL_PALETTE["bg"]
    )
    _apply_clinical_style(ax)

    bars = ax.barh(
        names, values, color=bar_colors, edgecolor="white", linewidth=0.7, height=0.55
    )
    ax.axvline(
        0.8,
        color=CLINICAL_PALETTE["positive"],
        lw=1.2,
        linestyle="--",
        alpha=0.6,
        label="Good (≥ 0.80)",
    )
    ax.axvline(
        0.6,
        color=CLINICAL_PALETTE["secondary"],
        lw=1.2,
        linestyle=":",
        alpha=0.6,
        label="Acceptable (≥ 0.60)",
    )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score", **_FONT_LABEL)
    ax.set_title(title, **_FONT_TITLE, pad=12)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
            color=CLINICAL_PALETTE["text"],
            fontweight="bold",
        )

    plt.tight_layout()
    _save_and_show(fig, save_path, dpi)
