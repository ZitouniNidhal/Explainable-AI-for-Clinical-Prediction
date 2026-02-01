# src/xai_clinical/visualization/plots.py (complété)
"""
Visualization utilities for XAI clinical predictions
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import shap
import logging

logger = logging.getLogger(__name__)


def plot_model_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Compare multiple models' performance"""
    models = list(results.keys())
    metrics = ['train_roc_auc', 'val_roc_auc', 'train_f1', 'val_f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results[model]['metrics'].get(metric, 0) for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, save_path: Optional[str] = None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = ['No Complication', 'Complication']
    
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                          n_bins: int = 10, save_path: Optional[str] = None):
    """Plot calibration curve"""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel('Fraction of Positives')
    plt.xlabel('Mean Predicted Value')
    plt.title('Calibration Plot')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance",
                          save_path: Optional[str] = None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, 
                     feature_names: List[str], save_path: Optional[str] = None):
    """Plot SHAP summary"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_waterfall(explanation: Dict, feature_names: List[str], 
                  save_path: Optional[str] = None):
    """Plot SHAP waterfall for individual prediction"""
    plt.figure(figsize=(10, 6))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=explanation['shap_values'],
            base_values=explanation['base_value'],
            data=explanation['feature_values'],
            feature_names=feature_names
        ),
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_stability_analysis(stability_results: Dict, save_path: Optional[str] = None):
    """Plot stability analysis results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Jaccard index plot
    noise_levels = []
    jaccard_means = []
    jaccard_stds = []
    
    for noise_level, metrics in stability_results['gaussian_noise'].items():
        noise_levels.append(float(noise_level.split('_')[1]))
        jaccard_means.append(metrics['jaccard_mean'])
        jaccard_stds.append(metrics['jaccard_std'])
    
    axes[0].errorbar(noise_levels, jaccard_means, yerr=jaccard_stds, 
                     marker='o', capsize=5)
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Jaccard Index')
    axes[0].set_title('Stability vs Gaussian Noise')
    axes[0].set_ylim(0, 1)
    
    # Spearman correlation plot
    spearman_means = []
    spearman_stds = []
    
    for noise_level, metrics in stability_results['gaussian_noise'].items():
        spearman_means.append(metrics['spearman_mean'])
        spearman_stds.append(metrics['spearman_std'])
    
    axes[1].errorbar(noise_levels, spearman_means, yerr=spearman_stds, 
                     marker='s', capsize=5)
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Spearman Correlation')
    axes[1].set_title('Rank Correlation vs Gaussian Noise')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()