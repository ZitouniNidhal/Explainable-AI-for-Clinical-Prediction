# src/xai_clinical/evaluation/clinical_metrics.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

def calculate_clinical_metrics(y_true, y_pred, y_proba):
    """
    Métriques cliniques complètes pour évaluation du modèle
    """
    metrics = {}
    
    # Métriques standard
    metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Sensibilité (Recall)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Spécificité
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # PPV (Precision)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # NPV
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F1-score
    precision = metrics['ppv']
    recall = metrics['sensitivity']
    metrics['f1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Likelihood ratios
    metrics['plr'] = metrics['sensitivity'] / (1 - metrics['specificity']) if metrics['specificity'] < 1 else float('inf')
    metrics['nlr'] = (1 - metrics['sensitivity']) / metrics['specificity'] if metrics['specificity'] > 0 else float('inf')
    
    return metrics


def print_clinical_report(metrics, model_name="Model"):
    """Affiche un rapport clinique formaté"""
    print(f"\n{'='*60}")
    print(f"CLINICAL VALIDATION REPORT - {model_name}")
    print(f"{'='*60}")
    print(f"AUC-ROC:          {metrics['auc_roc']:.3f}")
    print(f"Average Precision: {metrics['average_precision']:.3f}")
    print(f"Sensitivity:      {metrics['sensitivity']:.3f} (Recall)")
    print(f"Specificity:      {metrics['specificity']:.3f}")
    print(f"PPV (Precision):  {metrics['ppv']:.3f}")
    print(f"NPV:              {metrics['npv']:.3f}")
    print(f"F1-Score:         {metrics['f1']:.3f}")
    print(f"Positive LR:      {metrics['plr']:.2f}")
    print(f"Negative LR:      {metrics['nlr']:.2f}")
    print(f"{'='*60}")