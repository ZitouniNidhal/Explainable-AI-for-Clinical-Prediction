# src/xai_clinical/evaluation/metrics.py
"""
Clinical evaluation metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix
)
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """
    Comprehensive clinical evaluation metrics
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate all relevant clinical metrics
        """
        self.metrics = {
            # Basic metrics
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
            
            # Threshold-based metrics at optimal threshold
            'optimal_threshold': self._find_optimal_threshold(y_true, y_proba),
        }
        
        # Calculate metrics at optimal threshold
        y_pred_opt = (y_proba >= self.metrics['optimal_threshold']).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_opt)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics.update({
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        })
        
        # Clinical utility metrics
        self.metrics['net_benefit'] = self._calculate_net_benefit(y_true, y_proba)
        
        return self.metrics
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find optimal threshold using Youden's J statistic"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    def _calculate_net_benefit(self, y_true: np.ndarray, y_proba: np.ndarray,
                              threshold: float = 0.3) -> float:
        """Calculate net benefit for clinical decision making"""
        n = len(y_true)
        n_events = np.sum(y_true)
        
        # True positives and false positives at threshold
        tp = np.sum((y_proba >= threshold) & (y_true == 1))
        fp = np.sum((y_proba >= threshold) & (y_true == 0))
        
        # Net benefit calculation
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        
        return net_benefit
    
    def get_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray,
                               n_bins: int = 10) -> Dict[str, float]:
        """Calculate calibration metrics"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        max_calibration_error = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                max_calibration_error = max(max_calibration_error, 
                                          np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'expected_calibration_error': calibration_error,
            'max_calibration_error': max_calibration_error
        }