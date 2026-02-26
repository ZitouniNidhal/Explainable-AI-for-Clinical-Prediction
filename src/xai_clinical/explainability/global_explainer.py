# src/xai_clinical/evaluation/clinical_validation.py
"""
Clinical validation utilities
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ClinicalValidator:
    """
    Clinical validation and risk stratification
    """
    
    def __init__(self, config):
        self.config = config
        self.risk_thresholds = config.evaluation.clinical_thresholds
    
    def stratify_risk(self, probabilities: np.ndarray) -> np.ndarray:
        """Stratify patients into risk categories"""
        risk_categories = np.zeros_like(probabilities, dtype=int)
        
        high_risk = probabilities >= self.risk_thresholds['high_risk']
        medium_risk = (probabilities >= self.risk_thresholds['medium_risk']) & (~high_risk)
        
        risk_categories[medium_risk] = 1
        risk_categories[high_risk] = 2
        
        return risk_categories
    
    def validate_clinical_sense(self, model, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that model predictions make clinical sense
        """
        validation_results = {}
        
        # Test 1: Age effect
        young_patient = X_sample.mean().copy()
        old_patient = X_sample.mean().copy()
        young_patient['age'] = 25
        old_patient['age'] = 85
        
        young_risk = model.predict_proba(young_patient.values.reshape(1, -1))[0, 1]
        old_risk = model.predict_proba(old_patient.values.reshape(1, -1))[0, 1]
        
        validation_results['age_sensibility'] = {
            'young_risk': young_risk,
            'old_risk': old_risk,
            'age_effect_valid': old_risk > young_risk
        }
        
        # Test 2: ASA score effect
        low_asa = X_sample.mean().copy()
        high_asa = X_sample.mean().copy()
        low_asa['asa_score'] = 1
        high_asa['asa_score'] = 4
        
        low_risk = model.predict_proba(low_asa.values.reshape(1, -1))[0, 1]
        high_risk = model.predict_proba(high_asa.values.reshape(1, -1))[0, 1]
        
        validation_results['asa_sensibility'] = {
            'low_asa_risk': low_risk,
            'high_asa_risk': high_risk,
            'asa_effect_valid': high_risk > low_risk
        }
        
        # Test 3: Emergency surgery effect
        elective = X_sample.mean().copy()
        emergency = X_sample.mean().copy()
        elective['emergency'] = 0
        emergency['emergency'] = 1
        
        elective_risk = model.predict_proba(elective.values.reshape(1, -1))[0, 1]
        emergency_risk = model.predict_proba(emergency.values.reshape(1, -1))[0, 1]
        
        validation_results['emergency_sensibility'] = {
            'elective_risk': elective_risk,
            'emergency_risk': emergency_risk,
            'emergency_effect_valid': emergency_risk > elective_risk
        }
        
        # Overall validation
        all_valid = all([
            validation_results['age_sensibility']['age_effect_valid'],
            validation_results['asa_sensibility']['asa_effect_valid'],
            validation_results['emergency_sensibility']['emergency_effect_valid']
        ])
        
        validation_results['overall_valid'] = all_valid
        
        return validation_results
    
    def generate_clinical_report(self, model, X_test: pd.DataFrame, 
                               y_test: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """Generate clinical validation report"""
        from .metrics import ClinicalMetrics
        
        metrics = ClinicalMetrics()
        metric_results = metrics.calculate_all_metrics(y_test, 
                                                       (y_pred_proba >= 0.5).astype(int), 
                                                       y_pred_proba)
        
        validation_results = self.validate_clinical_sense(model, X_test)
        
        report = ["="*60]
        report.append("CLINICAL VALIDATION REPORT")
        report.append("="*60)
        
        report.append(f"\nModel Performance:")
        report.append(f"- AUC-ROC: {metric_results['auc_roc']:.3f}")
        report.append(f"- Sensitivity: {metric_results['sensitivity']:.3f}")
        report.append(f"- Specificity: {metric_results['specificity']:.3f}")
        report.append(f"- PPV: {metric_results['ppv']:.3f}")
        report.append(f"- NPV: {metric_results['npv']:.3f}")
        
        report.append(f"\nClinical Validation:")
        report.append(f"- Age effect valid: {validation_results['age_sensibility']['age_effect_valid']}")
        report.append(f"- ASA effect valid: {validation_results['asa_sensibility']['asa_effect_valid']}")
        report.append(f"- Emergency effect valid: {validation_results['emergency_sensibility']['emergency_effect_valid']}")
        report.append(f"- Overall clinical validity: {validation_results['overall_valid']}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)