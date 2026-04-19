"""
Clinical validation utilities with global explanation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import shap  # Ajout de SHAP pour l'explication globale
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
logger = logging.getLogger(__name__)

class ClinicalValidator:
    """
    Clinical validation and risk stratification with global explanation
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

    def generate_global_explanation(self, model, X_test: pd.DataFrame, method: str = "shap") -> Dict[str, Any]:
        """
        Generate global explanation of the model using SHAP or feature importance
        """
        explanation = {}

        if method == "shap":
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            explanation['shap_values'] = shap_values.values.mean(axis=0)
            explanation['feature_names'] = X_test.columns.tolist()
        elif method == "feature_importance":
            if hasattr(model, 'feature_importances_'):
                explanation['feature_importances'] = model.feature_importances_
                explanation['feature_names'] = X_test.columns.tolist()
            else:
                logger.warning("Model does not support feature_importances_")
                explanation['feature_importances'] = None
        else:
            logger.warning(f"Unknown explanation method: {method}")

        return explanation

    def generate_clinical_report(self, model, X_test: pd.DataFrame,
                               y_test: np.ndarray, y_pred_proba: np.ndarray,
                               explanation_method: Optional[str] = None) -> str:
        """Generate clinical validation report with optional global explanation"""
        from .metrics import ClinicalMetrics

        metrics = ClinicalMetrics()
        metric_results = metrics.calculate_all_metrics(y_test,(y_pred_proba >= 0.5).astype(int),y_pred_proba)

        validation_results = self.validate_clinical_sense(model, X_test)
        explanation = self.generate_global_explanation(model, X_test, method=explanation_method) if explanation_method else {}

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

        if explanation_method:
            report.append(f"\nGlobal Explanation ({explanation_method}):")
            if explanation_method == "shap":
                for feature, value in zip(explanation['feature_names'], explanation['shap_values']):
                    report.append(f"- {feature}: {value:.3f}")
            elif explanation_method == "feature_importance" and explanation.get('feature_importances') is not None:
                for feature, importance in zip(explanation['feature_names'], explanation['feature_importances']):
                    report.append(f"- {feature}: {importance:.3f}")

        report.append("\n" + "="*60)

        return "\n".join(report)
"""
Global explainer for model-wide interpretability
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class GlobalExplainer:
    """
    Global explainer providing model-wide feature importance
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.global_importance = None
        
    def explain(self, X: pd.DataFrame, method: str = 'permutation') -> pd.DataFrame:
        """
        Compute global feature importance
        
        Args:
            X: Feature matrix
            method: 'permutation', 'shap', or 'native'
        """
        if method == 'permutation':
            return self._permutation_importance(X)
        elif method == 'shap':
            return self._shap_importance(X)
        elif method == 'native':
            return self._native_importance()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _permutation_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate permutation importance"""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, X, np.zeros(len(X)),  # y n'est pas utilisé pour permutation
            n_repeats=10, random_state=42
        )
        
        importance = pd.DataFrame({
            'feature': self.feature_names or X.columns.tolist(),
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.global_importance = importance
        return importance
    
    def _shap_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate SHAP-based global importance"""
        import shap
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        importance = pd.DataFrame({
            'feature': self.feature_names or X.columns.tolist(),
            'importance_mean': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance_mean', ascending=False)
        
        self.global_importance = importance
        return importance
    
    def _native_importance(self) -> pd.DataFrame:
        """Get native feature importance from tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            raise ValueError("Model does not have native feature importance")
    
    def plot_importance(self, top_n: int = 20, save_path: str = None):
        """Plot global feature importance"""
        if self.global_importance is None:
            raise ValueError("Must call explain() first")
        
        plt.figure(figsize=(10, 8))
        data = self.global_importance.head(top_n)
        
        sns.barplot(data=data, y='feature', x='importance_mean')
        plt.title(f'Top {top_n} Global Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()