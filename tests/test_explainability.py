# tests/test_explainability.py
"""
Tests for explainability module
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from src.xai_clinical.explainability.shap_explainer import SHAPExplainer
from src.xai_clinical.explainability.lime_explainer import LIMEExplainer
from src.xai_clinical.explainability.stability_analyzer import StabilityAnalyzer


class TestSHAPExplainer:
    
    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data"""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X
    
    def test_shap_explainer_creation(self, sample_model_and_data):
        """Test SHAP explainer creation"""
        model, X = sample_model_and_data
        
        explainer = SHAPExplainer(model, X)
        assert explainer.explainer is not None
        assert explainer.expected_value is not None
    
    def test_shap_local_explanation(self, sample_model_and_data):
        """Test local SHAP explanation"""
        model, X = sample_model_and_data
        
        explainer = SHAPExplainer(model, X)
        explanation = explainer.explain_local(X, instance_idx=0)
        
        assert 'shap_values' in explanation
        assert 'base_value' in explanation
        assert 'prediction' in explanation
    
    def test_shap_global_explanation(self, sample_model_and_data):
        """Test global SHAP explanation"""
        model, X = sample_model_and_data
        
        explainer = SHAPExplainer(model, X)
        importance = explainer.explain_global(X)
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'shap_importance' in importance.columns


class TestLIMEExplainer:
    
    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data"""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X
    
    def test_lime_explainer_creation(self, sample_model_and_data):
        """Test LIME explainer creation"""
        model, X = sample_model_and_data
        
        explainer = LIMEExplainer(model, X)
        assert explainer.explainer is not None
    
    def test_lime_explanation(self, sample_model_and_data):
        """Test LIME explanation"""
        model, X = sample_model_and_data
        
        explainer = LIMEExplainer(model, X)
        explanation = explainer.explain_instance(X.iloc[0])
        
        assert 'explanation' in explanation
        assert 'features_weights' in explanation


class TestStabilityAnalyzer:
    
    @pytest.fixture
    def sample_explainer_and_data(self):
        """Create sample explainer and data"""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        shap_explainer = SHAPExplainer(model, X)
        
        return shap_explainer, X
    
    def test_stability_evaluation(self, sample_explainer_and_data):
        """Test stability evaluation"""
        explainer, X = sample_explainer_and_data
        
        analyzer = StabilityAnalyzer(explainer, X.columns.tolist())
        results = analyzer.evaluate_stability(X, n_perturbations=3)
        
        assert 'gaussian_noise' in results
        assert 'missing_values' in results
        assert 'baseline_explanation' in results