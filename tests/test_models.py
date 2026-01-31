# tests/test_models.py
"""
Tests for models module
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.xai_clinical.models.classifiers import ClassifierFactory
from src.xai_clinical.models.trainer import ModelTrainer


class TestClassifierFactory:
    
    def test_create_classifiers(self):
        """Test classifier creation"""
        # Test each classifier type
        classifiers = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        for clf_name in classifiers:
            model = ClassifierFactory.create_classifier(clf_name, random_state=42)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_param_distributions(self):
        """Test parameter distributions"""
        classifiers = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        for clf_name in classifiers:
            params = ClassifierFactory.get_param_distributions(clf_name)
            assert isinstance(params, dict)
            assert len(params) > 0


class TestModelTrainer:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        return pd.DataFrame(X), pd.Series(y)
    
    def test_model_training(self, sample_data):
        """Test model training"""
        X, y = sample_data
        
        # Mock config
        class MockConfig:
            def __init__(self):
                class Models:
                    classifiers = [
                        {'name': 'logistic_regression', 'enabled': True, 'params': {}}
                    ]
                    class HyperparameterTuning:
                        method = "grid"
                        n_trials = 2
                        cv_folds = 2
                        scoring = "accuracy"
                    hyperparameter_tuning = HyperparameterTuning()
                self.models = Models()
        
        config = MockConfig()
        trainer = ModelTrainer(config, random_state=42)
        
        results = trainer.train_all_models(X, y)
        
        assert 'logistic_regression' in results
        assert 'model' in results['logistic_regression']
        assert 'metrics' in results['logistic_regression']