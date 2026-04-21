# tests/test_data.py
"""
Tests for data module
"""

import pytest
import pandas as pd
import numpy as np
from src.xai_clinical.data.synthetic_generator import SyntheticDataGenerator
from src.xai_clinical.data.preprocessor import DataPreprocessor
from src.xai_clinical.config import Config


class TestSyntheticDataGenerator:

    def test_generate_data(self):
        """Test synthetic data generation"""
        generator = SyntheticDataGenerator(n_samples=100, random_state=42)
        X, y = generator.generate()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        assert set(y.unique()) == {0, 1}

    def test_feature_descriptions(self):
        """Test feature descriptions"""
        generator = SyntheticDataGenerator()
        descriptions = generator.get_feature_descriptions()

        assert isinstance(descriptions, dict)
        assert "age" in descriptions
        assert "sex" in descriptions


class TestDataPreprocessor:

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "age": np.random.normal(65, 10, 100),
                "sex": np.random.binomial(1, 0.5, 100),
                "bmi": np.random.normal(27, 5, 100),
            }
        )
        y = pd.Series(np.random.binomial(1, 0.3, 100))
        return X, y

    def test_preprocessing(self, sample_data):
        """Test preprocessing pipeline"""
        X, y = sample_data

        # Create mock config
        class MockConfig:
            def __init__(self):
                class Data:
                    imputation_strategy = "simple"

                class Preprocessing:
                    scaling = "standard"
                    balance_classes = False
                    feature_selection = None
                    max_features = 10
                    balance_method = "smote"

                self.data = Data()
                self.preprocessing = Preprocessing()

        config = MockConfig()
        preprocessor = DataPreprocessor(config)

        X_train_proc, y_train_proc = preprocessor.fit_transform(X, y)

        assert isinstance(X_train_proc, pd.DataFrame)
        assert isinstance(y_train_proc, pd.Series)
        assert X_train_proc.shape[0] == X.shape[0]
