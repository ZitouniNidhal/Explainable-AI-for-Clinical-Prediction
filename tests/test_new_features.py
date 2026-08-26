import pytest
import pandas as pd
import numpy as np
from src.xai_clinical.data.pancan_fusion import MultiOmicsFusion, SurvivalTargetBuilder
from src.xai_clinical.data.preprocessor import DataPreprocessor

def test_autoencoder_reduction():
    # Setup simple expression dataframe
    np.random.seed(42)
    expr_data = pd.DataFrame(np.random.rand(10, 50), columns=[f"gene_{i}" for i in range(50)])
    clinical_data = pd.DataFrame(np.random.rand(10, 5), columns=[f"clin_{i}" for i in range(5)])
    
    fusion = MultiOmicsFusion(expression_data=expr_data, clinical_data=clinical_data)
    
    reduced = fusion.reduce_expression_dimension(n_components=5, method="autoencoder")
    
    assert reduced.shape == (10, 5)
    assert "AE_expr_1" in reduced.columns

def test_knn_sensitivity_analysis():
    class DummyConfig:
        class data:
            knn_sensitivity_range = [2, 3]
    
    config = DummyConfig()
    preprocessor = DataPreprocessor(config)
    
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 5), columns=[f"feat_{i}" for i in range(5)])
    # Add some NaNs
    X.iloc[0, 0] = np.nan
    X.iloc[5, 2] = np.nan
    y = pd.Series(np.random.randint(0, 2, size=20))
    
    res = preprocessor.run_knn_sensitivity_analysis(X, y, k_values=[2, 3])
    
    assert len(res) == 2
    assert "mean_auc" in res.columns
    assert res.iloc[0]["k"] == 2

def test_clinical_subtype_mapping():
    data = {
        'er_status': ['Positive', 'Negative', 'Positive', 'Negative'],
        'pr_status': ['Positive', 'Negative', 'Negative', 'Negative'],
        'her2_status': ['Negative', 'Positive', 'Positive', 'Negative']
    }
    df = pd.DataFrame(data)
    
    subtypes = SurvivalTargetBuilder.clinical_subtype_mapping(df)
    
    assert len(subtypes) == 4
    assert subtypes.iloc[0] == 0 # Luminal A
    assert subtypes.iloc[1] == 2 # HER2-enriched
    assert subtypes.iloc[2] == 1 # Luminal B
    assert subtypes.iloc[3] == 3 # Basal
