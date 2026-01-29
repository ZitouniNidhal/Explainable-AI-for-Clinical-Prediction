# src/xai_clinical/data/preprocessor.py
"""
Clinical data preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Complete preprocessing pipeline for clinical data
    """
    
    def __init__(self, config):
        self.config = config
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.sampler = None
        self.feature_names = None
        self.selected_features = None
        
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     X_test: Optional[pd.DataFrame] = None) -> Tuple:
        """
        Fit and transform training data, then transform val/test
        """
        self.feature_names = X_train.columns.tolist()
        X_train_processed = X_train.copy()
        
        # 1. Missing values handling
        logger.info("Imputing missing values...")
        X_train_processed = self._fit_impute(X_train_processed)
        if X_val is not None:
            X_val_processed = self._transform_impute(X_val.copy())
        if X_test is not None:
            X_test_processed = self._transform_impute(X_test.copy())
            
        # 2. Normalization
        logger.info(f"Scaling ({self.config.preprocessing.scaling})...")
        X_train_processed = self._fit_scale(X_train_processed)
        if X_val is not None:
            X_val_processed = self._transform_scale(X_val_processed)
        if X_test is not None:
            X_test_processed = self._transform_scale(X_test_processed)
            
        # 3. Feature selection
        if self.config.preprocessing.feature_selection:
            logger.info("Feature selection...")
            X_train_processed = self._fit_feature_selection(X_train_processed, y_train)
            if X_val is not None:
                X_val_processed = self._transform_feature_selection(X_val_processed)
            if X_test is not None:
                X_test_processed = self._transform_feature_selection(X_test_processed)
                
        # 4. Class balancing (train only)
        if self.config.preprocessing.balance_classes:
            logger.info(f"Balancing classes ({self.config.preprocessing.balance_method})...")
            X_train_processed, y_train = self._balance_classes(X_train_processed, y_train)
            
        results = [X_train_processed, y_train]
        if X_val is not None:
            results.append(X_val_processed)
        if X_test is not None:
            results.append(X_test_processed)
            
        return tuple(results) if len(results) > 2 else (results[0], results[1])
    
    def _fit_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply imputation"""
        strategy = self.config.data.imputation_strategy
        
        if strategy == "simple":
            self.imputer = SimpleImputer(strategy='median')
        elif strategy == "iterative":
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        elif strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
            
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
       