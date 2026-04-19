"""
Clinical data preprocessing
"""
# Activer les features expérimentales de sklearn AVANT tous les imports sklearn
from sklearn.experimental import enable_iterative_imputer

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
        )
        return X_imputed
    
    def _transform_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation"""
        return pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def _fit_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and apply scaling"""
        scaling = self.config.preprocessing.scaling
        
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling}")
            
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
    
    def _transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling"""
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most informative features"""
        max_features = min(self.config.preprocessing.max_features, X.shape[1])
        
        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=max_features
        )
        
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        
        return pd.DataFrame(
            X_selected,
            columns=self.selected_features,
            index=X.index
        )
    
    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if self.selected_features:
            return X[self.selected_features]
        return X
    
    def _balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using SMOTE or other methods"""
        method = self.config.preprocessing.balance_method
        
        if method == "smote":
            sampler = SMOTE(random_state=42)
        elif method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
            
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        return (
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.Series(y_resampled, name=y.name)
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance if available"""
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'score': scores
            }).sort_values('score', ascending=False)
            return importance
        return None
        