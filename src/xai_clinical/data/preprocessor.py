"""
Clinical data preprocessing
"""

# Activer les features expérimentales de sklearn AVANT tous les imports sklearn
from sklearn.experimental import enable_iterative_imputer

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
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

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> Tuple:
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
            logger.info(
                f"Balancing classes ({self.config.preprocessing.balance_method})..."
            )
            X_train_processed, y_train = self._balance_classes(
                X_train_processed, y_train
            )

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
            self.imputer = SimpleImputer(strategy="median")
        elif strategy == "iterative":
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        elif strategy == "knn":
            k = getattr(self.config.data, 'knn_imputer_k', 5)
            self.imputer = KNNImputer(n_neighbors=k)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )
        return X_imputed

    def _transform_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation"""
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)

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
            self.scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        return X_scaled

    def _transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling"""
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most informative features"""
        method = getattr(self.config.preprocessing, 'feature_selection', 'f_classif')
        max_features = min(getattr(self.config.preprocessing, 'max_features', 50), X.shape[1])

        if method == "mutual_info":
            score_func = mutual_info_classif
        elif method == "chi2":
            from sklearn.feature_selection import chi2
            score_func = chi2
        else:
            from sklearn.feature_selection import f_classif
            score_func = f_classif

        selector = SelectKBest(score_func=score_func, k=max_features)
        
        # S'assurer que les données sont positives pour chi2 si nécessaire
        X_input = X.copy()
        if method == "chi2" and (X_input < 0).any().any():
            X_input = X_input - X_input.min().min()

        X_selected = selector.fit_transform(X_input, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector

        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)

    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if self.selected_features:
            return X[self.selected_features]
        return X

    def _balance_classes(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
            pd.Series(y_resampled, name=y.name, index=range(len(y_resampled))),
        )


    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance if available"""
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            importance = pd.DataFrame(
                {"feature": self.feature_names, "score": scores}
            ).sort_values("score", ascending=False)
            return importance
        return None

    def run_knn_sensitivity_analysis(self, X: pd.DataFrame, y: pd.Series, k_values: List[int] = None) -> pd.DataFrame:
        """
        Analyse de sensibilité : Comment le choix de k impacte l'AUC du modèle en aval.
        Répond à la critique sur le manque de justification du choix de k.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        if k_values is None:
            k_values = getattr(self.config.data, 'knn_sensitivity_range', [3, 5, 7, 9, 15])
        
        logger.info(f"Démarrage de l'analyse de sensibilité k-NN pour k={k_values}...")
        results = []
        
        for k in k_values:
            # 1. Imputation avec k
            imputer = KNNImputer(n_neighbors=k)
            X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # 2. Entraînement et évaluation d'un modèle de base (RandomForest)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Gestion du cas où on a des classes trop petites pour cv=5
            cv_folds = min(5, y.value_counts().min())
            if cv_folds < 2:
                # Si les classes sont extrêmement déséquilibrées (ex: 1 seul exemple)
                logger.warning(f"Classes trop peu représentées pour CV (min {cv_folds}). On saute l'évaluation.")
                auc_mean = np.nan
            else:
                try:
                    scores = cross_val_score(clf, X_imp, y, cv=cv_folds, scoring='roc_auc')
                    auc_mean = scores.mean()
                except Exception as e:
                    logger.warning(f"Erreur lors de l'évaluation CV pour k={k}: {e}")
                    auc_mean = np.nan
            
            results.append({
                'k': k,
                'mean_auc': auc_mean
            })
            
        sensitivity_df = pd.DataFrame(results)
        logger.info(f"Analyse de sensibilité terminée:\n{sensitivity_df}")
        return sensitivity_df


class ClinicalPreprocessor:
    """
    Simplified preprocessor for notebooks (v2 pipeline)
    """

    def __init__(self, n_features: int = 100):
        self.n_features = n_features
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selected_features = None

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> Tuple:
        """Fit on train and transform everything."""
        # Ensure only numeric
        X_train_num = X_train.select_dtypes(include=[np.number])
        
        # Impute
        X_train_imp = pd.DataFrame(self.imputer.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
        
        # Scale
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train_imp), columns=X_train_imp.columns, index=X_train_imp.index)
        
        # Select
        k = min(self.n_features, X_train_scaled.shape[1])
        self.selector.set_params(k=k)
        X_train_sel = self.selector.fit_transform(X_train_scaled, y_train)
        self.selected_features = X_train_scaled.columns[self.selector.get_support()].tolist()
        
        X_train_final = pd.DataFrame(X_train_sel, columns=self.selected_features, index=X_train.index)
        
        # We don't transform val/test here in this specific return signature but we return transformed train
        return X_train_final, y_train

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations."""
        X_num = X.select_dtypes(include=[np.number])
        X_imp = self.imputer.transform(X_num)
        X_scaled = self.scaler.transform(X_imp)
        X_sel = X_scaled[:, self.selector.get_support()]
        return pd.DataFrame(X_sel, columns=self.selected_features, index=X.index)
