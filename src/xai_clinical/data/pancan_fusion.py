
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MultiOmicsFusion:
    """Fusion multi-omiques: Expression + CNV + Clinique."""
    
    def __init__(self, 
                 expression_data: pd.DataFrame,
                 clinical_data: pd.DataFrame,
                 cnv_data: Optional[pd.DataFrame] = None,
                 fusion_strategy: str = "early"):
        self.expression = expression_data
        self.clinical = clinical_data
        self.cnv = cnv_data
        self.strategy = fusion_strategy
        self.fused_features = None
        self.scalers = {}
        
    def prepare_clinical_features(self, 
                                   categorical_cols: List[str] = None,
                                   numerical_cols: List[str] = None) -> pd.DataFrame:
        """Prépare les features cliniques (encodage, normalisation)."""
        df = self.clinical.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            exclude = ['patient_id', 'sample_id', 'vital_status', 'OS', 'OS.time']
            categorical_cols = [c for c in categorical_cols if c not in exclude]
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['OS', 'OS.time', 'DFS', 'DFS.time']
            numerical_cols = [c for c in numerical_cols if c not in exclude]
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        scaler = StandardScaler()
        for col in numerical_cols:
            if col in df.columns:
                df[col] = scaler.fit_transform(df[[col]])
        
        self.scalers['clinical'] = scaler
        feature_cols = [c for c in categorical_cols + numerical_cols if c in df.columns]
        return df[feature_cols]
    
    def reduce_expression_dimension(self, 
                                   n_components: int = 100,
                                   method: str = "pca") -> pd.DataFrame:
        """Réduit la dimensionnalité des données d'expression."""
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(self.expression)
            feature_names = [f"PC_expr_{i+1}" for i in range(n_components)]
            logger.info(f"PCA: variance expliquée = {reducer.explained_variance_ratio_.sum():.3f}")
        else:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.5)
            reduced = selector.fit_transform(self.expression)
            feature_names = self.expression.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(reduced, index=self.expression.index, columns=feature_names)
    
    def early_fusion(self,
                     clinical_features: pd.DataFrame,
                     expression_reduced: pd.DataFrame,
                     cnv_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Early fusion: concaténation simple."""
        common_idx = expression_reduced.index.intersection(clinical_features.index)
        fused = expression_reduced.loc[common_idx].copy()
        
        clinical_aligned = clinical_features.loc[common_idx]
        for col in clinical_aligned.columns:
            fused[f"clinical_{col}"] = clinical_aligned[col].values
        
        if cnv_features is not None and self.cnv is not None:
            cnv_aligned = cnv_features.loc[common_idx]
            for col in cnv_aligned.columns:
                fused[f"cnv_{col}"] = cnv_aligned[col].values
        
        logger.info(f"Early fusion: {fused.shape}")
        return fused
    
    def intermediate_fusion(self,
                         clinical_features: pd.DataFrame,
                         expression_reduced: pd.DataFrame,
                         cnv_features: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Intermediate fusion: features séparées pour modèle multi-input."""
        common_idx = expression_reduced.index.intersection(clinical_features.index)
        fused = {
            'expression': expression_reduced.loc[common_idx].values,
            'clinical': clinical_features.loc[common_idx].values
        }
        if cnv_features is not None:
            fused['cnv'] = cnv_features.loc[common_idx].values
        
        self.fused_features = fused
        logger.info(f"Intermediate: { {k: v.shape for k, v in fused.items()} }")
        return fused
    
    def late_fusion(self,
                   clinical_features: pd.DataFrame,
                   expression_reduced: pd.DataFrame,
                   cnv_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Late fusion: agrégation par modalité."""
        common_idx = expression_reduced.index.intersection(clinical_features.index)
        
        expr_agg = pd.DataFrame(index=common_idx)
        expr_agg['expr_mean'] = expression_reduced.loc[common_idx].mean(axis=1)
        expr_agg['expr_std'] = expression_reduced.loc[common_idx].std(axis=1)
        
        clin_agg = pd.DataFrame(index=common_idx)
        clin_agg['clinical_mean'] = clinical_features.loc[common_idx].mean(axis=1)
        
        fused = pd.concat([expr_agg, clin_agg], axis=1)
        
        if cnv_features is not None:
            cnv_agg = pd.DataFrame(index=common_idx)
            cnv_agg['cnv_mean'] = cnv_features.loc[common_idx].mean(axis=1)
            fused = pd.concat([fused, cnv_agg], axis=1)
        
        logger.info(f"Late fusion: {fused.shape}")
        return fused
    
    def fuse(self,
             clinical_features: pd.DataFrame,
             expression_reduced: pd.DataFrame,
             cnv_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Point d'entrée principal pour la fusion."""
        if self.strategy == "early":
            return self.early_fusion(clinical_features, expression_reduced, cnv_features)
        elif self.strategy == "intermediate":
            fused_dict = self.intermediate_fusion(clinical_features, expression_reduced, cnv_features)
            arrays = [fused_dict[k] for k in ['expression', 'clinical']]
            if 'cnv' in fused_dict:
                arrays.append(fused_dict['cnv'])
            combined = np.hstack(arrays)
            n_expr = fused_dict['expression'].shape[1]
            n_clin = fused_dict['clinical'].shape[1]
            cols = [f"expr_{i}" for i in range(n_expr)] + [f"clin_{i}" for i in range(n_clin)]
            if 'cnv' in fused_dict:
                cols += [f"cnv_{i}" for i in range(fused_dict['cnv'].shape[1])]
            return pd.DataFrame(combined, index=expression_reduced.index, columns=cols)
        elif self.strategy == "late":
            return self.late_fusion(clinical_features, expression_reduced, cnv_features)
        else:
            raise ValueError(f"Stratégie inconnue: {self.strategy}")


class FeatureSelector:
    """Sélection de features pour la prédiction du cancer."""
    
    def __init__(self, n_genes: int = 500, n_clinical: int = 10):
        self.n_genes = n_genes
        self.n_clinical = n_clinical
        self.selected_genes = None
        self.selected_clinical = None
        
    def select_genes_by_variance(self, expression: pd.DataFrame, y: pd.Series) -> List[str]:
        """Sélectionne les gènes les plus discriminants."""
        selector = SelectKBest(score_func=f_classif, k=min(self.n_genes, expression.shape[1]))
        selector.fit(expression, y)
        scores = pd.Series(selector.scores_, index=expression.columns)
        self.selected_genes = scores.nlargest(self.n_genes).index.tolist()
        logger.info(f"Top 10 gènes: {self.selected_genes[:10]}")
        return self.selected_genes
    
    def select_clinical_features(self, clinical: pd.DataFrame, y: pd.Series) -> List[str]:
        """Sélectionne les features cliniques les plus pertinentes."""
        selector = SelectKBest(score_func=mutual_info_classif, 
                              k=min(self.n_clinical, clinical.shape[1]))
        selector.fit(clinical, y)
        scores = pd.Series(selector.scores_, index=clinical.columns)
        self.selected_clinical = scores.nlargest(self.n_clinical).index.tolist()
        logger.info(f"Features cliniques: {self.selected_clinical}")
        return self.selected_clinical
    
    def get_selected_features(self, expression: pd.DataFrame, clinical: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Retourne le DataFrame avec uniquement les features sélectionnées."""
        genes = self.select_genes_by_variance(expression, y)
        clin = self.select_clinical_features(clinical, y)
        selected_expr = expression[genes]
        selected_clin = clinical[clin].add_prefix('clinical_')
        return pd.concat([selected_expr, selected_clin], axis=1)


class SurvivalTargetBuilder:
    """Construction de cibles de survie pour la prédiction."""
    
    @staticmethod
    def binary_survival(clinical_df: pd.DataFrame,
                       time_col: str = 'OS.time',
                       event_col: str = 'OS',
                       cutoff_months: float = 60) -> pd.Series:
        """Crée une cible binaire: décédé avant X mois vs survivant."""
        time = clinical_df[time_col] / 30.44
        event = clinical_df[event_col]
        target = ((event == 1) & (time <= cutoff_months)).astype(int)
        logger.info(f"Cible binaire (cutoff={cutoff_months}mo): {target.value_counts().to_dict()}")
        return target
    
    @staticmethod
    def risk_stratification(clinical_df: pd.DataFrame,
                           time_col: str = 'OS.time',
                           event_col: str = 'OS',
                           n_groups: int = 3) -> pd.Series:
        """Stratification du risque en groupes."""
        time = clinical_df[time_col] / 30.44
        event = clinical_df[event_col]
        risk_score = np.where(event == 1, 1 / (time + 1), 0)
        
        if n_groups == 2:
            threshold = np.median(risk_score)
            target = (risk_score > threshold).astype(int)
        else:
            target = pd.qcut(risk_score, q=n_groups, labels=range(n_groups))
        
        logger.info(f"Stratification ({n_groups} groupes): {pd.Series(target).value_counts().to_dict()}")
        return pd.Series(target, index=clinical_df.index)
