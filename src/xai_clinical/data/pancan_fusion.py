
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
                 fusion_strategy: str = "early",
                 weights: Optional[Dict[str, float]] = None):
        self.expression = expression_data
        self.clinical = clinical_data
        self.cnv = cnv_data
        self.strategy = fusion_strategy
        # Pondération des modalités (réponse à la critique de dominance de couche)
        self.weights = weights or {"expression": 1.0, "clinical": 1.0, "cnv": 1.0}
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
        if method == "autoencoder":
            reduced = self._train_autoencoder(self.expression, n_components)
            feature_names = [f"AE_expr_{i+1}" for i in range(n_components)]
            logger.info("Autoencoder: Réduction non-linéaire terminée.")
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(self.expression)
            feature_names = [f"PC_expr_{i+1}" for i in range(n_components)]
            logger.info(f"PCA: variance expliquée = {reducer.explained_variance_ratio_.sum():.3f}")
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(self.expression)
                feature_names = [f"UMAP_expr_{i+1}" for i in range(n_components)]
                logger.info("UMAP: Réduction non-linéaire terminée.")
            except ImportError:
                logger.warning("UMAP non installé. Repli sur PCA.")
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(self.expression)
                feature_names = [f"PC_expr_{i+1}" for i in range(n_components)]
        else:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.5)
            reduced = selector.fit_transform(self.expression)
            feature_names = self.expression.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(reduced, index=self.expression.index, columns=feature_names)

    def _train_autoencoder(self, X: pd.DataFrame, n_components: int) -> np.ndarray:
        """Entraîne un auto-encodeur simple pour la réduction de dimension."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Architecture bottleneck simple (ex: Input -> n_components -> Output)
        # sklearn.neural_network n'a pas de moyen direct d'extraire la couche cachée
        # Nous pouvons l'émuler avec un petit réseau pour l'instant ou utiliser les poids.
        # Une approche plus rigoureuse avec scikit-learn :
        # L'Autoencoder n'est pas nativement supporté pour extraction de features via MLPRegressor sans hack,
        # donc on utilise les activations de la première couche après un fit :
        
        ae = MLPRegressor(hidden_layer_sizes=(n_components,), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=200, 
                          random_state=42)
        ae.fit(X_scaled, X_scaled)
        
        # Obtenir les activations de la couche cachée (n_samples, n_components)
        # La matrice de poids W1 est ae.coefs_[0] de taille (n_features, n_components)
        # Le biais b1 est ae.intercepts_[0] de taille (n_components,)
        def relu(x): return np.maximum(0, x)
        hidden_activations = relu(np.dot(X_scaled, ae.coefs_[0]) + ae.intercepts_[0])
        
        return hidden_activations

    
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
        self._check_layer_dominance(fused)
        logger.info(f"Intermediate: { {k: v.shape for k, v in fused.items()} }")
        return fused

    def _check_layer_dominance(self, fused_dict: Dict[str, np.ndarray]):
        """
        Diagnostique si une couche écrase les autres (répond à la critique sur la fusion).
        Calcule la norme moyenne des vecteurs par modalité.
        """
        logger.info("--- Diagnostic de Dominance de Couche ---")
        for modality, data in fused_dict.items():
            norm = np.linalg.norm(data, axis=1).mean()
            logger.info(f"  - {modality}: Norme moyenne = {norm:.4f}")
    
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

            # Application des poids pour éviter l'écrasement (Réponse à la critique 3)
            # self.weights est défini dans __init__ (plus de dépendance à self.config)
            arrays = []
            if 'expression' in fused_dict:
                w_expr = self.weights.get('expression', 1.0)
                arrays.append(fused_dict['expression'] * w_expr)
            if 'clinical' in fused_dict:
                w_clin = self.weights.get('clinical', 1.0)
                arrays.append(fused_dict['clinical'] * w_clin)
            if 'cnv' in fused_dict:
                w_cnv = self.weights.get('cnv', 1.0)
                arrays.append(fused_dict['cnv'] * w_cnv)

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
                       time_col: str = 'OS_MONTHS',
                       event_col: str = 'OS_STATUS',
                       cutoff_months: float = 60) -> pd.Series:
        """Crée une cible binaire: décédé avant X mois vs survivant."""
        # Détection automatique des colonnes si les défauts ne sont pas présents
        if time_col not in clinical_df.columns:
            time_col = next((c for c in ['OS.time', 'OS_MONTHS', 'days_to_death'] if c in clinical_df.columns), time_col)
        if event_col not in clinical_df.columns:
            event_col = next((c for c in ['OS', 'OS_STATUS', 'vital_status'] if c in clinical_df.columns), event_col)
            
        if time_col not in clinical_df.columns or event_col not in clinical_df.columns:
            logger.warning(f"Colonnes de survie non trouvées: {time_col}, {event_col}")
            return pd.Series(np.nan, index=clinical_df.index)

        time = pd.to_numeric(clinical_df[time_col], errors='coerce')
        # Conversion en mois si c'est en jours
        if time.max() > 1000: # Probablement en jours
            time = time / 30.44
            
        event = clinical_df[event_col].astype(str)
        event_binary = event.str.contains('DECEASED|Dead|1|Progressed|Recurred', 
                                         case=False, na=False).astype(int)
        
        target = ((event_binary == 1) & (time <= cutoff_months)).astype(int)
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

    @staticmethod
    def clinical_subtype_mapping(clinical_df: pd.DataFrame) -> pd.Series:
        """
        Cartographie des sous-types cliniques du cancer du sein (PAM50-like)
        basée sur les statuts ER, PR, et HER2.
        """
        # Initialisation avec des NaN
        subtypes = pd.Series(np.nan, index=clinical_df.index)
        
        # Détection des colonnes pertinentes
        er_col = next((c for c in clinical_df.columns if 'er_status' in c.lower() or 'breast_carcinoma_estrogen_receptor_status' in c.lower()), None)
        pr_col = next((c for c in clinical_df.columns if 'pr_status' in c.lower() or 'breast_carcinoma_progesterone_receptor_status' in c.lower()), None)
        her2_col = next((c for c in clinical_df.columns if 'her2_status' in c.lower() or 'lab_proc_her2_neu_immunohistochemistry_receptor_status' in c.lower()), None)
        
        if er_col and pr_col and her2_col:
            er_pos = clinical_df[er_col].astype(str).str.contains('Positive', case=False, na=False)
            pr_pos = clinical_df[pr_col].astype(str).str.contains('Positive', case=False, na=False)
            her2_pos = clinical_df[her2_col].astype(str).str.contains('Positive', case=False, na=False)
            
            # Luminal A: ER+ and/or PR+, HER2- (Simplified without Ki-67)
            luminal_a = (er_pos | pr_pos) & (~her2_pos)
            # Luminal B: ER+ and/or PR+, HER2+ (Or high Ki-67 but we simplify here)
            luminal_b = (er_pos | pr_pos) & her2_pos
            # HER2-enriched: ER-, PR-, HER2+
            her2_enriched = (~er_pos) & (~pr_pos) & her2_pos
            # Basal-like / Triple Negative: ER-, PR-, HER2-
            basal = (~er_pos) & (~pr_pos) & (~her2_pos)
            
            subtypes[luminal_a] = 0 # Luminal A
            subtypes[luminal_b] = 1 # Luminal B
            subtypes[her2_enriched] = 2 # HER2+
            subtypes[basal] = 3 # Basal
            
            logger.info(f"Sous-types cliniques identifiés: {subtypes.value_counts().to_dict()}")
        else:
            logger.warning("Colonnes ER/PR/HER2 non trouvées pour la classification des sous-types.")
            
        return subtypes
