import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, Tuple, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Import local — importé ici pour éviter les imports circulaires
try:
    from .temporal_features import add_temporal_features
    from .clinical_features import add_granular_features
    _EXTENDED_FEATURES_AVAILABLE = True
except ImportError:
    _EXTENDED_FEATURES_AVAILABLE = False
    logger.warning("Modules de features étendues non disponibles")


class PANCANLoader:
    """
    Chargeur de données TCGA Pan-Cancer.
    
    Fichiers attendus dans data/raw/pancan/:
    - EBPlusPlusAdjustPANCAN.tsv (expression génique)
    - gdc_manifest.*.txt (manifest GDC)
    - TCGA_mastercalls.abs_tables_* (CNV/absolu)
    """
    
    def __init__(self, data_dir: str, cancer_type: str = "BRCA"):
        self.data_dir = Path(data_dir)
        self.cancer_type = cancer_type
        self.expression_data = None
        self.cnv_data = None
        self.clinical_data = None
        self.sample_ids = None
        
    def load_gene_expression(self, filepath: Optional[str] = None, top_genes: Optional[int] = 5000) -> pd.DataFrame:
        """Charge EBPlusPlusAdjustPANCAN.tsv et filtre par variance."""
        if filepath is None:
            candidates = list(self.data_dir.glob("EBPlusPlusAdjustPANCAN*")) + \
                        list(self.data_dir.glob("*geneEx*"))
            if not candidates:
                raise FileNotFoundError(f"Aucun fichier d'expression PANCAN trouvé dans {self.data_dir}")
            filepath = candidates[0]
        
        logger.info(f"Chargement PANCAN expression: {filepath}")
        
        # Lecture optimisée (les gènes sont en index, les patients en colonnes)
        df = pd.read_csv(filepath, sep='\t', index_col=0, compression='infer')
        df = df.T  # échantillons en lignes
        
        if self.cancer_type:
            sample_types = self._extract_cancer_types(df.index)
            mask = sample_types == self.cancer_type
            df = df[mask]
            logger.info(f"Filtrage {self.cancer_type}: {len(df)} échantillons")
        
        # Sélection des gènes par variance AVANT de continuer pour économiser la mémoire
        if top_genes and df.shape[1] > top_genes:
            logger.info(f"Sélection des {top_genes} gènes les plus variables...")
            variances = df.var()
            top_genes_idx = variances.nlargest(top_genes).index
            df = df[top_genes_idx]

        if df.max().max() > 100:
            logger.info("Application log2(x+1)")
            df = np.log2(df + 1)
        
        self.expression_data = df
        self.sample_ids = set(df.index)
        logger.info(f"Expression PANCAN finale: {df.shape}")
        return df
    
    def load_cnv_data(self, filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Charge TCGA_mastercalls.abs_tables."""
        if filepath is None:
            candidates = list(self.data_dir.glob("*mastercalls*")) + \
                        list(self.data_dir.glob("*abs_tables*"))
            if not candidates:
                logger.warning("Aucun fichier CNV PANCAN trouvé")
                return None
            filepath = candidates[0]
        
        logger.info(f"Chargement PANCAN CNV: {filepath}")
        df = pd.read_csv(filepath, sep='\t')
        
        id_col = next((c for c in ['Sample', 'sample', 'SampleID'] if c in df.columns), None)
        if id_col:
            df = df.set_index(id_col)
        
        if self.cancer_type and self.sample_ids:
            common = set(df.index) & self.sample_ids
            df = df.loc[list(common)]
        
        self.cnv_data = df
        logger.info(f"CNV PANCAN: {df.shape}")
        return df
    
    def load_clinical_from_manifest(self, filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Charge le manifest GDC."""
        if filepath is None:
            candidates = list(self.data_dir.glob("*manifest*"))
            if not candidates:
                logger.warning("Aucun manifest trouvé")
                return None
            filepath = candidates[0]
        
        logger.info(f"Chargement manifest: {filepath}")
        df = pd.read_csv(filepath, sep='\t')
        
        clinical_cols = ['case_id', 'case_submitter_id', 'primary_site', 
                        'disease_type', 'gender', 'race', 'ethnicity']
        available = [c for c in clinical_cols if c in df.columns]
        
        if available:
            df = df[available].drop_duplicates()
            if 'case_submitter_id' in df.columns:
                df = df.set_index('case_submitter_id')
        
        self.clinical_data = df
        logger.info(f"Clinical PANCAN: {df.shape}")
        return df
    
    def _extract_cancer_types(self, sample_ids: pd.Index) -> pd.Series:
        """Extrait le type de cancer depuis les IDs TCGA (TCGA-XX-XXXX)."""
        # Liste des codes TSS pour BRCA identifiés dans le dataset local
        brca_tss = {
            '3C', '4H', '5L', '5T', 'A1', 'A2', 'A7', 'A8', 'AC', 'AN', 'AO', 'AQ', 'AR', 
            'B6', 'BH', 'C8', 'D8', 'E2', 'E9', 'EW', 'GI', 'GM', 'HN', 'JL', 'LD', 'LL', 
            'LQ', 'MS', 'OK', 'OL', 'PE', 'PL', 'S3', 'UL', 'UU', 'V7', 'W8', 'WT', 'XX', 'Z7'
        }
        
        types = []
        for sid in sample_ids:
            parts = str(sid).split('-')
            if len(parts) >= 2:
                tss = parts[1]
                if self.cancer_type == "BRCA" and tss in brca_tss:
                    types.append("BRCA")
                else:
                    types.append(tss) # On garde le TSS par défaut
            else:
                types.append('UNKNOWN')
        return pd.Series(types, index=sample_ids)
    
    def get_common_samples(self) -> List[str]:
        """Retourne les échantillons communs entre toutes les modalités."""
        sets = []
        if self.expression_data is not None:
            sets.append(set(self.expression_data.index))
        if self.cnv_data is not None:
            sets.append(set(self.cnv_data.index))
        if self.clinical_data is not None:
            sets.append(set(self.clinical_data.index))
        
        if not sets:
            return []
        
        common = sets[0]
        for s in sets[1:]:
            common = common & s
        
        return sorted(list(common))
    
    def align_data(self) -> Dict[str, pd.DataFrame]:
        """Aligne toutes les modalités sur les échantillons communs."""
        common = self.get_common_samples()
        logger.info(f"Échantillons communs PANCAN: {len(common)}")
        
        aligned = {}
        if self.expression_data is not None:
            aligned['expression'] = self.expression_data.loc[common]
        if self.cnv_data is not None:
            aligned['cnv'] = self.cnv_data.loc[common]
        if self.clinical_data is not None:
            aligned['clinical'] = self.clinical_data.loc[common]
        
        return aligned
    
    def preprocess_expression(self, df: pd.DataFrame, 
                             min_variance: float = 0.5,
                             top_genes: Optional[int] = 5000) -> pd.DataFrame:
        """Prétraite l'expression génique PANCAN."""
        df = df.dropna(axis=1, thresh=len(df) * 0.8)
        df = df.fillna(df.median())
        
        variances = df.var()
        df = df.loc[:, variances > min_variance]
        
        if top_genes and len(df.columns) > top_genes:
            top = variances.nlargest(top_genes)
            df = df[top.index]
        
        df = (df - df.mean()) / df.std()
        df = df.fillna(0)
        
        logger.info(f"Expression PANCAN prétraitée: {df.shape}")
        return df
    
    def save_processed(self, output_dir: str, aligned_data: Dict[str, pd.DataFrame]):
        """Sauvegarde les données PANCAN prétraitées."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for modality, df in aligned_data.items():
            filepath = output_dir / f"pancan_{modality}_{self.cancer_type}.csv"
            df.to_csv(filepath)
            logger.info(f"Sauvegardé: {filepath}")


class BRCALoader:
    """
    Chargeur des données BRCA TCGA locales (format cBioPortal).
    
    Fichiers attendus dans data/raw/brca_tcga/:
    - data_clinical_patient.txt
    - data_clinical_sample.txt
    - data_mrna_seq_v2_rsem.txt
    - data_mutations.txt
    - data_cna.txt
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.clinical_patient = None
        self.clinical_sample = None
        self.mrna = None
        self.mutations = None
        self.cna = None
        
    def load_clinical_patient(self) -> pd.DataFrame:
        """Charge data_clinical_patient.txt (format cBioPortal)."""
        filepath = self.data_dir / "data_clinical_patient.txt"
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        # cBioPortal: métadonnées en commentaires (#)
        # On utilise PATIENT_ID comme index au lieu de la 1ère colonne (souvent un GUID)
        df = pd.read_csv(filepath, sep='\t', comment='#')
        if 'PATIENT_ID' in df.columns:
            df = df.set_index('PATIENT_ID')
        
        self.clinical_patient = df
        logger.info(f"Clinical patient: {df.shape}")
        return df
    
    def load_clinical_sample(self) -> pd.DataFrame:
        """Charge data_clinical_sample.txt."""
        filepath = self.data_dir / "data_clinical_sample.txt"
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        df = pd.read_csv(filepath, sep='\t', comment='#')
        if 'PATIENT_ID' in df.columns:
            df = df.set_index('PATIENT_ID')
        elif 'SAMPLE_ID' in df.columns:
            df = df.set_index('SAMPLE_ID')
            
        self.clinical_sample = df
        logger.info(f"Clinical sample: {df.shape}")
        return df
    
    def load_mrna_seq(self) -> Optional[pd.DataFrame]:
        """Charge data_mrna_seq_v2_rsem.txt."""
        filepath = self.data_dir / "data_mrna_seq_v2_rsem.txt"
        if not filepath.exists():
            candidates = list(self.data_dir.glob("*mrna*")) + list(self.data_dir.glob("*rsem*"))
            if candidates:
                filepath = candidates[0]
            else:
                logger.warning("Aucun fichier mRNA seq trouvé")
                return None
        
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        if df.shape[0] < df.shape[1]:
            df = df.T
        
        self.mrna = df
        logger.info(f"mRNA seq BRCA: {df.shape}")
        return df
    
    def load_mutations(self) -> Optional[pd.DataFrame]:
        """Charge data_mutations.txt."""
        filepath = self.data_dir / "data_mutations.txt"
        if not filepath.exists():
            candidates = list(self.data_dir.glob("*mutation*"))
            if candidates:
                filepath = candidates[0]
            else:
                logger.warning("Aucun fichier mutations trouvé")
                return None
        
        df = pd.read_csv(filepath, sep='\t', comment='#')
        self.mutations = df
        logger.info(f"Mutations: {df.shape}")
        return df
    
    def load_cna(self) -> Optional[pd.DataFrame]:
        """Charge data_cna.txt (Copy Number Alteration)."""
        filepath = self.data_dir / "data_cna.txt"
        if not filepath.exists():
            candidates = list(self.data_dir.glob("*cna*")) + list(self.data_dir.glob("*CNA*"))
            if candidates:
                filepath = candidates[0]
            else:
                logger.warning("Aucun fichier CNA trouvé")
                return None
        
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        self.cna = df
        logger.info(f"CNA: {df.shape}")
        return df
    
    def get_merged_clinical(self) -> pd.DataFrame:
        """Fusionne clinical_patient et clinical_sample."""
        if self.clinical_patient is None:
            self.load_clinical_patient()
        if self.clinical_sample is None:
            self.load_clinical_sample()
        
        merged = self.clinical_patient.merge(
            self.clinical_sample, 
            left_index=True, 
            right_index=True,
            how='outer',
            suffixes=('', '_sample')
        )
        
        logger.info(f"Clinical merged: {merged.shape}")
        return merged
    
    def extract_survival_target(self, 
                                clinical_df: pd.DataFrame,
                                time_col: str = 'OS_MONTHS',
                                event_col: str = 'OS_STATUS',
                                cutoff_months: float = 60) -> pd.Series:
        """
        Extrait une cible de survie binaire depuis les données cliniques BRCA.
        """
        time_candidates = ['OS_MONTHS', 'DFS_MONTHS', 'PFS_MONTHS', 'MONTHS_TO_LAST_FOLLOWUP']
        event_candidates = ['OS_STATUS', 'DFS_STATUS', 'PFS_STATUS', 'VITAL_STATUS']
        
        time_col_found = next((c for c in time_candidates if c in clinical_df.columns), None)
        event_col_found = next((c for c in event_candidates if c in clinical_df.columns), None)
        
        if time_col_found and event_col_found:
            # 1. Préparer les données
            clinical_df = clinical_df.copy()
            clinical_df.index = clinical_df.index.astype(str).str[:12].str.upper()
            
            time = pd.to_numeric(clinical_df[time_col_found], errors='coerce')
            event = clinical_df[event_col_found].astype(str)
            event_binary = event.str.contains('DECEASED|Dead|1|Progressed|Recurred', 
                                             case=False, na=False).astype(int)
            
            # 2. Cible Réelle Clinique (Gestion stricte de la censure)
            # Classe 1: Décès (ou événement) survenu AVANT ou À 60 mois
            # Classe 0: Suivi SANS événement pendant PLUS de 60 mois
            # NaN: Censure ambigüe (patient en vie mais suivi < 60 mois) -> exclus de l'étude binaire
            
            target = pd.Series(np.nan, index=clinical_df.index)
            target.loc[time > cutoff_months] = 0
            target.loc[(event_binary == 1) & (time <= cutoff_months)] = 1
            
            # 3. Dédoublonner par patient
            target = target.groupby(level=0).max().dropna()
            
            logger.info(f"Cible survie RÉELLE (cut-off {cutoff_months}m): {target.value_counts().to_dict()}")
            return target
        else:
            logger.warning("Colonnes de survie non trouvées")
            return pd.Series(np.nan, index=clinical_df.index)


class PANCANBRCAFusion:
    """
    Fusion des données PANCAN avec les données BRCA TCGA locales.
    """
    
    def __init__(self, pancan_loader: PANCANLoader, brca_loader: BRCALoader):
        self.pancan = pancan_loader
        self.brca = brca_loader
        self.fused_data = None
    
    def standardize_ids(self, df: pd.DataFrame, 
                       id_col: Optional[str] = None) -> pd.DataFrame:
        """Standardise les IDs au format TCGA court (12 caractères)."""
        df = df.copy()
        
        if id_col and id_col in df.columns:
            df.index = df[id_col].astype(str).str[:12]
        else:
            df.index = df.index.astype(str).str[:12]
        
        # S'assurer que l'index est en majuscules pour éviter les mismatches
        df.index = df.index.str.upper()
        
        df = df[~df.index.duplicated(keep='first')]
        return df
    
    def fuse_expression_clinical(self,
                                  pancan_expression: pd.DataFrame,
                                  brca_clinical: pd.DataFrame,
                                  target: Optional[pd.Series] = None,
                                  add_temporal: bool = True) -> pd.DataFrame:
        """
        Fusionne l'expression PANCAN avec les données cliniques BRCA.

        Parameters
        ----------
        add_temporal : bool
            Si True, ajoute les features temporelles (ordre des événements
            médicaux) via TemporalEventEncoder. Ces features capturent :
            - La rapidité de la récidive (précoce vs tardive)
            - L'ordre des traitements (néo-adjuvant vs adjuvant)
            - Le type de chirurgie initiale
            - Les changements de phénotype moléculaire (ex : ER+ → ER-)
        """
        pan_expr = self.standardize_ids(pancan_expression)
        brca_clin = self.standardize_ids(brca_clinical)

        # --- Ajout des features Étendues (Temporelles + Granulaires) ---
        if add_temporal and _EXTENDED_FEATURES_AVAILABLE:
            logger.info("[Features] Extraction des détails temporels et granulaires...")
            brca_clin = add_temporal_features(brca_clin)
            brca_clin = add_granular_features(brca_clin)
            logger.info(f"[Features] Total colonnes après extension: {len(brca_clin.columns)}")
        elif add_temporal and not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning("[Features] Modules non disponibles, fusion standard")

        common = pan_expr.index.intersection(brca_clin.index)
        logger.info(f"Échantillons communs PANCAN+BRCA: {len(common)}")

        if len(common) == 0:
            logger.error("Aucun échantillon commun trouvé!")
            return pd.DataFrame()

        fused = pan_expr.loc[common].merge(
            brca_clin.loc[common],
            left_index=True,
            right_index=True,
            how='inner'
        )

        if target is not None:
            target_std = target.copy()
            target_std.index = target_std.index.astype(str).str[:12].str.upper()
            target_std = target_std.groupby(level=0).max()

            common_target = target_std.index.intersection(fused.index)
            fused = fused.loc[common_target].copy()
            fused['TARGET'] = target_std.loc[common_target]

        self.fused_data = fused
        logger.info(f"Données fusionnées (avec temporel): {fused.shape}")
        return fused
    
    def prepare_ml_features(self,
                           fused_data: pd.DataFrame,
                           clinical_features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare X et y pour le machine learning."""
        if 'TARGET' in fused_data.columns:
            y = fused_data['TARGET']
            X = fused_data.drop(columns=['TARGET'])
        else:
            y = fused_data.iloc[:, -1]
            X = fused_data.iloc[:, :-1]
        
        # --- Nettoyage Impitoyable des Fuites (Leakage) ---
        # On supprime tout ce qui ressemble de près ou de loin à une information de suivi temporel
        bad_keywords = ['STATUS', 'MONTH', 'SURVIVAL', 'DEATH', 'DAYS', 'TIME', 'DFS', 'OS', 'PFS', 'DSS', 'VITAL', 'CONTACT', 'FOLLOWUP', 'NEOPLASM']
        
        leaks_found = []
        for c in X.columns:
            c_upper = c.upper()
            if any(k in c_upper for k in bad_keywords):
                # Exception : on garde l'âge
                if 'AGE' not in c_upper:
                    leaks_found.append(c)
        
        X = X.drop(columns=leaks_found)
        
        logger.info(f"[Sanitization] {len(leaks_found)} colonnes suspectes supprimées (Leakage Protection).")
        logger.info(f"Colonnes supprimées : {leaks_found[:10]}...")
        logger.info(f"Final Feature Space: {X.shape[1]} features")
        
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())
        y = y.dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"ML ready - X: {X.shape}, y: {y.value_counts().to_dict()}")
        return X, y
