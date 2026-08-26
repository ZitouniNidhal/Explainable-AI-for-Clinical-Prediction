"""
clinical_features.py
====================
Extraction de micro-features granulaires pour différencier les patients.
Transforme les détails textuels (pourcentages, stades, ratios) en données numériques.
"""

import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

class GranularClinicalEncoder:
    """
    Extrait les 'petits détails' qui font la différence entre les patients.
    """
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # 1. Conversion des pourcentages IHC (ex: "50-59%" -> 0.545)
        for col in ['ER_STATUS_IHC_PERCENT_POSITIVE', 'PR_STATUS_IHC_PERCENT_POSITIVE', 'HER2_IHC_PERCENT_POSITIVE']:
            if col in df.columns:
                feat_name = col.replace('_STATUS_', '_').replace('_PERCENT_POSITIVE', '_val').lower()
                features[feat_name] = df[col].apply(self._parse_percent_range)

        # 2. Ratios de ganglions lymphatiques
        if 'LYMPH_NODE_EXAMINED_COUNT' in df.columns and 'LYMPH_NODES_EXAMINED_HE_COUNT' in df.columns:
            total = pd.to_numeric(df['LYMPH_NODE_EXAMINED_COUNT'], errors='coerce')
            pos = pd.to_numeric(df['LYMPH_NODES_EXAMINED_HE_COUNT'], errors='coerce')
            features['lymph_node_ratio'] = (pos / total).fillna(0)
            features['lymph_node_pos_count'] = pos.fillna(0)

        # 3. Décomposition numérique du TNM (T1, T2... -> 1, 2...)
        if 'AJCC_TUMOR_PATHOLOGIC_PT' in df.columns:
            features['t_stage_numeric'] = df['AJCC_TUMOR_PATHOLOGIC_PT'].apply(self._extract_digit)
        if 'AJCC_NODES_PATHOLOGIC_PN' in df.columns:
            features['n_stage_numeric'] = df['AJCC_NODES_PATHOLOGIC_PN'].apply(self._extract_digit)
        if 'AJCC_METASTASIS_PATHOLOGIC_PM' in df.columns:
            features['m_stage_numeric'] = df['AJCC_METASTASIS_PATHOLOGIC_PM'].apply(self._extract_digit)

        # 4. Sous-types histologiques spécifiques (One-Hot simple pour les plus fréquents)
        if 'HISTOLOGICAL_DIAGNOSIS' in df.columns:
            diag = df['HISTOLOGICAL_DIAGNOSIS'].astype(str).str.upper()
            features['is_ductal'] = diag.str.contains('DUCTAL').astype(int)
            features['is_lobular'] = diag.str.contains('LOBULAR').astype(int)
            features['is_mixed'] = (features['is_ductal'] & features['is_lobular']).astype(int)

        return features

    def _parse_percent_range(self, val) -> float:
        """Convertit '50-59%' ou '<10%' en float 0.0-1.0."""
        s = str(val).replace('%', '').strip()
        if 'NOT AVAILABLE' in s.upper() or 'NAN' in s.upper():
            return 0.0
        
        try:
            if '-' in s:
                low, high = map(float, s.split('-'))
                return (low + high) / 200.0
            if '<' in s:
                return float(s.replace('<', '')) / 200.0 # On prend la moitié du max
            if '>' in s:
                return float(s.replace('>', '')) / 100.0
            return float(s) / 100.0
        except:
            return 0.0

    def _extract_digit(self, val) -> int:
        """Extrait le premier chiffre d'une chaîne (ex: 'T1b' -> 1)."""
        match = re.search(r'\d', str(val))
        return int(match.group()) if match else 0

def add_granular_features(df: pd.DataFrame) -> pd.DataFrame:
    encoder = GranularClinicalEncoder()
    granular_feats = encoder.fit_transform(df)
    return pd.concat([df, granular_feats], axis=1)
