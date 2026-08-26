"""
temporal_features.py
====================
Extraction de features temporelles / séquentielles à partir des données
cliniques TCGA-BRCA (cBioPortal format).

Problème résolu : L'ORDRE des événements médicaux change le risque prédit.
  - Récidive précoce (6 mois) ≠ récidive tardive (48 mois)
  - Traitement néo-adjuvant AVANT chirurgie ≠ traitement adjuvant APRÈS
  - Nouvelle tumeur survenant tôt = pronostic différent

Ce module encode la SÉQUENCE et la DURÉE inter-événements comme features ML.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class TemporalEventEncoder:
    """
    Encode les événements cliniques séquentiels en features numériques
    exploitables par les modèles ML.

    Événements disponibles dans TCGA-BRCA :
    - Diagnostic initial         (t=0, référence)
    - Chirurgie (type)           (SURGICAL_PROCEDURE_FIRST)
    - Traitement adjuvant        (PHARMACEUTICAL_TX_ADJUVANT, RADIATION_TREATMENT_ADJUVANT)
    - Nouvelle tumeur après trt  (NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT)
    - Progression / Récidive     (DFS_STATUS, DFS_MONTHS)
    - Décès / Fin de suivi       (OS_STATUS, OS_MONTHS)
    """

    # Valeurs manquantes TCGA
    MISSING_VALUES = {"[Not Available]", "[Not Applicable]", "[Unknown]", "nan", ""}

    def __init__(self):
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def fit_transform(self, clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme le DataFrame clinique brut en features temporelles.

        Parameters
        ----------
        clinical_df : pd.DataFrame
            Données cliniques issues de BRCALoader.get_merged_clinical()

        Returns
        -------
        pd.DataFrame
            Nouvelles features temporelles, alignées sur le même index.
        """
        df = clinical_df.copy()
        features = pd.DataFrame(index=df.index)

        # 1. Features de séquence des traitements
        features = pd.concat([features, self._encode_treatment_sequence(df)], axis=1)

        # 2. Features temporelles de survie / progression
        features = pd.concat([features, self._encode_survival_dynamics(df)], axis=1)

        # 3. Features de la nouvelle tumeur (événement post-traitement)
        features = pd.concat([features, self._encode_new_tumor_event(df)], axis=1)

        # 4. Features de la chirurgie initiale
        features = pd.concat([features, self._encode_surgical_event(df)], axis=1)

        # 5. Features d'âge au diagnostic
        features = pd.concat([features, self._encode_age_at_events(df)], axis=1)

        # 6. Score composite de temporalité
        features = pd.concat([features, self._compute_temporal_risk_score(features)], axis=1)

        self.feature_names_ = features.columns.tolist()
        logger.info(f"[TemporalEncoder] {len(self.feature_names_)} features temporelles créées")
        return features

    # ------------------------------------------------------------------
    # Encodeurs spécifiques
    # ------------------------------------------------------------------

    def _encode_treatment_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode l'ordre et la combinaison des traitements.

        Logique : un traitement néo-adjuvant (AVANT chirurgie) indique un
        stade avancé à la présentation → risque plus élevé que le traitement
        adjuvant seul (APRÈS chirurgie sur tumeur résécable).
        """
        feat = pd.DataFrame(index=df.index)

        # Traitement néo-adjuvant (avant chirurgie) = facteur de risque élevé
        if "HISTORY_NEOADJUVANT_TRTYN" in df.columns:
            feat["had_neoadjuvant_treatment"] = (
                df["HISTORY_NEOADJUVANT_TRTYN"]
                .astype(str)
                .str.upper()
                .eq("YES")
                .astype(int)
            )
        else:
            feat["had_neoadjuvant_treatment"] = 0

        # Traitement adjuvant pharmaceutique (après chirurgie)
        if "PHARMACEUTICAL_TX_ADJUVANT" in df.columns:
            feat["had_pharma_adjuvant"] = (
                df["PHARMACEUTICAL_TX_ADJUVANT"]
                .astype(str)
                .str.upper()
                .eq("YES")
                .astype(int)
            )
        else:
            feat["had_pharma_adjuvant"] = 0

        # Radiothérapie adjuvante
        if "RADIATION_TREATMENT_ADJUVANT" in df.columns:
            feat["had_radiation_adjuvant"] = (
                df["RADIATION_TREATMENT_ADJUVANT"]
                .astype(str)
                .str.upper()
                .eq("YES")
                .astype(int)
            )
        else:
            feat["had_radiation_adjuvant"] = 0

        # Séquence : néo-adjuvant ET adjuvant = traitement intensif = risque élevé initial
        feat["intensive_treatment_sequence"] = (
            feat["had_neoadjuvant_treatment"] & feat["had_pharma_adjuvant"]
        ).astype(int)

        # Nombre total de modalités de traitement reçues
        feat["n_treatment_modalities"] = (
            feat["had_neoadjuvant_treatment"]
            + feat["had_pharma_adjuvant"]
            + feat["had_radiation_adjuvant"]
        )

        return feat

    def _encode_survival_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode la dynamique temporelle de survie globale (OS) et
        sans progression (DFS).

        KEY INSIGHT : La VITESSE de progression est plus informative que
        la présence/absence de progression :
        - DFS_MONTHS faible + progression = récidive précoce (très mauvais)
        - DFS_MONTHS élevé + progression = récidive tardive (moins grave)
        """
        feat = pd.DataFrame(index=df.index)

        # --- OS (Overall Survival) ---
        if "OS_MONTHS" in df.columns:
            os_months = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
            feat["os_months"] = os_months.fillna(os_months.median())

            # Buckets temporels de survie (encode la rapidité de l'événement)
            feat["os_very_early"] = (os_months <= 12).astype(int)   # < 1 an
            feat["os_early"]      = ((os_months > 12) & (os_months <= 36)).astype(int)
            feat["os_mid"]        = ((os_months > 36) & (os_months <= 60)).astype(int)
            feat["os_late"]       = (os_months > 60).astype(int)    # > 5 ans

        if "OS_STATUS" in df.columns:
            os_event = df["OS_STATUS"].astype(str)
            feat["os_event_occurred"] = os_event.str.contains(
                "DECEASED|Dead|1:", case=False, na=False
            ).astype(int)

            # Interaction CRITIQUE : décédé tôt vs tard
            if "os_months" in feat.columns:
                feat["early_death"] = (
                    (feat["os_event_occurred"] == 1) & (feat["os_months"] <= 24)
                ).astype(int)
                feat["late_death"] = (
                    (feat["os_event_occurred"] == 1) & (feat["os_months"] > 60)
                ).astype(int)

        # --- DFS (Disease-Free Survival = Survie sans récidive) ---
        if "DFS_MONTHS" in df.columns:
            dfs_months = pd.to_numeric(df["DFS_MONTHS"], errors="coerce")
            feat["dfs_months"] = dfs_months.fillna(dfs_months.median())

            feat["dfs_very_early"] = (dfs_months <= 6).astype(int)   # < 6 mois
            feat["dfs_early"]      = ((dfs_months > 6) & (dfs_months <= 24)).astype(int)
            feat["dfs_mid"]        = ((dfs_months > 24) & (dfs_months <= 60)).astype(int)
            feat["dfs_late"]       = (dfs_months > 60).astype(int)

        if "DFS_STATUS" in df.columns:
            dfs_event = df["DFS_STATUS"].astype(str)
            feat["dfs_event_occurred"] = dfs_event.str.contains(
                "Recurred|Progressed|1:", case=False, na=False
            ).astype(int)

            # Récidive précoce = facteur pronostique majeur
            if "dfs_months" in feat.columns:
                feat["early_recurrence"] = (
                    (feat["dfs_event_occurred"] == 1) & (feat["dfs_months"] <= 12)
                ).astype(int)
                feat["very_early_recurrence"] = (
                    (feat["dfs_event_occurred"] == 1) & (feat["dfs_months"] <= 6)
                ).astype(int)

        # --- Gap OS - DFS (temps entre récidive et décès) ---
        if "os_months" in feat.columns and "dfs_months" in feat.columns:
            gap = feat["os_months"] - feat["dfs_months"]
            feat["os_dfs_gap_months"] = gap.clip(lower=0)  # Ne peut pas être négatif
            feat["rapid_death_after_recurrence"] = (
                (feat.get("dfs_event_occurred", pd.Series(0, index=feat.index)) == 1)
                & (gap <= 6)
            ).astype(int)

        return feat

    def _encode_new_tumor_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode l'événement 'nouvelle tumeur après traitement initial'.

        C'est un marqueur SÉQUENTIEL important : il survient APRÈS le
        traitement, indiquant une résistance ou une seconde tumeur primaire.
        """
        feat = pd.DataFrame(index=df.index)

        if "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT" in df.columns:
            nte = df["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].astype(str).str.upper()
            feat["new_tumor_after_treatment"] = nte.eq("YES").astype(int)

            # Statut ER/PR/HER2 de la nouvelle tumeur (changement de phénotype = résistance)
            nte_cols = {
                "NTE_ER_STATUS": "nte_er_positive",
                "NTE_PR_STATUS_BY_IHC": "nte_pr_positive",
                "NTE_HER2_STATUS": "nte_her2_positive",
            }
            for src_col, feat_name in nte_cols.items():
                if src_col in df.columns:
                    feat[feat_name] = (
                        df[src_col].astype(str).str.upper().eq("POSITIVE").astype(int)
                    )

            # Changement de phénotype moléculaire (ER+ → ER- = résistance endocrine)
            if "ER_STATUS_BY_IHC" in df.columns and "NTE_ER_STATUS" in df.columns:
                original_er_pos = df["ER_STATUS_BY_IHC"].astype(str).str.upper().eq("POSITIVE")
                nte_er_neg = df["NTE_ER_STATUS"].astype(str).str.upper().eq("NEGATIVE")
                feat["er_phenotype_switch"] = (original_er_pos & nte_er_neg).astype(int)

        else:
            feat["new_tumor_after_treatment"] = 0

        return feat

    def _encode_surgical_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode le type de la première intervention chirurgicale.

        L'ordre : mastectomie radicale = chirurgie agressive d'emblée
        (stade avancé) vs lumpectomie = chirurgie conservatrice (stade précoce).
        """
        feat = pd.DataFrame(index=df.index)

        if "SURGICAL_PROCEDURE_FIRST" in df.columns:
            surgery = df["SURGICAL_PROCEDURE_FIRST"].astype(str).str.upper()

            # Types de chirurgie par agressivité croissante
            feat["surgery_lumpectomy"] = surgery.str.contains(
                "LUMPECTOMY|EXCISION|PARTIAL", na=False
            ).astype(int)

            feat["surgery_simple_mastectomy"] = surgery.str.contains(
                "SIMPLE MASTECTOMY|TOTAL MASTECTOMY", na=False
            ).astype(int)

            feat["surgery_radical_mastectomy"] = surgery.str.contains(
                "RADICAL|MODIFIED RADICAL", na=False
            ).astype(int)

            # Score d'agressivité chirurgicale (0 = conservative, 2 = radicale)
            feat["surgery_aggressiveness_score"] = (
                feat["surgery_lumpectomy"] * 0
                + feat["surgery_simple_mastectomy"] * 1
                + feat["surgery_radical_mastectomy"] * 2
            )

        # Marges chirurgicales positives = risque de récidive locale
        if "PATH_MARGIN" in df.columns:
            margins = df["PATH_MARGIN"].astype(str).str.upper()
            feat["positive_surgical_margins"] = margins.str.contains(
                "POSITIVE|INVOLVED", na=False
            ).astype(int)
            feat["negative_surgical_margins"] = margins.eq("NEGATIVE").astype(int)

        return feat

    def _encode_age_at_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode l'âge comme facteur temporel.
        L'âge au diagnostic conditionne le risque différemment selon la
        tranche : jeunes femmes (< 40) = formes plus agressives, personnes
        âgées (> 70) = tolèrent moins les traitements.
        """
        feat = pd.DataFrame(index=df.index)

        if "AGE" in df.columns:
            age = pd.to_numeric(df["AGE"], errors="coerce")
            median_age = age.dropna().median() if not age.dropna().empty else 60
            age = age.fillna(median_age)

            feat["age_at_diagnosis"] = age
            feat["is_very_young"]    = (age < 35).astype(int)   # Formes agressives
            feat["is_young"]         = ((age >= 35) & (age < 50)).astype(int)
            feat["is_middle_aged"]   = ((age >= 50) & (age < 65)).astype(int)
            feat["is_elderly"]       = (age >= 65).astype(int)  # Tolérance réduite

            # Interaction âge × ménopause
            if "MENOPAUSE_STATUS" in df.columns:
                meno = df["MENOPAUSE_STATUS"].astype(str).str.upper()
                is_pre_meno = meno.str.contains("PRE", na=False).astype(int)
                feat["young_premenopausal"] = (feat["is_young"] & is_pre_meno).astype(int)

            # Durée totale de suivi normalisée par l'âge
            if "OS_MONTHS" in df.columns:
                os_months = pd.to_numeric(df["OS_MONTHS"], errors="coerce").fillna(0)
                # Ratio suivi/âge : patients jeunes ont plus d'années de suivi potentiel
                feat["followup_age_ratio"] = (os_months / 12) / age.replace(0, np.nan)
                feat["followup_age_ratio"] = feat["followup_age_ratio"].fillna(0).clip(0, 5)

        return feat

    def _compute_temporal_risk_score(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule un score composite de risque temporel basé sur
        l'ORDRE et la RAPIDITÉ des événements.
        """
        score = pd.DataFrame(index=features.index)
        risk = pd.Series(0.0, index=features.index)

        # Récidive très précoce = risque maximum (+3)
        if "very_early_recurrence" in features.columns:
            risk += features["very_early_recurrence"] * 3
        if "early_recurrence" in features.columns:
            risk += features["early_recurrence"] * 2

        # Décès précoce (+3)
        if "early_death" in features.columns:
            risk += features["early_death"] * 3

        # Nouvelle tumeur après traitement (+2)
        if "new_tumor_after_treatment" in features.columns:
            risk += features["new_tumor_after_treatment"] * 2

        # Changement de phénotype ER (+2 = résistance au traitement)
        if "er_phenotype_switch" in features.columns:
            risk += features["er_phenotype_switch"] * 2

        # Traitement néo-adjuvant (stade avancé à la présentation) (+1)
        if "had_neoadjuvant_treatment" in features.columns:
            risk += features["had_neoadjuvant_treatment"] * 1

        # Chirurgie radicale d'emblée (+1)
        if "surgery_radical_mastectomy" in features.columns:
            risk += features["surgery_radical_mastectomy"] * 1

        # Marges positives (+1)
        if "positive_surgical_margins" in features.columns:
            risk += features["positive_surgical_margins"] * 1

        # Âge très jeune (+1)
        if "is_very_young" in features.columns:
            risk += features["is_very_young"] * 1

        score["temporal_risk_score"] = risk
        score["high_temporal_risk"]  = (risk >= 4).astype(int)
        score["temporal_risk_normalized"] = (risk / risk.max()).fillna(0)

        return score


def add_temporal_features(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction utilitaire rapide pour ajouter les features temporelles
    directement au DataFrame clinique.

    Usage dans pipeline.py :
        from xai_clinical.data.temporal_features import add_temporal_features
        clinical_with_temporal = add_temporal_features(clinical_df)

    Parameters
    ----------
    clinical_df : pd.DataFrame
        DataFrame clinique brut (TCGA-BRCA format cBioPortal)

    Returns
    -------
    pd.DataFrame
        DataFrame original enrichi des features temporelles (colonnes ajoutées)
    """
    encoder = TemporalEventEncoder()
    temporal_feats = encoder.fit_transform(clinical_df)

    # Concaténer sans dupliquer les colonnes existantes
    new_cols = [c for c in temporal_feats.columns if c not in clinical_df.columns]
    result = pd.concat([clinical_df, temporal_feats[new_cols]], axis=1)

    logger.info(
        f"[TemporalFeatures] {len(new_cols)} features temporelles ajoutées au DataFrame clinique"
    )
    return result
