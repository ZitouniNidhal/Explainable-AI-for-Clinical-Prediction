"""
Clinical Validation Module — MOCIP Pipeline
============================================
Protocole ACVF (Analyse de Cohérence et de Validité Fonctionnelle) :
  1. Cohérence biologique : les prédictions haute-risque doivent s'aligner avec les
     sous-types BRCA cliniquement agressifs (Triple Négatif, HER2+, Stade III/IV).
  2. Discrimination clinique : AUC-ROC, calibration (Brier Score), courbe de calibration.
  3. Validation de clustering biologique : ARI entre clusters omiques et sous-types BRCA connus.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    adjusted_rand_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cohérence Biologique des Prédictions
# ─────────────────────────────────────────────────────────────────────────────

class BiologicalCoherenceValidator:
    """
    Vérifie que les prédictions haute-risque du modèle s'alignent avec
    les sous-types BRCA biologiquement agressifs (réponse à la critique de
    spécificité BRCA et de validation biologique des clusters).
    """

    # Sous-types considérés comme haut-risque biologiquement
    HIGH_RISK_SUBTYPES = {"Triple Negative", "HER2+", "Basal-like"}
    HIGH_RISK_STAGES   = {"Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC", "Stage IV"}

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def validate(
        self,
        y_proba: np.ndarray,
        clinical_df: pd.DataFrame,
        subtype_col: Optional[str] = None,
        stage_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calcule le taux d'alignement entre haute-probabilité prédite et
        sous-types biologiquement agressifs.

        Returns
        -------
        dict avec :
          - ``alignment_subtype``  : proportion des high-risk prédits qui sont bio-agressifs
          - ``alignment_stage``    : proportion des high-risk prédits en stade avancé
          - ``coverage_subtype``   : proportion des bio-agressifs capturés par le modèle
        """
        results: Dict[str, float] = {}
        predicted_high_risk = y_proba >= self.threshold

        # — Alignement par sous-type moléculaire ————————————————————————
        col = subtype_col or next(
            (c for c in ["SUBTYPE", "PAM50", "ONCOTREE_CODE", "CANCER_TYPE_DETAILED"]
             if c in clinical_df.columns),
            None,
        )
        if col:
            bio_aggressive = clinical_df[col].isin(self.HIGH_RISK_SUBTYPES)

            n_pred_hr = predicted_high_risk.sum()
            if n_pred_hr > 0:
                # Parmi ceux prédits high-risk, combien sont vraiment bio-agressifs ?
                alignment = (predicted_high_risk & bio_aggressive.values).sum() / n_pred_hr
                results["alignment_subtype"] = float(alignment)

            # Parmi les bio-agressifs, combien le modèle capture-t-il ?
            n_bio_hr = bio_aggressive.sum()
            if n_bio_hr > 0:
                coverage = (predicted_high_risk & bio_aggressive.values).sum() / n_bio_hr
                results["coverage_subtype"] = float(coverage)

            logger.info(f"[ACVF] Alignement sous-type : {results.get('alignment_subtype', 'N/A'):.3f}")
            logger.info(f"[ACVF] Couverture sous-type : {results.get('coverage_subtype', 'N/A'):.3f}")
        else:
            logger.warning("[ACVF] Colonne de sous-type non trouvée — alignement subtype ignoré.")

        # — Alignement par stade tumoral ————————————————————————————————
        s_col = stage_col or next(
            (c for c in ["AJCC_PATHOLOGIC_TUMOR_STAGE", "STAGE", "TUMOR_STAGE"]
             if c in clinical_df.columns),
            None,
        )
        if s_col:
            advanced_stage = clinical_df[s_col].isin(self.HIGH_RISK_STAGES)
            n_pred_hr = predicted_high_risk.sum()
            if n_pred_hr > 0:
                align_stage = (predicted_high_risk & advanced_stage.values).sum() / n_pred_hr
                results["alignment_stage"] = float(align_stage)
            logger.info(f"[ACVF] Alignement stade avancé : {results.get('alignment_stage', 'N/A'):.3f}")
        else:
            logger.warning("[ACVF] Colonne de stade non trouvée — alignement stade ignoré.")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation de Clustering Biologique (ARI vs Sous-types BRCA)
# ─────────────────────────────────────────────────────────────────────────────

class ClusterBiologicalValidator:
    """
    Calcule l'Adjusted Rand Index (ARI) entre les clusters omiques découverts
    par le pipeline et les sous-types BRCA cliniques connus.

    Répond à la critique : « le clustering n'est pas biologiquement validé ».
    """

    BRCA_SUBTYPE_MAPPING = {
        "Luminal A"     : 0,
        "Luminal B"     : 1,
        "HER2-enriched" : 2,
        "HER2+"         : 2,
        "Basal-like"    : 3,
        "Triple Negative": 3,
        "Normal-like"   : 4,
        "Unknown"       : -1,
    }

    def compute_ari(
        self,
        cluster_labels: np.ndarray,
        clinical_df: pd.DataFrame,
        subtype_col: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calcule l'ARI entre les clusters du pipeline et les sous-types BRCA.

        Parameters
        ----------
        cluster_labels : ndarray of shape (n_samples,)
            Étiquettes de cluster produites par le pipeline.
        clinical_df : DataFrame aligné sur les mêmes patients.
        subtype_col : colonne contenant le sous-type PAM50/BRCA.

        Returns
        -------
        dict avec ``ari``, ``n_clusters``, ``n_subtypes``, ``interpretation``
        """
        col = subtype_col or next(
            (c for c in ["SUBTYPE", "PAM50", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE"]
             if c in clinical_df.columns),
            None,
        )

        if col is None:
            logger.warning("[CLUSTER] Colonne de sous-type BRCA absente — ARI impossible.")
            return {"ari": None, "interpretation": "Données de sous-type absentes"}

        raw_subtypes = clinical_df[col].fillna("Unknown")
        # Encoder numériquement
        true_labels = raw_subtypes.map(
            lambda x: self.BRCA_SUBTYPE_MAPPING.get(x, -1)
        ).values

        # Exclure les inconnus
        valid = true_labels != -1
        if valid.sum() < 10:
            logger.warning("[CLUSTER] Trop peu de sous-types connus pour calculer l'ARI.")
            return {"ari": None, "interpretation": "Pas assez de patients annotés"}

        ari = adjusted_rand_score(true_labels[valid], cluster_labels[valid])

        interpretation = (
            "Excellent (≥0.6)" if ari >= 0.6 else
            "Bon (0.4–0.6)"    if ari >= 0.4 else
            "Modéré (0.2–0.4)" if ari >= 0.2 else
            "Faible (<0.2) — les clusters ne correspondent pas aux sous-types biologiques"
        )

        result = {
            "ari"           : float(ari),
            "n_clusters"    : int(len(np.unique(cluster_labels))),
            "n_subtypes"    : int(valid.sum()),
            "interpretation": interpretation,
        }

        logger.info(f"[CLUSTER] ARI = {ari:.4f} — {interpretation}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calibration du Modèle (Brier Score + courbe)
# ─────────────────────────────────────────────────────────────────────────────

class ModelCalibrationValidator:
    """
    Évalue la calibration des probabilités prédites.
    Un modèle bien calibré est indispensable pour un usage clinique.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, object]:
        """
        Returns
        -------
        dict avec :
          - ``brier_score``  : erreur de probabilité (0=parfait, <0.25=acceptable)
          - ``fraction_pos`` : fraction observée par bin (courbe de calibration)
          - ``mean_pred``    : probabilité moyenne prédite par bin
          - ``auc_roc``      : AUC-ROC pour référence
          - ``interpretation``: appréciation textuelle
        """
        brier = brier_score_loss(y_true, y_proba)
        auc   = roc_auc_score(y_true, y_proba)

        fraction_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )

        interp = (
            "Excellent (≤0.10)" if brier <= 0.10 else
            "Bon (0.10–0.20)"   if brier <= 0.20 else
            "Acceptable (0.20–0.25)" if brier <= 0.25 else
            "Mauvais (>0.25) — recalibration recommandée"
        )

        result = {
            "brier_score"  : float(brier),
            "auc_roc"      : float(auc),
            "fraction_pos" : fraction_pos.tolist(),
            "mean_pred"    : mean_pred.tolist(),
            "interpretation": interp,
        }

        logger.info(f"[CALIBRATION] Brier={brier:.4f} — {interp}")
        logger.info(f"[CALIBRATION] AUC-ROC={auc:.4f}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rapport ACVF Complet
# ─────────────────────────────────────────────────────────────────────────────

def run_acvf_validation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    clinical_df: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Point d'entrée unique pour le protocole ACVF complet.

    Parameters
    ----------
    y_true        : étiquettes binaires réelles
    y_proba       : probabilités prédites (classe positive)
    clinical_df   : données cliniques alignées (mêmes patients)
    cluster_labels: étiquettes de cluster omique (optionnel)

    Returns
    -------
    Dictionnaire structuré avec tous les résultats de validation
    """
    report: Dict[str, object] = {}

    logger.info("=" * 60)
    logger.info("PROTOCOLE ACVF — ANALYSE DE COHÉRENCE ET VALIDITÉ FONCTIONNELLE")
    logger.info("=" * 60)

    # 1. Cohérence biologique
    bio_val = BiologicalCoherenceValidator(threshold=0.5)
    report["biological_coherence"] = bio_val.validate(y_proba, clinical_df)

    # 2. Calibration
    cal_val = ModelCalibrationValidator()
    report["calibration"] = cal_val.evaluate(y_true, y_proba)

    # 3. Validation de clustering (si labels fournis)
    if cluster_labels is not None:
        cl_val = ClusterBiologicalValidator()
        report["cluster_biological_ari"] = cl_val.compute_ari(cluster_labels, clinical_df)
    else:
        report["cluster_biological_ari"] = {
            "ari": None,
            "interpretation": "Aucun label de cluster fourni"
        }

    # Résumé console
    logger.info("\n[ACVF RÉSUMÉ]")
    bio = report["biological_coherence"]
    cal = report["calibration"]
    logger.info(f"  Alignement sous-type : {bio.get('alignment_subtype', 'N/A')}")
    logger.info(f"  Couverture bio       : {bio.get('coverage_subtype', 'N/A')}")
    logger.info(f"  Brier Score          : {cal['brier_score']:.4f} — {cal['interpretation']}")
    logger.info(f"  AUC-ROC              : {cal['auc_roc']:.4f}")
    ari_res = report["cluster_biological_ari"]
    if ari_res["ari"] is not None:
        logger.info(f"  ARI clustering       : {ari_res['ari']:.4f} — {ari_res['interpretation']}")

    return report
