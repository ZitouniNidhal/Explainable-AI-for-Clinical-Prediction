# src/xai_clinical/explainability/stability_analyzer.py
"""
Explanation robustness and stability analysis
"""

import numpy as np
import pandas as pd
from typing import Any, List, Dict, Callable
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import jaccard_score


class StabilityAnalyzer:
    """
    Evaluates explanation stability against noise and perturbations
    """

    def __init__(self, explainer: Any, feature_names: List[str]):
        self.explainer = explainer
        self.feature_names = feature_names

    def add_gaussian_noise(self, X, noise_level=0.1):
        """Add Gaussian noise to features"""
        X_noisy = X.copy()
        
        # Convertir en numpy pour éviter les problèmes pandas
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            std_values = X.std().values  # Shape (n_features,)
            
            # Générer le bruit avec la bonne shape
            noise = np.random.normal(0, noise_level, X_values.shape)
            
            # Multiplier colonne par colonne
            for i in range(X_values.shape[1]):
                noise[:, i] *= std_values[i]
            
            X_noisy = pd.DataFrame(X_values + noise, columns=X.columns, index=X.index)
        else:
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise * X.std(axis=0)
        
        return X_noisy

    def add_missing_values(
        self, X: pd.DataFrame, missing_rate: float = 0.05
    ) -> pd.DataFrame:
        """Introduce random missing values"""
        X_missing = X.copy()
        mask = np.random.random(X.shape) < missing_rate
        X_missing[mask] = np.nan
        return X_missing

    def compute_jaccard_index(
        self, features1: List[str], features2: List[str], k: int = 10
    ) -> float:
        """
        Calculate Jaccard index between two top-k feature sets
        """
        set1 = set(features1[:k])
        set2 = set(features2[:k])

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def compute_rank_correlation(
        self, importance1: np.ndarray, importance2: np.ndarray
    ) -> float:
        """
        Calculate Spearman correlation between two importance vectors
        """
        corr, _ = spearmanr(importance1, importance2)
        return corr

    def evaluate_stability(
        self,
        X: pd.DataFrame,
        n_perturbations: int = 10,
        noise_levels: List[float] = None,
        missing_rates: List[float] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate explanation stability under different perturbations

        Returns:
            Dictionary with stability metrics
        """
        noise_levels = noise_levels or [0.01, 0.05, 0.1]
        missing_rates = missing_rates or [0.05, 0.1, 0.2]

        results = {
            "gaussian_noise": {},
            "missing_values": {},
            "baseline_explanation": None,
        }

        # Baseline explanation
        baseline_shap = self.explainer.explain_global(X)
        baseline_features = baseline_shap["feature"].tolist()
        baseline_importance = baseline_shap["shap_importance"].values

        results["baseline_explanation"] = baseline_shap

        # Test with Gaussian noise
        for noise in noise_levels:
            jaccard_scores = []
            rank_correlations = []

            for _ in range(n_perturbations):
                X_noisy = self.add_gaussian_noise(X, noise)

                try:
                    perturbed_shap = self.explainer.explain_global(X_noisy)
                    perturbed_features = perturbed_shap["feature"].tolist()
                    perturbed_importance = perturbed_shap["shap_importance"].values

                    # Align features for comparison
                    aligned_baseline = []
                    aligned_perturbed = []

                    for feat in baseline_features:
                        if feat in perturbed_shap["feature"].values:
                            aligned_baseline.append(
                                baseline_shap[baseline_shap["feature"] == feat][
                                    "shap_importance"
                                ].values[0]
                            )
                            aligned_perturbed.append(
                                perturbed_shap[perturbed_shap["feature"] == feat][
                                    "shap_importance"
                                ].values[0]
                            )

                    if len(aligned_baseline) > 1:
                        jaccard = self.compute_jaccard_index(
                            baseline_features, perturbed_features, top_k
                        )
                        corr = self.compute_rank_correlation(
                            np.array(aligned_baseline), np.array(aligned_perturbed)
                        )

                        jaccard_scores.append(jaccard)
                        rank_correlations.append(corr)

                except Exception as e:
                    continue

            results["gaussian_noise"][f"noise_{noise}"] = {
                "jaccard_mean": np.mean(jaccard_scores) if jaccard_scores else 0,
                "jaccard_std": np.std(jaccard_scores) if jaccard_scores else 0,
                "spearman_mean": np.mean(rank_correlations) if rank_correlations else 0,
                "spearman_std": np.std(rank_correlations) if rank_correlations else 0,
            }

        # Test with missing values
        for missing_rate in missing_rates:
            jaccard_scores = []
            rank_correlations = []

            for _ in range(n_perturbations):
                X_missing = self.add_missing_values(X, missing_rate)

                try:
                    # Simple imputation for testing
                    X_missing = X_missing.fillna(X.median())

                    perturbed_shap = self.explainer.explain_global(X_missing)
                    perturbed_features = perturbed_shap["feature"].tolist()

                    jaccard = self.compute_jaccard_index(
                        baseline_features, perturbed_features, top_k
                    )
                    jaccard_scores.append(jaccard)

                except Exception as e:
                    continue

            results["missing_values"][f"missing_{missing_rate}"] = {
                "jaccard_mean": np.mean(jaccard_scores) if jaccard_scores else 0,
                "jaccard_std": np.std(jaccard_scores) if jaccard_scores else 0,
            }

        return results

    def generate_stability_report(self, results: Dict[str, Any]) -> str:
        """
        Generate text report of stability
        """
        report = ["=" * 60]
        report.append("EXPLANATION STABILITY REPORT")
        report.append("=" * 60)

        report.append("\n1. GAUSSIAN NOISE STABILITY")
        report.append("-" * 40)
        for noise_level, metrics in results["gaussian_noise"].items():
            report.append(f"\n{noise_level}:")
            report.append(
                f"  Jaccard Index: {metrics['jaccard_mean']:.3f} ± {metrics['jaccard_std']:.3f}"
            )
            report.append(
                f"  Spearman Correlation: {metrics['spearman_mean']:.3f} ± {metrics['spearman_std']:.3f}"
            )

        report.append("\n2. MISSING VALUES STABILITY")
        report.append("-" * 40)
        for missing_level, metrics in results["missing_values"].items():
            report.append(f"\n{missing_level}:")
            report.append(
                f"  Jaccard Index: {metrics['jaccard_mean']:.3f} ± {metrics['jaccard_std']:.3f}"
            )

        report.append("\n" + "=" * 60)
        report.append("INTERPRETATION:")
        report.append("- Jaccard > 0.7: Excellent stability")
        report.append("- Jaccard 0.5-0.7: Good stability")
        report.append("- Jaccard < 0.5: Poor stability")
        report.append("=" * 60)

        return "\n".join(report)
