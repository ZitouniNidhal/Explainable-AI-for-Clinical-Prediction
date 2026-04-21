import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    def __init__(
        self, explainer: Any, feature_names: List[str], random_state: int = 42
    ):
        self.explainer = explainer
        self.feature_names = feature_names
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def add_gaussian_noise(
        self, X: pd.DataFrame, noise_level: float = 0.01
    ) -> pd.DataFrame:

        X_noisy = X.copy()

        # Calculate noise for each feature
        for col in X.columns:
            std = X[col].std()
            noise = self.rng.normal(0, noise_level * std, X.shape[0])
            X_noisy[col] = X[col] + noise

        return X_noisy

    def add_adversarial_noise(
        self, X: pd.DataFrame, y: np.ndarray, model: Any, epsilon: float = 0.01
    ) -> pd.DataFrame:

        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X)
        X_adv = X.copy()

        # Simple gradient-based adversarial attack
        for idx in range(min(100, len(X))):  # Limit for efficiency
            x = X_scaled[idx : idx + 1]

            # Get model prediction probability
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0, 1]
            else:
                proba = model.predict(x)[0]

            # Calculate gradient approximation
            gradient = np.zeros_like(x)
            delta = 0.001

            for i in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[0, i] += delta
                x_minus = x.copy()
                x_minus[0, i] -= delta

                if hasattr(model, "predict_proba"):
                    proba_plus = model.predict_proba(x_plus)[0, 1]
                    proba_minus = model.predict_proba(x_minus)[0, 1]
                else:
                    proba_plus = model.predict(x_plus)[0]
                    proba_minus = model.predict(x_minus)[0]

                gradient[0, i] = (proba_plus - proba_minus) / (2 * delta)

            # Apply adversarial perturbation
            perturbation = epsilon * np.sign(gradient)
            X_scaled[idx : idx + 1] = x + perturbation

        # Scale back to original range
        scaler = StandardScaler()
        scaler.fit(X)
        X_adv.iloc[: len(X_scaled)] = pd.DataFrame(
            scaler.inverse_transform(X_scaled),
            columns=X.columns,
            index=X.index[: len(X_scaled)],
        )

        return X_adv

    def add_missing_values(
        self, X: pd.DataFrame, missing_rate: float = 0.05, missing_pattern: str = "mcar"
    ) -> pd.DataFrame:

        X_missing = X.copy()

        if missing_pattern == "mcar":
            # Missing Completely At Random
            mask = self.rng.random(X.shape) < missing_rate

        elif missing_pattern == "mar":
            # Missing At Random - depends on observed values
            mask = np.zeros(X.shape, dtype=bool)

            # Make missingness depend on first few features
            n_dependent = min(3, X.shape[1])
            for i in range(n_dependent):
                dependent_prob = 0.2 + 0.3 * (X.iloc[:, i] > X.iloc[:, i].median())
                col_mask = self.rng.random(X.shape[0]) < dependent_prob
                mask[:, i] = col_mask

            # Add some random missingness for other features
            other_mask = self.rng.random(X.shape) < (missing_rate / 2)
            mask = mask | other_mask

        elif missing_pattern == "mnar":
            # Missing Not At Random - depends on missing values themselves
            mask = np.zeros(X.shape, dtype=bool)

            # Higher missing rate for extreme values
            for col in range(X.shape[1]):
                col_data = X.iloc[:, col]
                q25, q75 = col_data.quantile([0.25, 0.75])

                # Higher missing rate for extreme values
                extreme_mask = (col_data < q25) | (col_data > q75)
                extreme_prob = 0.3 if extreme_mask.any() else missing_rate
                col_mask = self.rng.random(X.shape[0]) < extreme_prob
                mask[:, col] = col_mask

        else:
            raise ValueError(f"Unknown missing pattern: {missing_pattern}")

        X_missing = X_missing.mask(mask)
        return X_missing

    def add_outliers(self, X: pd.DataFrame, outlier_rate: float = 0.02) -> pd.DataFrame:
        """
        Add outliers to data

        Args:
            X: Original data
            outlier_rate: Proportion of values to make outliers

        Returns:
            Data with outliers
        """
        X_outliers = X.copy()

        for col in X.columns:
            col_data = X[col]
            q1, q3 = col_data.quantile([0.25, 0.75])
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Select random points to make outliers
            n_outliers = int(outlier_rate * len(col_data))
            outlier_indices = self.rng.choice(X.index, n_outliers, replace=False)

            # Create outliers (extreme values)
            for idx in outlier_indices:
                if self.rng.random() < 0.5:
                    # Upper outlier
                    multiplier = self.rng.uniform(2, 5)
                    X_outliers.loc[idx, col] = upper_bound + multiplier * iqr
                else:
                    # Lower outlier
                    multiplier = self.rng.uniform(2, 5)
                    X_outliers.loc[idx, col] = lower_bound - multiplier * iqr

        return X_outliers

    def add_feature_noise(
        self, X: pd.DataFrame, feature_noise_rate: float = 0.1
    ) -> pd.DataFrame:
        """
        Add noise to specific features

        Args:
            X: Original data
            feature_noise_rate: Proportion of features to add noise to

        Returns:
            Data with feature noise
        """
        X_noisy = X.copy()

        n_noisy_features = int(feature_noise_rate * X.shape[1])
        noisy_features = self.rng.choice(X.columns, n_noisy_features, replace=False)

        for feature in noisy_features:
            # Add different types of noise
            noise_type = self.rng.choice(["gaussian", "uniform", "impulsive"])

            if noise_type == "gaussian":
                noise = self.rng.normal(0, 0.1 * X[feature].std(), X.shape[0])
            elif noise_type == "uniform":
                noise = self.rng.uniform(
                    -0.2 * X[feature].std(), 0.2 * X[feature].std(), X.shape[0]
                )
            else:  # impulsive
                noise = np.zeros(X.shape[0])
                n_impulses = int(0.05 * X.shape[0])
                impulse_indices = self.rng.choice(X.index, n_impulses, replace=False)
                noise[impulse_indices] = (
                    self.rng.uniform(-1, 1, n_impulses) * X[feature].std()
                )

            X_noisy[feature] = X[feature] + noise

        return X_noisy

    def compute_jaccard_index(
        self, features1: List[str], features2: List[str], k: int = 10
    ) -> float:
        """
        Calculate Jaccard index between two top-k feature sets

        Args:
            features1: First feature set
            features2: Second feature set
            k: Number of top features to consider

        Returns:
            Jaccard index (0-1)
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
        Calculate Spearman rank correlation between importance vectors

        Args:
            importance1: First importance vector
            importance2: Second importance vector

        Returns:
            Spearman correlation coefficient
        """
        corr, _ = spearmanr(importance1, importance2)
        return corr if not np.isnan(corr) else 0.0

    def compute_feature_stability(
        self, explanations_list: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute feature-level stability metrics across multiple explanations

        Args:
            explanations_list: List of explanation dictionaries

        Returns:
            Feature stability metrics
        """
        if len(explanations_list) < 2:
            return {}

        # Extract feature importances
        feature_importances = {}
        all_features = set()

        for i, exp in enumerate(explanations_list):
            if "shap_importance" in exp:
                importance_df = exp["shap_importance"]
                for _, row in importance_df.iterrows():
                    feature = row["feature"]
                    importance = row["shap_importance"]

                    if feature not in feature_importances:
                        feature_importances[feature] = []
                    feature_importances[feature].append(importance)
                    all_features.add(feature)

        # Compute stability metrics per feature
        feature_stability = {}

        for feature in all_features:
            importances = feature_importances[feature]
            if len(importances) > 1:
                feature_stability[feature] = {
                    "mean_importance": np.mean(importances),
                    "std_importance": np.std(importances),
                    "cv_importance": (
                        np.std(importances) / np.mean(importances)
                        if np.mean(importances) != 0
                        else np.inf
                    ),
                    "stability_score": 1
                    / (1 + np.std(importances)),  # Higher is more stable
                }

        return feature_stability

    def evaluate_comprehensive_robustness(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        perturbation_types: List[str] = None,
        n_perturbations: int = 10,
    ) -> Dict[str, Any]:
        """
        Comprehensive robustness evaluation

        Args:
            X: Original data
            y: Target values (optional)
            model: Trained model (optional, for adversarial attacks)
            perturbation_types: Types of perturbations to test
            n_perturbations: Number of perturbations per type

        Returns:
            Comprehensive robustness report
        """
        if perturbation_types is None:
            perturbation_types = ["gaussian", "missing", "outliers", "feature_noise"]

        logger.info(f"Evaluating robustness with perturbations: {perturbation_types}")

        comprehensive_results = {"baseline": {}, "perturbations": {}, "summary": {}}

        # Baseline explanations
        baseline_exp = self.explainer.explain_global(X)
        comprehensive_results["baseline"] = {
            "explanation": baseline_exp,
            "feature_ranking": baseline_exp["feature"].tolist(),
        }

        # Test each perturbation type
        for perturbation_type in perturbation_types:
            logger.info(f"Testing {perturbation_type} perturbation")

            perturbation_results = []

            for i in range(n_perturbations):
                # Apply perturbation
                if perturbation_type == "gaussian":
                    X_perturbed = self.add_gaussian_noise(X, noise_level=0.05)
                elif perturbation_type == "missing":
                    X_perturbed = self.add_missing_values(
                        X, missing_rate=0.1, missing_pattern="mcar"
                    )
                elif perturbation_type == "outliers":
                    X_perturbed = self.add_outliers(X, outlier_rate=0.02)
                elif perturbation_type == "feature_noise":
                    X_perturbed = self.add_feature_noise(X, feature_noise_rate=0.1)
                elif perturbation_type == "adversarial" and model is not None:
                    X_perturbed = self.add_adversarial_noise(X, y, model, epsilon=0.01)
                else:
                    continue

                # Get explanations for perturbed data
                perturbed_exp = self.explainer.explain_global(X_perturbed)

                # Calculate stability metrics
                jaccard = self.compute_jaccard_index(
                    comprehensive_results["baseline"]["feature_ranking"],
                    perturbed_exp["feature"].tolist(),
                    k=10,
                )

                # Rank correlation
                baseline_importance = baseline_exp["shap_importance"].values
                perturbed_importance = perturbed_exp["shap_importance"].values

                # Align importance vectors
                aligned_baseline = []
                aligned_perturbed = []

                for feat in baseline_exp["feature"].values:
                    if feat in perturbed_exp["feature"].values:
                        aligned_baseline.append(
                            baseline_exp[baseline_exp["feature"] == feat][
                                "shap_importance"
                            ].values[0]
                        )
                        aligned_perturbed.append(
                            perturbed_exp[perturbed_exp["feature"] == feat][
                                "shap_importance"
                            ].values[0]
                        )

                spearman = (
                    self.compute_rank_correlation(
                        np.array(aligned_baseline), np.array(aligned_perturbed)
                    )
                    if len(aligned_baseline) > 1
                    else 0.0
                )

                perturbation_results.append(
                    {
                        "jaccard_index": jaccard,
                        "spearman_correlation": spearman,
                        "perturbation_seed": i,
                    }
                )

            # Aggregate results
            if perturbation_results:
                jaccard_values = [r["jaccard_index"] for r in perturbation_results]
                spearman_values = [
                    r["spearman_correlation"] for r in perturbation_results
                ]

                comprehensive_results["perturbations"][perturbation_type] = {
                    "individual_results": perturbation_results,
                    "jaccard_mean": np.mean(jaccard_values),
                    "jaccard_std": np.std(jaccard_values),
                    "spearman_mean": np.mean(spearman_values),
                    "spearman_std": np.std(spearman_values),
                    "stability_score": np.mean(jaccard_values)
                    * np.mean(spearman_values),
                }

        # Overall summary
        if comprehensive_results["perturbations"]:
            all_stability_scores = [
                results["stability_score"]
                for results in comprehensive_results["perturbations"].values()
            ]

            comprehensive_results["summary"] = {
                "overall_stability": np.mean(all_stability_scores),
                "most_stable_perturbation": max(
                    comprehensive_results["perturbations"].items(),
                    key=lambda x: x[1]["stability_score"],
                )[0],
                "least_stable_perturbation": min(
                    comprehensive_results["perturbations"].items(),
                    key=lambda x: x[1]["stability_score"],
                )[0],
                "recommendation": self._generate_recommendation(comprehensive_results),
            }

        return comprehensive_results

    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on robustness analysis"""
        summary = results["summary"]

        if summary["overall_stability"] > 0.8:
            return "Excellent robustness - explanations are highly stable across perturbations"
        elif summary["overall_stability"] > 0.6:
            return "Good robustness - explanations are generally stable with minor variations"
        elif summary["overall_stability"] > 0.4:
            return "Moderate robustness - explanations show some sensitivity to perturbations"
        else:
            return "Poor robustness - explanations are highly sensitive to data perturbations"

    def generate_robustness_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive text report"""
        report = ["=" * 80]
        report.append("COMPREHENSIVE ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 80)

        # Summary
        if "summary" in results:
            summary = results["summary"]
            report.append(f"\nOVERALL ASSESSMENT:")
            report.append(
                f"- Overall Stability Score: {summary['overall_stability']:.3f}"
            )
            report.append(
                f"- Most Stable Perturbation: {summary['most_stable_perturbation']}"
            )
            report.append(
                f"- Least Stable Perturbation: {summary['least_stable_perturbation']}"
            )
            report.append(f"- Recommendation: {summary['recommendation']}")

        # Detailed results
        report.append(f"\nDETAILED PERTURBATION RESULTS:")
        report.append("-" * 50)

        for pert_type, pert_results in results["perturbations"].items():
            report.append(f"\n{pert_type.upper()} PERTURBATION:")
            report.append(
                f"  Jaccard Index: {pert_results['jaccard_mean']:.3f} ± {pert_results['jaccard_std']:.3f}"
            )
            report.append(
                f"  Spearman Correlation: {pert_results['spearman_mean']:.3f} ± {pert_results['spearman_std']:.3f}"
            )
            report.append(f"  Stability Score: {pert_results['stability_score']:.3f}")

        # Feature stability
        if "baseline" in results:
            report.append(f"\nBASELINE EXPLANATION:")
            baseline = results["baseline"]
            report.append(f"  Top 10 features: {baseline['feature_ranking'][:10]}")

        report.append("\n" + "=" * 80)
        report.append("INTERPRETATION GUIDE:")
        report.append("- Stability Score > 0.8: Excellent robustness")
        report.append("- Stability Score 0.6-0.8: Good robustness")
        report.append("- Stability Score 0.4-0.6: Moderate robustness")
        report.append("- Stability Score < 0.4: Poor robustness")
        report.append("=" * 80)

        return "\n".join(report)
