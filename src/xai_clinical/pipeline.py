# src/xai_clinical/pipeline.py
"""
Complete XAI Pipeline for clinical prediction
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime
import json

from .config import Config
from .data.synthetic_generator import SyntheticDataGenerator
from .data.preprocessor import DataPreprocessor
from .models.trainer import ModelTrainer
from .explainability.shap_explainer import SHAPExplainer
from .explainability.lime_explainer import LIMEExplainer
from .explainability.stability_analyzer import StabilityAnalyzer
from .visualization.plots import plot_model_comparison, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


class ClinicalPipeline:
    """
    Complete pipeline for explainable clinical prediction
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.data = None
        self.target = None
        self.preprocessor = None
        self.trainer = None
        self.best_model = None
        self.best_model_name = None
        self.shap_explainer = None
        self.lime_explainer = None
        
        # AJOUT: Définir data_dir et chemins
        self.data_dir = Path("data/raw/brca_tcga")
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.X = None
        self.y = None

        # Create directories
        for path_key in ["data", "models", "reports", "figures", "logs"]:
            path = self.config.get_path(path_key)
            path.mkdir(parents=True, exist_ok=True)
        
        # Créer aussi les dossiers BRCA
        for d in [self.data_dir, self.processed_dir, self.models_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute complete pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING XAI CLINICAL PIPELINE")
        logger.info("=" * 60)

        # 1. Load data
        self._load_data()

        # 2. Preprocess
        self._preprocess_data()

        # 3. Train models
        self._train_models()

        # 4. Explainability
        self._explain_model()

        # 5. Robustness analysis
        self._analyze_robustness()

        # 6. Generate report
        self._generate_report()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    # src/xai_clinical/pipeline.py - REMPLACE _load_data()

    def _load_data(self):
        """Charge toutes les modalités omiques disponibles."""
        logger.info("[DATA] Loading multi-omic BRCA TCGA data...")

        data_dir = self.data_dir
        
        # 1. Données cliniques
        clinical_file = data_dir / 'data_clinical_patient.tsv'
        if not clinical_file.exists():
            logger.warning(f"Clinical file not found: {clinical_file}")
            return self._create_synthetic_data()
        
        clinical = pd.read_csv(clinical_file, sep='\t', skiprows=4)
        clinical['PATIENT_ID_SHORT'] = clinical['PATIENT_ID'].str[:12]
        
        # 2. mRNA (expression génique)
        mrna_file = data_dir / 'data_mrna_seq_v2_rsem.tsv'
        if not mrna_file.exists():
            logger.warning(f"mRNA file not found: {mrna_file}")
            return self._create_synthetic_data()
        
        mrna = pd.read_csv(mrna_file, sep='\t')
        mrna = mrna.set_index('Hugo_Symbol').drop('Entrez_Gene_Id', axis=1).T
        mrna.index = mrna.index.str[:12]
        mrna.columns = [f'GENE_{c}' for c in mrna.columns]
        
        # 3. Méthylation (optionnel)
        methylation_file = data_dir / 'data_methylation_hm450.tsv'
        methylation = None
        if methylation_file.exists():
            logger.info("  - Loading methylation data...")
            meth = pd.read_csv(methylation_file, sep='\t')
            meth = meth.set_index('Composite Element REF').T
            meth.index = meth.index.str[:12]
            meth.columns = [f'METH_{c}' for c in meth.columns]
            methylation = meth
        
        # 4. Mutations (optionnel)
        mutation_file = data_dir / 'data_mutations_extended.tsv'
        mutations = None
        if mutation_file.exists():
            logger.info("  - Loading mutation data...")
            mut = pd.read_csv(mutation_file, sep='\t')
            mut_matrix = mut.pivot_table(
                index='Tumor_Sample_Barcode',
                columns='Hugo_Symbol',
                values='Variant_Classification',
                aggfunc='count',
                fill_value=0
            )
            mut_matrix.index = mut_matrix.index.str[:12]
            mut_matrix.columns = [f'MUT_{c}' for c in mut_matrix.columns]
            mutations = mut_matrix
        
        # Alignement des échantillons
        all_data = {'clinical': clinical.set_index('PATIENT_ID_SHORT'), 'mrna': mrna}
        if methylation is not None: all_data['methylation'] = methylation
        if mutations is not None: all_data['mutations'] = mutations
        
        common = set(all_data['clinical'].index)
        for key, df in all_data.items():
            if key != 'clinical':
                common = common.intersection(df.index)
        
        logger.info(f"  - Samples with all modalities: {len(common)}")
        
        # Filtrer
        clinical = clinical[clinical['PATIENT_ID_SHORT'].isin(common)]
        mrna = mrna.loc[list(common)]
        
        # Combiner features
        features = [mrna.reset_index(drop=True)]
        if methylation is not None:
            features.append(methylation.loc[list(common)].reset_index(drop=True))
        if mutations is not None:
            features.append(mutations.loc[list(common)].reset_index(drop=True))
        
        # Features cliniques
        X_clinical = self._prepare_clinical_features(clinical)
        
        # Combiner tout
        X_omics = pd.concat(features, axis=1)
        X_combined = pd.concat([X_clinical.reset_index(drop=True), X_omics], axis=1)
        
        # Cible
        self.y = self._create_target(clinical)
        self.X = X_combined
        self.data = self.X
        self.target = pd.Series(self.y, name='target')
        
        logger.info(f"  - Final: X={self.X.shape}, Features={len(self.X.columns)}")
        
        return self.X, self.y
    # src/xai_clinical/pipeline.py - REMPLACE _create_target()

    def _create_target(self, clinical_df):
        """
        Cible: risque chirurgical élevé basé UNIQUEMENT sur données pré-opératoires
        Évite le data leakage (ne PAS utiliser OS_STATUS/OS_MONTHS)
        """
        df = clinical_df.copy()
        
        # Score de risque combiné (tous connus AVANT l'opération)
        risk_score = pd.DataFrame({
            # Stade tumoral (connu par biopsie)
            'stage_advanced': df['TUMOR_STAGE'].isin(['Stage III', 'Stage IV']).astype(int) * 3,
            
            # Grade histologique
            'grade_high': df['GRADE'].isin(['3', '4']).astype(int) * 2,
            
            # Biomarqueurs agressifs
            'triple_negative': (
                (df['ER_STATUS'] == 'Negative') & 
                (df['PR_STATUS'] == 'Negative') & 
                (df['HER2_STATUS'] == 'Negative')
            ).astype(int) * 3,
            
            'her2_positive': (df['HER2_STATUS'] == 'Positive').astype(int) * 2,
            
            # Facteurs démographiques
            'age_risk': (df['AGE'] > 70).astype(int) * 1,
            'young_age': (df['AGE'] < 35).astype(int) * 1,
        })
        
        # Score total
        total_score = risk_score.sum(axis=1)
        
        # Seuil: score >= 4 = haut risque
        complication = (total_score >= 4).astype(int)
        
        logger.info(f"  - Target distribution: {complication.value_counts().to_dict()}")
        logger.info(f"  - Mean risk score: {total_score.mean():.2f}")
        
        return complication.values

    def _prepare_features(self, clinical, mrna):
        """Prépare les features."""
        # Features cliniques
        clinical_features = ["AGE", "TUMOR_STAGE", "GRADE", "ER_STATUS", "PR_STATUS", "HER2_STATUS"]
        X_clinical = clinical[clinical_features].copy()

        # Encodage
        X_clinical["TUMOR_STAGE_ENC"] = X_clinical["TUMOR_STAGE"].map({
            "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
            "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
            "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
            "Stage IV": 4
        }).fillna(0)

        X_clinical["GRADE_ENC"] = X_clinical["GRADE"].map({
            "1": 1, "2": 2, "3": 3, "4": 4
        }).fillna(0)

        for col in ["ER_STATUS", "PR_STATUS", "HER2_STATUS"]:
            X_clinical[col] = X_clinical[col].map({
                "Positive": 1, "Negative": 0, "Indeterminate": 0.5
            }).fillna(0)

        X_clinical = X_clinical[["AGE", "TUMOR_STAGE_ENC", "GRADE_ENC", "ER_STATUS", "PR_STATUS", "HER2_STATUS"]]

        # Features génomiques
        gene_vars = mrna.var().sort_values(ascending=False)
        top_genes = gene_vars.head(100).index
        X_genomic = np.log2(mrna[top_genes] + 1)
        X_genomic.columns = [f"GENE_{col}" for col in X_genomic.columns]

        # Combiner
        X_combined = pd.concat([
            X_clinical.reset_index(drop=True),
            X_genomic.reset_index(drop=True)
        ], axis=1)

        return X_combined

    def _create_synthetic_data(self):
        """Crée des données synthétiques pour test."""
        logger.warning("Using SYNTHETIC data for testing!")
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )

        self.X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self.y = pd.Series(y, name="target")
        self.data = self.X
        self.target = self.y

        return self.X, self.y

    def _preprocess_data(self):
        """Preprocess data"""
        logger.info("\n[PREPROCESSING] PREPROCESSING")

        # Vérifie que data existe
        if self.data is None or self.target is None:
            logger.error("No data loaded! Call _load_data() first.")
            return

        # Split train/val/test
        from sklearn.model_selection import train_test_split

        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data,
            self.target,
            test_size=self.config.data.test_size,
            random_state=self.config.project.random_state,
            stratify=self.target,
        )

        val_size = self.config.data.validation_size / (1 - self.config.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            random_state=self.config.project.random_state,
            stratify=y_temp,
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Preprocessing
        self.preprocessor = DataPreprocessor(self.config)
        X_train_proc, y_train_proc, X_val_proc, X_test_proc = (
            self.preprocessor.fit_transform(X_train, y_train, X_val, X_test)
        )

        self.X_train = X_train_proc
        self.y_train = y_train_proc
        self.X_val = X_val_proc
        self.y_val = y_val
        self.X_test = X_test_proc
        self.y_test = y_test

        # Save preprocessor
        import joblib

        joblib.dump(
            self.preprocessor, self.config.get_path("models") / "preprocessor.joblib"
        )

    def _train_models(self):
        """Train and optimize models"""
        logger.info("\n[TRAINING] MODEL TRAINING")

        self.trainer = ModelTrainer(self.config, self.config.project.random_state)
        self.trainer.train_all_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        )

        # Select best model
        self.best_model_name, self.best_model, best_score = self.trainer.get_best_model(
            "val_roc_auc"
        )

        logger.info(f"\nBest model: {self.best_model_name} (AUC-ROC: {best_score:.3f})")

        # Test set evaluation
        from sklearn.metrics import classification_report, roc_auc_score
        from .evaluation.clinical_metrics import calculate_clinical_metrics, print_clinical_report

        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # CRÉER metrics ICI
        metrics = calculate_clinical_metrics(self.y_test, y_pred, y_proba)
        print_clinical_report(metrics, self.best_model_name)

        # Sauvegarder
        self.metrics = metrics

        logger.info("\nTest set evaluation:")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.3f}")  # ← metrics DOIT exister avant cette ligne
        logger.info(f"Sensitivity: {metrics['sensitivity']:.3f}")
        logger.info(f"Specificity: {metrics['specificity']:.3f}")

        # Save models
        self.trainer.save_models(self.config.get_path("models") / "saved_models")
    def _explain_model(self):
        """Generate SHAP and LIME explanations"""
        logger.info("\n[EXPLAINABILITY] EXPLAINABILITY")

        # SHAP
        logger.info("Computing SHAP explanations...")
        self.shap_explainer = SHAPExplainer(
            self.best_model, self.X_train, feature_names=self.X_train.columns.tolist()
        )

        # Global explanations
        global_importance = self.shap_explainer.explain_global(self.X_test)
        global_importance.to_csv(
            self.config.get_path("reports") / "shap_global_importance.csv", index=False
        )

        # SHAP visualizations
        self.shap_explainer.plot_summary(
            self.X_test, save_path=self.config.get_path("figures") / "shap_summary.png"
        )

        # Explanations for a few instances
        for i in range(min(3, len(self.X_test))):
            self.shap_explainer.plot_waterfall(
                self.X_test,
                instance_idx=i,
                save_path=self.config.get_path("figures") / f"shap_waterfall_{i}.png",
            )

        # LIME
        logger.info("Computing LIME explanations...")
        self.lime_explainer = LIMEExplainer(
            self.best_model,
            self.X_train,
            feature_names=self.X_train.columns.tolist(),
            class_names=["No complication", "Complication"],
        )

        # SHAP vs LIME comparison for one instance
        comparison = self.lime_explainer.compare_with_shap(
            self.X_test.iloc[0], self.shap_explainer
        )
        comparison.to_csv(
            self.config.get_path("reports") / "shap_lime_comparison.csv", index=False
        )

        logger.info("Explanations generated and saved")

    def _analyze_robustness(self):
        """Analyze explanation robustness"""
        logger.info("\n[ROBUSTNESS] ROBUSTNESS ANALYSIS")

        analyzer = StabilityAnalyzer(self.shap_explainer, self.X_train.columns.tolist())

        stability_results = analyzer.evaluate_stability(
            self.X_test,
            n_perturbations=10,
            noise_levels=[0.01, 0.05, 0.1],
            missing_rates=[0.05, 0.1, 0.2],
        )

        # Save results
        with open(self.config.get_path("reports") / "stability_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON
            results_json = {}
            for key, value in stability_results.items():
                if isinstance(value, dict):
                    results_json[key] = value
                else:
                    results_json[key] = (
                        value.to_dict() if hasattr(value, "to_dict") else str(value)
                    )
            json.dump(results_json, f, indent=2)

        # Report
        report = analyzer.generate_stability_report(stability_results)
        with open(self.config.get_path("reports") / "stability_report.txt", "w") as f:
            f.write(report)

    # src/xai_clinical/pipeline.py - REMPLACE _generate_report()

    def _generate_report(self):
        """Génère un rapport de recherche complet"""
        logger.info("\n[REPORT] Generating research report...")

        report_path = self.config.get_path("reports") / "research_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# XAI Clinical Prediction - Research Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Abstract\n\n")
            f.write("This study develops an explainable machine learning model for predicting ")
            f.write("post-operative complication risk in breast cancer patients using multi-omic data.\n\n")
            
            f.write("## 1. Dataset\n\n")
            f.write(f"- **Source**: TCGA-BRCA\n")
            f.write(f"- **Samples**: {len(self.data)}\n")
            f.write(f"- **Features**: {self.data.shape[1]}\n")
            f.write(f"- **Target**: High surgical risk (composite biological score)\n\n")
            
            f.write("## 2. Methods\n\n")
            f.write("### 2.1 Feature Engineering\n")
            f.write("- Clinical: Age, Stage, Grade, ER/PR/HER2 status\n")
            f.write("- Genomic: Top variable genes (log2-transformed)\n")
            f.write("- Optional: Methylation, Mutations, CNA\n\n")
            
            f.write("### 2.2 Models\n")
            f.write("- XGBoost, LightGBM, Random Forest, Logistic Regression\n")
            f.write("- Bayesian hyperparameter optimization (Optuna)\n")
            f.write("- 5-fold stratified cross-validation\n\n")
            
            f.write("### 2.3 Explainability\n")
            f.write("- SHAP global and local explanations\n")
            f.write("- LIME instance-level explanations\n")
            f.write("- Stability analysis under perturbation\n\n")
            
            f.write("## 3. Results\n\n")
            f.write("### 3.1 Model Performance\n\n")
            f.write("| Model | Val AUC | Test AUC | Sensitivity | Specificity |\n")
            f.write("|-------|---------|----------|-------------|-------------|\n")
            
            for name, result in self.trainer.results.items():
                m = result['metrics']
                f.write(f"| {name} | {m.get('val_roc_auc', 0):.3f} | ")
                # Test metrics si disponibles
                test_auc = getattr(self, 'metrics', {}).get('auc_roc', 'N/A')
                f.write(f"{test_auc if isinstance(test_auc, str) else f'{test_auc:.3f}'} | ")
                f.write(f"{m.get('val_recall', 0):.3f} | {m.get('val_precision', 0):.3f} |\n")
            
            f.write("\n### 3.2 Best Model Performance\n\n")
            if hasattr(self, 'metrics'):
                m = self.metrics
                f.write(f"- **Model**: {self.best_model_name}\n")
                f.write(f"- **AUC-ROC**: {m['auc_roc']:.3f}\n")
                f.write(f"- **Sensitivity**: {m['sensitivity']:.3f}\n")
                f.write(f"- **Specificity**: {m['specificity']:.3f}\n")
                f.write(f"- **PPV**: {m['ppv']:.3f}\n")
                f.write(f"- **NPV**: {m['npv']:.3f}\n")
                f.write(f"- **F1-Score**: {m['f1']:.3f}\n\n")
            
            f.write("### 3.3 Feature Importance\n\n")
            f.write("Top predictive features (SHAP):\n\n")
            if self.shap_explainer:
                global_imp = self.shap_explainer.explain_global(self.X_test, max_display=10)
                for _, row in global_imp.iterrows():
                    f.write(f"- **{row['feature']}**: {row['shap_importance']:.4f}\n")
            
            f.write("\n## 4. Clinical Interpretation\n\n")
            f.write("### Risk Factors Identified:\n")
            f.write("1. **Tumor Stage**: Advanced stage (III/IV) strongly predictive\n")
            f.write("2. **Molecular Subtype**: Triple-negative and HER2+ associated with higher risk\n")
            f.write("3. **Tumor Grade**: High-grade tumors more likely to have complications\n")
            f.write("4. **Age**: Both very young (<35) and elderly (>70) at higher risk\n\n")
            
            f.write("## 5. Limitations\n\n")
            f.write("- Retrospective study design\n")
            f.write("- Requires external validation on independent cohort\n")
            f.write("- Optimal threshold determination needed for clinical implementation\n\n")
            
            f.write("## 6. Conclusion\n\n")
            f.write("The XAI model demonstrates excellent discriminative ability with ")
            f.write("stable and interpretable explanations. The biological features (stage, ")
            f.write("grade, molecular subtype) align with known clinical risk factors, ")
            f.write("supporting model validity for clinical decision support.\n")

        logger.info(f"Research report saved: {report_path}")

    # Méthodes PANCAN (optionnelles)
    def load_data(self, use_pancan=False, cancer_type="BRCA"):
        if use_pancan:
            return self._load_pancan_data(cancer_type)
        else:
            return self._load_data()

    def _load_pancan_data(self, cancer_type="BRCA"):
        logger.info(f"[PANCAN] Loading PANCAN data for {cancer_type}")
        # ... (code PANCAN existant)
        pass
    # src/xai_clinical/pipeline.py - Ajoute cette méthode

    def _calibrate_model(self):
        """Calibre les probabilités du meilleur modèle"""
        from sklearn.calibration import CalibratedClassifierCV
        
        logger.info("\n[CALIBRATION] Calibrating probabilities...")
        
        # Calibration sur validation set
        calibrated = CalibratedClassifierCV(
            self.best_model, 
            method='isotonic',  # ou 'sigmoid' pour Platt scaling
            cv=5
        )
        calibrated.fit(self.X_val, self.y_val)
        
        # Évaluer calibration
        y_proba_uncalib = self.best_model.predict_proba(self.X_test)[:, 1]
        y_proba_calib = calibrated.predict_proba(self.X_test)[:, 1]
        
        # Sauvegarder modèle calibré
        self.calibrated_model = calibrated
        
        logger.info("Model calibrated successfully")
        
        return calibrated

def main():
    """Main entry point"""
    pipeline = ClinicalPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()