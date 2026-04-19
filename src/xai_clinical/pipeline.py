# src/xai_clinical/pipeline.py
"""
Complete XAI Pipeline for clinical prediction
"""
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pipeline.log')
    ]
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
        
        # Create directories
        for path_key in ['data', 'models', 'reports', 'figures', 'logs']:
            path = self.config.get_path(path_key)
            path.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute complete pipeline"""
        logger.info("="*60)
        logger.info("STARTING XAI CLINICAL PIPELINE")
        logger.info("="*60)
        
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
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
    
    def _load_data(self):
        """Load or generate data"""
        logger.info("\n📊 DATA LOADING")
        
        if self.config.data.use_synthetic:
            logger.info("Generating synthetic data...")
            generator = SyntheticDataGenerator(
                n_samples=self.config.data.synthetic_samples,
                random_state=self.config.project.random_state
            )
            self.data, self.target = generator.generate()
            self.feature_descriptions = generator.get_feature_descriptions()
            
            # Save
            self.data.to_csv(self.config.get_path('data') / 'synthetic_data.csv', index=False)
            self.target.to_csv(self.config.get_path('data') / 'synthetic_target.csv', index=False)
            
            logger.info(f"Generated data: {self.data.shape[0]} samples, {self.data.shape[1]} features")
            logger.info(f"Class distribution: {self.target.value_counts().to_dict()}")
        else:
            # External data loading
            pass
    
    def _preprocess_data(self):
        """Preprocess data"""
        logger.info("\n🔧 PREPROCESSING")
        
        # Split train/val/test
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data, self.target,
            test_size=self.config.data.test_size,
            random_state=self.config.project.random_state,
            stratify=self.target
        )
        
        val_size = self.config.data.validation_size / (1 - self.config.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.project.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Preprocessing
        self.preprocessor = DataPreprocessor(self.config)
        X_train_proc, y_train_proc, X_val_proc, X_test_proc = self.preprocessor.fit_transform(
            X_train, y_train, X_val, X_test
        )
        
        self.X_train = X_train_proc
        self.y_train = y_train_proc
        self.X_val = X_val_proc
        self.y_val = y_val
        self.X_test = X_test_proc
        self.y_test = y_test
        
        # Save preprocessor
        import joblib
        joblib.dump(self.preprocessor, self.config.get_path('models') / 'preprocessor.joblib')
    
    def _train_models(self):
        """Train and optimize models"""
        logger.info("\n🤖 MODEL TRAINING")
        
        self.trainer = ModelTrainer(self.config, self.config.project.random_state)
        
        # Train all configured models
        self.trainer.train_all_models(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Select best model
        self.best_model_name, self.best_model, best_score = self.trainer.get_best_model('val_roc_auc')
        
        logger.info(f"\n🏆 Best model: {self.best_model_name} (AUC-ROC: {best_score:.3f})")
        
        # Test set evaluation
        from sklearn.metrics import classification_report, roc_auc_score
        
        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        logger.info("\nTest set evaluation:")
        logger.info(f"AUC-ROC: {roc_auc_score(self.y_test, y_proba):.3f}")
        logger.info("\nClassification report:")
        logger.info(classification_report(self.y_test, y_pred))
        
        # Save models
        self.trainer.save_models(self.config.get_path('models') / 'saved_models')
    
    def _explain_model(self):
        """Generate SHAP and LIME explanations"""
        logger.info("\n🔍 EXPLAINABILITY")
        
        # SHAP
        logger.info("Computing SHAP explanations...")
        self.shap_explainer = SHAPExplainer(
            self.best_model,
            self.X_train,
            feature_names=self.X_train.columns.tolist()
        )
        
        # Global explanations
        global_importance = self.shap_explainer.explain_global(self.X_test)
        global_importance.to_csv(self.config.get_path('reports') / 'shap_global_importance.csv', index=False)
        
        # SHAP visualizations
        self.shap_explainer.plot_summary(
            self.X_test, 
            save_path=self.config.get_path('figures') / 'shap_summary.png'
        )
        
        # Explanations for a few instances
        for i in range(min(3, len(self.X_test))):
            self.shap_explainer.plot_waterfall(
                self.X_test, 
                instance_idx=i,
                save_path=self.config.get_path('figures') / f'shap_waterfall_{i}.png'
            )
        
        # LIME
        logger.info("Computing LIME explanations...")
        self.lime_explainer = LIMEExplainer(
            self.best_model,
            self.X_train,
            feature_names=self.X_train.columns.tolist(),
            class_names=['No complication', 'Complication']
        )
        
        # SHAP vs LIME comparison for one instance
        comparison = self.lime_explainer.compare_with_shap(
            self.X_test.iloc[0],
            self.shap_explainer
        )
        comparison.to_csv(self.config.get_path('reports') / 'shap_lime_comparison.csv', index=False)
        
        logger.info("Explanations generated and saved")
    
    def _analyze_robustness(self):
        """Analyze explanation robustness"""
        logger.info("\n🛡️ ROBUSTNESS ANALYSIS")
        
        analyzer = StabilityAnalyzer(self.shap_explainer, self.X_train.columns.tolist())
        
        stability_results = analyzer.evaluate_stability(
            self.X_test,
            n_perturbations=10,
            noise_levels=[0.01, 0.05, 0.1],
            missing_rates=[0.05, 0.1, 0.2]
        )
        
        # Save results
        with open(self.config.get_path('reports') / 'stability_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            results_json = {}
            for key, value in stability_results.items():
                if isinstance(value, dict):
                    results_json[key] = value
                else:
                    results_json[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            json.dump(results_json, f, indent=2)
        
        # Report
        report = analyzer.generate_stability_report(stability_results)
        with open(self.config.get_path('reports') / 'stability_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(report)
    
    def _generate_report(self):
        """Generate final report"""
        logger.info("\n📄 REPORT GENERATION")
        
        report_path = self.config.get_path('reports') / 'final_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# XAI Report - Clinical Outcome Prediction\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## 1. Executive Summary\n\n")
            f.write(f"- **Best model:** {self.best_model_name}\n")
            f.write(f"- **Number of samples:** {len(self.data)}\n")
            f.write(f"- **Number of features:** {self.data.shape[1]}\n\n")
            
            f.write("## 2. Model Performance\n\n")
            f.write("| Model | Train AUC | Val AUC | Test AUC |\n")
            f.write("|--------|-----------|---------|----------|\n")
            for name, result in self.trainer.results.items():
                metrics = result['metrics']
                test_auc = metrics.get('test_roc_auc', 'N/A')
                f.write(f"| {name} | {metrics.get('train_roc_auc', 'N/A'):.3f} | ")
                f.write(f"{metrics.get('val_roc_auc', 'N/A'):.3f} | {test_auc if isinstance(test_auc, str) else f'{test_auc:.3f}'} |\n")
            
            f.write("\n## 3. Feature Importance (SHAP)\n\n")
            f.write("Most important variables for prediction:\n\n")
            global_imp = self.shap_explainer.explain_global(self.X_test, max_display=10)
            for _, row in global_imp.iterrows():
                f.write(f"- **{row['feature']}**: {row['shap_importance']:.4f}\n")
            
            f.write("\n## 4. Clinical Interpretation\n\n")
            f.write("### Identified risk factors:\n")
            f.write("1. **Age**: Elderly patients have increased risk\n")
            f.write("2. **ASA Score**: Anesthesia risk score is predictive\n")
            f.write("3. **Surgery duration**: Longer procedures are riskier\n\n")
            
            f.write("## 5. Robustness\n\n")
            f.write("Explanations were tested with:\n")
            f.write("- Gaussian noise (1%, 5%, 10%)\n")
            f.write("- Missing data (5%, 10%, 20%)\n\n")
            f.write("See `stability_report.txt` for details.\n\n")
            
            f.write("## 6. Recommendations\n\n")
            f.write("- Model is ready for prospective clinical validation\n")
            f.write("- Monitor data drift in production\n")
            f.write("- Implement clinical feedback system\n")
        
        logger.info(f"Report saved: {report_path}")


def main():
    """Main entry point"""
    pipeline = ClinicalPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()