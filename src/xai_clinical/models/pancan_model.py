
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class XAIPancanModel:
    """Pipeline XAI pour la prédiction basée sur les données PANCAN fusionnées."""
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.shap_values = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _build_model(self) -> Any:
        """Construit le modèle selon le type spécifié en utilisant la factory."""
        from xai_clinical.models.classifiers import ClassifierFactory
        
        # Mapping for naming consistency if needed, but we can use direct names
        return ClassifierFactory.create_classifier(self.model_type, random_state=self.random_state)

    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, scale: bool = True):
        """Prépare les données: split train/test et scaling."""
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if scale:
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns, index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns, index=self.X_test.index
            )
        
        logger.info(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self) -> Any:
        """Entraîne le modèle."""
        self.model = self._build_model()
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        
        logger.info(f"AUC-ROC test: {auc:.4f}")
        logger.info(f"\\n{classification_report(self.y_test, y_pred)}")
        return self.model
    
    def cross_validate(self, cv: int = 5) -> Dict[str, float]:
        """Validation croisée."""
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                cv=cv_splitter, scoring='roc_auc')
        
        results = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'all_scores': scores.tolist()
        }
        logger.info(f"CV AUC: {results['mean_auc']:.4f} (+/- {results['std_auc']*2:.4f})")
        return results
    
    def explain_with_shap(self, background_samples: int = 100, test_samples: int = 100):
        """Calcule les valeurs SHAP pour l'explicabilité."""
        logger.info("Calcul des valeurs SHAP...")
        
        background = shap.sample(self.X_train, background_samples)
        
        if self.model_type in ["random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost", "extra_trees", "ada_boost"]:
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)


        
        X_test_sample = self.X_test.iloc[:test_samples]
        self.shap_values = self.shap_explainer.shap_values(X_test_sample)
        
        logger.info(f"SHAP values: shape={np.array(self.shap_values).shape}")
        return self.shap_values
    
    def get_gene_importance(self, top_n: int = 50) -> pd.DataFrame:
        """Extrait l'importance des gènes depuis les valeurs SHAP."""
        if self.shap_values is None:
            self.explain_with_shap()
        
        if isinstance(self.shap_values, list):
            shap_array = np.array(self.shap_values[1])
        else:
            shap_array = np.array(self.shap_values)
        
        mean_shap = np.abs(shap_array).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        
        importance_df['feature_type'] = importance_df['feature'].apply(
            lambda x: 'gene' if not x.startswith('clinical_') and not x.startswith('PC_') 
            else 'clinical'
        )
        
        logger.info(f"Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['shap_importance']:.4f}")
        
        return importance_df.head(top_n)
    
    def plot_shap_summary(self, save_path: Optional[str] = None):
        """Génère le summary plot SHAP."""
        if self.shap_values is None:
            self.explain_with_shap()
        
        plt.figure(figsize=(12, 10))
        X_test_sample = self.X_test.iloc[:100]
        
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[1], X_test_sample, 
                            feature_names=self.feature_names, show=False)
        else:
            shap.summary_plot(self.shap_values, X_test_sample, 
                            feature_names=self.feature_names, show=False)
        
        plt.title("SHAP Summary Plot - Importance des Features")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot sauvegardé: {save_path}")
        plt.show()
    
    def plot_gene_importance(self, top_n: int = 30, save_path: Optional[str] = None):
        """Plot horizontal des gènes les plus importants."""
        importance = self.get_gene_importance(top_n)
        
        plt.figure(figsize=(10, 12))
        colors = ['#e74c3c' if t == 'gene' else '#3498db' for t in importance['feature_type']]
        
        sns.barplot(data=importance, y='feature', x='shap_importance', 
                   palette=colors, orient='h')
        plt.title(f"Top {top_n} Features Importantes (SHAP)")
        plt.xlabel("Importance SHAP moyenne (|valeur|)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot la courbe ROC."""
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc = roc_auc_score(self.y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_dir: str):
        """Sauvegarde les résultats et métriques."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        results = pd.DataFrame({
            'true': self.y_test.values,
            'predicted': y_pred,
            'probability': y_prob
        }, index=self.X_test.index)
        results.to_csv(output_dir / "predictions.csv")
        
        importance = self.get_gene_importance(top_n=200)
        importance.to_csv(output_dir / "shap_importance.csv", index=False)
        
        metrics = {
            'model_type': self.model_type,
            'auc_roc': roc_auc_score(self.y_test, y_prob),
            'n_features': len(self.feature_names),
            'n_train': len(self.X_train),
            'n_test': len(self.X_test)
        }
        
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans {output_dir}")


class GenePathwayAnalyzer:
    """Analyse des pathways biologiques à partir des gènes importants."""
    
    def __init__(self, gene_importance: pd.DataFrame):
        self.gene_importance = gene_importance
        self.top_genes = gene_importance[gene_importance['feature_type'] == 'gene']['feature'].tolist()
    
    def map_to_pathways(self, pathway_db: Optional[Dict] = None) -> pd.DataFrame:
        """Mappe les gènes importants vers des pathways connus."""
        known_pathways = {
            'PI3K-AKT': ['PIK3CA', 'AKT1', 'AKT2', 'PTEN', 'MTOR'],
            'MAPK': ['KRAS', 'BRAF', 'MAPK1', 'MAPK3', 'EGFR'],
            'Cell_Cycle': ['TP53', 'RB1', 'CDKN2A', 'CCND1', 'MYC'],
            'DNA_Repair': ['BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'PALB2'],
            'ER_Signaling': ['ESR1', 'ESR2', 'PGR', 'GREB1', 'TFF1'],
            'HER2': ['ERBB2', 'ERBB3', 'ERBB4', 'GRB7'],
            'Apoptosis': ['BCL2', 'BAX', 'CASP3', 'CASP8', 'FAS'],
            'Angiogenesis': ['VEGFA', 'VEGFR1', 'VEGFR2', 'HIF1A']
        }
        
        pathway_scores = []
        
        for pathway, genes in known_pathways.items():
            matched = [g for g in genes if g in self.top_genes]
            if matched:
                scores = self.gene_importance[
                    self.gene_importance['feature'].isin(matched)
                ]['shap_importance'].mean()
                
                pathway_scores.append({
                    'pathway': pathway,
                    'score': scores,
                    'n_genes': len(matched),
                    'genes': ', '.join(matched)
                })
        
        df = pd.DataFrame(pathway_scores).sort_values('score', ascending=False)
        return df
    
    def plot_pathway_enrichment(self, save_path: Optional[str] = None):
        """Plot l'enrichissement des pathways."""
        pathways = self.map_to_pathways()
        
        if pathways.empty:
            logger.warning("Aucun pathway enrichi trouvé")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pathways, x='score', y='pathway', palette='viridis')
        plt.title("Enrichissement des Pathways - Gènes Importants SHAP")
        plt.xlabel("Score d'importance moyen")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
