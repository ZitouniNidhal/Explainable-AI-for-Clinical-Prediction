import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import sys
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, confusion_matrix, average_precision_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Fix paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'figures'
REPORTS_DIR = PROJECT_ROOT / 'reports'

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

(MODELS_DIR / 'saved_models').mkdir(parents=True, exist_ok=True)

print("="*70)
print("MODÉLISATION ET EXPLAINABILITY")
print("="*70)

# Charger les données
X_train = np.load(PROCESSED_DIR / 'X_train.npy')
X_val = np.load(PROCESSED_DIR / 'X_val.npy')
X_test = np.load(PROCESSED_DIR / 'X_test.npy')
y_train = np.load(PROCESSED_DIR / 'y_train.npy')
y_val = np.load(PROCESSED_DIR / 'y_val.npy')
y_test = np.load(PROCESSED_DIR / 'y_test.npy')

try:
    feature_names = joblib.load(PROCESSED_DIR / 'selected_features.joblib')
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
except (NotImplementedError, FileNotFoundError):
    print("[WARNING] Erreur de chargement des features.")
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

scaler = joblib.load(PROCESSED_DIR / 'scaler.joblib')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("\n[TRAINING] Entraînement des modèles")

from src.xai_clinical.models.classifiers import ClassifierFactory

y_train_s = pd.Series(y_train)
counts = y_train_s.value_counts()
pos_label_weight = float(counts.get(0, 1) / counts.get(1, 1))
print(f"Poids de classe dynamique calculé : {pos_label_weight:.2f}")

models = {}
for name in ClassifierFactory.SUPPORTED_MODELS:
    try:
        models[name] = ClassifierFactory.create_classifier(name, random_state=42, pos_label_weight=pos_label_weight)
    except Exception as e:
        print(f"Skipping {name}: {e}")

results = {}

for name, model in models.items():
    print(f"\n{'-'*40}")
    print(f"Entraînement: {name}")
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Erreur d'entraînement pour {name}: {e}")
        continue
    
    # Évaluation
    sets = {
        'Train': (X_train, y_train),
        'Val': (X_val, y_val),
        'Test': (X_test, y_test)
    }
    
    metrics = {'model': model}
    try:
        for set_name, (X, y) in sets.items():
            y_pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
            else:
                y_proba = y_pred
            
            metrics[f'{set_name.lower()}_auc'] = roc_auc_score(y, y_proba)
            metrics[f'{set_name.lower()}_acc'] = (y_pred == y).mean()
            
            if set_name == 'Test':
                metrics['test_precision'] = precision_score(y, y_pred, zero_division=0)
                metrics['test_recall'] = recall_score(y, y_pred, zero_division=0)
                metrics['test_f1'] = f1_score(y, y_pred, zero_division=0)
                metrics['test_avg_precision'] = average_precision_score(y, y_proba)
        
        results[name] = metrics
        
        print(f"Train AUC: {metrics['train_auc']:.3f}")
        print(f"Val AUC:   {metrics['val_auc']:.3f}")
        print(f"Test AUC:  {metrics['test_auc']:.3f}")
        print(f"Test F1:   {metrics['test_f1']:.3f}")
    except Exception as e:
        print(f"Erreur d'évaluation pour {name}: {e}")

if not results:
    print("[ERROR] Aucun modèle n'a pu être entraîné.")
    exit(1)

comparison = pd.DataFrame({
    name: {k: v for k, v in m.items() if k != 'model'}
    for name, m in results.items()
}).T

print("\n[COMPARE] Comparaison:")
print(comparison[['train_auc', 'val_auc', 'test_auc', 'test_f1']])

# ROC Curves
plt.figure(figsize=(10, 8))
for name, metrics in results.items():
    model = metrics['model']
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
    else:
        y_proba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = metrics['test_auc']
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('Taux de Faux Positifs', fontsize=12)
plt.ylabel('Taux de Vrais Positifs', fontsize=12)
plt.title('Courbes ROC - Comparaison des modèles', fontsize=14)
plt.legend(loc='lower right', fontsize=10, bbox_to_anchor=(1.3, 0))
plt.grid(True, alpha=0.3)
plt.savefig(FIGURES_DIR / '03_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Sauvegarder tous les modèles
for name, metrics in results.items():
    joblib.dump(metrics['model'], MODELS_DIR / 'saved_models' / f'{name.lower()}_model.joblib')

best_name = comparison['test_auc'].idxmax()
best_model = results[best_name]['model']
print(f"\n🏆 Meilleur modèle: {best_name}")

print("\n[SHAP] SHAP Explanations")

try:
    # Utiliser TreeExplainer par défaut, sinon KernelExplainer
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        print("[INFO] TreeExplainer échoué, passage à KernelExplainer...")
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test)

    # Gérer le format des shap_values
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # Summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
    plt.title(f'SHAP Global Importance ({best_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Top features
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values_class1).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    shap_importance.to_csv(REPORTS_DIR / 'shap_importance.csv', index=False)
    print("\nTop 10 features SHAP:")
    print(shap_importance.head(10))
except Exception as e:
    print(f"[ERROR] SHAP error: {e}")

print("\n[SUCCESS] Modélisation terminée!")