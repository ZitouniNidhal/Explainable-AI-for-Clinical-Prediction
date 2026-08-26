import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('../data/processed')
MODELS_DIR = Path('../models')
FIGURES_DIR = Path('../figures')
REPORTS_DIR = Path('../reports')

print("="*70)
print("ROBUSTESSE ET ÉVALUATION CLINIQUE")
print("="*70)

# Charger les données
X_test = np.load(PROCESSED_DIR / 'X_test.npy')
y_test = np.load(PROCESSED_DIR / 'y_test.npy')

try:
    feature_names = joblib.load(PROCESSED_DIR / 'selected_features.joblib')
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
except (NotImplementedError, FileNotFoundError):
    print("[WARNING] Erreur de chargement des features.")
    feature_names = []

# Charger tous les modèles
models = {}
for f in (MODELS_DIR / 'saved_models').glob('*_model.joblib'):
    name = f.stem.replace('_model', '').upper()
    try:
        models[name] = joblib.load(f)
    except Exception as e:
        print(f"[ERROR] Impossible de charger {name}: {e}")

print(f"Modèles chargés: {list(models.keys())}")

if not models:
    print("[ERROR] Aucun modèle trouvé dans models/saved_models. Arrêt.")
    exit(1)

print("\n[ROBUSTNESS] Robustesse au bruit gaussien")

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
noise_results = {name: [] for name in models.keys()}

for noise in noise_levels:
    X_noisy = X_test + np.random.normal(0, noise, X_test.shape) * np.std(X_test, axis=0)
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_noisy)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_noisy)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
            else:
                y_proba = model.predict(X_noisy)
            
            auc = roc_auc_score(y_test, y_proba)
            noise_results[name].append(auc)
        except Exception as e:
            print(f"[ERROR] Erreur évaluation {name} (bruit {noise}): {e}")
            noise_results[name].append(0.5)

plt.figure(figsize=(14, 8))
for name, aucs in noise_results.items():
    plt.plot(noise_levels, aucs, marker='o', linewidth=2, label=name)

plt.xlabel('Niveau de bruit (σ)', fontsize=12)
plt.ylabel('AUC-ROC', fontsize=12)
plt.title('Robustesse au bruit gaussien', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)
plt.savefig(FIGURES_DIR / '04_robustness_noise.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[ROBUSTNESS] Robustesse aux données manquantes")

missing_rates = [0.0, 0.05, 0.1, 0.2, 0.3]
missing_results = {name: [] for name in models.keys()}

for rate in missing_rates:
    X_missing = X_test.copy()
    mask = np.random.random(X_missing.shape) < rate
    X_missing[mask] = np.nan
    
    # Imputation médiane
    col_medians = np.nanmedian(X_missing, axis=0)
    for i in range(X_missing.shape[1]):
        X_missing[np.isnan(X_missing[:, i]), i] = col_medians[i]
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_missing)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_missing)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
            else:
                y_proba = model.predict(X_missing)
            
            auc = roc_auc_score(y_test, y_proba)
            missing_results[name].append(auc)
        except Exception as e:
            print(f"[ERROR] Erreur évaluation {name} (manquant {rate}): {e}")
            missing_results[name].append(0.5)

plt.figure(figsize=(14, 8))
for name, aucs in missing_results.items():
    plt.plot(missing_rates, aucs, marker='s', linewidth=2, label=name)

plt.xlabel('Taux de données manquantes', fontsize=12)
plt.ylabel('AUC-ROC', fontsize=12)
plt.title('Robustesse aux données manquantes', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)
plt.savefig(FIGURES_DIR / '04_robustness_missing.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[CLINICAL] Métriques cliniques")

best_model_name = 'LIGHTGBM' if 'LIGHTGBM' in models else list(models.keys())[0]
best_model = models[best_model_name]
print(f"Meilleur modèle pour métriques cliniques: {best_model_name}")

y_pred = best_model.predict(X_test)
if hasattr(best_model, 'predict_proba'):
    y_proba = best_model.predict_proba(X_test)[:, 1]
elif hasattr(best_model, 'decision_function'):
    y_proba = best_model.decision_function(X_test)
    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
else:
    y_proba = y_pred

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

metrics = {
    'auc_roc': roc_auc_score(y_test, y_proba),
    'accuracy': accuracy_score(y_test, y_pred),
    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
    'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)
}

print(f"\n{'='*50}")
print("RAPPORT CLINIQUE")
print(f"{'='*50}")
for k, v in metrics.items():
    print(f"{k:15s}: {v:.3f}")

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Faible risque', 'Haut risque'],
            yticklabels=['Faible risque', 'Haut risque'])
plt.xlabel('Prédiction')
plt.ylabel('Réel')
plt.title('Matrice de Confusion')
plt.savefig(FIGURES_DIR / '04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8, label='Modèle')
plt.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite')
plt.xlabel('Probabilité prédite')
plt.ylabel('Fraction de positifs')
plt.title('Diagramme de Calibration')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig(FIGURES_DIR / '04_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

final_results = {
    'model': best_model_name,
    'metrics': metrics,
    'robustness_noise_10': noise_results[best_model_name][3] if len(noise_results[best_model_name]) > 3 else None,
    'robustness_missing_20': missing_results[best_model_name][3] if len(missing_results[best_model_name]) > 3 else None
}

pd.Series(final_results).to_json(REPORTS_DIR / 'final_results.json')

print("\n[SUCCESS] Évaluation clinique terminée!")
print(f"Résultats sauvegardés dans {REPORTS_DIR}/final_results.json")