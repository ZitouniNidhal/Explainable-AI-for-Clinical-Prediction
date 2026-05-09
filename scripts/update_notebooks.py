import nbformat as nbf
from pathlib import Path

def update_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Update the setup cell to import ConfigManager if needed, or just ensure we have paths
    # (The current setup seems fine)

    # 2. Update the training cell (index 4 in the provided view_file output, which corresponds to 5th cell)
    # Let's find the cell with source containing "factory.create_classifier"
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'factory.create_classifier("lightgbm")' in cell.source:
            cell.source = """from xai_clinical.utils.config import ConfigManager
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd

# Charger la configuration SOTA
config = ConfigManager().config
factory = ClassifierFactory()

results = []
trained_models = {}

print(f"Démarrage du benchmark sur 11 modèles (Optimisation: {config.models.hyperparameter_tuning.n_trials} essais)...")

# Liste des modèles activés dans le config
enabled_models = [c['name'] for c in config.models.classifiers if c.get('enabled', True)]

for model_name in enabled_models:
    print(f"\\n--- Entraînement de {model_name.upper()} ---")
    try:
        # Création et entraînement
        model = factory.create_classifier(model_name)
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Métriques
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "Modèle": model_name,
            "AUC": auc,
            "Accuracy": acc,
            "F1-Score": f1
        })
        trained_models[model_name] = model
        print(f"Résultat: AUC={auc:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"Erreur avec {model_name}: {e}")

# Affichage du tableau comparatif
df_results = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
print("\\n" + "="*40)
print("RÉSULTATS DU BENCHMARK SOTA")
print("="*40)
print(df_results.to_string(index=False))

# Sélectionner le meilleur modèle pour SHAP
best_model_name = df_results.iloc[0]["Modèle"]
model = trained_models[best_model_name]
print(f"\\nMeilleur modèle sélectionné pour XAI : {best_model_name.upper()}")
"""

    # 3. Update the SHAP cell to reflect that it uses the best model
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'Explicabilité avec SHAP' in cell.source:
            cell.source = f"## 3. Explicabilité SOTA avec SHAP\\n\\nNous utilisons ici le meilleur modèle identifié lors du benchmark pour générer des explications de niveau scientifique."

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook {nb_path} updated successfully.")

if __name__ == "__main__":
    nb_file = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks\03_modelisation_xai_v2.ipynb')
    update_notebook(nb_file)
