import nbformat
from pathlib import Path

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')
nb_path = base_dir / '04_robustesse_evaluation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
    
fallback = "model.predict_proba({0})[:, 1] if hasattr(model, 'predict_proba') else (model.decision_function({0}) - model.decision_function({0}).min()) / (model.decision_function({0}).max() - model.decision_function({0}).min() + 1e-8)"
fallback_best = "best_model.predict_proba({0})[:, 1] if hasattr(best_model, 'predict_proba') else (best_model.decision_function({0}) - best_model.decision_function({0}).min()) / (best_model.decision_function({0}).max() - best_model.decision_function({0}).min() + 1e-8)"

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Revert the old patch if it exists and apply the new one
        src = cell.source
        
        # Original sources might have "y_proba = model.predict_proba(X_noisy)[:, 1]"
        src = src.replace("y_proba = model.predict_proba(X_noisy)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_noisy)", "y_proba = model.predict_proba(X_noisy)[:, 1]")
        src = src.replace("y_proba = model.predict_proba(X_missing)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_missing)", "y_proba = model.predict_proba(X_missing)[:, 1]")
        src = src.replace("y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test)", "y_proba = best_model.predict_proba(X_test)[:, 1]")
        
        src = src.replace("y_proba = model.predict_proba(X_noisy)[:, 1]", f"y_proba = {fallback.format('X_noisy')}")
        src = src.replace("y_proba = model.predict_proba(X_missing)[:, 1]", f"y_proba = {fallback.format('X_missing')}")
        src = src.replace("y_proba = best_model.predict_proba(X_test)[:, 1]", f"y_proba = {fallback_best.format('X_test')}")
        
        cell.source = src

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Patched predict_proba with minmax scaler fallback.")
