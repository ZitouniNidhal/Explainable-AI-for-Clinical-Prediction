import nbformat
from pathlib import Path

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')
files = ['03_modelisation_xai.ipynb', '04_robustesse_evaluation.ipynb']

for f in files:
    nb_path = base_dir / f
    with open(nb_path, 'r', encoding='utf-8') as f_in:
        nb = nbformat.read(f_in, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if "feature_names = joblib.load(PROCESSED_DIR / 'selected_features.joblib')" in cell.source:
                replacement = """try:
    feature_names = joblib.load(PROCESSED_DIR / 'selected_features.joblib')
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
except NotImplementedError:
    print("⚠️ Avertissement: Erreur de chargement des features. Veuillez relancer la dernière cellule du notebook 02_preprocessing pour corriger le format.")
    feature_names = []"""
                cell.source = cell.source.replace("feature_names = joblib.load(PROCESSED_DIR / 'selected_features.joblib')", replacement)

    with open(nb_path, 'w', encoding='utf-8') as f_out:
        nbformat.write(nb, f_out)
        
print("Patched joblib.load in notebooks.")
