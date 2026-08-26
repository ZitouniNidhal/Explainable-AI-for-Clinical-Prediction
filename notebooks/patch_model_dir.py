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
            if "MODELS_DIR.glob('*_model.joblib')" in cell.source:
                cell.source = cell.source.replace(
                    "MODELS_DIR.glob('*_model.joblib')", 
                    "(MODELS_DIR / 'saved_models').glob('*_model.joblib')"
                )

    with open(nb_path, 'w', encoding='utf-8') as f_out:
        nbformat.write(nb, f_out)
        
print("Patched notebooks to load models from saved_models directory.")
