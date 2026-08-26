import nbformat
from pathlib import Path

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')
files = ['02_preprocessing.ipynb', '03_modelisation_xai.ipynb', '04_robustesse_evaluation.ipynb']

for f in files:
    nb_path = base_dir / f
    with open(nb_path, 'r', encoding='utf-8') as f_in:
        nb = nbformat.read(f_in, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'joblib.dump(selected_features' in cell.source or 'joblib.dump(' in cell.source and 'selected_features' in cell.source:
                cell.source = cell.source.replace('joblib.dump(selected_features', 'joblib.dump(list(selected_features)')
                
            if 'feature_names = joblib.load' in cell.source:
                # No change needed if we fix the saved file, but let's make sure to convert it to list if needed
                pass

    with open(nb_path, 'w', encoding='utf-8') as f_out:
        nbformat.write(nb, f_out)
        
print("Fixed joblib dumps to use list() instead of pandas Index.")
