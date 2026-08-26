import json
from pathlib import Path

nb_path = Path('notebooks/03_modelisation_xai.ipynb')
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 1 (Imports and Dirs)
cell1 = nb['cells'][3] # Index 3 is the first code cell
source1 = cell1['source']
new_source1 = []
for line in source1:
    if 'average_precision_score' in line:
        new_source1.append(line.replace('average_precision_score', 'average_precision_score,'))
        new_source1.append('    precision_score, recall_score, f1_score\n')
    elif 'for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:' in line:
        new_source1.append(line.replace('[MODELS_DIR, FIGURES_DIR, REPORTS_DIR]', "[MODELS_DIR, MODELS_DIR / 'saved_models', FIGURES_DIR, REPORTS_DIR]"))
    else:
        new_source1.append(line)
cell1['source'] = new_source1

# The path fix was already done by fix_nb.py earlier, but let's double check it's in saved_models
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [line.replace("MODELS_DIR / f'{name.lower()}_model.joblib'", "MODELS_DIR / 'saved_models' / f'{name.lower()}_model.joblib'") for line in cell['source']]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
