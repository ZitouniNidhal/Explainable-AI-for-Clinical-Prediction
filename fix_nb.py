import json

with open('notebooks/03_modelisation_xai.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['source'] = [line.replace("MODELS_DIR / f'{name.lower()}_model.joblib'", "MODELS_DIR / 'saved_models' / f'{name.lower()}_model.joblib'") for line in c['source']]

with open('notebooks/03_modelisation_xai.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
