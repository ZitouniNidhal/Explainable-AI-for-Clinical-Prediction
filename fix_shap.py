import json
from pathlib import Path

def fix_shap_logic(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'shap.TreeExplainer(best_model)' in ''.join(cell['source']):
            new_source = []
            for line in cell['source']:
                if 'explainer = shap.TreeExplainer(best_model)' in line:
                    new_source.append("try:\n")
                    new_source.append("    explainer = shap.TreeExplainer(best_model)\n")
                    new_source.append("except:\n")
                    new_source.append("    explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train, 50))\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_shap_logic('notebooks/03_modelisation_xai.ipynb')
