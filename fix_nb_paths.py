import json
from pathlib import Path

def fix_nb_paths(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Change relative paths to absolute ones using PROJECT_ROOT or direct literals
                line = line.replace("Path('../data/processed')", "Path(r'C:\\Users\\nidha\\Desktop\\xai_clinical_prediction\\data\\processed')")
                line = line.replace("Path('../models')", "Path(r'C:\\Users\\nidha\\Desktop\\xai_clinical_prediction\\models')")
                line = line.replace("Path('../figures')", "Path(r'C:\\Users\\nidha\\Desktop\\xai_clinical_prediction\\figures')")
                line = line.replace("Path('../reports')", "Path(r'C:\\Users\\nidha\\Desktop\\xai_clinical_prediction\\reports')")
                new_source.append(line)
            cell['source'] = new_source
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_nb_paths('notebooks/03_modelisation_xai.ipynb')
fix_nb_paths('notebooks/04_robustesse_evaluation.ipynb')
