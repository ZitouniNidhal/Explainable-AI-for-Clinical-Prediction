import nbformat
from pathlib import Path

notebooks = [
    '01_exploration_donnees.ipynb',
    '02_preprocessing.ipynb',
    '03_modelisation_xai.ipynb',
    '04_robustesse_evaluation.ipynb'
]

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')

for nb_name in notebooks:
    print(f"\n--- {nb_name} ---")
    nb_path = base_dir / nb_name
    if not nb_path.exists():
        print("File not found")
        continue
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    code_cells = 0
    md_cells = 0
    errors = 0
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            md_cells += 1
            # print("MD:", cell.source[:50].replace('\n', ' '))
        elif cell.cell_type == 'code':
            code_cells += 1
            for output in cell.get('outputs', []):
                if output.output_type == 'error':
                    errors += 1
                    
    print(f"Markdown cells: {md_cells}, Code cells: {code_cells}, Cells with errors: {errors}")
