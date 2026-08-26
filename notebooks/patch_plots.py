import nbformat
from pathlib import Path

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')
nb_path = base_dir / '04_robustesse_evaluation.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Fix plt.figure size and plt.legend for multiple models
        if "plt.figure(figsize=(10, 6))" in cell.source:
            cell.source = cell.source.replace("plt.figure(figsize=(10, 6))", "plt.figure(figsize=(14, 8))")
            
        if "plt.legend()" in cell.source:
            cell.source = cell.source.replace("plt.legend()", "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)\nplt.tight_layout()")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Patched plot display.")
