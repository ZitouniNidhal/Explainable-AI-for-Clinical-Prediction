import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path
import re

notebooks = [
    '01_exploration_donnees.ipynb',
    '02_preprocessing.ipynb',
    '03_modelisation_xai.ipynb',
    '04_robustesse_evaluation.ipynb'
]

base_dir = Path(r'c:\Users\nidha\Desktop\xai_clinical_prediction\notebooks')

for nb_name in notebooks:
    nb_path = base_dir / nb_name
    if not nb_path.exists():
        continue
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    if not nb.cells:
        continue
        
    # Get the source of the first cell (which contains all the code)
    # Sometimes it's a list of cells, we concatenate all code if multiple exist
    full_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            full_code += cell.source + "\n\n"
            
    # Split the code into chunks based on "# Cellule" or "# Etape"
    # We will use regex to find sections
    lines = full_code.split('\n')
    
    chunks = []
    current_chunk_code = []
    current_title = "Initialisation et Imports"
    
    for line in lines:
        match = re.match(r'^#\s*(?:Cellule|Etape|Étape)\s*\d*\s*:?\s*(.*)', line, re.IGNORECASE)
        if match:
            # Save the previous chunk if it has code
            if any(current_chunk_code):
                chunks.append((current_title, '\n'.join(current_chunk_code).strip()))
            
            # Start a new chunk
            current_title = match.group(1).strip()
            if not current_title:
                current_title = "Suite"
            current_chunk_code = []
        else:
            current_chunk_code.append(line)
            
    if any(current_chunk_code):
        chunks.append((current_title, '\n'.join(current_chunk_code).strip()))
        
    # Create a new notebook
    new_nb = new_notebook()
    
    # Add a main title markdown
    main_title = nb_name.replace('.ipynb', '').replace('_', ' ').title()
    new_nb.cells.append(new_markdown_cell(f"# {main_title}\n\nCe notebook a été automatiquement structuré pour une meilleure lisibilité académique."))
    
    for title, code in chunks:
        if not code:
            continue
            
        # Add markdown cell for the title
        if title:
            new_nb.cells.append(new_markdown_cell(f"### {title}"))
            
        # Add code cell
        new_nb.cells.append(new_code_cell(code))
        
    # Save the new notebook, overwriting the old one
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(new_nb, f)
        
    print(f"Refactored {nb_name} into {len(chunks)} sections.")
