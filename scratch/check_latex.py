import re

file_path = r"c:\Users\nidha\Desktop\mémoire_v_2 (1)\chapitre6_resultats.tex"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"File loaded successfully. Total length: {len(content)}")
except Exception as e:
    print(f"Error loading file: {e}")
    import sys
    sys.exit(1)

# Find all occurrences of \Rightarrow and \downarrow
for match in re.finditer(r"\\(Rightarrow|downarrow)\b", content):
    start = match.start()
    end = match.end()
    # Find line number
    line_num = content[:start].count("\n") + 1
    # Check if enclosed in $...$
    # We can check if the count of $ signs from the start of the document up to the command is odd or even.
    # Note: this is a simple heuristic.
    before = content[:start]
    dollar_count = before.count("$")
    is_in_math_mode = (dollar_count % 2 == 1)
    
    # Also get the surrounding context
    context_start = max(0, start - 40)
    context_end = min(len(content), end + 40)
    context = content[context_start:context_end].replace("\n", " ")
    
    print(f"Line {line_num}: command={match.group(0)}, math_mode={is_in_math_mode}")
    print(f"   Context: ... {context} ...")
