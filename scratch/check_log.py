log_path = r"c:\Users\nidha\Desktop\mémoire_v_2 (1)\chapitre6_resultats.log"

try:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_lines = f.readlines()
    print(f"Log file loaded. Total lines: {len(log_lines)}")
except Exception as e:
    print(f"Error loading log: {e}")
    import sys
    sys.exit(1)

errors = []
for idx, line in enumerate(log_lines):
    if line.startswith("!"):
        # Collect the error and some context
        context = log_lines[max(0, idx-2):min(len(log_lines), idx+6)]
        errors.append((idx + 1, line.strip(), context))

print(f"Found {len(errors)} error markers starting with '!':")
for line_num, err, ctx in errors:
    print(f"\n--- Error on log line {line_num}: {err} ---")
    print("".join(ctx))
