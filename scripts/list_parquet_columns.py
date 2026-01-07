from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "graficos" / "arquivos"
parquets = list(DATA_DIR.rglob("*.parquet"))
if not parquets:
    print("No parquet files found")
    raise SystemExit(1)

all_cols = {}
for p in parquets:
    try:
        df = pd.read_parquet(p)
        cols = list(df.columns)
        all_cols[str(p.relative_to(ROOT))] = cols
    except Exception as e:
        all_cols[str(p.relative_to(ROOT))] = f"ERROR: {e}"

# Print per-file
for path, cols in all_cols.items():
    if isinstance(cols, list):
        print(f"FILE: {path}\nCOLUMNS: {', '.join(cols)}\n")
    else:
        print(f"FILE: {path}\n{cols}\n")

# Aggregate unique columns
unique = set()
for cols in all_cols.values():
    if isinstance(cols, list):
        unique.update(cols)

print("AGGREGATED COLUMNS:\n", '\n'.join(sorted(unique)))
