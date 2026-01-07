#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

SRC_DIR = Path('graficos/arquivos')
OUT_DIR = Path('graficos/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = list(SRC_DIR.rglob('*.parquet'))
if not parquet_files:
    print('No parquet files found under', SRC_DIR)
    sys.exit(2)

results = []
total = 0
for p in sorted(parquet_files):
    count = None
    # try pandas
    if HAS_PANDAS:
        try:
            df = pd.read_parquet(p)
            count = len(df)
        except Exception:
            count = None
    # try pyarrow metadata
    if count is None and HAS_PYARROW:
        try:
            m = pq.ParquetFile(str(p)).metadata
            count = m.num_rows
        except Exception:
            count = None
    # fallback: try reading in chunks
    if count is None and HAS_PANDAS:
        try:
            # read with columns=None just to count rows; may still load
            df = pd.read_parquet(p)
            count = len(df)
        except Exception:
            count = None
    if count is None:
        results.append({'file': str(p), 'count': ''})
        print('Could not determine rows for', p)
    else:
        results.append({'file': str(p), 'count': int(count)})
        total += int(count)

# write CSV per-file
csv_path = OUT_DIR / 'predictions_by_file.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.DictWriter(cf, fieldnames=['file','count'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# write total
txt_path = OUT_DIR / 'total_predictions.txt'
with open(txt_path, 'w', encoding='utf-8') as tf:
    tf.write(str(total))

print('Total predictions (sum of rows across parquet files):', total)
print('Wrote:', csv_path, txt_path)

