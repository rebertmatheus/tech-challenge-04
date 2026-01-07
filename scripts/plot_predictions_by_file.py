#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt

IN = Path('graficos/output/predictions_by_file.csv')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

rows = []
with open(IN, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            c = int(r['count']) if r['count']!='' else 0
        except Exception:
            c = 0
        rows.append((r['file'], c))

if not rows:
    print('No rows found in', IN)
    raise SystemExit(2)

# sort desc
rows.sort(key=lambda x: x[1], reverse=True)
# take top 20
top = rows[:20]
files = [Path(r[0]).name for r in top]
counts = [r[1] for r in top]

plt.figure(figsize=(max(6, len(files)*0.6),4))
plt.bar(range(len(files)), counts, color='C0')
plt.xticks(range(len(files)), files, rotation=45, ha='right')
plt.ylabel('Predictions (rows)')
plt.title('Top 20 files by prediction count')
plt.tight_layout()

png = OUT / 'predictions_by_file_top20.png'
svg = OUT / 'predictions_by_file_top20.svg'
plt.savefig(png)
plt.savefig(svg)
print('Wrote', png, svg)
