#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

SRC = Path('graficos/arquivos/models/json')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob('*.json'))
if not files:
    print('No model JSON files found in', SRC)
    raise SystemExit(2)

versions = []
mae_valid = []
mae_test = []

for f in files:
    name = f.stem
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        print('Failed to read', f, e)
        continue

    # heuristics: prefer 'validacao' then 'validation' then 'val'
    v_mae = None
    t_mae = None
    for key in ['validacao', 'validation', 'val', 'valid']:
        if key in data and isinstance(data[key], dict) and 'mae' in data[key]:
            v_mae = data[key]['mae']
            break
    for key in ['teste', 'test', 'tests']:
        if key in data and isinstance(data[key], dict) and 'mae' in data[key]:
            t_mae = data[key]['mae']
            break

    # fallback: look for mae at top-level
    if v_mae is None and 'mae' in data:
        v_mae = data.get('mae')

    versions.append(name)
    mae_valid.append(v_mae)
    mae_test.append(t_mae)

# Plot
x = range(len(versions))
plt.figure(figsize=(max(6, len(versions)*1.2), 4))
if any(m is not None for m in mae_valid):
    plt.plot(x, [m if m is not None else float('nan') for m in mae_valid], marker='o', label='validation MAE')
if any(m is not None for m in mae_test):
    plt.plot(x, [m if m is not None else float('nan') for m in mae_test], marker='s', label='test MAE')

plt.xticks(x, versions, rotation=45, ha='right')
plt.xlabel('Version')
plt.ylabel('MAE')
plt.title('MAE Evolution by Version')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out_path = OUT / 'mae_by_version.png'
plt.savefig(out_path)
print('Wrote', out_path)
