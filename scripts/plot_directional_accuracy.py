#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC = Path('graficos/arquivos/models/json')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob('*.json'))
if not files:
    print('No model JSON files found in', SRC)
    raise SystemExit(2)

# collect per-version per-ticker directional_accuracy if present
data = {}  # version -> {ticker: accuracy}
all_tickers = set()
for f in files:
    name = f.stem
    with open(f, 'r', encoding='utf-8') as fh:
        j = json.load(fh)
    found = False
    # look for explicit per-ticker sections
    # common possibilities: 'by_ticker', 'per_ticker', 'tickers', 'metrics_per_ticker'
    for section in ['by_ticker','per_ticker','tickers','metrics_per_ticker','per_ticker_metrics']:
        if section in j and isinstance(j[section], dict):
            tickers = {}
            for t, metrics in j[section].items():
                if isinstance(metrics, dict) and 'directional_accuracy' in metrics:
                    tickers[t] = metrics['directional_accuracy']
            if tickers:
                data[name] = tickers
                all_tickers.update(tickers.keys())
                found = True
                break
    if found:
        continue
    # also check if j contains many keys that look like tickers (e.g., uppercase len 4-6)
    # heuristic: keys that are all-uppercase and length 3-6 and each value is a dict with directional_accuracy
    ticker_candidates = {}
    for k, v in j.items():
        if isinstance(k, str) and k.isupper() and 1 < len(k) <= 6 and isinstance(v, dict) and 'directional_accuracy' in v:
            ticker_candidates[k] = v['directional_accuracy']
    if ticker_candidates:
        data[name] = ticker_candidates
        all_tickers.update(ticker_candidates.keys())
        found = True

if not data:
    print('No per-ticker directional_accuracy found in model JSONs. Cannot produce plot.')
    raise SystemExit(0)

# Build matrix: rows=versions, cols=tickers
versions = sorted(data.keys())
tickers = sorted(all_tickers)
mat = np.full((len(versions), len(tickers)), np.nan)
for i, v in enumerate(versions):
    for j_idx, t in enumerate(tickers):
        mat[i, j_idx] = data.get(v, {}).get(t, np.nan)

# Plot heatmap
fig, ax = plt.subplots(figsize=(max(6, len(tickers)*0.6), max(4, len(versions)*0.6)))
c = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(len(tickers)))
ax.set_xticklabels(tickers, rotation=45, ha='right')
ax.set_yticks(range(len(versions)))
ax.set_yticklabels(versions)
ax.set_xlabel('Ticker')
ax.set_ylabel('Version')
ax.set_title('Directional Accuracy by Version and Ticker')
fig.colorbar(c, ax=ax, label='Directional accuracy')
plt.tight_layout()
out = OUT / 'directional_accuracy_by_version_ticker.png'
plt.savefig(out)
print('Wrote', out)
