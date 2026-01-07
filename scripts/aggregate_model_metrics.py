#!/usr/bin/env python3
import json
from pathlib import Path
import csv

SRC = Path('graficos/arquivos/models/json')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob('*.json'))
if not files:
    print('No model JSON files found in', SRC)
    raise SystemExit(2)

rows = []
for f in files:
    name = f.stem
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        print('Failed to read', f, e)
        continue

    def get_metric(section_keys, metric_name):
        for key in section_keys:
            if key in data and isinstance(data[key], dict) and metric_name in data[key]:
                return data[key].get(metric_name)
        return None

    row = {'version': name}
    metrics = ['mae','mse','rmse','mape','r2','directional_accuracy']
    for m in metrics:
        row[f'validation_{m}'] = get_metric(['validacao','validation','val','valid'], m)
        row[f'test_{m}'] = get_metric(['teste','test','tests'], m)

    # include config as compact JSON
    cfg = data.get('config') or data.get('configuration') or {}
    try:
        row['config'] = json.dumps(cfg, ensure_ascii=False)
    except Exception:
        row['config'] = str(cfg)

    rows.append(row)

# write CSV
csv_path = OUT / 'versions_metrics.csv'
fieldnames = ['version'] + [x for m in ['mae','mse','rmse','mape','r2','directional_accuracy'] for x in (f'validation_{m}', f'test_{m}')] + ['config']
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# attempt HTML table (simple)
html_path = OUT / 'versions_metrics.html'
with open(html_path, 'w', encoding='utf-8') as hf:
    hf.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>Versions metrics</title></head><body>\n')
    hf.write('<h2>Versions metrics</h2>\n')
    hf.write('<table border="1" cellspacing="0" cellpadding="4">\n')
    # header
    hf.write('<tr>')
    for col in fieldnames:
        hf.write(f'<th>{col}</th>')
    hf.write('</tr>\n')
    # rows
    for r in rows:
        hf.write('<tr>')
        for col in fieldnames:
            v = r.get(col, '')
            if v is None:
                v = ''
            hf.write(f'<td>{str(v)}</td>')
        hf.write('</tr>\n')
    hf.write('</table>\n</body></html>')

print('Wrote:', csv_path, html_path)
