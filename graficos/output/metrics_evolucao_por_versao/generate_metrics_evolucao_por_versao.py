#!/usr/bin/env python3
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

SRC = Path('graficos/arquivos/models/json')
OUT = Path('graficos/output/metrics_evolucao_por_versao')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob('*.json'))
if not files:
    print('Nenhum arquivo JSON de modelo encontrado em', SRC)
    raise SystemExit(2)

metrics = ['mse','rmse','mape','r2','directional_accuracy']
rows = []
for f in files:
    name = f.stem
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        print('Falha ao ler', f, e)
        continue

    def find_metric(section_keys, metric_name):
        for k in section_keys:
            if k in data and isinstance(data[k], dict) and metric_name in data[k]:
                return data[k].get(metric_name)
        return None

    row = {'versao': name}
    for m in metrics:
        row[f'validacao_{m}'] = find_metric(['validacao','validation','val','valid'], m)
        row[f'teste_{m}'] = find_metric(['teste','test','tests'], m)
    rows.append(row)

# write CSV
csv_path = OUT / 'metrics_evolucao_por_versao.csv'
fieldnames = ['versao'] + [f'validacao_{m}' for m in metrics] + [f'teste_{m}' for m in metrics]
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# generate one plot per metric
versoes = [r['versao'] for r in rows]
for m in metrics:
    val = [r.get(f'validacao_{m}') for r in rows]
    tst = [r.get(f'teste_{m}') for r in rows]
    x = range(len(versoes))
    plt.figure(figsize=(max(6, len(versoes)*1.2), 4))
    if any(v is not None for v in val):
        plt.plot(x, [v if v is not None else float('nan') for v in val], marker='o', label='Validação')
    if any(t is not None for t in tst):
        plt.plot(x, [t if t is not None else float('nan') for t in tst], marker='s', label='Teste')
    plt.xticks(x, versoes, rotation=45, ha='right')
    plt.xlabel('Versão')
    # portuguese ylabel: metric name
    ylabel = m
    plt.ylabel(ylabel)
    plt.title(f'Evolução de {m} por versão')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = OUT / f'{m}_evolucao_por_versao.png'
    plt.savefig(png_path)
    plt.close()

print('Escrito CSV e gráficos em', OUT)
print(csv_path)
