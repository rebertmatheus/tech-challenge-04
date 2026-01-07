#!/usr/bin/env python3
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

SRC = Path('graficos/arquivos/models/json')
OUT = Path('graficos/output/mae_evolucao_por_versao')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob('*.json'))
if not files:
    print('Nenhum arquivo JSON de modelo encontrado em', SRC)
    raise SystemExit(2)

rows = []
for f in files:
    name = f.stem
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        print('Falha ao ler', f, e)
        continue

    # buscar MAE em validação e teste
    def busca_mae(sec_keys):
        for k in sec_keys:
            if k in data and isinstance(data[k], dict) and 'mae' in data[k]:
                return data[k]['mae']
        return None

    v_mae = busca_mae(['validacao', 'validation', 'val', 'valid'])
    t_mae = busca_mae(['teste', 'test', 'tests'])

    rows.append({'versao': name, 'mae_validacao': v_mae, 'mae_teste': t_mae})

# salvar CSV
csv_path = OUT / 'mae_evolucao_por_versao.csv'
fieldnames = ['versao','mae_validacao','mae_teste']
with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# plot
versoes = [r['versao'] for r in rows]
mae_val = [r['mae_validacao'] for r in rows]
mae_test = [r['mae_teste'] for r in rows]

x = range(len(versoes))
plt.figure(figsize=(max(6, len(versoes)*1.2), 4))
if any(v is not None for v in mae_val):
    plt.plot(x, [v if v is not None else float('nan') for v in mae_val], marker='o', label='Validação')
if any(v is not None for v in mae_test):
    plt.plot(x, [v if v is not None else float('nan') for v in mae_test], marker='s', label='Teste')

plt.xticks(x, versoes, rotation=45, ha='right')
plt.xlabel('Versão')
plt.ylabel('MAE')
plt.title('Evolução do MAE por Versão')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

png_path = OUT / 'mae_evolucao_por_versao.png'
plt.savefig(png_path)
print('Escrito:', csv_path, png_path)
