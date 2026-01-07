#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Pasta de entrada e saída
SRC = Path('graficos/arquivos')
OUT = Path('graficos/output/predicao_por_dia_pelo_ticker')
OUT.mkdir(parents=True, exist_ok=True)

parquet_files = list(SRC.rglob('*.parquet'))
if not parquet_files:
    print('Nenhum arquivo parquet encontrado em', SRC)
    raise SystemExit(2)

frames = []
for p in sorted(parquet_files):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print('Falha ao ler', p, '->', e)
        continue
    # normalizar nomes de coluna de timestamp
    if 'execution_timestamp' not in df.columns:
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp':'execution_timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date':'execution_timestamp'})
    if 'ticker' not in df.columns or 'execution_timestamp' not in df.columns:
        continue
    df = df[['ticker','execution_timestamp']].dropna()
    df['execution_timestamp'] = pd.to_datetime(df['execution_timestamp'], errors='coerce')
    df = df.dropna(subset=['execution_timestamp'])
    frames.append(df)

if not frames:
    print('Nenhuma frame utilizável encontrada (colunas `ticker` e `execution_timestamp` necessárias)')
    raise SystemExit(0)

all_df = pd.concat(frames, ignore_index=True)
all_df['date'] = all_df['execution_timestamp'].dt.date

# tabela agregada: linhas=data, colunas=tickers
per_day_ticker = all_df.groupby(['date','ticker']).size().unstack(fill_value=0)
# salvar CSV na pasta específica
csv_path = OUT / 'predicoes_por_dia_por_ticker.csv'
per_day_ticker.to_csv(csv_path)
print('Escrito CSV:', csv_path)

# gráfico empilhado por dia (top N tickers + OUTROS)
TOP_N = 8
top_tickers = per_day_ticker.sum().sort_values(ascending=False).head(TOP_N).index.tolist()
others = [c for c in per_day_ticker.columns if c not in top_tickers]
per_ticker_top = per_day_ticker[top_tickers].copy()
if others:
    per_ticker_top['OUTROS'] = per_day_ticker[others].sum(axis=1)
per_ticker_top = per_ticker_top.sort_index()

plt.figure(figsize=(12,5))
per_ticker_top.plot(kind='area', stacked=True, figsize=(12,5))
plt.title('Predições por dia por ticker (top %d + OUTROS)' % TOP_N)
plt.xlabel('Data')
plt.ylabel('Contagem de predições')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
png_path = OUT / 'predicoes_por_dia_por_ticker.png'
plt.savefig(png_path, bbox_inches='tight')
plt.close()
print('Escrito PNG:', png_path)
