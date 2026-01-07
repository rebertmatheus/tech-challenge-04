#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

SRC = Path('graficos/arquivos')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

parquet_files = list(SRC.rglob('*.parquet'))
if not parquet_files:
    print('No parquet files found under', SRC)
    raise SystemExit(2)

frames = []
for p in sorted(parquet_files):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print('Failed to read', p, '->', e)
        continue
    if 'ticker' not in df.columns or 'execution_timestamp' not in df.columns:
        # try alternative timestamp names
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp':'execution_timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date':'execution_timestamp'})
        else:
            continue
        if 'ticker' not in df.columns:
            continue
    # ensure timestamp
    df['execution_timestamp'] = pd.to_datetime(df['execution_timestamp'], errors='coerce')
    frames.append(df[['ticker','execution_timestamp']].dropna())

if not frames:
    print('No usable frames found')
    raise SystemExit(0)

all_df = pd.concat(frames, ignore_index=True)
all_df['date'] = all_df['execution_timestamp'].dt.date
per_day_ticker = all_df.groupby(['date','ticker']).size().unstack(fill_value=0)
csv_path = OUT / 'predictions_per_day_by_ticker.csv'
per_day_ticker.to_csv(csv_path)
print('Wrote', csv_path)
