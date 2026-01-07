#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        # try to read only relevant columns to save memory
        cols_to_try = ['ticker','execution_timestamp','timestamp','date','pred','prediction','y_pred']
        df = pd.read_parquet(p, columns=[c for c in cols_to_try if c in pd.read_parquet(p, engine='auto').columns])
    except Exception:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print('Failed to read', p, '->', e)
            continue

    # normalize column names
    if 'execution_timestamp' not in df.columns:
        for alt in ['timestamp','date']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'execution_timestamp'})
                break
    if 'ticker' not in df.columns:
        # try uppercase ticker
        for col in df.columns:
            if isinstance(col, str) and col.isupper() and len(col) <= 6:
                # this is unsafe; skip
                pass
    if 'ticker' not in df.columns or 'execution_timestamp' not in df.columns:
        # skip this file
        continue

    df = df[['ticker','execution_timestamp']].copy()
    frames.append(df)

if not frames:
    print('No files with `ticker` and `execution_timestamp` found.')
    raise SystemExit(0)

all_df = pd.concat(frames, ignore_index=True)
# convert timestamps
all_df['execution_timestamp'] = pd.to_datetime(all_df['execution_timestamp'], errors='coerce')
all_df = all_df.dropna(subset=['execution_timestamp','ticker'])

# last timestamp per ticker
last = all_df.groupby('ticker', as_index=False)['execution_timestamp'].max()
last = last.sort_values('execution_timestamp', ascending=True)
# save CSV
csv_path = OUT / 'last_prediction_per_ticker.csv'
last.to_csv(csv_path, index=False)

# plot: scatter with date x axis
plt.figure(figsize=(10, max(4, len(last)*0.25)))
ys = range(len(last))
xd = mdates.date2num(last['execution_timestamp'].dt.to_pydatetime())
plt.scatter(xd, ys, c='C0')
plt.yticks(ys, last['ticker'])
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.xlabel('Last prediction timestamp')
plt.title('Last prediction per ticker')
plt.tight_layout()

png = OUT / 'last_prediction_per_ticker.png'
plt.savefig(png)
print('Wrote:', csv_path, png)
