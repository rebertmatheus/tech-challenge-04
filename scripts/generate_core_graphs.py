#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# try seaborn but fall back gracefully
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
import numpy as np

SRC = Path('graficos/arquivos')
OUT = Path('graficos/output')
OUT.mkdir(parents=True, exist_ok=True)

parquet_files = list(SRC.rglob('*.parquet'))
if not parquet_files:
    print('No parquet files found under', SRC)
    raise SystemExit(2)

cols_of_interest = ['ticker','execution_timestamp','Close','Adj Close','return_1d','volatility_5d',
                    'rsi_14','ma3','ma9','relative_volume','volume_ratio_5','bb_position','stoch_k','gap','Volume']

frames = []
for p in sorted(parquet_files):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print('Failed to read', p, '->', e)
        continue
    present = [c for c in cols_of_interest if c in df.columns]
    if not present:
        continue
    df = df[present].copy()
    # normalize timestamp
    if 'execution_timestamp' in df.columns:
        df['execution_timestamp'] = pd.to_datetime(df['execution_timestamp'], errors='coerce')
    frames.append(df)

if not frames:
    print('No data frames with needed columns found')
    raise SystemExit(0)

data = pd.concat(frames, ignore_index=True)
# drop rows without ticker or timestamp
if 'ticker' in data.columns:
    data = data.dropna(subset=['ticker'])
else:
    print('No ticker column found; aborting')
    raise SystemExit(0)

# ----------------- Determine top tickers -----------------
ticker_counts = data['ticker'].value_counts()
top_tickers = ticker_counts.head(6).index.tolist()

# 1) Price time series by ticker (top 6)
for t in top_tickers:
    sub = data[data['ticker']==t].dropna(subset=['execution_timestamp'])
    if 'Close' in sub.columns:
        sub = sub.sort_values('execution_timestamp')
        plt.figure(figsize=(10,3))
        plt.plot(sub['execution_timestamp'], sub['Close'], label='Close')
        if 'Adj Close' in sub.columns:
            plt.plot(sub['execution_timestamp'], sub['Adj Close'], label='Adj Close', alpha=0.7)
        plt.title(f'Price series: {t}')
        plt.xlabel('time')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f'price_series_{t}.png')
        plt.close()

# 2) Return daily distribution: global + per top tickers
if 'return_1d' in data.columns:
    plt.figure(figsize=(8,4))
    if HAS_SEABORN:
        sns.histplot(data['return_1d'].dropna(), bins=80, kde=True)
    else:
        plt.hist(data['return_1d'].dropna(), bins=80)
    plt.title('Distribution of return_1d (all tickers)')
    plt.tight_layout()
    plt.savefig(OUT / 'return_1d_distribution.png')
    plt.close()

    # boxplot per top tickers
    plt.figure(figsize=(max(6,len(top_tickers)*1.2),4))
    if HAS_SEABORN:
        sns.boxplot(x='ticker', y='return_1d', data=data[data['ticker'].isin(top_tickers)])
    else:
        pd.DataFrame({t: data.loc[data['ticker']==t,'return_1d'].dropna() for t in top_tickers}).boxplot(rot=45)
    plt.title('return_1d by ticker (top)')
    plt.tight_layout()
    plt.savefig(OUT / 'return_1d_boxplot_top_tickers.png')
    plt.close()

# 3) Volatility over time for top tickers
if 'volatility_5d' in data.columns:
    for t in top_tickers:
        sub = data[data['ticker']==t].dropna(subset=['execution_timestamp','volatility_5d'])
        if sub.empty:
            continue
        sub = sub.sort_values('execution_timestamp')
        plt.figure(figsize=(10,3))
        plt.plot(sub['execution_timestamp'], sub['volatility_5d'])
        plt.title(f'Volatility 5d: {t}')
        plt.xlabel('time')
        plt.ylabel('volatility_5d')
        plt.tight_layout()
        plt.savefig(OUT / f'volatility_5d_{t}.png')
        plt.close()

# 6) Correlation heatmap of numeric features
numeric_cols = [c for c in ['Close','Adj Close','return_1d','volatility_5d','rsi_14','ma3','ma9','relative_volume','volume_ratio_5','bb_position','stoch_k','gap','Volume'] if c in data.columns]
if numeric_cols:
    corr = data[numeric_cols].corr()
    plt.figure(figsize=(max(6,len(numeric_cols)*0.6), max(4,len(numeric_cols)*0.6)))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    else:
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
        plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Feature correlation')
    plt.tight_layout()
    plt.savefig(OUT / 'feature_correlation_heatmap.png')
    plt.close()

# 10) Predictions per day (using execution_timestamp)
if 'execution_timestamp' in data.columns:
    data['date'] = pd.to_datetime(data['execution_timestamp']).dt.date
    counts = data.groupby('date').size()
    plt.figure(figsize=(10,3))
    counts.plot()
    plt.title('Predictions per day (total)')
    plt.xlabel('date')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(OUT / 'predictions_per_day.png')
    plt.close()

    # per-ticker stacked/area plot (top tickers + OTHER)
    per_ticker = data.groupby(['date', 'ticker']).size().unstack(fill_value=0)
    # choose top N tickers by total count
    TOP_N = 8
    top_tickers = per_ticker.sum().sort_values(ascending=False).head(TOP_N).index.tolist()
    others = [c for c in per_ticker.columns if c not in top_tickers]
    per_ticker_top = per_ticker[top_tickers].copy()
    if others:
        per_ticker_top['OTHER'] = per_ticker[others].sum(axis=1)
    per_ticker_top = per_ticker_top.sort_index()

    plt.figure(figsize=(12,4))
    # use area plot for stacked view
    per_ticker_top.plot(kind='area', stacked=True, figsize=(12,4))
    plt.title(f'Predictions per day by ticker (top {TOP_N} + OTHER)')
    plt.xlabel('date')
    plt.ylabel('count')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(OUT / 'predictions_per_day_by_ticker.png', bbox_inches='tight')
    plt.close()

print('Wrote core graphs to', OUT)
