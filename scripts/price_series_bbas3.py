#!/usr/bin/env python3
"""Gerar série de preços para BBAS3 procurando em pastas comuns de dados."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
SEARCH_DIRS = [
    ROOT / 'graficos' / 'arquivos',
    ROOT / 'data',
    ROOT / 'data' / 'train',
    ROOT / 'data-service',
    ROOT / 'stock-service'
]
OUT_DIR = ROOT / 'graficos' / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_parquet_files():
    files = []
    for d in SEARCH_DIRS:
        if d.exists():
            files += list(d.rglob('*.parquet'))
    # also search entire repo as fallback (but keep reasonable limit)
    if not files:
        files = list(ROOT.rglob('*.parquet'))[:1000]
    return sorted(set(files))


def collect_price_series(ticker: str) -> pd.DataFrame:
    parquet_files = find_parquet_files()
    if not parquet_files:
        print('No parquet files found in search dirs')
        return pd.DataFrame()

    frames = []
    for p in parquet_files:
        try:
            sample = pd.read_parquet(p, engine='auto')
        except Exception:
            try:
                sample = pd.read_parquet(p)
            except Exception:
                continue

        cols = sample.columns.tolist()
        if 'ticker' not in cols and 'TICKER' not in cols:
            # if file named with ticker, quick include
            if p.name.lower().startswith(ticker.lower()):
                try:
                    df = pd.read_parquet(p)
                except Exception:
                    continue
            else:
                continue
        else:
            # read only needed cols if available
            want = [c for c in ['ticker', 'timestamp', 'date', 'execution_timestamp', 'close', 'price'] if c in cols]
            try:
                if want:
                    df = pd.read_parquet(p, columns=want)
                else:
                    df = sample
            except Exception:
                df = sample

        # normalize
        for ts in ['execution_timestamp', 'timestamp', 'date']:
            if ts in df.columns:
                df = df.rename(columns={ts: 'timestamp'})
                break
        for pr in ['close', 'price']:
            if pr in df.columns:
                df = df.rename(columns={pr: 'close'})
                break
        if 'ticker' not in df.columns and 'TICKER' in df.columns:
            df = df.rename(columns={'TICKER': 'ticker'})

        if 'ticker' not in df.columns:
            # possibly file contains only one ticker
            if p.name.lower().startswith(ticker.lower()):
                # try to infer prices in column named after ticker
                possible_price_cols = [c for c in df.columns if c.lower() in ('close', 'price', ticker.lower())]
                if possible_price_cols:
                    df = df.rename(columns={possible_price_cols[0]: 'close'})
                    df['ticker'] = ticker.upper()
                else:
                    continue
            else:
                continue

        # filter by ticker
        try:
            sel = df[df['ticker'].astype(str).str.upper() == ticker.upper()]
        except Exception:
            continue
        if sel.empty:
            continue
        # ensure timestamp and close exist
        if 'timestamp' not in sel.columns or 'close' not in sel.columns:
            continue
        sel = sel[['timestamp', 'close']].copy()
        frames.append(sel)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], errors='coerce')
    all_df = all_df.dropna(subset=['timestamp', 'close'])
    all_df = all_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    return all_df


def plot_price_series(df: pd.DataFrame, ticker: str, out_png: Path):
    if df.empty:
        print('Empty dataframe, nothing to plot')
        return
    plt.figure(figsize=(12,4))
    plt.plot(df['timestamp'], df['close'], label=ticker.upper())
    plt.fill_between(df['timestamp'], df['close'], alpha=0.1)
    plt.xlabel('Timestamp')
    plt.ylabel('Close')
    plt.title(f'Price series for {ticker.upper()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print('Wrote', out_png)


if __name__ == '__main__':
    ticker = 'BBAS3'
    df = collect_price_series(ticker)
    out_png = OUT_DIR / f'price_series_{ticker.lower()}.png'
    out_csv = OUT_DIR / f'price_series_{ticker.lower()}.csv'
    if df.empty:
        print('No price series found for', ticker)
    else:
        df.to_csv(out_csv, index=False)
        plot_price_series(df, ticker, out_png)
