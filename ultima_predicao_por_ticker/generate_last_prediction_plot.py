#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
from pandas.api.types import is_datetime64tz_dtype

ROOT = Path('.')
SRC = ROOT / 'graficos' / 'arquivos'
OUT = ROOT / 'graficos' / 'output'
OUT.mkdir(parents=True, exist_ok=True)

def collect_last_predictions(src: Path) -> pd.DataFrame:
    parquet_files = list(src.rglob('*.parquet'))
    if not parquet_files:
        raise SystemExit(f'No parquet files found under {src}')

    frames = []
    for p in sorted(parquet_files):
        try:
            # attempt to read minimal columns first
            sample = pd.read_parquet(p, engine='auto')
            cols = sample.columns.tolist()
            needed = [c for c in ['ticker', 'execution_timestamp', 'timestamp', 'date'] if c in cols]
            if needed:
                df = pd.read_parquet(p, columns=needed)
            else:
                df = sample
        except Exception:
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue

        # normalize column names
        if 'execution_timestamp' not in df.columns:
            for alt in ['timestamp', 'date']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'execution_timestamp'})
                    break

        if 'ticker' not in df.columns or 'execution_timestamp' not in df.columns:
            continue

        df = df[['ticker', 'execution_timestamp']].copy()
        frames.append(df)

    if not frames:
        raise SystemExit('No files with `ticker` and `execution_timestamp` found.')

    all_df = pd.concat(frames, ignore_index=True)
    all_df['execution_timestamp'] = pd.to_datetime(all_df['execution_timestamp'], errors='coerce')
    all_df = all_df.dropna(subset=['execution_timestamp', 'ticker'])

    last = all_df.groupby('ticker', as_index=False)['execution_timestamp'].max()
    return last


def plot_last_predictions(last: pd.DataFrame, out_png: Path, out_csv: Path, top_n: int = None):
    now = pd.Timestamp.now()
    last = last.copy()
    # ensure timestamps are timezone-naive for arithmetic
    last['execution_timestamp'] = pd.to_datetime(last['execution_timestamp'], errors='coerce')
    if is_datetime64tz_dtype(last['execution_timestamp']):
        last['execution_timestamp'] = last['execution_timestamp'].dt.tz_convert(None)
    # compute age in hours
    last['age_hours'] = (now - last['execution_timestamp']).dt.total_seconds() / 3600.0

    if top_n is not None and len(last) > top_n:
        # keep most recent top_n
        last = last.sort_values('execution_timestamp', ascending=False).head(top_n)

    # sort so most recent on top
    last = last.sort_values('execution_timestamp', ascending=True)

    # save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    last.to_csv(out_csv, index=False)

    # colors: map age -> color (more recent = darker)
    ages = last['age_hours'].values
    if len(ages) == 0:
        print('No data to plot')
        return
    # normalize where smaller age->larger value for colormap
    norm = mcolors.Normalize(vmin=ages.min(), vmax=ages.max())
    cmap = plt.get_cmap('viridis_r')
    colors = cmap(norm(ages))

    tickers = last['ticker'].astype(str).tolist()
    dates = last['execution_timestamp'].dt.to_pydatetime()
    y_pos = np.arange(len(tickers))

    figsize = (10, max(4, len(tickers) * 0.25))
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y_pos, mdates.date2num(dates), color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers)

    # x-axis as dates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    ax.set_xlabel('Last prediction timestamp')
    ax.set_title('Última predição por ticker')
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # annotate with relative age
    for i, (dt, age) in enumerate(zip(dates, ages)):
        ax.text(mdates.date2num(dt), i, f'  {pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M")} ({age:.1f}h)',
                va='center', fontsize=8, color='#222222')

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('Wrote:', out_csv, out_png)


if __name__ == '__main__':
    try:
        last = collect_last_predictions(SRC)
    except SystemExit as e:
        print(e)
        raise

    csv_path = OUT / 'ultima_predicao_por_ticker.csv'
    png_path = OUT / 'ultima_predicao_por_ticker.png'
    plot_last_predictions(last, png_path, csv_path, top_n=200)
