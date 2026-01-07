#!/usr/bin/env python3
"""Regenerate price_series CSV and PNG for given tickers using
functions from scripts/price_series_bbas3.py (collect_price_series, plot_price_series).
"""
import runpy
from pathlib import Path

g = runpy.run_path("scripts/price_series_bbas3.py")
collect = g['collect_price_series']
plot = g['plot_price_series']
OUT_DIR = g['OUT_DIR']

for ticker in ('PETR4', 'VALE3'):
    print('Processing', ticker)
    df = collect(ticker)
    out_csv = OUT_DIR / f'price_series_{ticker.lower()}.csv'
    out_png = OUT_DIR / f'price_series_{ticker.lower()}.png'
    if df.empty:
        print('No data found for', ticker)
    else:
        df.to_csv(out_csv, index=False)
        plot(df, ticker, out_png)
        print('Wrote', out_csv, out_png)
