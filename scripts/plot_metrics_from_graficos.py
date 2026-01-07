"""Generate requested plots from Parquet files under graficos/arquivos.

Outputs (in graficos/output):
- versions_table.csv
- versions_table.html
- mae_evolution.png
- metrics_comparison.png
- directional_accuracy_heatmap.png
- total_predictions.png
- last_prediction_per_ticker.csv
- last_prediction_per_ticker.png

Usage:
  python scripts/plot_metrics_from_graficos.py

The script attempts to detect common column names for version, ticker, timestamp and metrics.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns_available = True
    sns.set(style='whitegrid')
except Exception:
    sns_available = False
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "graficos" / "arquivos"
OUT_DIR = ROOT / "graficos" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# discover parquet files
parquet_files = list(DATA_DIR.rglob("*.parquet"))
if not parquet_files:
    print("No parquet files found under", DATA_DIR)
    sys.exit(1)

# read files
dfs = []
for p in parquet_files:
    try:
        df = pd.read_parquet(p)
        df['_source_file'] = str(p.relative_to(ROOT))
        dfs.append(df)
    except Exception as e:
        print(f"Warning: failed to read {p}: {e}")

if not dfs:
    print("No dataframes loaded; exiting")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True, sort=False)
print("Loaded rows:", len(df))

# helper to find a column among candidates
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

version_col = find_col(["version","model_version","ver","v","model","model_name"]) or "version"
ticker_col = find_col(["ticker","symbol","code","asset"]) or "ticker"
mae_col = find_col(["mae","MAE","mean_absolute_error"])
mse_col = find_col(["mse","MSE","mean_squared_error"])
rmse_col = find_col(["rmse","RMSE","root_mean_squared_error"])
acc_col = find_col(["accuracy","acc","directional_accuracy","direction_accuracy","directional_acc"]) 

# timestamp column
ts_col = find_col(["timestamp","prediction_time","pred_time","datetime","date","ts"]) 

# normalize columns
if version_col not in df.columns:
    df['version'] = df.get('version', df.get('model', 'unknown'))
    version_col = 'version'

if ticker_col not in df.columns:
    df['ticker'] = df.get('ticker', df.get('symbol', None))
    ticker_col = 'ticker'

# ensure timestamp is datetime
if ts_col and ts_col in df.columns:
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception:
        pass

# Create aggregated versions table
agg_metrics = {}
metrics = []
for mcol,name in ((mae_col,'mae'),(mse_col,'mse'),(rmse_col,'rmse'),(acc_col,'accuracy')):
    if mcol:
        metrics.append((mcol,name))

if not metrics:
    print("No metric columns found among candidates. Will compute counts only.")

versions = df[version_col].fillna('unknown')
summary = df.groupby(version_col).agg({
    **({m[0]: 'mean' for m in metrics} if metrics else {}),
    '_source_file': 'count'
}).rename(columns={'_source_file': 'predictions_count'})

# rename metric columns
for mcol,name in metrics:
    if mcol in summary.columns:
        summary = summary.rename(columns={mcol: name})

summary = summary.reset_index().sort_values(by='predictions_count', ascending=False)
summary.to_csv(OUT_DIR / 'versions_table.csv', index=False)
summary.to_html(OUT_DIR / 'versions_table.html', index=False)
print('Wrote versions table to', OUT_DIR)


# MAE evolution by version (if mae present)
if 'mae' in summary.columns:
    plt.figure(figsize=(8,4))
    s = summary.sort_values(by=version_col)
    if sns_available:
        sns.lineplot(data=s, x=version_col, y='mae', marker='o')
    else:
        plt.plot(s[version_col].astype(str), s['mae'], marker='o')
    plt.xticks(rotation=45)
    plt.title('MAE médio por versão')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mae_evolution.png')
    plt.close()
    print('Saved mae_evolution.png')

# Comparison of metrics between versions
metric_cols = [c for c in ['mae','mse','rmse','accuracy'] if c in summary.columns]
if metric_cols:
    mdf = summary[[version_col]+metric_cols].melt(id_vars=[version_col], value_vars=metric_cols, var_name='metric', value_name='value')
    plt.figure(figsize=(10,5))
    if sns_available:
        sns.barplot(data=mdf, x=version_col, y='value', hue='metric')
    else:
        # simple grouped bar plot with matplotlib
        versions = mdf[version_col].unique().tolist()
        metrics_u = mdf['metric'].unique().tolist()
        x = np.arange(len(versions))
        width = 0.8 / len(metrics_u)
        for i, met in enumerate(metrics_u):
            vals = [mdf[(mdf[version_col]==v)&(mdf['metric']==met)]['value'].mean() for v in versions]
            plt.bar(x + i*width, vals, width=width, label=met)
        plt.xticks(x + width*(len(metrics_u)-1)/2, versions, rotation=45)
        plt.legend()
    plt.xticks(rotation=45)
    plt.title('Comparação de métricas por versão')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'metrics_comparison.png')
    plt.close()
    print('Saved metrics_comparison.png')

# Directional accuracy by version and ticker
if acc_col and acc_col in df.columns:
    acc_df = df[[version_col, ticker_col, acc_col]].dropna()
    # convert to numeric if possible
    try:
        acc_df[acc_col] = pd.to_numeric(acc_df[acc_col])
    except Exception:
        pass
    pivot = acc_df.groupby([version_col, ticker_col])[acc_col].mean().unstack(fill_value=np.nan)
    plt.figure(figsize=(10, max(4, pivot.shape[0]*0.4)))
    if sns_available:
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
    else:
        plt.imshow(pivot.fillna(0).values, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.yticks(np.arange(pivot.shape[0]), pivot.index)
        plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=45)
    plt.title('Acurácia direcional média por versão e ticker')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'directional_accuracy_heatmap.png')
    plt.close()
    print('Saved directional_accuracy_heatmap.png')

# Total predictions
total_by_version = df.groupby(version_col).size().reset_index(name='total_predictions').sort_values('total_predictions', ascending=False)
plt.figure(figsize=(8,4))
if sns_available:
    sns.barplot(data=total_by_version, x=version_col, y='total_predictions')
else:
    plt.bar(total_by_version[version_col].astype(str), total_by_version['total_predictions'])
plt.xticks(rotation=45)
plt.title('Total de predições por versão')
plt.tight_layout()
plt.savefig(OUT_DIR / 'total_predictions.png')
plt.close()
print('Saved total_predictions.png')

# Last prediction per ticker
if ts_col and ts_col in df.columns:
    last_preds = df.sort_values(ts_col).groupby(ticker_col).tail(1).set_index(ticker_col)
    cols = [c for c in [version_col, ts_col, 'prediction', 'predicted_value', 'y_pred'] if c in last_preds.columns]
    last_df = last_preds[[version_col, ts_col] + cols[2:]] if cols else last_preds[[version_col, ts_col]]
    last_df.reset_index().to_csv(OUT_DIR / 'last_prediction_per_ticker.csv', index=False)
    # simple bar showing recency
    try:
        last_df2 = last_df.reset_index()
        last_df2[ts_col] = pd.to_datetime(last_df2[ts_col])
        last_df2 = last_df2.sort_values(ts_col)
        plt.figure(figsize=(8, max(4, last_df2.shape[0]*0.3)))
        sns.barplot(data=last_df2, x=ts_col, y=ticker_col, hue=version_col)
        plt.title('Última predição por ticker')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'last_prediction_per_ticker.png')
        plt.close()
        print('Saved last_prediction_per_ticker.png')
    except Exception as e:
        print('Could not render last_prediction_per_ticker.png:', e)
else:
    print('No timestamp column found; cannot compute last prediction per ticker')

print('All done. Outputs in', OUT_DIR)
