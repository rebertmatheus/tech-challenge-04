from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

from modeling.dataio import read_parquet_from_blob
from modeling.dataset import create_windows

from feature_engineering import FeatureEngineer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference using a trained LSTM model. The model can be a Keras/" +
            "TensorFlow model (.h5) or a PyTorch Lightning model (.pt), depending on the " +
            "training framework used."
        )
    )
    parser.add_argument(
        "--ticker", default="ITUB4", help="Stock ticker symbol (without .SA suffix)"
    )
    parser.add_argument(
        "--conn-str", required=True, help="Azure Blob Storage connection string"
    )
    parser.add_argument(
        "--container", default="techchallenge04storage", help="Blob container name"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Directory containing the saved model (model.h5 or model.pt), scaler.pkl and feature_spec.json"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of past days to fetch for inference (should be >= lookback)",
    )
    parser.add_argument(
        "--framework",
        choices=["tensorflow", "pytorch"],
        default="tensorflow",
        help=(
            "Specify the framework used during training: 'tensorflow' expects a model.h5 Keras file; "
            "'pytorch' expects a model.pt file saved via torch.save(state_dict)."
        ),
    )
    return parser.parse_args()


def find_latest_daily_blob(container_client, ticker: str) -> str:
    from azure.storage.blob import BlobServiceClient

    def list_blobs(prefix: str):
        return container_client.list_blobs(name_starts_with=prefix)

    today = datetime.now()
    for offset in range(0, 30):
        dt = today - timedelta(days=offset)
        prefix = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
        for blob in list_blobs(prefix):
            if blob.name.endswith(f"/{ticker}.parquet"):
                return blob.name
    raise FileNotFoundError(f"No daily data found for {ticker} in the last 30 days")


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()

    if args.framework == 'tensorflow':
        model_filename = 'model.h5'
    else:
        model_filename = 'model.pt'
    model_path = os.path.join(args.model_dir, model_filename)
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    spec_path = os.path.join(args.model_dir, 'feature_spec.json')

    scaler = joblib.load(scaler_path)
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    lookback = spec['lookback']
    feature_cols = spec['feature_columns']
    target_col = spec['target_column']

    if args.framework == 'tensorflow':
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TensorFlow must be installed to load a Keras model. Install tensorflow and retry."
            ) from exc
        model = tf.keras.models.load_model(model_path)
    else:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyTorch must be installed to load a PyTorch model. Install torch and retry."
            ) from exc
        from modeling.lstm_pl import LSTMModel
        input_size = len(feature_cols)
        model = LSTMModel(input_size=input_size)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

    from azure.storage.blob import BlobServiceClient
    container_client = BlobServiceClient.from_connection_string(args.conn_str).get_container_client(args.container)
    blob_path = find_latest_daily_blob(container_client, ticker)
    df = read_parquet_from_blob(args.conn_str, args.container, blob_path)

    engineer = FeatureEngineer()
    if not all(col in df.columns for col in feature_cols):
        df_feats = engineer.create_features(df, is_training_data=False)
    else:
        df_feats = df.copy()

    df_feats = df_feats.sort_index()
    df_feats = df_feats.tail(lookback)

    missing = [c for c in feature_cols if c not in df_feats.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in daily data: {missing}")

    df_window = df_feats[feature_cols]

    scaled = scaler.transform(df_window.values)
    X = scaled.reshape(1, lookback, len(feature_cols)).astype(np.float32)

    if args.framework == 'tensorflow':
        pred_target = float(model.predict(X)[0][0])
    else:
        import torch  # type: ignore
        x_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred_tensor = model(x_tensor)
        pred_target = float(pred_tensor.detach().cpu().numpy().reshape(-1)[0])

    latest_close = float(df_feats['Close'].iloc[-1]) if 'Close' in df_feats.columns else None
    if latest_close:
        pred_close = latest_close * (1 + pred_target / 100)
        print(f"Predicted next-day close for {ticker}: {pred_close:.2f} BRL (target {pred_target:.2f}% change)")
    else:
        print(f"Predicted next-day target for {ticker}: {pred_target:.2f}% change")


if __name__ == "__main__":
    main()