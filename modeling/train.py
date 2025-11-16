from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from modeling.dataio import read_parquet_from_blob, upload_local_file_to_blob, ensure_local_dir
from modeling.dataset import split_train_val_test, create_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM model for stock price prediction using TensorFlow or PyTorch Lightning."
    )
    # Default ticker set to ITUB4. You can override this via the command line.
    parser.add_argument(
        "--ticker", default="ITUB4", help="Stock ticker symbol (without .SA suffix)"
    )
    parser.add_argument(
        "--conn-str", required=False, default=None, help="Azure Blob Storage connection string"
    )
    parser.add_argument(
        "--container", default="techchallenge04storage", help="Blob container name"
    )
    parser.add_argument(
        "--blob-path",
        default=None,
        help="Path to parquet file inside the container (defaults to history/{ticker}.parquet)",
    )
    parser.add_argument(
        "--lookback", type=int, default=60, help="Number of past days to use in each input window"
    )
    parser.add_argument(
        "--output-dir", default="artifacts", help="Directory to save model artefacts"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload artefacts back to Blob Storage after training",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--framework",
        choices=["tensorflow", "pytorch"],
        default="pytorch",
        help=(
            "Deep learning framework to use: 'tensorflow' (default) uses Keras; "
            "'pytorch' uses PyTorch Lightning.  Ensure the necessary packages are installed "
            "before selecting 'pytorch'."
        ),
    )
    return parser.parse_args()


def load_dataset(ticker: str, conn_str: Optional[str], container: str, blob_path: Optional[str]) -> pd.DataFrame:
    resolved_blob_path = blob_path or f"history/{ticker}.parquet"
    if conn_str:
        return read_parquet_from_blob(conn_str, container, resolved_blob_path)
    else:
        if not os.path.isfile(resolved_blob_path):
            raise FileNotFoundError(
                f"Local parquet file {resolved_blob_path} not found. Provide --conn-str or place file locally.")
        return pd.read_parquet(resolved_blob_path)


def main() -> None:
    args = parse_args()

    ticker = args.ticker.upper()
    blob_path = args.blob_path or f"history/{ticker}.parquet"

    print(f"Loading dataset for {ticker}...")
    df = load_dataset(ticker, args.conn_str, args.container, blob_path)

    df = df.sort_index()

    drop_cols = [col for col in ['Date', 'execution_timestamp', 'ticker'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    if 'target' not in df.columns:
        raise ValueError("The dataset must contain a 'target' column. Ensure that feature engineering was run with is_training_data=True.")

    train_df, val_df, test_df = split_train_val_test(df, train_ratio=0.7, val_ratio=0.15, target_col='target')

    scaler = StandardScaler()
    lookback = args.lookback

    X_train, y_train = create_windows(train_df, target_col='target', lookback=lookback, scaler=scaler, fit_scaler=True)
    X_val, y_val = create_windows(val_df, target_col='target', lookback=lookback, scaler=scaler, fit_scaler=False)
    X_test, y_test = create_windows(test_df, target_col='target', lookback=lookback, scaler=scaler, fit_scaler=False)

    output_dir = os.path.join(
        args.output_dir, ticker, datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    )
    ensure_local_dir(output_dir)

    mae: float
    rmse: float
    mape: float

    if args.framework == 'tensorflow':
        # Importing TensorFlow only when needed to avoid unnecessary dependencies and heavyness
        import tensorflow as tf
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

        from modeling.model import build_lstm_model
        model_tf = build_lstm_model((lookback, X_train.shape[2]))

        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor='val_loss'
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, monitor='val_loss'
        )

        print("Starting training (TensorFlow)...")
        model_tf.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=2,
        )

        print("Evaluating on test set (TensorFlow)...")
        y_pred = model_tf.predict(X_test).reshape(-1)
        y_true = y_test.reshape(-1)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))

        # Saving model
        model_filename = 'model.h5'
        model_tf.save(os.path.join(output_dir, model_filename))

    else:
        try:
            import torch
            import pytorch_lightning as pl
        except ImportError as exc:
            raise ImportError(
                "PyTorch and PyTorch Lightning must be installed to use the 'pytorch' framework"
            ) from exc

        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
        from modeling.lstm_pl import TimeSeriesDataset, LSTMModel

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        model_pl = LSTMModel(input_size=X_train.shape[2])

        early_stop_cb = pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, mode='min', verbose=False
        )
        lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            callbacks=[early_stop_cb, lr_monitor_cb],
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=True,
        )

        print("Starting training (PyTorch Lightning)...")
        trainer.fit(model_pl, train_loader, val_loader)

        print("Evaluating on test set (PyTorch Lightning)...")
        model_pl.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                y_hat = model_pl(x_batch).detach().cpu().numpy()
                preds.append(y_hat)
                trues.append(y_batch.detach().cpu().numpy())
        y_pred = np.concatenate(preds, axis=0).reshape(-1)
        y_true = np.concatenate(trues, axis=0).reshape(-1)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))

        model_filename = 'model.pt'
        ensure_local_dir(output_dir)
        torch.save(model_pl.state_dict(), os.path.join(output_dir, model_filename))

    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    feature_spec = {
        'ticker': ticker,
        'lookback': lookback,
        'feature_columns': [col for col in train_df.columns if col != 'target'],
        'target_column': 'target',
    }
    with open(os.path.join(output_dir, 'feature_spec.json'), 'w') as f:
        json.dump(feature_spec, f, indent=2)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    artefact_files = [model_filename, 'scaler.pkl', 'feature_spec.json', 'metrics.json']
    if args.upload and args.conn_str:
        for filename in artefact_files:
            local_path = os.path.join(output_dir, filename)
            blob_dest = f"models/{ticker}/{os.path.basename(output_dir)}/{filename}"
            upload_local_file_to_blob(
                args.conn_str, args.container, local_path, blob_dest, overwrite=True
            )
        print(f"Uploaded artefacts to container '{args.container}'")

    print(f"Training finished. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    main()