from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def create_windows(
    df: pd.DataFrame,
    target_col: str,
    lookback: int,
    scaler: StandardScaler | MinMaxScaler,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_df = df.drop(columns=[target_col])
    target = df[target_col].to_numpy()

    if fit_scaler:
        scaler.fit(feature_df.values)

    scaled_features = scaler.transform(feature_df.values)

    X, y = [], []
    for idx in range(lookback, len(df)):
        X.append(scaled_features[idx - lookback:idx])
        y.append(target[idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
