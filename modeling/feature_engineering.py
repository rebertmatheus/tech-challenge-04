from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def create_features(
        self,
        df: Optional[pd.DataFrame] = None,
        is_training_data: bool = True,
        target_days: int = 1,
    ) -> Optional[pd.DataFrame]:
        # Validate input
        if df is None:
            self.logger.warning("DataFrame is None, skipping feature creation.")
            return None
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

        required_cols = ['Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        try:
            data = df.copy()

            if isinstance(data.columns, pd.MultiIndex):
                ticker = data.columns.get_level_values(1)[0]
                data = data.xs(ticker, axis=1, level=1)
                self.logger.info(f"MultiIndex detected, using ticker: {ticker}")

            has_ohlc = all(col in data.columns for col in ['Open', 'High', 'Low', 'Close'])
            has_open = 'Open' in data.columns

            price_col = 'Close'

            delta = data[price_col].diff().astype(float)

            gain_7 = delta.clip(lower=0).ewm(alpha=1/7, adjust=False).mean()
            loss_7 = (-delta.clip(upper=0)).ewm(alpha=1/7, adjust=False).mean()
            rs_7 = gain_7 / loss_7.replace(0, np.nan)
            data['rsi_7'] = 100 - (100 / (1 + rs_7))
            data['rsi_7'] = data['rsi_7'].fillna(50)

            gain_14 = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss_14 = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs_14 = gain_14 / loss_14.replace(0, np.nan)
            data['rsi_14'] = 100 - (100 / (1 + rs_14))
            data['rsi_14'] = data['rsi_14'].fillna(50)

            ema12 = data[price_col].ewm(span=12, adjust=False).mean()
            ema26 = data[price_col].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            data['macd_histogram'] = macd - macd_signal

            data['ma3'] = data[price_col].rolling(window=3).mean()
            data['ma5'] = data[price_col].rolling(window=5).mean()
            data['ma9'] = data[price_col].rolling(window=9).mean()

            data['distance_ma3'] = np.where(
                data['ma3'] != 0,
                (data[price_col] - data['ma3']) / data['ma3'] * 100,
                0,
            )
            data['distance_ma9'] = np.where(
                data['ma9'] != 0,
                (data[price_col] - data['ma9']) / data['ma9'] * 100,
                0,
            )

            data['return_1d'] = data[price_col].pct_change(periods=1) * 100
            data['return_3d'] = data[price_col].pct_change(periods=3) * 100
            data['roc_3'] = np.where(
                data[price_col].shift(3) != 0,
                (data[price_col] - data[price_col].shift(3)) / data[price_col].shift(3) * 100,
                0,
            )

            returns_pct = data[price_col].pct_change()
            data['volatility_5d'] = returns_pct.rolling(window=5).std() * 100
            volatility_10d = returns_pct.rolling(window=10).std() * 100
            data['volatility_ratio'] = np.where(
                volatility_10d != 0,
                data['volatility_5d'] / volatility_10d,
                1.0,
            )

            vol_ma_5 = data['Volume'].rolling(window=5).mean()
            vol_ma_20 = data['Volume'].rolling(window=20).mean()
            data['relative_volume'] = np.where(
                vol_ma_20 != 0,
                data['Volume'] / vol_ma_20,
                1.0,
            )
            data['volume_ratio_5'] = np.where(
                vol_ma_5 != 0,
                data['Volume'] / vol_ma_5,
                1.0,
            )

            bb_ma = data[price_col].rolling(window=20).mean()
            bb_std = data[price_col].rolling(window=20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)
            bb_range = bb_upper - bb_lower
            data['bb_position'] = np.where(
                bb_range != 0,
                (data[price_col] - bb_lower) / bb_range,
                0.5,
            )

            if has_ohlc:
                low_5 = data['Low'].rolling(window=5).min()
                high_5 = data['High'].rolling(window=5).max()
                data['stoch_k'] = np.where(
                    (high_5 - low_5) != 0,
                    (data['Close'] - low_5) / (high_5 - low_5) * 100,
                    50,
                )
                self.logger.info("Stochastic %K computed (OHLC available)")
            else:
                data['stoch_k'] = np.nan
                self.logger.info("Stochastic %K not computed - OHLC not available")

            if has_open:
                data['gap'] = np.where(
                    data['Close'].shift(1) != 0,
                    (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100,
                    0,
                )
                self.logger.info("Gap computed (Open available)")
            else:
                data['gap'] = 0.0
                self.logger.info("Gap set to 0 (Open not available)")

            if is_training_data:
                data['target'] = np.where(
                    data[price_col] != 0,
                    (data[price_col].shift(-target_days) / data[price_col] - 1) * 100,
                    0,
                )

                initial_len = len(data)
                data = data.dropna()
                removed = initial_len - len(data)
                self.logger.info(f"Target computed: {target_days}-day ahead; dropped {removed} rows")
            else:
                # Fill missing values for inference
                data = data.ffill().bfill()
                self.logger.info("Inference mode: NaNs forward/backward filled")
                if len(data) > 0:
                    try:
                        last_date = data.index[-1]
                    except Exception:
                        last_date = 'N/A'
                    self.logger.info(f"Last date in inference data: {last_date}")

            # Ensure DataFrame is returned
            if isinstance(data, pd.Series):
                self.logger.warning("Converting Series to DataFrame")
                data = data.to_frame()

            return data

        except Exception as exc:
            self.logger.exception(f"Error computing features: {exc}")
            raise
