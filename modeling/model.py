from __future__ import annotations

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    lookback, n_features = input_shape
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='mae',
                  metrics=[
                      'mse',
                      'mae',
                  ])
    return model
