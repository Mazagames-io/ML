from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
except Exception:
    tf = None


class LSTMForecastModel:
    name = "lstm"

    def __init__(self, lookback: int = 30, epochs: int = 15, batch_size: int = 32) -> None:
        if tf is None:
            raise ImportError("tensorflow is not installed")
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        xs, ys = [], []
        for i in range(self.lookback, len(X)):
            xs.append(X[i - self.lookback : i])
            if y is not None:
                ys.append(y[i])
        return (np.array(xs), np.array(ys)) if y is not None else np.array(xs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = self.scaler_X.fit_transform(X.values)
        ys = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        X_seq, y_seq = self._make_sequences(Xs, ys)

        self.model = Sequential(
            [
                Input(shape=(self.lookback, X.shape[1])),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit LSTM model before prediction")
        Xs = self.scaler_X.transform(X.values)
        if len(Xs) < self.lookback:
            pad = np.repeat(Xs[[0]], self.lookback - len(Xs), axis=0)
            Xs = np.vstack([pad, Xs])

        X_seq = self._make_sequences(Xs)
        yhat_scaled = self.model.predict(X_seq, verbose=0).flatten()
        yhat = self.scaler_y.inverse_transform(yhat_scaled.reshape(-1, 1)).flatten()
        if len(yhat) < len(X):
            yhat = np.hstack([np.repeat(yhat[0], len(X) - len(yhat)), yhat])
        return yhat
