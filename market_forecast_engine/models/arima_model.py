from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAForecastModel:
    name = "arima"

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 0, 1, 5)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model_fit = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False).fit(
            disp=False
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_fit is None:
            raise RuntimeError("Fit ARIMA model before prediction")
        return np.asarray(self.model_fit.forecast(steps=len(X)))
