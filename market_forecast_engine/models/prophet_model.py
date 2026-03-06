from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception:
    Prophet = None


class ProphetForecastModel:
    name = "prophet"

    def __init__(self) -> None:
        if Prophet is None:
            raise ImportError("prophet is not installed")
        self.model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        train_df = pd.DataFrame({"ds": X.index, "y": y.values})
        self.model.fit(train_df)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(pd.DataFrame({"ds": X.index}))
        return pred["yhat"].values
