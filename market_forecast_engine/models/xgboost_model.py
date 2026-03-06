from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class XGBoostForecastModel:
    name = "xgboost"

    def __init__(self, random_state: int = 42) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        self.model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
