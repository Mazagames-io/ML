from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from market_forecast_engine.models.arima_model import ARIMAForecastModel
from market_forecast_engine.models.random_forest_model import RandomForestForecastModel
from market_forecast_engine.models.xgboost_model import XGBoostForecastModel


@dataclass
class ModelBacktestMetrics:
    rmse: float
    mape: float
    directional_accuracy: float


class BacktestEngine:
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, baseline_price: np.ndarray) -> ModelBacktestMetrics:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        true_dir = np.sign(y_true - baseline_price)
        pred_dir = np.sign(y_pred - baseline_price)
        da = float((true_dir == pred_dir).mean())
        return ModelBacktestMetrics(rmse=rmse, mape=mape, directional_accuracy=da)

    def run(self, feature_df: pd.DataFrame) -> dict[str, ModelBacktestMetrics]:
        recent = feature_df.last("3Y") if len(feature_df) > 780 else feature_df.copy()
        y = recent["target_30d"]
        X = recent.drop(columns=["target_30d"])

        split = int(len(recent) * 0.75)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        baseline = recent.loc[y_test.index, "Adj Close"].to_numpy()

        models = {"random_forest": RandomForestForecastModel(), "arima": ARIMAForecastModel()}
        try:
            models["xgboost"] = XGBoostForecastModel()
        except Exception:
            pass

        results: dict[str, ModelBacktestMetrics] = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = np.resize(model.predict(X_test), len(y_test))
            results[name] = self._evaluate(y_test.to_numpy(), pred, baseline)

        return results
