from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from market_forecast_engine.ensemble.ensemble_predictor import EnsemblePredictor
from market_forecast_engine.models.arima_model import ARIMAForecastModel
from market_forecast_engine.models.lstm_model import LSTMForecastModel
from market_forecast_engine.models.prophet_model import ProphetForecastModel
from market_forecast_engine.models.random_forest_model import RandomForestForecastModel
from market_forecast_engine.models.xgboost_model import XGBoostForecastModel
from market_forecast_engine.simulation.monte_carlo import MonteCarloSimulator


@dataclass
class ForecastOutput:
    future_dates: pd.DatetimeIndex
    price_forecast: np.ndarray
    lower_band: np.ndarray
    upper_band: np.ndarray
    bullish_probability: float
    bearish_probability: float
    expected_volatility: float
    support_levels: list[float]
    resistance_levels: list[float]
    confidence_score: float
    trend_classification: str
    model_weights: dict[str, float]
    monte_carlo_paths: np.ndarray


class ForecastEngine:
    def __init__(self, horizon: int = 30):
        self.horizon = horizon

    @staticmethod
    def _trend_label(last: float, future: float) -> str:
        change = (future - last) / last
        if change > 0.08:
            return "Strong Bullish"
        if change > 0.02:
            return "Bullish"
        if change < -0.08:
            return "Strong Bearish"
        if change < -0.02:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _support_resistance(prices: pd.Series, lookback: int = 180) -> tuple[list[float], list[float]]:
        window = prices.tail(lookback)
        q = window.quantile([0.1, 0.2, 0.8, 0.9]).values
        return [round(q[0], 2), round(q[1], 2)], [round(q[2], 2), round(q[3], 2)]

    def _build_models(self) -> list:
        models = [RandomForestForecastModel(), ARIMAForecastModel()]
        for ctor in (XGBoostForecastModel, ProphetForecastModel, LSTMForecastModel):
            try:
                models.append(ctor())
            except Exception:
                continue
        return models

    def run(self, feature_df: pd.DataFrame) -> ForecastOutput:
        y = feature_df["target_30d"]
        X = feature_df.drop(columns=["target_30d"])

        split = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = self._build_models()
        val_preds: dict[str, np.ndarray] = {}
        rmse: dict[str, float] = {}

        for model in models:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if len(pred) != len(y_test):
                pred = np.resize(pred, len(y_test))
            val_preds[model.name] = pred
            rmse[model.name] = float(np.sqrt(mean_squared_error(y_test, pred)))

        ensemble_val = EnsemblePredictor().combine(val_preds, rmse)
        residual_std = float(np.std(y_test.values - ensemble_val.ensemble_prediction))

        future_dates = pd.bdate_range(feature_df.index[-1] + pd.Timedelta(days=1), periods=self.horizon)
        future_X = pd.DataFrame(
            np.repeat(X.iloc[[-1]].to_numpy(), self.horizon, axis=0),
            columns=X.columns,
            index=future_dates,
        )

        future_preds: dict[str, np.ndarray] = {}
        for model in models:
            model.fit(X, y)
            pred = model.predict(future_X)
            future_preds[model.name] = np.resize(pred, self.horizon)

        ensemble_future = EnsemblePredictor().combine(future_preds, rmse)
        forecast = ensemble_future.ensemble_prediction
        lower_band = forecast - 1.96 * residual_std
        upper_band = forecast + 1.96 * residual_std

        daily_returns = feature_df["returns"].dropna()
        annual_return = float(daily_returns.mean() * 252)
        annual_vol = float(daily_returns.std() * np.sqrt(252))
        last_price = float(feature_df["Adj Close"].iloc[-1])

        mc_paths = MonteCarloSimulator.simulate_paths(last_price, annual_return, annual_vol, horizon=self.horizon, n_paths=1500)
        bullish = float((mc_paths[:, -1] > last_price).mean())
        bearish = 1.0 - bullish

        support, resistance = self._support_resistance(feature_df["Adj Close"])
        conf = max(0.0, min(1.0, 1.0 / (1.0 + residual_std / max(last_price, 1e-6))))

        return ForecastOutput(
            future_dates=future_dates,
            price_forecast=forecast,
            lower_band=lower_band,
            upper_band=upper_band,
            bullish_probability=bullish,
            bearish_probability=bearish,
            expected_volatility=annual_vol,
            support_levels=support,
            resistance_levels=resistance,
            confidence_score=conf,
            trend_classification=self._trend_label(last_price, float(forecast[-1])),
            model_weights=ensemble_future.weights,
            monte_carlo_paths=mc_paths,
        )
