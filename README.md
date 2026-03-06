# Market Forecast Engine (Global, 30 Trading Days)

Production-ready, modular Python project for forecasting financial assets across global markets using only free/open data sources.

## Supported assets
- US equities: `AAPL`, `TSLA`
- India (NSE/BSE): `TCS.NS`, `RELIANCE.NS`
- Europe / Asia tickers supported by Yahoo suffixes
- ETFs: `SPY`, `QQQ`, `INDA`
- Crypto: `BTC-USD`, `ETH-USD`

## Data sources (free)
- Primary: `yfinance`
- Optional fallback hooks:
  - Alpha Vantage (`ALPHAVANTAGE_API_KEY`)
  - TwelveData (`TWELVEDATA_API_KEY`)

## Project structure
```text
market_forecast_engine/
  data/
    data_fetcher.py
  features/
    feature_engineering.py
  models/
    lstm_model.py
    xgboost_model.py
    random_forest_model.py
    prophet_model.py
    arima_model.py
  ensemble/
    ensemble_predictor.py
  forecast/
    forecast_engine.py
  visualization/
    charts.py
  backtest/
    backtest_engine.py
  simulation/
    monte_carlo.py
  utils/
    helpers.py
main.py
requirements.txt
README.md
```

## Implemented capabilities
- 5–10 years OHLCV retrieval with local parquet cache
- Market index + sector ETF + VIX + optional macro ingestion
- Technical features: RSI, MACD, Bollinger, SMA/EMA, ATR, momentum, OBV, volume spikes
- Market/time features: relative strength, sector correlation, volatility clustering, trend regime, calendar effects
- Models: LSTM, XGBoost, RandomForest, Prophet, ARIMA/SARIMA
- Inverse-RMSE weighted ensemble
- Monte Carlo simulation for next 30 trading days
- Outputs: forecast path, confidence bands, bullish/bearish probabilities, support/resistance, trend class
- Backtesting (last 3 years): RMSE, MAPE, directional accuracy, per-model comparison
- Visualization: Plotly interactive + Matplotlib static charts including technical overlays and Monte Carlo paths

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage
```bash
python main.py --ticker AAPL
python main.py --ticker TCS.NS --period 8y
python main.py --ticker BTC-USD --horizon 30 --skip-backtest
```

## Example output (abridged)
```json
{
  "ticker": "AAPL",
  "forecast_start": "2026-03-09",
  "forecast_end": "2026-04-17",
  "trend_classification": "Bullish",
  "bullish_probability": 0.61,
  "bearish_probability": 0.39,
  "support_levels": [170.2, 176.1],
  "resistance_levels": [203.4, 209.8],
  "backtest": {
    "random_forest": {"rmse": 8.5, "mape": 0.04, "directional_accuracy": 0.62},
    "arima": {"rmse": 9.1, "mape": 0.05, "directional_accuracy": 0.58}
  }
}
```

## Notes
- Optional heavy dependencies (TensorFlow, Prophet, XGBoost) are loaded opportunistically and skipped if unavailable.
- For best speed in CPU-only environments, use ARIMA + RandomForest + XGBoost.
