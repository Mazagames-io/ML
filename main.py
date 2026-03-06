from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from market_forecast_engine.backtest.backtest_engine import BacktestEngine
from market_forecast_engine.data.data_fetcher import DataFetcher
from market_forecast_engine.features.feature_engineering import FeatureEngineering
from market_forecast_engine.forecast.forecast_engine import ForecastEngine
from market_forecast_engine.utils.helpers import setup_logging
from market_forecast_engine.visualization.charts import Charts


def run_pipeline(ticker: str, period: str, horizon: int, skip_backtest: bool) -> dict:
    fetcher = DataFetcher()
    bundle = fetcher.fetch_bundle(ticker=ticker, period=period)

    features = FeatureEngineering().transform(
        asset_df=bundle["asset"],
        index_df=bundle["index"],
        sector_df=bundle["sector"],
        vix_df=bundle["vix"],
    )

    forecast_output = ForecastEngine(horizon=horizon).run(features)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{ticker}_forecast.html"
    png_path = out_dir / f"{ticker}_forecast.png"
    Charts().plot_interactive(bundle["asset"], features, forecast_output, str(html_path))
    Charts().plot_static(bundle["asset"], features, forecast_output, str(png_path))

    backtest_results = None
    if not skip_backtest:
        bt = BacktestEngine().run(features)
        backtest_results = {name: asdict(metrics) for name, metrics in bt.items()}

    payload = {
        "ticker": ticker,
        "forecast_start": str(forecast_output.future_dates[0].date()),
        "forecast_end": str(forecast_output.future_dates[-1].date()),
        "last_forecast_price": float(forecast_output.price_forecast[-1]),
        "trend_classification": forecast_output.trend_classification,
        "bullish_probability": forecast_output.bullish_probability,
        "bearish_probability": forecast_output.bearish_probability,
        "expected_volatility_annualized": forecast_output.expected_volatility,
        "support_levels": forecast_output.support_levels,
        "resistance_levels": forecast_output.resistance_levels,
        "confidence_score": forecast_output.confidence_score,
        "model_weights": forecast_output.model_weights,
        "outputs": {
            "interactive_chart": str(html_path),
            "static_chart": str(png_path),
        },
        "backtest": backtest_results,
    }
    return payload


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Global 30-trading-day market forecast engine")
    parser.add_argument("--ticker", required=True, help="e.g., AAPL, TCS.NS, BTC-USD")
    parser.add_argument("--period", default="10y", help="Historical period to fetch (default: 10y)")
    parser.add_argument("--horizon", default=30, type=int, help="Forecast horizon in trading days")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtesting step")
    args = parser.parse_args()

    result = run_pipeline(args.ticker, args.period, args.horizon, args.skip_backtest)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
