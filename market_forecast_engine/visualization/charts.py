from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from market_forecast_engine.forecast.forecast_engine import ForecastOutput


class Charts:
    def plot_interactive(self, history: pd.DataFrame, features: pd.DataFrame, forecast: ForecastOutput, output_html: str) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history["Adj Close"], name="Historical Price", mode="lines"))
        fig.add_trace(go.Scatter(x=forecast.future_dates, y=forecast.price_forecast, name="Forecast", mode="lines"))
        fig.add_trace(go.Scatter(x=features.index, y=features["sma_20"], name="SMA 20", mode="lines", opacity=0.6))
        fig.add_trace(go.Scatter(x=features.index, y=features["sma_50"], name="SMA 50", mode="lines", opacity=0.6))
        fig.add_trace(go.Scatter(x=forecast.future_dates, y=forecast.upper_band, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=forecast.future_dates,
                y=forecast.lower_band,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(52, 152, 219, 0.2)",
                name="95% CI",
            )
        )

        for i in range(min(100, forecast.monte_carlo_paths.shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=forecast.future_dates,
                    y=forecast.monte_carlo_paths[i],
                    mode="lines",
                    line=dict(color="rgba(127,140,141,0.15)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(title="30-Trading-Day Forecast", xaxis_title="Date", yaxis_title="Price")
        fig.write_html(output_html)

    def plot_static(self, history: pd.DataFrame, features: pd.DataFrame, forecast: ForecastOutput, output_png: str) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

        axes[0].plot(history.index, history["Adj Close"], label="Historical")
        axes[0].plot(features.index, features["sma_20"], label="SMA20", alpha=0.8)
        axes[0].plot(features.index, features["sma_50"], label="SMA50", alpha=0.8)
        axes[0].plot(forecast.future_dates, forecast.price_forecast, label="Forecast", color="tab:orange")
        axes[0].fill_between(forecast.future_dates, forecast.lower_band, forecast.upper_band, alpha=0.2, label="95% CI")
        axes[0].set_title("Price Forecast + Bands")
        axes[0].legend()

        axes[1].plot(features.index, features["rsi_14"], label="RSI(14)")
        axes[1].axhline(70, linestyle="--", color="red", alpha=0.5)
        axes[1].axhline(30, linestyle="--", color="green", alpha=0.5)
        axes[1].set_title("RSI")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        plt.close(fig)
