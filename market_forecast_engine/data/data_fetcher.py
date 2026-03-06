from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from market_forecast_engine.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    market: str
    index_ticker: str
    sector_proxy: str


class DataFetcher:
    """Download and cache historical market data from free providers."""

    def __init__(self, cache_dir: str = "market_forecast_engine/cache") -> None:
        self.cache_dir = ensure_dir(cache_dir)

    @staticmethod
    def infer_market_context(ticker: str) -> MarketContext:
        t = ticker.upper()
        if t.endswith(".NS") or t.endswith(".BO"):
            return MarketContext("India", "^NSEI", "INDA")
        if t.endswith(".L") or t.endswith(".PA") or t.endswith(".DE"):
            return MarketContext("Europe", "^STOXX50E", "VGK")
        if t.endswith(".HK") or t.endswith(".T"):
            return MarketContext("Asia", "^N225", "AAXJ")
        if "-USD" in t:
            return MarketContext("Crypto", "BTC-USD", "BITO")
        return MarketContext("US", "^GSPC", "XLK")

    def _cache_path(self, ticker: str, period: str, interval: str) -> Path:
        safe = ticker.replace("^", "IDX_").replace("/", "_")
        return self.cache_dir / f"{safe}_{period}_{interval}.parquet"

    def fetch_ohlcv(self, ticker: str, period: str = "10y", interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_path(ticker, period, interval)
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.rename(columns={col: col.title() for col in df.columns})
        required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = df["Close"] if col == "Adj Close" else 0.0

        cleaned = df[required].copy()
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index().dropna(how="any")
        cleaned.to_parquet(cache_path)
        return cleaned

    def fetch_alpha_vantage_daily(self, ticker: str, api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
        key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            return None

        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={ticker}&outputsize=full&apikey={key}"
        )
        raw = pd.read_json(url)
        block_name = "Time Series (Daily)"
        if block_name not in raw:
            return None

        mapping = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        }
        df = pd.DataFrame(raw[block_name]).T
        df = df.rename(columns=mapping)[list(mapping.values())].astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def fetch_twelve_data_daily(self, ticker: str, api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
        # Lightweight optional hook; returns None when API key is absent.
        key = api_key or os.getenv("TWELVEDATA_API_KEY")
        if not key:
            return None
        try:
            import requests
        except Exception:
            return None

        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 5000,
            "apikey": key,
        }
        response = requests.get(url, params=params, timeout=20)
        payload = response.json()
        values = payload.get("values")
        if not values:
            return None

        df = pd.DataFrame(values)
        df = df.rename(
            columns={
                "datetime": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Adj Close"] = df["Close"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index().dropna()
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    def fetch_bundle(self, ticker: str, period: str = "10y") -> dict[str, pd.DataFrame]:
        context = self.infer_market_context(ticker)
        logger.info("Detected market context: %s", context)

        asset = self.fetch_ohlcv(ticker, period=period)
        index_df = self.fetch_ohlcv(context.index_ticker, period=period)
        sector = self.fetch_ohlcv(context.sector_proxy, period=period)

        try:
            vix = self.fetch_ohlcv("^VIX", period=period)
        except Exception:
            vix = pd.DataFrame()

        macro = {}
        for key, mticker in {"us10y": "^TNX", "dxy": "DX-Y.NYB"}.items():
            try:
                macro[key] = self.fetch_ohlcv(mticker, period=period)
            except Exception:
                macro[key] = pd.DataFrame()

        return {
            "asset": asset,
            "index": index_df,
            "sector": sector,
            "vix": vix,
            "macro": macro,
            "context": pd.DataFrame([context.__dict__]),
        }
