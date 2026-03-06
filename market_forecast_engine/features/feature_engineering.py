from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Compute technical, market, and calendar features for supervised forecasting."""

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        return (np.sign(close.diff().fillna(0)) * volume).cumsum()

    def transform(
        self,
        asset_df: pd.DataFrame,
        index_df: pd.DataFrame,
        sector_df: pd.DataFrame,
        vix_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = asset_df.copy()

        df["returns"] = df["Adj Close"].pct_change()
        df["log_returns"] = np.log1p(df["returns"])

        for w in [20, 50, 200]:
            df[f"sma_{w}"] = df["Adj Close"].rolling(w).mean()
        for w in [12, 20, 50, 200]:
            df[f"ema_{w}"] = df["Adj Close"].ewm(span=w, adjust=False).mean()

        df["rsi_14"] = self._rsi(df["Adj Close"], 14)
        df["atr_14"] = self._atr(df, 14)
        df["obv"] = self._obv(df["Adj Close"], df["Volume"])
        df["momentum_5"] = df["Adj Close"].pct_change(5)
        df["momentum_20"] = df["Adj Close"].pct_change(20)

        ema12 = df["ema_12"]
        ema26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        bb_mid = df["Adj Close"].rolling(20).mean()
        bb_std = df["Adj Close"].rolling(20).std()
        df["bb_mid"] = bb_mid
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

        vol_mean = df["Volume"].rolling(20).mean()
        vol_std = df["Volume"].rolling(20).std()
        df["volume_spike"] = ((df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)).fillna(0)

        idx_ret = index_df["Adj Close"].pct_change().rename("idx_returns")
        sec_ret = sector_df["Adj Close"].pct_change().rename("sec_returns")
        df = df.join(idx_ret, how="left").join(sec_ret, how="left")
        df["rel_strength_idx_20"] = (1 + df["returns"]).rolling(20).apply(np.prod, raw=True) / (
            (1 + df["idx_returns"]).rolling(20).apply(np.prod, raw=True)
        )
        df["corr_sector_60"] = df["returns"].rolling(60).corr(df["sec_returns"])
        df["volatility_20"] = df["returns"].rolling(20).std()
        df["vol_cluster_10"] = df["volatility_20"].rolling(10).mean()
        df["trend_regime"] = np.where(df["sma_50"] >= df["sma_200"], 1, -1)

        if vix_df is not None and not vix_df.empty:
            df = df.join(vix_df[["Adj Close"]].rename(columns={"Adj Close": "vix_close"}), how="left")
            df["vix_close"] = df["vix_close"].ffill()

        df["dow"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)

        df["target_30d"] = df["Adj Close"].shift(-30)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df
