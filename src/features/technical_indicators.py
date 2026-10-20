"""Feature store: candle shape + normalized technical indicators.

v2 (2026-10-19, full-code audit at Kass's request):
- The v1 recipe fed the PPO raw price-scale values (vwap ~80,000, sma_50
  ~81,000, macd in the hundreds) next to tiny returns (~0.001) -- a scale
  ratio near 10^8 that an MLP cannot learn from. Every feature here is now
  small and roughly stationary: ratios, percents and 0-1 ranges.
- Candle/wick shape added at Kass's request: body, upper wick, lower wick,
  and full range, all as fractions of the open.
- volume_change is clipped: yfinance zero-volume candles used to produce
  absurd spikes (observed max +2,198,700%).

All features are causal (no lookahead beyond the row itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "body_pct",       # (close-open)/open -- candle body, signed
    "upper_wick",     # (high-max(open,close))/open
    "lower_wick",     # (min(open,close)-low)/open
    "range_pct",      # (high-low)/open
    "rsi_14n",        # RSI / 100 -> 0..1
    "macd_hist_rel",  # (macd - signal) / close
    "dist_vwap",      # (close - vwap) / close
    "dist_sma50",     # (close - sma_50) / close
    "atr_rel",        # atr / close
    "price_change",   # 1-candle close return
    "volatility_21",  # rolling std/mean of close
    "volume_change",  # volume pct change, clipped to [-1, 3]
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to an OHLCV frame. Returns a new frame."""
    df = df.copy()
    open_, high, low, close, volume = (
        df["open"], df["high"], df["low"], df["close"], df["volume"])

    # -- candle shape (fractions of the open) --
    top = pd.concat([open_, close], axis=1).max(axis=1)
    bot = pd.concat([open_, close], axis=1).min(axis=1)
    df["body_pct"] = (close - open_) / open_
    df["upper_wick"] = (high - top) / open_
    df["lower_wick"] = (bot - low) / open_
    df["range_pct"] = (high - low) / open_

    # -- normalized indicators --
    vwap = (close * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    sma_50 = close.rolling(50).mean()
    df["rsi_14n"] = _rsi(close, 14) / 100.0
    df["macd_hist_rel"] = (macd - signal) / close
    df["dist_vwap"] = (close - vwap) / close
    df["dist_sma50"] = (close - sma_50) / close
    df["atr_rel"] = _atr(df, 14) / close
    df["price_change"] = close.pct_change()
    roll = close.rolling(21)
    df["volatility_21"] = roll.std() / roll.mean()
    vol = volume.replace(0, np.nan)
    df["volume_change"] = vol.pct_change().fillna(0.0).clip(-1.0, 3.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stacker's target: 1 if the next candle closes higher.

    Drops rows with any NaN (warmup + last row).
    """
    df = add_features(df)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X, y ready for the stacker. Calling make_supervised first is required."""
    return df[FEATURE_COLUMNS].copy(), df["target"].astype(int).copy()
