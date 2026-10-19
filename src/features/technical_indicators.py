"""Feature store: technical indicators + price features.

This is the Project-Bot feature recipe (Bot1.ipynb), cleaned up and made
reusable: vwap, macd/signal_line, rsi, atr, sma_50, price_change, volatility,
volume_change. The stacker and the PPO state both build on these.

`add_features` is pure: input OHLCV in, features out, no lookahead beyond the
row itself (all indicators are causal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "vwap",
    "macd",
    "signal_line",
    "rsi_14",
    "atr_14",
    "sma_50",
    "price_change",
    "volatility_21",
    "volume_change",
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
    """Add indicator columns to an OHLCV frame. Returns a new frame."""
    df = df.copy()
    close, volume = df["close"], df["volume"]

    df["vwap"] = (close * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi_14"] = _rsi(close, 14)
    df["atr_14"] = _atr(df, 14)
    df["sma_50"] = close.rolling(50).mean()
    df["price_change"] = close.pct_change()
    roll = close.rolling(21)
    df["volatility_21"] = roll.std() / roll.mean()
    # yfinance leaves volume at 0 for many crypto intervals; treat "no
    # measurable volume" as 0 change instead of dropping the row.
    vol = volume.replace(0, np.nan)
    df["volume_change"] = vol.pct_change().fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stacker's target: 1 if the next candle closes higher.

    Same labeling as Project-Bot. Drops rows with any NaN (warmup + last row).
    """
    df = add_features(df)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X, y ready for the stacker. Calling make_supervised first is required."""
    return df[FEATURE_COLUMNS].copy(), df["target"].astype(int).copy()
