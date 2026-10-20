"""Feature store: candle shape + normalized technical indicators.

v3 (2026-10-20, Kass's explicit request): STRICT [-1, +1] contract on every
value the brain sees. Kass remembered normalizing indicators to -1..1 from
earlier ML work; v2 was only loosely normalized (RSI sat in 0..1 uncentered,
volume_change clipped to [-1, 3], price-relative features unbounded), so v3
enforces the contract mechanically:

- rsi_c = (RSI - 50) / 50 -- centered at 0, in [-1, 1] by construction
  (renamed from v2's rsi_14n = RSI / 100).
- volume_change clipped symmetric to [-1, 1].
- Every price-relative feature is divided by a per-feature scale constant,
  then hard-clipped to [-1, 1]. The scales are the 99.5th percentile of
  |raw value| over 17,478 hourly BTC-USD candles (2024-10-21..2026-10-20,
  yfinance 1h), derived by scripts/derive_feature_scales.py -- so ~99.5%
  of history lands strictly inside the rails and only genuine extremes
  saturate. Re-derive if the timeframe or universe changes.
  Note: dist_vwap's scale is large because VWAP here is cumulative over
  the fetched window; on multi-year windows it drifts far from price.

History: v1 fed the PPO raw price-scale values (vwap ~80,000, macd in the
hundreds) next to tiny returns (~0.001) -- a scale ratio near 10^8 that an
MLP cannot learn from. v2 (2026-10-19 audit) made every feature small and
roughly stationary and added candle/wick shape. v3 keeps the v2 recipes
and adds the strict contract on top.

All features are causal (no lookahead beyond the row itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "body_pct",       # (close-open)/open, scaled -> [-1, 1]
    "upper_wick",     # (high-max(open,close))/open, scaled -> [0, 1]
    "lower_wick",     # (min(open,close)-low)/open, scaled -> [0, 1]
    "range_pct",      # (high-low)/open, scaled -> [0, 1]
    "rsi_c",          # (RSI-50)/50 -> [-1, 1], centered (was rsi_14n in v2)
    "macd_hist_rel",  # (macd - signal) / close, scaled -> [-1, 1]
    "dist_vwap",      # (close - vwap) / close, scaled -> [-1, 1]
    "dist_sma50",     # (close - sma_50) / close, scaled -> [-1, 1]
    "atr_rel",        # atr / close, scaled -> [0, 1]
    "price_change",   # 1-candle close return, scaled -> [-1, 1]
    "volatility_21",  # rolling std/mean of close, scaled -> [0, 1]
    "volume_change",  # volume pct change, clipped to [-1, 1]
]

# Per-feature scale constants: 99.5th percentile of |raw value| on ~2y of
# 1h BTC-USD (see module docstring / scripts/derive_feature_scales.py).
FEATURE_SCALES = {
    "body_pct": 0.021527,
    "upper_wick": 0.008957,
    "lower_wick": 0.010104,
    "range_pct": 0.027746,
    "macd_hist_rel": 0.006395,
    "dist_vwap": 0.591223,
    "dist_sma50": 0.069489,
    "atr_rel": 0.015577,
    "price_change": 0.021449,
    "volatility_21": 0.030799,
}


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


def _scale(series: pd.Series, name: str) -> pd.Series:
    """raw / scale, hard-clipped to [-1, 1] -- the v3 contract."""
    return (series / FEATURE_SCALES[name]).clip(-1.0, 1.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to an OHLCV frame. Returns a new frame.

    Every column in FEATURE_COLUMNS is guaranteed to lie in [-1, 1]
    (verified by scripts/audit_features_v3.py over ~2y of 1h data).
    """
    df = df.copy()
    open_, high, low, close, volume = (
        df["open"], df["high"], df["low"], df["close"], df["volume"])

    # -- candle shape (fractions of the open), scaled to the contract --
    top = pd.concat([open_, close], axis=1).max(axis=1)
    bot = pd.concat([open_, close], axis=1).min(axis=1)
    df["body_pct"] = _scale((close - open_) / open_, "body_pct")
    df["upper_wick"] = _scale((high - top) / open_, "upper_wick")
    df["lower_wick"] = _scale((bot - low) / open_, "lower_wick")
    df["range_pct"] = _scale((high - low) / open_, "range_pct")

    # -- normalized indicators --
    vwap = (close * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    sma_50 = close.rolling(50).mean()
    df["rsi_c"] = ((_rsi(close, 14) - 50.0) / 50.0).clip(-1.0, 1.0)
    df["macd_hist_rel"] = _scale((macd - signal) / close, "macd_hist_rel")
    df["dist_vwap"] = _scale((close - vwap) / close, "dist_vwap")
    df["dist_sma50"] = _scale((close - sma_50) / close, "dist_sma50")
    df["atr_rel"] = _scale(_atr(df, 14) / close, "atr_rel")
    df["price_change"] = _scale(close.pct_change(), "price_change")
    roll = close.rolling(21)
    df["volatility_21"] = _scale(roll.std() / roll.mean(), "volatility_21")
    vol = volume.replace(0, np.nan)
    df["volume_change"] = vol.pct_change().fillna(0.0).clip(-1.0, 1.0)

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
