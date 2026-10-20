"""Derive Features-v3 scale constants from real 1h BTC-USD data.

For each price-relative feature, the scale is the 99.5th percentile of
|raw value| over ~2 years of hourly candles, computed with the v2 raw
formulas (no clipping). Features v3 then uses clip(raw / scale, -1, 1),
so ~99.5% of history lands strictly inside [-1, 1] and only true extremes
saturate at the rails.

Usage: .venv/bin/python scripts/derive_feature_scales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.yfinance_adapter import fetch_klines


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


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """The v2 raw formulas, unclipped (volume_change left raw too)."""
    df = df.copy()
    open_, high, low, close, volume = (
        df["open"], df["high"], df["low"], df["close"], df["volume"])
    top = pd.concat([open_, close], axis=1).max(axis=1)
    bot = pd.concat([open_, close], axis=1).min(axis=1)
    df["body_pct"] = (close - open_) / open_
    df["upper_wick"] = (high - top) / open_
    df["lower_wick"] = (bot - low) / open_
    df["range_pct"] = (high - low) / open_
    vwap = (close * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    sma_50 = close.rolling(50).mean()
    df["rsi_raw"] = _rsi(close, 14)
    df["macd_hist_rel"] = (macd - signal) / close
    df["dist_vwap"] = (close - vwap) / close
    df["dist_sma50"] = (close - sma_50) / close
    df["atr_rel"] = _atr(df, 14) / close
    df["price_change"] = close.pct_change()
    roll = close.rolling(21)
    df["volatility_21"] = roll.std() / roll.mean()
    vol = volume.replace(0, np.nan)
    df["volume_change"] = vol.pct_change()
    return df.replace([np.inf, -np.inf], np.nan)


SCALED = ["body_pct", "upper_wick", "lower_wick", "range_pct",
          "macd_hist_rel", "dist_vwap", "dist_sma50", "atr_rel",
          "price_change", "volatility_21"]


def main() -> None:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    print(f"rows={len(df)}  {df.index[0]} .. {df.index[-1]}")
    raw = raw_features(df)
    print(f"{'feature':<16} {'p99.5|x|':>12} {'max|x|':>12} {'min':>12} {'max':>12} {'n':>7}")
    scales = {}
    for c in SCALED:
        col = raw[c].dropna()  # per-column: warmup NaNs differ per feature
        s = col.abs()
        p = float(s.quantile(0.995))
        scales[c] = p
        print(f"{c:<16} {p:>12.6f} {float(s.max()):>12.6f} "
              f"{float(col.min()):>12.6f} {float(col.max()):>12.6f} {len(col):>7}")
    vc = raw["volume_change"].dropna()
    print(f"volume_change raw: min={vc.min():.3f} max={vc.max():.3f} "
          f"frac outside [-1,1]={(vc.abs() > 1).mean():.4f} (clip-only per spec)")
    rsi = raw["rsi_raw"]
    print(f"rsi raw: min={rsi.min():.2f} max={rsi.max():.2f} "
          f"-> rsi_c=(rsi-50)/50 is in [-1,1] by construction")
    print("\nFEATURE_SCALES = {")
    for k, v in scales.items():
        print(f'    "{k}": {v:.6f},')
    print("}")


if __name__ == "__main__":
    main()
