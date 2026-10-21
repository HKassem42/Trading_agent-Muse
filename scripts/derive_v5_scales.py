"""Derive Features-v5 scale constants from real 1h BTC-USD data.

v5 adds five features (Kass's indicator set, 2026-10-20): Heikin-Ashi
direction, OBV slope, ADL slope, net-volume share, and the Elliott Wave
Oscillator. Only two of them need a scale constant:

- ha_dir raw = (ha_close - ha_open) / ATR(14) -- the Heikin-Ashi body in
  ATR units; unbounded in principle, so it gets the standard treatment:
  scale = 99.5th percentile of |raw| over ~2y of hourly candles
  (Coinbase feed, the v4+ primary source), then clip(raw/scale, -1, 1).
- ewo raw = (EMA5 - EMA35) / close -- the classic EWO construction;
  same p99.5 treatment.

obv_slope / adl_slope / netvol_rel are ratios of signed volume to total
volume over the same trailing 24h window, so they live in [-1, 1] by
construction and need no scale (printed here as a sanity check).

Usage: .venv/bin/python scripts/derive_v5_scales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.market_data import fetch_klines
from features.technical_indicators import _atr


def heikin_ashi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recursive Heikin-Ashi open/close (causal by construction)."""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    ha_close = (o + h + lo + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for t in range(1, len(ha_close)):
        ha_open[t] = 0.5 * (ha_open[t - 1] + ha_close[t - 1])
    return (pd.Series(ha_open, index=df.index),
            pd.Series(ha_close, index=df.index))


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """The v5 raw formulas, unclipped."""
    df = df.copy()
    close, volume = df["close"], df["volume"]
    atr_pct = _atr(df, 14) / close
    ha_open, ha_close = heikin_ashi(df)
    df["ha_dir"] = (ha_close - ha_open) / (atr_pct * close).replace(0, np.nan)

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema35 = close.ewm(span=35, adjust=False).mean()
    df["ewo"] = (ema5 - ema35) / close

    # Naturally bounded volume ratios (no scale; sanity check only).
    obv = (volume * np.sign(close.diff().fillna(0.0))).cumsum()
    vol24 = volume.rolling(24).sum().replace(0, np.nan)
    df["obv_slope"] = obv.diff(24) / vol24
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((close - df["low"]) - (df["high"] - close)) / rng
    adl = (mfm.fillna(0.0) * volume).cumsum()
    df["adl_slope"] = adl.diff(24) / vol24
    signed_bar = volume * np.sign(close - df["open"])
    df["netvol_rel"] = signed_bar.rolling(24).sum() / vol24
    return df.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    print(f"rows={len(df)}  {df.index[0]} .. {df.index[-1]}")
    raw = raw_features(df)
    print(f"{'feature':<12} {'p99.5|x|':>12} {'max|x|':>12} {'min':>12} {'max':>12} {'n':>7}")
    for c in ("ha_dir", "ewo"):
        col = raw[c].dropna()
        s = col.abs()
        print(f"{c:<12} {float(s.quantile(0.995)):>12.6f} {float(s.max()):>12.6f} "
              f"{float(col.min()):>12.6f} {float(col.max()):>12.6f} {len(col):>7}")
    for c in ("obv_slope", "adl_slope", "netvol_rel"):
        col = raw[c].dropna()
        print(f"{c:<12} bounded: min={col.min():.4f} max={col.max():.4f} "
              f"frac outside [-1,1]={(col.abs() > 1).mean():.6f} n={len(col)}")


if __name__ == "__main__":
    main()
