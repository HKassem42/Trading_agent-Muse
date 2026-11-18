"""Muse's Pick #13 — HA + Bollinger/RelVol feature matrix.

Pre-registered in refs/muse-pick-13-preregistration-2026-11-12.md.
The 10 Heikin-Ashi candle values of Pick #10/#11
(`build_ha_matrix`, unchanged) PLUS exactly two columns — the
two dashboard indicators that were never model inputs:

- ha_boll_pos: Bollinger position of the HA close. With
  sma20/std20 the rolling-20 mean/std of the HA close,
  position p = (hc - (sma20 - 2*std20)) / (4*std20) (0 = lower
  band, 0.5 = the mean, 1 = upper band); the feature is
  (2p - 1) clipped to [-1, 1] — identically (hc - sma20) /
  (2*std20) before clipping, so +-1 is a +-2-sigma excursion.
  Same definition as the Book #5 indicators feed's Bollinger
  panel, squeezed to the feature contract.
- rel_vol: relative volume on REAL volume. r = volume /
  (rolling-24 mean of volume) — the dashboards' Relative
  Volume window; the feature is (r - 1) clipped to [-1, 1]
  (0 at average volume, +1 at >= 2x average, -> -1 as
  volume -> 0).

Every column at row t depends only on raw candles <= t
(trailing windows only; pinned by tests/test_muse_pick13.py
truncation checks, the v5.1 lesson).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.ha_seq import heikin_ashi
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix

EXTRA_COLUMNS = ["ha_boll_pos", "rel_vol"]
HA12_COLUMNS = HA_COLUMNS + EXTRA_COLUMNS


def build_ha12_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = build_ha_matrix(df)
    ha = heikin_ashi(df)
    hc = ha["ha_close"]
    vol = df["volume"].astype(float)

    sma20 = hc.rolling(20).mean()
    std20 = hc.rolling(20).std()
    boll_pos = ((hc - (sma20 - 2.0 * std20)) / (4.0 * std20))
    rel_vol = vol / vol.rolling(24).mean()

    out = pd.DataFrame(index=df.index)
    out["ha_boll_pos"] = (2.0 * boll_pos - 1.0).clip(-1.0, 1.0)
    out["rel_vol"] = (rel_vol - 1.0).clip(-1.0, 1.0)
    return pd.concat([base, out[EXTRA_COLUMNS]], axis=1)[HA12_COLUMNS]
