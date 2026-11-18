"""Muse's Pick #10 — HA feature matrix for the sequence LSTM.

Pre-registered in refs/muse-pick-10-preregistration-2026-11-05.md.
Ten per-candle values computed from the Heikin-Ashi candle
(program's own causal recursion, models.ha_seq.heikin_ashi),
including the signed same-color run length — Kass's "candles
form in a row" thesis given to the model explicitly. Every
column at row t depends only on raw candles <= t (pinned by
tests/test_muse_pick10.py truncation checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.ha_seq import heikin_ashi

HA_COLUMNS = ["ha_dir", "ha_body_pct", "ha_range_pct",
              "ha_upper_wick", "ha_lower_wick", "ha_streak",
              "ha_slope", "ha_disp", "raw_ret", "ha_body_rel"]
STREAK_CAP = 12


def build_ha_matrix(df: pd.DataFrame) -> pd.DataFrame:
    ha = heikin_ashi(df)
    ho = ha["ha_open"].to_numpy(dtype=float)
    hh = ha["ha_high"].to_numpy(dtype=float)
    hl = ha["ha_low"].to_numpy(dtype=float)
    hc = ha["ha_close"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    rng = hh - hl
    safe_rng = np.where(rng > 0, rng, 1.0)
    direction = np.where(hc >= ho, 1.0, -1.0)
    streak = np.empty(len(df), dtype=float)
    run = 0.0
    prev = 0.0
    for t in range(len(df)):
        run = run + 1.0 if direction[t] == prev else 1.0
        prev = direction[t]
        streak[t] = direction[t] * min(run, STREAK_CAP) / STREAK_CAP
    ha_slope = np.zeros(len(df))
    ha_slope[1:] = (hc[1:] - hc[:-1]) / hc[:-1]
    raw_ret = np.zeros(len(df))
    raw_ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    out = pd.DataFrame({
        "ha_dir": direction,
        "ha_body_pct": (hc - ho) / ho,
        "ha_range_pct": rng / ho,
        "ha_upper_wick": np.where(rng > 0,
                                  (hh - np.maximum(ho, hc)) / safe_rng,
                                  0.0),
        "ha_lower_wick": np.where(rng > 0,
                                  (np.minimum(ho, hc) - hl) / safe_rng,
                                  0.0),
        "ha_streak": streak,
        "ha_slope": ha_slope,
        "ha_disp": (hc - close) / close,
        "raw_ret": raw_ret,
        "ha_body_rel": np.where(rng > 0, (hc - ho) / safe_rng, 0.0),
    }, index=df.index)
    return out[HA_COLUMNS]
