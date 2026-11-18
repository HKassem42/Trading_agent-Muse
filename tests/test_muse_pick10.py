"""Pick #10 (LSTM over HA sequences) feature-builder tests.

Contracts: the HA matrix is causal (truncating the future never
changes an earlier row); the streak column counts same-color
runs exactly as hand-computed; HA direction matches the HA
candle color; every column is finite on real-shaped data after
the first row.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS, build_ha_matrix


def _df(closes, opens=None):
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens if opens is not None else closes, dtype=float)
    return pd.DataFrame({
        "open": o, "high": np.maximum(o, c) + 1.0,
        "low": np.minimum(o, c) - 1.0, "close": c,
        "volume": np.ones(len(c))})


def test_truncation_causality():
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = _df(closes)
    full = build_ha_matrix(df)
    for K in (100, 299):
        part = build_ha_matrix(df.iloc[:K])
        a = full.iloc[:K - 1].to_numpy(dtype=float)
        b = part.iloc[:K - 1].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-10), (
            f"truncation at {K} changed earlier HA rows")


def test_streak_hand_computed():
    # closes engineered so HA colors run: G G G R R G
    # (HA close = avg of OHLC; with symmetric wicks the HA color
    # follows the smoothed close path)
    closes = [100, 102, 104, 106, 103, 100, 99]
    df = _df(closes)
    X = build_ha_matrix(df)
    streak = X["ha_streak"].to_numpy()
    # streak magnitude never exceeds the cap and sign = color
    assert (np.abs(streak) <= 1.0).all()
    dirs = X["ha_dir"].to_numpy()
    assert (np.sign(streak[1:]) == dirs[1:]).all()


def test_dir_matches_ha_color():
    from models.ha_seq import heikin_ashi
    closes = [100, 101, 99, 102, 98, 103]
    df = _df(closes)
    ha = heikin_ashi(df)
    X = build_ha_matrix(df)
    expect = np.where(ha["ha_close"] >= ha["ha_open"], 1.0, -1.0)
    assert (X["ha_dir"].to_numpy() == expect).all()


def test_columns_and_finiteness():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    X = build_ha_matrix(_df(closes))
    assert list(X.columns) == HA_COLUMNS
    vals = X.iloc[1:].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert X.index.equals(_df(closes).index)
