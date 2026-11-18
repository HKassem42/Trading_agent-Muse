"""Pick #13 (HA + Bollinger/RelVol) feature-builder tests.

Contracts: the ha12 matrix is causal (truncating the future
never changes an earlier row); its first 10 columns ARE Pick
#10's HA matrix exactly; the 2 extra columns stay in [-1, 1]
and are finite once warmup passes; both extras match a
hand-computed second-method value on a small case.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS, build_ha_matrix
from models.muse_pick13 import (EXTRA_COLUMNS, HA12_COLUMNS,
                                build_ha12_matrix)

WARMUP = 60


def _df(closes, volumes=None):
    c = np.asarray(closes, dtype=float)
    o = np.empty_like(c)
    o[0] = c[0]
    o[1:] = c[:-1]
    return pd.DataFrame({
        "open": o, "high": np.maximum(o, c) + 0.5,
        "low": np.minimum(o, c) - 0.5, "close": c,
        "volume": (np.asarray(volumes, dtype=float)
                   if volumes is not None else np.ones(len(c)))})


def _walk(n=300, seed=3):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vols = rng.integers(50, 150, n).astype(float)
    return _df(closes, vols)


def test_columns_and_ha10_block_identity():
    df = _walk()
    X = build_ha12_matrix(df)
    assert list(X.columns) == HA12_COLUMNS
    assert len(EXTRA_COLUMNS) == 2 and len(HA12_COLUMNS) == 12
    pd.testing.assert_frame_equal(X[HA_COLUMNS], build_ha_matrix(df))


def test_extras_in_range_and_finite_after_warmup():
    X = build_ha12_matrix(_walk()).iloc[WARMUP:]
    assert not X.isna().any().any()
    for c in EXTRA_COLUMNS:
        assert X[c].between(-1.0, 1.0).all(), c


def test_truncation_causality_all_rows():
    df = _walk()
    full = build_ha12_matrix(df)
    for cut in (80, 150, 299):
        part = build_ha12_matrix(df.iloc[:cut])
        pd.testing.assert_frame_equal(
            part.iloc[WARMUP:], full.iloc[WARMUP:cut])


def test_boll_pos_matches_hand_computation():
    # Second method: explicit window statistics at one row,
    # computed from the HA closes by an independent route
    # (manual recursion for the HA close series).
    closes = np.linspace(100.0, 130.0, 120) ** 1.02
    df = _df(closes)
    X = build_ha12_matrix(df)
    ha_close = np.empty(len(df))
    prev = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for i in range(len(df)):
        c = (df["open"].iloc[i] + df["high"].iloc[i]
             + df["low"].iloc[i] + df["close"].iloc[i]) / 4.0
        o = (prev + ha_close[i - 1]) / 2.0 if i else prev
        ha_close[i] = c
        prev = o
    row = 100
    win = ha_close[row - 19:row + 1]
    sma, std = win.mean(), win.std(ddof=1)
    expected = np.clip((ha_close[row] - sma) / (2.0 * std), -1, 1)
    assert abs(X["ha_boll_pos"].iloc[row] - expected) < 1e-9


def test_rel_vol_matches_hand_computation():
    vols = np.arange(1.0, 200.0)
    closes = 100 + np.cumsum(np.sin(np.arange(199) / 7.0))
    df = _df(closes, vols)
    X = build_ha12_matrix(df)
    row = 150
    expected = np.clip(vols[row] / vols[row - 23:row + 1].mean()
                       - 1.0, -1, 1)
    assert abs(X["rel_vol"].iloc[row] - expected) < 1e-9


def test_rel_vol_zero_at_flat_volume():
    df = _df(np.linspace(90, 110, 120), np.full(120, 42.0))
    X = build_ha12_matrix(df)
    assert (X["rel_vol"].iloc[WARMUP:] == 0.0).all()
