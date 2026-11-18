"""Pick #12 (HA Suite) feature-builder tests.

Contracts: the suite matrix is causal (truncating the future
never changes an earlier row); its first 10 columns ARE Pick
#10's HA matrix exactly; the 9 suite columns stay in [-1, 1]
and are finite once warmup passes; the volume-flow columns
match a hand-computed all-green-bars case exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS, build_ha_matrix
from models.muse_pick12 import (SUITE_ALL_COLUMNS, SUITE_COLUMNS,
                                build_ha_suite_matrix)

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
    X = build_ha_suite_matrix(df)
    assert list(X.columns) == SUITE_ALL_COLUMNS
    assert len(SUITE_COLUMNS) == 9 and len(SUITE_ALL_COLUMNS) == 19
    pd.testing.assert_frame_equal(X[HA_COLUMNS], build_ha_matrix(df))


def test_truncation_causality():
    df = _walk()
    full = build_ha_suite_matrix(df)
    for K in (150, 299):
        part = build_ha_suite_matrix(df.iloc[:K])
        a = full.iloc[:K - 1].to_numpy(dtype=float)
        b = part.iloc[:K - 1].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-10, equal_nan=True), (
            f"truncation at {K} changed earlier suite rows")


def test_suite_range_and_finite_after_warmup():
    X = build_ha_suite_matrix(_walk(400, seed=7))
    suite = X[SUITE_COLUMNS].iloc[WARMUP:]
    assert np.isfinite(suite.to_numpy(dtype=float)).all()
    assert (suite.abs() <= 1.0 + 1e-12).all().all()


def test_volume_flow_hand_computed_all_green():
    # Every candle strongly up -> every HA bar green, HA close
    # strictly rising: signed volume is all-positive, so the
    # 24h shares are exactly 1.0 once the window is full.
    closes = 100 * 1.02 ** np.arange(120)
    X = build_ha_suite_matrix(_df(closes, np.full(120, 10.0)))
    tail = X.iloc[WARMUP:]
    assert (tail["ha_netvol_rel"] == 1.0).all()
    assert (tail["ha_obv_slope"] == 1.0).all()


def test_rsi_matches_independent_computation():
    # Second-method check of the wiring: RSI(14) on the HA close
    # via a hand-written Wilder-EWM loop (the production _rsi
    # semantics: ewm alpha=1/14, adjust=False), centered.
    from models.ha_seq import heikin_ashi
    df = _walk(400, seed=11)
    hc = heikin_ashi(df)["ha_close"].to_numpy(dtype=float)
    d = np.diff(hc)
    gain = np.clip(d, 0, None)
    loss = np.clip(-d, 0, None)
    alpha = 1 / 14
    ag = np.full(len(hc), np.nan)
    al = np.full(len(hc), np.nan)
    g, l = gain[0], loss[0]
    for t in range(len(d)):
        g = alpha * gain[t] + (1 - alpha) * g
        l = alpha * loss[t] + (1 - alpha) * l
        ag[t + 1], al[t + 1] = g, l
    rsi = 100 - 100 / (1 + ag / al)
    expect = np.clip((rsi - 50) / 50, -1, 1)
    got = build_ha_suite_matrix(df)["ha_rsi_c"].to_numpy(dtype=float)
    assert np.allclose(got[WARMUP:], expect[WARMUP:], atol=1e-8)
