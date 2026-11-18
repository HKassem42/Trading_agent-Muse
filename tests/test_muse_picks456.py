"""Adversarial tests for Muse's Pick #4/#5/#6 machinery.

Pre-registered in refs/muse-picks-456-preregistration-2026-11-04.md.
The load-bearing risks are (a) horizon labels leaking or being
misaligned, (b) recency weights running the wrong way in time,
(c) the label-cleaning threshold being computed from anything
but the train rows handed to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_picks456 import (  # noqa: E402
    build_bag_learners,
    build_wide_stack,
    clean_keep_mask,
    horizon_labels,
    quartile_threshold,
    recency_weights,
)


def _close_fixture() -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    return pd.Series(
        [100.0, 101.0, 99.0, 102.0, 103.0, 98.0, 104.0, 105.0],
        index=idx)


def test_horizon_labels_hand_computed():
    close = _close_fixture()
    y1 = horizon_labels(close, 1)
    # next-candle up?: 101>100 T, 99>101 F, 102>99 T, 103>102 T,
    # 98>103 F, 104>98 T, 105>104 T, last row undefined
    assert y1.iloc[:7].tolist() == [1, 0, 1, 1, 0, 1, 1]
    assert np.isnan(y1.iloc[7])
    y2 = horizon_labels(close, 2)
    # close[t+2] > close[t]: 99>100 F, 102>101 T, 103>99 T,
    # 98>102 F, 104>103 T, 105>98 T, last two undefined
    assert y2.iloc[:6].tolist() == [0, 1, 1, 0, 1, 1]
    assert np.isnan(y2.iloc[6]) and np.isnan(y2.iloc[7])


def test_horizon_label_uses_only_its_own_endpoint():
    close = _close_fixture()
    base = horizon_labels(close, 4)
    mutated = close.copy()
    mutated.iloc[7] = 1.0  # far-future crash, beyond t+4 for t<=2
    after = horizon_labels(mutated, 4)
    assert base.iloc[:3].tolist() == after.iloc[:3].tolist()
    mutated2 = close.copy()
    mutated2.iloc[4] = 50.0  # the t+4 endpoint for t=0: 103 -> 50
    after2 = horizon_labels(mutated2, 4)
    assert base.iloc[0] == 1 and after2.iloc[0] == 0


def test_horizon_labels_truncation_invariant():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(
        rng.normal(0, 0.004, 500))), index=idx)
    for h in (1, 4, 24):
        full = horizon_labels(close, h)
        cut = horizon_labels(close.iloc[:300], h)
        shared = cut.dropna()
        assert (full.reindex(shared.index) == shared).all()


def test_recency_weights_direction_and_halflife():
    idx = pd.date_range("2026-01-01", periods=241, freq="D", tz="UTC")
    w = recency_weights(idx, half_life_days=120)
    assert w[-1] == pytest.approx(1.0)          # newest row
    assert w[-121] == pytest.approx(0.5, abs=1e-9)  # exactly 120d old
    assert np.all(np.diff(w) >= 0)              # grows toward present
    assert w[0] == pytest.approx(0.25, abs=1e-9)    # 240 days old


def test_quartile_threshold_and_clean_mask():
    vals = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    thr = quartile_threshold(vals)
    assert thr == pytest.approx(np.quantile(vals, 0.25))
    keep = clean_keep_mask(vals, thr)
    # strictly greater than the quartile survives; the two smallest
    # (0.01, 0.02 <= thr=0.0275) are dropped
    assert keep.tolist() == [False, False, True, True, True, True,
                             True, True]


def test_bag_learners_are_the_champion_four():
    learners = build_bag_learners(42)
    assert [n for n, _ in learners] == ["rf", "xgb", "lgbm", "et"]
    for _, est in learners:
        assert est.n_estimators == 200
    assert learners[0][1].max_depth == 8   # RF mirrors the recipe
    assert learners[1][1].max_depth == 5   # XGB mirrors the recipe


def test_wide_stack_has_seven_diverse_learners():
    stack = build_wide_stack(42)
    names = [n for n, _ in stack.estimators]
    assert names == ["rf", "xgb", "lgbm", "et", "knn", "mlp", "lr"]
    assert stack.cv == 3
    assert type(stack.final_estimator).__name__ == "LogisticRegression"
    knn = dict(stack.estimators)["knn"]
    assert knn.n_neighbors == 40
