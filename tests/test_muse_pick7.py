"""Muse's Pick #7 ("Slow Clock") — adversarial tests for the 24h
split machinery, written BEFORE the module per the quality
standard. The load-bearing risks at a 24h horizon are silent:
a train label that reads a close inside the exam period, and
walk-forward folds whose 24h labels straddle the fold boundary.
Each invariant test is paired with the naive construction it
guards against, and asserts the naive one VIOLATES the invariant
— a test that cannot fail on the wrong code is worthless.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick7 import (anatomy, exam_split, is_clean,
                               purged_wf_splits)
from models.muse_picks456 import horizon_labels

H = 24


def _index(n):
    return pd.date_range("2025-01-01", periods=n, freq="h")


def test_horizon_labels_hand_computed():
    close = pd.Series(
        [100.0, 101.0, 99.0, 102.0, 98.0, 103.0], index=_index(6))
    y2 = horizon_labels(close, 2)
    # t0: 99 > 100? no -> 0 ; t1: 102 > 101 -> 1 ; t2: 98 > 99? no
    # t3: 103 > 102 -> 1 ; t4/t5: no t+2 close -> NaN
    assert y2.iloc[:4].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert y2.iloc[4:].isna().all()
    y24 = horizon_labels(
        pd.Series(np.arange(100.0, 130.0), index=_index(30)), H)
    assert int(y24.isna().sum()) == H  # exactly the last 24 rows
    assert y24.iloc[:6].tolist() == [1.0] * 6  # rising series


def test_labels_are_not_silently_lagged():
    # A one-row shift in the close series must change the labels;
    # if the construction were accidentally lagged/aligned by
    # label (the v5.1 bug class), this catches it.
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(size=200)),
                      index=_index(200))
    y = horizon_labels(close, H)
    y_shifted_input = horizon_labels(close.shift(1), H)
    same = (y.dropna() == y_shifted_input.reindex(y.dropna().index)
            ).mean()
    assert same < 0.95


def test_exam_split_purges_boundary_labels():
    n, hold_pos = 400, 300
    idx = _index(n)
    close = pd.Series(100 + np.sin(np.arange(n) / 9.0), index=idx)
    y = horizon_labels(close, H)
    train_pos, exam_pos = exam_split(idx, y, hold_pos, horizon=H)
    # every exam row is at/after the holdout start with a label
    assert exam_pos.min() >= hold_pos
    assert y.iloc[exam_pos].notna().all()
    # THE invariant: no train label reads a close inside the exam
    assert train_pos.max() + H < hold_pos
    assert y.iloc[train_pos].notna().all()
    # adversarial control: the naive split (all rows before the
    # holdout) violates the invariant on exactly H rows
    naive_train = np.arange(0, hold_pos)
    naive_train = naive_train[y.iloc[naive_train].notna().to_numpy()]
    assert naive_train.max() + H >= hold_pos
    assert int((naive_train + H >= hold_pos).sum()) == H


def test_exam_split_drops_undefined_exam_tail():
    n, hold_pos = 400, 300
    idx = _index(n)
    close = pd.Series(100 + np.cos(np.arange(n) / 7.0), index=idx)
    y = horizon_labels(close, H)
    _, exam_pos = exam_split(idx, y, hold_pos, horizon=H)
    # last H rows of the frame have no t+24 close: excluded
    assert exam_pos.max() == n - H - 1
    assert len(exam_pos) == (n - hold_pos) - H


def test_purged_wf_embargo():
    n = 1200
    folds = purged_wf_splits(n, horizon=H, n_splits=3)
    assert len(folds) == 3
    for tr, va in folds:
        assert len(tr) > 0 and len(va) > 0
        # no train label straddles the boundary, and the first
        # validation rows (whose features sit inside the last
        # train labels' span) are embargoed too
        assert tr.max() + H < va.min()
        assert va.min() > tr.max()
    # adversarial control: raw TimeSeriesSplit violates it
    raw = list(TimeSeriesSplit(n_splits=3).split(np.zeros((n, 1))))
    assert any(tr.max() + H >= va.min() for tr, va in raw)


def test_anatomy_and_clean_rule():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    proba = np.where(y == 1, 0.8, 0.2)  # perfect caller
    m = anatomy(proba, y)
    assert m["acc"] == 1.0 and m["precision"] == 1.0
    assert m["up_calls"] == int(y.sum())
    assert is_clean(m)
    # degenerate: only 3 up-calls — accuracy is meaningless
    proba2 = np.full(len(y), 0.4)
    proba2[:3] = 0.9
    m2 = anatomy(proba2, y)
    assert m2["up_calls"] == 3
    assert not is_clean(m2)
    # all-up caller (Next Try #1's tuned-theta anatomy)
    m3 = anatomy(np.full(len(y), 0.6), y)
    assert m3["up_calls"] == len(y)
    assert m3["acc"] == pytest.approx(y.mean())
