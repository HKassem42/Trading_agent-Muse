"""Pick #9 (LSTM sequence stacker) machinery tests.

The load-bearing contracts: a sequence for row t contains
exactly rows t-window+1..t (never a future row); sequences built
from a truncated matrix match the full build on every shared
row (no look-ahead through the windowing); the scaler is fit on
train rows only; exam rows never enter the training span.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick9 import (fit_scaler, make_sequences,
                               sequence_rows, transform)


def _matrix(n=60, d=3):
    # value encodes its own row index so alignment is checkable
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = np.arange(n) * 10 + j
    return X


def test_sequence_alignment_and_window():
    X = _matrix()
    seqs, rows = make_sequences(X, window=24)
    assert rows[0] == 23 and rows[-1] == 59
    assert seqs.shape == (60 - 23, 24, 3)
    # sequence for row t ends at row t and starts at t-23
    k = 10  # rows[10] == 33
    assert seqs[k][-1, 0] == 33 * 10
    assert seqs[k][0, 0] == (33 - 23) * 10
    # full ordering inside the window
    assert (np.diff(seqs[k][:, 0]) == 10).all()


def test_sequence_truncation_causality():
    X = _matrix(80)
    full, rows_full = make_sequences(X, window=24)
    for K in (40, 79):
        part, rows_part = make_sequences(X[:K], window=24)
        assert rows_part[-1] == K - 1
        shared = len(rows_part)
        assert np.array_equal(part, full[:shared]), (
            f"truncation at {K} changed earlier sequences")


def test_sequence_rows_helper_matches():
    X = _matrix(30)
    _, rows = make_sequences(X, window=5)
    assert sequence_rows(30, 5) == rows.tolist()
    assert sequence_rows(4, 5) == []


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)) + np.linspace(0, 5, 100)[:, None]
    mean, std = fit_scaler(X[:60])
    assert np.allclose(mean, X[:60].mean(axis=0))
    assert np.allclose(std, X[:60].std(axis=0))
    # full-data stats differ (drift), proving the fit is local
    assert not np.allclose(mean, X.mean(axis=0))
    Z = transform(X, mean, std)
    # float32 arithmetic: the standardized train mean is 0 to ~1e-7
    assert np.allclose(Z[:60].mean(axis=0), 0, atol=1e-6)


def test_train_exam_sequence_spans_disjoint():
    # the build's split contract: every supervised row is either
    # a train row or an exam row; sequences never relabel a row
    n, holdout, window = 1000, 792, 24
    rows = sequence_rows(n, window)
    train = [r for r in rows if r < n - holdout]
    exam = [r for r in rows if r >= n - holdout]
    assert len(train) + len(exam) == len(rows)
    assert max(train) < min(exam)
    # the first exam row's window reaches back into train rows —
    # causal (past candles only), which this assert pins exactly
    assert min(exam) - (window - 1) < n - holdout


def test_zero_window_rejected():
    with pytest.raises(ValueError):
        make_sequences(_matrix(10), window=0)
