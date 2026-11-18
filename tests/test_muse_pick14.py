"""Pick #14 ("The Committee") builder tests.

Contracts: the ha10 frame the committee consumes is causal
(truncating the future never changes an earlier row); the
committee is exactly the mean of its 7 fixed seeds' Pick-recipe
probabilities; the same seeds on the same rows reproduce
identical averaged probabilities; and the cross-fitted votes
the PPO phase consumes are look-ahead-free (nightly
crossfit_p_up semantics: leading segment neutral 0.5, every
fold's votes from a committee fit only on earlier rows --
flipping labels in the final block changes NO vote anywhere).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS, build_ha_matrix
from models.muse_pick14 import (SEEDS, CommitteeStacker,
                                crossfit_committee_p_up,
                                fit_committee, fit_seed_pipeline,
                                member_probas)

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


def _frame(n=300, seed=3):
    df = _walk(n=n, seed=seed)
    X = build_ha_matrix(df).iloc[WARMUP:]
    nxt = np.sign(df["close"].to_numpy()[1:]
                  - df["close"].to_numpy()[:-1])
    y = (nxt > 0).astype(int)
    y = np.concatenate([y, [y[-1]]])[WARMUP:]
    return X.reset_index(drop=True), y


def test_seeds_frozen():
    assert SEEDS == [42, 7, 123, 2024, 555, 9001, 31337]


def test_ha10_frame_truncation_causality():
    df = _walk()
    full = build_ha_matrix(df)
    assert list(full.columns) == HA_COLUMNS
    for cut in (80, 150, 299):
        part = build_ha_matrix(df.iloc[:cut])
        pd.testing.assert_frame_equal(
            part.iloc[WARMUP:], full.iloc[WARMUP:cut])


def test_averaging_correctness():
    X, y = _frame()
    committee = fit_committee(X, y)
    assert isinstance(committee, CommitteeStacker)
    assert len(committee.pipelines) == 7
    assert committee.seeds == SEEDS
    manual = np.vstack([
        fit_seed_pipeline(X, y, s).predict_proba(X)[:, 1]
        for s in SEEDS])
    got = committee.predict_proba(X)[:, 1]
    np.testing.assert_allclose(got, manual.mean(axis=0),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(member_probas(committee, X), manual,
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        committee.predict(X), (got >= 0.5).astype(int))
    assert np.all((got >= 0.0) & (got <= 1.0))


def test_committee_determinism():
    X, y = _frame()
    a = fit_committee(X, y).predict_proba(X)[:, 1]
    b = fit_committee(X, y).predict_proba(X)[:, 1]
    # Same seeds, same rows -> the same probabilities, up to
    # 1-ulp jitter (measured max 1.1e-16) from the parallel
    # boosters' reduction order on a contended box. Anything
    # larger would mean a seed is not actually fixed.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def test_crossfit_neutral_head_and_no_lookahead():
    X, y = _frame()
    votes = crossfit_committee_p_up(X, y, folds=2)
    # TimeSeriesSplit(2) on 240 rows: the first 80 rows precede
    # every test fold -> neutral 0.5, the production pattern.
    assert (votes[:80] == 0.5).all()
    assert np.all((votes >= 0.0) & (votes <= 1.0))
    # Flip the labels of the final block: no fold trains on
    # those rows, so NO vote anywhere may change (beyond the
    # 1-ulp booster jitter measured at 1.1e-16) -- a vote that
    # moved materially would prove a model saw its vote rows
    # in training.
    y2 = y.copy()
    y2[-48:] = 1 - y2[-48:]
    votes2 = crossfit_committee_p_up(X, y2, folds=2)
    np.testing.assert_allclose(votes, votes2, rtol=0, atol=1e-9)
