"""Meta-Label Gate — run 84, the activity-calibrated gate
(2026-10-30).

Written BEFORE the implementation, against the design frozen
in refs/meta-label-study-84-2026-10-30.md. Run 84 changes only
the operating point of run 83's frozen gate: tau84 is the 70th
percentile of the frozen model's TRAINING-REGION out-of-fold
probabilities on stream-1 candidates (the line that would
have passed ~30% of stream-1 attempts in training), and a
secondary arm gates BOTH proposal streams at tau84. Pins:

- the quantile calibration on a hand-computed example,
  including the gate's own >= allow convention for the
  implied pass rate;
- calibration provenance: tau depends only on the stream-1
  OOF probabilities passed in — stream-2 rows, NaN (OOF-
  uncovered) rows, and row order cannot move it;
- Gate84's per-stream plumbing: proba reads the CURRENT
  stream's history and stream-id feature; attempts/vetoes
  are counted per stream;
- veto accounting: the gate object's totals equal the env's
  entries_attempted / entries_vetoed in the duel mirror;
- stream-2 duel integration: with a holding PPO, a vote
  crossing fires exactly one stream-2 proposal; a deny-all
  gate vetoes it and the book stays flat; PPO precedence —
  a non-hold PPO action consumes the step and the rule
  never fires;
- pair mode with a deny-all gate: every attempt vetoed,
  zero P&L, zero time-in-position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from training import meta_label83 as m83  # noqa: E402
from training import meta_label84 as m84  # noqa: E402


# ------------------------------------------------------------ fixtures
def _sup(n=264, closes=None, atr=0.01):
    if closes is None:
        closes = np.full(n, 100.0)
    closes = np.asarray(closes, dtype=float)
    df = pd.DataFrame(index=pd.RangeIndex(n))
    df["open"] = closes.copy()
    df["high"] = closes * 1.001
    df["low"] = closes * 0.999
    df["close"] = closes.copy()
    df["volume"] = 1.0
    for c in FEATURE_COLUMNS:
        df[c] = 0.0
    df["atr_pct_raw"] = atr
    df["target"] = 0
    return df


class _ConstModel:
    """predict_proba always returns class-1 probability p;
    asserts the gate really builds the 114-vector."""

    classes_ = np.array([0, 1])

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == m83.N_FEATURES_83, X.shape
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


class _StreamIdModel:
    """predict_proba returns the row's stream-id feature
    (index 113) as the class-1 probability."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.clip(X[:, 113], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _extras(sup, votes):
    closes = sup["close"].to_numpy(dtype=float)
    hours = (np.arange(len(sup)) % 24).astype(float)
    return m83.build_extra_context(
        votes, sup["atr_pct_raw"].to_numpy(dtype=float),
        np.zeros(len(sup)), closes, hours)


def _gate(model, threshold, sup, votes):
    g = m84.Gate84(model, threshold)
    g.set_window(0, np.zeros((len(sup), 98)), votes,
                 np.zeros(len(sup)), np.zeros(len(sup)),
                 sup["atr_pct_raw"].to_numpy(dtype=float),
                 _extras(sup, votes))
    return g


def _crossing_votes(n=264):
    """0.50 up to t=29, 0.60 from t=30: one up-crossing of the
    rule's 0.55 band, inside the episode starting at 24."""
    v = np.full(n, 0.50)
    v[30:] = 0.60
    return v


# ------------------------------------------------- quantile calibration
def test_calibrated_threshold_hand_computed():
    # Stream-1 OOF probabilities 0.1..1.0 (10 values). The 70th
    # percentile (numpy linear interpolation) sits at position
    # 0.7 * 9 = 6.3 -> 0.7 + 0.3 * (0.8 - 0.7) = 0.73. Under the
    # gate's >= rule, {0.8, 0.9, 1.0} pass -> exactly 30%.
    proba = np.arange(0.1, 1.01, 0.1)
    streams = np.ones(len(proba), dtype=int)
    out = m84.calibrated_threshold(proba, streams)
    assert out["threshold"] == pytest.approx(0.73, abs=1e-9)
    assert out["n_calibration"] == 10
    assert out["implied_pass_rate"] == pytest.approx(0.30, abs=1e-9)
    assert out["quantile"] == pytest.approx(0.70, abs=1e-12)


def test_calibrated_threshold_ignores_stream2_and_nan():
    s1 = np.arange(0.1, 1.01, 0.1)
    proba = np.concatenate([s1, [0.99, 0.99, np.nan, np.nan]])
    streams = np.array([1] * 10 + [0, 0, 1, 1])
    out = m84.calibrated_threshold(proba, streams)
    assert out["threshold"] == pytest.approx(0.73, abs=1e-9)
    assert out["n_calibration"] == 10  # stream-2 + NaN rows excluded


def test_calibrated_threshold_order_invariant_and_target():
    rng = np.random.default_rng(7)
    proba = rng.uniform(0, 1, 400)
    streams = np.ones(400, dtype=int)
    a = m84.calibrated_threshold(proba, streams)["threshold"]
    perm = rng.permutation(400)
    b = m84.calibrated_threshold(proba[perm], streams[perm])["threshold"]
    assert a == b  # the SET of training probabilities, nothing else
    med = m84.calibrated_threshold(proba, streams, target_pass_rate=0.5)
    assert med["threshold"] == pytest.approx(float(np.median(proba)))
    assert med["quantile"] == pytest.approx(0.5)


def test_calibrated_threshold_empty_raises():
    with pytest.raises((ValueError, AssertionError)):
        m84.calibrated_threshold(np.array([np.nan, np.nan]),
                                 np.array([1, 1]))


# ------------------------------------------------------- Gate84 streams
def test_gate84_proba_reads_current_stream():
    sup = _sup()
    votes = np.full(len(sup), 0.5)
    g = _gate(_StreamIdModel(), 0.5, sup, votes)
    g.current_stream = 1
    assert g.proba(10, 1) == pytest.approx(1.0)
    assert g.allow(10, 1) is True
    g.current_stream = 0
    assert g.proba(10, 1) == pytest.approx(0.0)
    assert g.allow(10, 1) is False
    assert g.attempts == {1: 1, 0: 1}
    assert g.vetoes == {1: 0, 0: 1}


def test_gate84_histories_are_per_stream():
    sup = _sup()
    votes = np.full(len(sup), 0.5)
    g = _gate(_ConstModel(0.9), 0.5, sup, votes)
    g.seed_history(1, [(5, 1), (6, 1)])
    g.seed_history(0, [(5, 0)])
    assert g.histories[1] == [(5, 1), (6, 1)]
    assert g.histories[0] == [(5, 0)]


# ------------------------------------------------------- duel integration
def test_combined_deny_all_vetoes_stream2_and_stays_flat():
    sup = _sup()
    votes = _crossing_votes()
    gate = _gate(_ConstModel(0.9), 2.0, sup, votes)  # deny-all
    res = m84.evaluate_windows_84(
        None, [sup], [votes], gate_factory=lambda w: gate,
        mode="combined", policy_fn=lambda obs: 0)
    gs = res["gate_stats"]
    assert gs["entries_attempted"] == 1          # the one crossing
    assert gs["entries_vetoed"] == 1
    assert gs["entries_filled"] == 0
    assert res["total_pnl"] == 0.0
    assert res["time_in_position"] == 0.0
    # veto accounting: gate object == env counters
    assert sum(gate.attempts.values()) == gs["entries_attempted"]
    assert sum(gate.vetoes.values()) == gs["entries_vetoed"]
    assert gate.attempts[0] == 1 and gate.attempts[1] == 0


def test_combined_allow_all_fills_stream2_entry():
    sup = _sup()
    votes = _crossing_votes()
    gate = _gate(_ConstModel(0.9), 0.5, sup, votes)
    res = m84.evaluate_windows_84(
        None, [sup], [votes], gate_factory=lambda w: gate,
        mode="combined", policy_fn=lambda obs: 0)
    gs = res["gate_stats"]
    assert gs["entries_attempted"] == 1
    assert gs["entries_vetoed"] == 0
    assert gs["entries_filled"] == 1
    assert gate.attempts[0] == 1 and gate.attempts[1] == 0
    assert res["time_in_position"] > 0.0


def test_combined_ppo_precedence_rule_never_fires():
    sup = _sup()
    votes = _crossing_votes()
    gate = _gate(_ConstModel(0.9), 0.5, sup, votes)
    res = m84.evaluate_windows_84(
        None, [sup], [votes], gate_factory=lambda w: gate,
        mode="combined", policy_fn=lambda obs: 1)  # always buy
    gs = res["gate_stats"]
    assert gate.attempts[0] == 0                  # rule never proposed
    assert gate.attempts[1] == gs["entries_attempted"] >= 1
    assert sum(gate.vetoes.values()) == gs["entries_vetoed"]


def test_pair_mode_deny_all_flat():
    sup = _sup()
    votes = np.full(len(sup), 0.5)
    gate = _gate(_ConstModel(0.9), 2.0, sup, votes)
    res = m84.evaluate_windows_84(
        None, [sup], [votes], gate_factory=lambda w: gate,
        mode="pair", policy_fn=lambda obs: 1)     # always buy
    gs = res["gate_stats"]
    assert gs["entries_attempted"] >= 1
    assert gs["entries_vetoed"] == gs["entries_attempted"]
    assert gs["entries_filled"] == 0
    assert res["total_pnl"] == 0.0
    assert res["time_in_position"] == 0.0
    assert gate.attempts[0] == 0                  # pair mode: stream 1 only
