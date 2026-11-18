"""Tests for the two-step decision hierarchy (2026-10-24).

The hierarchy executes refs/hierarchy-plan-2026-10-24.md exactly as
pre-registered; these tests pin the mechanics the plan depends on:

(1) Stage targets: gate = label in {BUY, SHORT}; direction defined
    only on TRADE rows (BUY -> UP=1, SHORT -> DOWN=0).
(2) The fixed decision rule, including exact boundaries: P(TRADE)
    = 0.50 fires, just below holds; stage-2 argmax with an exact
    probability tie going UP (LONG-first). The duel conviction
    encoding (+1/-1/0) must equal the end-to-end labels.
(3) Fit discipline: a fit consumes exactly the rows handed to it,
    and stage 2 consumes only those rows' TRADE subset -- proven
    with poisoned out-of-fold rows and the fit audits.
(4) Learning sanity: on a synthetic frame where tradeability and
    direction are separable by construction, the hierarchy's
    end-to-end walk-forward must recover the signal (a broken
    two-stage wiring cannot).
(5) Verdict mechanics: PASS requires all three pre-registered
    rules; each rule fails independently.

Run: .venv/bin/python -m pytest tests/test_hierarchy.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import gbm_focus as gf  # noqa: E402
from models import hierarchy as hier  # noqa: E402


# ---------------------------------------------------------------- (1)
def test_stage_targets():
    y = np.array(["BUY", "HOLD", "SHORT", "HOLD", "BUY"])
    assert list(hier.gate_target(y)) == [1, 0, 1, 0, 1]
    # direction is only meaningful on TRADE rows: BUY -> 1, SHORT -> 0
    trade = y[hier.gate_target(y) == 1]
    assert list(hier.direction_target(trade)) == [1, 0, 1]


# ---------------------------------------------------------------- (2)
def test_decision_rule_boundaries_exact():
    p_trade = np.array([0.50, 0.499999, 0.50, 0.99, 0.0, 0.75])
    p_up = np.array([0.90, 0.90, 0.50, 0.499999, 0.90, 0.10])
    want = ["BUY",    # fires, up
            "HOLD",   # gate just below the bar
            "BUY",    # fires, exact direction tie -> UP
            "SHORT",  # fires, down
            "HOLD",   # gate at zero
            "SHORT"]  # fires, down
    assert list(hier.end_to_end_labels(p_trade, p_up)) == want


def test_conviction_matches_end_to_end():
    class _M:
        def predict(self, X):
            return hier.end_to_end_labels(
                np.array([0.9, 0.2, 0.8]), np.array([0.7, 0.7, 0.2]))

    c = hier.hierarchy_conviction(_M(), np.zeros((3, 2)))
    assert list(c) == [1.0, 0.0, -1.0]


def test_binary_params_are_lgbm_params_adapted():
    shared = {k: v for k, v in gf.LGBM_PARAMS.items()
              if k not in ("objective", "num_class")}
    assert hier.BINARY_PARAMS == {**shared, "objective": "binary"}


# ---------------------------------------------------------------- (3)
def _toy_xy(n=600, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Tradeable when feature 0 is extreme; direction by feature 1.
    y = np.full(n, "HOLD", dtype=object)
    trade = np.abs(X[:, 0]) > 0.8
    y[trade & (X[:, 1] > 0)] = "BUY"
    y[trade & (X[:, 1] <= 0)] = "SHORT"
    return X, y


def test_fit_consumes_only_given_rows_and_trade_subset():
    X, y = _toy_xy()
    ts = pd.date_range("2025-01-01", periods=len(y), freq="h")
    Xp = X.copy()
    Xp[500:] = 1e9  # poison everything from row 500 on
    model, audit = hier.fit_hierarchy(Xp[:300], y[:300], ts=ts[:300])
    assert audit["n_fed"] == 300
    assert audit["fit_max_abs"] < 1e6, "poison reached the fit"
    assert audit["fed_ts_max"] == ts[299]
    g = hier.gate_target(y[:300])
    assert audit["n_trade_fed"] == int(g.sum())
    assert audit["trade_fed_ts_max"] == ts[:300][g == 1].max()
    counts = audit["direction_class_counts"]
    assert counts[0] + counts[1] == audit["n_trade_fed"]
    assert counts[1] == int((y[:300][g == 1] == "BUY").sum())
    assert model is not None


# ---------------------------------------------------------------- (4)
def test_walk_forward_recovers_separable_signal():
    rng = np.random.default_rng(11)
    n = 2400
    X = rng.normal(size=(n, 3))
    y = np.full(n, "HOLD", dtype=object)
    trade = np.abs(X[:, 0]) > 0.7
    y[trade & (X[:, 1] > 0)] = "BUY"
    y[trade & (X[:, 1] <= 0)] = "SHORT"
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame(index=idx)
    df[gf.TARGET_COL] = y
    # arm machinery needs the arm's columns; alias the focus-11
    # names onto the signal features plus noise columns.
    for j, name in enumerate(gf.FOCUS_FEATURES):
        df[name] = X[:, j] if j < 3 else rng.normal(size=n)
    rec = hier.walk_forward_arm(df, "focus")
    p = rec["pooled"]
    assert p["n_graded"] > 1000
    assert p["accuracy"] > 0.85, p
    assert p["balanced_accuracy"] > 0.85, p
    d = p["diagnostics"]
    assert d["stage2_acc_on_true_trade"] > 0.9, d
    assert d["gate_balanced_accuracy"] > 0.9, d
    assert 0.2 < d["trade_base_rate"] < 0.6


# ---------------------------------------------------------------- (5)
def _wf(acc=0.70, bal=0.40, folds=(0.40, 0.41, 0.39, 0.42, 0.38),
        majority=0.669):
    return {"pooled": {"accuracy": acc, "balanced_accuracy": bal,
                       "majority_baseline": majority},
            "folds": [{"balanced_accuracy": b} for b in folds]}


def test_verdict_requires_all_three_rules():
    duel_ok = {"total_pnl": 100.0, "bench_pnl": 50.0,
               "non_hold_share": 0.25}
    rob_ok = {"total_pnl": 0.0}
    assert hier.apply_verdict(_wf(), duel_ok, rob_ok)["verdict"] \
        == "PASS"
    # rule 1 fails on accuracy below the majority baseline
    v = hier.apply_verdict(_wf(acc=0.60), duel_ok, rob_ok)
    assert v["verdict"] == "FAIL" and not v["rule1_walk_forward"]
    # rule 1 fails when the fold spread collapses (2 folds <= 1/3)
    v = hier.apply_verdict(
        _wf(folds=(0.40, 0.30, 0.41, 0.32, 0.44)), duel_ok, rob_ok)
    assert v["verdict"] == "FAIL" and not v["rule1_walk_forward"]
    # rule 2 fails on P&L not beating buy-and-hold
    v = hier.apply_verdict(
        _wf(), {"total_pnl": 10.0, "bench_pnl": 50.0,
                "non_hold_share": 0.25}, rob_ok)
    assert v["verdict"] == "FAIL" and not v["rule2_duel"]
    # rule 2 fails on the activity floor (boundary: 0.10 passes)
    v = hier.apply_verdict(
        _wf(), {"total_pnl": 100.0, "bench_pnl": 50.0,
                "non_hold_share": 0.099}, rob_ok)
    assert v["verdict"] == "FAIL" and not v["rule2_duel"]
    v = hier.apply_verdict(
        _wf(), {"total_pnl": 100.0, "bench_pnl": 50.0,
                "non_hold_share": 0.10}, rob_ok)
    assert v["rule2_duel"]
    # rule 3 fails on negative robustness (boundary: 0.0 passes)
    v = hier.apply_verdict(_wf(), duel_ok, {"total_pnl": -0.01})
    assert v["verdict"] == "FAIL" and not v["rule3_robustness"]
