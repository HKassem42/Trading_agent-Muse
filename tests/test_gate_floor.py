"""Two-part promotion floor for the PPO gate (Kass decision 2026-10-22).

A challenger is promotable only if the v4 relative formula passes AND
(1) it actually traded -- non-hold share >= 10% of its recorded action
mix, (2) its total duel P&L is strictly > $0, and (3) its total duel
P&L strictly beats buy-and-hold over the same windows. Fixtures are
shaped from the recorded duels in data/training.db (runs 20/24/26 are
the formula-passes that lost money or never traded). Run:
.venv/bin/python -m pytest tests/test_gate_floor.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.nightly import gate_decision  # noqa: E402

BENCH = 717.5366148134078  # buy-and-hold over the lab duel windows


def _chall(pnl, bench, hold):
    return {"total_pnl": pnl, "bench_pnl": bench,
            "action_mix": {"hold": hold, "buy": 0.0, "sell": 0.0,
                           "short": round(1.0 - hold, 3)}}


# -- recorded-duel fixtures -------------------------------------------------

def test_run20_silent_challenger_blocked_by_activity():
    # HA-Seq run 20: 100% hold, $0 P&L, formal score pass vs a losing
    # champion. Blocked by the activity floor (first failing condition).
    ok, why = gate_decision(_chall(0.0, BENCH, 1.0), formula_pass=True)
    assert ok is False
    assert why is not None and why.startswith("floor: activity")
    assert "0%" in why and "< 10%" in why


def test_run24_active_but_losing_blocked_by_profit_floor():
    # GBM Focus D run 24: 11.7% non-hold (clears activity), P&L -$3,178.
    ok, why = gate_decision(
        _chall(-3178.4057403288753, BENCH, 0.883), formula_pass=True)
    assert ok is False
    assert why is not None and why.startswith("floor: P&L")
    assert "≤ $0" in why


def test_run26_active_smote_loser_blocked():
    # GBM Lag F run 26: 14.2% non-hold, 40 real round trips, P&L -$4,417
    # while buy-and-hold made +$718. Still not promotable.
    ok, why = gate_decision(
        _chall(-4416.975650035325, BENCH, 0.858), formula_pass=True)
    assert ok is False
    assert why is not None and why.startswith("floor: P&L")


def test_profitable_but_below_buy_and_hold_blocked():
    # Pins condition 3 on its own: positive P&L that still trails B&H.
    ok, why = gate_decision(_chall(500.0, BENCH, 0.45), formula_pass=True)
    assert ok is False
    assert why is not None and "≤ B&H" in why


# -- the genuine article ------------------------------------------------------

def test_genuine_pass_is_promotable():
    ok, why = gate_decision(_chall(2000.0, 700.0, 0.45), formula_pass=True)
    assert ok is True
    assert why is None


# -- boundaries ---------------------------------------------------------------

def test_activity_exactly_ten_percent_passes():
    # non-hold exactly 0.10 meets the floor (>=, not >). The recorded
    # mix is rounded to 3 decimals, so hold=0.90 must not fall to a
    # float-subtraction artifact (1 - 0.9 = 0.09999...).
    ok, why = gate_decision(_chall(2000.0, 700.0, 0.90), formula_pass=True)
    assert ok is True, why


def test_activity_just_below_ten_percent_blocked():
    ok, why = gate_decision(_chall(2000.0, 700.0, 0.901), formula_pass=True)
    assert ok is False
    assert why is not None and why.startswith("floor: activity")


def test_pnl_exactly_zero_blocked():
    ok, why = gate_decision(_chall(0.0, -100.0, 0.45), formula_pass=True)
    assert ok is False
    assert why is not None and "≤ $0" in why


def test_pnl_exactly_equal_to_bench_blocked():
    # Must be strictly greater than buy-and-hold; a tie is not beating
    # doing nothing.
    ok, why = gate_decision(_chall(700.0, 700.0, 0.45), formula_pass=True)
    assert ok is False
    assert why is not None and "≤ B&H" in why


def test_formula_failure_short_circuits_without_floor_reason():
    # Perfect floor numbers cannot rescue a formula failure, and the
    # recorded reason for a formula failure stays the formula's own.
    ok, why = gate_decision(_chall(5000.0, 100.0, 0.45), formula_pass=False)
    assert ok is False
    assert why is None


def test_missing_action_mix_is_zero_activity_not_a_crash():
    ok, why = gate_decision({"total_pnl": 2000.0, "bench_pnl": 700.0},
                            formula_pass=True)
    assert ok is False
    assert why is not None and why.startswith("floor: activity")


# -- gate feed renders the recorded verdict for a floor-blocked run ---------

def test_gate_feed_shows_not_promoted_for_floor_blocked_run(monkeypatch):
    """A nightly run recorded under the new rule (formula pass in its
    windows, promoted=0, floor reason) must render through build_gate
    as the recorded verdict, never an unqualified PASSED."""
    from dashboards import compact

    champ_side = {"score": -2062.97, "total_pnl": -4484.72,
                  "max_drawdown": 2000.0,
                  "windows": [{"score": -900.0}, {"score": -1500.0},
                              {"score": 337.03}]}
    snap = {
        "runs": [
            {"id": 28, "ts": "2026-10-23T04:05:00+00:00", "kind": "ppo",
             "promoted": False,
             "metrics": {
                 "challenger": {
                     "score": -1871.18, "total_pnl": -4416.98,
                     "max_drawdown": 1500.0,
                     "bench_pnl": BENCH,
                     "action_mix": {"hold": 0.858, "buy": 0.087,
                                    "sell": 0.013, "short": 0.042},
                     "windows": [{"score": -800.0}, {"score": -1400.0},
                                 {"score": 328.82}]},
                 "champion": champ_side,
                 "reason": ("challenger won the relative gate but was "
                            "not promoted — floor: P&L -$4,417 ≤ $0; "
                            "champion kept")}},
            {"id": 17, "ts": "2026-10-21T13:40:45+00:00", "kind": "ppo",
             "promoted": True,
             "metrics": {"challenger": champ_side,
                         "champion": {"score": -1e9}}},
        ],
        "champion": {"accuracy": 0.4659, "precision": 0.4656, "folds": 5},
    }
    monkeypatch.setattr(compact.snapshot, "build_training_snapshot",
                        lambda: snap)
    # The duel is read from the full registry (2026-11-06), so the
    # fabricated world stubs the registry read too — the same
    # pattern as the gate tests in test_compact_feeds.py.
    monkeypatch.setattr(compact, "_all_runs_desc",
                        lambda: snap["runs"])
    g = compact.build_gate()
    assert g["chall"]["i"] == 28
    assert g["duel"]["fpass"] == 1  # formula really did pass
    assert g["duel"]["pass"] == 0
    assert g["duel"]["rec"] == "NOT PROMOTED"
