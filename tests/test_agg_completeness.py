"""Duel-record completeness (perfection pass, 2026-10-25): everything
evaluate_ppo computes must survive _agg into the registry. The old
flattening silently dropped the per-window episodes count (the
Training Lab rendered "Episodes judged —" for every modern duel),
the per-window drawdowns, mean_drawdown, and round_trips_per_day --
decision data Kass never got to see. These tests pin the full record
shape with hand-computed expectations, and reconcile the aggregate
arithmetic two ways.

Run: .venv/bin/python -m pytest tests/test_agg_completeness.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.nightly import _agg  # noqa: E402


def _w(score, pnl, bench, dd, mdd, mr, trips, episodes=10, mix=None):
    return {"score": score, "total_pnl": pnl, "bench_pnl": bench,
            "max_drawdown": dd, "mean_drawdown": mdd,
            "mean_reward": mr, "round_trips_per_day": trips,
            "episodes": episodes,
            "action_mix": mix or {"hold": 0.5, "buy": 0.2,
                                  "sell": 0.2, "short": 0.1}}


# Distinct values per window so a mix-up of windows or fields cannot
# accidentally pass.
RES = {
    "windows": [
        _w(111.11, 1000.10, 300.30, 400.40, 40.04, 0.111, 1.5),
        _w(222.22, 2000.20, 300.30, 900.90, 90.09, 0.222, 2.5),
        _w(333.33, 3000.30, 400.40, 500.50, 50.05, 0.333, 3.5,
           mix={"hold": 0.25, "buy": 0.25, "sell": 0.25,
                "short": 0.25}),
    ],
    "agg_score": (111.11 + 222.22 + 333.33) / 3,
    "total_pnl": 1000.10 + 2000.20 + 3000.30,
    "mean_reward": (0.111 + 0.222 + 0.333) / 3,
}


def test_agg_records_episodes_total_and_per_window():
    out = _agg(RES)
    # 3 windows x 10 episodes judged = 30; per-window count rides along.
    assert out["episodes"] == 30
    assert [w["episodes"] for w in out["windows"]] == [10, 10, 10]


def test_agg_records_drawdowns_and_trips():
    out = _agg(RES)
    assert out["max_drawdown"] == 900.90  # worst window
    assert out["mean_drawdown"] == pytest.approx(
        (40.04 + 90.09 + 50.05) / 3)
    assert [w["dd"] for w in out["windows"]] == [400.40, 900.90, 500.50]
    assert [w["mdd"] for w in out["windows"]] == [40.04, 90.09, 50.05]
    assert out["round_trips_per_day"] == pytest.approx(2.5)


def test_agg_arithmetic_reconciles_two_ways():
    out = _agg(RES)
    # Way 1: from the evaluate_windows aggregate fields.
    assert out["score"] == RES["agg_score"]
    assert out["total_pnl"] == RES["total_pnl"]
    # Way 2: recomputed from the window records alone (what a reader
    # of training.db can verify): mean of window scores, sum of
    # window P&Ls and benches.
    assert out["score"] == pytest.approx(
        sum(w["score"] for w in RES["windows"]) / 3)
    assert out["windows"][0]["pnl"] == 1000.10
    assert sum(w["pnl"] for w in out["windows"]) == pytest.approx(
        RES["total_pnl"], abs=0.02)  # window P&Ls are rounded to cents
    assert out["bench_pnl"] == pytest.approx(300.30 + 300.30 + 400.40)
    assert [w["mr"] for w in out["windows"]] == [0.111, 0.222, 0.333]
    # Action mix stays the LAST window's, by documented design.
    assert out["action_mix"] == {"hold": 0.25, "buy": 0.25,
                                 "sell": 0.25, "short": 0.25}
    assert out["runnable"] is True


def test_agg_none_keeps_the_full_schema():
    """A non-runnable challenger still records the complete field set
    (zeros, not missing keys), so consumers never branch on absence."""
    out = _agg(None)
    assert out["runnable"] is False
    for key in ("score", "total_pnl", "mean_reward", "max_drawdown",
                "mean_drawdown", "bench_pnl", "episodes",
                "round_trips_per_day"):
        assert key in out, f"fallback record dropped {key}"
    assert out["episodes"] == 0
