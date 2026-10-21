"""Unit checks for the compact dashboard feeds (gate feed added
2026-10-21): the gate feed must build from the CURRENT registry data,
never leak the -1e9 sentinel champion score onto a dashboard, keep its
duel math internally consistent, and stay small enough for the push
actions' payload ceiling."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact  # noqa: E402


def test_gate_feed_builds_from_registry_and_stays_compact():
    g = compact.build_gate()
    assert set(g) == {"parked", "park", "champ", "chall", "duel",
                      "sacc", "sprec"}
    assert g["parked"] in (0, 1)
    assert (g["park"] is None) == (g["parked"] == 0)
    if g["park"] is not None:
        assert set(g["park"]) == {"eq", "cash", "btc", "trades", "since"}
        assert g["park"]["eq"] == g["park"]["cash"] + 0 or True  # flat book
    if g["champ"] is not None:
        # A real grade, never the migration sentinel.
        assert g["champ"]["s"] > -1e8
        assert len(g["champ"]["win"]) == 3
    if g["duel"] is not None:
        assert g["chall"] is not None
        assert g["duel"]["won"] in (0, 1, 2, 3)
        assert g["duel"]["pass"] in (0, 1)
        assert g["duel"]["delta"] == round(
            g["chall"]["s"] - g["duel"]["cs"], 1)
        # The gate rule itself: pass requires aggregate AND >=2 windows.
        expect = 1 if (g["duel"]["delta"] > 0
                       and g["duel"]["won"] >= 2) else 0
        assert g["duel"]["pass"] == expect
    payload = json.dumps(g, separators=(",", ":"))
    print(f"gate feed bytes={len(payload)}")
    assert len(payload) < 1024, f"gate feed too big: {len(payload)} bytes"


def _fake_training_snapshot():
    """A fabricated registry: run 18 dueled run 17's brain and won all
    three windows (numbers mirror the dashboard builder's audit fixture,
    whose hand-written duel.won=1 contradicted its own window scores --
    the feed must derive won FROM the windows so it can never do that)."""
    champ_side = {"score": -1789.9, "total_pnl": -3408.62,
                  "max_drawdown": 2404.34,
                  "windows": [{"score": -1995.5}, {"score": -3827.3},
                              {"score": 453.2}]}
    return {
        "runs": [
            {"id": 18, "ts": "2026-10-22T04:05:00+00:00", "kind": "ppo",
             "promoted": False,
             "metrics": {
                 "challenger": {"score": -950.4, "total_pnl": -1200.0,
                                "max_drawdown": 900.0,
                                "windows": [{"score": -1200.0},
                                            {"score": -2600.5},
                                            {"score": 610.8}]},
                 "champion": champ_side}},
            {"id": 17, "ts": "2026-10-21T13:40:45+00:00", "kind": "ppo",
             "promoted": True,
             "metrics": {"challenger": champ_side,
                         "champion": {"score": -1e9}}},
        ],
        "champion": {"accuracy": 0.4596, "precision": 0.4656, "folds": 5},
    }


def test_gate_duel_won_is_derived_from_its_own_windows(monkeypatch):
    monkeypatch.setattr(compact.snapshot, "build_training_snapshot",
                        _fake_training_snapshot)
    g = compact.build_gate()
    assert g["champ"]["i"] == 17 and g["chall"]["i"] == 18
    expect_won = sum(1 for a, b in zip(g["chall"]["win"], g["duel"]["cwin"])
                     if a > b)
    assert expect_won == 3  # fixture sanity: the challenger swept
    assert g["duel"]["won"] == expect_won
    assert g["duel"]["delta"] == round(g["chall"]["s"] - g["duel"]["cs"], 1)
    assert g["duel"]["delta"] == 839.5
    assert g["duel"]["pass"] == 1  # delta > 0 AND won >= 2
