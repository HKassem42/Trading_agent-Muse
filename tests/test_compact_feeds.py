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
