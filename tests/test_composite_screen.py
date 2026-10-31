"""Tests for the Matrix v3 composite screen (2026-10-24).

The screen executes refs/composite-screen-plan-v3-2026-10-23.md
exactly as pre-registered; these tests pin the new mechanics the
plan depends on:

(1) The 60/40 rule mapping, including EXACT boundary behavior:
    60.0 is BUY (>=), 40.0 is SHORT (<=), values strictly inside
    the band are HOLD. The duel conviction encoding must carry the
    same decisions (+1 / -1 / 0).
(2) Composite loading from the real v3 CSV: the four composite
    columns exist, are complete and inside [0, 100], and the v3
    frame's rows / labels / OHLC are IDENTICAL to v2's (v3 changed
    only the indicator columns' representation).
(3) The pre-fee P&L accounting identity the plan's fee transparency
    rests on: for fills written by the real PaperBroker,
    net P&L == gross flows - fees, where gross flows are the signed
    price x qty cash flows recomputed from the ledger. The duel
    driver reports gross = net + fees_total; this test proves that
    decomposition against the broker itself, long and short.

Run: .venv/bin/python -m pytest tests/test_composite_screen.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.composite_screen import (COMPOSITE_COLS, load_v3,  # noqa: E402
                                     rule_conviction, rule_labels)
from models.exit_study import _read_new_fills  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v3.csv"
V2 = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v2.csv"
TARGET = "Target_Action (BUY/SELL/SHORT/HOLD)"


# ---------------------------------------------------------------- (1)
def test_rule_mapping_boundaries_exact():
    scores = np.array([60.0, 59.999999, 40.0, 40.000001, 50.0,
                       100.0, 0.0, 75.5, 24.5, 60.000001, 39.999999])
    want = ["BUY", "HOLD", "SHORT", "HOLD", "HOLD",
            "BUY", "SHORT", "BUY", "SHORT", "BUY", "SHORT"]
    assert list(rule_labels(scores)) == want


def test_rule_conviction_matches_labels():
    scores = np.array([80.0, 50.0, 20.0, 60.0, 40.0])
    assert list(rule_conviction(scores)) == [1.0, 0.0, -1.0, 1.0, -1.0]
    lab = rule_labels(scores)
    conv = rule_conviction(scores)
    for label, c in zip(lab, conv):
        assert (label == "BUY") == (c == 1.0)
        assert (label == "SHORT") == (c == -1.0)
        assert (label == "HOLD") == (c == 0.0)


def test_rule_thresholds_are_the_preregistered_60_40():
    from models.composite_screen import RULE_HIGH, RULE_LOW
    assert RULE_HIGH == 60.0 and RULE_LOW == 40.0
    assert COMPOSITE_COLS == ["score_master", "score_trend",
                              "score_momentum", "score_volume"]


# ---------------------------------------------------------------- (2)
def test_v3_loading_identity_vs_v2():
    use3 = ["Date", TARGET, "Open", "High", "Low", "Close"] + COMPOSITE_COLS
    use2 = ["Date", TARGET, "Open", "High", "Low", "Close"]
    v3 = load_v3(V3)
    v2 = pd.read_csv(V2, parse_dates=["Date"], usecols=use2)
    v2 = v2.set_index("Date").sort_index()
    assert len(v3) == len(v2) == 16_735
    assert v3.index.equals(v2.index)
    # Labels identical on every row.
    assert (v3[TARGET].to_numpy() == v2[TARGET].to_numpy()).all()
    counts = {c: int((v3[TARGET] == c).sum())
              for c in ("BUY", "HOLD", "SHORT")}
    assert counts == {"BUY": 2915, "HOLD": 10976, "SHORT": 2844}
    # Prices carried over exactly (v3 build's identity claim).
    for col in ("Open", "High", "Low", "Close"):
        assert np.array_equal(v3[col].to_numpy(dtype=float),
                              v2[col].to_numpy(dtype=float)), col
    # Composites: complete, in range.
    comp = v3[COMPOSITE_COLS]
    assert comp.notna().all().all()
    assert float(comp.min().min()) >= 0.0
    assert float(comp.max().max()) <= 100.0


# ---------------------------------------------------------------- (3)
def _gross_flows(fills) -> float:
    """Signed price x qty cash flows from ledger fills: SELL fills
    pay cash in, BUY fills pay cash out (a short open is a SELL, a
    cover a BUY -- broker.py's side convention)."""
    total = 0.0
    for f in fills:
        sign = 1.0 if f["side"] == "SELL" else -1.0
        total += sign * f["price"] * f["qty"]
    return total


def test_prefee_identity_long_round_trip(tmp_path):
    db = tmp_path / "fixture_long.db"
    b = PaperBroker(db, starting_balance=100_000.0)
    b.buy(100.0, ts="2026-01-01T00:00:00Z", atr_pct=0.01)
    assert b.btc > 0
    b.sell(110.0, pct=1.0, ts="2026-01-01T01:00:00Z")
    assert b.btc == 0
    fills, _ = _read_new_fills(db, 0)
    fees = float(sum(f["fee"] for f in fills))
    net = b.cash - 100_000.0
    gross = _gross_flows(fills)
    assert fees > 0
    assert net == gross - fees or abs(net - (gross - fees)) < 1e-6
    # and the no-fee trade itself was profitable at these prices
    assert gross > 0


def test_prefee_identity_short_round_trip(tmp_path):
    db = tmp_path / "fixture_short.db"
    b = PaperBroker(db, starting_balance=100_000.0)
    b.short_open(100.0, ts="2026-01-01T00:00:00Z", atr_pct=0.01)
    assert b.btc < 0
    b.cover(90.0, pct=1.0, ts="2026-01-01T01:00:00Z")
    assert b.btc == 0
    fills, _ = _read_new_fills(db, 0)
    fees = float(sum(f["fee"] for f in fills))
    net = b.cash - 100_000.0
    gross = _gross_flows(fills)
    assert fees > 0
    assert abs(net - (gross - fees)) < 1e-6
    assert gross > 0  # short sold at 100, covered at 90
