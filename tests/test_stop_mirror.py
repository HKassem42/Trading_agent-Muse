"""Dashboard stop mirrors: every feed that displays a position
stop must show the broker's EFFECTIVE stop -- PaperBroker.
stop_level() computed on the persisted live state -- not the raw
stop_anchor and not a locally re-derived figure.

History: the 2026-10-24 fix pinned every mirror to the persisted
stop ANCHOR because the three displays disagreed with each other
(anchor $86,607.68 vs trading snapshot $88,720.07 vs market feed
$82,118.42 on the same live short). That pin made the feeds agree
with EACH OTHER while disagreeing with the BROKER: the anchor is
only the locked INITIAL stop, and once the trailing arm activates
(price TRAIL_ACTIVATE_PCT+ in the position's favor) the level that actually fires
is the tighter trailing level. On that same short the broker's
effective stop was $81,285.08 (low_water $79,691.25 x 1.02) while
the dashboard drew $86,607.68 -- Kass spotted that the displayed
stop "doesn't trail." Superseded the same day: stop_price is now
the effective level, and the anchor rides along as its own field
(stop_anchor, plus stop_trailing 0/1) so no information is lost.
The computation is delegated to PaperBroker.stop_level itself
(restored from the persisted fields exactly as live.py restores
it), so the display can never drift from the exit logic again;
anchor-less legacy states get the broker's own entry/ATR fallback
from the same call. Flat positions show no stop anywhere.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact, snapshot  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

SHORT_ANCHOR = 82400.55   # locked initial stop (entry 80000 + ~3%)
LONG_ANCHOR = 77600.25    # locked initial stop (entry 80000 - ~3%)
# Trailed levels the broker computes from the water marks below
# (atr 0.005 -> trail_pct clamps to the 2% floor):
SHORT_TRAILED = 71400.0   # low_water 70000 x 1.02
LONG_TRAILED = 88200.0    # high_water 90000 x 0.98

SHORT_LIVE = {
    "last_ts": "2026-10-24T03:00:00", "last_stance": "sell",
    "last_p_up": 0.5, "cash": 124003.84, "btc": -0.5,
    "entry_price": 80000.0, "last_entry_price": 80000.0,
    "entry_atr_pct": 0.005, "high_water": None,
    "low_water": 70000.0,  # fell 12.5% past entry: trailing armed
    "stop_anchor": SHORT_ANCHOR,
}

LONG_LIVE = {
    "last_ts": "2026-10-24T03:00:00", "last_stance": "buy",
    "last_p_up": 0.5, "cash": 80000.0, "btc": 0.25,
    "entry_price": 80000.0, "last_entry_price": 80000.0,
    "entry_atr_pct": 0.005, "high_water": 90000.0,  # +12.5%: armed
    "low_water": None,
    "stop_anchor": LONG_ANCHOR,
}

# Same anchors, but the water marks never moved the activation
# distance (2% since 2026-11-02) past entry, so the trailing arm is NOT active and the
# effective stop IS the anchor.
SHORT_UNTRAILED = {
    **SHORT_LIVE, "low_water": 79000.0,   # only -1.25% from entry
}
LONG_UNTRAILED = {
    **LONG_LIVE, "high_water": 81000.0,   # only +1.25% from entry
}

FLAT_LIVE = {
    "last_ts": "2026-10-24T03:00:00", "last_stance": "sell",
    "last_p_up": 0.5, "cash": 100000.0, "btc": 0.0,
    "entry_price": None, "last_entry_price": 80000.0,  # stale on purpose
    "entry_atr_pct": 0.005, "high_water": None, "low_water": None,
    "stop_anchor": None,
}


def _make_data_dir(tmp_path: Path, live: dict) -> Path:
    """A fixture data dir: paper.db (real schema, one equity row
    whose signed btc matches the live_state) + live_state.json."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(data / "paper.db")
    con.execute(
        "CREATE TABLE equity(ts TEXT NOT NULL, equity REAL NOT NULL,"
        " cash REAL NOT NULL, btc REAL NOT NULL, price REAL NOT NULL)")
    con.execute(
        "CREATE TABLE trades(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL,"
        " qty REAL NOT NULL, fee REAL NOT NULL, cash_after REAL NOT NULL,"
        " btc_after REAL NOT NULL, reason TEXT NOT NULL, context TEXT)")
    btc = live["btc"]
    con.execute(
        "INSERT INTO equity(ts, equity, cash, btc, price)"
        " VALUES (?, ?, ?, ?, ?)",
        ("2026-10-24T03:00:00+00:00", 100261.15, 124003.84, btc, 81000.0))
    con.commit()
    con.close()
    (data / "live_state.json").write_text(json.dumps(live))
    return data


def _trading_feed(tmp_path, monkeypatch, live) -> dict:
    data = _make_data_dir(tmp_path, live)
    monkeypatch.setattr(snapshot, "DATA", data)
    return compact.build_compact_trading()


def _market_feed(tmp_path, monkeypatch, live) -> dict:
    data = _make_data_dir(tmp_path, live)
    monkeypatch.setattr(snapshot, "DATA", data)
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    candles = [[80000, 80100, 79900, 80050, 10]] * 96
    monkeypatch.setattr(compact, "_market_candles",
                        lambda: (1_700_000_000, candles, None))
    return compact.build_market(1)


def _broker_stop_level(live: dict):
    """Ground truth: PaperBroker.stop_level() on a broker restored
    from the persisted fields the way live.py restores them."""
    b = PaperBroker(":memory:", starting_balance=100_000,
                    max_position_pct=0.25, stop_loss_pct=0.05)
    b.cash = float(live["cash"])
    b.btc = float(live["btc"])
    b.entry_price = live.get("entry_price")
    b.entry_atr_pct = live.get("entry_atr_pct")
    b.high_water = live.get("high_water")
    b.low_water = live.get("low_water")
    b.stop_anchor = live.get("stop_anchor")
    return b.stop_level()


# --------------------------------------- trailed: effective != anchor
def test_trading_feed_stop_is_broker_effective_level_short(
        tmp_path, monkeypatch):
    out = _trading_feed(tmp_path, monkeypatch, SHORT_LIVE)
    assert out["position"] == "SHORT"
    level, trailing = _broker_stop_level(SHORT_LIVE)
    assert trailing and round(level, 2) == SHORT_TRAILED  # fixture truth
    # The headline stop is the level that would actually fire...
    assert out["stop_price"] == round(level, 2)
    assert out["stop_price"] != SHORT_ANCHOR
    # ...and the anchor + trailing flag ride along as their own data.
    assert out["stop_anchor"] == SHORT_ANCHOR
    assert out["stop_trailing"] == 1


def test_market_feed_stop_is_broker_effective_level_short(
        tmp_path, monkeypatch):
    out = _market_feed(tmp_path, monkeypatch, SHORT_LIVE)
    assert out["entry_price"] == 80000.0
    assert out["stop_price"] == SHORT_TRAILED


def test_trading_feed_stop_is_broker_effective_level_long(
        tmp_path, monkeypatch):
    out = _trading_feed(tmp_path, monkeypatch, LONG_LIVE)
    assert out["position"] == "LONG"
    level, trailing = _broker_stop_level(LONG_LIVE)
    assert trailing and round(level, 2) == LONG_TRAILED
    assert out["stop_price"] == round(level, 2)
    assert out["stop_price"] != LONG_ANCHOR
    assert out["stop_anchor"] == LONG_ANCHOR
    assert out["stop_trailing"] == 1


def test_market_feed_stop_is_broker_effective_level_long(
        tmp_path, monkeypatch):
    out = _market_feed(tmp_path, monkeypatch, LONG_LIVE)
    assert out["entry_price"] == 80000.0
    assert out["stop_price"] == LONG_TRAILED


# --------------------------- untrailed: effective == anchor, flag off
def test_trading_feed_stop_is_anchor_until_trailing_arms_short(
        tmp_path, monkeypatch):
    out = _trading_feed(tmp_path, monkeypatch, SHORT_UNTRAILED)
    level, trailing = _broker_stop_level(SHORT_UNTRAILED)
    assert not trailing and round(level, 2) == SHORT_ANCHOR
    assert out["stop_price"] == SHORT_ANCHOR
    assert out["stop_anchor"] == SHORT_ANCHOR
    assert out["stop_trailing"] == 0


def test_trading_feed_stop_is_anchor_until_trailing_arms_long(
        tmp_path, monkeypatch):
    out = _trading_feed(tmp_path, monkeypatch, LONG_UNTRAILED)
    level, trailing = _broker_stop_level(LONG_UNTRAILED)
    assert not trailing and round(level, 2) == LONG_ANCHOR
    assert out["stop_price"] == LONG_ANCHOR
    assert out["stop_anchor"] == LONG_ANCHOR
    assert out["stop_trailing"] == 0


def test_market_feed_stop_is_anchor_until_trailing_arms_short(
        tmp_path, monkeypatch):
    out = _market_feed(tmp_path, monkeypatch, SHORT_UNTRAILED)
    assert out["stop_price"] == SHORT_ANCHOR


# --------------------------------------------------------------- flat
def test_trading_feed_stop_is_none_when_flat(tmp_path, monkeypatch):
    out = _trading_feed(tmp_path, monkeypatch, FLAT_LIVE)
    assert out["position"] == "FLAT"
    assert out["stop_price"] is None
    assert out["stop_anchor"] is None
    assert out["stop_trailing"] == 0


def test_market_feed_stop_is_none_when_flat(tmp_path, monkeypatch):
    out = _market_feed(tmp_path, monkeypatch, FLAT_LIVE)
    assert out["stop_price"] is None
