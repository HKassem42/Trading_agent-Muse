"""Master live-desk feed (2026-10-31): ONE combined payload for the
three paper books, built by REUSING the per-book snapshot/stats code
-- the master numbers must equal what each book's own dashboard
shows at the same moment, identities must derive from the seat
manifests, Book #3's vetoes must match its gate_decisions rows, and
a book with no closed trades must yield null (never NaN, never a
crash) for the stats that don't exist yet.

Vision is stubbed off in these tests (include_vision=False): the
vision builders fetch market data and compute frames, and the
master contract's vision-derived fields (p_up fallback, top_gauges)
have their own fallback semantics tested here.
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact, snapshot  # noqa: E402


def _master():
    return compact.build_master(include_vision=False)


def test_master_totals_equal_per_book_snapshots():
    """Combined equity/pnl are the sums of the three books' OWN
    snapshot values built in the same process -- no forked math."""
    m = _master()
    assert [b["id"] for b in m["books"]] == ["book1", "book2", "book3"]
    paths = [("book1", None, None),
             ("book2", compact.SHADOW_DB, compact.SHADOW_STATE),
             ("book3", compact.GATED_DB, compact.GATED_STATE)]
    total_eq = 0.0
    total_pnl = 0.0
    for book, (bid, db, st) in zip(m["books"], paths):
        s = (snapshot.build_trading_snapshot()
             if db is None else
             snapshot.build_trading_snapshot(db_path=db, state_path=st))
        assert book["equity"] == s["equity"], bid
        assert book["cash"] == s["cash"], bid
        assert book["btc_held"] == s["btc"], bid
        assert book["side"] == s["position"], bid
        assert book["p_up"] == (round(s["stacker_p_up"], 4)
                                if s["stacker_p_up"] is not None
                                else None), bid
        assert book["closed_trades"] == s["trade_stats"]["total_trades"], bid
        assert book["win_rate"] == s["trade_stats"]["win_pct"], bid
        total_eq += s["equity"]
        total_pnl += book["pnl"]
    assert m["combined"]["equity"] == round(total_eq, 2)
    assert m["combined"]["pnl"] == round(total_pnl, 2)
    assert m["btc"]["price"] == m["books"][0]["mark_price"]


def test_master_identities_derive_from_manifests():
    m = _master()
    b1, b2, b3 = m["books"]
    seat = compact._seat_manifest()["seat"]
    assert b1["identity"]["brain_run"] == seat["ppo"]["run_id"] == 44
    assert b1["identity"]["eyes_run"] == seat["stacker"]["run_id"] == 40
    assert b1["identity"]["feature_set"] == seat["stacker"]["feature_set"]
    assert b1["kind"] == "driver"
    assert b1["identity"]["gate_run"] is None
    desk2 = compact._shadow_desk_block()
    assert b2["kind"] == "shadow"
    assert b2["identity"]["eyes_run"] == desk2["eyes"]["run"] == 73
    assert "Champion #4" in b2["identity"]["eyes_label"]
    assert b2["identity"]["indicators_n"] == desk2["eyes"]["n_features"] == 98
    desk3 = compact._gated_desk_block()
    assert b3["kind"] == "gated"
    assert b3["identity"]["gate_run"] == desk3["gate"]["run"] == 82
    assert b3["identity"]["gate_threshold"] == desk3["gate"]["threshold"] == 0.55
    assert "vetoes" not in b1 and "vetoes" not in b2


def test_master_book3_vetoes_match_gate_decisions():
    m = _master()
    b3 = m["books"][2]
    con = sqlite3.connect(compact.GATED_DB)
    rows = con.execute(
        "SELECT ts, proba, threshold, verdict FROM gate_decisions "
        "ORDER BY id DESC LIMIT 10").fetchall()
    con.close()
    expect = [{"ts": ts, "proba": round(p, 4), "threshold": thr,
               "verdict": v} for ts, p, thr, v in rows]
    assert b3["vetoes"] == expect


def test_master_curves_and_lists_bounded():
    m = _master()
    for b in m["books"]:
        assert len(b["equity_curve"]) <= 120
        assert len(b["dd_curve"]) <= 120
        assert len(b["recent_fills"]) <= 12
        assert len(b["recent_decisions"]) <= 12
        if b["equity_curve"]:
            assert b["equity_curve"][-1][1] == b["equity"]
    assert len(m["books"][2]["vetoes"]) <= 10
    for b in m["books"]:
        assert len(b["decision_log"]) <= 24
    payload = json.dumps(m, separators=(",", ":"))
    print(f"master feed (no vision) bytes={len(payload)}")
    # Budget rebaselined 20,000 -> 27,000 on 2026-10-31: per-candle decision
    # logging (Kass's order) added decision_log (<=24 records/book) +
    # latest_thought (features once per book). Undieted the no-vision
    # payload measured 28,077B with all three books logging; a feed-only
    # precision diet in _decision_log_block (features 4dp, positions 6dp;
    # the JSONL on disk keeps full precision) brought it to 25,676B.
    # SENTINEL: decision logs still grow ~300B/record/book until each book
    # holds 24 records (~46KB projected at full depth) — when this trips,
    # the next diet is a depth/precision decision for the parent, not a
    # blind raise.
    assert len(payload) < 27_000


def test_master_zero_closed_trades_yields_null_stats(tmp_path):
    """A fresh book (equity rows, zero trades) must produce null
    PF/expectancy/avg/payoff/best/worst -- not NaN, not 0.0, no crash."""
    db = tmp_path / "paper_test.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT, "
                "side TEXT, price REAL, qty REAL, fee REAL, "
                "cash_after REAL, btc_after REAL, reason TEXT, "
                "context TEXT)")
    con.execute("CREATE TABLE equity (ts TEXT, equity REAL, cash REAL, "
                "btc REAL, price REAL)")
    for i, eq in enumerate((100000.0, 100010.0, 99990.0)):
        con.execute("INSERT INTO equity VALUES (?, ?, ?, 0.0, ?)",
                    (f"2026-10-31T0{i}:00:00", eq, eq, 77000.0))
    con.commit()
    con.close()
    state = tmp_path / "state.json"
    state.write_text("{}")
    book = compact._master_book(
        "bookX", db, state,
        {"brain_run": 1, "brain_label": "PPO · Run 1", "eyes_run": 2,
         "eyes_label": "Stacker · Run 2", "feature_set": "prod21",
         "indicators_n": 21, "gate_run": None, "gate_threshold": None},
        kind="driver", label="Book X", start_equity=100000.0,
        include_vision=False)
    assert book["closed_trades"] == 0
    assert book["profit_factor"] is None
    assert book["expectancy"] is None
    assert book["avg_win"] is None and book["avg_loss"] is None
    assert book["payoff"] is None
    assert book["best_trade"] is None and book["worst_trade"] is None
    assert book["equity"] == 99990.0
    json.dumps(book)  # must be JSON-clean (no NaN/Infinity)
