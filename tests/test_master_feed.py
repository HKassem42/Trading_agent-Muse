"""Master live-desk feed (2026-10-31): ONE combined payload for the
five paper books, built by REUSING the per-book snapshot/stats code
-- the master numbers must equal what each book's own dashboard
shows at the same moment, identities must derive from the seat
manifests, Book #3's vetoes must match its gate_decisions rows, and
a book with no closed trades must yield null (never NaN, never a
crash) for the stats that don't exist yet.

Vision is stubbed off in most of these tests (include_vision=False):
the vision builders fetch market data and compute frames. Since
2026-11-03 each book's p_up/vision_ts are ALWAYS its own recorded
loop state (never the recomputed vision's vote -- see
test_master_book_p_up_is_own_state_not_recomputed_vision); vision
supplies only top_gauges.
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact, snapshot  # noqa: E402
from tests.test_shadow_desk import _ohlcv  # noqa: E402


def _master():
    return compact.build_master(include_vision=False)


def test_master_totals_equal_per_book_snapshots():
    """Combined equity/pnl are the sums of the five books' OWN
    snapshot values built in the same process -- no forked math."""
    m = _master()
    assert [b["id"] for b in m["books"]] == [
        "book1", "book2", "book3", "book4", "book5"]
    paths = [("book1", None, None),
             ("book2", compact.SHADOW_DB, compact.SHADOW_STATE),
             ("book3", compact.GATED_DB, compact.GATED_STATE),
             ("book4", compact.HA_DB, compact.HA_STATE),
             ("book5", compact.CHAMP_DB, compact.CHAMP_STATE)]
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
    b1, b2, b3, b4, b5 = m["books"]
    seat = compact._seat_manifest()["seat"]
    assert b1["identity"]["brain_run"] == seat["ppo"]["run_id"] == 44
    assert b1["identity"]["eyes_run"] == seat["stacker"]["run_id"] == 40
    assert b1["identity"]["feature_set"] == seat["stacker"]["feature_set"]
    assert b1["kind"] == "driver"
    assert b1["identity"]["gate_run"] is None
    desk2 = compact._shadow_desk_block()
    assert b2["kind"] == "shadow"
    # Book #2's eyes: run 119 (Champion #11, ha10) since Kass's
    # 2026-11-09 seat swap; run 73 / Champion #4 / 98 scored was
    # the pin before it.
    assert b2["identity"]["eyes_run"] == desk2["eyes"]["run"] == 119
    assert "Champion #11" in b2["identity"]["eyes_label"]
    assert b2["identity"]["feature_set"] == desk2["eyes"]["feature_set"] == "ha10"
    assert b2["identity"]["indicators_n"] == desk2["eyes"]["n_features"] == 10
    desk3 = compact._gated_desk_block()
    assert b3["kind"] == "gated"
    assert b3["identity"]["gate_run"] == desk3["gate"]["run"] == 82
    assert b3["identity"]["gate_threshold"] == desk3["gate"]["threshold"] == 0.55
    desk4 = compact._ha_desk_block()
    assert b4["kind"] == "ha"
    assert b4["identity"]["brain_run"] == desk4["driver"]["run"] == 44
    assert b4["identity"]["eyes_run"] == desk4["eyes"]["run"] == 110
    assert "Champion #6" in b4["identity"]["eyes_label"]
    assert b4["identity"]["feature_set"] == desk4["eyes"]["feature_set"] == "ha10"
    assert b4["identity"]["indicators_n"] == desk4["eyes"]["n_features"] == 10
    assert b4["identity"]["gate_run"] is None
    desk5 = compact._champ_desk_block()
    assert b5["kind"] == "champ"
    # RE-PINNED 2026-11-21 (run 147 crowning): Muse crowned the
    # 2026-11-21 retrain's PPO winner run 147 as lab PPO champion
    # and re-pinned Book #5's follows-the-crown seat (Era 2), so
    # the Book #5 driver IS run 147; eyes unchanged (run 126).
    assert b5["identity"]["brain_run"] == desk5["driver"]["run"] == 147
    assert b5["identity"]["eyes_run"] == desk5["eyes"]["run"] == 126
    assert "Champion #13" in b5["identity"]["eyes_label"]
    assert b5["identity"]["feature_set"] == desk5["eyes"]["feature_set"] == "ha10"
    assert b5["identity"]["indicators_n"] == desk5["eyes"]["n_features"] == 10
    assert b5["identity"]["gate_run"] is None
    assert b5["label"] == desk5["book_name"]
    assert "vetoes" not in b1 and "vetoes" not in b2 and "vetoes" not in b4
    assert "vetoes" not in b5


def _expand_rows(rows, cols):
    """Diet round 5: expand positional feed rows back to dicts
    (None where the array slot is null / the dict key was elided
    under the round-2 rule -- the client treats them identically)."""
    return [{c: (row[i] if i < len(row) else None)
             for i, c in enumerate(cols)} if isinstance(row, list)
            else row for row in rows]


def test_master_book3_vetoes_match_gate_decisions():
    m = _master()
    b3 = m["books"][2]
    con = sqlite3.connect(compact.GATED_DB)
    rows = con.execute(
        "SELECT ts, proba, threshold, verdict FROM gate_decisions "
        "ORDER BY id DESC LIMIT 8").fetchall()
    con.close()
    expect = [{"ts": ts, "proba": round(p, 4), "threshold": thr,
               "verdict": v} for ts, p, thr, v in rows]
    assert _expand_rows(b3["vetoes"], compact.MASTER_VETO_COLS) == expect


def test_master_rows_are_positional_arrays_lossless():
    """Diet round 5 (2026-11-05): the master's row sections ship
    as POSITIONAL ARRAYS -- the key names ride once in the client
    column lists instead of repeating on every row (at the four-
    book plateau the repeated keys cost ~10.6KB, more than the
    whole remaining headroom). Expansion must reproduce the dict
    rows EXACTLY: every value the desk renders, None where the
    round-2 elision dropped a null key."""
    m = _master()
    for b in m["books"]:
        log, _latest = compact._decision_log_block(b["id"])
        expect = [{c: r.get(c)
                   for c in compact.MASTER_DECISION_LOG_COLS}
                  for r in log]
        assert all(isinstance(r, list) for r in b["decision_log"])
        assert _expand_rows(
            b["decision_log"],
            compact.MASTER_DECISION_LOG_COLS) == expect
        assert all(isinstance(r, list) and
                   len(r) == len(compact.MASTER_FILL_COLS)
                   for r in b["recent_fills"])
        assert all(isinstance(r, list) and
                   len(r) == len(compact.MASTER_DECISION_COLS)
                   for r in b["recent_decisions"])
    assert all(isinstance(r, list) and
               len(r) == len(compact.MASTER_VETO_COLS)
               for r in m["books"][2]["vetoes"])


def test_master_fills_and_decisions_expand_to_source_rows():
    """The expanded recent_fills must equal each book's own
    snapshot recent_trades, and the expanded recent_decisions the
    newest-10 trades-table rows (p_up from the fill context;
    depth 12 until diet round 7, 2026-11-17) -- the same source
    rows the dict format carried."""
    m = _master()
    paths = [(None, None), (compact.SHADOW_DB, compact.SHADOW_STATE),
             (compact.GATED_DB, compact.GATED_STATE),
             (compact.HA_DB, compact.HA_STATE)]
    for b, (db, st) in zip(m["books"], paths):
        s = (snapshot.build_trading_snapshot()
             if db is None else
             snapshot.build_trading_snapshot(db_path=db, state_path=st))
        expect_fills = [
            {"ts": r["ts"], "action": r["side"], "price": r["price"],
             "qty": r["qty"], "reason": r["reason"]}
            for r in s["recent_trades"][:10]]
        assert _expand_rows(
            b["recent_fills"], compact.MASTER_FILL_COLS) == expect_fills
        con = snapshot._paper_db(db)
        raw = con.execute(
            "SELECT ts, side, reason, context FROM trades "
            "ORDER BY id ASC").fetchall()
        con.close()
        expect_dec = []
        for ts, fside, reason, ctx in reversed(raw[-10:]):
            p = None
            if ctx:
                try:
                    p = json.loads(ctx).get("p")
                except Exception:
                    p = None
            expect_dec.append({"ts": ts, "action": fside,
                               "reason": reason, "p_up": p})
        assert _expand_rows(
            b["recent_decisions"],
            compact.MASTER_DECISION_COLS) == expect_dec


def test_master_row_array_fallback_keeps_unknown_keys():
    """A row carrying a key outside the column list must pass
    through as the original dict -- a future field degrades that
    one row to the old format instead of being silently dropped
    (the client expands arrays and passes objects through)."""
    row = {"ts": "2026-11-05T00:00:00", "future_field": 1}
    assert compact._master_row_array(
        row, compact.MASTER_DECISION_LOG_COLS) == row
    known = {"ts": "2026-11-05T00:00:00", "close": 1.5}
    arr = compact._master_row_array(
        known, compact.MASTER_DECISION_LOG_COLS)
    assert isinstance(arr, list)
    assert len(arr) == len(compact.MASTER_DECISION_LOG_COLS)
    assert arr[0] == "2026-11-05T00:00:00" and arr[1] == 1.5
    assert all(v is None for v in arr[2:])


def test_master_gauges_are_positional_pairs(tmp_path):
    """top_gauges ships as [name, value] pairs (the client maps
    them through GAUGE_COLS); the values are the vision's own."""
    book = _sentinel_book(tmp_path, "book1", 0.1111, 0.9999)
    assert book["top_gauges"] == [["sentinel_gauge", 42.0]]


def test_master_curves_and_lists_bounded():
    m = _master()
    for b in m["books"]:
        assert len(b["equity_curve"]) <= 120
        # Diet round 4 (2026-11-05): master dd cap 40 -> 32, so
        # the bound is cap + 2 endpoint guards = 34 (was 42).
        # Diet round 7 (2026-11-17): dd cap 24 -> 20, bound 22.
        assert len(b["dd_curve"]) <= 22
        assert len(b["recent_fills"]) <= 10
        assert len(b["recent_decisions"]) <= 10
        if b["equity_curve"]:
            assert b["equity_curve"][-1][1] == b["equity"]
    # Diet round 4: the master's race_bh copy is strided to the
    # same cap 32 master-side (build_shadow_race's own cap-40
    # curve still serves the shadow desk); every point stays a
    # verbatim point of that race curve.
    if m["race_bh"]:
        # Diet round 7 (2026-11-17): race_bh cap 32 -> 24, bound 26.
        assert len(m["race_bh"]) <= 26
        race_pts = {tuple(pt) for pt in
                    compact.build_shadow_race()["bh"]["curve"]}
        assert all(tuple(pt) in race_pts for pt in m["race_bh"])
    assert len(m["books"][2]["vetoes"]) <= 8
    for b in m["books"]:
        assert len(b["decision_log"]) <= 8
    payload = json.dumps(m, separators=(",", ":"))
    print(f"master feed (no vision) bytes={len(payload)}")
    # Budget history: 20,000 -> 27,000 on 2026-10-31 (decision logging
    # added decision_log + latest_thought; a feed-only precision diet,
    # features 4dp / positions 6dp, brought the no-vision build to
    # 25,676B). Rebaselined 27,000 -> 30,000 on 2026-11-01: the logs
    # filling from 1 to 10 records/book plus drawdown-curve growth
    # took the WITH-VISION payload to ~37.7KB and push_master
    # REJECTED it in transport (a 34.3KB build had been accepted
    # earlier the same night -- the ceiling sits between the two).
    # Feed-only diet in _decision_log_block (the on-disk JSONL is
    # untouched): latest_thought features trimmed to the strongest
    # 24 by |value| with features_total carrying the full count,
    # and decision_log depth 24 -> 16. Target: total payload
    # <= 30,000B with headroom for a future Book #4. SENTINEL: at
    # full depth-16 steady state plus full 120-point drawdown
    # curves this trips again by design -- the next diet (dd-curve
    # stride / record slimming) is a parent decision, not a raise.
    # Diet round 2 LANDED 2026-11-02 (the parent decision, taken in
    # the daily dashboard review): _compress_flat_runs drops the
    # interior points of constant dd-curve runs losslessly, the
    # master dd-curve cap went 120 -> 64 (tail full-res kept), and
    # decision_log feed rows omit None-valued keys. Budget stays
    # 30,000 -- it was NOT raised. No-vision build after round 2:
    # see the printed bytes when this test runs.
    # Diet round 3 LANDED 2026-11-05 (Kass's 2026-11-04 standing
    # directive: feed fixes need no approval): the round-2 steady
    # state outgrew the transport anyway -- the no-vision build
    # reached 34,238B and the WITH-VISION payload 35,360B, which
    # push_master rejected twice in transport (worker tool-call
    # truncation; the largest verified accepted push is 34.3KB,
    # 2026-11-01). Round 3: dd-curve cap 64 -> 40 (the module's
    # race-curve cap; points surviving the stride stay verbatim
    # points of the flat-run-compressed series, pinned by
    # test_master_dd_curve_stride_diet_keeps_points_verbatim)
    # and decision-block price/money floats (stop_level / close
    # / equity) round to 2dp in the feed copies, the precision
    # prices render at (pinned by test_decision_log's
    # price-precision test). The budget is REBASELINED to 32,500
    # on the no-vision build with the basis stated, not silently:
    # post-diet no-vision measures 32,241B and vision adds only
    # top_gauges (~1.1KB), putting the pushed payload at ~33.4KB
    # -- ~0.9KB under the largest accepted push and ~2.0KB under
    # the observed truncation. Deliberately NOT cut (display-
    # changing, flagged to Kass instead): decision_log depth
    # 16 -> 12 (~2.9KB) and latest_thought features 24 -> 16
    # (~0.75KB). If this trips again, those are the next levers;
    # do not just raise it.
    # Diet round 4 LANDED 2026-11-05 (same directive): measured
    # section-by-section against the round-3 pushed payload, the
    # 2026-11-05 swell was NOT the equity curves (stride-bounded
    # by _downsample_curve at <= 19 pts; +38B total across books
    # in 8 volatile hours) and NOT a 36.6KB payload -- that figure
    # was the cron worker's default-separator serialization of a
    # 33,670B compact payload (+2,980B of spaces). The one
    # compression-dependent section was dd_curve: Book #3 rode
    # 30 -> 39 points (917B -> 1,193B) as BTC's drop ended the
    # flat runs compression feeds on, and the no-vision build
    # drifted 32,241B -> 32,507B, onto this sentinel. Round 4
    # makes the bound deterministic instead of luck-based: dd
    # cap 40 -> 32 and the master's race_bh copy strided to cap
    # 32 master-side (both verbatim-point strides, newest 8
    # full-res; pinned by the round-4 worst-case tests below,
    # which use a zero-flat-run series where compression saves
    # nothing). Budget REBASELINED to 32,250 on the no-vision
    # build, basis: post-round-4 no-vision measures ~31.9KB and
    # the arithmetic worst case (every curve at its deterministic
    # cap with max-length value reprs: dd 3 x 34 x 31B, equity
    # 3 x 19 x 33B, race 34 x 33B over the ~26.0KB of count-
    # bounded non-curve content) is ~32.2KB; vision adds ~1.1KB,
    # so the worst-case pushed payload is ~33.3KB -- ~1.0KB
    # under the largest verified accepted push (34.3KB) and
    # ~2.1KB under the observed 35,360B truncation.
    # Diet round 5 LANDED 2026-11-05 (same directive): Book #4
    # (HA desk, added that day) took the with-vision payload to
    # ~34.5KB while its sections were still nearly empty --
    # filling it to a mature book's caps projected ~40.9KB, and
    # this sentinel's no-vision build was already over budget.
    # The cut is structural, not a trim: the row sections
    # (decision_log, recent_fills, recent_decisions, vetoes,
    # top_gauges) now ship as POSITIONAL ARRAYS -- key names
    # ride once in compact.MASTER_*_COLS, mirrored by the Master
    # client's feed.ts column lists, which expand rows back to
    # objects at read time (and pass object rows through, so
    # deploy order never breaks the desk). Every value is
    # verbatim -- expansion reproduces the dict rows exactly,
    # pinned by the round-5 tests -- and the flagged display-
    # changing levers (decision depth 16 -> 12, features
    # 24 -> 16) stayed untouched. Measured: with-vision
    # 34,502B -> 25,859B on the live payload; plateau simulation
    # (Book #4 at caps) 40,929B -> 30,306B. Budget REBASELINED
    # to 31,000 on the no-vision build, basis: post-round-5 it
    # measures 25,122B and Book #4's fill-to-caps adds ~4.8KB
    # (decision_log +~1.9KB, dd +~1.0KB, fills +~0.7KB,
    # decisions +~0.6KB, equity +~0.6KB) for a ~29.9KB plateau;
    # vision adds ~0.9KB of gauge pairs, so the worst-case
    # pushed payload is ~30.8KB -- ~3.5KB under the largest
    # verified accepted push and ~4.6KB under the observed
    # 35,360B truncation. If this trips again the payload has
    # grown past the four-book plateau -- investigate before
    # reaching for the flagged display-changing levers.
    # Diet round 6 LANDED 2026-11-12 (Kass approved connecting
    # Book #5 to the master desk): the five-book no-vision build
    # measured 30,673B with Book #5 still nearly empty -- onto
    # this sentinel. Round 6 takes the two levers flagged since
    # round 3 (decision_log depth 16 -> 12, latest_thought
    # features 24 -> 16 -- the panel displays the strongest 12,
    # so no visible change) plus dd cap 32 -> 24 (worst case 26
    # points/book, ~16px/point at phone width). Measured:
    # no-vision 30,673B -> 27,296B. Budget UNCHANGED at 31,000,
    # basis: Book #5's fill-to-caps at the new caps adds ~2.5KB
    # for a ~29.8KB five-book plateau; vision adds ~1.1KB, so
    # the worst-case pushed payload is ~30.9KB -- ~3.4KB under
    # the largest verified accepted push and ~4.5KB under the
    # observed truncation.
    # Diet round 7 LANDED 2026-11-17 (Kass's order: stop running
    # into size limits, for good): the five-book build had crept
    # back over this sentinel (31,302B measured) as the books
    # matured toward the round-6 plateau estimate, so round 7
    # cuts the plateau itself, structurally: decision_log depth
    # 12 -> 8, latest_thought features 16 -> 12 (the panel
    # displays the strongest 12 -- invisible), dd cap 24 -> 20,
    # recent_fills / recent_decisions 12 -> 10, Book #3 vetoes
    # 10 -> 8, race_bh cap 32 -> 24. Every master section now
    # has a fixed count cap, so the plateau is a constant, not a
    # date: measured post-round-7 no-vision = 26,981B, and the
    # synthetic all-caps plateau (test_master_plateau_all_caps_
    # bounded below: five volatile books, full logs, max-length
    # values) stays under this sentinel with margin; vision adds
    # ~1.1KB of gauge pairs, keeping the worst-case pushed
    # payload ~5KB under the largest verified accepted push
    # (34.3KB) and ~6.5KB under the observed 35,360B truncation.
    # The budget stays 31,000 -- it was NOT raised.
    assert len(payload) < 31_000


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


def _sentinel_book(tmp_path, book_id, sentinel_p_up, vision_p_up):
    """One _master_book on a tmp book whose OWN state file records
    `sentinel_p_up` as its last_p_up (the value its loop wrote,
    == its latest decision-log p_up), with a vision_fn that
    returns a DIFFERENT (conflicting) recomputed value."""
    d = tmp_path / book_id
    d.mkdir()
    db = d / "paper.db"
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
    state = d / "state.json"
    state.write_text(json.dumps({
        "last_p_up": sentinel_p_up, "last_ts": "2026-10-31T02:00:00",
        "last_stance": "hold", "last_entry_price": None}))
    gauges = [{"name": "sentinel_gauge", "value": 42.0}]
    return compact._master_book(
        book_id, db, state,
        {"brain_run": 1, "brain_label": "PPO · Run 1", "eyes_run": 2,
         "eyes_label": "Stacker · Run 2", "feature_set": "prod21",
         "indicators_n": 21, "gate_run": None, "gate_threshold": None},
        kind="driver", label=f"Book {book_id}", start_equity=100000.0,
        include_vision=True,
        vision_fn=lambda: (vision_p_up, "2099-01-01T00:00:00", gauges))


def test_master_book_p_up_is_own_state_not_recomputed_vision(tmp_path):
    """REGRESSION (2026-11-03): _master_book overwrote each book's
    p_up with the freshly RECOMPUTED vision value. Books #2/#3
    share identical run-73 eyes, so the recomputation returns the
    same value for both -- and, built a candle ahead of a book
    whose loop had not processed that candle yet, it made Book
    #2's master headline carry Book #3's recorded vote instead of
    Book #2's own (saved payload 2026-11-03_1307: book2 p_up
    0.4875 == book3 latest_thought 0.4875 @17:00, while book2's
    own latest_thought was 0.4906 @16:00). Each book's master
    p_up must be sourced from THAT book's own state/decision log
    -- distinct sentinels per book, one conflicting vision value
    for all: on the buggy code every book below reads 0.9999."""
    sentinels = {"book1": 0.1111, "book2": 0.2222, "book3": 0.3333}
    for book_id, sentinel in sentinels.items():
        book = _sentinel_book(tmp_path, book_id, sentinel, 0.9999)
        assert book["p_up"] == sentinel, book_id
        # vision_ts belongs to the same recorded vote as p_up.
        assert book["vision_ts"] == "2026-10-31T02:00:00", book_id
        # Vision still supplies its gauges (display garnish) --
        # as diet-round-5 positional [name, value] pairs.
        assert book["top_gauges"] == [
            ["sentinel_gauge", 42.0]], book_id


def test_master_p_up_matches_each_books_own_decision_log():
    """Integration pin: in a real (no-vision) master build, every
    book's headline p_up equals the p_up that book's OWN decision
    log recorded for the book's vision_ts candle -- in
    particular Book #2's equals book2.jsonl's record, never a
    value sourced from another book's file."""
    m = _master()
    for b in m["books"]:
        path = compact.DECISIONS_DIR / f"{b['id']}.jsonl"
        records = [json.loads(line) for line in
                   path.read_text().splitlines() if line.strip()]
        by_ts = {r["ts"]: r for r in records}
        rec_ts = b["vision_ts"]
        if b["id"] == "book5":
            # Book #5's vote-recording stamp follows its decision
            # mode (trading.live_champ.DECISION_MODE). Close mode
            # (the production default again since Kass's
            # 2026-11-17 revert): the vote computed ON candle T is
            # recorded under T itself. Open-decision mode
            # (2026-11-13 -> 2026-11-17): the vote computed ON
            # candle T was recorded under the decision it drove --
            # the log record stamped T+1h, the candle whose open
            # filled at that vote. The primary stamp is the current
            # mode's; during the revert transition the state's last
            # vote may still have been written under the open
            # convention, so fall back to the other stamp when the
            # primary record is absent or carries a different
            # vote. Either way the headline p_up must equal the
            # book's OWN recorded vote exactly.
            from datetime import datetime, timedelta

            from src.trading.live_champ import DECISION_MODE
            plus1 = (datetime.fromisoformat(b["vision_ts"])
                     + timedelta(hours=1)).isoformat()
            primary, fallback = (
                (plus1, b["vision_ts"]) if DECISION_MODE == "open"
                else (b["vision_ts"], plus1))
            rec_ts = primary
            if (rec_ts not in by_ts
                    or by_ts[rec_ts]["p_up"] != b["p_up"]):
                if (fallback in by_ts
                        and by_ts[fallback]["p_up"] == b["p_up"]):
                    rec_ts = fallback
        assert rec_ts in by_ts, b["id"]
        assert b["p_up"] == by_ts[rec_ts]["p_up"], b["id"]
        if b["latest_thought"] is not None:
            assert b["p_up"] == b["latest_thought"]["p_up"], b["id"]


def test_master_dd_curve_flat_runs_compress_losslessly():
    """2026-11-02 diet round 2 (dd-curve stride, named in the
    sentinel comment above): the master dd_curve drops interior
    points of constant-value runs ONLY -- endpoints, every turning
    point, and min/max survive, so the rendered drawdown line is
    point-for-point identical. A fully flat curve (Book #3 at 0%
    since inception) collapses to its two endpoints."""
    compress = compact._compress_flat_runs
    flat = [[f"2026-11-0{d}T{h:02d}:00:00", 0] for d in (1, 2)
            for h in range(0, 24, 4)][:13]
    out = compress(flat)
    assert len(out) == 2
    assert out[0] == flat[0] and out[-1] == flat[-1]
    curve = [["t0", 0], ["t1", 0], ["t2", 0], ["t3", -0.5],
             ["t4", -0.5], ["t5", -1.25], ["t6", -0.5],
             ["t7", -0.5], ["t8", -0.5], ["t9", 0]]
    out = compress(curve)
    vals = [p[1] for p in out]
    assert min(vals) == -1.25 and max(vals) == 0
    assert out[0] == curve[0] and out[-1] == curve[-1]
    # Step reconstruction: every original point's value equals the
    # last compressed point at or before it -- the drawn line cannot
    # change anywhere.
    j = 0
    for ts, v in curve:
        while j + 1 < len(out) and out[j + 1][0] <= ts:
            j += 1
        assert out[j][1] == v, (ts, v, out)


# ------------------------------------------- shared master fetch (2026-11-04)

def _counting_market(monkeypatch):
    """Patch data.market_data.fetch_klines with a counting fake:
    every (symbol, interval, period) call is tallied and answered
    with a fresh copy of one synthetic 1600-candle frame (the size
    test_vision_raws proves warms all 21 production features and
    the drift block). Returns the Counter. All of compact's vision
    paths import fetch_klines from the module at call time, so the
    patch reaches every one of them."""
    import data.market_data as md

    counts = Counter()
    frame = _ohlcv(1600)

    def fake_fetch(symbol, interval="1h", period="730d"):
        counts[(symbol, interval, period)] += 1
        return frame.copy()

    monkeypatch.setattr(md, "fetch_klines", fake_fetch)
    return counts


def _fake_scored_book(monkeypatch, seat_fn_name, builder_fn_name,
                      meta_rel):
    """Stub one scored book's seat stacker + frame-builder factory
    (the test_vision_own_candle pattern): a constant-score frame
    over the seat's recorded columns and a fixed-vote model. The
    vision builders themselves -- and every market fetch they
    make -- run for real."""
    meta = json.loads((ROOT / meta_rel).read_text())
    columns = list(meta["feature_columns"])

    def fake_builder(df):
        frame = pd.DataFrame(63.2, index=df.index, columns=columns)
        return frame, list(columns)

    class _Vote:
        def predict_proba(self, X):
            return np.array([[0.4877, 0.5123]])

    monkeypatch.setattr(compact, seat_fn_name, lambda: (_Vote(), meta))
    monkeypatch.setattr(compact, builder_fn_name, lambda: fake_builder)


def _mock_master_market(monkeypatch):
    counts = _counting_market(monkeypatch)
    _fake_scored_book(monkeypatch, "_shadow_seat_stacker",
                      "_shadow_frame_builder",
                      "models/shadow_seat/stacker.json")
    _fake_scored_book(monkeypatch, "_gated_seat_stacker",
                      "_gated_frame_builder",
                      "models/gated_seat/stacker.json")
    return counts


def test_master_build_fetches_730d_series_once(monkeypatch):
    """PERF REGRESSION (2026-11-04): build_master used to fetch the
    730d BTC series THREE times per build -- once inside Book #1's
    vision via _drift_block, plus once per scored book because each
    _master_vision_scored call minted its own memo cache. One fetch
    is 59 sequential chunk requests; at the ~3.6s/request proxy
    latency measured that morning the build took ~676s (~97%
    network wait) and the 15-minute master-feed job timed out three
    runs in a row. The master build must thread ONE memoized fetch
    through all three vision paths: exactly ONE 730d fetch per
    build, however many books read the series."""
    counts = _mock_master_market(monkeypatch)
    m = compact.build_master(include_vision=True)
    assert counts[("BTC-USD", "1h", "730d")] == 1, dict(counts)
    # The shared fetch must not starve any book: every book still
    # gets its vision gauges.
    for b in m["books"]:
        assert b["top_gauges"], b["id"]


def test_master_shared_fetch_matches_unshared_vision(monkeypatch):
    """Output equivalence for the shared fetch: with identical
    mocked market data, the master build's per-book values must
    equal the UNSHARED path's -- each vision helper called on its
    own (own fetch, own memo cache), and each book's equity/p_up
    read from its own snapshot in the same process. Sharing a
    fetch may change WHEN candles arrive, never WHAT a book
    reports (consumers read closed candles, iloc[:-1])."""
    _mock_master_market(monkeypatch)
    m = compact.build_master(include_vision=True)
    refs = [
        compact._master_vision_book1(),
        # Book #2's eyes are ha10 since Champion #11 took the
        # shadow seat (2026-11-09): its master vision path is
        # the HA one, like Book #4's.
        compact._master_vision_ha(compact.build_shadow_vision),
        compact._master_vision_scored(compact.build_gated_vision,
                                      compact._gated_frame_builder),
    ]
    paths = [(None, None),
             (compact.SHADOW_DB, compact.SHADOW_STATE),
             (compact.GATED_DB, compact.GATED_STATE)]
    for book, ref, (db, st) in zip(m["books"], refs, paths):
        # Diet round 5: the feed ships gauges as [name, value]
        # pairs; expand before comparing with the vision's dicts.
        assert _expand_rows(
            book["top_gauges"], compact.MASTER_GAUGE_COLS) == ref[2], \
            book["id"]
        s = (snapshot.build_trading_snapshot() if db is None else
             snapshot.build_trading_snapshot(db_path=db, state_path=st))
        assert book["equity"] == s["equity"], book["id"]
        assert book["p_up"] == (round(s["stacker_p_up"], 4)
                                if s["stacker_p_up"] is not None
                                else None), book["id"]


def test_master_dd_curve_stride_diet_keeps_points_verbatim(tmp_path):
    """2026-11-05 diet round 3 (dd-curve stride -- the lever the
    sentinel comment named on 2026-11-01): the master dd_curve
    cap drops 64 -> 40, the module's own race-curve cap
    (_downsample_pairs default). Every dieted point must be a
    VERBATIM point of the flat-run-compressed series (the
    lossless round-2 stage) built from the same snapshot: same
    first and last points, the newest 8 points at full
    resolution, no point invented, moved, or revalued, and the
    result strictly shorter than the round-2 (cap-64) curve --
    the same stride class the 120 -> 64 cut already shipped."""
    import math

    d = tmp_path / "bookD"
    d.mkdir()
    db = d / "paper.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT, "
                "side TEXT, price REAL, qty REAL, fee REAL, "
                "cash_after REAL, btc_after REAL, reason TEXT, "
                "context TEXT)")
    con.execute("CREATE TABLE equity (ts TEXT, equity REAL, cash REAL, "
                "btc REAL, price REAL)")
    # 240 hourly rows of sawtooth equity, so the drawdown curve
    # has long non-flat runs (flat runs alone would compress
    # away and never reach the stride).
    for i in range(240):
        eq = 100000.0 + 900.0 * math.sin(i / 9.0) - 6.0 * i
        con.execute("INSERT INTO equity VALUES (?, ?, ?, 0.0, ?)",
                    (f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00",
                     round(eq, 2), round(eq, 2), 80000.0))
    con.commit()
    con.close()
    state = d / "state.json"
    state.write_text(json.dumps({
        "last_p_up": 0.5, "last_ts": "2026-10-10T23:00:00",
        "last_stance": "hold", "last_entry_price": None}))
    s = snapshot.build_trading_snapshot(db_path=db, state_path=state)
    raw_dd = [[p["t"], p["dd_pct"]] for p in s["drawdown_curve"]]
    assert len(raw_dd) > 64  # precondition: the stride engages
    compressed = compact._compress_flat_runs(raw_dd)
    round2 = compact._downsample_pairs(compressed, cap=64)
    book = compact._master_book(
        "bookD", db, state,
        {"brain_run": 1, "brain_label": "PPO · Run 1", "eyes_run": 2,
         "eyes_label": "Stacker · Run 2", "feature_set": "prod21",
         "indicators_n": 21, "gate_run": None, "gate_threshold": None},
        kind="driver", label="Book D", start_equity=100000.0,
        include_vision=False)
    dieted = book["dd_curve"]
    # Round-3 bound was <= 42 (cap 40); diet round 4 (2026-11-05)
    # tightened the master dd cap to 32 -> bound 34; diet round 6
    # (2026-11-12) cap 24 -> bound 26; diet round 7 (2026-11-17)
    # cap 20 -> bound 22.
    assert len(dieted) <= 22
    assert len(dieted) < len(round2)
    assert dieted[0] == compressed[0]
    assert dieted[-1] == compressed[-1]
    assert dieted[-8:] == compressed[-8:]
    compressed_set = {tuple(pt) for pt in compressed}
    assert all(tuple(pt) in compressed_set for pt in dieted)


# ------------------------------------------- diet round 4 (2026-11-05)

def _volatile_book(tmp_path, book_id="bookV", rows=720):
    """A synthetic book whose equity declines STRICTLY every row
    (varying decrements), so its drawdown deepens every candle:
    the dd series has NO flat runs at all -- the worst case for
    the round-2 flat-run compression, which then saves exactly
    zero bytes and the stride cap is the only size protection.
    Returns (book_dict, snapshot_dict)."""
    d = tmp_path / book_id
    d.mkdir()
    db = d / "paper.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT, "
                "side TEXT, price REAL, qty REAL, fee REAL, "
                "cash_after REAL, btc_after REAL, reason TEXT, "
                "context TEXT)")
    con.execute("CREATE TABLE equity (ts TEXT, equity REAL, cash REAL, "
                "btc REAL, price REAL)")
    eq = 100000.0
    for i in range(rows):
        if i:
            eq -= 13.7 + ((i * 7919) % 431) / 10.0  # 13.7..56.7, always > 0
        con.execute("INSERT INTO equity VALUES (?, ?, ?, 0.0, ?)",
                    (f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00",
                     round(eq, 2), round(eq, 2), 80000.0))
    con.commit()
    con.close()
    state = d / "state.json"
    state.write_text(json.dumps({
        "last_p_up": 0.5, "last_ts": "2026-10-30T23:00:00",
        "last_stance": "hold", "last_entry_price": None}))
    s = snapshot.build_trading_snapshot(db_path=db, state_path=state)
    book = compact._master_book(
        book_id, db, state,
        {"brain_run": 1, "brain_label": "PPO · Run 1", "eyes_run": 2,
         "eyes_label": "Stacker · Run 2", "feature_set": "prod21",
         "indicators_n": 21, "gate_run": None, "gate_threshold": None},
        kind="driver", label="Book V", start_equity=100000.0,
        include_vision=False)
    return book, s


def test_master_dd_curve_worst_case_volatile_series_bounded(tmp_path):
    """2026-11-05 diet round 4 -- WORST-CASE SIZE: BTC's hard drop
    on 2026-11-05 took Book #3's dd_curve from 917B (30 pts, book
    near its peak, flat runs compressing) to 1,193B (39 pts) in
    eight hours: under cap 40 the dd sections' size still rode on
    compression luck. Round 4 tightens the master dd cap 40 ->
    32 (~12px/point at phone width, past chart resolution; the
    stride class rounds 2-3 already shipped). This book is the
    adversarial case -- dd deepens EVERY candle, compression is
    the identity -- and the dieted curve must STILL land at the
    deterministic bound: <= 22 points (cap 20 + endpoint guards,
    diet round 7, 2026-11-17; the cap was 24 / bound 26 in round
    6), <= 22*31+2 serialized bytes (a [19-char ts, 3dp value]
    point costs at most 31B), every point a verbatim point of
    the RAW series, endpoints exact, newest 8 full-res, strictly
    shorter than the round-3 (cap-40) curve of the same series."""
    book, s = _volatile_book(tmp_path)
    raw_dd = [[p["t"], p["dd_pct"]] for p in s["drawdown_curve"]]
    # snapshot builds curves over its last-500 equity window, so
    # the 720 inserted rows arrive as 500 dd points -- still >14x
    # the cap, so the stride is fully engaged.
    assert len(raw_dd) == 500
    # Adversarial precondition: NO flat runs -- compression saves
    # nothing, so only the stride cap bounds the size.
    assert compact._compress_flat_runs(raw_dd) == raw_dd
    round3 = compact._downsample_pairs(raw_dd, cap=40)
    dieted = book["dd_curve"]
    assert len(dieted) <= 22
    assert len(dieted) < len(round3)
    blob = json.dumps(dieted, separators=(",", ":"))
    assert len(blob) <= 22 * 31 + 2
    assert dieted[0] == raw_dd[0]
    assert dieted[-1] == raw_dd[-1]
    assert dieted[-8:] == raw_dd[-8:]
    raw_set = {tuple(pt) for pt in raw_dd}
    assert all(tuple(pt) in raw_set for pt in dieted)


def test_master_equity_curve_worst_case_volatile_series_bounded(tmp_path):
    """Equity-curve invariant on the same no-flat-runs book:
    _downsample_curve already bounds the master equity curve
    deterministically (leading-flat collapse <= 2 pts + strided
    tail <= 8 + newest 8 full-res + final-point guard = <= 19
    points) -- it never depended on compression luck, which is
    why the 2026-11-05 swell measured +38B across all three
    books' equity curves while dd moved +332B. Pin the bound and
    the verbatim/endpoints invariants so a future 'simplification'
    of the curve path cannot silently unbound it."""
    book, s = _volatile_book(tmp_path)
    src = [[p["t"], round(p["equity"], 2)] for p in s["equity_curve"]]
    curve = book["equity_curve"]
    assert 0 < len(curve) <= 19
    assert curve[0] == src[0]
    assert curve[-1] == src[-1]
    assert curve[-1][1] == book["equity"]
    src_set = {tuple(pt) for pt in src}
    assert all(tuple(pt) in src_set for pt in curve)
    blob = json.dumps(curve, separators=(",", ":"))
    assert len(blob) <= 19 * 33 + 2


# --------------------------------- plateau growth test (2026-11-17)

def test_master_plateau_all_caps_bounded(tmp_path, monkeypatch):
    """PLATEAU, not today (2026-11-17 size fix, diet round 7):
    five synthetic books with EVERY master section at its cap --
    720-row strictly-declining equity (no flat runs for the dd
    compression to hide behind), 15 trades (fills/decisions cap
    at 10), 30 decision records per book (log caps at 8) whose
    newest carries 100 features (latest_thought caps at the
    strongest 12), one book carrying 8 vetoes -- must serialize
    UNDER the 31,000 sentinel. The point is the bound is a
    constant of the caps, not a function of the date: the same
    payload built after 2 more years of hourly rows differs only
    in timestamp digits, which are fixed-width. Vision is off;
    its top_gauges add ~1.1KB (<= 8 gauge pairs/book, pinned by
    the round-5 tests), keeping the worst-case pushed payload
    ~5KB under the largest verified accepted push (34.3KB,
    2026-11-01) and ~6.5KB under the observed 35,360B transport
    truncation."""
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    feats = {f"feat_{i:02d}": (i + 1) / 100.0 for i in range(100)}
    for b in range(1, 6):
        recs = []
        for i in range(30):
            recs.append({
                "ts": f"2026-11-{1 + i // 24:02d}T{i % 24:02d}:00:00",
                "close": 65432.10, "p_up": 0.4987, "stance": "hold",
                "action": 0, "taken": False, "reason": "hold",
                "gate_proba": None, "gate_threshold": None,
                "gate_verdict": None, "pos_before": -0.393754,
                "pos_after": -0.393754, "equity": 104254.99,
                "stop_level": 66551.05, "exposure_pct": 25.71,
                "features": feats if i == 29 else None})
        (dec_dir / f"book{b}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(compact, "DECISIONS_DIR", dec_dir)
    books = []
    for b in range(1, 6):
        book, _s = _volatile_book(tmp_path, book_id=f"book{b}")
        # _volatile_book's DB has no trades; add 15 so the fills
        # and decisions sections fill to their caps too.
        db = tmp_path / f"book{b}" / "paper.db"
        con = sqlite3.connect(db)
        for i in range(15):
            con.execute(
                "INSERT INTO trades (ts, side, price, qty, fee,"
                " cash_after, btc_after, reason, context) VALUES"
                " (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00",
                 "BUY" if i % 2 else "SELL", 65432.10, 0.393754,
                 38.23, 99217.14, -0.393754, "signal",
                 json.dumps({"p": 0.4987})))
        con.commit()
        con.close()
        vetoes = None
        if b == 3:
            vetoes = [{"ts": "2026-11-17T13:00:00", "proba": 0.4681,
                       "threshold": 0.55, "verdict": "VETOED"}
                      for _ in range(8)]
        books.append(compact._master_book(
            f"book{b}", db, tmp_path / f"book{b}" / "state.json",
            {"brain_run": 99, "brain_label": "PPO · Run 99",
             "eyes_run": 73, "eyes_label": "Stacker · Run 73",
             "feature_set": "prod21", "indicators_n": 21,
             "gate_run": 82 if b == 3 else None,
             "gate_threshold": 0.55 if b == 3 else None},
            kind="driver", label=f"Book #{b}",
            start_equity=100000.0, include_vision=False,
            vetoes=vetoes))
    for b in books:
        assert len(b["equity_curve"]) <= 19
        assert len(b["dd_curve"]) <= 22
        assert len(b["recent_fills"]) <= 10
        assert len(b["recent_decisions"]) <= 10
        assert len(b["decision_log"]) == 8
        assert len(b["latest_thought"]["features"]) == 12
        assert b["latest_thought"]["features_total"] == 100
    payload = json.dumps(
        {"books": books,
         "race_bh": [["2026-11-17T00:00:00", 65432.10]] * 25,
         "combined": {"equity": 508256.11, "pnl": 8256.11,
                      "exposure_usd": 128000.0, "books_in_profit": 4,
                      "books_in_drawdown": 1}},
        separators=(",", ":"))
    print(f"master plateau (all caps, no vision) bytes={len(payload)}")
    assert len(payload) < 31_000, (
        f"master plateau over the sentinel: {len(payload)}")
