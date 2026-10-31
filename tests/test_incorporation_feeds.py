"""Feed builders for the 2026-10-26 research incorporation:
friction/risk (Live ledger), PSI drift blocks (Live vision +
Shadow snapshot), per-fill attribution ranking (fillctx), the
Overfitting Auditor + telemetry blocks (Training), and the
Walk-Forward Efficiency block (Shadow race).

Friction math is asserted against HAND-COMPUTED values on a
synthetic ledger; the auditor is asserted against the real
registry's recorded facts (champions, declines, trial counts).
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dashboards import compact  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402


# ---------------------------------------------------------------
# Friction + risk (item 1)
# ---------------------------------------------------------------

def _toy_book(tmp_path):
    db = tmp_path / "paper.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, ts TEXT,"
                " side TEXT, price REAL, qty REAL, fee REAL,"
                " cash_after REAL, btc_after REAL, reason TEXT,"
                " context TEXT)")
    con.execute("CREATE TABLE equity(ts TEXT, equity REAL, cash REAL,"
                " btc REAL, price REAL)")
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    t2 = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    old = (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S")
    # Fill 1: price == decision close (drift 0). Fill 2: BUY filled
    # $1 above its decision close of $99 -> +101.01 bps, +$2.00 cost.
    # Fill 3 (40d old): no equity row at its ts -> excluded.
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, t1, "BUY", 100.0, 1.0, 10.0, 0, 0, "ppo_live", None),
        (2, t2, "BUY", 100.0, 2.0, 20.0, 0, 0, "ppo_live", None),
        (3, old, "SELL", 50.0, 1.0, 5.0, 0, 0, "ppo_live", None),
    ])
    con.executemany("INSERT INTO equity VALUES (?,?,?,?,?)", [
        (t1, 100000.0, 100000.0, 0.0, 100.0),
        (t2, 90000.0, 90000.0, -0.5, 99.0),
    ])
    con.commit()
    return con


def test_friction_hand_computed(tmp_path):
    con = _toy_book(tmp_path)
    out = compact._friction_and_risk(con)
    fr = out["friction"]
    assert fr["fees_all"] == 35.0
    assert fr["fees_30d"] == 30.0  # the 40-day-old fee is outside
    assert fr["n"] == 2 and fr["excl"] == 1
    assert fr["drift_bps"] == pytest.approx(50.51, abs=0.01)
    assert fr["drift_usd"] == pytest.approx(1.0, abs=1e-9)
    rk = out["risk"]
    assert rk["peak"] == 100000.0
    assert rk["floor_gap"] == -5000.0
    assert rk["floor_gap_pct"] == pytest.approx(-5.26, abs=0.01)
    assert rk["exp"] == pytest.approx(0.055, abs=0.01)
    assert rk["cap"] == 75
    con.close()


def test_ledger_shadow_shape_has_no_friction(tmp_path):
    _toy_book(tmp_path)
    out = compact.build_ledger(db_path=tmp_path / "paper.db")
    assert "friction" not in out and "risk" not in out
    assert out["ledger"] and out["closed"] == []


# ---------------------------------------------------------------
# Drift windows + blocks (item 4)
# ---------------------------------------------------------------

def test_drift_windows_anchor_to_grading():
    idx = pd.date_range("2026-01-01", periods=1200, freq="1h")
    feat = pd.DataFrame({"x": np.arange(1200.0)}, index=idx)
    base_end = idx[1000].strftime("%Y-%m-%dT%H:%M:%S")
    base, live = compact._drift_windows(feat, base_end)
    assert len(live) == 168 and live.index[-1] == idx[-1]
    assert len(base) == 792 and base.index[-1] == idx[1000]
    base2, _ = compact._drift_windows(feat, None)
    assert len(base2) == 792 and base2.index[-1] == idx[1031]


def _fake_raw(n=1500, seed=5):
    idx = pd.date_range("2025-09-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    close = 80000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)


def test_drift_block_shape_with_fake_fetch():
    raw = _fake_raw()
    block = compact._drift_block(fetch=lambda *a, **k: raw)
    assert block is not None
    assert len(block["psi"]) == 21
    assert sum(block["bands"]) == 21
    assert block["live_h"] == 168 and block["base_h"] == 792
    assert all(v is None or v >= 0 for v in block["psi"])


def test_shadow_drift_block_worst_first():
    raw = _fake_raw()
    _model, meta = compact._shadow_seat_stacker()
    cols = list(meta["feature_columns"])
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {c: rng.normal(50, 10, len(raw)) for c in cols}, index=raw.index)
    # One column's live window jumps two sigma: it must top worst.
    frame.iloc[-168:, 0] += 40.0
    block = compact._shadow_drift_block(
        fetch=lambda *a, **k: raw,
        scored_builder=lambda r: (frame, cols))
    assert block is not None
    assert block["n"] == len(cols)
    assert sum(block["bands"]) + block["nb"] == len(cols)
    assert len(block["worst"]) == 10
    psis = [v for _c, v in block["worst"]]
    assert psis == sorted(psis, reverse=True)
    assert block["worst"][0][0] == cols[0]


# ---------------------------------------------------------------
# Fill attribution (item 5)
# ---------------------------------------------------------------

def test_seat_importances_are_a_distribution():
    imp = compact._seat_stacker_importances()
    assert imp is not None and len(imp) == 21
    assert all(v >= 0 for v in imp)
    assert sum(imp) == pytest.approx(1.0, abs=1e-6)


def test_fill_top_ranks_by_value_times_importance():
    imp = [0.01] * 21
    imp[3] = 0.5   # the stacker leans hard on feature 3
    imp[7] = 0.3
    f = [0.1] * 21
    f[3] = 0.9     # strong value x strong importance -> rank 1, up
    f[7] = -0.8    # strong x important, negative -> rank 2, down
    f[0] = 1.0     # strongest raw value, tiny importance -> rank 3
    top = compact._fill_top(f, imp, None, "2026-10-21T14:00:00")
    assert [t[0] for t in top[:3]] == [
        FEATURE_COLUMNS[3], FEATURE_COLUMNS[7], FEATURE_COLUMNS[0]]
    assert top[0][3] == 1 and top[1][3] == -1
    assert all(t[2] is None for t in top)  # no raw frame -> no units


# ---------------------------------------------------------------
# Auditor + telemetry (items 3 and 6)
# ---------------------------------------------------------------

def test_audit_matches_registry_facts():
    runs = compact._all_runs_desc()
    a = compact.build_audit(runs)
    assert a["n"] == len(runs)
    assert a["ppo"][0] + a["stk"][0] == len(runs)
    # Recorded facts (2026-10-26 registry): 3 stacker declines
    # (runs 61/63/67); no stacker pending -- run 69's pending was
    # resolved the same evening when Kass approved it and it was
    # crowned lab stacker Champion #3.
    # RE-PINNED 2026-10-27 (Muse's Pick #2): run 73 (Pick #2
    # stacker) beat the champion and pends for Kass, so one
    # stacker is pending again, and best-of-the-rest is run 73's
    # 0.5354 -- ABOVE the champion's 0.5265 while it pends, which
    # is exactly the luckiest-draw context the auditor exists to
    # show. Declines unchanged (61/63/67).
    # RE-PINNED again 2026-10-27 (Champion #4 crowning): Kass
    # delegated the ruling to Muse, who crowned run 73 lab
    # stacker Champion #4 and resolved its pending as approved
    # on the record -- no stacker pends, the champion IS run 73
    # at 0.5354, and best-of-the-rest is now former champion run
    # 69's 0.5265.
    assert a["stk"][2] == 3 and a["ppo"][2] == 0
    assert a["ppo"][3] == 0 and a["stk"][3] == 0
    # Champions: stacker #4 is run 73 at 0.5354; PPO champion run 44
    # dueled -615.6. Best-of-the-rest stacker (post-fix) is run
    # 69's 0.5265 (the displaced Champion #3).
    assert a["cs"][0] == 73 and a["cs"][1] == pytest.approx(0.5354)
    assert a["cs"][3] == pytest.approx(0.5265)
    assert a["cs"][2] < a["cs"][3]  # median below the luckiest draw
    assert a["cp"][0] == 44 and a["cp"][1] == pytest.approx(-615.6)


def test_telemetry_block_reads_records_verbatim():
    runs = [
        {"id": 3, "kind": "ppo", "promoted": False,
         "metrics": {"telemetry": {"ent_start": 1.3, "ent_end": 0.9,
                                   "ent_min": 0.8, "kl_end": 0.01,
                                   "clip_end": 0.2, "ev_end": 0.5,
                                   "vl_end": 0.3}}},
        {"id": 2, "kind": "ppo", "promoted": False,
         "metrics": {"telemetry": None}},
        {"id": 1, "kind": "stacker", "promoted": False, "metrics": {}},
    ]
    out = compact._telemetry_block(runs)
    assert out["from"] == 3
    assert out["rows"] == [[3, 1.3, 0.9, 0.8, 0.01, 0.2, 0.5, 0.3]]
    empty = compact._telemetry_block(runs[1:])
    assert empty == {"from": None, "rows": []}


def test_training_feed_carries_audit_and_telemetry():
    feed = compact.build_compact_training()
    assert feed["audit"]["n"] == len(compact._all_runs_desc())
    # Telemetry is carried verbatim from the run records -- never
    # reconstructed. Recorded fact (2026-10-26): run 70 (Muse's
    # Pick PPO) is the FIRST run ever to carry a telemetry record
    # (the capture shipped that day); every older run reads "Not
    # recorded" and stays out of the block. Updated 2026-10-27:
    # the midnight retrain's run 72 is the first NIGHTLY run with
    # telemetry, so the block now carries runs 72 and 70 (newest
    # first); "from" stays 70, the first run that ever recorded.
    # Updated again 2026-10-27 (Muse's Pick #2): run 74 (Pick #2
    # PPO, telemetry capture on per the pre-registration) joins
    # the block, so it now carries runs 74, 72 and 70.
    # Updated again 2026-10-27 (Muse's Pick #3): run 75 (Pick #3
    # PPO, telemetry on per its pre-registration) joins the block,
    # so it now carries runs 75, 74, 72 and 70.
    keys = ("ent_start", "ent_end", "ent_min", "kl_end", "clip_end",
            "ev_end", "vl_end")

    def _row(rid):
        rec = next(r for r in compact._all_runs_desc()
                   if r["id"] == rid)
        t = rec["metrics"]["telemetry"]
        return [rid] + [
            round(t[k], 4) if isinstance(t.get(k), (int, float))
            else None for k in keys]

    assert feed["telemetry"] == {"from": 70,
                                  "rows": [_row(75), _row(74), _row(72),
                                           _row(70)]}
    import json
    assert len(json.dumps(feed, separators=(",", ":"))) < 16384


# ---------------------------------------------------------------
# WFE (item 2)
# ---------------------------------------------------------------

def _wfe_setup(tmp_path, monkeypatch, started, fills):
    db = tmp_path / "paper_shadow.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, ts TEXT,"
                " side TEXT, price REAL, qty REAL, fee REAL,"
                " cash_after REAL, btc_after REAL, reason TEXT,"
                " context TEXT)")
    for i in range(fills):
        con.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (i + 1, "2026-10-26T02:00:00", "BUY", 100.0, 0.1,
                     1.0, 0, 0, "ppo_live", None))
    con.commit()
    con.close()
    monkeypatch.setattr(compact, "SHADOW_DB", db)
    monkeypatch.setattr(compact, "_all_runs_desc", lambda: [{
        "id": 66, "ts": "2026-10-25T22:00:00", "kind": "ppo",
        "promoted": False,
        "metrics": {"family": "scored_eyes_ppo",
                    "champion": {"score": 39.76, "total_pnl": 461.81}}}])
    return {"started": started, "starting_balance": 100000}


def test_wfe_early_days_ratio_is_none(tmp_path, monkeypatch):
    man = _wfe_setup(tmp_path, monkeypatch, "2026-10-26T01:00:00", 0)
    w = compact._wfe_block(man, {"pnl": 0.0, "pct": 0.0})
    assert w["exam_pnl"] == 461.81 and w["exam_pct"] == 0.4618
    assert w["exam_days"] == 33
    assert w["fills"] == 0 and w["ratio"] is None


def test_wfe_ratio_once_mature(tmp_path, monkeypatch):
    started = (datetime.now(timezone.utc)
               - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    man = _wfe_setup(tmp_path, monkeypatch, started, 3)
    w = compact._wfe_block(man, {"pnl": 230.9, "pct": 0.2309})
    assert w["fills"] == 3 and w["days"] >= 29.9
    assert w["ratio"] == pytest.approx(0.2309 / 0.4618, abs=0.002)


def test_wfe_exam_of_previous_pairing_is_reference_only(
        tmp_path, monkeypatch):
    """Kass seated Champion #3's eyes (run 69) in the shadow desk
    2026-10-26. The registry's scored-eyes exam graded the PREVIOUS
    pairing (PPO 44 + run 64 eyes, recorded on the manifest as
    wfe_exam at the swap): the current pairing (44 + 69) has NO
    graded exam, so the exam fields go null and the old exam rides
    along as a labeled previous-pairing reference only."""
    man = _wfe_setup(tmp_path, monkeypatch, "2026-10-26T01:00:00", 0)
    man["driver"] = {"brain": "PPO", "run_id": 44}
    man["eyes"] = {"brain": "Stacker", "run_id": 69,
                   "champion": "Champion #3"}
    man["wfe_exam"] = {"driver_run": 44, "eyes_run": 64}
    w = compact._wfe_block(man, {"pnl": 0.0, "pct": 0.0})
    assert w["pair"] == [44, 69]
    assert w["exam_pnl"] is None and w["exam_pct"] is None
    assert w["exam_days"] is None
    assert w["pexam"] == {"driver": 44, "eyes": 64, "pnl": 461.81,
                          "pct": 0.4618, "days": 33}
    assert w["fills"] == 0 and w["ratio"] is None


def test_wfe_pair_defaults_to_the_original_pairing(
        tmp_path, monkeypatch):
    """A manifest with no recorded exam attribution (the pre-swap
    shape) attributes the registry exam to its own pairing."""
    man = _wfe_setup(tmp_path, monkeypatch, "2026-10-26T01:00:00", 0)
    w = compact._wfe_block(man, {"pnl": 0.0, "pct": 0.0})
    assert w["pair"] == [44, 64]
    assert w["pexam"] is None
