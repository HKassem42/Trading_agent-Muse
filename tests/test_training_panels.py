"""Training-dashboard upgrade checks (2026-10-25): the training feed
gains a Current Recipe panel (the live training configuration, read
from the code itself so it cannot silently drift), a Gate History
strip (the last 6 recorded PPO gate duels), and per-run settings on
the detail rows (steps/deep/family/config notes the registry actually
recorded). Expectations here are derived INDEPENDENTLY of compact.py's
own derivations -- from the source files and training.db directly --
so a drift on either side fails."""
import inspect
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact  # noqa: E402


# --- Current Recipe panel -------------------------------------------------

def test_recipe_panel_matches_code_constants_two_ways():
    t = compact.build_compact_training()
    assert "recipe" in t, "training feed carries no recipe panel"
    r = t["recipe"]

    # Way 1: the modules' own constants, imported directly.
    from features.technical_indicators import FEATURE_COLUMNS
    from models import ppo_train
    from trading.broker import FEE_RATE, PaperBroker

    assert r["feat_n"] == len(FEATURE_COLUMNS) == 21
    assert r["fee"] == FEE_RATE == 0.0015
    assert r["chunk"] == ppo_train.CHUNK_SIZE == 25_000
    assert r["stop"] == [PaperBroker.STOP_ATR_MULT,
                         PaperBroker.STOP_MIN_PCT,
                         PaperBroker.STOP_MAX_PCT] == [2.0, 0.025, 0.08]
    assert r["trail"] == [PaperBroker.TRAIL_ACTIVATE_PCT,
                          PaperBroker.TRAIL_ATR_MULT,
                          PaperBroker.TRAIL_MIN_PCT,
                          PaperBroker.TRAIL_MAX_PCT] == [0.02, 1.5, 0.02, 0.06]  # activation 2% per Kass 2026-11-02 (was 3%)
    sig = inspect.signature(PaperBroker.__init__).parameters
    assert r["tranche"] == sig["max_position_pct"].default == 0.25
    assert r["exposure"] == sig["max_exposure_pct"].default == 0.75

    # Way 2: the test's own parse of nightly.py (the step counts, warm
    # start, holdout geometry, gate + floor numbers live only there --
    # as literals inside main()/gate_decision(), not importable
    # constants -- so both sides parse the file independently).
    src = (ROOT / "src" / "training" / "nightly.py").read_text()
    m = re.search(r"timesteps = (\d[\d_]*) if deep else (\d[\d_]*)", src)
    assert r["deep_steps"] == int(m.group(1).replace("_", "")) == 300_000
    assert r["steps"] == int(m.group(2).replace("_", "")) == 60_000
    assert r["warm"] == (1 if re.search(
        r"train_ppo\(\s*timesteps=timesteps, warm_start=True", src) else 0)
    assert r["warm"] == 1
    m = re.search(r"split = len\(sup\) - (\d+)", src)
    assert r["holdout_d"] == int(m.group(1)) // 24 == 33
    m = re.search(r"windows_sup = \[hold_sup\.iloc\[k \* \d+:\(k \+ 1\) "
                  r"\* \d+\] for k in range\((\d+)\)\]", src)
    assert r["windows"] == int(m.group(1)) == 3
    m = re.search(r"def evaluate_ppo\(model_path: Path, sup, p_up, "
                  r"episodes: int = (\d+),\s*feature_columns=None\)", src)
    assert r["episodes"] == int(m.group(1)) == 10
    assert r["fills"] == re.search(r'fill_mode="(\w+)"', src).group(1)
    assert r["fills"] == "next_open"
    m = re.search(r'"score": total - ([\d.]+) \* max_dd', src)
    assert r["dd_pen"] == float(m.group(1)) == 0.5
    m = re.search(r'and wins >= (\d+)\)', src)
    assert r["gate_wins"] == int(m.group(1)) == 2
    # The two-part floor, exactly as gate_decision() enforces it:
    # non-hold share floor, P&L strictly > $0, P&L strictly > buy&hold.
    m = re.search(r"if non_hold < ([\d.]+):", src)
    assert r["floor"] == [float(m.group(1)), 0, 1] == [0.1, 0, 1]
    assert re.search(r"if pnl <= 0\.0:", src)
    assert re.search(r"if pnl <= bench:", src)
    # Stacker promotion: strict holdout win, nothing softer.
    assert re.search(r"stacker_promoted = chall_acc > champ_acc", src)
    assert r["stacker"] == "strict_win"
    # Feature-set version: no FEATURE_VERSION constant exists in code;
    # the feed must carry the version the production lineage attests
    # (nightly run records, runs 16-44 era: the v5.1 recipe). The
    # naive all-runs walk is a trap -- run 45's record attests "3"
    # because its STUDY was named "V3 composite screen", the
    # features-v6 lab runs 61/62 (2026-10-25) attest "6" the same way
    # ("features v6" in their reason text), and run 85 (2026-10-30)
    # attests "2" ("Eyes v2") -- being newest it now wins the naive
    # walk; no study label must ever name the feature set.
    # Both halves are pinned here so removing the lineage restriction
    # fails loudly.
    assert r["feat_v"] == "5.1"
    registry = compact._all_runs_desc()
    notes = {n["ts"]: n for n in compact._lab_notes_raw()}
    naive = None
    for run in reversed(registry):
        own = compact._pick_version(
            compact._version_candidates(run, notes.get(run["ts"])))
        if own:
            naive = own
    assert naive == "2"  # the study-label pollution, documented above
    assert r["feat_v"] != naive


# --- Gate History strip ---------------------------------------------------

def _db_nightly_duels():
    """The recorded gate duels, derived straight from training.db:
    nightly PPO runs (no lab family) where BOTH sides recorded a real
    score (migration-seated champions carry a -1e9 sentinel and never
    dueled), newest first."""
    con = sqlite3.connect(ROOT / "data" / "training.db")
    con.row_factory = sqlite3.Row
    out = []
    for row in con.execute("SELECT * FROM runs ORDER BY id DESC"):
        m = json.loads(row["metrics"])
        if row["kind"] != "ppo" or m.get("family"):
            continue
        c, h = m.get("challenger") or {}, m.get("champion") or {}
        if not (isinstance(c.get("score"), (int, float))
                and isinstance(h.get("score"), (int, float))
                and c["score"] > -1e8 and h["score"] > -1e8):
            continue
        out.append((row["id"], row["ts"], bool(row["promoted"]), m))
    con.close()
    return out


def test_gate_history_matches_recorded_duels():
    t = compact.build_compact_training()
    assert "gate_hist" in t, "training feed carries no gate history"
    hist = t["gate_hist"]
    duels = _db_nightly_duels()
    assert len(duels) >= 6
    expect = duels[:6]
    # Population: the last 6 recorded duels, newest first.
    assert [row[0] for row in hist] == [d[0] for d in expect]
    for row, (rid, ts, promoted, m) in zip(hist, expect):
        c, h = m["challenger"], m["champion"]
        assert row[1] == ts[5:10]
        # Windows won, derived from the recorded window scores the
        # same way the gate feed derives them (1dp zip, strict >).
        cw = [round(w["score"], 1) for w in c["windows"]]
        hw = [round(w["score"], 1) for w in h["windows"]]
        won = sum(1 for a, b in zip(cw, hw) if a > b)
        assert row[2] == won
        # Aggregates at gate-feed precision; P&Ls in whole dollars.
        assert row[3] == round(c["score"], 1)
        assert row[4] == round(c["total_pnl"])
        assert row[5] == round(h["score"], 1)
        assert row[6] == round(h["total_pnl"])
        assert row[7] == round(c["bench_pnl"])
        fpass = 1 if (c["score"] > h["score"] and won >= 2) else 0
        assert row[8] == fpass
        # Floor column: only reached when the formula passed; then
        # the recorded promotion flag IS the floor's verdict (nightly
        # promotes only through gate_decision). Formula fail -> null.
        if fpass:
            assert row[9] == (1 if promoted else 0)
        else:
            assert row[9] is None
        # The verdict text is the RECORDED outcome, never recomputed.
        assert row[10] == ("PROMOTED" if promoted else "NOT PROMOTED")
    # The one floor-blocked duel in history: run 44 passed the
    # formula, the floor stopped it, and the recorded floor reason
    # rides along verbatim from the run's own recorded reason.
    # RE-PINNED 2026-11-21: run 44 rolled out of the 6-row strip
    # long ago (the strip now starts at run 147), so the row is
    # verified by running the same builder over run 44's own
    # record instead of fishing it out of the strip -- the
    # floor-reason extraction is what this block pins, not strip
    # membership. Run 44 remains the only floor-blocked duel in
    # the registry (checked 2026-11-21).
    r44_rec = next(r for r in compact._all_runs_desc()
                   if r["id"] == 44)
    r44 = compact._gate_history([r44_rec])[0]
    assert r44[8] == 1 and r44[9] == 0 and r44[10] == "NOT PROMOTED"
    assert r44[11] == "P&L -$650 ≤ $0"
    # Consistency with the live gate feed: the strip's newest row is
    # the same duel the gate chip shows, with the same verdict.
    g = compact.build_gate()
    assert hist[0][0] == g["chall"]["i"]
    assert hist[0][2] == g["duel"]["won"]
    assert hist[0][8] == g["duel"]["fpass"]
    assert hist[0][10] == g["duel"]["rec"]


# --- Models index naming: production lineage only ---------------------------
# The index's version walk inherits a version forward from the newest
# run that attests one. Lab-family records attest STUDY labels, not
# feature versions -- run 45's record attests "3" because its study was
# NAMED "V3 composite screen" -- and letting one into the walk renamed
# every later nightly run "PPO v3" (live defect, 2026-10-25: runs
# 57-60). The walk must mirror _current_feat_version's restriction:
# production lineage (runs with no lab `family`) only. Lab runs keep
# their own study names, which never read the inherited version.

_FIXTURE_DUEL = {"score": -100.0, "total_pnl": -50.0,
                 "max_drawdown": 10.0, "mean_reward": -0.5,
                 "bench_pnl": 0.0, "windows": [{"score": -1.0}]}


def _fixture_run(rid, ts, kind, metrics):
    return {"id": rid, "ts": ts, "kind": kind, "metrics": metrics,
            "promoted": False}


def test_models_index_names_come_from_production_lineage_only():
    runs_desc = [  # newest first, as _all_runs_desc returns
        _fixture_run(3, "2026-11-03T04:00:00+00:00", "ppo",
                     {"challenger": dict(_FIXTURE_DUEL),
                      "champion": dict(_FIXTURE_DUEL)}),
        _fixture_run(2, "2026-11-02T05:00:00+00:00", "ppo",
                     {"family": "composite_screen",
                      "composite_screen": {"label": "Rule Master"},
                      # A study-label attestation on the run's own
                      # record (the real run 45 attests "3" the same
                      # way, via its lab note "Matrix v3 composite
                      # screen" -- the study's NAME, not a feature
                      # version).
                      "features": "v3 matrix composites",
                      "challenger": dict(_FIXTURE_DUEL),
                      "champion": dict(_FIXTURE_DUEL)}),
        _fixture_run(1, "2026-11-01T13:00:00+00:00", "ppo",
                     {"reason": "Features v5.1 migration",
                      "challenger": dict(_FIXTURE_DUEL),
                      "champion": dict(_FIXTURE_DUEL)}),
    ]
    names = {row[0]: row[2] for row in compact._models_index(runs_desc)}
    assert names[1] == "PPO v5.1"
    # The lab run keeps its own study name...
    assert names[2] == "Composite Screen — Rule Master"
    # ...and its "V3" study label must NOT rename the nightly run
    # that attests nothing of its own (the live defect).
    assert names[3] == "PPO v5.1"


def test_models_index_live_registry_names_current_lineage():
    # The index ships in push_models_page pages since 2026-11-17;
    # assemble it the way the desk does.
    t = compact.build_compact_training()
    names = {m[0]: m[2] for p in compact.build_models_pages()
             for m in p["rows"]}
    # The newest nightly runs (57-60) ran the v5.1 recipe; the "V3
    # composite screen" study label must not rename them.
    assert names[57] == "Stacker v5.1" and names[59] == "Stacker v5.1"
    assert names[58] == "PPO v5.1" and names[60] == "PPO v5.1"
    # Study runs keep their own study names. Run 45 left the
    # shipped feed index under Kass's 2026-11-05 trim (failed,
    # id <= 50), so its name is asserted on the naming layer,
    # which stays complete.
    full_names = {row[0]: row[2] for row in
                  compact._models_index(compact._all_runs_desc())}
    assert full_names[45] == "Composite Screen — Rule Master"
    assert names[56].startswith("Two-Step Hierarchy")
    # The index's newest nightly name agrees with the recipe panel's
    # feature version (both read the production lineage).
    assert names[60] == f"PPO v{t['recipe']['feat_v']}"


# --- Run detail: recorded settings surface --------------------------------

def test_detail_rows_surface_recorded_settings():
    t = compact.build_compact_training()
    by_id = {r["i"]: r for r in t["runs"]}
    # Expectations derive from the registry itself (the detail
    # window is the newest 4 runs and rolls as the registry grows --
    # id-pinned assertions broke twice that way): every windowed run
    # that RECORDED training steps + a deep flag must surface them
    # verbatim on its detail row; stacker runs record no step count,
    # so the field must be ABSENT there, never fabricated.
    con = sqlite3.connect(ROOT / "data" / "training.db")
    rows = con.execute(
        "SELECT id, kind, metrics FROM runs ORDER BY id DESC "
        "LIMIT 4").fetchall()
    con.close()
    assert [r["i"] for r in t["runs"]] == [r[0] for r in rows]
    checked_steps = 0
    steps_in_window = 0
    for rid, kind, raw in rows:
        m = json.loads(raw)
        row = by_id[rid]
        if isinstance(m.get("timesteps"), int):
            steps_in_window += 1
            assert row["steps"] == m["timesteps"]
            assert row["deep"] == (1 if m.get("deep") else 0)
            checked_steps += 1
        if kind == "stacker":
            assert "steps" not in row
    # Data-aware non-vacuity guard -- the same pattern as the cfg
    # guard in test_compact_feeds: every steps-recording run in the
    # window must have been checked. The old absolute demand (the
    # window must hold a steps-recording run) tripped on window
    # composition alone: the 2026-11-05 detail window (runs 107-110,
    # all Pick-study stackers) records no steps anywhere. The
    # surfacing path itself stays pinned deterministically by
    # test_detail_run_settings_from_fixture below.
    assert checked_steps == steps_in_window


def test_detail_run_settings_from_fixture():
    base = {"score": -1.0, "total_pnl": -2.0, "max_drawdown": 1.0,
            "mean_reward": -0.5, "bench_pnl": 3.0,
            "windows": [{"score": -1.0}, {"score": -2.0},
                        {"score": 0.5}]}
    lab = {"id": 900, "ts": "2026-10-25T00:00:00+00:00", "kind": "ppo",
           "promoted": False,
           "metrics": {"family": "hierarchy",
                       "hierarchy": {"design": "gate + direction split"},
                       "challenger": dict(base), "champion": dict(base),
                       "timesteps": 60_000, "deep": False,
                       "reason": "lab duel"}}
    out = compact._duel_run(lab)
    assert out["fam"] == 10  # same family code as the models index
    assert out["cfg"] == "gate + direction split"
    assert out["steps"] == 60_000 and out["deep"] == 0

    ex = {"id": 901, "ts": "2026-10-25T00:00:00+00:00", "kind": "ppo",
          "promoted": False,
          "metrics": {"family": "exit_study",
                      "exit_study": {"config": {
                          "stop_mult": 1.5, "tp_mode": None,
                          "tp_value": 0.0, "trail": "current",
                          "transition": "T0"}},
                      "challenger": dict(base), "champion": dict(base),
                      "reason": "exit duel"}}
    out = compact._duel_run(ex)
    assert out["fam"] == 5
    assert out["cfg"] == ("stop_mult=1.5, tp_value=0, trail=current, "
                          "transition=T0")

    plain = {"id": 902, "ts": "2026-10-25T00:00:00+00:00",
             "kind": "stacker", "promoted": False,
             "metrics": {"challenger_acc": 0.46, "champion_acc": 0.48,
                         "reason": "stacker duel"}}
    out = compact._duel_run(plain)
    assert "fam" not in out and "cfg" not in out and "steps" not in out
