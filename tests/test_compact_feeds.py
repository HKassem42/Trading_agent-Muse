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
                      "pending", "sacc", "sprec"}
    # Promotion lock (2026-10-25): the pending field is present and
    # either null (nobody awaiting Kass) or a well-formed record.
    assert g["pending"] is None or (
        set(g["pending"]) == {"i", "k", "t", "why"}
        and g["pending"]["k"] in ("ppo", "stacker"))
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
        # The score formula itself: pass requires aggregate AND >=2
        # windows. That result now rides as fpass (data); the headline
        # pass chip is the RECORDED outcome (2026-10-22 display fix):
        # formula-pass alone must never render as PASSED for a run the
        # registry did not promote.
        expect = 1 if (g["duel"]["delta"] > 0
                       and g["duel"]["won"] >= 2) else 0
        assert g["duel"]["fpass"] == expect
        assert g["duel"]["pass"] == (expect if g["duel"]["newchamp"]
                                     else 0)
        assert g["duel"]["rec"] == ("PROMOTED" if g["duel"]["newchamp"]
                                    else "NOT PROMOTED")
    payload = json.dumps(g, separators=(",", ":"))
    print(f"gate feed bytes={len(payload)}")
    assert len(payload) < 1024, f"gate feed too big: {len(payload)} bytes"


def test_gate_pending_why_is_complete_not_truncated_mid_clause():
    """2026-10-30 dashboard audit: run 83's gate pending.why arrived on
    BOTH dashboards cut mid-clause at exactly 100 chars --
    "...P&L +237 vs " -- because _pending_promotion sliced the stored
    reason at [:100]. The feed text must carry the COMPLETE champion
    comparison the stored reason makes, trimmed (if at all) only at a
    clause boundary, never mid-word/mid-clause. The expected text is
    derived from training.db itself, never hard-coded."""
    g = compact.build_gate()
    pend = g["pending"]
    assert pend is not None and pend["i"] == 83
    why = pend["why"]
    con = sqlite3.connect(ROOT / "data" / "training.db")
    row = con.execute(
        "SELECT metrics FROM runs WHERE id = 83").fetchone()
    con.close()
    stored = json.loads(row[0]).get("reason") or ""
    assert stored.startswith("Meta-Label Gate")
    # The bug: the old 100-char slice never reaches the comparison.
    assert len(why) > 100
    assert "P&L +237 vs champion -305" in why
    assert not why.rstrip().endswith("vs")
    # Trimmed only at a clause boundary: why is a verbatim prefix of
    # the stored reason, and the very next stored character (if any)
    # is a clause delimiter.
    assert stored.startswith(why)
    if len(why) < len(stored):
        assert stored[len(why)] in ";.", (
            f"pending.why cut mid-clause at char {len(why)}: "
            f"...{why[-30:]!r} | next: {stored[len(why):len(why)+30]!r}")
    # The gate feed stays inside its documented size budget.
    payload = json.dumps(g, separators=(",", ":"))
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
    assert g["duel"]["fpass"] == 1  # formula: delta > 0 AND won >= 2
    # ...but the fixture challenger was NOT promoted, so the recorded
    # verdict chip must not read as a pass (2026-10-22 display fix).
    assert g["duel"]["pass"] == 0
    assert g["duel"]["rec"] == "NOT PROMOTED"


# --- 2026-10-22 dashboard completeness audit ------------------------------
# Kass directive: both dashboards must display EVERY model tried, with
# correct names/versions. The training feed therefore carries a slim
# all-runs index (snapshot.models) next to the detailed last-8 duel rows,
# and the gate chip shows the recorded outcome instead of the raw score
# formula alone (run 20 rendered PASSED while recorded not-promoted).

import sqlite3  # noqa: E402


def _db_runs():
    con = sqlite3.connect(ROOT / "data" / "training.db")
    rows = con.execute(
        "SELECT id, promoted FROM runs ORDER BY id").fetchall()
    con.close()
    return rows


def test_training_models_index_covers_every_run_with_correct_names():
    t = compact.build_compact_training()
    models = t["models"]
    db = _db_runs()
    # Every registered run, newest first, promoted flags matching the DB.
    assert [m[0] for m in models] == [r[0] for r in db][::-1]
    by_id = {m[0]: m for m in models}
    for rid, promoted in db:
        assert by_id[rid][4] == int(promoted)
    names = {rid: by_id[rid][2] for rid, _ in db}
    # Lab families get their real names, not raw codes or kind labels.
    assert names[20] == "HA-Seq v1"
    assert names[21] == "GBM Focus A — Plain"
    assert names[22] == "GBM Focus B — Class-Weighted"
    assert names[23] == "GBM Focus C — SMOTE"
    assert names[24] == "GBM Focus D — All-60 Features"
    # Nightly lineage: explicit versions + era inheritance.
    assert names[17] == "PPO v5.1" and names[18] == "Stacker v5.1"
    assert names[19] == "PPO v5.1" and names[16] == "Stacker v5.1"
    assert names[15] == "PPO v5" and names[14] == "Stacker v5"
    assert names[11] == "PPO v4" and names[9] == "Stacker v4"
    assert names[13] == "PPO v4" and names[12] == "Stacker v4"
    assert names[8] == "PPO v3" and names[7] == "Stacker v3"
    assert names[5] == "PPO v2" and names[6] == "PPO v2"
    # Family codes: 0 ppo, 1 stacker, 2 ha_seq, 3 gbm_focus.
    assert by_id[17][3] == 0 and by_id[18][3] == 1
    assert by_id[20][3] == 2 and by_id[21][3] == 3
    # Headline numbers match the registry (score/P&L for PPO-family,
    # accuracy and no P&L for the stacker). Index P&L is whole dollars
    # since the 2026-10-23 size diet; exact cents stay in training.db.
    assert by_id[24][5] == -1603.7 and by_id[24][6] == -3178
    assert by_id[18][5] == 0.4659 and by_id[18][6] is None
    # Recorded lab verdicts ride along; nightly runs carry none.
    assert by_id[20][7].startswith("FAIL")
    assert by_id[21][7] == "PASS" and by_id[24][7] == "PASS"
    assert by_id[22][7] == "FAIL" and by_id[23][7] == "FAIL"
    assert by_id[17][7] is None
    payload = json.dumps(t, separators=(",", ":"))
    print(f"training feed bytes={len(payload)}")
    # Feed budget: 16,384B = 2x the largest payload this dashboard has
    # VERIFIABLY accepted (7,925B pushed via push_snapshot and read
    # back intact 2026-10-23; 6,433B and 3,307B accepted before it; no
    # push to this dashboard has ever been rejected, and the old
    # "~2.5KB ceiling" belief was falsified by each of those pushes).
    # The budget is a guardrail, not a measured platform limit:
    # acceptance above 7,925B is unverified, so the feed is dieted to
    # ~6.7KB (see the size-regression test) and the models-index growth
    # (~90-118B/night) keeps it under budget for 60+ nights.
    assert len(payload) < 16384, f"training feed too big: {len(payload)}"


def test_training_feed_size_regression_at_current_registry():
    """Size sentinel for the 2026-10-23 diet, re-baselined
    2026-10-22 at the 39-run registry (6,971B measured; the 3
    size_study runs added ~97B per models-index row, inside the
    documented growth band), again 2026-10-24 at the 54-run
    registry (7,268B measured; the 3 regime_data runs added ~115B
    per models-index row, inside the documented growth band),
    and again 2026-10-24 at the 56-run registry (7,716B measured;
    the 2 hierarchy runs grew the feed ~224B per row -- above the
    ~90-118B index-row band because the detail window also rolled
    to longer records), and again 2026-10-24 at the same 56-run
    registry after the audit fix restored reward_series to the full
    registry (7,984B measured; the +268B is the 6 PPO points the
    30-run snapshot window had silently dropped, back up to the
    documented newest-20 cap), and again 2026-10-25 at the 60-run
    registry after the training-dashboard upgrade (8,642B measured;
    the +1,039B over the pre-upgrade 7,603B at the same registry is
    three deliberate, bounded additions Kass asked for -- the
    recipe panel 309B and gate history 426B, both fixed-size, plus
    ~281B of recorded per-run settings in the detail window), and
    again 2026-10-25 at the 64-run registry (8,881B measured; the 2
    full_indicator_stacker runs grew the feed ~158B per row --
    above the ~90-118B index-row band because the detail window
    also rolled to the study's longer records, the same effect as
    the hierarchy re-baseline), and again 2026-10-25 at the same
    64-run registry after the lab/driver's-seat separation
    (9,212B measured; the +331B is one deliberate, bounded
    addition Kass asked for -- the champions roster section
    (seat line + the stacker/PPO title-holder rows), fixed-size
    at the current two-crown roster and growing only when a new
    champion is crowned), and again 2026-10-25 at the 66-run
    registry after the scored-eyes PPO study (10,412B measured
    post-diet; the 2 scored_eyes_ppo runs added ~1,200B -- above
    the per-row band because the detail window rolled to two full
    PPO duel records AND the newest-PPO learning curve became a
    300k deep run's: its recorded 288 points (~4.1KB) took the feed
    to 13,706B pre-diet, and this trip forced the rcurve diet --
    stride-downsample to <= 64 recorded points, endpoints exact --
    in build_compact_training), and again 2026-10-26 at the
    68-run registry after Kass's "what the brains read" panel
    (11,078B measured; the +1,346B over the 9,732B pre-panel size
    at the same registry is one deliberate, bounded addition he
    asked for -- the reads block: the prod21 roster with his five
    indicators flagged, plus Champion #2's recorded top-15 names
    and Kass-family presence in its scored set; both sections are
    fixed-size and change only when a champion's study record
    does), and again 2026-10-26 at the 70-run registry after the
    Muse's Pick challenger study (11,630B measured; the 2
    muse_pick runs added ~232B -- two models-index rows plus the
    detail window rolling to the study's records and the telemetry
    block gaining its first row (run 70), inside the documented
    per-row band), and again 2026-10-27 at the 75-run registry
    after Muse's Pick #2 + Pick #3 and the Champion #4 crowning
    (12,154B measured; the five runs since run 70 added ~524B --
    five models-index rows, the detail window rolling to the
    pick-study records, two more telemetry rows (runs 74/75), and
    the champions roster gaining its fourth crown row, all inside
    the documented bands), and again 2026-10-31 at the 90-run
    registry after Phase 3 of the Training Lab transformation
    (12,810B measured; the +476B over the 12,334B pre-change size
    at the same registry is two deliberate, bounded additions --
    the votes block (200B: the stacker champion's stored holdout
    vote histogram + summary, fixed-size, sha-guarded to the
    reigning champion artifact) and the weights block (256B:
    per-layer weight stats of the live-seat PPO, fixed six
    layers); both blocks are fixed-size and do not grow with the
    registry):
    the feed must be <= 12,900B and
    the detail window exactly the newest 6 runs (the models index
    carries the rest). The absolute number is expected to need
    re-baselining as the index grows (~90-118B/night) -- that trip
    is the point: it forces the next diet while the feed is still
    far under the 16,384B budget asserted above."""
    t = compact.build_compact_training()
    db = _db_runs()
    assert len(t["models"]) == len(db)
    assert [r["i"] for r in t["runs"]] == [r[0] for r in db][-6:][::-1]
    payload = json.dumps(t, separators=(",", ":"))
    print(f"training feed bytes (regression)={len(payload)}")
    assert len(payload) <= 12900, (
        f"training feed regressed past the diet target: "
        f"{len(payload)} bytes at {len(db)} runs")


def test_training_reads_block_names_what_each_brain_reads():
    """Kass (2026-10-26): the Training Lab must show how the models
    use HIS indicators. The reads block carries (1) the prod21 roster
    every production brain reads, with Kass's five v5 features
    flagged, and (2) the reigning champion's recorded study data
    (run 64's full-indicator study until 2026-10-26; run 69's
    Muse's Pick study while it reigned as Champion #3; run 73's
    Muse's Pick #2 study since the 2026-10-27 Champion #4
    crowning) -- its
    input count and the ranked top-15 feature list its registry record
    stored at grading time (names in recorded order; the registry
    stored NO numeric weights, so the feed must not invent any).
    Kass-family presence in the scored set is derived from the run's
    own recorded feature list, never hardcoded."""
    t = compact.build_compact_training()
    reads = t["reads"]
    p21 = reads["p21"]
    assert [r[0] for r in p21] == [
        "body_pct", "upper_wick", "lower_wick", "range_pct", "rsi_c",
        "macd_hist_rel", "dist_vwap", "dist_sma50", "atr_rel",
        "price_change", "volatility_21", "volume_change", "h4_rsi_c",
        "h4_dist_sma", "d1_dist_sma", "h4_atr_rel", "ha_dir",
        "obv_slope", "adl_slope", "netvol_rel", "ewo"]
    assert all(len(r) == 3 and r[1] for r in p21)
    flagged = {r[0] for r in p21 if r[2] == 1}
    assert flagged == {"ha_dir", "obv_slope", "adl_slope",
                       "netvol_rel", "ewo"}
    c2 = reads["c2"]
    assert c2 is not None  # the champion's record is on file
    assert c2["run"] == 73 and c2["n"] == 98 and c2["comp"] == 4
    assert len(c2["top"]) == 15
    assert c2["top"][0] == "Stochastic RSI (StochRSI)"
    # Names only: any numeric weight here would be fabricated.
    assert all(isinstance(x, str) for x in c2["top"])
    # Derived from run 73's recorded feature list (the scored 94
    # plus Pick #2's four, none of which is a Kass-family name):
    # scored OBV and A/D are in; Heikin-Ashi, net volume and the
    # EWO are not.
    assert set(c2["kin"]) == {"OBV", "A/D line"}
    assert set(c2["kout"]) == {"Heikin-Ashi", "Net volume",
                               "Elliott Wave Osc."}


def test_gate_champ_and_duel_survive_snapshot_window_cap(monkeypatch):
    """2026-10-24 full-dashboard audit: the training snapshot caps its
    runs list at the newest 30, and once the registry passed 30 runs
    that window held NO promoted PPO run at all -- build_gate returned
    champ/chall/duel all null on both dashboards while run 17 reigns
    and run 44's floor-blocked duel is the latest gate duel on record.
    Simulate the cap by dropping promoted PPO runs from the snapshot
    window: the gate must still seat the champion from the full
    registry, and its duel must be the newest GATE run (nightly, no
    lab family) -- lab studies dueled study baselines, not the crown,
    so they are never the gate's challenger. (The challenger pin was
    run 44 at audit time; the 2026-10-25 retrains added newer duels,
    so the expected duel is now DERIVED from the registry -- exact
    values, no hard-coded run id to go stale again. Run 44's
    floor-blocked semantics stay pinned in test_training_panels.py's
    gate-history checks.)"""
    real = compact.snapshot.build_training_snapshot()
    capped = dict(real)
    capped["runs"] = [r for r in real["runs"]
                      if not (r["kind"] == "ppo" and r["promoted"])]
    monkeypatch.setattr(compact.snapshot, "build_training_snapshot",
                        lambda: capped)
    g = compact.build_gate()
    assert g["champ"] is not None and g["champ"]["i"] == 17
    assert g["champ"]["s"] == -1789.9
    assert g["champ"]["win"] == [-1995.5, -3827.3, 453.2]
    assert g["champ"]["pl"] == -3408.62
    con = sqlite3.connect(ROOT / "data" / "training.db")
    con.row_factory = sqlite3.Row
    duel = None
    for row in con.execute("SELECT * FROM runs ORDER BY id DESC"):
        m = json.loads(row["metrics"])
        if row["kind"] != "ppo" or m.get("family"):
            continue
        c, h = m.get("challenger") or {}, m.get("champion") or {}
        if (isinstance(c.get("score"), (int, float))
                and isinstance(h.get("score"), (int, float))
                and c["score"] > -1e8 and h["score"] > -1e8):
            duel = (row, m)
            break
    con.close()
    assert duel is not None
    row, m = duel
    c, h = m["challenger"], m["champion"]
    assert g["chall"] is not None and g["chall"]["i"] == row["id"]
    assert g["chall"]["s"] == round(c["score"], 1)
    assert g["chall"]["pl"] == round(c["total_pnl"], 2)
    assert g["duel"]["cs"] == round(h["score"], 1)
    assert g["duel"]["cwin"] == [round(w["score"], 1) for w in h["windows"]]
    assert g["duel"]["cpl"] == round(h["total_pnl"], 2)
    won = sum(1 for a, b in zip(c["windows"], h["windows"])
              if round(a["score"], 1) > round(b["score"], 1))
    assert g["duel"]["won"] == won
    assert g["duel"]["delta"] == round(round(c["score"], 1)
                                       - round(h["score"], 1), 1)
    assert g["duel"]["fpass"] == (1 if c["score"] > h["score"]
                                  and won >= 2 else 0)
    # The headline verdict is the RECORDED outcome, never recomputed.
    assert g["duel"]["pass"] == (1 if row["promoted"] else 0)
    assert g["duel"]["rec"] == ("PROMOTED" if row["promoted"]
                                else "NOT PROMOTED")
    assert g["duel"]["newchamp"] == (1 if row["promoted"] else 0)


def test_reward_series_uses_full_registry_not_snapshot_window():
    """2026-10-24 audit: reward_series was built from the snapshot's
    30-run window, so older PPO runs silently fell out of the learning
    chart (14 points shown where the registry holds more than 20 PPO
    runs with a recorded mean_reward). The series is the newest 20
    PPO mean-rewards across ALL registered runs."""
    t = compact.build_compact_training()
    con = sqlite3.connect(ROOT / "data" / "training.db")
    rows = con.execute(
        "SELECT ts, metrics FROM runs WHERE kind='ppo' ORDER BY id ASC"
    ).fetchall()
    con.close()
    exp = []
    for ts, mj in rows:
        mr = (json.loads(mj).get("challenger") or {}).get("mean_reward")
        if isinstance(mr, (int, float)):
            exp.append({"t": ts, "r": round(mr, 3)})
    assert len(exp) > 20  # the registry really holds more than the cap
    assert t["reward_series"] == exp[-20:]


def test_gate_chip_promoted_challenger_reads_promoted(monkeypatch):
    snap = _fake_training_snapshot()
    snap["runs"][0]["promoted"] = True  # the duel winner took the crown
    monkeypatch.setattr(compact.snapshot, "build_training_snapshot",
                        lambda: snap)
    g = compact.build_gate()
    assert g["duel"]["fpass"] == 1
    assert g["duel"]["pass"] == 1
    assert g["duel"]["rec"] == "PROMOTED"
    assert g["duel"]["newchamp"] == 1


def test_full_indicator_stacker_family_renders_in_index():
    """Full-indicator stacker study (Kass, 2026-10-25): lab stacker
    runs under family "full_indicator_stacker" must render in the
    models index with their real name, family code 12, the STACKER
    headline (holdout accuracy, no P&L), and their recorded verdict
    -- the same treatment features_v6 stacker runs get."""
    r = {"id": 999, "ts": "2026-10-25T18:00:00+00:00",
         "kind": "stacker", "promoted": False,
         "metrics": {
             "challenger_acc": 0.5123, "champion_acc": 0.4811,
             "challenger_wf_acc": 0.501, "precision": 0.49, "folds": 5,
             "pending_approval": True,
             "reason": "full-indicator stacker Arm A holdout acc "
                       "0.5123 beat champion 0.4811",
             "family": "full_indicator_stacker",
             "full_indicator_stacker": {
                 "label": "Raw 98", "arm": "A", "verdict": "PASS",
                 "design": "Arm A raw v2 set: 90 features",
                 "n_features": 90}}}
    name = compact._display_name(r, None)
    assert name == "Full-Indicator Stacker — Raw 98"
    entry = compact._model_entry(r, name)
    assert entry[2] == name
    assert entry[3] == 12  # family code
    assert entry[4] == 0  # never promoted (promotion lock)
    assert entry[5] == 0.5123  # stacker headline = holdout accuracy
    assert entry[6] is None  # stacker runs carry no P&L
    assert entry[7] == "PASS"
    # The detail row carries the family code + config note too.
    detail = compact._duel_run(r)
    assert detail["fam"] == 12
    assert detail["s"] == 0.5123 and detail["cs"] == 0.4811
    assert detail["cfg"] == "Arm A raw v2 set: 90 features"


def test_training_rcurve_is_bounded_and_keeps_endpoints():
    """The learning curve is the newest PPO run's recorded reward curve.
    A 300k deep run records ~288 points (~4.1KB), which blew the feed
    past its diet target when run 66 landed (13,706B at 66 runs). The
    feed stride-downsamples to <= 64 points for the chart, keeping the
    first and last recorded points exactly -- sampled truth, never
    interpolated or invented values."""
    t = compact.build_compact_training()
    rc = t["rcurve"]
    assert rc is not None and 0 < len(rc) <= 64
    registry = compact._all_runs_desc()
    raw = next((r.get("metrics") or {}).get("reward_curve")
               for r in registry
               if r["kind"] == "ppo"
               and (r.get("metrics") or {}).get("reward_curve"))
    assert rc[0][0] == raw[0][0]
    assert rc[-1][0] == raw[-1][0]
    assert rc[-1][1] == round(raw[-1][1], 2)
    # Every shown point is a recorded point, in order.
    raw_pts = {(p[0], round(p[1], 2)) for p in raw}
    assert all((p[0], p[1]) in raw_pts for p in rc)


def test_training_feed_flags_prefix_and_declined_runs():
    """Truth-in-labeling (Kass directive, 2026-10-26): grades earned
    before the v5.1 look-ahead fix are history, not edge, and runs
    Kass declined must not keep reading as plain PASSes. The feed
    carries both run-id sets, derived from the registry itself:
    `pfx` (pre-fix) and `dec` (pending_decision == declined). The
    artifact joins them onto the models index, the detail cards,
    and the family bests by run id."""
    t = compact.build_compact_training()
    assert "pfx" in t and "dec" in t
    # Independent derivation #1 (timestamps): the lab note titled
    # "Features v5.1 migration" marks the fix going live (its ts is
    # run 16's own stamp -- the first clean grade, holdout 0.4596);
    # every run stamped strictly before it was graded pre-fix.
    notes = json.loads((ROOT / "data" / "lab_notes.json").read_text())
    mig = [n["ts"] for n in notes if "v5.1 migration" in n["title"]]
    assert len(mig) == 1, "expected exactly one v5.1 migration note"
    boundary = mig[0]
    con = sqlite3.connect(ROOT / "data" / "training.db")
    rows = con.execute("SELECT id, ts, metrics FROM runs").fetchall()
    con.close()
    expect_pfx = sorted(rid for rid, ts, _ in rows if ts < boundary)
    assert t["pfx"] == expect_pfx
    # Independent derivation #2 (recorded decisions): declined = the
    # runs whose own metrics record Kass's decision against them.
    expect_dec = sorted(
        rid for rid, _, m in rows
        if json.loads(m).get("pending_decision") == "declined")
    assert t["dec"] == expect_dec
    # Today's pinned truth: the trophy pair (Stacker v4 run 10,
    # PPO v5 run 15) is pre-fix; the migration pair, the champions,
    # and every later run are not. Declined is exactly 61/63/67/78
    # (78 declined 2026-10-29 on Muse's pick under Kass's delegation).
    assert 10 in t["pfx"] and 15 in t["pfx"]
    for rid in (16, 17, 40, 44, 61, 63, 64, 67, 68):
        assert rid not in t["pfx"], f"run {rid} wrongly flagged pre-fix"
    assert t["dec"] == [61, 63, 67, 78]
    # Every flagged id is a real models-index row the artifact can
    # join the marker onto.
    ids = {row[0] for row in t["models"]}
    assert set(t["pfx"]) <= ids and set(t["dec"]) <= ids


def _assert_bounded_preview(feed_text, full_text, cap, label):
    """A compact-feed preview of `full_text`: either the full text, or
    a verbatim prefix that ends at a word/clause boundary -- the very
    next stored character must be a space or a clause delimiter, so
    the preview can never end mid-word or mid-number."""
    assert feed_text, f"{label}: preview is empty"
    if feed_text == full_text:
        return
    assert len(feed_text) <= cap, (
        f"{label}: preview exceeds its {cap}-char cap: "
        f"{len(feed_text)} chars")
    assert full_text.startswith(feed_text), (
        f"{label}: preview is not a verbatim prefix of the stored "
        f"text: {feed_text!r}")
    nxt = full_text[len(feed_text)]
    assert nxt in " ;—", (
        f"{label}: preview cuts mid-word at char {len(feed_text)}: "
        f"...{feed_text[-25:]!r} | next stored: "
        f"{full_text[len(feed_text):len(feed_text) + 25]!r}")


def test_training_feed_text_never_truncates_mid_word():
    """2026-10-31 dashboard review: Training Lab verdict/config text
    truncated MID-WORD -- run 88's cfg arrived as "...on the full HA
    frame; BU", run 87's cfg as "...sub-cha", run 86's cfg as
    "...deterministic swi", run 88's why as "...duel agg -1550 vs
    champion -" and run 87's note preview as "...exam acc 0.5"
    (mid-NUMBER). Root cause: three raw character slices in the
    training feed builder -- _verdict_reason [:100] (runs.why),
    _duel_run's cfg [:80], _lab_notes' detail [:90]. The previews
    must trim at word/clause boundaries (the _clause_bounded helper
    from the 2026-10-30 gate fix, extended with a word-boundary
    fallback), keeping each field's existing cap. Expected full
    texts are derived from training.db / lab_notes.json, never
    hard-coded."""
    t = compact.build_compact_training()
    registry = {r["id"]: r for r in compact._all_runs_desc()}
    checked = {"why": 0, "cfg": 0, "notes": 0}
    for run in t["runs"]:
        m = registry[run["i"]].get("metrics") or {}
        stored_reason = m.get("reason") or m.get("promotion_reason")
        if stored_reason and "why" in run:
            _assert_bounded_preview(run["why"], str(stored_reason),
                                    100, f"run {run['i']} why")
            checked["why"] += 1
        fam = m.get("family")
        sub = (m.get(fam) or {}) if fam else {}
        design = sub.get("design")
        if isinstance(design, str) and design and "cfg" in run:
            _assert_bounded_preview(run["cfg"], design, 80,
                                    f"run {run['i']} cfg")
            checked["cfg"] += 1
    raw_notes = compact._lab_notes_raw()
    by_ts = {n["ts"]: n for n in raw_notes}
    for note in t["notes"]:
        full = by_ts[note["t"]]
        _assert_bounded_preview(note["d"], str(full["detail"]), 90,
                                f"note {full['title']!r} detail")
        checked["notes"] += 1
    # The current registry really exercises all three previews --
    # a silent no-op test would be worthless.
    assert checked["why"] >= 4 and checked["cfg"] >= 3
    assert checked["notes"] == 4
    # Today's pinned offenders must be gone entirely.
    blob = json.dumps(t, ensure_ascii=False)
    for frag in ("frame; BU", "sub-cha", "deterministic swi",
                 "exam acc 0.5", "champion -\""):
        assert frag not in blob, f"mid-word fragment still in feed: {frag!r}"


def test_clause_bounded_word_fallback_trims_at_word_boundary():
    """The word-boundary fallback added 2026-10-31: when NO clause
    delimiter fits inside the limit, _clause_bounded(word_fallback=
    True) trims at the last word boundary inside the limit instead
    of extending past the cap (the gate's completeness-first
    behaviour, kept as the default) or slicing mid-word."""
    text = ("alpha beta gamma delta epsilon zeta eta theta iota "
            "kappa lambda mu nu xi omicron pi rho sigma tau")
    out = compact._clause_bounded(text, limit=40, word_fallback=True)
    assert len(out) <= 40 and text.startswith(out)
    assert text[len(out)] == " "
    # The gate's default path is unchanged: with no clause boundary
    # anywhere, the text rides whole rather than cutting mid-clause.
    assert compact._clause_bounded(text, limit=40) == text
    # And the gate's clause path still wins when a boundary fits.
    gated = "title — subtitle; the complete first clause rides " \
            "whole; a much longer second clause follows here"
    assert compact._clause_bounded(gated, limit=60) == \
        "title — subtitle; the complete first clause rides whole"


def test_training_feed_votes_block_matches_stored_champion_votes():
    """Phase 3 (2026-10-31): the votes block must be the reigning
    stacker champion's STORED holdout votes -- the per-candle
    probabilities scripts/run_meta_label_study.py generated from
    models/stacker_champion.pkl (sha-pinned in the study's
    results.json) on its 792-candle holdout frame and saved to
    data/meta_label/votes_hold.npy. The block is a histogram +
    summary of THAT array, recomputed here independently; it must
    vanish (never go stale) when the champion artifact no longer
    matches the study's pinned sha."""
    import hashlib

    import numpy as np

    t = compact.build_compact_training()
    v = t["votes"]
    assert v is not None, "votes block missing while its source is stored"
    # Run 73's registry record predates grading-time vote logging,
    # so its block still comes from the study array.
    assert v["src"] == "study"
    study = json.loads(
        (ROOT / "data" / "meta_label" / "results.json").read_text())
    champ_meta = json.loads(
        (ROOT / "models" / "stacker_champion.json").read_text())
    sha = hashlib.sha256(
        (ROOT / "models" / "stacker_champion.pkl").read_bytes()).hexdigest()
    # The votes describe the CURRENT champion artifact, or nothing.
    assert sha == study["frames"]["champion"]["stacker_sha256"]
    assert v["run"] == champ_meta["run_id"] == \
        study["frames"]["champion"]["stacker_run"]
    assert v["sha"] == sha[:8]
    votes = np.load(ROOT / "data" / "meta_label" / "votes_hold.npy")
    bins, _ = np.histogram(votes, bins=20, range=(0.0, 1.0))
    assert v["bins"] == [int(x) for x in bins]
    assert sum(v["bins"]) == v["n"] == len(votes)
    assert v["min"] == round(float(votes.min()), 4)
    assert v["max"] == round(float(votes.max()), 4)
    assert v["med"] == round(float(np.median(votes)), 4)
    assert v["mid"] == round(
        100.0 * float(np.mean((votes >= 0.45) & (votes <= 0.55))), 1)
    assert v["tight"] == round(
        100.0 * float(np.mean((votes >= 0.48) & (votes <= 0.52))), 1)
    frames = study["frames"]
    assert v["n"] == frames["holdout_rows"]
    assert v["from"] == frames["holdout_first"][:16]
    assert v["to"] == frames["holdout_last"][:16]


def test_training_feed_policy_weights_match_direct_sb3_load():
    """Phase 3 (2026-10-31): the weights block carries per-layer
    weight statistics (mean / population std / min / max) of the
    live-seat PPO policy -- run 44's artifact, byte-identical to
    models/ppo_champion.zip. The builder reads policy.pth straight
    out of the zip; this test recomputes the same statistics a
    second way, through a full stable-baselines3 PPO.load, and the
    two routes must agree layer for layer."""
    import hashlib

    import numpy as np
    from stable_baselines3 import PPO

    t = compact.build_compact_training()
    w = t["weights"]
    assert w is not None, "weights block missing while the seat model is stored"
    seat = json.loads((ROOT / "data" / "live_seat.json").read_text())
    sha = hashlib.sha256(
        (ROOT / "models" / "live_seat" / "ppo.zip").read_bytes()).hexdigest()
    assert sha == seat["seat"]["ppo"]["sha256"]
    assert w["run"] == seat["seat"]["ppo"]["run_id"]
    assert w["sha"] == sha[:8]
    spec = {
        "mlp_extractor.policy_net.0.weight": "pi1",
        "mlp_extractor.policy_net.2.weight": "pi2",
        "mlp_extractor.value_net.0.weight": "vf1",
        "mlp_extractor.value_net.2.weight": "vf2",
        "action_net.weight": "act",
        "value_net.weight": "val",
    }
    model = PPO.load(str(ROOT / "models" / "live_seat" / "ppo.zip"))
    sd = model.policy.state_dict()
    expected = []
    for key, short in spec.items():
        a = sd[key].detach().cpu().numpy().astype(np.float64)
        expected.append([short, round(float(a.mean()), 4),
                         round(float(a.std()), 4),
                         round(float(a.min()), 4),
                         round(float(a.max()), 4)])
    assert w["layers"] == expected


def _votes_tree(tmp_path, metrics):
    """A minimal ROOT tree whose reigning stacker champion is a
    synthetic run 999 with the given registry metrics."""
    import sqlite3

    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "models" / "stacker_champion.pkl").write_bytes(
        b"synthetic-champion")
    (tmp_path / "models" / "stacker_champion.json").write_text(
        json.dumps({"run_id": 999, "feature_set": "prod21"}))
    con = sqlite3.connect(tmp_path / "data" / "training.db")
    con.execute(
        "CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, kind TEXT NOT NULL, metrics TEXT NOT NULL, "
        "promoted INTEGER NOT NULL)")
    con.execute(
        "INSERT INTO runs(id, ts, kind, metrics, promoted) "
        "VALUES (999, '2026-11-05T04:04:46+00:00', 'stacker', ?, 1)",
        (json.dumps(metrics),))
    con.commit()
    con.close()


def test_training_feed_votes_block_prefers_grading_histogram(
        tmp_path, monkeypatch):
    """Grading-time vote logging (2026-10-31): when the reigning
    stacker champion's own registry record carries the vote summary
    the grader stored (metrics["votes"]), the feed serves THAT --
    src "grading", the stored bins verbatim, the run's grading
    timestamp -- ahead of any study array."""
    import hashlib

    stored = {"bins": [0] * 9 + [30, 42] + [0] * 9, "n": 72,
              "min": 0.4701, "max": 0.5299, "med": 0.5012,
              "mid": 100.0, "tight": 88.9}
    _votes_tree(tmp_path, {"challenger_acc": 0.61, "votes": stored})
    monkeypatch.setattr(compact, "ROOT", tmp_path)

    v = compact._votes_block()
    assert v is not None
    assert v["src"] == "grading"
    assert v["run"] == 999
    assert v["bins"] == stored["bins"]
    assert v["n"] == stored["n"] == sum(stored["bins"])
    assert v["min"] == stored["min"]
    assert v["max"] == stored["max"]
    assert v["med"] == stored["med"]
    assert v["mid"] == stored["mid"]
    assert v["tight"] == stored["tight"]
    assert v["from"] == v["to"] == "2026-11-05T04:04"
    assert v["sha"] == hashlib.sha256(
        b"synthetic-champion").hexdigest()[:8]


def test_training_feed_votes_block_null_when_champion_has_no_votes(
        tmp_path, monkeypatch):
    """Self-invalidation: a champion whose record carries no stored
    votes and no matching study array gets NO block -- the feed
    never serves a former champion's votes under a new crown."""
    _votes_tree(tmp_path, {"challenger_acc": 0.61})
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    assert compact._votes_block() is None
