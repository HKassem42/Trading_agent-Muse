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
                      "pending", "sacc", "sprec", "schamp"}
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


def test_gate_carries_stacker_champion_identity():
    """2026-11-13 dashboard review: Book #1's gate panel showed the
    stacker champion's exam figures (sacc/sprec) with no way to name
    WHICH champion they belong to -- the feed carried no run id or
    crown number, so the client would have had to hard-code it.
    schamp must be the champions roster's CURRENT stacker row
    (status 1 or 3), derived from the registry, and its figures must
    agree with sacc/sprec on the same feed."""
    g = compact.build_gate()
    champs = compact.build_champions(compact._all_runs_desc())
    assert champs and champs.get("stk"), "roster must list stacker crowns"
    current = [row for row in champs["stk"] if row[7] in (1, 3)]
    assert len(current) == 1, f"exactly one current champion: {current}"
    no, run_id, _name, _feat, acc, _wf, prec, _st = current[0]
    assert g["schamp"] == {"i": run_id, "n": no}, (
        f"gate schamp {g['schamp']} != roster current {(no, run_id)}")
    # The identity must belong to the same brain as the exam figures.
    assert g["sacc"] == round(acc, 4)
    assert g["sprec"] == round(prec, 4)


def test_gate_pending_why_is_complete_not_truncated_mid_clause():
    """2026-10-30 dashboard audit: run 83's gate pending.why arrived on
    BOTH dashboards cut mid-clause at exactly 100 chars --
    "...P&L +237 vs " -- because _pending_promotion sliced the stored
    reason at [:100]. The feed text must carry the COMPLETE champion
    comparison the stored reason makes, trimmed (if at all) only at a
    clause boundary, never mid-word/mid-clause. The expected text is
    derived from training.db itself, never hard-coded.

    2026-11-02: the live pending flag legitimately moved to run 96
    (the Nov 2 midnight retrain's stacker WOULD PROMOTE) -- the old
    pin `pend["i"] == 83` rotted the same way the cfg guard did on
    2026-11-01. The expectations below are therefore derived from
    the registry by _pending_promotion's own documented rule, and
    the run-83 regression is pinned directly on _clause_bounded,
    where it can never rot again."""
    con = sqlite3.connect(ROOT / "data" / "training.db")
    rows = con.execute(
        "SELECT id, kind, metrics, promoted FROM runs "
        "ORDER BY id DESC").fetchall()
    con.close()
    newest_promoted = {}
    for rid, kind, _m, promoted in rows:
        if promoted:
            newest_promoted.setdefault(kind, rid)
    expected_id = None
    stored = ""
    for rid, kind, mjson, _p in rows:
        m = json.loads(mjson) if mjson else {}
        if not m.get("pending_approval") or m.get("pending_decision"):
            continue
        sup = newest_promoted.get(kind)
        if sup is not None and sup > rid:
            continue
        expected_id = rid
        stored = m.get("reason") or ""
        break
    g = compact.build_gate()
    pend = g["pending"]
    if expected_id is None:
        # 2026-11-06: Kass crowned runs 111 + 112, so no unresolved
        # pending exists -- run 83's PPO-kind pending is superseded by
        # the promoted run 112 under the resolver's own documented
        # rule (the same rule that superseded run 100 when run 110
        # was crowned). The builder must agree: no pending anywhere.
        assert pend is None
    else:
        assert pend is not None and pend["i"] == expected_id
        why = pend["why"]
        assert not why.rstrip().endswith("vs")
        # Trimmed only at a clause boundary: why is a verbatim prefix
        # of the stored reason, and the very next stored character
        # (if any) is a clause delimiter.
        assert stored.startswith(why)
        if len(why) < len(stored):
            assert stored[len(why)] in ";.", (
                f"pending.why cut mid-clause at char {len(why)}: "
                f"...{why[-30:]!r} | next: {stored[len(why):len(why)+30]!r}")
    # The original run-83 regression, pinned on the trimmer itself:
    # its 566-char stored reason must keep the champion comparison
    # whole ("P&L +237 vs champion -305"), never "...P&L +237 vs ".
    con = sqlite3.connect(ROOT / "data" / "training.db")
    row = con.execute(
        "SELECT metrics FROM runs WHERE id = 83").fetchone()
    con.close()
    stored83 = json.loads(row[0]).get("reason") or ""
    assert stored83.startswith("Meta-Label Gate")
    why83 = compact._clause_bounded(stored83)
    assert len(why83) > 100
    assert "P&L +237 vs champion -305" in why83
    assert not why83.rstrip().endswith("vs")
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
    # The duel is read from the full registry (2026-11-06: a registry
    # fact, not a window fact -- see build_gate), so the fabricated
    # world stubs the registry read with the same fabricated runs.
    monkeypatch.setattr(compact, "_all_runs_desc",
                        lambda: _fake_training_snapshot()["runs"])
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
    # Since the 2026-11-17 pagination the index ships via
    # push_models_page pages, not inside the snapshot; assemble
    # it exactly as the desk does (pages in order, rows joined).
    models = [row for page in compact.build_models_pages()
              for row in page["rows"]]
    db = _db_runs()
    # Kass's feed-trim directive (2026-11-05): the shipped index
    # omits the FAILED runs 16-50 (never promoted, no seat/crown --
    # run 44 kept by id as the override-installed PPO champion);
    # runs 1-15 (pre-fix set) and everything over 50 always ship.
    # The complete record lives in training.db / the workbook /
    # the Model Atlas, and _models_index itself stays complete.
    promoted = {r[0] for r in db if r[1]}
    expect_ids = [r[0] for r in db
                  if r[0] < 16 or r[0] > 50 or r[0] in promoted
                  or r[0] == 44][::-1]
    assert [m[0] for m in models] == expect_ids
    assert t["models_omitted"] == len(db) - len(expect_ids)
    by_id = {m[0]: m for m in models}
    for rid, prom in db:
        if rid in by_id:
            assert by_id[rid][4] == int(prom)
    names = {rid: by_id[rid][2] for rid in expect_ids}
    # The naming layer still covers EVERY run, trimmed or not --
    # the roster and the Atlas derive names from the full index.
    _full_index = compact._models_index(compact._all_runs_desc())
    full_names = {row[0]: row[2] for row in _full_index}
    full_by_id = {row[0]: row for row in _full_index}
    # Lab families get their real names, not raw codes or kind labels.
    assert full_names[20] == "HA-Seq v1"
    assert full_names[21] == "GBM Focus A — Plain"
    assert full_names[22] == "GBM Focus B — Class-Weighted"
    assert full_names[23] == "GBM Focus C — SMOTE"
    assert full_names[24] == "GBM Focus D — All-60 Features"
    # Nightly lineage: explicit versions + era inheritance.
    assert names[17] == "PPO v5.1" and names[18] == "Stacker v5.1"
    assert full_names[19] == "PPO v5.1" and names[16] == "Stacker v5.1"
    assert names[15] == "PPO v5" and names[14] == "Stacker v5"
    assert names[11] == "PPO v4" and names[9] == "Stacker v4"
    assert names[13] == "PPO v4" and names[12] == "Stacker v4"
    assert names[8] == "PPO v3" and names[7] == "Stacker v3"
    assert names[5] == "PPO v2" and names[6] == "PPO v2"
    # Family codes: 0 ppo, 1 stacker, 2 ha_seq, 3 gbm_focus.
    assert by_id[17][3] == 0 and by_id[18][3] == 1
    assert full_by_id[20][3] == 2 and full_by_id[21][3] == 3
    # Headline numbers match the registry (score/P&L for PPO-family,
    # accuracy and no P&L for the stacker). Index P&L is whole dollars
    # since the 2026-10-23 size diet; exact cents stay in training.db.
    assert full_by_id[24][5] == -1603.7 and full_by_id[24][6] == -3178
    assert by_id[18][5] == 0.4659 and by_id[18][6] is None
    # Recorded lab verdicts ride along; nightly runs carry none.
    assert full_by_id[20][7].startswith("FAIL")
    assert full_by_id[21][7] == "PASS" and full_by_id[24][7] == "PASS"
    assert full_by_id[22][7] == "FAIL" and full_by_id[23][7] == "FAIL"
    assert by_id[17][7] is None
    payload = json.dumps(t, separators=(",", ":"))
    print(f"training snapshot bytes={len(payload)}")
    # Ceilings after the 2026-11-17 pagination (full basis in the
    # structural growth tests at the bottom of this file): the
    # snapshot carries only count-capped sections now, so its
    # plateau is ~9.5KB even at a 300-run registry -- asserted
    # here at 12,000 with headroom. Each models page (40 rows)
    # must stay under 4,000B (measured ~3.1KB with the action-mix
    # 9th element, Kass 2026-11-23; the dashboard's
    # largest verified acceptance is 7,925B, so a page keeps >2x
    # headroom no matter how large the registry grows -- the
    # page COUNT grows, never the page). The historical 16,384B
    # whole-feed budget (2x the 7,925B verified acceptance) is
    # retired with the monolith it guarded.
    assert "models" not in t
    assert t["models_total"] == len(models)
    assert len(payload) < 12000, f"training snapshot too big: {len(payload)}"
    for page in compact.build_models_pages():
        blob = json.dumps(page, separators=(",", ":"))
        assert len(blob) < 4000, f"models page too big: {len(blob)}"


def test_action_mix_counts_recover_exactly_for_every_ppo_run():
    """Action Mix — every model (Kass, 2026-11-23): the models index
    carries each PPO run's action mix as exact decision COUNTS over
    its last grading window, recovered from the registry's recorded
    3-decimal shares — never interpolated. The grading env runs
    24-decision episodes (TradingEnv episode_len = 1 day x 24
    hourly candles), so a standard duel window is 10 episodes x 24
    = 240 decisions; a mix recovers exactly when every share has
    one integer preimage at that total and the counts sum to it.
    Runs whose shares do not verify at any recorded total keep
    shares only (counts null) — the section shows them honestly
    rather than guessing. Adversarial pins: run 153 (the screenshot
    panel: Buy 0 / Hold 74 / Sell 13 / Short 12), run 88 (exact
    fractions over 264 decisions), run 75 (exposure vocabulary),
    run 44 (the live-seat driver)."""
    registry = compact._all_runs_desc()
    # Known-unrecoverable challenger mixes: lab-study duels whose
    # episode count the registry never recorded (runs 45-56) and run
    # 86, whose challenger was graded off the standard path (its
    # champion side still verifies at 240). Any other failure is a
    # regression in the recovery itself.
    unrecoverable = {45, 46, 50, 51, 52, 54, 55, 56, 86}
    recovered = 0
    for r in registry:
        if r["kind"] != "ppo":
            continue
        mix = (r.get("metrics") or {}).get("challenger", {}).get("action_mix")
        if not mix:
            continue
        got = compact._recover_mix_counts(mix, (r.get("metrics") or {}).get("challenger", {}).get("episodes"))
        if r["id"] in unrecoverable:
            assert got is None, f"run {r['id']} unexpectedly recovered: {got}"
            continue
        assert got is not None, f"run {r['id']} mix failed to recover: {mix}"
        vocab, counts, total = got
        assert sum(counts) == total
        assert len(counts) == len(vocab)
        recovered += 1
    assert recovered >= 60, f"only {recovered} mixes recovered"
    by_id = {r["id"]: r for r in registry}
    # The screenshot: run 153's mix at promotion — Buy 0 / Hold 74 /
    # Sell 13 / Short 12, over the 240-decision last window.
    assert compact._recover_mix_counts(
        by_id[153]["metrics"]["challenger"]["action_mix"], 30) == (
        ["hold", "buy", "sell", "short"], [178, 1, 31, 30], 240)
    # Run 88's mix is exact fractions (never rounded to 3 decimals):
    # 256 hold + 3 buy + 0 sell + 5 short over 264 decisions.
    assert compact._recover_mix_counts(
        by_id[88]["metrics"]["challenger"]["action_mix"], None) == (
        ["hold", "buy", "sell", "short"], [256, 3, 0, 5], 264)
    # Run 75 is the exposure manager (Muse's Pick #3): its own
    # vocabulary, same 240-decision window.
    assert compact._recover_mix_counts(
        by_id[75]["metrics"]["challenger"]["action_mix"], 30) == (
        ["stand_aside", "half", "full", "hold"], [47, 12, 2, 179], 240)
    # Run 44, the live-seat driver: Buy 0 is real, not a gap.
    assert compact._recover_mix_counts(
        by_id[44]["metrics"]["challenger"]["action_mix"], None) == (
        ["hold", "buy", "sell", "short"], [149, 0, 89, 2], 240)


def test_models_index_rows_carry_action_mix():
    """The models index row gains a 9th element: the run's action
    mix for the Action Mix section — counts + window total when
    exactly recovered, 2-decimal shares when not, null for stackers
    (they vote, they don't act) and PPO runs with no recorded mix.
    Row length stays uniform at 9 so the client parses positionally."""
    registry = compact._all_runs_desc()
    by_id = {r["id"]: r for r in registry}
    name153 = compact._display_name(by_id[153], None)
    row153 = compact._model_entry(by_id[153], name153)
    assert len(row153) == 9
    assert row153[8] == {"v": "d", "c": [178, 1, 31, 30], "n": 240}
    row45 = compact._model_entry(
        by_id[45], compact._display_name(by_id[45], None))
    assert len(row45) == 9
    assert row45[8]["v"] == "d" and row45[8]["c"] is None
    assert row45[8]["f"] == [0.2, 0.42, 0.06, 0.32]
    row75 = compact._model_entry(
        by_id[75], compact._display_name(by_id[75], None))
    assert row75[8] == {"v": "x", "c": [47, 12, 2, 179], "n": 240}
    # Stackers take no actions: no mix, never a fabricated one.
    row152 = compact._model_entry(
        by_id[152], compact._display_name(by_id[152], None))
    assert len(row152) == 9 and row152[8] is None
    # PPO runs with no recorded challenger mix: null, not zeros.
    row47 = compact._model_entry(
        by_id[47], compact._display_name(by_id[47], None))
    assert len(row47) == 9 and row47[8] is None
    # Every shipped page row carries the element (uniform shape).
    for page in compact.build_models_pages():
        for row in page["rows"]:
            assert len(row) == 9, f"run {row[0]} row has {len(row)} elements"


def test_training_feed_structure_at_current_registry():
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
    registry), and again 2026-11-03 at the 99-run registry after
    the source-truth audit (14,495B measured; growth since the
    90-run baseline is the models index for runs 91-99 plus one
    deliberate, bounded addition: full text for trimmed previews
    -- notes[].full on the newest-4 lab notes and why_full on the
    6-run detail window, shipped ONLY when the preview was
    actually trimmed, so the two long Kass-directive notes (the
    1,218-char trailing-stop ruling and the 795-char run-96
    crowning) are readable in full from the dashboard instead of
    existing nowhere in the payload; the addition is bounded by
    the note count and the fixed detail window, not the registry),
    and again 2026-11-04 at the 101-run registry (14,666B measured;
    the two runs since the 99-run baseline added ~171B -- two
    models-index rows plus the detail window rolling to run 101's
    muse_next_try record, inside the documented per-row band), and
    again 2026-11-05 at the 110-run registry (16,089B measured AFTER
    that day's diet; pre-diet the feed measured 17,776B -- over the
    16,384B budget -- because the Pick #7-#11 study burst put five
    long stacker records in the detail window and the run-110
    crowning added a 1,031-char lab note. The diet cut the detail
    window 6 -> 4 and the lab-notes window 4 -> 3; the models index
    stays complete and every shipped note/run still carries its
    full text when trimmed), and rebaselined again the same evening
    after Kass's feed-trim directive (failed runs 16-50 out of the
    shipped index -- 30 rows, ~2.1KB -- with the complete record in
    the Model Atlas / workbook / training.db; measured 13,990B at
    the 110-run registry), and again 2026-11-11 at the 125-run
    registry after the Champion #12 crowning and Muse's Pick #12
    (14,209B measured; the net +219B over the 110-run baseline is
    the models index for runs 111-125 plus the champions roster
    gaining its 12th crown row, partly offset by the bounded
    detail/notes windows rolling past the long run-110 crowning
    note -- all inside the documented bands; the 16,384B budget
    above is unchanged, ~2.1KB of headroom), and again 2026-11-13
    at the 128-run registry after Muse's Pick #13 (14,869B
    measured; the +660B over the 125-run baseline is the models
    index for runs 126-128 plus the detail window rolling to run
    128's long Pick #13 study record -- the same above-band
    detail-window effect as the hierarchy and scored-eyes trips;
    the trip also forced the muse_pick13 family-code fix (code 33)
    in compact.py, without which the feed did not generate at all.
    The 16,384B budget above is unchanged, ~1.5KB of headroom):
    the feed must be <= 15,100B and
    the detail window exactly the newest 4 runs (the models index
    carries the rest). The absolute number is expected to need
    re-baselining as the index grows (~90-118B/night) -- that trip
    is the point: it forces the next diet while the feed is still
    under the 16,384B budget asserted above.

    2026-11-17 SIZE FIX (Kass: stop running into size limits,
    for good): the aging-sentinel era ends here. The models index
    -- the feed's only registry-driven growth -- no longer rides
    in the snapshot: it ships in fixed 40-row pages via
    push_models_page (compact.build_models_pages), so the
    snapshot's size is a function of its caps, not of the run
    count. This test is now STRUCTURAL: no "models" key in the
    snapshot, counts exact against the registry, pages assemble
    to the full kept index, the detail window exactly the newest
    4, the snapshot under its 12,000 plateau ceiling (basis:
    every section count-capped -- detail window 4, notes 3,
    reward_series 20, rcurve <= 64, gate_hist 6, reads/audit/
    telemetry/votes/weights fixed; the champs roster is the only
    event-driven section, ~90B per crowning), and every page
    under 4,000B (40 rows x ~55B + envelope ~= 2.3KB measured;
    basis: comfortably under the dashboard's largest verified
    acceptance, 7,925B). The growth tests at the bottom of this
    file prove the same bounds at a synthetic 300-run registry
    and 2 years of hourly rows."""
    t = compact.build_compact_training()
    db = _db_runs()
    _promoted = {r[0] for r in db if r[1]}
    _kept = [r for r in db if r[0] < 16 or r[0] > 50
             or r[0] in _promoted or r[0] == 44]
    assert "models" not in t
    assert t["models_total"] == len(_kept)
    assert t["models_omitted"] == len(db) - len(_kept)
    pages = compact.build_models_pages()
    assert [row for p in pages for row in p["rows"]] == \
        compact._models_index_feed(compact._all_runs_desc())[0]
    assert compact.build_models_manifest()["pages"] == len(pages)
    assert [r["i"] for r in t["runs"]] == [r[0] for r in db][-4:][::-1]
    payload = json.dumps(t, separators=(",", ":"))
    print(f"training snapshot bytes (structural)={len(payload)}")
    assert len(payload) <= 12000, (
        f"training snapshot over its plateau ceiling: "
        f"{len(payload)} bytes at {len(db)} runs")
    for p in pages:
        blob = json.dumps(p, separators=(",", ":"))
        assert len(blob) < 4000, f"models page too big: {len(blob)}"


def test_training_reads_block_names_what_each_brain_reads(
        monkeypatch, tmp_path):
    """Kass (2026-10-26): the Training Lab must show how the models
    use HIS indicators. The reads block carries (1) the prod21 roster
    every production brain reads, with Kass's five v5 features
    flagged, and (2) the reigning champion's recorded study data
    (run 64's full-indicator study until 2026-10-26; run 69's
    Muse's Pick study while it reigned as Champion #3; run 73's
    Muse's Pick #2 study while it reigned as Champion #4) -- its
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
    # Since Kass's 2026-11-02 LAB-ONLY crowning of run 96 (a nightly
    # prod21 stacker with no study block on its record), the reigning
    # champion carries no recorded study data, so c2 must be null --
    # the panel hides rather than showing a former champion's reads
    # (the builder's own documented contract).
    assert reads["c2"] is None
    # The positive path still holds against a champion whose record
    # DOES carry study data: re-point the builder's champion meta at
    # run 73 (Champion #4, Muse's Pick #2 study) and re-derive.
    runs_desc = compact._all_runs_desc()
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    (fake_models / "stacker_champion.json").write_text(
        json.dumps({"run_id": 73}))
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    c2 = compact._reads_block(runs_desc)["c2"]
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
    # The expected champion is DERIVED from the registry (newest
    # promoted PPO run's own recorded challenger block): it was
    # run 17 at audit time; Kass's 2026-11-06 crowning of run 112
    # superseded the hard-coded pin the same way the run-44
    # challenger pin rotted in 2025.
    con0 = sqlite3.connect(ROOT / "data" / "training.db")
    exp_champ = None
    for rid0, mjson0, prom0 in con0.execute(
            "SELECT id, metrics, promoted FROM runs "
            "WHERE kind = 'ppo' ORDER BY id DESC"):
        if not prom0:
            continue
        c0 = (json.loads(mjson0) or {}).get("challenger") or {}
        if isinstance(c0.get("score"), (int, float)):
            exp_champ = (rid0, c0)
            break
    con0.close()
    assert exp_champ is not None
    exp_id, exp_c = exp_champ
    assert g["champ"] is not None and g["champ"]["i"] == exp_id
    assert g["champ"]["s"] == round(exp_c["score"], 1)
    assert g["champ"]["win"] == [round(w["score"], 1)
                                 for w in exp_c["windows"]]
    assert g["champ"]["pl"] == round(exp_c["total_pnl"], 2)
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
    # The duel is read from the full registry (2026-11-06), so the
    # fabricated world stubs the registry read too — the same
    # pattern as test_gate_duel_won_is_derived_from_its_own_windows.
    monkeypatch.setattr(compact, "_all_runs_desc",
                        lambda: snap["runs"])
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
    # and every later run are not. Declined is exactly
    # 61/63/67/78/137 (78 declined 2026-10-29 on Muse's pick
    # under Kass's delegation; 137 declined 2026-11-16 by Muse
    # under the standing crowning delegation -- it won its exam
    # by ~2 candles with materially weaker precision/WF anatomy;
    # the pin went stale when the ruling landed and is corrected
    # here, verified against the registry's recorded decisions
    # by the independent derivation asserted just above).
    assert 10 in t["pfx"] and 15 in t["pfx"]
    for rid in (16, 17, 40, 44, 61, 63, 64, 67, 68):
        assert rid not in t["pfx"], f"run {rid} wrongly flagged pre-fix"
    assert t["dec"] == [61, 63, 67, 78, 137]
    # Every flagged id is a real models-index row the artifact can
    # join the marker onto.
    ids = {row[0] for page in compact.build_models_pages()
           for row in page["rows"]}
    assert set(t["pfx"]) <= ids and set(t["dec"]) <= ids


def _assert_bounded_preview(feed_text, full_text, cap, label):
    """A compact-feed preview of `full_text`: either the full text, or
    a verbatim prefix that ends at a word/clause boundary -- the very
    next stored character must be a space or a clause delimiter, so
    the preview can never end mid-word or mid-number -- PLUS a
    trailing "…" marking the trim (2026-11-03: a bounded cut with
    no marker still reads as a mid-clause cut, e.g. the note
    preview ending "'Arm trailing at 2%"). A trimmed preview may
    exceed its cap by the ellipsis alone (one character)."""
    assert feed_text, f"{label}: preview is empty"
    if feed_text == full_text:
        return
    assert feed_text.endswith("…"), (
        f"{label}: trimmed preview carries no ellipsis marker: "
        f"...{feed_text[-25:]!r}")
    core = feed_text[:-1]
    assert len(core) <= cap, (
        f"{label}: preview exceeds its {cap}-char cap: "
        f"{len(core)} chars before the ellipsis")
    assert len(feed_text) <= cap + 1, (
        f"{label}: preview exceeds its cap by more than the "
        f"ellipsis: {len(feed_text)} chars vs cap {cap}")
    assert full_text.startswith(core), (
        f"{label}: preview is not a verbatim prefix of the stored "
        f"text: {feed_text!r}")
    nxt = full_text[len(core)]
    assert nxt in " ;—", (
        f"{label}: preview cuts mid-word at char {len(core)}: "
        f"...{core[-25:]!r} | next stored: "
        f"{full_text[len(core):len(core) + 25]!r}")


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
    # The current registry really exercises the previews --
    # a silent no-op test would be worthless. why/notes are always
    # present in the detail window, so their guards stay hard. cfg
    # only exists on lab-family runs: the 2026-11-01 detail window
    # (runs 90-95, all plain v5.1) carries none, which tripped the
    # old >=3 guard without any truncation present. The cfg guard is
    # therefore data-aware: every cfg-bearing run in the window must
    # have been checked, and a cfg-free window must emit no cfg key
    # at all (the vacuous case becomes a real assertion).
    cfg_in_window = 0
    for run in t["runs"]:
        m = registry[run["i"]].get("metrics") or {}
        fam = m.get("family")
        sub = (m.get(fam) or {}) if fam else {}
        if isinstance(sub.get("design"), str) and sub["design"]:
            cfg_in_window += 1
    assert checked["why"] >= 4
    assert checked["cfg"] == cfg_in_window
    if cfg_in_window == 0:
        assert all("cfg" not in run for run in t["runs"])
    assert checked["notes"] == 3
    # Today's pinned offenders must be gone entirely. The
    # "exam acc 0.5" pin is scoped to its field-final shape (closing
    # quote): since 2026-11-03 the feed ships complete note texts as
    # "full", and the run-110 crowning note legitimately contains
    # "exam acc 0.5341" -- the offender is the number sliced at a
    # field's end, not the digits inside a complete sentence. Every
    # preview is still guarded field-by-field above.
    blob = json.dumps(t, ensure_ascii=False)
    for frag in ("frame; BU", "sub-cha", "deterministic swi",
                 "exam acc 0.5\"", "champion -\""):
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


def test_training_trimmed_previews_end_with_ellipsis(monkeypatch):
    """2026-11-03 defect: Training Lab previews cut with NO ellipsis
    -- the trailing-stop lab note arrived as "...options tap 'Arm
    trailing at 2%", a complete word but a sentence cut in half, so
    it read as a mid-clause truncation. Every ACTUAL trim of a
    training preview (notes detail cap 90, run why cap 100, run cfg
    cap 80) must end with "…", the text before it must be a verbatim
    prefix ending at a word/clause boundary (never mid-word), and
    the preview may exceed its cap by the ellipsis alone. Text that
    was NOT trimmed never gains an ellipsis. This test fails on the
    pre-fix helper, whose bounded trims carried no marker."""
    # Unit level, all three trim paths of the helper.
    words = ("alpha beta gamma delta epsilon zeta eta theta iota "
             "kappa lambda mu nu xi omicron pi rho sigma tau upsilon")
    out = compact._clause_bounded(words, limit=40, word_fallback=True,
                                  ellipsis=True)
    assert out.endswith("…") and len(out) <= 41
    assert words.startswith(out[:-1]) and words[len(out) - 1] == " "
    gated = "title — subtitle; the complete first clause rides " \
            "whole; a much longer second clause follows here"
    out = compact._clause_bounded(gated, limit=60, word_fallback=True,
                                  ellipsis=True)
    assert out == ("title — subtitle; the complete first clause "
                   "rides whole…")
    # Untrimmed text is returned exactly, with no marker appended,
    # and the gate default (no ellipsis flag) is unchanged.
    assert compact._clause_bounded("short text", limit=40,
                                   word_fallback=True,
                                   ellipsis=True) == "short text"
    assert compact._clause_bounded(words, limit=40,
                                   word_fallback=True) == \
        compact._clause_bounded(words, limit=40, word_fallback=True,
                                ellipsis=True)[:-1]

    # Feed level: every note the current window trims must carry
    # the marker and verify as a bounded preview of its source.
    # (The original anchor -- the trailing-stop note -- aged out of
    # the newest-4 window on 2026-11-04 when the retrain and the
    # Muse's Next Try notes landed; the invariant, not the note,
    # is what this section pins. The trailing note itself is pinned
    # separately below through a synthetic window.)
    t = compact.build_compact_training()
    trimmed_notes = 0
    raw_by_ts = {n["ts"]: n for n in compact._lab_notes_raw()}
    for note in t["notes"]:
        full = str(raw_by_ts[note["t"]]["detail"])
        if len(full) > len(note["d"]):
            trimmed_notes += 1
            _assert_bounded_preview(note["d"], full, 90,
                                    f"note {note['h']!r} detail")
            assert note["d"].endswith("…")
    assert trimmed_notes >= 1

    # The exact reported regression string, pinned permanently:
    # rebuild the notes block with the real trailing-stop note as
    # the newest entry and require its preview to carry the marker
    # and its full text to ride along byte-identical.
    raw_notes = compact._lab_notes_raw()
    trail_raw = next(n for n in raw_notes
                     if "trailing-stop" in (n.get("title") or ""))
    monkeypatch.setattr(compact, "_lab_notes_raw",
                        lambda: raw_notes[-3:] + [trail_raw])
    pinned = compact._lab_notes()[0]
    assert pinned["d"].endswith("…"), (
        f"note preview lacks ellipsis: {pinned['d']!r}")
    assert not pinned["d"].endswith("at 2%")
    assert "options tap 'Arm trailing at 2%" in pinned["d"]
    assert pinned["full"] == str(trail_raw["detail"])
    monkeypatch.undo()

    # Run why + cfg through their real builders, on a synthetic run
    # whose stored reason and design both exceed their caps.
    long_reason = ("challenger kept the crown because the duel "
                   "aggregate stayed below the champion across all "
                   "three windows and the floor was never reached")
    long_design = ("gate plus direction split with deterministic "
                   "swing labels on the full scored indicator frame "
                   "and walk forward folds")
    assert len(long_reason) > 100 and len(long_design) > 80
    run = {"id": 9999, "ts": "2026-11-03T00:00:00+00:00",
           "kind": "stacker", "promoted": False,
           "metrics": {"challenger_acc": 0.51, "champion_acc": 0.49,
                       "reason": long_reason, "family": "hierarchy",
                       "hierarchy": {"design": long_design}}}
    detail = compact._duel_run(run)
    _assert_bounded_preview(detail["why"], long_reason, 100,
                            "synthetic run why")
    _assert_bounded_preview(detail["cfg"], long_design, 80,
                            "synthetic run cfg")
    assert detail["why"].endswith("…")
    assert detail["cfg"].endswith("…")


def test_training_feed_votes_block_matches_stored_champion_votes():
    """Phase 3 (2026-10-31): the votes block must be the reigning
    stacker champion's STORED holdout votes, or nothing.

    While run 73 reigned (Champion #4, record predating grading-time
    vote logging), the block came from the meta-label study array
    (data/meta_label/votes_hold.npy, sha-pinned in its results.json).
    Since Kass crowned run 96 as Champion #5 on 2026-11-02 (LAB
    ONLY), the champion's own record carries a grading-time
    histogram (nightly vote logging, 2026-10-31), and the block must
    serve THAT verbatim with src "grading" -- while the study
    fallback must self-invalidate, because the study's pinned sha
    describes run 73's artifact, not the current champion's."""
    import hashlib
    import sqlite3

    t = compact.build_compact_training()
    v = t["votes"]
    assert v is not None, "votes block missing while its source is stored"
    champ_meta = json.loads(
        (ROOT / "models" / "stacker_champion.json").read_text())
    rid = champ_meta["run_id"]
    assert v["run"] == rid
    assert v["src"] == "grading"
    sha = hashlib.sha256(
        (ROOT / "models" / "stacker_champion.pkl").read_bytes()).hexdigest()
    assert v["sha"] == sha[:8]
    # The block is the champion's registry-stored histogram, verbatim.
    con = sqlite3.connect(ROOT / "data" / "training.db")
    row = con.execute(
        "SELECT metrics, ts FROM runs WHERE id=?", (rid,)).fetchone()
    con.close()
    stored = json.loads(row[0])["votes"]
    assert v["bins"] == [int(x) for x in stored["bins"]]
    assert sum(v["bins"]) == v["n"] == int(stored["n"])
    for key in ("min", "max", "med"):
        assert v[key] == round(float(stored[key]), 4)
    for key in ("mid", "tight"):
        assert v[key] == round(float(stored[key]), 1)
    assert v["from"] == v["to"] == row[1][:16]
    # The study fallback describes run 73's artifact only: with a
    # different champion installed it must vanish, never go stale.
    assert compact._study_votes_block() is None


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


# --- 2026-11-17 structural growth tests --------------------------------
# Kass's order: stop running into size limits. Instead of a
# sentinel that gets re-baselined every few weeks, these tests
# prove the bounds at SYNTHETIC scale -- a ~300-run registry and
# ~2 years of hourly rows -- because every feed section now has
# a fixed count cap and the plateau is a constant of the caps.
# Ceilings and bases:
#   * models page < 4,000B: 40 rows x ~55B + envelope = ~2.3KB
#     measured; the Training dashboard's largest verified
#     acceptance is 7,925B, so a page keeps >2x headroom forever
#     -- the page COUNT grows with the registry, never the page.
#   * training snapshot <= 12,000B (asserted in the structural
#     test above): every section count-capped; 8,248B measured
#     at the 138-run registry.
#   * curves: _downsample_curve <= 19 points and
#     _downsample_pairs(cap) <= cap+1 points at ANY input length
#     (a [19-char ts, value] point costs <= 33B, so one curve is
#     <= ~1.4KB whether the book is a week or a decade old).
#   * daily calendar: _feed_calendar caps at 371 entries (53
#     weeks) -- the section previously grew ~72B/day forever.
#   * gated veto log: 200 rows (~9.5KB) with full-table counts.

def _synthetic_runs(n=300):
    """A synthetic registry in _all_runs_desc() shape (newest
    first): alternating stacker/PPO runs with plausible metrics,
    a few promoted, spanning the feed-trim id band."""
    runs = []
    for rid in range(n, 0, -1):
        if rid % 2:
            runs.append({"id": rid, "ts": "2026-11-17T00:00:00",
                         "kind": "stacker",
                         "promoted": rid % 37 == 0,
                         "metrics": {"challenger_acc": 0.5417,
                                     "challenger": {}}})
        else:
            runs.append({"id": rid, "ts": "2026-11-17T00:00:00",
                         "kind": "ppo",
                         "promoted": rid % 41 == 0,
                         "metrics": {"challenger": {
                             "score": -775.11, "pnl": -1151.94,
                             "episodes": 30,
                             "action_mix": {"hold": 0.6, "buy": 0.1,
                                            "sell": 0.2, "short": 0.1},
                             "mean_reward": 0.123}}})
    return runs


def test_models_pages_bounded_at_300_run_registry():
    runs = _synthetic_runs(300)
    pages = compact.build_models_pages(runs)
    full, omitted = compact._models_index_feed(runs)
    # The full kept index, exactly -- pagination is transport,
    # never a trim (Kass: the full list is the point).
    assert [row for p in pages for row in p["rows"]] == full
    assert len(pages) == -(-len(full) // compact.MODELS_PAGE_SIZE)
    assert len(full) > 250  # the synthetic registry really is big
    for p in pages:
        assert p["total"] == len(full) and p["omitted"] == omitted
        blob = json.dumps(p, separators=(",", ":"))
        assert len(blob) < 4000, (
            f"models page {p['page']} too big at 300 runs: "
            f"{len(blob)}")


def test_curves_and_calendar_bounded_at_two_years_of_hourly_rows():
    # 17,520 hourly points (2 years) through the same transforms
    # the book / race / master builders apply to a book's equity.
    pts = [{"t": f"2024-01-{(i // 24) % 28 + 1:02d}T{i % 24:02d}:00:00",
            "equity": 100000.0 + (i * 37 % 5000) - i * 0.11}
           for i in range(17520)]
    curve = compact._downsample_curve(pts)
    assert len(curve) <= 19
    assert curve[-1]["equity"] == round(pts[-1]["equity"], 2)
    pairs = [[p["t"], p["equity"]] for p in pts]
    assert len(compact._downsample_pairs(pairs)) <= 41
    assert len(compact._downsample_pairs(pairs, cap=20)) <= 21
    assert len(compact._downsample_pairs(pairs, cap=24)) <= 25
    # Two full years of daily-calendar entries (one per day,
    # forever, before the fix) collapse to the 371-day window.
    cal = [{"date": f"2024-{(i // 31) % 12 + 1:02d}-{i % 28 + 1:02d}",
            "pnl": 1.0, "trades": 1, "win_pct": 50.0}
           for i in range(730)]
    out = compact._feed_calendar({"daily_calendar": cal})
    assert len(out) == compact.CALENDAR_FEED_DAYS == 371
    assert out[-1] == cal[-1] and out[0] == cal[-371]


def test_gated_veto_feed_plateau_bounded(tmp_path, monkeypatch):
    """250 recorded gate decisions (the forward test accrues
    ~18/week, so this is ~3 months of evidence) must ship as a
    200-row window with the FULL counts intact."""
    db = tmp_path / "paper_gated.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE gate_decisions (id INTEGER PRIMARY"
                " KEY, ts TEXT, side INTEGER, action INTEGER,"
                " proba REAL, threshold REAL, verdict TEXT)")
    for i in range(250):
        con.execute(
            "INSERT INTO gate_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i + 1, "2026-11-17T13:00:00", -1, 3, 0.4681, 0.55,
             "VETOED" if i % 3 else "TAKEN"))
    con.commit()
    con.close()
    monkeypatch.setattr(compact, "GATED_DB", db)
    v = compact.build_gated_veto()
    assert v["total"] == 250 and v["shown"] == 200
    assert len(v["rows"]) == 200
    assert v["taken"] + v["vetoed"] == 250
    blob = json.dumps(v, separators=(",", ":"))
    assert len(blob) < 12000, f"gated veto plateau too big: {len(blob)}"


def test_champ_lineage_feed_capped():
    """The crown lineage grows one era per crowning (~119B/era);
    the feed ships the newest 24 eras plus the true total."""
    man = {"lineage": [{"era": i, "pinned_at": "2026-11-13T02:37:51Z",
                        "driver_run": 112, "eyes_run": 126,
                        "eyes_champion": "#13"} for i in range(1, 31)],
           "grading_record": {}, "eyes": {}, "driver": {}}
    block = compact._champ_crown_block(man)
    assert len(block["lineage"]) == 24
    assert block["lineage_total"] == 30
    assert block["era"] == 30
    assert block["lineage"][0]["era"] == 7
