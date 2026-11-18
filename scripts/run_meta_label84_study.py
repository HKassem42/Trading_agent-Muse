#!/usr/bin/env python
"""Meta-Label Gate run 84 — the activity-calibrated gate:
study runner (2026-10-30).

Kass's order: make the models trade ~30% of the time instead
of ~10%, rerun it, and get clear results. Design frozen in
refs/meta-label-study-84-2026-10-30.md BEFORE any run-84
holdout run. Run 84 keeps run 83's frozen fitted meta-model
byte-identical and changes only the operating point: tau84 is
the 70th percentile of the model's TRAINING-REGION
TimeSeriesSplit(4) out-of-fold probabilities over stream-1
candidates (the line that would have passed ~30% of stream-1
attempts in training), computed and published before the
duel, never adjusted after seeing holdout behavior.

Arms: primary (graded) = stream 1 at tau84; secondary
(reported) = BOTH streams at tau84 with PPO precedence in
the same duel windows. Machinery pins run first: pass-all ==
production champion (1e-6), and the run-84 machinery at run
83's 0.55 reproduces run 83's recorded 0.55 arm. Verdict on
the primary arm: formula + gate_decision()'s two-part floor,
identical to runs 82/83.

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded
before/after, now including Book #3's gated seat; the three
paper books are information-only — their own loops write
them — and are never opened). Writes live only under
data/meta_label84/, plus the registry run and the lab note.

Phases checkpoint under data/meta_label84/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  prep -> calibrate -> duel -> register
The duel phase REFUSES to run until tau84 (from the calibrate
phase) is published in refs/meta-label-study-84-2026-10-30.md
— the pre-registration is literal, not ceremonial.

Usage: .venv/bin/python scripts/run_meta_label84_study.py [phase]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import meta_label as ml  # noqa: E402
from training import meta_label83 as m83  # noqa: E402
from training import meta_label84 as m84  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import _agg, gate_decision  # noqa: E402

LAB82 = ROOT / "data" / "meta_label"
LAB83 = ROOT / "data" / "meta_label83"
LAB = ROOT / "data" / "meta_label84"
RESULTS_PATH = LAB / "results.json"
RESULTS82_PATH = LAB82 / "results.json"
RESULTS83_PATH = LAB83 / "results.json"
REFS_PATH = ROOT / "refs" / "meta-label-study-84-2026-10-30.md"
TRAIN_SUP_PATH = LAB82 / "train_sup.pkl"
HOLD_SUP_PATH = LAB82 / "hold_sup.pkl"
FRAME_HOLD_PATH = LAB82 / "frame_hold.npy"
VOTES_HOLD_PATH = LAB82 / "votes_hold.npy"
CTX_PATH = LAB82 / "context.npz"
EXTRAS83_PATH = LAB83 / "extras.npz"
CANDIDATES83_PATH = LAB83 / "candidates83.json"
ASSEMBLED83_PATH = LAB83 / "candidates83_assembled.json"
DATASET83_PATH = LAB83 / "dataset83.npz"
MODEL83_PATH = LAB83 / "meta_model83.pkl"
OOF84_PATH = LAB / "oof_proba84.npz"
LEDGERS = LAB / "eval_ledgers"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"
MODEL83_SHA256 = ("ddbe2b4870f5464f80ea4fa2bb60386a322cfbc43743"
                  "cc89b6f8cb4326c744ae")

GUARD_FILES = {
    "live_seat_ppo": ROOT / "models" / "live_seat" / "ppo.zip",
    "live_seat_stacker": ROOT / "models" / "live_seat" / "stacker.pkl",
    "live_seat_meta": ROOT / "models" / "live_seat" / "stacker.json",
    "shadow_seat_ppo": ROOT / "models" / "shadow_seat" / "ppo.zip",
    "shadow_seat_stacker": ROOT / "models" / "shadow_seat" / "stacker.pkl",
    "shadow_seat_meta": ROOT / "models" / "shadow_seat" / "stacker.json",
    "gated_seat_ppo": ROOT / "models" / "gated_seat" / "ppo.zip",
    "gated_seat_stacker": ROOT / "models" / "gated_seat" / "stacker.pkl",
    "gated_seat_meta": ROOT / "models" / "gated_seat" / "stacker.json",
    "gated_seat_gate_model": ROOT / "models" / "gated_seat" / "gate_model.pkl",
    "gated_seat_manifest": ROOT / "data" / "gated_seat.json",
    "lab_ppo_champion": PPO_CHAMPION,
    "lab_stacker_champion": STACKER_CHAMPION,
    "lab_stacker_meta": ROOT / "models" / "stacker_champion.json",
    "live_state": ROOT / "data" / "live_state.json",
    "live_state_shadow": ROOT / "data" / "live_state_shadow.json",
    "live_state_gated": ROOT / "data" / "live_state_gated.json",
    "paper_db_info_only": ROOT / "data" / "paper.db",
    "paper_shadow_db_info_only": ROOT / "data" / "paper_shadow.db",
    "paper_gated_db_info_only": ROOT / "data" / "paper_gated.db",
}


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard() -> dict:
    return {k: _sha(p) for k, p in GUARD_FILES.items()}


def results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    LAB.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1))


def _windows():
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    p_ups = [votes_hold[k * 264:(k + 1) * 264] for k in range(3)]
    return hold_sup, windows, p_ups


def _extras_ho() -> dict:
    ex = np.load(EXTRAS83_PATH)
    return {k[3:]: ex[k] for k in ex.files if k.startswith("ho_")}


def _seed_outcomes():
    raw = json.loads(CANDIDATES83_PATH.read_text())
    s1 = [(o["close_pos"], o["label"]) for o in raw["outcomes1"]]
    s2 = [(o["close_pos"], o["label"]) for o in raw["outcomes2"]]
    return s1, s2


def _resum_arm(tag: str, windows) -> tuple[float, dict]:
    per_window = {}
    total = 0.0
    for w, w_sup in enumerate(windows):
        closes = w_sup["close"].to_numpy(dtype=float)
        s = 0.0
        for ep in range(10):
            db = LEDGERS / f"{tag}w{w}_ep{ep}.db"
            s += ml.resum_episode_pnl(db, float(closes[ep * 24 + 24]))
        per_window[str(w)] = s
        total += s
    return total, per_window


def _gate_factory(model, threshold: float, seed_both: bool):
    """One Gate84 for a whole arm; the factory swaps each
    window's arrays in, run 83's runner pattern."""
    frame_hold = np.load(FRAME_HOLD_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    ctx = np.load(CTX_PATH)
    extras_ho = _extras_ho()
    gate = m84.Gate84(model, threshold)
    s1, s2 = _seed_outcomes()
    gate.seed_history(m84.STREAM1, s1)
    if seed_both:
        gate.seed_history(m84.STREAM2, s2)

    def factory(w):
        sl = slice(w * 264, (w + 1) * 264)
        gate.set_window(w, frame_hold[sl], votes_hold[sl],
                        ctx["atr_rel_ho"][sl], ctx["vol21_ho"][sl],
                        ctx["atr_raw_ho"][sl],
                        {k: v[sl] for k, v in extras_ho.items()})
        return gate

    return gate, factory


def _arm_record(gres, windows, tag: str, champ_res) -> dict:
    resumed, per_window = _resum_arm(tag, windows)
    diff = abs(resumed - gres["total_pnl"])
    assert diff < 0.01, (
        f"TWO-WAY CHECK FAIL {tag}: ledger re-sum {resumed:.2f} "
        f"vs evaluator {gres['total_pnl']:.2f} (diff {diff:.4f})")
    gs = gres["gate_stats"]
    by = gres.get("gate_by_stream", {"attempts": {}, "vetoes": {}})
    att_total = sum(by["attempts"].values()) if by["attempts"] else None
    vet_total = sum(by["vetoes"].values()) if by["vetoes"] else None
    if att_total is not None:
        assert att_total == gs["entries_attempted"], (
            f"VETO ACCOUNTING FAIL {tag}: gate counted {att_total} "
            f"attempts, env counted {gs['entries_attempted']}")
        assert vet_total == gs["entries_vetoed"], (
            f"VETO ACCOUNTING FAIL {tag}: gate counted {vet_total} "
            f"vetoes, env counted {gs['entries_vetoed']}")
    agg = _agg(gres)
    wins = sum(1 for gw, cw in zip(gres["windows"],
                                   champ_res["windows"])
               if gw["score"] > cw["score"])
    return {
        "agg": agg, "wins": wins,
        "attempted": gs["entries_attempted"],
        "vetoed": gs["entries_vetoed"],
        "filled": gs["entries_filled"],
        "pass_rate": ((gs["entries_attempted"] - gs["entries_vetoed"])
                      / gs["entries_attempted"]
                      if gs["entries_attempted"] else 0.0),
        "veto_rate": (gs["entries_vetoed"] / gs["entries_attempted"]
                      if gs["entries_attempted"] else 0.0),
        "fills": gs["fills"], "fees": gs["fees"],
        "time_in_position": gres["time_in_position"],
        "gate_by_stream": by,
        "resum_total_pnl": resumed, "resum_diff": diff,
        "resum_per_window": per_window}


# ---------------------------------------------------------------- phases
def phase_prep() -> None:
    if results().get("prep"):
        print("prep: checkpoint exists, skipping", flush=True)
        return
    r82 = json.loads(RESULTS82_PATH.read_text())
    r83 = json.loads(RESULTS83_PATH.read_text())
    meta = nightly._champion_meta()
    assert meta.get("run_id") == 73, meta.get("run_id")
    assert meta.get("feature_set") == "pick2_98", meta.get("feature_set")
    assert _sha(PPO_CHAMPION) == r82["frames"]["champion"]["ppo_sha256"]
    assert _sha(STACKER_CHAMPION) == \
        r82["frames"]["champion"]["stacker_sha256"]
    assert _sha(MODEL83_PATH) == MODEL83_SHA256, (
        "run 83's frozen fitted model changed on disk — run 84 "
        "grades THAT model or nothing")

    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    frame_hold = np.load(FRAME_HOLD_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    assert len(train_sup) == r82["frames"]["train_rows"] == 12079
    assert len(hold_sup) == r82["frames"]["holdout_rows"] == 792
    assert str(hold_sup.index[0]) == r82["frames"]["holdout_first"]
    assert str(hold_sup.index[-1]) == r82["frames"]["holdout_last"]
    assert frame_hold.shape == (792, 98)
    assert len(votes_hold) == 792

    d = np.load(DATASET83_PATH)
    X, y = d["X"], d["y"]
    recs = json.loads(ASSEMBLED83_PATH.read_text())
    assert X.shape == (640, m83.N_FEATURES_83), X.shape
    assert len(recs) == len(y) == 640
    streams = np.array([r["stream"] for r in recs], dtype=int)
    assert all((r["stream"] == 1) == (X[i, 113] == 1.0)
               for i, r in enumerate(recs)), \
        "dataset83 rows and assembled records disagree on stream"
    assert int(np.sum(streams == 1)) == 615
    assert int(np.sum(streams == 0)) == 25
    assert r83["fit"]["selected_variant"] == "stacker"
    save_result("prep", {
        "train_rows": len(train_sup), "holdout_rows": len(hold_sup),
        "holdout_first": str(hold_sup.index[0]),
        "holdout_last": str(hold_sup.index[-1]),
        "model83_sha256": MODEL83_SHA256,
        "n_candidates": int(len(y)),
        "n_stream1": int(np.sum(streams == 1)),
        "n_stream2": int(np.sum(streams == 0)),
        "run83_reference": {
            "oof_stacker": r83["fit"]["oof_selected"],
            "g55": {k: r83["duel"]["arms"]["0.55"][k] for k in
                    ("attempted", "vetoed", "filled", "fills", "fees")},
            "g55_score": r83["duel"]["arms"]["0.55"]["agg"]["score"],
            "g55_total_pnl":
                r83["duel"]["arms"]["0.55"]["agg"]["total_pnl"]},
        "run82_reference": {
            "champion_score": r82["duel"]["champion"]["score"],
            "champion_total_pnl": r82["duel"]["champion"]["total_pnl"],
            "champion_bench_pnl": r82["duel"]["champion"]["bench_pnl"],
            "g55_score": r82["duel"]["arms"]["0.55"]["agg"]["score"],
            "g55_total_pnl":
                r82["duel"]["arms"]["0.55"]["agg"]["total_pnl"]},
        "guard_before": _guard()})
    print(f"prep: run 82/83 artifacts verified — holdout "
          f"{hold_sup.index[0]} -> {hold_sup.index[-1]}, frozen "
          f"model83 sha ok, 640 candidates (615 stream 1 / 25 "
          f"stream 2)", flush=True)


def phase_calibrate() -> None:
    if results().get("calibrate"):
        print("calibrate: checkpoint exists, skipping", flush=True)
        return
    prep = results()["prep"]
    d = np.load(DATASET83_PATH)
    X, y = d["X"], d["y"]
    recs = json.loads(ASSEMBLED83_PATH.read_text())
    streams = np.array([r["stream"] for r in recs], dtype=int)
    print("regenerating run 83's stacker OOF probabilities "
          "(TimeSeriesSplit(4), seeded recipe) ...", flush=True)
    t0 = time.time()
    proba = m83.oof_proba_base("stacker", X, y, folds=4)
    metrics = m83.oof_metrics(proba, y)
    ref = prep["run83_reference"]["oof_stacker"]
    assert metrics["n_oof"] == ref["n_oof"] == 512, metrics
    assert abs(metrics["balanced_accuracy"]
               - ref["balanced_accuracy"]) <= 0.005, (
        f"OOF REPRODUCTION FAIL: regenerated balanced accuracy "
        f"{metrics['balanced_accuracy']:.4f} vs run 83's recorded "
        f"{ref['balanced_accuracy']:.4f} — the recipe does not "
        f"reproduce; calibration would be built on sand")
    assert abs((metrics["auc"] or 0) - (ref["auc"] or 0)) <= 0.005, (
        f"OOF REPRODUCTION FAIL (AUC): {metrics['auc']} vs "
        f"{ref['auc']}")
    print(f"  OOF reproduced: n {metrics['n_oof']}, bal_acc "
          f"{metrics['balanced_accuracy']:.4f} (recorded "
          f"{ref['balanced_accuracy']:.4f}), auc {metrics['auc']:.4f} "
          f"(recorded {ref['auc']:.4f}), {time.time() - t0:.0f}s",
          flush=True)
    LAB.mkdir(parents=True, exist_ok=True)
    np.savez(OOF84_PATH, proba=proba, streams=streams)
    cal = m84.calibrated_threshold(proba, streams)
    cal["oof_reproduction"] = {
        "n_oof": metrics["n_oof"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auc": metrics["auc"],
        "recorded_balanced_accuracy": ref["balanced_accuracy"],
        "recorded_auc": ref["auc"]}
    cal["seconds"] = round(time.time() - t0, 1)
    save_result("calibrate", cal)
    print(f"calibrate: tau84 = {cal['threshold']:.6f} (70th "
          f"percentile of {cal['n_calibration']} stream-1 "
          f"OOF-covered probabilities; implied training pass "
          f"rate {cal['implied_pass_rate']:.1%})", flush=True)
    print(f"  PUBLISH BEFORE DUEL — append to {REFS_PATH}: "
          f"tau84 = {cal['threshold']:.6f}", flush=True)


def phase_duel() -> None:
    if results().get("duel"):
        print("duel: checkpoint exists, skipping", flush=True)
        return
    res = results()
    cal = res.get("calibrate")
    assert cal, "calibrate phase has not run"
    tau = float(cal["threshold"])
    refs_text = REFS_PATH.read_text()
    assert f"tau84 = {tau:.6f}" in refs_text, (
        f"PRE-REGISTRATION INCOMPLETE: tau84 = {tau:.6f} is not "
        f"published in {REFS_PATH} — publish it (training-region "
        f"evidence only) before any holdout run")
    ref82 = res["prep"]["run82_reference"]
    ref83 = res["prep"]["run83_reference"]
    _hold, windows, p_ups = _windows()
    model = joblib.load(MODEL83_PATH)

    print("grading champion pair (production evaluator) ...",
          flush=True)
    champ_res = nightly.evaluate_windows(PPO_CHAMPION, windows, p_ups)
    assert champ_res is not None, "champion pair failed to run"
    champ = _agg(champ_res)
    for key, refkey in (("score", "champion_score"),
                        ("total_pnl", "champion_total_pnl"),
                        ("bench_pnl", "champion_bench_pnl")):
        diff = abs(champ[key] - ref82[refkey])
        assert diff < 0.01, (
            f"CHAMPION IDENTITY FAIL {key}: run 84 sees "
            f"{champ[key]} vs the recorded {ref82[refkey]} — the "
            f"holdout is not the same; study aborts")
    print(f"  champion identity confirmed (agg "
          f"{champ['score']:.2f}, P&L {champ['total_pnl']:+.2f})",
          flush=True)

    print("pin 1: pass-all mirror84 == production ...", flush=True)
    passall = m84.evaluate_windows_84(
        PPO_CHAMPION, windows, p_ups, gate_factory=None,
        mode="pair", ledger_dir=LEDGERS, tag="passall84")
    for w in range(3):
        pw, cw = passall["windows"][w], champ_res["windows"][w]
        for key in ("score", "total_pnl", "max_drawdown"):
            assert abs(pw[key] - cw[key]) <= 1e-6, (
                f"EQUIVALENCE FAIL window {w} {key}: mirror84 "
                f"{pw[key]} vs production {cw[key]} — study aborts")
        assert pw["action_mix"] == cw["action_mix"], (
            f"EQUIVALENCE FAIL window {w} action_mix")
    print("  pass-all mirror84 == production champion on all 3 "
          "windows", flush=True)

    print("pin 2: run-84 machinery @ 0.55 reproduces run 83 ...",
          flush=True)
    _g, fac55 = _gate_factory(model, ml.GATE_THRESHOLD, seed_both=True)
    repro = m84.evaluate_windows_84(
        PPO_CHAMPION, windows, p_ups, gate_factory=fac55,
        mode="pair", ledger_dir=LEDGERS, tag="repro83")
    repro_rec = _arm_record(repro, windows, "repro83", champ_res)
    r55 = ref83["g55"]
    assert abs(repro_rec["agg"]["score"] - ref83["g55_score"]) < 0.01
    assert abs(repro_rec["agg"]["total_pnl"]
               - ref83["g55_total_pnl"]) < 0.01
    for k in ("attempted", "vetoed", "filled"):
        assert repro_rec[k] == r55[k], (
            f"REPRODUCTION FAIL {k}: run-84 machinery counts "
            f"{repro_rec[k]}, run 83 recorded {r55[k]}")
    assert abs(repro_rec["fees"] - r55["fees"]) < 0.01
    print(f"  run 83 @0.55 reproduced (agg "
          f"{repro_rec['agg']['score']:.2f}, P&L "
          f"{repro_rec['agg']['total_pnl']:+.2f}, "
          f"{repro_rec['vetoed']}/{repro_rec['attempted']} vetoed)",
          flush=True)

    print(f"grading PRIMARY arm (stream 1 @ tau84 "
          f"{tau:.6f}) ...", flush=True)
    _g, fac_p = _gate_factory(model, tau, seed_both=False)
    prim = m84.evaluate_windows_84(
        PPO_CHAMPION, windows, p_ups, gate_factory=fac_p,
        mode="pair", ledger_dir=LEDGERS, tag="primary84")
    prim_rec = _arm_record(prim, windows, "primary84", champ_res)
    print(f"  primary: agg {prim_rec['agg']['score']:.2f}, P&L "
          f"{prim_rec['agg']['total_pnl']:+.2f}, wins "
          f"{prim_rec['wins']}/3, vetoed {prim_rec['vetoed']}/"
          f"{prim_rec['attempted']} (pass "
          f"{prim_rec['pass_rate']:.0%}), filled "
          f"{prim_rec['filled']}, fees ${prim_rec['fees']:.2f}, "
          f"time-in-position {prim_rec['time_in_position']:.1%}",
          flush=True)

    print(f"grading SECONDARY arm (both streams @ tau84 "
          f"{tau:.6f}) ...", flush=True)
    _g, fac_s = _gate_factory(model, tau, seed_both=True)
    sec = m84.evaluate_windows_84(
        PPO_CHAMPION, windows, p_ups, gate_factory=fac_s,
        mode="combined", ledger_dir=LEDGERS, tag="both84")
    sec_rec = _arm_record(sec, windows, "both84", champ_res)
    print(f"  secondary: agg {sec_rec['agg']['score']:.2f}, P&L "
          f"{sec_rec['agg']['total_pnl']:+.2f}, wins "
          f"{sec_rec['wins']}/3, vetoed {sec_rec['vetoed']}/"
          f"{sec_rec['attempted']} (pass "
          f"{sec_rec['pass_rate']:.0%}), filled "
          f"{sec_rec['filled']}, fees ${sec_rec['fees']:.2f}, "
          f"time-in-position {sec_rec['time_in_position']:.1%}, "
          f"by stream {sec_rec['gate_by_stream']}", flush=True)

    formula = prim_rec["agg"]["score"] > champ["score"] \
        and prim_rec["wins"] >= 2
    promotable, floor_fail = gate_decision(prim_rec["agg"], formula)
    cgs = passall["gate_stats"]
    save_result("duel", {
        "champion": champ,
        "champion_gate_stats": {
            "attempted": cgs["entries_attempted"],
            "filled": cgs["entries_filled"], "fills": cgs["fills"],
            "fees": cgs["fees"]},
        "champion_time_in_position": passall["time_in_position"],
        "repro83": repro_rec, "primary": prim_rec,
        "secondary": sec_rec,
        "formula_pass": bool(formula), "floor_fail": floor_fail,
        "promotable": bool(promotable)})
    print(f"duel: formula={formula} floor_fail={floor_fail} "
          f"promotable={promotable}", flush=True)


def phase_register() -> None:
    res = results()
    if res.get("registered"):
        print("register: already done, skipping", flush=True)
        return
    duel, cal = res["duel"], res["calibrate"]
    ref82, ref83 = res["prep"]["run82_reference"], \
        res["prep"]["run83_reference"]
    champ = duel["champion"]
    prim, sec = duel["primary"], duel["secondary"]
    promotable, formula = duel["promotable"], duel["formula_pass"]
    floor_fail = duel["floor_fail"]
    tau = float(cal["threshold"])
    cgs = duel["champion_gate_stats"]

    rel = (f"activity-calibrated gate (tau84 {tau:.4f} = 70th "
           f"percentile of {cal['n_calibration']} stream-1 "
           f"training OOF probabilities, implied pass "
           f"{cal['implied_pass_rate']:.0%}) on run 83's frozen "
           f"model: aggregate {prim['agg']['score']:.0f} vs "
           f"champion {champ['score']:.0f}, won {prim['wins']}/3 "
           f"windows, P&L {prim['agg']['total_pnl']:+,.0f} vs "
           f"champion {champ['total_pnl']:+,.0f} (buy-and-hold "
           f"{prim['agg']['bench_pnl']:+,.0f}); realized activity: "
           f"{prim['attempted']} attempts, pass "
           f"{prim['pass_rate']:.0%} ({prim['vetoed']} vetoed), "
           f"{prim['filled']} filled, fees ${prim['fees']:.0f} vs "
           f"champion ${cgs['fees']:.0f}, time-in-position "
           f"{prim['time_in_position']:.0%}; vs run 83 @0.55 (agg "
           f"{ref83['g55_score']:.0f}, P&L "
           f"{ref83['g55_total_pnl']:+,.0f}, 1 filled) and run 82 "
           f"@0.55 (agg {ref82['g55_score']:.0f}, P&L "
           f"{ref82['g55_total_pnl']:+,.0f}); secondary both-"
           f"streams arm: agg {sec['agg']['score']:.0f}, P&L "
           f"{sec['agg']['total_pnl']:+,.0f}, {sec['filled']} "
           f"filled, time-in-position "
           f"{sec['time_in_position']:.0%}")
    if promotable:
        verdict = "PASS"
        reason = (f"Meta-Label Gate — Pick A-84: {rel} (gate "
                  f"pass incl. two-part floor) — WOULD PROMOTE: "
                  f"held as PENDING KASS APPROVAL under the "
                  f"promotion lock; champion kept, nothing "
                  f"installed")
    elif formula and floor_fail is not None:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A-84: passed the "
                  f"relative gate ({rel}) but was not promoted — "
                  f"{floor_fail}; champion kept")
    else:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A-84: champion kept: "
                  f"{rel}")
    metrics = {
        "family": ml.FAMILY,
        "challenger": prim["agg"],
        "champion": champ,
        "reason": reason,
        "deep": False,
        "timesteps": 0,
        "pending_approval": bool(promotable),
        "reward_curve": None,
        "telemetry": None,
        "meta_label": {
            "label": "Pick A-84",
            "verdict": verdict,
            "design": {
                "threshold": tau,
                "threshold_rule": "70th percentile of the frozen "
                                  "model's TimeSeriesSplit(4) "
                                  "OOF probabilities over the "
                                  "stream-1 training candidates "
                                  "(~30% training pass rate); "
                                  "chosen on training-region "
                                  "evidence only, published before "
                                  "the duel, never adjusted on "
                                  "holdout behavior",
                "meta_model": "run 83's frozen fitted stacker "
                              "(sha256 ddbe2b48…), unchanged — no "
                              "refit; run 84 changes the operating "
                              "point only",
                "features": "run 83's frozen 114-vector",
                "label": "run 83's FIFO in-context tranche "
                         "attribution (unchanged)",
                "signal": "primary (graded): stream 1, the "
                          "champion pair's entries, at tau84; "
                          "secondary (reported): both streams at "
                          "tau84, PPO precedence, the vote-"
                          "crossing rule proposes only on steps "
                          "where the PPO holds",
            },
            "calibration": {
                "tau84": tau,
                "quantile": cal["quantile"],
                "n_calibration": cal["n_calibration"],
                "implied_pass_rate": cal["implied_pass_rate"],
                "oof_reproduction": cal["oof_reproduction"]},
            "primary": {
                "agg_score": prim["agg"]["score"],
                "total_pnl": prim["agg"]["total_pnl"],
                "wins": prim["wins"],
                "attempted": prim["attempted"],
                "vetoed": prim["vetoed"], "filled": prim["filled"],
                "pass_rate": prim["pass_rate"],
                "fills": prim["fills"], "fees": prim["fees"],
                "time_in_position": prim["time_in_position"]},
            "secondary_both_streams": {
                "agg_score": sec["agg"]["score"],
                "total_pnl": sec["agg"]["total_pnl"],
                "wins": sec["wins"],
                "attempted": sec["attempted"],
                "vetoed": sec["vetoed"], "filled": sec["filled"],
                "pass_rate": sec["pass_rate"],
                "fills": sec["fills"], "fees": sec["fees"],
                "time_in_position": sec["time_in_position"],
                "gate_by_stream": sec["gate_by_stream"]},
            "run83_reproduction_pin": {
                "agg_score": duel["repro83"]["agg"]["score"],
                "total_pnl": duel["repro83"]["agg"]["total_pnl"],
                "attempted": duel["repro83"]["attempted"],
                "vetoed": duel["repro83"]["vetoed"],
                "filled": duel["repro83"]["filled"]},
            "comparisons": {
                "run83_055": {"agg_score": ref83["g55_score"],
                              "total_pnl": ref83["g55_total_pnl"]},
                "run82_055": {"agg_score": ref82["g55_score"],
                              "total_pnl": ref82["g55_total_pnl"]},
                "champion": {"agg_score": champ["score"],
                             "total_pnl": champ["total_pnl"],
                             "bench_pnl": champ["bench_pnl"]}},
            "champion_entries": {"attempted": cgs["attempted"],
                                 "filled": cgs["filled"],
                                 "fills": cgs["fills"],
                                 "fees": cgs["fees"]},
            "two_way_check": {
                "primary_resum_total_pnl": prim["resum_total_pnl"],
                "primary_evaluator_total_pnl":
                    prim["agg"]["total_pnl"],
                "primary_abs_diff": prim["resum_diff"],
                "secondary_resum_total_pnl": sec["resum_total_pnl"],
                "secondary_abs_diff": sec["resum_diff"]},
        },
    }
    nightly.log_run("ppo", metrics, False)
    import sqlite3
    with sqlite3.connect(nightly.registry_db()) as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    nightly.add_lab_note(
        "Meta-Label Gate — Pick A-84 — PENDING KASS APPROVAL"
        if promotable else "Meta-Label Gate — Pick A-84", reason)
    guard_after = _guard()
    before = res["prep"]["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("registered", {"run_id": int(run_id),
                               "verdict": verdict, "reason": reason,
                               "guard_after_ok": True})
    print(f"registered run {run_id}: verdict {verdict} — {reason}",
          flush=True)


PHASES = {"prep": phase_prep, "calibrate": phase_calibrate,
          "duel": phase_duel, "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== meta-label84 study runner done ===", flush=True)


if __name__ == "__main__":
    main()
