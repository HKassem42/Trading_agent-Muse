#!/usr/bin/env python
"""Meta-Label Gate run 83 — the strengthened gate: study runner
(2026-10-30).

Kass's order: "Fix whatever you want with 82 and run a 83 and
see if you can improve it." Design frozen in
refs/meta-label-study-83-2026-10-30.md BEFORE any run-83 code
ran. Run 83 fixes run 82's four diagnosed weaknesses: (1) near-
chance meta-model eyes -> a six-variant model zoo (run 82's
LightGBM control, the program's stacker recipe, each with
isotonic/Platt calibration) selected on training-region OOF
balanced accuracy only; (2) thin data -> two candidate streams
(the champion pair replayed at episode offsets 0/8/16, plus a
stacker-vote crossing rule policy); (3) standalone labels ->
FIFO tranche attribution inside each replay trajectory (the
in-context label); (4) uncalibrated probabilities -> the
calibration variants + ten causal context features (114-vector).

Data is run 82's persisted artifacts, byte-identical — the
holdout (2026-09-27 -> 2026-10-30) is literally the same
candles, votes and frames run 82 was graded on; NO fresh
market fetch (a fresh fetch would shift the holdout). Grading
is run 82's frozen protocol: champion pair via the production
evaluator, the 83 mirror pass-all equivalence pin, gated arms
at 0.50/0.55/0.60 with 0.55 graded, formula + gate_decision()
floor, ledger re-sum two-way check per arm. Improvement bar
(frozen): the 0.55 arm's aggregate AND total P&L must beat run
82's -36.79 / +$37.31.

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded before/
after; the study never opens paper.db / paper_shadow.db).
Writes live only under data/meta_label83/, plus the registry
run and the lab note at the end. Run 82's module, runner and
artifacts are read-only inputs — run 82 stays reproducible.

Phases checkpoint under data/meta_label83/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  prep -> extract -> assemble -> compare -> fit -> duel ->
  register

Usage: .venv/bin/python scripts/run_meta_label83_study.py [phase]
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
from training import nightly  # noqa: E402
from training.nightly import _agg, gate_decision  # noqa: E402

LAB82 = ROOT / "data" / "meta_label"
LAB = ROOT / "data" / "meta_label83"
RESULTS_PATH = LAB / "results.json"
RESULTS82_PATH = LAB82 / "results.json"
TRAIN_SUP_PATH = LAB82 / "train_sup.pkl"
HOLD_SUP_PATH = LAB82 / "hold_sup.pkl"
FRAME_TRAIN_PATH = LAB82 / "frame_train.npy"
FRAME_HOLD_PATH = LAB82 / "frame_hold.npy"
VOTES_TRAIN_PATH = LAB82 / "votes_train.npy"
VOTES_HOLD_PATH = LAB82 / "votes_hold.npy"
CTX_PATH = LAB82 / "context.npz"
EXTRAS_PATH = LAB / "extras.npz"
CANDIDATES_PATH = LAB / "candidates83.json"
DATASET_PATH = LAB / "dataset83.npz"
MODEL_PATH = LAB / "meta_model83.pkl"
LEDGERS = LAB / "eval_ledgers"
SCRATCH = LAB / "scratch"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"

GUARD_FILES = {
    "live_seat_ppo": ROOT / "models" / "live_seat" / "ppo.zip",
    "live_seat_stacker": ROOT / "models" / "live_seat" / "stacker.pkl",
    "live_seat_meta": ROOT / "models" / "live_seat" / "stacker.json",
    "shadow_seat_ppo": ROOT / "models" / "shadow_seat" / "ppo.zip",
    "shadow_seat_stacker": ROOT / "models" / "shadow_seat" / "stacker.pkl",
    "shadow_seat_meta": ROOT / "models" / "shadow_seat" / "stacker.json",
    "lab_ppo_champion": PPO_CHAMPION,
    "lab_stacker_champion": STACKER_CHAMPION,
    "lab_stacker_meta": ROOT / "models" / "stacker_champion.json",
    "live_state": ROOT / "data" / "live_state.json",
    "live_state_shadow": ROOT / "data" / "live_state_shadow.json",
    "paper_db_info_only": ROOT / "data" / "paper.db",
    "paper_shadow_db_info_only": ROOT / "data" / "paper_shadow.db",
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


def results82() -> dict:
    return json.loads(RESULTS82_PATH.read_text())


def _windows():
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    p_ups = [votes_hold[k * 264:(k + 1) * 264] for k in range(3)]
    return hold_sup, windows, p_ups


def _extras(sup, votes, ctx, prefix: str) -> dict:
    closes = sup["close"].to_numpy(dtype=float)
    hours = np.asarray(sup.index.hour, dtype=float)
    return m83.build_extra_context(votes, ctx[f"atr_raw_{prefix}"],
                                   ctx[f"vol21_{prefix}"], closes, hours)


# ---------------------------------------------------------------- phases
def phase_prep() -> None:
    if results().get("prep"):
        print("prep: checkpoint exists, skipping", flush=True)
        return
    r82 = results82()
    meta = nightly._champion_meta()
    assert meta.get("run_id") == 73, meta.get("run_id")
    assert meta.get("feature_set") == "pick2_98", meta.get("feature_set")
    assert _sha(PPO_CHAMPION) == r82["frames"]["champion"]["ppo_sha256"]
    assert _sha(STACKER_CHAMPION) == \
        r82["frames"]["champion"]["stacker_sha256"]

    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    frame_train = np.load(FRAME_TRAIN_PATH)
    frame_hold = np.load(FRAME_HOLD_PATH)
    votes_train = np.load(VOTES_TRAIN_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    ctx = np.load(CTX_PATH)
    assert len(train_sup) == r82["frames"]["train_rows"] == 12079
    assert len(hold_sup) == r82["frames"]["holdout_rows"] == 792
    assert str(hold_sup.index[0]) == r82["frames"]["holdout_first"]
    assert str(hold_sup.index[-1]) == r82["frames"]["holdout_last"]
    assert str(train_sup.index[-1]) == r82["frames"]["train_last"]
    assert frame_train.shape == (len(train_sup), 98)
    assert frame_hold.shape == (792, 98)
    assert len(votes_train) == len(train_sup)
    assert len(votes_hold) == 792
    extras_tr = _extras(train_sup, votes_train, ctx, "tr")
    extras_ho = _extras(hold_sup, votes_hold, ctx, "ho")
    LAB.mkdir(parents=True, exist_ok=True)
    np.savez(EXTRAS_PATH,
             **{f"tr_{k}": v for k, v in extras_tr.items()},
             **{f"ho_{k}": v for k, v in extras_ho.items()})
    save_result("prep", {
        "train_rows": len(train_sup), "holdout_rows": len(hold_sup),
        "train_last": str(train_sup.index[-1]),
        "holdout_first": str(hold_sup.index[0]),
        "holdout_last": str(hold_sup.index[-1]),
        "run82_reference": {
            "champion_score": r82["duel"]["champion"]["score"],
            "champion_total_pnl": r82["duel"]["champion"]["total_pnl"],
            "champion_bench_pnl": r82["duel"]["champion"]["bench_pnl"],
            "g55_score": r82["duel"]["arms"]["0.55"]["agg"]["score"],
            "g55_total_pnl":
                r82["duel"]["arms"]["0.55"]["agg"]["total_pnl"],
            "g55_gate_stats": {
                k: r82["duel"]["arms"]["0.55"][k]
                for k in ("attempted", "vetoed", "filled", "fills",
                          "fees")},
            "oof82": r82["fit"]["oof"]},
        "guard_before": _guard()})
    print(f"prep: run 82 artifacts verified — train {len(train_sup)} "
          f"rows, holdout {len(hold_sup)} rows "
          f"({hold_sup.index[0]} -> {hold_sup.index[-1]}); extras built",
          flush=True)


def phase_extract() -> None:
    if CANDIDATES_PATH.exists():
        print("extract: checkpoint exists, skipping", flush=True)
        return
    from stable_baselines3 import PPO

    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    votes = np.load(VOTES_TRAIN_PATH)
    model = PPO.load(PPO_CHAMPION)

    def policy(obs):
        return int(model.predict(obs, deterministic=True)[0])

    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    db1 = SCRATCH / "stream1.db"
    db1.unlink(missing_ok=True)
    c1, o1, d1 = m83.extract_stream_episodes(
        train_sup, votes, policy, db1, offsets=(0, 8, 16),
        stream=m83.STREAM_PAIR)
    t1 = time.time()
    print(f"  stream 1 (pair, offsets 0/8/16): {len(c1)} candidates, "
          f"{d1} refused, {t1 - t0:.0f}s", flush=True)
    db2 = SCRATCH / "stream2.db"
    db2.unlink(missing_ok=True)
    c2, o2, d2 = m83.extract_stream_episodes(
        train_sup, votes, None, db2, offsets=(0,),
        stream=m83.STREAM_VOTES, rule=True)
    print(f"  stream 2 (vote-crossing rule): {len(c2)} candidates, "
          f"{d2} refused, {time.time() - t1:.0f}s", flush=True)
    ml.validate_candidates(
        [(c["decision_pos"], c["side"]) for c in c1 + c2],
        len(train_sup))
    CANDIDATES_PATH.write_text(json.dumps({
        "stream1": c1, "stream2": c2, "outcomes1": o1,
        "outcomes2": o2, "dropped1": d1, "dropped2": d2}))
    save_result("extract", {
        "stream1_candidates": len(c1), "stream2_candidates": len(c2),
        "stream1_dropped_refused": d1, "stream2_dropped_refused": d2,
        "seconds": round(time.time() - t0, 1)})
    print(f"extract: {len(c1) + len(c2)} candidates total", flush=True)


def phase_assemble() -> None:
    if DATASET_PATH.exists():
        print("assemble: checkpoint exists, skipping", flush=True)
        return
    raw = json.loads(CANDIDATES_PATH.read_text())
    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    frame = np.load(FRAME_TRAIN_PATH)
    votes = np.load(VOTES_TRAIN_PATH)
    ctx = np.load(CTX_PATH)
    ex = np.load(EXTRAS_PATH)
    extras_tr = {k[3:]: ex[k] for k in ex.files if k.startswith("tr_")}
    outcomes_by_stream = {
        m83.STREAM_PAIR: [(o["close_pos"], o["label"])
                          for o in raw["outcomes1"]],
        m83.STREAM_VOTES: [(o["close_pos"], o["label"])
                           for o in raw["outcomes2"]],
    }
    label_by_tag = {}
    for o in raw["outcomes1"] + raw["outcomes2"]:
        label_by_tag[o["tag"]] = o

    pooled, deduped = [], 0
    for stream_key, cands in (("stream1", raw["stream1"]),
                              ("stream2", raw["stream2"])):
        seen = set()
        for c in cands:
            key = (c["decision_pos"], c["side"], c["stream"])
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            pooled.append(c)
    pooled.sort(key=lambda c: (c["decision_pos"], c["stream"]))
    X, y, recs = [], [], []
    for c in pooled:
        o = label_by_tag[c["tag"]]
        assert o["close_pos"] > c["decision_pos"], \
            f"outcome closes before its decision: {c} vs {o}"
        win20, winexp = m83.stream_win_features(
            outcomes_by_stream[c["stream"]], c["decision_pos"])
        X.append(m83.feature_row_83(
            frame, votes, ctx["atr_rel_tr"], ctx["vol21_tr"],
            ctx["atr_raw_tr"], extras_tr, c["decision_pos"], c["side"],
            win20, winexp, float(c["stream"])))
        y.append(o["label"])
        recs.append({**c, "label": o["label"],
                     "contribution": o["contribution"],
                     "close_pos": o["close_pos"]})
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    assert X.shape == (len(pooled), m83.N_FEATURES_83), X.shape
    assert len(np.unique(y)) == 2, "pooled labels are single-class"
    np.savez(DATASET_PATH, X=X, y=y)
    (LAB / "candidates83_assembled.json").write_text(json.dumps(recs))
    per_stream = {}
    for s in (m83.STREAM_PAIR, m83.STREAM_VOTES):
        ys = [r["label"] for r in recs if r["stream"] == s]
        per_stream[str(s)] = {
            "n": len(ys), "base_rate": float(np.mean(ys)) if ys else None}
    save_result("assemble", {
        "n_pooled": len(pooled), "deduped_dropped": deduped,
        "base_rate": float(y.mean()), "per_stream": per_stream,
        "mean_contribution": float(np.mean(
            [r["contribution"] for r in recs]))})
    print(f"assemble: {len(pooled)} pooled candidates "
          f"({deduped} exact duplicates dropped), base rate "
          f"{y.mean():.3f}", flush=True)


def phase_compare() -> None:
    if results().get("compare"):
        print("compare: checkpoint exists, skipping", flush=True)
        return
    d = np.load(DATASET_PATH)
    X, y = d["X"], d["y"]
    print(f"comparing {len(m83.VARIANTS)} variants on training OOF "
          f"({len(y)} candidates) ...", flush=True)
    t0 = time.time()
    table = m83.oof_table(X, y)
    for v in m83.VARIANTS:
        m = table[v]
        print(f"  {v}: bal_acc {m['balanced_accuracy']:.4f}, "
              f"auc {m['auc']}, n_oof {m['n_oof']}", flush=True)
    save_result("compare", {"oof_table": table,
                            "seconds": round(time.time() - t0, 1)})
    print(f"compare: done in {time.time() - t0:.0f}s", flush=True)


def phase_fit() -> None:
    if MODEL_PATH.exists():
        print("fit: checkpoint exists, skipping", flush=True)
        return
    table = results()["compare"]["oof_table"]
    selected = m83.select_variant(table)
    d = np.load(DATASET_PATH)
    X, y = d["X"], d["y"]
    t0 = time.time()
    model = m83.fit_final_variant(selected, X, y)
    joblib.dump(model, MODEL_PATH)
    save_result("fit", {"selected_variant": selected,
                        "oof_selected": table[selected],
                        "n_train": int(len(y)),
                        "seconds": round(time.time() - t0, 1)})
    print(f"fit: selected {selected} (OOF bal_acc "
          f"{table[selected]['balanced_accuracy']:.4f}); final model "
          f"fit on {len(y)} candidates", flush=True)


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


def phase_duel() -> None:
    if results().get("duel"):
        print("duel: checkpoint exists, skipping", flush=True)
        return
    res = results()
    ref = res["prep"]["run82_reference"]
    _hold, windows, p_ups = _windows()
    frame_hold = np.load(FRAME_HOLD_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    ctx = np.load(CTX_PATH)
    ex = np.load(EXTRAS_PATH)
    extras_ho = {k[3:]: ex[k] for k in ex.files if k.startswith("ho_")}
    model = joblib.load(MODEL_PATH)
    raw = json.loads(CANDIDATES_PATH.read_text())
    seed = [(o["close_pos"], o["label"]) for o in raw["outcomes1"]]

    print("grading champion pair (production evaluator) ...", flush=True)
    champ_res = nightly.evaluate_windows(PPO_CHAMPION, windows, p_ups)
    assert champ_res is not None, "champion pair failed to run"
    champ = _agg(champ_res)
    for key, refkey in (("score", "champion_score"),
                        ("total_pnl", "champion_total_pnl"),
                        ("bench_pnl", "champion_bench_pnl")):
        diff = abs(champ[key] - ref[refkey])
        assert diff < 0.01, (
            f"CHAMPION IDENTITY FAIL {key}: run 83 sees {champ[key]} "
            f"vs run 82's recorded {ref[refkey]} — the holdout is not "
            f"the same; study aborts, nothing registers")
    print(f"  champion identity vs run 82 confirmed "
          f"(agg {champ['score']:.2f}, P&L {champ['total_pnl']:+.2f}, "
          f"bench {champ['bench_pnl']:+.2f})", flush=True)

    print("grading pass-all mirror83 (equivalence pin) ...", flush=True)
    passall = m83.evaluate_gated_windows_83(
        PPO_CHAMPION, windows, p_ups, gate_factory=None,
        ledger_dir=LEDGERS, tag="passall83")
    for w in range(3):
        pw, cw = passall["windows"][w], champ_res["windows"][w]
        for key in ("score", "total_pnl", "max_drawdown"):
            assert abs(pw[key] - cw[key]) <= 1e-6, (
                f"EQUIVALENCE FAIL window {w} {key}: mirror83 "
                f"{pw[key]} vs production {cw[key]} — study aborts")
        assert pw["action_mix"] == cw["action_mix"], (
            f"EQUIVALENCE FAIL window {w} action_mix")
    print("  pass-all mirror83 == production champion on all 3 windows",
          flush=True)

    arms: dict = {}
    for t in ml.SENSITIVITY_THRESHOLDS:
        tag = f"g83_{int(round(t * 100))}"
        gate = m83.MetaGate83(model, threshold=t)
        gate.seed_history(seed)

        def factory(w, gate=gate):
            sl = slice(w * 264, (w + 1) * 264)
            gate.set_window(w, frame_hold[sl], votes_hold[sl],
                            ctx["atr_rel_ho"][sl], ctx["vol21_ho"][sl],
                            ctx["atr_raw_ho"][sl],
                            {k: v[sl] for k, v in extras_ho.items()})
            return gate

        print(f"grading gated arm83 @ {t:.2f} ...", flush=True)
        gres = m83.evaluate_gated_windows_83(
            PPO_CHAMPION, windows, p_ups, gate_factory=factory,
            ledger_dir=LEDGERS, tag=tag)
        resumed, per_window = _resum_arm(tag, windows)
        diff = abs(resumed - gres["total_pnl"])
        assert diff < 0.01, (
            f"TWO-WAY CHECK FAIL @ {t}: ledger re-sum {resumed:.2f} "
            f"vs evaluator {gres['total_pnl']:.2f} (diff {diff:.4f})")
        agg = _agg(gres)
        wins = sum(1 for gw, cw in zip(gres["windows"],
                                       champ_res["windows"])
                   if gw["score"] > cw["score"])
        gs = gres["gate_stats"]
        arms[f"{t:.2f}"] = {
            "agg": agg, "wins": wins,
            "attempted": gs["entries_attempted"],
            "vetoed": gs["entries_vetoed"],
            "filled": gs["entries_filled"],
            "veto_rate": (gs["entries_vetoed"] / gs["entries_attempted"]
                          if gs["entries_attempted"] else 0.0),
            "fills": gs["fills"], "fees": gs["fees"],
            "resum_total_pnl": resumed, "resum_diff": diff,
            "resum_per_window": per_window}
        print(f"  @ {t:.2f}: agg {agg['score']:.2f} vs champion "
              f"{champ['score']:.2f}, P&L {agg['total_pnl']:+.2f}, "
              f"wins {wins}/3, vetoed {gs['entries_vetoed']}/"
              f"{gs['entries_attempted']}, fees ${gs['fees']:.2f}, "
              f"re-sum diff ${diff:.4f}", flush=True)

    g55 = arms["0.55"]
    formula = g55["agg"]["score"] > champ["score"] and g55["wins"] >= 2
    promotable, floor_fail = gate_decision(g55["agg"], formula)
    improved = bool(g55["agg"]["score"] > ref["g55_score"]
                    and g55["agg"]["total_pnl"] > ref["g55_total_pnl"])
    cgs = passall["gate_stats"]
    save_result("duel", {
        "champion": champ,
        "champion_gate_stats": {
            "attempted": cgs["entries_attempted"],
            "filled": cgs["entries_filled"], "fills": cgs["fills"],
            "fees": cgs["fees"]},
        "arms": arms, "formula_pass": bool(formula),
        "floor_fail": floor_fail, "promotable": bool(promotable),
        "improved_vs_run82": improved})
    print(f"duel: formula={formula} floor_fail={floor_fail} "
          f"promotable={promotable} improved_vs_82={improved}",
          flush=True)


def phase_register() -> None:
    res = results()
    if res.get("registered"):
        print("register: already done, skipping", flush=True)
        return
    duel, ref = res["duel"], res["prep"]["run82_reference"]
    champ, arms = duel["champion"], duel["arms"]
    g55 = arms["0.55"]
    promotable, formula = duel["promotable"], duel["formula_pass"]
    floor_fail = duel["floor_fail"]
    improved = duel["improved_vs_run82"]
    asm, fit = res["assemble"], res["fit"]
    table = res["compare"]["oof_table"]
    selected = fit["selected_variant"]
    sel_oof = table[selected]
    cgs = duel["champion_gate_stats"]

    rel = (f"gated pair aggregate {g55['agg']['score']:.0f} vs champion "
           f"{champ['score']:.0f}, won {g55['wins']}/3 windows, P&L "
           f"{g55['agg']['total_pnl']:+,.0f} vs champion "
           f"{champ['total_pnl']:+,.0f} (buy-and-hold "
           f"{g55['agg']['bench_pnl']:+,.0f}); vetoed "
           f"{g55['vetoed']}/{g55['attempted']} candidate entries "
           f"({g55['veto_rate']:.0%}), fees ${g55['fees']:.0f} vs "
           f"champion ${cgs['fees']:.0f}; vs run 82's gate (agg "
           f"{ref['g55_score']:.0f}, P&L {ref['g55_total_pnl']:+,.0f}): "
           + ("IMPROVED on both frozen numbers"
              if improved else
              "did NOT improve on both frozen numbers"))
    model_rel = (f"meta-model '{selected}' selected from "
                 f"{len(table)} variants on training OOF (bal acc "
                 f"{sel_oof['balanced_accuracy']:.3f} vs run 82's "
                 f"LightGBM {ref['oof82'].get('balanced_accuracy'):.3f}); "
                 f"{asm['n_pooled']} in-context-labeled candidates "
                 f"(base rate {asm['base_rate']:.3f})")
    if promotable:
        verdict = "PASS"
        reason = (f"Meta-Label Gate — Pick A-83: {rel}; {model_rel} "
                  f"(gate pass incl. two-part floor) — WOULD PROMOTE: "
                  f"held as PENDING KASS APPROVAL under the promotion "
                  f"lock; champion kept, nothing installed")
    elif formula and floor_fail is not None:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A-83: passed the relative "
                  f"gate ({rel}) but was not promoted — {floor_fail}; "
                  f"{model_rel}; champion kept")
    else:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A-83: champion kept: "
                  f"{rel}; {model_rel}")

    sens = {}
    for t, a in arms.items():
        sens[t] = {"agg_score": a["agg"]["score"],
                   "total_pnl": a["agg"]["total_pnl"], "wins": a["wins"],
                   "vetoed": a["vetoed"], "attempted": a["attempted"],
                   "veto_rate": a["veto_rate"], "fees": a["fees"],
                   "resum_total_pnl": a["resum_total_pnl"],
                   "resum_diff": a["resum_diff"]}
    metrics = {
        "family": ml.FAMILY,
        "challenger": g55["agg"],
        "champion": champ,
        "reason": reason,
        "deep": False,
        "timesteps": 0,
        "pending_approval": bool(promotable),
        "reward_curve": None,
        "telemetry": None,
        "meta_label": {
            "label": "Pick A-83",
            "verdict": verdict,
            "design": {
                "threshold": ml.GATE_THRESHOLD,
                "sensitivity_thresholds": list(ml.SENSITIVITY_THRESHOLDS),
                "meta_model": f"{selected} (zoo of {len(table)}, "
                              f"selected on training OOF balanced "
                              f"accuracy, ties to AUC)",
                "features": "114 = run 82's 104 + [vote slope 6, "
                            "vote slope 24, stream win-rate last-20, "
                            "stream win-rate expanding, stop distance "
                            "(ATR units), volatility percentile, "
                            "drawdown from 100-high, hour sin, hour "
                            "cos, stream id]",
                "label": "FIFO tranche attribution inside the "
                         "replay trajectory: the candidate entry "
                         "tranche's realized contribution > 0 "
                         "(pro-rata fees; open tranches marked at "
                         "episode end, entry fee only)",
                "signal": "stream 1: champion pair (PPO run 44 + "
                          "stacker run 73 votes) replayed at "
                          "episode offsets 0/8/16; stream 2: "
                          "vote-crossing rule policy (training "
                          "only); the duel gates stream 1 only",
            },
            "candidates": asm["n_pooled"],
            "candidates_by_stream": {
                k: v["n"] for k, v in asm["per_stream"].items()},
            "deduped_dropped": asm["deduped_dropped"],
            "refused_dropped": (res["extract"]
                                ["stream1_dropped_refused"]
                                + res["extract"]
                                ["stream2_dropped_refused"]),
            "base_rate": asm["base_rate"],
            "oof_table": table,
            "selected_variant": selected,
            "oof": sel_oof,
            "run82_comparison": {
                "run82_agg_score": ref["g55_score"],
                "run82_total_pnl": ref["g55_total_pnl"],
                "improved": improved},
            "champion_entries": {"attempted": cgs["attempted"],
                                 "filled": cgs["filled"],
                                 "fills": cgs["fills"],
                                 "fees": cgs["fees"]},
            "thresholds": sens,
            "two_way_check": {
                "gated_055_resum_total_pnl": g55["resum_total_pnl"],
                "gated_055_evaluator_total_pnl":
                    g55["agg"]["total_pnl"],
                "abs_diff": g55["resum_diff"]},
        },
    }
    nightly.log_run("ppo", metrics, False)
    import sqlite3
    with sqlite3.connect(nightly.registry_db()) as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    nightly.add_lab_note(
        "Meta-Label Gate — Pick A-83 — PENDING KASS APPROVAL"
        if promotable else "Meta-Label Gate — Pick A-83", reason)
    guard_after = _guard()
    before = res["prep"]["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("registered", {"run_id": int(run_id), "verdict": verdict,
                               "reason": reason,
                               "guard_after_ok": True})
    print(f"registered run {run_id}: verdict {verdict} — {reason}",
          flush=True)


PHASES = {"prep": phase_prep, "extract": phase_extract,
          "assemble": phase_assemble, "compare": phase_compare,
          "fit": phase_fit, "duel": phase_duel,
          "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== meta-label83 study runner done ===", flush=True)


if __name__ == "__main__":
    main()
