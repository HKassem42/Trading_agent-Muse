#!/usr/bin/env python
"""Meta-Label Gate — Pick A: the study runner (2026-10-30).

Kass delegated the next-system pick on 2026-10-30 ("You go ahead
and pick our next model. You choose"); Muse picked A, the
tradeability gate / meta-labeling system. The design was frozen
in refs/meta-label-study-2026-10-30.md BEFORE any of this code
ran. In one paragraph: replay the champion pair (PPO run 44 on
stacker run 73's pick2_98 votes) over the TRAINING region
exactly as the nightly duel replays it; every entry attempt is
a candidate; label each candidate by a standalone production-
broker simulation of that single trade (1 iff net P&L > 0 after
fees); fit a LightGBM meta-model on the 104-vector at each
decision candle (the 98 pick2 values the champion reads +
vote/context); grade the gated pair against the ungated
champion pair in the standard 3-window duel — an entry executes
only if P(profitable) >= 0.55 — under the formula + gate_
decision()'s two-part floor.

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded before/
after; the study never opens paper.db / paper_shadow.db).
Writes live only under data/meta_label/, plus the registry run
and the lab note at the end. A pass records WOULD PROMOTE —
PENDING KASS APPROVAL (promoted stays 0); anything less is the
fifteenth null, recorded as such.

Phases checkpoint under data/meta_label/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  frames -> votes -> candidates -> labels -> fit -> duel ->
  register

Usage: .venv/bin/python scripts/run_meta_label_study.py [phase]
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

from data.market_data import fetch_klines  # noqa: E402
from features import pick2_features  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, make_supervised)
from models.muse_pick import crossfit_p_up_pick  # noqa: E402
from training import meta_label as ml  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import _agg, gate_decision  # noqa: E402

LAB = ROOT / "data" / "meta_label"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "df.pkl"
TRAIN_SUP_PATH = LAB / "train_sup.pkl"
HOLD_SUP_PATH = LAB / "hold_sup.pkl"
FRAME_TRAIN_PATH = LAB / "frame_train.npy"
FRAME_HOLD_PATH = LAB / "frame_hold.npy"
VOTES_TRAIN_PATH = LAB / "votes_train.npy"
VOTES_HOLD_PATH = LAB / "votes_hold.npy"
CTX_PATH = LAB / "context.npz"
CANDIDATES_PATH = LAB / "candidates.json"
DATASET_PATH = LAB / "dataset.npz"
RECORDS_PATH = LAB / "candidate_records.json"
MODEL_PATH = LAB / "meta_model.pkl"
LEDGERS = LAB / "eval_ledgers"
SCRATCH = LAB / "scratch"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"
HOLDOUT = 792

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


def _fetch_df() -> pd.DataFrame:
    if DF_PATH.exists():
        return pd.read_pickle(DF_PATH)
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h", period="730d")
            break
        except Exception as e:
            print(f"fetch attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(DF_PATH)
    return df


def _ctx(sup: pd.DataFrame) -> dict:
    """The context arrays feature_row consumes, aligned to a
    supervised frame: production atr_rel / volatility_21 and the
    raw ATR fraction the production env itself feeds the
    broker."""
    return {
        "atr_rel": sup["atr_rel"].to_numpy(dtype=float),
        "vol21": sup["volatility_21"].to_numpy(dtype=float),
        "atr_raw": sup["atr_pct_raw"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------- phases
def phase_frames() -> None:
    if results().get("frames"):
        print("frames: checkpoint exists, skipping", flush=True)
        return
    meta = nightly._champion_meta()
    assert meta.get("run_id") == 73, f"lab stacker champion is not run 73: {meta.get('run_id')}"
    assert meta.get("feature_set") == "pick2_98", meta.get("feature_set")
    want = list(meta.get("feature_columns") or [])
    assert len(want) == 98, len(want)

    df = _fetch_df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    train_sup_all, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup_all.index.max() < hold_sup.index.min()
    holdout_start = hold_sup.index[0]

    print("building pick2 frame ...", flush=True)
    t0 = time.time()
    frame_all, built_cols, evidence = pick2_features.build_pick2_frame(df)
    assert list(built_cols) == want, "pick2 columns differ from the champion's recorded set"
    frame_on_sup = frame_all.reindex(sup.index)[want]
    vals_all = frame_on_sup.to_numpy(dtype=float)
    print(f"  pick2 frame in {time.time() - t0:.0f}s; evidence={evidence}",
          flush=True)

    # Training region: sup rows before the holdout whose 98
    # values are ALL finite, restricted to the contiguous suffix
    # (the scored region opens after warmup and never closes).
    finite = np.isfinite(vals_all).all(axis=1)
    pre = np.nonzero(finite[:split])[0]
    assert len(pre) > 5_000, f"finite training rows too few: {len(pre)}"
    first = int(pre[0])
    bad = np.nonzero(~finite[first:split])[0]
    assert len(bad) == 0, "non-finite pick2 rows inside the scored region"
    train_sup = sup.iloc[first:split]
    # The raw feed carries two known 6h holes (2025-10-25,
    # 2026-05-08 — yfinance gaps in 17,510 candles). The whole
    # duel machinery (nightly included) is POSITIONAL — iloc
    # slices, positional votes — so a hole misaligns nothing;
    # assert the gap structure is exactly that and record it,
    # rather than demanding a contiguity the feed never had.
    diffs_h = (np.diff(train_sup.index.values)
               / np.timedelta64(1, "h"))
    gap_sizes = sorted({float(g) for g in diffs_h if g != 1.0})
    assert all(g <= 6.0 for g in gap_sizes) and \
        int((diffs_h != 1.0).sum()) <= 5, \
        f"unexpected timestamp gaps in the training region: {gap_sizes}"
    assert train_sup.index.max() < holdout_start

    champ_frame, tag = nightly.champion_holdout_frame(df, hold_sup, meta)
    assert tag == "pick2_98" and list(champ_frame.columns) == want
    frame_hold = champ_frame.to_numpy(dtype=float)
    assert np.isfinite(frame_hold).all(), "non-finite pick2 value in the holdout"
    champ_model = joblib.load(STACKER_CHAMPION)
    votes_hold = champ_model.predict_proba(champ_frame)[:, 1]

    train_sup.to_pickle(TRAIN_SUP_PATH)
    hold_sup.to_pickle(HOLD_SUP_PATH)
    np.save(FRAME_TRAIN_PATH, vals_all[first:split])
    np.save(FRAME_HOLD_PATH, frame_hold)
    np.save(VOTES_HOLD_PATH, votes_hold)
    ctx_tr, ctx_ho = _ctx(train_sup), _ctx(hold_sup)
    np.savez(CTX_PATH, atr_rel_tr=ctx_tr["atr_rel"], vol21_tr=ctx_tr["vol21"],
             atr_raw_tr=ctx_tr["atr_raw"], atr_rel_ho=ctx_ho["atr_rel"],
             vol21_ho=ctx_ho["vol21"], atr_raw_ho=ctx_ho["atr_raw"])
    save_result("frames", {
        "rows_raw": len(df), "rows_sup": len(sup), "split": split,
        "train_rows": len(train_sup), "holdout_rows": len(hold_sup),
        "warmup_dropped_train_rows": int(split - len(train_sup)),
        "train_first": str(train_sup.index[0]),
        "train_last": str(train_sup.index[-1]),
        "holdout_first": str(hold_sup.index[0]),
        "holdout_last": str(hold_sup.index[-1]),
        "frame_evidence": evidence,
        "train_gap_hours": gap_sizes,
        "train_gap_count": int((diffs_h != 1.0).sum()),
        "champion": {"stacker_run": 73, "feature_set": "pick2_98",
                     "ppo_sha256": _sha(PPO_CHAMPION),
                     "stacker_sha256": _sha(STACKER_CHAMPION)},
        "guard_before": _guard(),
    })
    print(f"frames: train {len(train_sup)} rows "
          f"({train_sup.index[0]} -> {train_sup.index[-1]}), holdout "
          f"{len(hold_sup)} rows; champion votes mean "
          f"{votes_hold.mean():.3f}", flush=True)


def phase_votes() -> None:
    if VOTES_TRAIN_PATH.exists():
        print("votes: checkpoint exists, skipping", flush=True)
        return
    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    frame = np.load(FRAME_TRAIN_PATH)
    want = list(nightly._champion_meta()["feature_columns"])
    X = pd.DataFrame(frame, columns=want, index=train_sup.index)
    y = train_sup["target"].astype(int)
    print(f"cross-fitting pick votes over {len(X)} training rows ...",
          flush=True)
    t0 = time.time()
    votes = crossfit_p_up_pick(X, y, folds=4)
    np.save(VOTES_TRAIN_PATH, votes)
    save_result("votes", {"seconds": round(time.time() - t0, 1),
                          "mean": float(votes.mean()),
                          "frac_neutral": float(np.mean(votes == 0.5))})
    print(f"votes: crossfit done in {time.time() - t0:.0f}s, "
          f"mean={votes.mean():.3f}", flush=True)


def phase_candidates() -> None:
    if CANDIDATES_PATH.exists():
        print("candidates: checkpoint exists, skipping", flush=True)
        return
    from stable_baselines3 import PPO

    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    votes = np.load(VOTES_TRAIN_PATH)
    model = PPO.load(PPO_CHAMPION)

    def policy(obs):
        return int(model.predict(obs, deterministic=True)[0])

    SCRATCH.mkdir(parents=True, exist_ok=True)
    all_cands: list = []
    t0 = time.time()
    for bstart in range(0, len(train_sup) - 263, 264):
        block_sup = train_sup.iloc[bstart:bstart + 264]
        block_votes = votes[bstart:bstart + 264]
        db = SCRATCH / f"cand_{bstart}.db"
        db.unlink(missing_ok=True)
        local = ml.extract_candidates(block_sup, block_votes, policy, db)
        all_cands.extend((bstart + i, s) for i, s in local)
        print(f"  block @{bstart}: {len(local)} candidates "
              f"({len(all_cands)} total, {time.time() - t0:.0f}s)",
              flush=True)
    ml.validate_candidates(all_cands, len(train_sup))
    CANDIDATES_PATH.write_text(json.dumps(all_cands))
    save_result("candidates", {
        "n": len(all_cands),
        "longs": int(sum(1 for _p, s in all_cands if s > 0)),
        "shorts": int(sum(1 for _p, s in all_cands if s < 0)),
        "seconds": round(time.time() - t0, 1)})
    print(f"candidates: {len(all_cands)} entry attempts extracted",
          flush=True)


def phase_labels() -> None:
    if DATASET_PATH.exists():
        print("labels: checkpoint exists, skipping", flush=True)
        return
    train_sup = pd.read_pickle(TRAIN_SUP_PATH)
    frame = np.load(FRAME_TRAIN_PATH)
    votes = np.load(VOTES_TRAIN_PATH)
    ctx = np.load(CTX_PATH)
    candidates = [tuple(c) for c in json.loads(CANDIDATES_PATH.read_text())]
    arrays = {c: train_sup[c].to_numpy(dtype=float)
              for c in ("open", "high", "low", "close", "atr_pct_raw")}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    db = SCRATCH / "labels.db"
    db.unlink(missing_ok=True)
    print(f"labeling {len(candidates)} candidates ...", flush=True)
    t0 = time.time()
    X, y, recs = ml.build_candidate_dataset(
        candidates, frame, votes, ctx["atr_rel_tr"], ctx["vol21_tr"],
        ctx["atr_raw_tr"], arrays, db)
    np.savez(DATASET_PATH, X=X, y=y)
    RECORDS_PATH.write_text(json.dumps(recs))
    exits: dict = {}
    for r in recs:
        exits[r["exit_reason"]] = exits.get(r["exit_reason"], 0) + 1
    save_result("labels", {
        "n_candidates": len(candidates), "n_labeled": len(recs),
        "dropped": len(candidates) - len(recs),
        "base_rate": float(y.mean()) if len(y) else None,
        "mean_pnl": float(np.mean([r["pnl"] for r in recs])) if recs else None,
        "mean_fees": float(np.mean([r["fees"] for r in recs])) if recs else None,
        "exit_reasons": exits,
        "seconds": round(time.time() - t0, 1)})
    print(f"labels: {len(recs)} labeled, base rate "
          f"{float(y.mean()):.3f}, exits {exits}", flush=True)


def phase_fit() -> None:
    if MODEL_PATH.exists():
        print("fit: checkpoint exists, skipping", flush=True)
        return
    d = np.load(DATASET_PATH)
    X, y = d["X"], d["y"]
    assert len(np.unique(y)) == 2, \
        f"candidate labels are single-class (base rate {y.mean():.3f}); " \
        "the meta-model cannot be trained — study stops here"
    t0 = time.time()
    model = ml.fit_meta_model(X, y)
    joblib.dump(model, MODEL_PATH)
    oof = ml.oof_diagnostics(X, y, folds=4)
    save_result("fit", {"oof": oof, "seconds": round(time.time() - t0, 1),
                        "n_train": int(len(y))})
    print(f"fit: meta-model trained on {len(y)} candidates; OOF {oof}",
          flush=True)


def _windows():
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    p_ups = [votes_hold[k * 264:(k + 1) * 264] for k in range(3)]
    return hold_sup, windows, p_ups


def _resum_arm(tag: str, windows) -> tuple[float, dict]:
    """The two-way check: re-sum one arm's total P&L from its
    persisted episode ledgers ONLY (no env numbers), marking each
    episode at the close the production env marks it at
    (positional start + 24). Returns (resumed_total, per_window)."""
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
    hold_sup, windows, p_ups = _windows()
    frame_hold = np.load(FRAME_HOLD_PATH)
    votes_hold = np.load(VOTES_HOLD_PATH)
    ctx = np.load(CTX_PATH)
    model = joblib.load(MODEL_PATH)

    print("grading champion pair (production evaluator) ...", flush=True)
    champ_res = nightly.evaluate_windows(PPO_CHAMPION, windows, p_ups)
    assert champ_res is not None, "champion pair failed to run"
    champ = _agg(champ_res)

    print("grading pass-all mirror (equivalence pin) ...", flush=True)
    passall = ml.evaluate_gated_windows(
        PPO_CHAMPION, windows, p_ups, [None] * 3,
        ledger_dir=LEDGERS, tag="passall")
    assert passall is not None, "pass-all mirror failed to run"
    for w in range(3):
        pw, cw = passall["windows"][w], champ_res["windows"][w]
        for key in ("score", "total_pnl", "max_drawdown"):
            assert abs(pw[key] - cw[key]) <= 1e-6, (
                f"EQUIVALENCE FAIL window {w} {key}: mirror {pw[key]} "
                f"vs production {cw[key]} — study aborts, nothing registers")
        assert pw["action_mix"] == cw["action_mix"], (
            f"EQUIVALENCE FAIL window {w} action_mix: {pw['action_mix']} "
            f"vs {cw['action_mix']}")
    print("  pass-all mirror == production champion on all 3 windows",
          flush=True)

    arms: dict = {}
    for t in ml.SENSITIVITY_THRESHOLDS:
        tag = f"g{int(round(t * 100))}"
        gates, fns = [], []
        for w in range(3):
            sl = slice(w * 264, (w + 1) * 264)
            g = ml.MetaGate(model, frame_hold[sl], votes_hold[sl],
                            ctx["atr_rel_ho"][sl], ctx["vol21_ho"][sl],
                            ctx["atr_raw_ho"][sl], threshold=t)
            gates.append(g)
            fns.append(g.allow)
        print(f"grading gated arm @ {t:.2f} ...", flush=True)
        res = ml.evaluate_gated_windows(PPO_CHAMPION, windows, p_ups, fns,
                                        ledger_dir=LEDGERS, tag=tag)
        assert res is not None, f"gated arm {t} failed to run"
        resumed, per_window = _resum_arm(tag, windows)
        diff = abs(resumed - res["total_pnl"])
        assert diff < 0.01, (
            f"TWO-WAY CHECK FAIL @ {t}: ledger re-sum {resumed:.2f} vs "
            f"evaluator {res['total_pnl']:.2f} (diff {diff:.4f})")
        agg = _agg(res)
        wins = sum(1 for gw, cw in zip(res["windows"], champ_res["windows"])
                   if gw["score"] > cw["score"])
        gs = res["gate_stats"]
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
              f"{champ['score']:.2f}, P&L {agg['total_pnl']:+.2f}, wins "
              f"{wins}/3, vetoed {gs['entries_vetoed']}/"
              f"{gs['entries_attempted']}, fees ${gs['fees']:.2f}, "
              f"re-sum diff ${diff:.4f}", flush=True)

    g55 = arms["0.55"]
    formula = g55["agg"]["score"] > champ["score"] and g55["wins"] >= 2
    promotable, floor_fail = gate_decision(g55["agg"], formula)
    cgs = passall["gate_stats"]
    save_result("duel", {
        "champion": champ,
        "champion_gate_stats": {
            "attempted": cgs["entries_attempted"],
            "filled": cgs["entries_filled"], "fills": cgs["fills"],
            "fees": cgs["fees"]},
        "arms": arms, "formula_pass": bool(formula),
        "floor_fail": floor_fail, "promotable": bool(promotable)})
    print(f"duel: formula={formula} floor_fail={floor_fail} "
          f"promotable={promotable}", flush=True)


def phase_register() -> None:
    res = results()
    if res.get("registered"):
        print("register: already done, skipping", flush=True)
        return
    duel = res["duel"]
    champ, arms = duel["champion"], duel["arms"]
    g55 = arms["0.55"]
    promotable = duel["promotable"]
    formula = duel["formula_pass"]
    floor_fail = duel["floor_fail"]
    lab = res["labels"]
    fit = res["fit"]
    oof = fit["oof"]
    cgs = duel["champion_gate_stats"]

    rel = (f"gated pair aggregate {g55['agg']['score']:.0f} vs champion "
           f"{champ['score']:.0f}, won {g55['wins']}/3 windows, P&L "
           f"{g55['agg']['total_pnl']:+,.0f} vs champion "
           f"{champ['total_pnl']:+,.0f} (buy-and-hold "
           f"{g55['agg']['bench_pnl']:+,.0f}); vetoed "
           f"{g55['vetoed']}/{g55['attempted']} candidate entries "
           f"({g55['veto_rate']:.0%}), fees ${g55['fees']:.0f} vs "
           f"champion ${cgs['fees']:.0f}; meta-model on "
           f"{lab['n_labeled']} labeled training candidates (base rate "
           f"{lab['base_rate']:.3f}, OOF balanced acc "
           f"{oof.get('balanced_accuracy')})")
    if promotable:
        verdict = "PASS"
        reason = (f"Meta-Label Gate — Pick A: {rel} (gate pass incl. "
                  f"two-part floor) — WOULD PROMOTE: held as PENDING "
                  f"KASS APPROVAL under the promotion lock; champion "
                  f"kept, nothing installed")
    elif formula and floor_fail is not None:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A: passed the relative gate "
                  f"({rel}) but was not promoted — {floor_fail}; "
                  f"champion kept — fifteenth null")
    else:
        verdict = "FAIL"
        reason = (f"Meta-Label Gate — Pick A: champion kept: {rel} — "
                  f"fifteenth null")

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
            "label": "Pick A",
            "verdict": verdict,
            "design": {
                "threshold": ml.GATE_THRESHOLD,
                "sensitivity_thresholds": list(ml.SENSITIVITY_THRESHOLDS),
                "meta_model": "LGBMClassifier(seed 42)",
                "features": "pick2_98 at decision candle + [p_up, "
                            "|p_up-0.5|, atr_rel, volatility_21, "
                            "atr_pct_raw, side]",
                "label": "standalone production-broker trade, net "
                         "P&L > 0 after fees",
                "signal": "champion pair (PPO run 44 + stacker run 73 "
                          "votes) candidate entries, duel replay",
            },
            "candidates": lab["n_candidates"],
            "labeled": lab["n_labeled"],
            "dropped": lab["dropped"],
            "base_rate": lab["base_rate"],
            "oof": oof,
            "champion_entries": {"attempted": cgs["attempted"],
                                 "filled": cgs["filled"],
                                 "fills": cgs["fills"],
                                 "fees": cgs["fees"]},
            "thresholds": sens,
            "two_way_check": {
                "gated_055_resum_total_pnl": g55["resum_total_pnl"],
                "gated_055_evaluator_total_pnl": g55["agg"]["total_pnl"],
                "abs_diff": g55["resum_diff"]},
        },
    }
    nightly.log_run("ppo", metrics, False)
    import sqlite3
    with sqlite3.connect(nightly.registry_db()) as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    nightly.add_lab_note(
        "Meta-Label Gate — Pick A — PENDING KASS APPROVAL"
        if promotable else "Meta-Label Gate — Pick A", reason)
    guard_after = _guard()
    before = res["frames"]["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue  # the live loops write the paper books on candle
            # closes; those hashes are information, not a guard
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("registered", {"run_id": int(run_id), "verdict": verdict,
                               "reason": reason,
                               "guard_after_ok": True})
    print(f"registered run {run_id}: verdict {verdict} — {reason}",
          flush=True)


PHASES = {"frames": phase_frames, "votes": phase_votes,
          "candidates": phase_candidates, "labels": phase_labels,
          "fit": phase_fit, "duel": phase_duel,
          "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== meta-label study runner done ===", flush=True)


if __name__ == "__main__":
    main()
