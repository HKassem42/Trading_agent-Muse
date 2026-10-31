"""GBM lag-columns experiment (2026-10-22, Kass green-lit).

Follow-up to the GBM focus-list experiment (scripts/build_gbm_focus.py,
refs/gbm-focus-2026-10-22.md). Kass's hypothesis: every model so far
saw a single-row SNAPSHOT. Add TRAJECTORY -- for each focus feature,
its values 1/2/3 hours ago and its changes over 1h/3h -- plus raw
close-return lags, and grade identically so the comparison isolates
the one changed variable. Labels are UNCHANGED (next-candle return,
+-0.3% fee band) -- one variable at a time.

Trading is PARKED: lab-only. Never runs the live loop, never writes
the paper ledger or live state, never promotes (every registration is
promoted=0 with the verdict in the reason, the HA-Seq run 20 / GBM
focus runs 21-24 pattern). Champion artifacts are read (graded),
never written.

Arms (LightGBM, SAME fixed hyperparameters and seed as the focus
build, RAW values):
  E  lag-augmented focus set -- 11 base features x [now, lag1, lag2,
     lag3, delta1, delta3] + 6 close-return lags (ret_1h..ret_24h)
     = 72 columns; balanced class weights
  F  same 72 columns; SMOTE inside each training fold only
     (scaler->SMOTE, the stacker recipe) -- does trajectory rescue
     SMOTE, or does SMOTE amplify lag noise too?
  G  lag-augmented ALL-60: each of the 60 matrix features x [now,
     lag1, delta1] + 6 return lags = 186 columns; class-weighted
     (context arm: trajectory for the kitchen sink)

Validation: chronological TimeSeriesSplit (5 folds) over the pre-duel
rows only (the last-792h duel region is held out from ALL fitting) --
identical protocol to the focus build. Duel: final models fit on ALL
pre-duel rows, graded by training.nightly.evaluate_windows on the
standard 3x264 gate windows through the same threshold policy zip
(models.gbm_policy) with the SAME PRE-REGISTERED thresholds as the
focus build (LONG at P(BUY) >= 0.40, SHORT at P(SHORT) >= 0.40, both
clear -> larger wins, tie -> LONG) -- written in the focus brief
before any grading and carried over unchanged here.

ACTIVITY RULE (pre-registered, carried over): a formula pass
(aggregate beats the champion AND >= 2/3 windows won) with fewer than
3 total round trips is labelled VACUOUS, not a pass; a formula pass
that LOST money in absolute terms is flagged as such in the report.

Causality: every lag/delta/return column is a positive shift (or a
difference of positive shifts) of a source column -- past-only by
construction. phase_setup additionally runs the truncation
self-check (rebuild on the first 9,000 rows, assert exact equality on
shared rows for sampled lag/delta/return columns) and prints
PASS/FAIL; tests/test_gbm_lag.py pins the same property on fixtures.

Usage: .venv/bin/python scripts/build_gbm_lag.py
Phases checkpoint to data/gbm_lag/; reruns skip completed phases.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import gbm_focus as gf  # noqa: E402
from models import gbm_lag as gl  # noqa: E402
from models.gbm_policy import build_rule_model, conviction_np  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "gbm_lag"
FOCUS_LAB = ROOT / "data" / "gbm_focus"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
MATRIX_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"
HA_LAB = ROOT / "data" / "ha_seq"
RULE_ZIP = LAB / "gbm_lag_rule_model"

SEED = 42
HOLDOUT = 792
WINDOW = 264
EPISODES = 10
VACUOUS_TRIP_FLOOR = 3

ARMS = gl.ARMS
ARM_DESIGNS = gl.ARM_DESIGNS


# ---------------------------------------------------------------- state
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    _save_json(PROGRESS_PATH, {"done": sorted(progress() | {phase})})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


def _mtimes() -> dict:
    out = {}
    for p in (MODELS / "ppo_champion.zip", MODELS / "stacker_champion.pkl",
              MODELS / "stacker_champion.json", ROOT / "data" / "paper.db",
              ROOT / "data" / "live_state.json"):
        out[p.name] = p.stat().st_mtime if p.exists() else None
    return out


# ---------------------------------------------------------------- data
def get_matrix() -> pd.DataFrame:
    return gf.load_matrix(MATRIX_PATH)


def get_sup() -> pd.DataFrame:
    """The cached supervised frame the HA-Seq / focus builds graded on
    (same fetch, same windows -> champion numbers cross-checkable)."""
    return pd.read_pickle(HA_LAB / "sup.pkl")


def split_frames(sup: pd.DataFrame):
    split = len(sup) - HOLDOUT
    windows = [sup.iloc[split + k * WINDOW: split + (k + 1) * WINDOW]
               for k in range(3)]
    return windows


def duel_start_ts(sup: pd.DataFrame) -> pd.Timestamp:
    return sup.index[len(sup) - HOLDOUT]


def arm_data(matrix: pd.DataFrame, sup: pd.DataFrame, arm: str):
    """(kept lag frame, pre-duel mask over the KEPT frame's rows)."""
    frame = gl.drop_warmup(gl.arm_frame(matrix, arm))
    mask = np.asarray(frame.index < duel_start_ts(sup))
    return frame, mask


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    matrix, sup = get_matrix(), get_sup()
    windows = split_frames(sup)
    sample_cols = ["Volume__lag3", "Volume__d1",
                   "Force Index__d3", "On-Balance Volume (OBV)__lag1",
                   "Historical Volatility (HV)__lag2",
                   "Choppiness Index (CHOP)__d1", "ret_1h", "ret_24h"]
    trunc = gl.truncation_check(matrix, 9000, sample_cols)
    per_arm = {}
    for arm in ARMS:
        frame, mask = arm_data(matrix, sup, arm)
        per_arm[arm] = {
            "n_features": int(frame.shape[1]),
            "rows_kept": int(len(frame)),
            "rows_dropped_warmup": int(len(matrix) - len(frame)),
            "pre_duel_rows": int(mask.sum()),
            "duel_region_rows": int((~mask).sum()),
        }
    # Every duel DECISION row must have a lag-frame row (arms share the
    # kept timestamps: warmup ends years before the duel region).
    kept = set(per_arm and gl.drop_warmup(gl.focus_lag_frame(matrix)).index)
    missing = [str(ts) for w in windows for ts in w.index[:240]
               if ts not in kept]
    checks = {
        "matrix_monotonic": bool(matrix.index.is_monotonic_increasing),
        "truncation_causality": bool(trunc["pass"]),
        "focus_arm_width_72": per_arm["E"]["n_features"] == 72
        and per_arm["F"]["n_features"] == 72,
        "wide_arm_width_186": per_arm["G"]["n_features"] == 186,
        "duel_decision_rows_covered": not missing,
        "label_counts_unchanged": {
            c: int((matrix[gf.TARGET_COL] == c).sum())
            for c in gf.CLASSES} == {"BUY": 2915, "HOLD": 10976,
                                    "SHORT": 2844},
    }
    save_result("meta", {
        "matrix_rows": int(len(matrix)),
        "matrix_start": str(matrix.index.min()),
        "matrix_end": str(matrix.index.max()),
        "sup_rows": int(len(sup)),
        "duel_start": str(duel_start_ts(sup)),
        "truncation_check": trunc,
        "per_arm": per_arm,
        "lgbm_params": gf.LGBM_PARAMS,
        "checks": checks, "integrity_start": _mtimes(),
    })
    print(f"setup: truncation causality PASS={trunc['pass']} "
          f"(max diff {trunc['max_abs_diff']}), per-arm "
          f"{ {a: (v['n_features'], v['pre_duel_rows']) for a, v in per_arm.items()} }, "
          f"checks={checks}", flush=True)
    assert all(checks.values()), f"self-checks failed: {checks}"


def _wf_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_score, recall_score)
    pred = np.array(gf.CLASSES)[np.argmax(proba, axis=1)]
    dist = {c: float((pred == c).mean()) for c in gf.CLASSES}
    return {
        "n_oof": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "majority_baseline": float(max(
            (y_true == c).mean() for c in gf.CLASSES)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_buy": float(precision_score(
            y_true, pred, labels=["BUY"], average="macro", zero_division=0)),
        "recall_buy": float(recall_score(
            y_true, pred, labels=["BUY"], average="macro", zero_division=0)),
        "precision_short": float(precision_score(
            y_true, pred, labels=["SHORT"], average="macro", zero_division=0)),
        "recall_short": float(recall_score(
            y_true, pred, labels=["SHORT"], average="macro", zero_division=0)),
        "pred_distribution": dist,
        "true_distribution": {c: float((y_true == c).mean())
                              for c in gf.CLASSES},
    }


def phase_wf(arm: str) -> None:
    from sklearn.model_selection import TimeSeriesSplit

    matrix, sup = get_matrix(), get_sup()
    frame, mask = arm_data(matrix, sup, arm)
    treat = gl.ARM_TREATMENT[arm]
    pre = frame.loc[mask]
    X = pre.to_numpy(dtype=float)
    y = matrix.loc[pre.index, gf.TARGET_COL].to_numpy()
    ts = pre.index
    proba = np.full((len(pre), 3), np.nan)
    audits = []
    tss = TimeSeriesSplit(n_splits=5)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, audit = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, treat)
        proba[va_idx] = model.predict_proba(X[va_idx])
        audits.append(audit)
        print(f"wf_{arm} fold {k}: train {len(tr_idx)} -> "
              f"resampled {audit['n_after_resample']}, val {len(va_idx)}",
              flush=True)
    oof = ~np.isnan(proba[:, 0])
    rec = _wf_metrics(y[oof], proba[oof])
    rec["features"] = list(frame.columns)
    rec["n_features"] = int(frame.shape[1])
    rec["fold_audits"] = audits
    save_result(f"wf_{arm}", rec)
    print(f"wf_{arm}: acc {rec['accuracy']:.4f} vs majority "
          f"{rec['majority_baseline']:.4f} | balanced "
          f"{rec['balanced_accuracy']:.4f} | pred dist "
          f"{ {k: round(v, 3) for k, v in rec['pred_distribution'].items()} }",
          flush=True)


def phase_duel_champion() -> None:
    """Reuse the focus build's champion re-grade: identical windows,
    champion zip and stacker votes, already cross-checked to the cent
    against HA-Seq run 20's stored numbers there."""
    focus = _load_json(FOCUS_LAB / "results.json", {})
    champ = focus.get("duel_champion")
    assert champ and champ["champion"]["runnable"], \
        "focus build champion record missing -- run build_gbm_focus first"
    rec = dict(champ)
    rec["source"] = ("reused from data/gbm_focus/results.json "
                     "(identical windows/champion; cross-checked vs "
                     "HA-Seq run 20 in that build)")
    save_result("duel_champion", rec)
    c = champ["champion"]
    print(f"duel_champion: agg={c['score']:,.2f} "
          f"pnl={c['total_pnl']:+,.2f} bench={c['bench_pnl']:+,.2f} "
          f"(from focus build)", flush=True)


def _agg(res) -> dict:
    if res is None:
        return {"mean_reward": -999.0, "total_pnl": 0.0,
                "max_drawdown": 0.0, "score": -1e9, "runnable": False}
    ws = res["windows"]
    return {"score": res["agg_score"], "total_pnl": res["total_pnl"],
            "mean_reward": res["mean_reward"],
            "max_drawdown": float(max(w["max_drawdown"] for w in ws)),
            "bench_pnl": float(sum(w["bench_pnl"] for w in ws)),
            "action_mix": ws[-1]["action_mix"],
            "windows": [{"score": round(w["score"], 2),
                         "pnl": round(w["total_pnl"], 2),
                         "bench": round(w["bench_pnl"], 2)} for w in ws],
            "runnable": True}


def _total_round_trips(res) -> float:
    return float(sum(w["round_trips_per_day"] * EPISODES
                     for w in res["windows"]))


def _rule_zip() -> Path:
    zip_file = LAB / "gbm_lag_rule_model.zip"
    if not zip_file.exists():
        build_rule_model().save(RULE_ZIP)  # SB3 appends .zip
    assert zip_file.exists()
    return zip_file


def phase_duel_arm(arm: str) -> None:
    from training.nightly import evaluate_windows

    matrix, sup = get_matrix(), get_sup()
    windows = split_frames(sup)
    frame, mask = arm_data(matrix, sup, arm)
    treat = gl.ARM_TREATMENT[arm]
    pre = frame.loc[mask]
    y_pre = matrix.loc[pre.index, gf.TARGET_COL].to_numpy()
    cache = LAB / f"probs_{arm}.npz"
    if cache.exists():
        z = np.load(cache)
        prob_ts, proba = z["ts"], z["proba"]
        audit = {"cached": True}
    else:
        model, audit = gf.fit_arm_final(
            pre.to_numpy(dtype=float), y_pre, treat)
        duel_rows = frame.loc[~mask]
        proba = model.predict_proba(duel_rows.to_numpy(dtype=float))
        prob_ts = duel_rows.index.to_numpy()
        np.savez(cache, ts=prob_ts, proba=proba)
    prob_by_ts = {pd.Timestamp(t): p for t, p in zip(prob_ts, proba)}
    p_ups = []
    for w in windows:
        conv = np.zeros(len(w))
        for i, ts in enumerate(w.index):
            p = prob_by_ts.get(ts)
            if p is not None:
                conv[i] = conviction_np(p[0:1], p[2:3])[0]
        p_ups.append((conv + 1.0) / 2.0)
    res = evaluate_windows(_rule_zip(), windows, p_ups)
    assert res is not None, f"arm {arm} policy failed to grade"
    champ = results()["duel_champion"]["champion"]
    bench = float(sum(w["bench_pnl"] for w in res["windows"]))
    assert abs(bench - champ["bench_pnl"]) < 1e-6, "bench mismatch"
    wins = sum(1 for cw, aw in zip(champ["windows"], res["windows"])
               if aw["score"] > cw["score"])
    trips = _total_round_trips(res)
    formula_pass = bool(res["agg_score"] > champ["score"] and wins >= 2)
    vacuous = bool(formula_pass and trips < VACUOUS_TRIP_FLOOR)
    verdict = ("PASS" if formula_pass and not vacuous else
               "VACUOUS PASS (formula only)" if vacuous else "FAIL")
    rec = {"arm": _agg(res), "windows_won": wins,
           "total_round_trips": trips,
           "traded": bool(trips > 0),
           "formula_pass": formula_pass, "vacuous": vacuous,
           "verdict": verdict, "fit_audit": audit,
           "action_mix_by_window": [w["action_mix"]
                                    for w in res["windows"]]}
    save_result(f"duel_{arm}", rec)
    print(f"duel_{arm}: agg={res['agg_score']:,.2f} "
          f"pnl={res['total_pnl']:+,.2f} trips={trips:.0f} | champ "
          f"agg={champ['score']:,.2f} | won {wins}/3 | {verdict}",
          flush=True)


def phase_importance() -> None:
    """OOF permutation importance for the best lag arm (manual loop,
    same scheme as the focus build) + grouping by feature kind
    (now / lag / delta / ret) and by base feature, so the report can
    say whether TRAJECTORY columns carried anything."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    wf = {a: results()[f"wf_{a}"] for a in ARMS}
    best = max(wf, key=lambda a: wf[a]["balanced_accuracy"])
    matrix, sup = get_matrix(), get_sup()
    frame, mask = arm_data(matrix, sup, best)
    treat = gl.ARM_TREATMENT[best]
    feats = list(frame.columns)
    pre = frame.loc[mask]
    X = pre.to_numpy(dtype=float)
    y = matrix.loc[pre.index, gf.TARGET_COL].to_numpy()
    ts = pre.index
    n_repeats = 3 if best == "G" else 5
    imps = np.zeros((5, len(feats)))
    tss = TimeSeriesSplit(n_splits=5)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, _ = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, treat)
        Xv, yv = X[va_idx], y[va_idx]
        base = balanced_accuracy_score(yv, model.predict(Xv))
        for j in range(len(feats)):
            drops = []
            for rep in range(n_repeats):
                rng = np.random.default_rng(SEED + 1000 * rep + j)
                Xp = Xv.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - balanced_accuracy_score(
                    yv, model.predict(Xp)))
            imps[k, j] = float(np.mean(drops))
        print(f"importance fold {k} done (arm {best}, base "
              f"{base:.4f})", flush=True)
    mean = imps.mean(axis=0)
    std = imps.std(axis=0)
    order = np.argsort(-mean)
    total_pos = float(mean[mean > 0].sum()) or 1.0
    table = []
    for rank, j in enumerate(order, start=1):
        table.append({
            "feature": feats[j], "measured_rank": rank,
            "kind": gl.feature_kind(feats[j]),
            "importance_mean": float(mean[j]),
            "importance_std": float(std[j]),
            "importance_share_pct": float(100 * max(mean[j], 0) / total_pos),
        })
    # Group shares: what fraction of positive importance is trajectory?
    kind_tot = {}
    for row in table:
        kind_tot[row["kind"]] = kind_tot.get(row["kind"], 0.0) + \
            max(row["importance_mean"], 0.0)
    kind_share = {k: float(100 * v / total_pos)
                  for k, v in kind_tot.items()}
    traj = [r for r in table if r["kind"] in ("lag", "delta", "ret")]
    best_traj = traj[0] if traj else None
    volume_rows = [r for r in table if r["feature"] == "Volume"
                   or r["feature"].startswith("Volume__")]
    save_result("importance", {
        "best_lag_arm": best, "table": table,
        "kind_share_pct": kind_share,
        "best_trajectory_feature": best_traj,
        "volume_family": volume_rows,
        "scoring": "balanced_accuracy", "n_repeats": n_repeats})
    print(f"importance: best lag arm {best}; kind shares "
          f"{ {k: round(v, 1) for k, v in kind_share.items()} }; "
          f"best trajectory = "
          f"{best_traj['feature'] if best_traj else None}", flush=True)


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run

    res = results()
    champ = res["duel_champion"]["champion"]
    run_ids = {}
    summaries = []
    for arm in ARMS:
        wf, duel = res[f"wf_{arm}"], res[f"duel_{arm}"]
        a = duel["arm"]
        reason = (
            f"GBM lag-columns challenger (family=gbm_lag, arm {arm}: "
            f"{ARM_DESIGNS[arm]}): duel {duel['verdict']} -- won "
            f"{duel['windows_won']}/3 windows, aggregate "
            f"{a['score']:.0f} vs champion {champ['score']:.0f}, "
            f"P&L ${a['total_pnl']:+,.0f}, {duel['total_round_trips']:.0f} "
            f"round trips (buy-and-hold same windows: "
            f"${champ['bench_pnl']:+,.0f}); walk-forward acc "
            f"{wf['accuracy']:.4f} vs majority baseline "
            f"{wf['majority_baseline']:.4f}, balanced acc "
            f"{wf['balanced_accuracy']:.4f}; lab only, promoted=0")
        metrics = {
            "challenger": a, "champion": champ, "reason": reason,
            "family": "gbm_lag", "deep": False, "timesteps": 0,
            "reward_curve": None,
            "gbm_lag": {
                "arm": arm, "design": ARM_DESIGNS[arm],
                "wf_accuracy": wf["accuracy"],
                "wf_majority_baseline": wf["majority_baseline"],
                "wf_balanced_accuracy": wf["balanced_accuracy"],
                "verdict": duel["verdict"],
                "windows_won": duel["windows_won"],
                "total_round_trips": duel["total_round_trips"],
                "thresholds": {"long_p": 0.40, "short_p": 0.40},
            },
        }
        log_run("ppo", metrics, False)  # lab challenger: never promoted
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        run_ids[arm] = int(run_id)
        summaries.append(f"Arm {arm} ({ARM_DESIGNS[arm]}): "
                         f"{duel['verdict']}, agg {a['score']:.0f} vs "
                         f"champ {champ['score']:.0f}, P&L "
                         f"${a['total_pnl']:+,.0f}, "
                         f"{duel['total_round_trips']:.0f} round trips, "
                         f"WF acc {wf['accuracy']:.4f} vs "
                         f"{wf['majority_baseline']:.4f} baseline "
                         f"(run {run_id}).")
        print(f"register: arm {arm} -> run {run_id} (promoted=0)",
              flush=True)
    imp = res.get("importance", {})
    ks = imp.get("kind_share_pct", {})
    note = ("GBM lag-columns experiment (Kass's trajectory hypothesis; "
            "labels and grading identical to the GBM focus build): "
            "LightGBM 3-class on lag-augmented matrix features, "
            "thresholds pre-registered at 0.40/0.40, graded on the "
            "standard 3 windows vs champion and buy-and-hold "
            f"(${champ['bench_pnl']:+,.0f}). " + " ".join(summaries) +
            f" Best lag arm by OOF balanced accuracy: "
            f"{imp.get('best_lag_arm')}; importance kind shares: "
            f"{ {k: round(v, 1) for k, v in ks.items()} }. "
            "Full report: refs/gbm-lag-2026-10-22.md")
    add_lab_note("GBM lag-columns challenger (family=gbm_lag)", note)
    save_result("register", {"run_ids": run_ids, "lab_note": note})
    print("register: lab note added", flush=True)


def phase_verify() -> None:
    start = results()["meta"]["integrity_start"]
    now = _mtimes()
    same = all(start[k] == now[k] for k in start)
    save_result("verify", {"mtimes_unchanged": bool(same),
                           "start": start, "end": now})
    print(f"verify: champion/paper mtimes unchanged = {same}", flush=True)
    assert same, f"INTEGRITY: mtimes changed {start} -> {now}"


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    phases = [("setup", phase_setup)]
    phases += [(f"wf_{a}", (lambda a: lambda: phase_wf(a))(a))
               for a in ARMS]
    phases += [("duel_champion", phase_duel_champion)]
    phases += [(f"duel_{a}", (lambda a: lambda: phase_duel_arm(a))(a))
               for a in ARMS]
    phases += [("importance", phase_importance),
               ("register", phase_register),
               ("verify", phase_verify)]
    done = progress()
    print(f"=== GBM lag build (phases done: {sorted(done) or 'none'}) "
          f"===", flush=True)
    for name, fn in phases:
        if name in done:
            print(f"{name}: already done -- skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"--- phase {name} done in {time.time() - t0:.0f}s ---",
              flush=True)
    print("=== GBM lag build complete ===", flush=True)


if __name__ == "__main__":
    main()
