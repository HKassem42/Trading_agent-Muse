"""GBM focus-list challenger experiment (2026-10-22, Kass green-lit).

Trading is PARKED: lab-only. Never runs the live loop, never writes the
paper ledger or live state, never promotes (every registration is
promoted=0 with the verdict in the reason, the HA-Seq run 20 pattern).
Champion artifacts are read (graded), never written.

Kass pasted an external top-10 feature ranking (item 1's name was cut
off; its description is a raw-volume move-confirmation feature, covered
here by Volume + On-Balance Volume). This experiment trains a LightGBM
multiclass classifier (BUY/HOLD/SHORT) on the real BTC matrix
(data/timeseries_matrix/btc_usd_1h_matrix.csv) in four arms:

  A  focus-11, plain
  B  focus-11, balanced class weights
  C  focus-11, SMOTE inside each training fold (stacker recipe:
     scaler fit on the fold's train rows -> SMOTE -> model) --
     Kass's SMOTE-for-aggression question, answered empirically
  D  all 60 matrix feature columns, class-weighted (context arm)

Validation: chronological TimeSeriesSplit (5 folds) over the pre-duel
rows only (the last-792h duel region is held out from ALL fitting).
Signal metrics always print next to the majority-class baseline.

Duel: each arm's final model (fit on ALL pre-duel rows) is graded by
training.nightly.evaluate_windows on the standard 3x264 gate windows
(the cached supervised frame from the HA-Seq build, so the champion
re-grade is directly comparable to run 20's), through a threshold
policy zip (models.gbm_policy) with PRE-REGISTERED thresholds --
written here before any grading and never changed afterward:

  LONG  when P(BUY)   >= 0.40
  SHORT when P(SHORT) >= 0.40   (both clear -> larger probability wins;
                                 exact tie -> LONG)
  otherwise FLAT: an open position whose signal flipped or vanished is
  exited by the wrapper's position-aware mapping (SELL a long, BUY to
  cover a short, HOLD when flat), mirroring the HA wrapper.

The probability pair rides the env's vote channel as a signed
conviction c in {-P(SHORT), 0, +P(BUY)} encoded p_up = (c+1)/2; see
models/gbm_policy.py for why the encoding is lossless for this rule.

ACTIVITY RULE (pre-registered): a formula pass (aggregate beats the
champion AND >= 2/3 windows won) with fewer than 3 total round trips
across the duel is labelled VACUOUS, not a pass -- the gate's missing
activity floor is a known design flaw (decision pending with Kass) and
this experiment does not exploit it.

Usage: .venv/bin/python scripts/build_gbm_focus.py
Phases checkpoint to data/gbm_focus/; reruns skip completed phases.
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

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models import gbm_focus as gf  # noqa: E402
from models.gbm_policy import build_rule_model, conviction_np  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "gbm_focus"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
MATRIX_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"
HA_LAB = ROOT / "data" / "ha_seq"
RULE_ZIP = LAB / "gbm_rule_model"

SEED = 42
HOLDOUT = 792
WINDOW = 264
EPISODES = 10          # evaluate_ppo default: 10 episode-days per window
VACUOUS_TRIP_FLOOR = 3  # pre-registered activity floor (total round trips)

ARMS = {
    "A": "focus-11, plain",
    "B": "focus-11, balanced class weights",
    "C": "focus-11, SMOTE inside each training fold (scaler->SMOTE)",
    "D": "all-60 features, balanced class weights",
}


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
    """The cached supervised frame the HA-Seq build graded on (same
    fetch, same windows -> champion numbers cross-checkable)."""
    return pd.read_pickle(HA_LAB / "sup.pkl")


def split_frames(sup: pd.DataFrame):
    split = len(sup) - HOLDOUT
    windows = [sup.iloc[split + k * WINDOW: split + (k + 1) * WINDOW]
               for k in range(3)]
    return windows


def duel_start_ts(sup: pd.DataFrame) -> pd.Timestamp:
    return sup.index[len(sup) - HOLDOUT]


def pre_duel_mask(matrix: pd.DataFrame, sup: pd.DataFrame) -> np.ndarray:
    return np.asarray(matrix.index < duel_start_ts(sup))


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    matrix, sup = get_matrix(), get_sup()
    windows = split_frames(sup)
    mask = pre_duel_mask(matrix, sup)
    y = matrix[gf.TARGET_COL]
    counts = {c: int((y == c).sum()) for c in gf.CLASSES}
    corr = float(matrix["Force Index"].corr(matrix["Elder Force Index (EFI)"]))
    # Every duel DECISION row (first 240 of each 264-row window) must
    # have a matrix row to take a probability from.
    midx = set(matrix.index)
    missing = [str(ts) for w in windows for ts in w.index[:240]
               if ts not in midx]
    checks = {
        "matrix_monotonic": bool(matrix.index.is_monotonic_increasing),
        "focus_columns_present": all(
            c in matrix.columns for c in gf.FOCUS_FEATURES),
        "pre_duel_rows_positive": bool(mask.sum() > 10_000),
        "duel_decision_rows_covered": not missing,
        "label_counts_match_report": counts == {
            "BUY": 2915, "HOLD": 10976, "SHORT": 2844},
    }
    save_result("meta", {
        "matrix_rows": int(len(matrix)),
        "matrix_start": str(matrix.index.min()),
        "matrix_end": str(matrix.index.max()),
        "sup_rows": int(len(sup)),
        "duel_start": str(duel_start_ts(sup)),
        "duel_end": str(sup.index.max()),
        "pre_duel_rows": int(mask.sum()),
        "duel_region_matrix_rows": int((~mask).sum()),
        "label_counts": counts,
        "majority_baseline_full": counts["HOLD"] / len(matrix),
        "force_efi_corr": corr,
        "lgbm_params": gf.LGBM_PARAMS,
        "checks": checks, "integrity_start": _mtimes(),
    })
    print(f"setup: matrix {len(matrix)} rows, pre-duel {int(mask.sum())}, "
          f"duel from {duel_start_ts(sup)}, Force/EFI corr {corr:.3f}, "
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
    mask = pre_duel_mask(matrix, sup)
    feats = gf.arm_features(matrix, arm)
    pre = matrix.loc[mask]
    X = pre[feats].to_numpy(dtype=float)
    y = pre[gf.TARGET_COL].to_numpy()
    ts = pre.index
    proba = np.full((len(pre), 3), np.nan)
    audits = []
    tss = TimeSeriesSplit(n_splits=5)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, audit = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, arm)
        proba[va_idx] = model.predict_proba(X[va_idx])
        audits.append(audit)
        print(f"wf_{arm} fold {k}: train {len(tr_idx)} -> "
              f"resampled {audit['n_after_resample']}, val {len(va_idx)}",
              flush=True)
    oof = ~np.isnan(proba[:, 0])
    rec = _wf_metrics(y[oof], proba[oof])
    rec["features"] = feats
    rec["n_features"] = len(feats)
    rec["fold_audits"] = audits
    save_result(f"wf_{arm}", rec)
    print(f"wf_{arm}: acc {rec['accuracy']:.4f} vs majority "
          f"{rec['majority_baseline']:.4f} | balanced "
          f"{rec['balanced_accuracy']:.4f} | pred dist "
          f"{ {k: round(v, 3) for k, v in rec['pred_distribution'].items()} }",
          flush=True)


def _agg(res) -> dict:
    """Flatten a 3-window evaluate_windows result into the nightly
    run-record shape (mirrors build_ha_seq._agg)."""
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
    zip_file = LAB / "gbm_rule_model.zip"
    if not zip_file.exists():
        build_rule_model().save(RULE_ZIP)  # SB3 appends .zip
    assert zip_file.exists()
    return zip_file


def phase_duel_champion() -> None:
    import joblib
    from training.nightly import evaluate_windows

    sup = get_sup()
    windows = split_frames(sup)
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    champ_p = [stacker_model.predict_proba(w[FEATURE_COLUMNS])[:, 1]
               for w in windows]
    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows, champ_p)
    assert champ_res is not None, "champion failed to grade"
    ha = _load_json(HA_LAB / "results.json", {}).get("duel", {})
    rec = {"champion": _agg(champ_res),
           "total_round_trips": _total_round_trips(champ_res),
           "ha_run20_crosscheck": {
               "ha_champion_score": ha.get("champion", {}).get("score"),
               "my_champion_score": champ_res["agg_score"],
               "ha_bench": ha.get("bench_pnl_total"),
               "my_bench": float(sum(
                   w["bench_pnl"] for w in champ_res["windows"])),
           }}
    save_result("duel_champion", rec)
    print(f"duel_champion: agg={champ_res['agg_score']:,.2f} "
          f"pnl={champ_res['total_pnl']:+,.2f} bench="
          f"{rec['champion']['bench_pnl']:+,.2f} (HA run 20 stored: "
          f"score {rec['ha_run20_crosscheck']['ha_champion_score']}, "
          f"bench {rec['ha_run20_crosscheck']['ha_bench']})", flush=True)


def phase_duel_arm(arm: str) -> None:
    from training.nightly import evaluate_windows

    matrix, sup = get_matrix(), get_sup()
    windows = split_frames(sup)
    mask = pre_duel_mask(matrix, sup)
    feats = gf.arm_features(matrix, arm)
    pre = matrix.loc[mask]
    cache = LAB / f"probs_{arm}.npz"
    if cache.exists():
        z = np.load(cache)
        prob_ts, proba = z["ts"], z["proba"]
        audit = {"cached": True}
    else:
        model, audit = gf.fit_arm_final(
            pre[feats].to_numpy(dtype=float),
            pre[gf.TARGET_COL].to_numpy(), arm)
        duel_rows = matrix.loc[~mask]
        proba = model.predict_proba(duel_rows[feats].to_numpy(dtype=float))
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
    """OOF permutation importance for the best focus arm, computed
    manually (shuffle one validation column at a time, re-score with
    balanced accuracy): the arm predictor is a thin wrapper, not an
    sklearn estimator, so sklearn.inspection's estimator checks are
    the wrong tool -- the manual loop is exact and deterministic."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    wf = {a: results()[f"wf_{a}"] for a in ("A", "B", "C")}
    best = max(wf, key=lambda a: wf[a]["balanced_accuracy"])
    matrix, sup = get_matrix(), get_sup()
    mask = pre_duel_mask(matrix, sup)
    feats = gf.arm_features(matrix, best)
    pre = matrix.loc[mask]
    X = pre[feats].to_numpy(dtype=float)
    y = pre[gf.TARGET_COL].to_numpy()
    ts = pre.index
    n_repeats = 5
    imps = np.zeros((5, len(feats)))
    tss = TimeSeriesSplit(n_splits=5)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, _ = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, best)
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
        f = feats[j]
        claimed = gf.KASS_CLAIMED.get(f, {})
        table.append({
            "feature": f, "measured_rank": rank,
            "importance_mean": float(mean[j]),
            "importance_std": float(std[j]),
            "importance_share_pct": float(100 * max(mean[j], 0) / total_pos),
            "kass_rank": claimed.get("rank"),
            "kass_claimed_pct": claimed.get("pct"),
        })
    save_result("importance", {"best_focus_arm": best, "table": table,
                               "scoring": "balanced_accuracy",
                               "n_repeats": 5})
    print(f"importance: best focus arm {best}; top3 = "
          f"{[t['feature'] for t in table[:3]]}", flush=True)


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run

    res = results()
    champ = res["duel_champion"]["champion"]
    run_ids = {}
    summaries = []
    for arm in gf.ARMS:
        wf, duel = res[f"wf_{arm}"], res[f"duel_{arm}"]
        a = duel["arm"]
        reason = (
            f"GBM focus challenger (family=gbm_focus, arm {arm}: "
            f"{ARMS[arm]}): duel {duel['verdict']} -- won "
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
            "family": "gbm_focus", "deep": False, "timesteps": 0,
            "reward_curve": None,
            "gbm_focus": {
                "arm": arm, "design": ARMS[arm],
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
        summaries.append(f"Arm {arm} ({ARMS[arm]}): {duel['verdict']}, "
                         f"agg {a['score']:.0f} vs champ "
                         f"{champ['score']:.0f}, P&L "
                         f"${a['total_pnl']:+,.0f}, "
                         f"{duel['total_round_trips']:.0f} round trips, "
                         f"WF acc {wf['accuracy']:.4f} vs "
                         f"{wf['majority_baseline']:.4f} baseline "
                         f"(run {run_id}).")
        print(f"register: arm {arm} -> run {run_id} (promoted=0)",
              flush=True)
    imp = res.get("importance", {})
    note = ("GBM focus-list experiment (Kass's pasted top-10; item 1 = "
            "Volume + OBV): LightGBM 3-class on the real BTC matrix, "
            "thresholds pre-registered at 0.40/0.40, graded on the "
            "standard 3 windows vs champion and buy-and-hold "
            f"(${champ['bench_pnl']:+,.0f}). " + " ".join(summaries) +
            f" Best focus arm by OOF balanced accuracy: "
            f"{imp.get('best_focus_arm')}; measured #1 feature: "
            f"{imp.get('table', [{}])[0].get('feature')} "
            f"(Kass's list had Force Index as its top named item at "
            f"2.9%). Full report: refs/gbm-focus-2026-10-22.md")
    add_lab_note("GBM focus-list challenger (family=gbm_focus)", note)
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
               for a in gf.ARMS]
    phases += [("duel_champion", phase_duel_champion)]
    phases += [(f"duel_{a}", (lambda a: lambda: phase_duel_arm(a))(a))
               for a in gf.ARMS]
    phases += [("importance", phase_importance),
               ("register", phase_register),
               ("verify", phase_verify)]
    done = progress()
    print(f"=== GBM focus build (phases done: {sorted(done) or 'none'}) "
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
    print("=== GBM focus build complete ===", flush=True)


if __name__ == "__main__":
    main()
