"""Muse's Pick #4 / #5 / #6 -- LAB study (2026-11-04, Kass's
order: "Name the 3 models you did, Muse's Pick #4 #5 #6... build
a model that will beat my champions... train the models however
you want").

Pre-registered in refs/muse-picks-456-preregistration-2026-11-04.md
BEFORE any candidate was fitted. Three stacker challengers on
Pick #2's 98 scored eyes and the Next Try study's exam (train =
12,045 rows finite on the full frame; holdout = last 792 candles
from 2026-10-02 06:00 UTC; champions re-scored on identical
timestamps: run 96 = 0.5265, run 73 = 0.5227; all-time clean-era
bar = 0.5354):

- #4 "Three Clocks" (family muse_pick4): Pick-recipe pipelines
  fitted to 1h/4h/24h direction labels; equal-weight vote.
- #5 "Fresh Eyes" (family muse_pick5): the champion four trees x
  5 seeds, recency sample weights (half-life 120d), no SMOTE,
  plain probability vote.
- #6 "Wide Net, Clean Labels" (family muse_pick6): 7-learner
  stack (champion four + kNN + MLP + LR) trained only on rows
  whose |next-candle return| clears the train bottom quartile.

Threshold fixed 0.50 everywhere. A pick is PENDING KASS only if
it beats both re-scored champions with clean anatomy (up-calls
>= 8, precision > 0.05); beating 0.5354 is flagged BEAT THE BAR.
LAB ONLY under the promotion lock; guards hash-checked; each
pick registers as soon as it grades so a later failure cannot
lose an earlier result.

Usage: .venv/bin/python scripts/build_muse_picks456.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import next_try_features, pick2_features, scored_features  # noqa: E402
from models.muse_pick import build_pick_pipeline  # noqa: E402
from models.muse_picks456 import (  # noqa: E402
    BAG_SEEDS, HALF_LIFE_DAYS, HORIZONS, build_bag_learners,
    build_wide_stack, clean_keep_mask, horizon_labels,
    quartile_threshold, recency_weights)
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB1 = ROOT / "data" / "muse_next_try"
LAB = ROOT / "data" / "muse_picks456"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
COLS98 = list(scored_features.scored_columns()) + list(
    pick2_features.NEW_COLUMNS)
CHAMP96_ACC, CHAMP73_ACC, BAR = 0.5265151515151515, 0.5227272727272727, 0.5354
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
    "shadow_stacker": MODELS / "shadow_seat" / "stacker.pkl",
    "shadow_ppo": MODELS / "shadow_seat" / "ppo.zip",
    "gated_stacker": MODELS / "gated_seat" / "stacker.pkl",
}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard():
    return {k: _sha(p) for k, p in GUARD_FILES.items()}


def _progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"done": []}


def _mark(pick: str) -> None:
    pr = _progress()
    if pick not in pr["done"]:
        pr["done"].append(pick)
    PROGRESS.write_text(json.dumps(pr, indent=1))


def _setup():
    frame = pd.read_pickle(LAB1 / "frame120.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    split = len(prod_sup) - HOLDOUT
    train_idx_all = prod_sup.index[:split]
    Xall = frame.reindex(train_idx_all)
    common = Xall.dropna(subset=COLS98 + next_try_features.NEW_COLUMNS)
    train_idx = common.index
    ytr = prod_sup.reindex(train_idx)["target"].astype(int)
    hold_idx = prod_sup.index[split:]
    yhold = prod_sup.iloc[split:]["target"].astype(int)
    X = frame.reindex(train_idx)[COLS98]
    Xhold = frame.reindex(hold_idx)[COLS98]
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert np.isfinite(Xhold.to_numpy(dtype=float)).all()
    majority = float(max(yhold.mean(), 1 - yhold.mean()))
    assert abs(majority - 0.5101) < 5e-5, f"exam anchor moved: {majority}"
    close = prod_sup["close"]
    # integrity: the stored target IS the 1h horizon label
    y1_all = horizon_labels(close, 1).reindex(prod_sup.index)
    both = y1_all.notna()
    assert (y1_all[both].astype(int).to_numpy()
            == prod_sup["target"][both].to_numpy()).all()
    print(f"setup: train {len(train_idx)} rows, holdout {len(hold_idx)} "
          f"from {hold_idx[0]}; majority {majority:.4f}; "
          f"target==horizon_labels(1) verified", flush=True)
    return X, ytr, Xhold, yhold, close, train_idx


def _metrics(proba, yhold):
    from sklearn.metrics import accuracy_score, precision_score
    pred = (np.asarray(proba) >= 0.5).astype(int)
    acc = float(accuracy_score(yhold, pred))
    prec = float(precision_score(yhold, pred, zero_division=0))
    win_accs = [float(accuracy_score(
        yhold.iloc[k * 264:(k + 1) * 264], pred[k * 264:(k + 1) * 264]))
        for k in range(3)]
    return {"acc": acc, "precision": prec, "up_calls": int(pred.sum()),
            "window_accs": win_accs}


def _verdict(m: dict) -> tuple[bool, bool, bool]:
    clean = m["up_calls"] >= 8 and m["precision"] > 0.05
    pending = bool(m["acc"] > CHAMP96_ACC and m["acc"] > CHAMP73_ACC
                   and clean)
    beat_bar = bool(m["acc"] > BAR and clean)
    return pending, beat_bar, clean


def _register(family, label, design, m, wf_acc, wf_prec, extra,
              guard_before):
    guard_after = _guard()
    assert guard_after == guard_before, "GUARD TRIPPED"
    pending, beat_bar, clean = _verdict(m)
    acc = m["acc"]
    if beat_bar:
        reason = (f"{label} holdout acc {acc:.4f} BEAT THE BAR "
                  f"(all-time clean-era best 0.5354) and both "
                  f"champions re-scored on the shared holdout "
                  f"(run 96: {CHAMP96_ACC:.4f}; run 73 / Muse's "
                  f"Pick #2: {CHAMP73_ACC:.4f}) — WOULD PROMOTE: "
                  f"held as PENDING KASS APPROVAL under the "
                  f"promotion lock; champion kept")
    elif pending:
        reason = (f"{label} holdout acc {acc:.4f} beat both "
                  f"champions re-scored on the shared holdout "
                  f"(run 96: {CHAMP96_ACC:.4f}; run 73 / Muse's "
                  f"Pick #2: {CHAMP73_ACC:.4f}) but not the all-time "
                  f"bar 0.5354 — WOULD PROMOTE: held as PENDING "
                  f"KASS APPROVAL under the promotion lock; "
                  f"champion kept")
    elif acc > CHAMP96_ACC and not clean:
        reason = (f"{label} holdout acc {acc:.4f} beat both "
                  f"champions BUT anatomy is DEGENERATE (up-calls "
                  f"{m['up_calls']}/792, precision "
                  f"{m['precision']:.4f}) — counted as a LOSS under "
                  f"the pre-registered run-63 rule")
        pending = False
    else:
        reason = (f"{label} holdout acc {acc:.4f} did not beat "
                  f"both champions re-scored on the shared holdout "
                  f"(run 96 {CHAMP96_ACC:.4f}, run 73 "
                  f"{CHAMP73_ACC:.4f}); champion kept (study "
                  f"verdict only)")
    metrics = {
        "challenger_acc": acc, "champion_acc": CHAMP96_ACC,
        "challenger_wf_acc": wf_acc, "challenger_wf_precision": wf_prec,
        "precision": m["precision"], "folds": 3,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": family,
        "votes": vote_histogram(extra.pop("proba")),
        family: {
            "label": label, "verdict": "PASS" if pending else "FAIL",
            "design": design, "beat_bar": beat_bar,
            "champion73_acc": CHAMP73_ACC, "bar": BAR,
            "window_accs": m["window_accs"], "up_calls": m["up_calls"],
            "majority_baseline": 0.5101, **extra,
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        f"{label} (family={family})",
        f"{design}. Holdout acc {acc:.4f} vs run 96 "
        f"{CHAMP96_ACC:.4f} and run 73 (Muse's Pick #2) "
        f"{CHAMP73_ACC:.4f} re-scored on identical timestamps; "
        f"all-time bar 0.5354; precision {m['precision']:.4f}; "
        f"up-calls {m['up_calls']}/792; walk-forward(3) "
        f"{wf_acc:.4f}; majority baseline 0.5101. "
        f"{'BEAT THE BAR — PENDING KASS APPROVAL.' if beat_bar else 'PENDING KASS APPROVAL under the promotion lock.' if pending else 'No win.'} "
        f"Nothing installed.")
    return pending, beat_bar


def pick4(X, ytr, Xhold, yhold, close, guard_before):
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit
    t0 = time.time()
    labels = {h: horizon_labels(close, h).reindex(X.index)
              for h in HORIZONS}
    # ---- WF3: the 3-head ensemble, fold by fold ----
    tss = TimeSeriesSplit(n_splits=3)
    Xv, yv = X.to_numpy(dtype=float), ytr.to_numpy()
    fold_accs, fold_precs = [], []
    for tr, te in tss.split(Xv):
        probs = []
        for h in HORIZONS:
            lab = labels[h].iloc[tr]
            ok = lab.notna().to_numpy()
            pipe = build_pick_pipeline(random_state=42)
            pipe.fit(Xv[tr][ok], lab.to_numpy()[ok].astype(int))
            probs.append(pipe.predict_proba(Xv[te])[:, 1])
        pred = (np.mean(probs, axis=0) >= 0.5).astype(int)
        fold_accs.append(accuracy_score(yv[te], pred))
        fold_precs.append(precision_score(yv[te], pred, zero_division=0))
        print(f"  pick4 WF fold: acc {fold_accs[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    wf_acc = float(np.mean(fold_accs))
    wf_prec = float(np.mean(fold_precs))
    # ---- full fits + the one holdout grade ----
    heads, probs = {}, []
    for h in HORIZONS:
        lab = labels[h]
        ok = lab.notna().to_numpy()
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(X.to_numpy(dtype=float)[ok],
                 lab.to_numpy()[ok].astype(int))
        heads[f"h{h}"] = pipe
        probs.append(pipe.predict_proba(Xhold.to_numpy(dtype=float))[:, 1])
        print(f"  pick4 head h{h} fitted on {int(ok.sum())} rows "
              f"({time.time() - t0:.0f}s)", flush=True)
    proba = np.mean(probs, axis=0)
    m = _metrics(proba, yhold)
    joblib.dump(heads, LAB / "pick4_heads.pkl", compress=3)
    np.save(LAB / "proba_pick4.npy", proba)
    (LAB / "results_pick4.json").write_text(json.dumps(
        {"metrics": m, "wf3_acc": wf_acc, "wf3_precision": wf_prec,
         "wf3_folds": [float(a) for a in fold_accs]}, indent=1))
    pending, beat_bar = _register(
        "muse_pick4", "Muse's Pick #4",
        "Muse's Pick #4 stacker 'Three Clocks': Pick recipe "
        "(RF+XGB+LGBM+ET -> LR, SMOTE) fitted at 1h/4h/24h label "
        "horizons on Pick #2's 98 scored eyes; equal-weight "
        "probability vote; threshold 0.50",
        m, wf_acc, wf_prec, {"proba": proba,
                             "wf3_folds": [float(a) for a in fold_accs]},
        guard_before)
    print(f"final: PICK #4 acc={m['acc']:.4f} prec={m['precision']:.4f} "
          f"up={m['up_calls']}/792 wf3={wf_acc:.4f} -> "
          f"{'BEAT THE BAR / PENDING' if beat_bar else 'PENDING KASS' if pending else 'NO WIN'}",
          flush=True)


def pick5(X, ytr, Xhold, yhold, guard_before):
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()

    def bag_proba(Xtr_df, y, Xte):
        scaler = StandardScaler().fit(Xtr_df)
        w = recency_weights(Xtr_df.index, HALF_LIFE_DAYS)
        Xs, Xes = scaler.transform(Xtr_df), scaler.transform(Xte)
        probs, fitted = [], []
        for seed in BAG_SEEDS:
            for name, est in build_bag_learners(seed):
                est.fit(Xs, y, sample_weight=w)
                fitted.append((f"{name}_s{seed}", est))
                probs.append(est.predict_proba(Xes)[:, 1])
        return np.mean(probs, axis=0), scaler, fitted

    tss = TimeSeriesSplit(n_splits=3)
    fold_accs, fold_precs = [], []
    idx = np.arange(len(X))
    for tr, te in tss.split(idx):
        proba_te, _, _ = bag_proba(X.iloc[tr], ytr.iloc[tr].to_numpy(),
                                   X.iloc[te])
        pred = (proba_te >= 0.5).astype(int)
        fold_accs.append(accuracy_score(ytr.iloc[te], pred))
        fold_precs.append(precision_score(ytr.iloc[te], pred,
                                          zero_division=0))
        print(f"  pick5 WF fold: acc {fold_accs[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    wf_acc, wf_prec = float(np.mean(fold_accs)), float(np.mean(fold_precs))
    proba, scaler, fitted = bag_proba(X, ytr.to_numpy(), Xhold)
    m = _metrics(proba, yhold)
    joblib.dump({"scaler": scaler, "learners": fitted},
                LAB / "pick5_bag.pkl", compress=3)
    np.save(LAB / "proba_pick5.npy", proba)
    (LAB / "results_pick5.json").write_text(json.dumps(
        {"metrics": m, "wf3_acc": wf_acc, "wf3_precision": wf_prec,
         "wf3_folds": [float(a) for a in fold_accs]}, indent=1))
    pending, beat_bar = _register(
        "muse_pick5", "Muse's Pick #5",
        "Muse's Pick #5 stacker 'Fresh Eyes': champion four "
        "trees (RF/XGB/LGBM/ET) x 5 seeds = 20 learners on Pick "
        "#2's 98 scored eyes, StandardScaler, NO SMOTE, recency "
        "sample weights (half-life 120d), plain probability vote; "
        "threshold 0.50",
        m, wf_acc, wf_prec, {"proba": proba,
                             "wf3_folds": [float(a) for a in fold_accs]},
        guard_before)
    print(f"final: PICK #5 acc={m['acc']:.4f} prec={m['precision']:.4f} "
          f"up={m['up_calls']}/792 wf3={wf_acc:.4f} -> "
          f"{'BEAT THE BAR / PENDING' if beat_bar else 'PENDING KASS' if pending else 'NO WIN'}",
          flush=True)


def pick6(X, ytr, Xhold, yhold, close, guard_before):
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    abs_next = ((close.shift(-1) / close - 1.0).abs()
                ).reindex(X.index)

    def fit_clean(Xdf, y):
        thr = quartile_threshold(abs_next.reindex(Xdf.index))
        keep = clean_keep_mask(abs_next.reindex(Xdf.index), thr)
        model = SkPipeline([("scaler", StandardScaler()),
                            ("stack", build_wide_stack(42))])
        model.fit(Xdf.to_numpy(dtype=float)[keep],
                  np.asarray(y)[keep])
        return model, float(thr), int(keep.sum())

    tss = TimeSeriesSplit(n_splits=3)
    fold_accs, fold_precs = [], []
    for tr, te in tss.split(np.arange(len(X))):
        model, thr, kept = fit_clean(X.iloc[tr], ytr.iloc[tr].to_numpy())
        proba_te = model.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
        pred = (proba_te >= 0.5).astype(int)
        fold_accs.append(accuracy_score(ytr.iloc[te], pred))
        fold_precs.append(precision_score(ytr.iloc[te], pred,
                                          zero_division=0))
        print(f"  pick6 WF fold: acc {fold_accs[-1]:.4f} "
              f"(kept {kept}, thr {thr:.5f}; {time.time() - t0:.0f}s)",
              flush=True)
    wf_acc, wf_prec = float(np.mean(fold_accs)), float(np.mean(fold_precs))
    model, thr, kept = fit_clean(X, ytr.to_numpy())
    proba = model.predict_proba(Xhold.to_numpy(dtype=float))[:, 1]
    m = _metrics(proba, yhold)
    joblib.dump(model, LAB / "pick6_stack.pkl", compress=3)
    np.save(LAB / "proba_pick6.npy", proba)
    (LAB / "results_pick6.json").write_text(json.dumps(
        {"metrics": m, "wf3_acc": wf_acc, "wf3_precision": wf_prec,
         "wf3_folds": [float(a) for a in fold_accs],
         "clean_threshold": thr, "train_rows_kept": kept}, indent=1))
    pending, beat_bar = _register(
        "muse_pick6", "Muse's Pick #6",
        "Muse's Pick #6 stacker 'Wide Net, Clean Labels': "
        "7-learner stack (RF/XGB/LGBM/ET + kNN-40 + MLP(48,24) + "
        "LR -> LR meta, cv=3) on Pick #2's 98 scored eyes, trained "
        "only on rows whose |next-candle return| clears the train "
        "bottom quartile; StandardScaler, no SMOTE; threshold 0.50",
        m, wf_acc, wf_prec, {"proba": proba,
                             "wf3_folds": [float(a) for a in fold_accs],
                             "clean_threshold": thr,
                             "train_rows_kept": kept},
        guard_before)
    print(f"final: PICK #6 acc={m['acc']:.4f} prec={m['precision']:.4f} "
          f"up={m['up_calls']}/792 wf3={wf_acc:.4f} -> "
          f"{'BEAT THE BAR / PENDING' if beat_bar else 'PENDING KASS' if pending else 'NO WIN'}",
          flush=True)


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    guard_before = _guard()
    X, ytr, Xhold, yhold, close, _ = _setup()
    done = set(_progress()["done"])
    print(f"=== Muse's Pick #4/#5/#6 study (done: {sorted(done)}) ===",
          flush=True)
    if "pick4" not in done:
        pick4(X, ytr, Xhold, yhold, close, guard_before)
        _mark("pick4")
        gc.collect()
    if "pick5" not in done:
        pick5(X, ytr, Xhold, yhold, guard_before)
        _mark("pick5")
        gc.collect()
    if "pick6" not in done:
        pick6(X, ytr, Xhold, yhold, close, guard_before)
        _mark("pick6")
        gc.collect()
    assert _guard() == guard_before, "GUARD TRIPPED at close"
    print("=== Muse's Pick #4/#5/#6 study complete; guard intact ===",
          flush=True)


if __name__ == "__main__":
    main()
