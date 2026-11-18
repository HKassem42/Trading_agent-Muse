"""Muse's Next Try #2 -- LAB study (2026-11-04, Kass's order).

Pre-registered in refs/muse-next-try-2-preregistration-2026-11-04.md.
Try #1 (run 101) failed cleanly; its two design defects, visible
from train data alone, are the only things changed here:

1. Frame selection uses the REAL Pick recipe's walk-forward
   (3-fold TimeSeriesSplit) on the identical train rows -- no
   fast proxy. Candidates: control 98, info 112, full 120 (all
   on the uniform 0..100 scored scale).
2. The decision threshold is fixed at 0.50 -- the OOF threshold
   knob is removed.

The winner gets the standard 5-fold walk-forward, a full-train
fit, and ONE grade on the shared 792-candle holdout against the
champions' Try-#1 re-scores (run 96: 0.5265, run 73: 0.5227,
two-way agreed on the identical timestamps). Exam reuse is
disclosed in the pre-registration and the lab note. LAB ONLY
under the promotion lock; guards hash-checked.

Usage: .venv/bin/python scripts/build_muse_next_try2.py
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

from features import next_try_features, pick2_features, scored_features  # noqa: E402
from models.muse_pick import build_pick_pipeline  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

MODELS = ROOT / "models"
LAB1 = ROOT / "data" / "muse_next_try"
LAB = ROOT / "data" / "muse_next_try2"
RESULTS_PATH = LAB / "results.json"
STACKER_PATH = LAB / "stacker_next_try2.pkl"
HOLDOUT = 792
COLS98 = list(scored_features.scored_columns()) + list(
    pick2_features.NEW_COLUMNS)
NT = next_try_features
FRAMES = {
    "a_control_98": COLS98,
    "b_info_112": COLS98 + NT.INFO_COLUMNS,
    "d_full_120": COLS98 + NT.NEW_COLUMNS,
}
CHAMP96_ACC, CHAMP73_ACC = 0.5265151515151515, 0.5227272727272727
DESIGN2 = ("Muse's Next Try #2 stacker: Pick recipe (RF+XGB+LGBM+ET "
           "-> LR, SMOTE) on an all-scored frame (98/112/120) selected "
           "by the recipe's own 3-fold train walk-forward; threshold "
           "fixed 0.50")
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


def _save(obj):
    RESULTS_PATH.write_text(json.dumps(obj, indent=1, default=str))


def main() -> None:
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit

    LAB.mkdir(parents=True, exist_ok=True)
    guard_before = _guard()
    frame = pd.read_pickle(LAB1 / "frame120.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    split = len(prod_sup) - HOLDOUT
    try1 = json.loads((LAB1 / "results.json").read_text())
    train_idx_all = prod_sup.index[:split]
    Xall = frame.reindex(train_idx_all)
    common = Xall.dropna(subset=COLS98 + NT.NEW_COLUMNS)
    train_idx = common.index
    ytr = prod_sup.reindex(train_idx)["target"].astype(int)
    hold_idx = prod_sup.index[split:]
    yhold = prod_sup.iloc[split:]["target"].astype(int)
    print(f"train rows {len(train_idx)} (identical for all frames); "
          f"holdout {len(hold_idx)} from {hold_idx[0]}", flush=True)

    # ---- selection: the recipe's own 3-fold WF, train only ----
    sel_scores = {}
    tss3 = TimeSeriesSplit(n_splits=3)
    for name, cols in FRAMES.items():
        X = frame.reindex(train_idx)[cols]
        accs = []
        for tr, te in tss3.split(X):
            pipe = build_pick_pipeline(random_state=42)
            pipe.fit(X.iloc[tr], ytr.iloc[tr])
            accs.append(accuracy_score(
                ytr.iloc[te], pipe.predict(X.iloc[te])))
        sel_scores[name] = {"wf3_acc": float(np.mean(accs)),
                            "folds": [float(a) for a in accs],
                            "n_cols": len(cols)}
        print(f"  select {name}: recipe WF3 acc "
              f"{sel_scores[name]['wf3_acc']:.4f} "
              f"{[round(a, 4) for a in accs]}", flush=True)
    order = ["a_control_98", "b_info_112", "d_full_120"]
    selected = max(order, key=lambda k: (sel_scores[k]["wf3_acc"],
                                         -sel_scores[k]["n_cols"],
                                         -order.index(k)))
    cols = FRAMES[selected]
    print(f"selected: {selected} ({len(cols)} cols)", flush=True)

    # ---- winner: 5-fold WF + full fit + one holdout grade ----
    X = frame.reindex(train_idx)[cols]
    tss5 = TimeSeriesSplit(n_splits=5)
    fold_accs, fold_precs = [], []
    for tr, te in tss5.split(X):
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(X.iloc[tr], ytr.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        fold_accs.append(accuracy_score(ytr.iloc[te], pred))
        fold_precs.append(precision_score(ytr.iloc[te], pred,
                                          zero_division=0))
    wf_acc, wf_prec = float(np.mean(fold_accs)), float(np.mean(fold_precs))
    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(X, ytr)
    joblib.dump(pipe, STACKER_PATH)
    print(f"  winner WF5 acc={wf_acc:.4f} prec={wf_prec:.4f}; "
          f"full fit in {time.time() - t0:.0f}s", flush=True)

    Xhold = frame.reindex(hold_idx)[cols]
    assert np.isfinite(Xhold.to_numpy(dtype=float)).all()
    proba = pipe.predict_proba(Xhold)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yhold, pred))
    prec = float(precision_score(yhold, pred, zero_division=0))
    up = int(pred.sum())
    win_accs = [float(accuracy_score(
        yhold.iloc[k * 264:(k + 1) * 264], pred[k * 264:(k + 1) * 264]))
        for k in range(3)]
    majority = float(max(yhold.mean(), 1 - yhold.mean()))
    win = bool(acc > CHAMP96_ACC and acc > CHAMP73_ACC)
    degenerate = bool(win and (up <= 7 or prec <= 0.05))
    eff_win = bool(win and not degenerate)

    scaler = pipe.named_steps["scaler"]
    stack = pipe.named_steps["stack"]
    Xs = scaler.transform(Xhold)
    imps = {}
    for nm, est in stack.named_estimators_.items():
        imp = np.asarray(est.feature_importances_, dtype=float)
        imps[nm] = imp / imp.sum() if imp.sum() > 0 else imp
    agg = np.mean([imps[k] for k in ("rf", "xgb", "lgbm", "et")], axis=0)
    ordr = np.argsort(-agg)
    top15 = [{"feature": cols[i], "importance": float(agg[i])}
             for i in ordr[:15]]
    new_rank = {c: int(np.where(ordr == cols.index(c))[0][0]) + 1
                for c in NT.NEW_COLUMNS if c in cols}

    out = {
        "selected_frame": selected, "selection_wf3": sel_scores,
        "features": cols, "n_features": len(cols),
        "acc": acc, "precision": prec, "up_calls": up,
        "wf_acc": wf_acc, "wf_precision": wf_prec,
        "window_accs": win_accs, "majority_baseline": majority,
        "champion96_acc": CHAMP96_ACC, "champion73_acc": CHAMP73_ACC,
        "try1_acc": try1["final"]["acc"],
        "win": win, "degenerate": degenerate, "effective_win": eff_win,
        "votes": vote_histogram(proba), "top15": top15,
        "new_feature_ranks": new_rank,
        "guard_before": guard_before,
    }
    _save(out)
    print(f"final: NEXT TRY #2 ({selected}) acc={acc:.4f} prec={prec:.4f} "
          f"up={up}/792 wf5={wf_acc:.4f} windows="
          f"{[round(a, 4) for a in win_accs]} | run96 {CHAMP96_ACC:.4f} "
          f"| run73 {CHAMP73_ACC:.4f} | try1 {out['try1_acc']:.4f} | "
          f"majority {majority:.4f} -> {'WIN' if eff_win else 'NO WIN'}",
          flush=True)

    # ---- register ----
    guard_after = _guard()
    assert guard_after == guard_before, "GUARD TRIPPED"
    if eff_win:
        reason = (
            f"Muse's Next Try #2 stacker holdout acc {acc:.4f} beat "
            f"BOTH champions re-scored on the shared holdout (run 96: "
            f"{CHAMP96_ACC:.4f}; run 73 / Muse's Pick #2: "
            f"{CHAMP73_ACC:.4f}) — WOULD PROMOTE: held as PENDING "
            f"KASS APPROVAL under the promotion lock; champion kept "
            f"(study verdict only; second challenger graded on this "
            f"exam — see pre-registration)")
    elif win and degenerate:
        reason = (
            f"Muse's Next Try #2 stacker holdout acc {acc:.4f} "
            f"strictly beat both champions BUT anatomy is DEGENERATE "
            f"(up-calls {up}/792, precision {prec:.4f}) — counted as "
            f"a LOSS under the pre-registered run-63 rule")
    else:
        reason = (
            f"Muse's Next Try #2 stacker holdout acc {acc:.4f} did "
            f"not beat both champions re-scored on the shared holdout "
            f"(run 96 {CHAMP96_ACC:.4f}, run 73 {CHAMP73_ACC:.4f}); "
            f"champion kept (study verdict only)")
    metrics = {
        "challenger_acc": acc, "champion_acc": CHAMP96_ACC,
        "challenger_wf_acc": wf_acc, "challenger_wf_precision": wf_prec,
        "precision": prec, "folds": 5, "pending_approval": eff_win,
        "reason": reason, "deep": False, "family": "muse_next_try2",
        "votes": out["votes"],
        "muse_next_try2": {
            "label": "Muse's Next Try #2", "verdict":
            "PASS" if eff_win else "FAIL", "design": DESIGN2,
            "selected_frame": selected, "selection_wf3": sel_scores,
            "features": cols, "n_features": len(cols),
            "champion73_acc": CHAMP73_ACC, "try1_acc": out["try1_acc"],
            "window_accs": win_accs, "up_calls": up,
            "majority_baseline": majority,
            "new_feature_ranks": new_rank,
            "top_features": [t["feature"] for t in top15],
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        "Muse's Next Try #2 Stacker (family=muse_next_try2)",
        f"{DESIGN2}. Selected frame {selected} by the recipe's own "
        f"train walk-forward. Holdout acc {acc:.4f} vs run 96 "
        f"{CHAMP96_ACC:.4f} and run 73 (Muse's Pick #2) "
        f"{CHAMP73_ACC:.4f} re-scored on identical timestamps; "
        f"precision {prec:.4f}; up-calls {up}/792; walk-forward "
        f"{wf_acc:.4f}; majority baseline {majority:.4f}. "
        f"{'WIN — PENDING KASS APPROVAL under the promotion lock.' if eff_win else 'No win.'} "
        f"Nothing installed.")
    out["registered"] = {"win": eff_win, "guard_after": guard_after}
    _save(out)
    print(f"register: Next Try #2 logged (promoted=0, "
          f"pending={eff_win}); guard intact", flush=True)
    print("=== Muse's Next Try #2 complete ===", flush=True)


if __name__ == "__main__":
    main()
