"""Muse's Pick #8 — "Plain 21, Slow Clock" — LAB study
(2026-11-05, Kass's order: "Run Pick #8: test the plain 21
features at 24h as the candidate").

Pre-registered in refs/muse-pick-8-preregistration-2026-11-05.md
BEFORE any candidate was fitted. In Pick #7 (run 106, FAIL),
control arm C1 — the Pick recipe (scaler -> SMOTE -> RF + XGB +
LGBM + ExtraTrees -> LR meta) on the plain 21 production
features, target next-24h direction — printed the program's
highest clean exam number (0.5612) but was a control: no
walk-forward of its own, and the 24h exam's effective sample is
~32 independent rows. Pick #8 promotes that exact configuration
to CANDIDATE on the IDENTICAL exam (machine-checked against
Pick #7's saved split) and gives it the missing grading: a
purged, embargoed walk-forward on the candidate itself.

Frozen bar: PENDING KASS only if exam acc beats Pick #7's
registered candidate on the same exam (0.5404) AND the exam
majority (0.5091), with clean anatomy (run-63 rule) AND
walk-forward agreement. Staged and resumable; guards checked
before registration. LAB ONLY under the promotion lock.

Usage: .venv/bin/python scripts/build_muse_pick8.py
"""

from __future__ import annotations

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

from models.muse_pick import build_pick_pipeline  # noqa: E402
from models.muse_pick7 import (HORIZON, anatomy, exam_split,  # noqa: E402
                               is_clean, purged_wf_splits)
from models.muse_picks456 import horizon_labels  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB1 = ROOT / "data" / "muse_next_try"
LAB7 = ROOT / "data" / "muse_pick7"
LAB = ROOT / "data" / "muse_pick8"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick8"
LABEL = "Stacker — Plain 21 (24h)"
DESIGN = ("Muse's Pick #8 'Plain 21, Slow Clock': the Pick #7 "
          "control arm promoted to candidate — Pick recipe on "
          "the plain 21 production features only, fitted to "
          "next-24h direction on Pick #7's identical exam "
          "(24-row boundary purge), now WITH its own purged, "
          "embargoed walk-forward")
BAR = 0.5354
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


def _done() -> list:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text()).get("done", [])
    return []


def _mark(stage: str) -> None:
    done = _done()
    if stage not in done:
        done.append(stage)
    LAB.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"done": done}, indent=1))


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pick7_refs() -> dict:
    meta7 = json.loads((LAB7 / "meta.json").read_text())
    res7 = json.loads((LAB7 / "results.json").read_text())
    return {"meta": meta7, "c1_acc": res7["control_prod21"]["acc"],
            "pick7_acc": res7["pick"]["acc"],
            "pick7_run_majority": meta7["exam_majority"]}


# ------------------------------------------------------------- setup
def stage_setup_real(refs: dict) -> dict:
    frame = pd.read_pickle(LAB1 / "frame120.pkl")
    raw = pd.read_pickle(LAB1 / "raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    assert frame.index.equals(raw.index), "frame/raw index mismatch"
    raw_pos = raw.index.get_indexer(prod_sup.index)
    assert (raw_pos >= 0).all() and (np.diff(raw_pos) == 1).all(), \
        "supervised grid is not a contiguous slice of the label grid"
    close_full = raw["close"]
    y1 = horizon_labels(close_full, 1).reindex(prod_sup.index)
    assert (y1.dropna().astype(int).to_numpy()
            == prod_sup["target"].reindex(y1.dropna().index)
            .astype(int).to_numpy()).all(), "stored target != 1h labels"
    y24 = horizon_labels(close_full, HORIZON).reindex(prod_sup.index)
    sup = frame.reindex(prod_sup.index)
    # finiteness discipline identical to Pick #7's setup: rows
    # must be finite on the FULL 120-column frame so the row set
    # (and therefore the exam) is byte-identical to Pick #7's.
    from features import next_try_features, pick2_features, scored_features
    cols120 = (list(scored_features.scored_columns())
               + list(pick2_features.NEW_COLUMNS)
               + list(next_try_features.NEW_COLUMNS))
    X120 = sup[cols120]
    finite_rows = np.isfinite(X120.to_numpy(dtype=float)).all(axis=1)
    y_masked = y24.to_numpy(dtype=float).copy()
    y_masked[~finite_rows] = np.nan
    y24 = pd.Series(y_masked, index=prod_sup.index)
    split = len(prod_sup) - HOLDOUT
    train_pos, exam_pos = exam_split(prod_sup.index, y24, split,
                                     horizon=HORIZON)
    assert finite_rows[train_pos].all() and finite_rows[exam_pos].all()
    from features.technical_indicators import FEATURE_COLUMNS
    prod_cols = list(FEATURE_COLUMNS)
    assert len(prod_cols) == 21
    assert all(c in prod_sup.columns for c in prod_cols)
    y = y24.to_numpy(dtype=float)
    exam_y = y[exam_pos].astype(int)
    majority = float(max(exam_y.mean(), 1 - exam_y.mean()))
    meta = {
        "guard_before": _guard(),
        "n_supervised": int(len(prod_sup)),
        "holdout_pos": int(split),
        "n_train": int(len(train_pos)),
        "n_exam": int(len(exam_pos)),
        "exam_majority": majority,
        "prod_cols": prod_cols,
        "train_first": str(prod_sup.index[train_pos[0]]),
        "train_last": str(prod_sup.index[train_pos[-1]]),
        "exam_first": str(prod_sup.index[exam_pos[0]]),
        "exam_last": str(prod_sup.index[exam_pos[-1]]),
    }
    # THE SAME-EXAM CHECK: this split must equal Pick #7's saved
    # split exactly, or the comparison the study exists for is void.
    m7 = refs["meta"]
    assert meta["n_train"] == m7["n_train"], "train size != Pick #7"
    assert meta["n_exam"] == m7["n_exam"], "exam size != Pick #7"
    assert abs(meta["exam_majority"] - m7["exam_majority"]) < 1e-12
    assert meta["exam_first"] == m7["exam_first"], "exam start differs"
    assert meta["train_last"] == m7["train_last"], "train end differs"
    np.savez_compressed(
        LAB / "setup.npz", train_pos=train_pos, exam_pos=exam_pos,
        y=y, prod_cols=np.array(prod_cols))
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: train {len(train_pos)} rows, exam {len(exam_pos)} "
         f"rows from {meta['exam_first']}, majority {majority:.4f} "
         f"— identical to Pick #7's exam (checked)")
    return meta


def _load_setup():
    z = np.load(LAB / "setup.npz", allow_pickle=True)
    meta = json.loads((LAB / "meta.json").read_text())
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    return z, meta, prod_sup


# ------------------------------------------------------------- fit
def stage_fit_candidate(refs: dict) -> dict:
    z, meta, prod_sup = _load_setup()
    train_pos, exam_pos, y = z["train_pos"], z["exam_pos"], z["y"]
    src = prod_sup[list(z["prod_cols"])]
    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(src.iloc[train_pos], y[train_pos].astype(int))
    proba = pipe.predict_proba(src.iloc[exam_pos])[:, 1]
    np.save(LAB / "proba_pick8.npy", proba)
    joblib.dump(pipe, LAB / "pick8_stacker.pkl")
    m = anatomy(proba, y[exam_pos].astype(int))
    delta = m["acc"] - refs["c1_acc"]
    _say(f"candidate: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{len(exam_pos)} "
         f"({time.time() - t0:.0f}s); determinism delta vs Pick "
         f"#7's C1 control: {delta:+.2e}")
    return {**m, "determinism_delta": float(delta)}


# ------------------------------------------------------------- wf
def stage_wf() -> dict:
    z, meta, prod_sup = _load_setup()
    train_pos, y = z["train_pos"], z["y"]
    X = prod_sup[list(z["prod_cols"])].iloc[train_pos] \
        .reset_index(drop=True)
    ytr = y[train_pos].astype(int)
    folds = purged_wf_splits(len(train_pos), horizon=HORIZON,
                             n_splits=3)
    fold_accs, fold_majs, proba_all = [], [], np.full(len(ytr), np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(X.iloc[tr], ytr[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        proba_all[va] = p
        m = anatomy(p, ytr[va])
        maj = float(max(ytr[va].mean(), 1 - ytr[va].mean()))
        fold_accs.append(m["acc"])
        fold_majs.append(maj)
        _say(f"wf fold {k}: acc {m['acc']:.4f} vs majority "
             f"{maj:.4f} ({time.time() - t0:.0f}s)")
    pooled = anatomy(proba_all[np.isfinite(proba_all)],
                     ytr[np.isfinite(proba_all)])
    out = {"fold_accs": fold_accs, "fold_majorities": fold_majs,
           "wf_acc": float(np.mean(fold_accs)),
           "wf_majority": float(np.mean(fold_majs)),
           "wf_precision_pooled": pooled["precision"]}
    (LAB / "wf.json").write_text(json.dumps(out, indent=1))
    _say(f"wf: mean acc {out['wf_acc']:.4f} vs majority "
         f"{out['wf_majority']:.4f}")
    return out


# ------------------------------------------------------------- register
def stage_register(meta, cand, wf, refs) -> dict:
    z, _, _ = _load_setup()
    exam_pos, y = z["exam_pos"], z["y"]
    exam_y = y[exam_pos].astype(int)
    proba = np.load(LAB / "proba_pick8.npy")
    m = anatomy(proba, exam_y)  # grade #1 recomputed from saved proba
    assert abs(m["acc"] - cand["acc"]) < 1e-12
    bar_acc = refs["pick7_acc"]
    clean = is_clean(m)
    wf_ok = wf["wf_acc"] > wf["wf_majority"]
    pending = bool(m["acc"] > bar_acc
                   and m["acc"] > meta["exam_majority"] and clean
                   and wf_ok)
    beat_bar = bool(pending and m["acc"] > BAR)
    pred = (proba >= 0.5).astype(int)
    w = len(exam_y) // 3
    window_accs = [float((pred[k * w:(k + 1) * w]
                          == exam_y[k * w:(k + 1) * w]).mean())
                   for k in range(3)]
    nonov = anatomy(proba[::HORIZON], exam_y[::HORIZON])
    guard_after = _guard()
    assert guard_after == meta["guard_before"], "GUARD TRIPPED"
    det = cand["determinism_delta"]
    det_txt = (f" Candidate fit reproduced Pick #7's C1 control "
               f"exactly (delta {det:+.1e})." if abs(det) < 1e-9
               else f" NOTE: candidate exam acc differs from Pick "
                    f"#7's C1 control by {det:+.4f} (pipeline "
                    f"nondeterminism) — the candidate's own numbers "
                    f"stand as measured.")
    if pending:
        reason = (f"Muse's Pick #8 holdout(24h) acc {m['acc']:.4f} "
                  f"beat Pick #7's candidate on the identical exam "
                  f"({bar_acc:.4f}) and the exam majority "
                  f"({meta['exam_majority']:.4f}) with clean "
                  f"anatomy and purged walk-forward agreement "
                  f"({wf['wf_acc']:.4f} vs {wf['wf_majority']:.4f}) "
                  f"— WOULD PROMOTE: held as PENDING KASS "
                  f"APPROVAL under the promotion lock; champion "
                  f"kept. NOTE: 24h-horizon model — not a drop-in "
                  f"for the 1h champions; seating needs its own "
                  f"design decision.")
    elif m["acc"] > bar_acc and not clean:
        reason = (f"Muse's Pick #8 holdout(24h) acc {m['acc']:.4f} "
                  f"beat the same-exam bar BUT anatomy is "
                  f"DEGENERATE (up-calls {m['up_calls']}/"
                  f"{len(exam_y)}, precision {m['precision']:.4f}) "
                  f"— counted as a LOSS under the run-63 rule")
    elif m["acc"] > bar_acc and not wf_ok:
        reason = (f"Muse's Pick #8 holdout(24h) acc {m['acc']:.4f} "
                  f"beat Pick #7's candidate on the identical exam "
                  f"({bar_acc:.4f}) with clean anatomy BUT its "
                  f"purged walk-forward disagrees ({wf['wf_acc']:.4f} "
                  f"vs majority {wf['wf_majority']:.4f}) — FAIL "
                  f"under the pre-registered bar; champion kept "
                  f"(study verdict only)")
    else:
        reason = (f"Muse's Pick #8 holdout(24h) acc {m['acc']:.4f} "
                  f"did not clear the pre-registered bar (Pick #7 "
                  f"candidate {bar_acc:.4f} and majority "
                  f"{meta['exam_majority']:.4f} on the identical "
                  f"exam); champion kept (study verdict only)")
    metrics = {
        "challenger_acc": m["acc"], "champion_acc": bar_acc,
        "challenger_wf_acc": wf["wf_acc"],
        "challenger_wf_precision": wf["wf_precision_pooled"],
        "precision": m["precision"], "folds": 3,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": FAMILY,
        "votes": vote_histogram(proba),
        FAMILY: {
            "label": LABEL, "verdict": "PASS" if pending else "FAIL",
            "design": DESIGN, "beat_bar": beat_bar,
            "pick7_candidate_acc": bar_acc,
            "pick7_control_c1_acc": refs["c1_acc"],
            "determinism_delta": det,
            "majority_baseline": meta["exam_majority"],
            "exam_rows": int(len(exam_y)), "horizon": HORIZON,
            "window_accs": window_accs,
            "up_calls": m["up_calls"],
            "wf_majority": wf["wf_majority"],
            "wf_fold_accs": wf["fold_accs"],
            "nonoverlap_acc": nonov["acc"],
            "nonoverlap_n": int(len(exam_y[::HORIZON])),
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        f"Muse's Pick #8 (family={FAMILY})",
        f"{DESIGN}. Exam (24h, {len(exam_y)} rows) acc "
        f"{m['acc']:.4f} vs Pick #7 candidate {bar_acc:.4f} / "
        f"majority {meta['exam_majority']:.4f} on the identical "
        f"exam; precision {m['precision']:.4f}; up-calls "
        f"{m['up_calls']}/{len(exam_y)}; purged walk-forward "
        f"{wf['wf_acc']:.4f} vs majority {wf['wf_majority']:.4f} "
        f"(folds {['%.4f' % a for a in wf['fold_accs']]}); "
        f"non-overlap diagnostic {nonov['acc']:.4f} "
        f"(n={len(exam_y[::HORIZON])}).{det_txt} "
        f"{'PENDING KASS APPROVAL under the promotion lock.' if pending else 'No win.'} "
        f"Nothing installed.")
    out = {"pick": m, "wf": wf, "pending": pending,
           "beat_bar": beat_bar, "window_accs": window_accs,
           "nonoverlap": nonov, "meta": meta, "refs": refs}
    (LAB / "results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered: acc {m['acc']:.4f} — "
         f"{'PENDING KASS' if pending else 'NO WIN'}")
    return out


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    refs = _pick7_refs()
    done = _done()
    if "setup" not in done:
        meta = stage_setup_real(refs)
        _mark("setup")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("setup: resumed from checkpoint")
    if "fit_candidate" not in done:
        cand = stage_fit_candidate(refs)
        _mark("fit_candidate")
    else:
        z, _, _ = _load_setup()
        proba = np.load(LAB / "proba_pick8.npy")
        cand = anatomy(proba, z["y"][z["exam_pos"]].astype(int))
        cand = {**cand, "determinism_delta":
                float(cand["acc"] - refs["c1_acc"])}
        _say("candidate: resumed from saved probabilities")
    if "wf" not in done:
        wf = stage_wf()
        _mark("wf")
    else:
        wf = json.loads((LAB / "wf.json").read_text())
        _say("wf: resumed from checkpoint")
    if "register" not in done:
        stage_register(meta, cand, wf, refs)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
