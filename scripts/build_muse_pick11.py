"""Muse's Pick #11 — Flat Recipe on HA Candles — LAB study
(2026-11-05, Kass's order: "Run Pick #11: the flat recipe on HA
candles as the candidate, with its own walk-forward").

Pre-registered in refs/muse-pick-11-preregistration-2026-11-05.md
BEFORE any candidate was fitted. In Pick #10 (run 109, FAIL),
the control arm — the standard Pick recipe on the 10-value
Heikin-Ashi candle matrix, read flat — printed 0.5341 on the
exam, above both re-scored champions, but it was a control with
no walk-forward. Pick #11 promotes that exact configuration to
CANDIDATE on the IDENTICAL exam (Pick #9's saved frame; last
792 supervised rows from 2026-10-03 19:00 UTC; majority
0.5088; champions run 96 0.5215 / run 73 0.5189, two-way
verified in Pick #9 on the same rows) and gives it the missing
grading: a purged 5-fold walk-forward (24-row embargo).

The exam fit is the same computation as Pick #10's control, so
its accuracy is expected to reproduce 0.5341 exactly; the
determinism delta is recorded either way.

Staged and resumable; guards checked before registration.
LAB ONLY under the promotion lock.

Usage: .venv/bin/python scripts/build_muse_pick11.py
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
from models.muse_pick7 import anatomy, is_clean, purged_wf_splits  # noqa: E402
from models.muse_pick10 import HA_COLUMNS  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB9 = ROOT / "data" / "muse_pick9"
LAB10 = ROOT / "data" / "muse_pick10"
LAB = ROOT / "data" / "muse_pick11"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick11"
LABEL = "Stacker — Flat Recipe on HA Candles"
DESIGN = ("Muse's Pick #11: Pick #10's control arm promoted to "
          "candidate — the standard Pick recipe (RF+XGB+LGBM+ET "
          "-> LR, SMOTE) on the 10-value Heikin-Ashi candle "
          "matrix read flat, graded on Picks #9/#10's identical "
          "exam, now WITH its own purged 5-fold walk-forward")
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
    "run73_artifact": ROOT / "data" / "muse_pick2" / "stacker_pick2.pkl",
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


def _refs() -> dict:
    meta9 = json.loads((LAB9 / "meta.json").read_text())
    champs9 = json.loads((LAB9 / "champs.json").read_text())
    exam10 = json.loads((LAB10 / "exam.json").read_text())
    assert meta9["holdout_start"] == "2026-10-03 19:00:00"
    return {"split": meta9["split"],
            "exam_majority": meta9["exam_majority"],
            "holdout_start": meta9["holdout_start"],
            "champ96_acc": champs9["run96"]["acc"],
            "champ73_acc": champs9["run73"]["acc"],
            "control10_acc": exam10["flat_control"]["acc"],
            "control10_anatomy": exam10["flat_control"]}


def _XY():
    prod_sup = pd.read_pickle(LAB9 / "prod_sup.pkl")
    Xdf = pd.read_pickle(LAB10 / "ha_matrix.pkl")
    assert list(Xdf.columns) == HA_COLUMNS
    assert Xdf.index.equals(prod_sup.index)
    y = prod_sup["target"].astype(int).to_numpy()
    return Xdf, y


# ------------------------------------------------------------- stages
def stage_setup(refs) -> dict:
    Xdf, y = _XY()
    meta = {"guard_before": _guard(), **refs,
            "rows_sup": int(len(y))}
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: HA matrix {Xdf.shape} on the identical exam "
         f"(majority {refs['exam_majority']:.4f}); Pick #10 "
         f"control acc on record {refs['control10_acc']:.4f} "
         f"(up {refs['control10_anatomy']['up_calls']}, prec "
         f"{refs['control10_anatomy']['precision']:.4f})")
    return meta


def stage_fit_candidate(refs) -> dict:
    Xdf, y = _XY()
    split = refs["split"]
    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(Xdf.iloc[:split], y[:split])
    proba = pipe.predict_proba(Xdf.iloc[split:])[:, 1]
    np.save(LAB / "proba_pick11.npy", proba)
    joblib.dump(pipe, LAB / "pick11_stacker.pkl")
    m = anatomy(proba, y[split:])
    delta = m["acc"] - refs["control10_acc"]
    _say(f"candidate: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{HOLDOUT} "
         f"({time.time() - t0:.0f}s); determinism delta vs Pick "
         f"#10's control: {delta:+.2e}")
    return {**m, "determinism_delta": float(delta)}


def stage_wf(refs) -> dict:
    Xdf, y = _XY()
    split = refs["split"]
    Xtr_df = Xdf.iloc[:split]
    ytr = y[:split]
    folds = purged_wf_splits(split, horizon=24, n_splits=5)
    fold_accs, fold_majs, proba_all = [], [], np.full(split, np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(Xtr_df.iloc[tr], ytr[tr])
        p = pipe.predict_proba(Xtr_df.iloc[va])[:, 1]
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


def stage_register(meta, cand, wf) -> dict:
    _, y = _XY()
    split = meta["split"]
    proba = np.load(LAB / "proba_pick11.npy")
    m = anatomy(proba, y[split:])  # grade #1 recomputed
    assert abs(m["acc"] - cand["acc"]) < 1e-12
    a96, a73 = meta["champ96_acc"], meta["champ73_acc"]
    best_champ = max(a96, a73)
    clean = is_clean(m)
    wf_ok = wf["wf_acc"] > wf["wf_majority"]
    pending = bool(m["acc"] > best_champ
                   and m["acc"] > meta["exam_majority"] and clean
                   and wf_ok)
    beat_bar = bool(pending and m["acc"] > BAR)
    pred = (proba >= 0.5).astype(int)
    ey = y[split:]
    w = len(ey) // 3
    window_accs = [float((pred[k * w:(k + 1) * w]
                          == ey[k * w:(k + 1) * w]).mean())
                   for k in range(3)]
    guard_after = _guard()
    assert guard_after == meta["guard_before"], "GUARD TRIPPED"
    det = cand["determinism_delta"]
    det_txt = (f" Candidate fit reproduced Pick #10's control "
               f"exactly (delta {det:+.1e})." if abs(det) < 1e-9
               else f" NOTE: candidate exam acc differs from Pick "
                    f"#10's control by {det:+.4f} — the candidate's "
                    f"own numbers stand as measured.")
    if pending:
        reason = (f"Muse's Pick #11 holdout acc {m['acc']:.4f} beat "
                  f"both champions on the identical exam (run 96 "
                  f"{a96:.4f}; run 73 {a73:.4f}) and the exam "
                  f"majority {meta['exam_majority']:.4f} with clean "
                  f"anatomy and purged walk-forward agreement "
                  f"({wf['wf_acc']:.4f} vs {wf['wf_majority']:.4f}) "
                  f"— WOULD PROMOTE: held as PENDING KASS "
                  f"APPROVAL under the promotion lock; champion kept")
    elif m["acc"] > best_champ and not clean:
        reason = (f"Muse's Pick #11 holdout acc {m['acc']:.4f} beat "
                  f"the champions BUT anatomy is DEGENERATE "
                  f"(up-calls {m['up_calls']}/{len(ey)}, precision "
                  f"{m['precision']:.4f}) — counted as a LOSS under "
                  f"the run-63 rule")
    elif m["acc"] > best_champ and not wf_ok:
        reason = (f"Muse's Pick #11 holdout acc {m['acc']:.4f} beat "
                  f"both champions on the identical exam (run 96 "
                  f"{a96:.4f}; run 73 {a73:.4f}) with clean anatomy "
                  f"BUT its purged walk-forward disagrees "
                  f"({wf['wf_acc']:.4f} vs majority "
                  f"{wf['wf_majority']:.4f}) — FAIL under the "
                  f"pre-registered bar; champion kept")
    else:
        reason = (f"Muse's Pick #11 holdout acc {m['acc']:.4f} did "
                  f"not beat both champions on the identical exam "
                  f"(run 96 {a96:.4f}; run 73 {a73:.4f}; majority "
                  f"{meta['exam_majority']:.4f}); champion kept "
                  f"(study verdict only)")
    metrics = {
        "challenger_acc": m["acc"], "champion_acc": best_champ,
        "challenger_wf_acc": wf["wf_acc"],
        "challenger_wf_precision": wf["wf_precision_pooled"],
        "precision": m["precision"], "folds": 5,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": FAMILY,
        "votes": vote_histogram(proba),
        FAMILY: {
            "label": LABEL, "verdict": "PASS" if pending else "FAIL",
            "design": DESIGN, "beat_bar": beat_bar,
            "champ96_acc": a96, "champ73_acc": a73,
            "pick10_control_acc": meta["control10_acc"],
            "determinism_delta": det,
            "majority_baseline": meta["exam_majority"],
            "exam_rows": int(len(ey)), "horizon": 1,
            "window_accs": window_accs,
            "up_calls": m["up_calls"],
            "wf_majority": wf["wf_majority"],
            "wf_fold_accs": wf["fold_accs"],
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        f"Muse's Pick #11 (family={FAMILY})",
        f"{DESIGN}. Exam (1h, {len(ey)} rows — Picks #9/#10's "
        f"identical exam) acc {m['acc']:.4f} vs champions run 96 "
        f"{a96:.4f} / run 73 {a73:.4f} / majority "
        f"{meta['exam_majority']:.4f}; precision "
        f"{m['precision']:.4f}; up-calls {m['up_calls']}/{len(ey)}; "
        f"purged walk-forward {wf['wf_acc']:.4f} vs majority "
        f"{wf['wf_majority']:.4f} (folds "
        f"{['%.4f' % a for a in wf['fold_accs']]}).{det_txt} "
        f"{'BEAT THE BAR — PENDING KASS APPROVAL.' if beat_bar else 'PENDING KASS APPROVAL under the promotion lock.' if pending else 'No win.'} "
        f"Nothing installed.")
    out = {"pick": m, "wf": wf, "pending": pending,
           "beat_bar": beat_bar, "meta": meta}
    (LAB / "results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered: acc {m['acc']:.4f} — "
         f"{'BEAT THE BAR / PENDING KASS' if beat_bar else 'PENDING KASS' if pending else 'NO WIN'}")
    return out


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    refs = _refs()
    done = _done()
    if "setup" not in done:
        meta = stage_setup(refs)
        _mark("setup")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("setup: resumed from checkpoint")
    if "fit_candidate" not in done:
        cand = stage_fit_candidate(refs)
        _mark("fit_candidate")
    else:
        proba = np.load(LAB / "proba_pick11.npy")
        _, y = _XY()
        cand = anatomy(proba, y[refs["split"]:])
        cand = {**cand, "determinism_delta":
                float(cand["acc"] - refs["control10_acc"])}
        _say("candidate: resumed from saved probabilities")
    if "wf" not in done:
        wf = stage_wf(refs)
        _mark("wf")
    else:
        wf = json.loads((LAB / "wf.json").read_text())
        _say("wf: resumed from checkpoint")
    if "register" not in done:
        stage_register(meta, cand, wf)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
