"""Muse's Pick #12 — HA Suite — LAB study (2026-11-11, Kass's
order: "Go ahead and run it and if they fail let's just leave
our current champ to keep retraining").

Pre-registered in refs/muse-pick-12-preregistration-2026-11-11.md
BEFORE any candidate was fitted. The candidate is the exact
Pick recipe (build_pick_pipeline, seed 42) on 19 columns: the
10 HA candle values plus 9 suite columns computed ON the HA
candles (models.muse_pick12). Graded on Picks #9/#10/#11's
IDENTICAL exam (last 792 supervised rows from 2026-10-03
19:00 UTC, majority 0.5088).

Setup stage (before any candidate fit): the rebuilt ha10 block
must equal Pick #10's saved matrix exactly and Pick #11's
artifact must reproduce 0.5341 on the exam — otherwise the
study aborts. The sitting champion (run 123) and former
champion (run 119) are re-scored on the same exam rows.

Bar (frozen): PENDING KASS only if exam acc beats run 123
re-scored AND 0.5341 AND the exam majority, with clean anatomy
and purged walk-forward agreement (5 folds, 24-row embargo).
On FAIL nothing changes (Kass's instruction): champion 123
keeps retraining nightly. Staged and resumable; guards
checked before registration. LAB ONLY.

Usage: .venv/bin/python scripts/build_muse_pick12.py
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
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix  # noqa: E402
from models.muse_pick12 import (SUITE_ALL_COLUMNS,  # noqa: E402
                                build_ha_suite_matrix)
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB9 = ROOT / "data" / "muse_pick9"
LAB11 = ROOT / "data" / "muse_pick11"
LAB = ROOT / "data" / "muse_pick12"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick12"
LABEL = "Stacker — HA Suite (Flat Recipe)"
DESIGN = ("Muse's Pick #12: the standard Pick recipe (RF+XGB+"
          "LGBM+ET -> LR, SMOTE) on 19 columns — the 10 HA "
          "candle values plus Kass's 9-indicator suite computed "
          "on the HA candles (RSI, MACD, EWO, OBV, ADL, net "
          "volume, ATR, SMA50 distance, volatility) — graded on "
          "Picks #9/#10/#11's identical exam with a purged "
          "5-fold walk-forward")
PICK11_ACC = 0.5340909090909091
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


def _frames():
    raw = pd.read_pickle(LAB9 / "raw_ohlcv.pkl")
    sup = pd.read_pickle(LAB9 / "prod_sup.pkl")
    X = build_ha_suite_matrix(raw).loc[sup.index]
    X10 = build_ha_matrix(raw).loc[sup.index]
    y = sup["target"].astype(int).to_numpy()
    return X, X10, y, sup


def _score(pkl_path, X10, y, split) -> dict:
    pipe = joblib.load(pkl_path)
    proba = pipe.predict_proba(X10.iloc[split:])[:, 1]
    return anatomy(proba, y[split:])


# ------------------------------------------------------------- stages
def stage_setup() -> dict:
    meta9 = json.loads((LAB9 / "meta.json").read_text())
    assert meta9["holdout_start"] == "2026-10-03 19:00:00"
    split = meta9["split"]
    X, X10, y, _ = _frames()
    assert list(X.columns) == SUITE_ALL_COLUMNS
    assert X.index.equals(X10.index)
    assert not X.isna().any().any(), "suite frame has NaNs"
    # Frame identity: rebuilt ha10 == Pick #10's saved matrix.
    saved10 = pd.read_pickle(ROOT / "data" / "muse_pick10"
                             / "ha_matrix.pkl")
    pd.testing.assert_frame_equal(X10, saved10)
    # Pick #11's artifact must reproduce its print on this exam.
    m110 = _score(LAB11 / "pick11_stacker.pkl", X10, y, split)
    assert abs(m110["acc"] - PICK11_ACC) < 1e-12, m110
    # Champion re-scores on the identical exam rows.
    champ_now = json.loads((MODELS / "stacker_champion.json")
                           .read_text())
    assert champ_now["run_id"] == 123, champ_now["run_id"]
    m123 = _score(MODELS / "stacker_champion.pkl", X10, y, split)
    arch = sorted((MODELS / "archive").glob(
        "*_pre_champion12/stacker_champion.pkl"))
    assert len(arch) == 1, arch
    m119 = _score(arch[0], X10, y, split)
    meta = {"guard_before": _guard(), "split": split,
            "exam_majority": meta9["exam_majority"],
            "holdout_start": meta9["holdout_start"],
            "rows_sup": int(len(y)),
            "pick11_acc": PICK11_ACC, "pick11_rescore": m110,
            "champ123_rescore": m123, "champ119_rescore": m119}
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: suite frame {X.shape} on the identical exam "
         f"(majority {meta['exam_majority']:.4f}); frame identity "
         f"OK (run 110 reproduces {m110['acc']:.4f}); re-scores: "
         f"run 123 {m123['acc']:.4f} (prec {m123['precision']:.4f}, "
         f"up {m123['up_calls']}), run 119 {m119['acc']:.4f}")
    return meta


def stage_fit_candidate(meta) -> dict:
    X, _, y, _ = _frames()
    split = meta["split"]
    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(X.iloc[:split], y[:split])
    proba = pipe.predict_proba(X.iloc[split:])[:, 1]
    np.save(LAB / "proba_pick12.npy", proba)
    joblib.dump(pipe, LAB / "pick12_stacker.pkl")
    m = anatomy(proba, y[split:])
    _say(f"candidate: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{HOLDOUT} "
         f"({time.time() - t0:.0f}s) vs bar: run 123 re-scored "
         f"{meta['champ123_rescore']['acc']:.4f}, Pick #11 "
         f"{PICK11_ACC:.4f}, majority {meta['exam_majority']:.4f}")
    return m


def stage_wf(meta) -> dict:
    X, _, y, _ = _frames()
    split = meta["split"]
    Xtr = X.iloc[:split]
    ytr = y[:split]
    folds = purged_wf_splits(split, horizon=24, n_splits=5)
    fold_accs, fold_majs, proba_all = [], [], np.full(split, np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(Xtr.iloc[tr], ytr[tr])
        p = pipe.predict_proba(Xtr.iloc[va])[:, 1]
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
    _, _, y, _ = _frames()
    split = meta["split"]
    proba = np.load(LAB / "proba_pick12.npy")
    m = anatomy(proba, y[split:])  # grade recomputed, second way
    assert abs(m["acc"] - cand["acc"]) < 1e-12
    a123 = meta["champ123_rescore"]["acc"]
    a119 = meta["champ119_rescore"]["acc"]
    clean = is_clean(m)
    wf_ok = wf["wf_acc"] > wf["wf_majority"]
    pending = bool(m["acc"] > a123 and m["acc"] > PICK11_ACC
                   and m["acc"] > meta["exam_majority"] and clean
                   and wf_ok)
    pred = (proba >= 0.5).astype(int)
    ey = y[split:]
    w = len(ey) // 3
    window_accs = [float((pred[k * w:(k + 1) * w]
                          == ey[k * w:(k + 1) * w]).mean())
                   for k in range(3)]
    guard_after = _guard()
    assert guard_after == meta["guard_before"], "GUARD TRIPPED"
    base = (f"Muse's Pick #12 holdout acc {m['acc']:.4f} on the "
            f"identical exam (run 123 re-scored {a123:.4f}; run "
            f"119 re-scored {a119:.4f}; Pick #11 {PICK11_ACC:.4f}; "
            f"majority {meta['exam_majority']:.4f})")
    if pending:
        reason = (f"{base} with clean anatomy and purged "
                  f"walk-forward agreement ({wf['wf_acc']:.4f} vs "
                  f"{wf['wf_majority']:.4f}) — WOULD PROMOTE: held "
                  f"as PENDING KASS APPROVAL under the promotion "
                  f"lock; champion kept")
    elif not clean and m["acc"] > a123:
        reason = (f"{base} BUT anatomy is DEGENERATE (up-calls "
                  f"{m['up_calls']}/{len(ey)}, precision "
                  f"{m['precision']:.4f}) — counted as a LOSS under "
                  f"the run-63 rule")
    elif not wf_ok and m["acc"] > a123 and m["acc"] > PICK11_ACC:
        reason = (f"{base} with clean anatomy BUT its purged "
                  f"walk-forward disagrees ({wf['wf_acc']:.4f} vs "
                  f"majority {wf['wf_majority']:.4f}) — FAIL under "
                  f"the pre-registered bar; champion kept")
    else:
        reason = (f"{base} — did not clear the pre-registered "
                  f"bar; FAIL (study verdict only). Per Kass's "
                  f"2026-11-11 instruction, champion 123 keeps "
                  f"retraining; nothing changes")
    metrics = {
        "challenger_acc": m["acc"], "champion_acc": a123,
        "challenger_wf_acc": wf["wf_acc"],
        "challenger_wf_precision": wf["wf_precision_pooled"],
        "precision": m["precision"], "folds": 5,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": FAMILY,
        "feature_set": "hasuite",
        "votes": vote_histogram(proba),
        FAMILY: {
            "label": LABEL, "verdict": "PASS" if pending else "FAIL",
            "design": DESIGN,
            "champ123_rescore_acc": a123,
            "champ119_rescore_acc": a119,
            "pick11_acc": PICK11_ACC,
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
        f"Muse's Pick #12 (family={FAMILY})",
        f"{DESIGN}. Exam (1h, {len(ey)} rows — the identical "
        f"Picks #9/#10/#11 exam) acc {m['acc']:.4f} vs run 123 "
        f"re-scored {a123:.4f} / run 119 re-scored {a119:.4f} / "
        f"Pick #11 {PICK11_ACC:.4f} / majority "
        f"{meta['exam_majority']:.4f}; precision "
        f"{m['precision']:.4f}; up-calls {m['up_calls']}/{len(ey)}; "
        f"purged walk-forward {wf['wf_acc']:.4f} vs majority "
        f"{wf['wf_majority']:.4f} (folds "
        f"{['%.4f' % a for a in wf['fold_accs']]}). "
        f"{'PENDING KASS APPROVAL under the promotion lock.' if pending else 'FAIL — per Kass 2026-11-11, champion 123 keeps retraining; nothing changes.'} "
        f"Nothing installed.")
    out = {"pick": m, "wf": wf, "pending": pending, "meta": meta}
    (LAB / "results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered: acc {m['acc']:.4f} — "
         f"{'PENDING KASS' if pending else 'FAIL'}")
    return out


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = _done()
    if "setup" not in done:
        meta = stage_setup()
        _mark("setup")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("setup: resumed from checkpoint")
    if "fit_candidate" not in done:
        cand = stage_fit_candidate(meta)
        _mark("fit_candidate")
    else:
        proba = np.load(LAB / "proba_pick12.npy")
        _, _, y, _ = _frames()
        cand = anatomy(proba, y[meta["split"]:])
        _say("candidate: resumed from saved probabilities")
    if "wf" not in done:
        wf = stage_wf(meta)
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
