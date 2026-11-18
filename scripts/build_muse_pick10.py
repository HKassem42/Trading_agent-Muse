"""Muse's Pick #10 — LSTM over Heikin-Ashi sequences — LAB study
(2026-11-05, Kass's order: "Try the exact thing but with the HA
sequence... they form in a row when it's HA").

Pre-registered in refs/muse-pick-10-preregistration-2026-11-05.md
BEFORE any candidate was fitted. Pick #9's exact design with ONE
variable changed: the input matrix. Instead of the 21 production
features, each candle is 10 values computed from its Heikin-Ashi
candle (color, body, range, wicks, signed same-color streak,
slope, displacement, raw return, body share). Everything else
is Pick #9 verbatim: the SAME saved frame and exam (last 792
supervised rows from 2026-10-03 19:00 UTC, majority 0.5088), the
champions' re-scores from Pick #9 (run 96 0.5215, run 73 0.5189 —
two-way verified there, on literally the same rows), the same
LSTM (hidden 64, dropout 0.2, class-weighted BCE, seeds
42/7/123 ensembled), 24-candle windows, purged 5-fold
walk-forward, frozen bar. Control: the flat Pick recipe on the
HA rows.

Staged and resumable; guards checked before registration.
LAB ONLY under the promotion lock.

Usage: .venv/bin/python scripts/build_muse_pick10.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.muse_pick import build_pick_pipeline  # noqa: E402
from models.muse_pick7 import anatomy, is_clean, purged_wf_splits  # noqa: E402
from models.muse_pick9 import WINDOW, fit_predict_ensemble, sequence_rows  # noqa: E402
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB9 = ROOT / "data" / "muse_pick9"
LAB = ROOT / "data" / "muse_pick10"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick10"
LABEL = "Stacker — LSTM HA Sequence (24×10)"
DESIGN = ("Muse's Pick #10: Pick #9's LSTM (hidden 64, dropout "
          "0.2, class-weighted BCE, seeds 42/7/123 ensembled) "
          "over 24-candle windows of a 10-value Heikin-Ashi "
          "candle matrix (color / body / range / wicks / signed "
          "same-color streak / slope / displacement / raw return "
          "/ body share) — Kass's HA-runs thesis; graded on Pick "
          "#9's identical exam against its re-scored champions")
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


def _load_exam():
    """Pick #9's saved frame + verified champion re-scores. The
    exam identity is asserted, not assumed."""
    meta9 = json.loads((LAB9 / "meta.json").read_text())
    champs9 = json.loads((LAB9 / "champs.json").read_text())
    prod_sup = pd.read_pickle(LAB9 / "prod_sup.pkl")
    raw = pd.read_pickle(LAB9 / "raw_ohlcv.pkl")
    assert meta9["split"] == len(prod_sup) - HOLDOUT
    assert meta9["holdout_start"] == "2026-10-03 19:00:00"
    return meta9, champs9, prod_sup, raw


# ------------------------------------------------------------- setup
def stage_setup() -> dict:
    meta9, champs9, prod_sup, raw = _load_exam()
    ha = build_ha_matrix(raw)
    X = ha.reindex(prod_sup.index)
    assert np.isfinite(X.to_numpy(dtype=float)).all(), \
        "non-finite HA values inside the supervised rows"
    X.to_pickle(LAB / "ha_matrix.pkl")
    meta = {
        "guard_before": _guard(),
        "split": meta9["split"],
        "exam_majority": meta9["exam_majority"],
        "holdout_start": meta9["holdout_start"],
        "champ96_acc": champs9["run96"]["acc"],
        "champ73_acc": champs9["run73"]["acc"],
        "rows_sup": int(len(prod_sup)),
    }
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: HA matrix on Pick #9's exam (split {meta['split']}, "
         f"majority {meta['exam_majority']:.4f}); champions run96 "
         f"{meta['champ96_acc']:.4f} / run73 {meta['champ73_acc']:.4f}")
    return meta


def _XY():
    meta = json.loads((LAB / "meta.json").read_text())
    _, _, prod_sup, _ = _load_exam()
    Xdf = pd.read_pickle(LAB / "ha_matrix.pkl")
    assert list(Xdf.columns) == HA_COLUMNS
    y = prod_sup["target"].astype(int).to_numpy()
    return meta, Xdf, y


# ------------------------------------------------------------- fits
def stage_fit_exam() -> dict:
    meta, Xdf, y = _XY()
    split = meta["split"]
    X = Xdf.to_numpy(dtype=float)
    rows = sequence_rows(len(X), WINDOW)
    train_rows = np.array([r for r in rows if r < split], dtype=int)
    exam_rows = np.array([r for r in rows if r >= split], dtype=int)
    assert len(exam_rows) == HOLDOUT
    t0 = time.time()
    proba = fit_predict_ensemble(X, y, train_rows, exam_rows)
    np.save(LAB / "proba_pick10.npy", proba)
    m = anatomy(proba, y[exam_rows])
    _say(f"candidate: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{len(exam_rows)} "
         f"({time.time() - t0:.0f}s)")
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(Xdf.iloc[:split], y[:split])
    pc = pipe.predict_proba(Xdf.iloc[split:])[:, 1]
    mc = anatomy(pc, y[split:])
    _say(f"flat HA control: exam acc {mc['acc']:.4f}")
    out = {"pick10": m, "flat_control": mc}
    (LAB / "exam.json").write_text(json.dumps(out, indent=1))
    return out


def stage_wf() -> dict:
    meta, Xdf, y = _XY()
    split = meta["split"]
    X = Xdf.to_numpy(dtype=float)
    rows = sequence_rows(len(X), WINDOW)
    train_rows = np.array([r for r in rows if r < split], dtype=int)
    folds = purged_wf_splits(len(train_rows), horizon=WINDOW,
                             n_splits=5)
    fold_accs, fold_majs, proba_all = [], [], np.full(len(train_rows), np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        p = fit_predict_ensemble(X, y, train_rows[tr], train_rows[va])
        proba_all[va] = p
        m = anatomy(p, y[train_rows[va]])
        maj = float(max(y[train_rows[va]].mean(),
                        1 - y[train_rows[va]].mean()))
        fold_accs.append(m["acc"])
        fold_majs.append(maj)
        _say(f"wf fold {k}: acc {m['acc']:.4f} vs majority "
             f"{maj:.4f} ({time.time() - t0:.0f}s)")
    pooled = anatomy(proba_all[np.isfinite(proba_all)],
                     y[train_rows][np.isfinite(proba_all)])
    out = {"fold_accs": fold_accs, "fold_majorities": fold_majs,
           "wf_acc": float(np.mean(fold_accs)),
           "wf_majority": float(np.mean(fold_majs)),
           "wf_precision_pooled": pooled["precision"]}
    (LAB / "wf.json").write_text(json.dumps(out, indent=1))
    _say(f"wf: mean acc {out['wf_acc']:.4f} vs majority "
         f"{out['wf_majority']:.4f}")
    return out


# ------------------------------------------------------------- register
def stage_register(meta, exam, wf) -> dict:
    _, Xdf, y = _XY()
    split = meta["split"]
    X = Xdf.to_numpy(dtype=float)
    rows = sequence_rows(len(X), WINDOW)
    exam_rows = np.array([r for r in rows if r >= split], dtype=int)
    proba = np.load(LAB / "proba_pick10.npy")
    m = anatomy(proba, y[exam_rows])  # grade #1 recomputed
    assert abs(m["acc"] - exam["pick10"]["acc"]) < 1e-12
    a96, a73 = meta["champ96_acc"], meta["champ73_acc"]
    best_champ = max(a96, a73)
    clean = is_clean(m)
    wf_ok = wf["wf_acc"] > wf["wf_majority"]
    pending = bool(m["acc"] > best_champ
                   and m["acc"] > meta["exam_majority"] and clean
                   and wf_ok)
    beat_bar = bool(pending and m["acc"] > BAR)
    pred = (proba >= 0.5).astype(int)
    ey = y[exam_rows]
    w = len(ey) // 3
    window_accs = [float((pred[k * w:(k + 1) * w]
                          == ey[k * w:(k + 1) * w]).mean())
                   for k in range(3)]
    guard_after = _guard()
    assert guard_after == meta["guard_before"], "GUARD TRIPPED"
    if pending:
        reason = (f"Muse's Pick #10 holdout acc {m['acc']:.4f} beat "
                  f"both champions on Pick #9's identical exam "
                  f"(run 96 {a96:.4f}; run 73 {a73:.4f}) and the "
                  f"exam majority {meta['exam_majority']:.4f} with "
                  f"clean anatomy and walk-forward agreement — "
                  f"WOULD PROMOTE: held as PENDING KASS APPROVAL "
                  f"under the promotion lock; champion kept")
    elif m["acc"] > best_champ and not clean:
        reason = (f"Muse's Pick #10 holdout acc {m['acc']:.4f} beat "
                  f"the champions BUT anatomy is DEGENERATE "
                  f"(up-calls {m['up_calls']}/{len(ey)}, precision "
                  f"{m['precision']:.4f}) — counted as a LOSS under "
                  f"the run-63 rule")
    elif m["acc"] > best_champ and not wf_ok:
        reason = (f"Muse's Pick #10 holdout acc {m['acc']:.4f} beat "
                  f"both champions on Pick #9's identical exam "
                  f"(run 96 {a96:.4f}; run 73 {a73:.4f}) with clean "
                  f"anatomy BUT walk-forward disagrees "
                  f"({wf['wf_acc']:.4f} vs majority "
                  f"{wf['wf_majority']:.4f}) — FAIL under the "
                  f"pre-registered bar; champion kept")
    else:
        reason = (f"Muse's Pick #10 holdout acc {m['acc']:.4f} did "
                  f"not beat both champions on Pick #9's identical "
                  f"exam (run 96 {a96:.4f}; run 73 {a73:.4f}; "
                  f"majority {meta['exam_majority']:.4f}); champion "
                  f"kept (study verdict only)")
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
            "flat_control_acc": exam["flat_control"]["acc"],
            "majority_baseline": meta["exam_majority"],
            "exam_rows": int(len(ey)), "horizon": 1,
            "window": WINDOW,
            "window_accs": window_accs,
            "up_calls": m["up_calls"],
            "wf_majority": wf["wf_majority"],
            "wf_fold_accs": wf["fold_accs"],
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        f"Muse's Pick #10 (family={FAMILY})",
        f"{DESIGN}. Exam (1h, {len(ey)} rows — Pick #9's "
        f"identical exam) acc {m['acc']:.4f} vs champions run 96 "
        f"{a96:.4f} / run 73 {a73:.4f} / majority "
        f"{meta['exam_majority']:.4f}; flat HA control "
        f"{exam['flat_control']['acc']:.4f}; precision "
        f"{m['precision']:.4f}; up-calls {m['up_calls']}/{len(ey)}; "
        f"purged walk-forward {wf['wf_acc']:.4f} vs majority "
        f"{wf['wf_majority']:.4f}. "
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
    done = _done()
    if "setup" not in done:
        meta = stage_setup()
        _mark("setup")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("setup: resumed from checkpoint")
    if "fit_exam" not in done:
        exam = stage_fit_exam()
        _mark("fit_exam")
    else:
        exam = json.loads((LAB / "exam.json").read_text())
        _say("fit_exam: resumed from checkpoint")
    if "wf" not in done:
        wf = stage_wf()
        _mark("wf")
    else:
        wf = json.loads((LAB / "wf.json").read_text())
        _say("wf: resumed from checkpoint")
    if "register" not in done:
        stage_register(meta, exam, wf)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
