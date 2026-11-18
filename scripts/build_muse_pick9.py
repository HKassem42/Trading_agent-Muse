"""Muse's Pick #9 — LSTM sequence stacker — LAB study
(2026-11-05, Kass's order: "return back to the 1 hr matrix but
pick a complete new technique from our rl combo document...
something we have not tried before at all").

Pre-registered in refs/muse-pick-9-preregistration-2026-11-05.md
BEFORE any candidate was fitted. The candidate is the
combinations document's LSTM entry — "time-windowed matrices,
e.g. 24-hour lookbacks": an LSTM (hidden 64, dropout 0.2,
class-weighted BCE, seeds 42/7/123 ensembled) reading the
ordered last-24-rows window of the production 21-feature matrix,
graded on the standard 792-row holdout at the 1h horizon.
Champions (run 96 prod21, run 73 pick2_98) are re-scored on the
identical exam TWO independent ways each; the ways must agree
exactly. Control: the flat Pick recipe on prod21 rows.
Walk-forward: 5 purged folds (24-row embargo) on the training
region, candidate = the same 3-seed ensemble.

Staged and resumable; guards checked before registration.
LAB ONLY under the promotion lock.

Usage: .venv/bin/python scripts/build_muse_pick9.py
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

from data.market_data import fetch_klines  # noqa: E402
from features import pick2_features, scored_features  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, feature_matrix, make_supervised)
from models.muse_pick import build_pick_pipeline  # noqa: E402
from models.muse_pick7 import anatomy, is_clean, purged_wf_splits  # noqa: E402
from models.muse_pick9 import WINDOW, fit_predict_ensemble, sequence_rows  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB = ROOT / "data" / "muse_pick9"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick9"
LABEL = "Stacker — LSTM Sequence (24×21)"
DESIGN = ("Muse's Pick #9: LSTM (hidden 64, dropout 0.2, "
          "class-weighted BCE, seeds 42/7/123 ensembled) over "
          "24-candle windows of the production 21-feature matrix "
          "— the RL/ML combinations document's LSTM entry "
          "('time-windowed matrices, 24-hour lookbacks'), first "
          "time built in this program; 1h horizon, standard "
          "792-row holdout, purged 5-fold walk-forward")
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
COLS98 = list(scored_features.scored_columns()) + list(
    pick2_features.NEW_COLUMNS)


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


# ------------------------------------------------------------- data
def stage_data() -> dict:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    prod_sup = make_supervised(df)
    split = len(prod_sup) - HOLDOUT
    df.to_pickle(LAB / "raw_ohlcv.pkl")
    prod_sup.to_pickle(LAB / "prod_sup.pkl")
    meta = {
        "guard_before": _guard(),
        "rows_raw": int(len(df)), "rows_sup": int(len(prod_sup)),
        "split": int(split),
        "holdout_start": str(prod_sup.index[split]),
        "data_first": str(df.index[0]), "data_last": str(df.index[-1]),
        "exam_majority": float(max(
            prod_sup["target"].iloc[split:].mean(),
            1 - prod_sup["target"].iloc[split:].mean())),
    }
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"data: raw {len(df)} rows, supervised {len(prod_sup)}, "
         f"holdout from {meta['holdout_start']}, exam majority "
         f"{meta['exam_majority']:.4f}")
    return meta


def _load_frames():
    meta = json.loads((LAB / "meta.json").read_text())
    df = pd.read_pickle(LAB / "raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(LAB / "prod_sup.pkl")
    return meta, df, prod_sup


# ------------------------------------------------------------- frames
def stage_frames() -> dict:
    meta, df, prod_sup = _load_frames()
    # prod21 truncation causality at two cuts: features of shared
    # supervised rows must be IDENTICAL when the future is cut.
    X_full, _ = feature_matrix(prod_sup)
    assert list(X_full.columns) == list(FEATURE_COLUMNS)
    ev = {}
    for K in (9000, len(df) - 5000):
        sup_t = make_supervised(df.iloc[:K])
        X_t, _ = feature_matrix(sup_t)
        shared = X_t.index.intersection(X_full.index)
        # the truncated build's last row may lack a label twin;
        # compare FEATURE values on shared rows only
        a = X_full.loc[shared].to_numpy(dtype=float)
        b = X_t.loc[shared].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isnan(a) == np.isnan(b)).all(), f"NaN map differs at {K}"
        worst = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        assert worst <= 1e-9, f"prod21 causality FAIL at K={K}: {worst}"
        ev[str(K)] = worst
        _say(f"frames: prod21 truncation K={K} PASS (max diff {worst})")
    # cols98 frame for run 73's way-A re-score: build_pick2_frame
    # returns the full joined 98-column frame directly
    frame98, cols98, _ev = pick2_features.build_pick2_frame(df)
    assert list(cols98) == COLS98, "build_pick2_frame cols != COLS98"
    frame98 = frame98[COLS98]
    frame98.to_pickle(LAB / "frame98.pkl")
    out = {"causality": ev, "frame98_cols": len(COLS98)}
    (LAB / "frames.json").write_text(json.dumps(out, indent=1))
    _say(f"frames: cols98 built ({len(COLS98)} columns)")
    return out


# ------------------------------------------------------------- champions
def stage_champions() -> dict:
    meta, df, prod_sup = _load_frames()
    split = meta["split"]
    hold = prod_sup.iloc[split:]
    yhold = hold["target"].astype(int)
    frame98 = pd.read_pickle(LAB / "frame98.pkl")
    # run 96 (prod21), two ways
    champ96 = joblib.load(MODELS / "stacker_champion.pkl")
    meta96 = json.loads((MODELS / "stacker_champion.json").read_text())
    assert meta96["run_id"] == 96 and meta96["feature_set"] == "prod21"
    Xa, tag = nightly.champion_holdout_frame(df, hold, meta96)
    assert tag == "prod21"
    Xb, _ = feature_matrix(hold)
    pa = champ96.predict_proba(Xa)[:, 1]
    pb = champ96.predict_proba(Xb)[:, 1]
    assert np.max(np.abs(pa - pb)) <= 1e-12, "run96 two-way mismatch"
    m96 = anatomy(pa, yhold.to_numpy())
    # run 73 (pick2_98), two ways
    champ73 = joblib.load(ROOT / "data" / "muse_pick2" /
                          "stacker_pick2.pkl")
    X73a = frame98.reindex(hold.index)[COLS98]
    assert np.isfinite(X73a.to_numpy(dtype=float)).all()
    p73a = champ73.predict_proba(X73a)[:, 1]
    meta73 = {"feature_set": "pick2_98", "feature_columns": COLS98}
    X73b, tag73 = nightly.champion_holdout_frame(df, hold, meta73)
    assert tag73 == "pick2_98"
    seat73 = joblib.load(MODELS / "shadow_seat" / "stacker.pkl")
    p73b = seat73.predict_proba(X73b[COLS98])[:, 1]
    assert np.max(np.abs(p73a - p73b)) <= 1e-12, "run73 two-way mismatch"
    m73 = anatomy(p73a, yhold.to_numpy())
    out = {"run96": m96, "run73": m73}
    (LAB / "champs.json").write_text(json.dumps(out, indent=1))
    _say(f"champions re-scored (two ways each, exact): run96 "
         f"{m96['acc']:.4f}, run73 {m73['acc']:.4f}")
    return out


# ------------------------------------------------------------- fits
def _XY():
    meta, df, prod_sup = _load_frames()
    X, _ = feature_matrix(prod_sup)
    y = prod_sup["target"].astype(int).to_numpy()
    return meta, X.to_numpy(dtype=float), y


def stage_fit_exam() -> dict:
    meta, X, y = _XY()
    split = meta["split"]
    rows = sequence_rows(len(X), WINDOW)
    train_rows = np.array([r for r in rows if r < split], dtype=int)
    exam_rows = np.array([r for r in rows if r >= split], dtype=int)
    assert len(exam_rows) == HOLDOUT
    t0 = time.time()
    proba = fit_predict_ensemble(X, y, train_rows, exam_rows)
    np.save(LAB / "proba_pick9.npy", proba)
    m = anatomy(proba, y[exam_rows])
    _say(f"candidate: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{len(exam_rows)} "
         f"({time.time() - t0:.0f}s)")
    # flat control: Pick recipe on flat prod21 rows
    _, _, prod_sup = _load_frames()
    Xdf, _ = feature_matrix(prod_sup)
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(Xdf.iloc[:split], y[:split])
    pc = pipe.predict_proba(Xdf.iloc[split:])[:, 1]
    mc = anatomy(pc, y[split:])
    _say(f"flat control: exam acc {mc['acc']:.4f}")
    out = {"pick9": m, "flat_control": mc}
    (LAB / "exam.json").write_text(json.dumps(out, indent=1))
    return out


def stage_wf() -> dict:
    meta, X, y = _XY()
    split = meta["split"]
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
def stage_register(meta, exam, champs, wf) -> dict:
    _, X, y = _XY()
    split = meta["split"]
    rows = sequence_rows(len(X), WINDOW)
    exam_rows = np.array([r for r in rows if r >= split], dtype=int)
    proba = np.load(LAB / "proba_pick9.npy")
    m = anatomy(proba, y[exam_rows])  # grade #1 recomputed
    assert abs(m["acc"] - exam["pick9"]["acc"]) < 1e-12
    m96, m73 = champs["run96"], champs["run73"]
    best_champ = max(m96["acc"], m73["acc"])
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
        reason = (f"Muse's Pick #9 holdout acc {m['acc']:.4f} beat "
                  f"both champions re-scored on the identical exam "
                  f"(run 96 {m96['acc']:.4f}; run 73 {m73['acc']:.4f}) "
                  f"and the exam majority {meta['exam_majority']:.4f} "
                  f"with clean anatomy and walk-forward agreement — "
                  f"WOULD PROMOTE: held as PENDING KASS APPROVAL "
                  f"under the promotion lock; champion kept")
    elif m["acc"] > best_champ and not clean:
        reason = (f"Muse's Pick #9 holdout acc {m['acc']:.4f} beat "
                  f"the champions BUT anatomy is DEGENERATE "
                  f"(up-calls {m['up_calls']}/{len(ey)}, precision "
                  f"{m['precision']:.4f}) — counted as a LOSS under "
                  f"the run-63 rule")
    elif m["acc"] > best_champ and not wf_ok:
        reason = (f"Muse's Pick #9 holdout acc {m['acc']:.4f} beat "
                  f"both champions re-scored on the identical exam "
                  f"(run 96 {m96['acc']:.4f}; run 73 {m73['acc']:.4f}) "
                  f"with clean anatomy BUT walk-forward disagrees "
                  f"({wf['wf_acc']:.4f} vs majority "
                  f"{wf['wf_majority']:.4f}) — FAIL under the "
                  f"pre-registered bar; champion kept")
    else:
        reason = (f"Muse's Pick #9 holdout acc {m['acc']:.4f} did "
                  f"not beat both champions re-scored on the "
                  f"identical exam (run 96 {m96['acc']:.4f}; run 73 "
                  f"{m73['acc']:.4f}; majority "
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
            "champ96_acc": m96["acc"], "champ73_acc": m73["acc"],
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
        f"Muse's Pick #9 (family={FAMILY})",
        f"{DESIGN}. Exam (1h, {len(ey)} rows) acc {m['acc']:.4f} "
        f"vs champions re-scored run 96 {m96['acc']:.4f} / run 73 "
        f"{m73['acc']:.4f} / majority {meta['exam_majority']:.4f}; "
        f"flat prod21 control {exam['flat_control']['acc']:.4f}; "
        f"precision {m['precision']:.4f}; up-calls "
        f"{m['up_calls']}/{len(ey)}; purged walk-forward "
        f"{wf['wf_acc']:.4f} vs majority {wf['wf_majority']:.4f}. "
        f"{'BEAT THE BAR — PENDING KASS APPROVAL.' if beat_bar else 'PENDING KASS APPROVAL under the promotion lock.' if pending else 'No win.'} "
        f"Nothing installed.")
    out = {"pick": m, "champs": champs, "wf": wf,
           "pending": pending, "beat_bar": beat_bar, "meta": meta}
    (LAB / "results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered: acc {m['acc']:.4f} — "
         f"{'BEAT THE BAR / PENDING KASS' if beat_bar else 'PENDING KASS' if pending else 'NO WIN'}")
    return out


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = _done()
    if "data" not in done:
        meta = stage_data()
        _mark("data")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("data: resumed from checkpoint")
    if "frames" not in done:
        stage_frames()
        _mark("frames")
    if "champions" not in done:
        champs = stage_champions()
        _mark("champions")
    else:
        champs = json.loads((LAB / "champs.json").read_text())
        _say("champions: resumed from checkpoint")
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
        stage_register(meta, exam, champs, wf)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
