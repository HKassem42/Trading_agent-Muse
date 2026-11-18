"""Muse's Pick #7 — "Slow Clock" — LAB study (2026-11-05,
Kass's order: "try muses pick the next one for training a model.
Pick whatever u want and give it another shot").

Pre-registered in refs/muse-pick-7-preregistration-2026-11-05.md
BEFORE any candidate was fitted. The Pick recipe (scaler ->
SMOTE -> RF + XGB + LGBM + ExtraTrees -> LR meta) is fitted to
next-24h direction on the Next Try study's causal 120-column
frame (98 scored eyes + 22 positioning/cross-asset columns that
failed at the 1h horizon in runs 101/102 and were never tested
at 24h). Controls on the identical exam: the same recipe on the
21 production features and on the 98 eyes alone. Diagnostic
ablation: the 22 new-information columns alone.

Split discipline (pinned by tests/test_muse_pick7.py): exam =
the standard 792-row holdout window restricted to rows with a
defined 24h label (expected 768); train = supervised rows
before it MINUS a 24-row purge, so no train label reads a close
inside the exam period. Walk-forward: 3 folds with a 24-row
embargo at every boundary. Threshold fixed 0.50.

Staged and resumable (the picks456 lesson — that build died in a
runtime restart mid-fold and lost everything): each stage
writes its outputs under data/muse_pick7/ and marks progress;
a relaunch skips finished stages. Guards (champion/seat/shadow/
gated hashes) are checked before registration. LAB ONLY under
the promotion lock: a win registers pending_approval only.

Usage: .venv/bin/python scripts/build_muse_pick7.py
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

from features import next_try_features, pick2_features, scored_features  # noqa: E402
from models.muse_pick import build_pick_pipeline  # noqa: E402
from models.muse_pick7 import (HORIZON, anatomy, exam_split,  # noqa: E402
                               is_clean, purged_wf_splits)
from models.muse_picks456 import horizon_labels  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB1 = ROOT / "data" / "muse_next_try"
LAB = ROOT / "data" / "muse_pick7"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick7"
LABEL = "Stacker — Slow Clock (24h)"
DESIGN = ("Muse's Pick #7 'Slow Clock': Pick recipe fitted to "
          "next-24h direction on the causal 120-column frame "
          "(98 scored eyes + 22 derivatives-positioning / ETH / "
          "calendar columns), 24-row boundary purge, embargoed "
          "walk-forward — the positioning lever's first test at "
          "its natural horizon after failing at 1h (runs 101/102)")
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
COLS98 = list(scored_features.scored_columns()) + list(
    pick2_features.NEW_COLUMNS)
COLS_NEW = list(next_try_features.NEW_COLUMNS)
COLS120 = COLS98 + COLS_NEW


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


# ------------------------------------------------------------- setup
def stage_setup() -> dict:
    frame = pd.read_pickle(LAB1 / "frame120.pkl")
    raw = pd.read_pickle(LAB1 / "raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    assert frame.index.equals(raw.index), "frame/raw index mismatch"
    # The purge/embargo arithmetic is POSITIONAL (the program's
    # convention: labels are positional shifts, holes included —
    # the raw feed has two documented 6h holes). What must hold
    # is that the supervised grid is one contiguous slice of the
    # label grid, so positional offsets are identical on both:
    raw_pos = raw.index.get_indexer(prod_sup.index)
    assert (raw_pos >= 0).all() and (np.diff(raw_pos) == 1).all(), \
        "supervised grid is not a contiguous slice of the label grid"
    diffs = frame.index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(hours=1)]
    # integrity: the stored 1h target IS horizon_labels(close, 1)
    close_full = raw["close"]
    y1 = horizon_labels(close_full, 1).reindex(prod_sup.index)
    assert (y1.dropna().astype(int).to_numpy()
            == prod_sup["target"].reindex(y1.dropna().index)
            .astype(int).to_numpy()).all(), "stored target != 1h labels"
    y24 = horizon_labels(close_full, HORIZON).reindex(prod_sup.index)
    sup = frame.reindex(prod_sup.index)
    X120 = sup[COLS120]
    finite_rows = np.isfinite(X120.to_numpy(dtype=float)).all(axis=1)
    # Supervised rows = finite on the full frame AND label defined
    # (the Next Try study's own row discipline: its 12,045 train
    # rows are exactly the pre-holdout finite set). Mask the label
    # series so exam_split's finiteness filter enforces both.
    y_masked = y24.to_numpy(dtype=float).copy()
    y_masked[~finite_rows] = np.nan
    y24 = pd.Series(y_masked, index=prod_sup.index)
    split = len(prod_sup) - HOLDOUT
    train_pos, exam_pos = exam_split(prod_sup.index, y24, split,
                                     horizon=HORIZON)
    assert finite_rows[train_pos].all() and finite_rows[exam_pos].all(), \
        "non-finite frame values inside the Pick #7 row set"
    from features.technical_indicators import FEATURE_COLUMNS
    prod_cols = list(FEATURE_COLUMNS)  # the canonical production 21
    assert len(prod_cols) == 21
    assert all(c in prod_sup.columns for c in prod_cols), \
        "prod_sup is missing a production feature column"
    y = y24.to_numpy(dtype=float)
    exam_y = y[exam_pos].astype(int)
    majority = float(max(exam_y.mean(), 1 - exam_y.mean()))
    meta = {
        "guard_before": _guard(),
        "n_supervised": int(len(prod_sup)),
        "holdout_pos": int(split),
        "holdout_start": str(prod_sup.index[split]),
        "n_train": int(len(train_pos)),
        "n_exam": int(len(exam_pos)),
        "exam_majority": majority,
        "exam_up_rate": float(exam_y.mean()),
        "prod_cols": prod_cols,
        "train_first": str(prod_sup.index[train_pos[0]]),
        "train_last": str(prod_sup.index[train_pos[-1]]),
        "exam_first": str(prod_sup.index[exam_pos[0]]),
        "exam_last": str(prod_sup.index[exam_pos[-1]]),
        "index_gaps_over_1h": int(len(gaps)),
        "index_gap_hours_total": float(
            (gaps / pd.Timedelta(hours=1)).sum()) if len(gaps) else 0.0,
    }
    np.savez_compressed(
        LAB / "setup.npz", train_pos=train_pos, exam_pos=exam_pos,
        y=y, prod_cols=np.array(prod_cols))
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: train {len(train_pos)} rows "
         f"({meta['train_first']} -> {meta['train_last']}), exam "
         f"{len(exam_pos)} rows from {meta['exam_first']}, "
         f"majority {majority:.4f}")
    return meta


def _load_setup():
    z = np.load(LAB / "setup.npz", allow_pickle=True)
    meta = json.loads((LAB / "meta.json").read_text())
    frame = pd.read_pickle(LAB1 / "frame120.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    sup = frame.reindex(prod_sup.index)
    return z, meta, sup, prod_sup


def _fit_proba(cols, tag):
    z, meta, sup, prod_sup = _load_setup()
    train_pos, exam_pos, y = z["train_pos"], z["exam_pos"], z["y"]
    if cols == "prod21":
        src = prod_sup[list(z["prod_cols"])]
    else:
        src = sup[cols]
    Xtr = src.iloc[train_pos]
    Xex = src.iloc[exam_pos]
    ytr = y[train_pos].astype(int)
    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xex)[:, 1]
    np.save(LAB / f"proba_{tag}.npy", proba)
    if tag == "pick7":
        joblib.dump(pipe, LAB / "pick7_stacker.pkl")
    m = anatomy(proba, y[exam_pos].astype(int))
    _say(f"{tag}: exam acc {m['acc']:.4f} prec {m['precision']:.4f} "
         f"up {m['up_calls']}/{len(exam_pos)} "
         f"({time.time() - t0:.0f}s)")
    return m


# ------------------------------------------------------------- wf
def stage_wf() -> dict:
    z, meta, sup, prod_sup = _load_setup()
    train_pos, y = z["train_pos"], z["y"]
    X = sup[COLS120].iloc[train_pos].reset_index(drop=True)
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
def stage_register(meta, m_pick, m_c1, m_c2, m_a1, wf) -> dict:
    z, _, _, _ = _load_setup()
    exam_pos, y = z["exam_pos"], z["y"]
    exam_y = y[exam_pos].astype(int)
    proba = np.load(LAB / "proba_pick7.npy")
    # grade #1 recomputed here from the saved probabilities
    m = anatomy(proba, exam_y)
    assert abs(m["acc"] - m_pick["acc"]) < 1e-12
    best_control = max(m_c1["acc"], m_c2["acc"])
    clean = is_clean(m)
    wf_ok = wf["wf_acc"] > wf["wf_majority"]
    pending = bool(m["acc"] > best_control
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
    if beat_bar:
        reason = (f"Muse's Pick #7 holdout(24h) acc {m['acc']:.4f} "
                  f"BEAT THE BAR (clean-era best 0.5354, cross-exam "
                  f"reference) and both same-exam controls "
                  f"(prod21 {m_c1['acc']:.4f}; eyes98 "
                  f"{m_c2['acc']:.4f}) with clean anatomy and "
                  f"walk-forward agreement — WOULD PROMOTE: held "
                  f"as PENDING KASS APPROVAL under the promotion "
                  f"lock; champion kept")
    elif pending:
        reason = (f"Muse's Pick #7 holdout(24h) acc {m['acc']:.4f} "
                  f"beat both same-exam controls (prod21 "
                  f"{m_c1['acc']:.4f}; eyes98 {m_c2['acc']:.4f}) "
                  f"and the exam majority "
                  f"{meta['exam_majority']:.4f} with clean anatomy "
                  f"and walk-forward agreement — WOULD PROMOTE: "
                  f"held as PENDING KASS APPROVAL under the "
                  f"promotion lock; champion kept")
    elif m["acc"] > best_control and not clean:
        reason = (f"Muse's Pick #7 holdout(24h) acc {m['acc']:.4f} "
                  f"beat the controls BUT anatomy is DEGENERATE "
                  f"(up-calls {m['up_calls']}/{len(exam_y)}, "
                  f"precision {m['precision']:.4f}) — counted as a "
                  f"LOSS under the pre-registered run-63 rule")
    elif m["acc"] > best_control and not wf_ok:
        reason = (f"Muse's Pick #7 holdout(24h) acc {m['acc']:.4f} "
                  f"beat the controls (best {best_control:.4f}) "
                  f"with clean anatomy BUT walk-forward disagrees "
                  f"({wf['wf_acc']:.4f} vs majority "
                  f"{wf['wf_majority']:.4f}) — FAIL under the "
                  f"pre-registered bar; champion kept")
    else:
        reason = (f"Muse's Pick #7 holdout(24h) acc {m['acc']:.4f} "
                  f"did not beat both same-exam controls (prod21 "
                  f"{m_c1['acc']:.4f}; eyes98 {m_c2['acc']:.4f}; "
                  f"majority {meta['exam_majority']:.4f}); "
                  f"champion kept (study verdict only)")
    metrics = {
        "challenger_acc": m["acc"], "champion_acc": best_control,
        "challenger_wf_acc": wf["wf_acc"],
        "challenger_wf_precision": wf["wf_precision_pooled"],
        "precision": m["precision"], "folds": 3,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": FAMILY,
        "votes": vote_histogram(proba),
        FAMILY: {
            "label": LABEL, "verdict": "PASS" if pending else "FAIL",
            "design": DESIGN, "beat_bar": beat_bar,
            "control_prod21_acc": m_c1["acc"],
            "control_eyes98_acc": m_c2["acc"],
            "ablation_info22_acc": m_a1["acc"],
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
        f"Muse's Pick #7 (family={FAMILY})",
        f"{DESIGN}. Exam (24h, {len(exam_y)} rows) acc "
        f"{m['acc']:.4f} vs controls prod21 {m_c1['acc']:.4f} / "
        f"eyes98 {m_c2['acc']:.4f} / majority "
        f"{meta['exam_majority']:.4f}; ablation info22-only "
        f"{m_a1['acc']:.4f}; precision {m['precision']:.4f}; "
        f"up-calls {m['up_calls']}/{len(exam_y)}; purged "
        f"walk-forward {wf['wf_acc']:.4f} vs majority "
        f"{wf['wf_majority']:.4f}; non-overlap diagnostic "
        f"{nonov['acc']:.4f} (n={len(exam_y[::HORIZON])}). "
        f"{'BEAT THE BAR — PENDING KASS APPROVAL.' if beat_bar else 'PENDING KASS APPROVAL under the promotion lock.' if pending else 'No win.'} "
        f"Nothing installed.")
    out = {"pick": m, "control_prod21": m_c1,
           "control_eyes98": m_c2, "ablation_info22": m_a1,
           "wf": wf, "pending": pending, "beat_bar": beat_bar,
           "window_accs": window_accs, "nonoverlap": nonov,
           "meta": meta}
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
    results = {}
    for tag, cols in (("control_prod21", "prod21"),
                      ("control_eyes98", COLS98),
                      ("ablation_info22", COLS_NEW),
                      ("pick7", COLS120)):
        stage = f"fit_{tag}"
        if stage not in done:
            results[tag] = _fit_proba(cols, tag)
            _mark(stage)
        else:
            z, _, _, _ = _load_setup()
            proba = np.load(LAB / f"proba_{tag}.npy")
            results[tag] = anatomy(proba, z["y"][z["exam_pos"]]
                                   .astype(int))
            _say(f"{tag}: resumed from saved probabilities")
    if "wf" not in done:
        wf = stage_wf()
        _mark("wf")
    else:
        wf = json.loads((LAB / "wf.json").read_text())
        _say("wf: resumed from checkpoint")
    if "register" not in done:
        stage_register(meta, results["pick7"],
                       results["control_prod21"],
                       results["control_eyes98"],
                       results["ablation_info22"], wf)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
