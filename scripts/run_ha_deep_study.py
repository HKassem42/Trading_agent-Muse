#!/usr/bin/env python
"""HA-Deep — the Heikin-Ashi study (runs 87/88): study runner
(2026-10-30).

Kass called this study himself ("Call the Heikin-Ashi study
now", 2026-10-30 ~16:53 ET) and confirmed the consolidated
spec ("Yes build this and test it", ~16:59 ET). Design frozen
in refs/ha-deep-study-2026-10-30.md BEFORE any grading ran,
including his three amendments (the full HA-native indicator
suite; his confirmed-shift trading rule; the judged form with
the BUY/SHORT/HOLD-only vocabulary). In one paragraph: a
58-feature Heikin-Ashi frame (anatomy/streaks + main-chart
indicators ON the HA candles + sub-chart indicators ON the
HA candles) feeds a four-arm zoo selected on training-region
TimeSeriesSplit(4) OOF balanced accuracy (stacker on
triple-barrier labels / stacker on next-candle labels /
24-candle HA sequence LSTM on triple-barrier labels /
LightGBM on triple-barrier labels); the winner faces the
768-row exam vs run 73 re-scored on its own label definition
plus the standard duel with PPO run 44 — BOTH bars required
for WOULD PROMOTE — with a training-region walk-forward as
the primary corroboration (this holdout's sixth grading
use). Part II grades Kass's HA Shift Trader walk-forward
first (pure L=2 control, judged form registered, pure L=3
sensitivity), holdout duel second.

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded
before/after incl. Book #3's gated files; the paper books
are never opened for writing). Writes live only under
data/ha_deep/, plus the two registry runs and lab notes.

Phases checkpoint under data/ha_deep/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  frames -> oof -> fit_exam -> duel_eyes -> shift -> register

Usage: .venv/bin/python scripts/run_ha_deep_study.py [phase]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import make_supervised  # noqa: E402
from models.muse_pick import build_pick_pipeline  # noqa: E402
from training import eyes_v2 as ev  # noqa: E402
from training import ha_deep as hd  # noqa: E402
from training import meta_label as ml  # noqa: E402
from training import nightly  # noqa: E402
from training.meta_label import LGBM_PARAMS  # noqa: E402
from training.nightly import _agg, evaluate_windows, gate_decision  # noqa: E402

LAB82 = ROOT / "data" / "meta_label"
LAB = ROOT / "data" / "ha_deep"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB82 / "df.pkl"
HOLD_SUP82_PATH = LAB82 / "hold_sup.pkl"
REFS_PATH = ROOT / "refs" / "ha-deep-study-2026-10-30.md"
FRAME_PATH = LAB / "frame58.npy"
SEQ_PATH = LAB / "seq6.npy"
LABELS_PATH = LAB / "labels_tb.npy"
ROWS_PATH = LAB / "rows_train.npy"
HOLD_SUP_PATH = LAB / "hold_sup.pkl"
OOF_PATH = LAB / "oof.npz"
VOTES_NEW_PATH = LAB / "votes_new.npy"
VOTES_73_PATH = LAB / "votes_73.npy"
MODEL_PKL = LAB / "ha_deep_model.pkl"
MODEL_PT = LAB / "ha_sequence_eyes.pt"
JUDGE_PKL = LAB / "judge_model.pkl"
JUDGE_LOG_PATH = LAB / "shift_judge_log.json"
LEDGERS = LAB / "ledgers"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"
HOLDOUT = 792
N_EXAM = 768
# Runs 82–86's recorded champion pair on this byte-identical
# holdout — the two-way pin for both money tests.
CHAMP_AGG_PIN = -210.17
CHAMP_PNL_PIN = -305.28

ARM_NAMES = {
    "a": "HA frame + stacker recipe, triple-barrier labels",
    "b": "HA frame + stacker recipe, next-candle labels",
    "c": "HA sequence LSTM (24 candles), triple-barrier labels",
    "c_fallback": "flattened-lag LightGBM (substituted for c)",
    "d": "HA frame + LightGBM, triple-barrier labels",
}
BLOCK_OF = {}
for _i, _c in enumerate(hd.BLOCK1_COLUMNS):
    BLOCK_OF[_c] = 1
for _i, _c in enumerate(hd.BLOCK2_COLUMNS):
    BLOCK_OF[_c] = 2
for _i, _c in enumerate(hd.BLOCK3_COLUMNS):
    BLOCK_OF[_c] = 3
BLOCK_IDX = {
    "full58": list(range(58)),
    "B1": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
           if BLOCK_OF[c] == 1],
    "B2": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
           if BLOCK_OF[c] == 2],
    "B3": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
           if BLOCK_OF[c] == 3],
    "B1+B2": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
              if BLOCK_OF[c] in (1, 2)],
    "B1+B3": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
              if BLOCK_OF[c] in (1, 3)],
    "B2+B3": [i for i, c in enumerate(hd.FEATURE_COLUMNS_HA)
              if BLOCK_OF[c] in (2, 3)],
}


def results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    LAB.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1))


def _df() -> pd.DataFrame:
    return pd.read_pickle(DF_PATH)


def _pick2_sup(df: pd.DataFrame, sup: pd.DataFrame) -> pd.DataFrame:
    """The pick2_98 frame (champion's 98 columns in recorded
    order) aligned to the supervised rows — the eyes_v2
    construction."""
    from features import pick2_features
    frame98, _cols98 = pick2_features.build_pick2_frame_pair(df)
    champ_cols = list(nightly._champion_meta().get("feature_columns")
                      or [])
    assert len(champ_cols) == 98, len(champ_cols)
    return frame98.reindex(sup.index)[champ_cols]


def _oof_metrics(proba: np.ndarray, mask: np.ndarray,
                 y: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)
    p, t = proba[mask], y[mask]
    return {"balanced_accuracy": float(balanced_accuracy_score(t, p >= 0.5)),
            "auc": float(roc_auc_score(t, p)),
            "accuracy": float(accuracy_score(t, p >= 0.5)),
            "n_oof": int(mask.sum())}


def _segments(rows: np.ndarray, split: int) -> list[np.ndarray]:
    """The walk-forward segmentation: 5 contiguous near-equal
    segments over the common-row span [rows.min(), split),
    segment 0 = burn-in. DEVIATION FROM THE LITERAL PRE-REG
    TEXT, recorded before any walk-forward number existed:
    the pre-reg splits [0, split), but the common rows start
    at position 4176 (the pick2_98 warmup — the same 4,176-row
    head run 82 recorded), so segment 0 of the literal split
    holds zero common rows and fold 1 has no purged fit rows
    at all — ungradeable. Segmenting the span the graded rows
    actually occupy preserves every other frozen element
    (5 segments, burn-in first, folds 1-4, the t + 24 purge,
    identical fit/graded row definitions)."""
    return list(np.array_split(np.arange(int(rows.min()), split), 5))


# ---------------------------------------------------------------- frames
def phase_frames() -> None:
    if results().get("frames"):
        print("frames: checkpoint exists, skipping", flush=True)
        return
    assert REFS_PATH.exists(), "pre-registration missing — design not frozen"
    meta = nightly._champion_meta()
    assert meta.get("run_id") == 73, \
        f"lab stacker champion is not run 73: {meta.get('run_id')}"
    assert meta.get("feature_set") == "pick2_98", meta.get("feature_set")

    df = _df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    hold_sup = sup.iloc[split:]
    hold82 = pd.read_pickle(HOLD_SUP82_PATH)
    assert list(hold_sup.index) == list(hold82.index), \
        "holdout differs from run 82's persisted holdout"
    assert len(hold_sup) == HOLDOUT
    df_pos = df.index.get_indexer(sup.index)
    assert (np.diff(df_pos) == 1).all(), "sup rows not a contiguous tail"

    print("building the 58-feature HA frame ...", flush=True)
    t0 = time.time()
    frame_df, seq_df, aux = hd.build_ha_frame(df)
    assert list(frame_df.columns) == list(hd.FEATURE_COLUMNS_HA)
    frame = frame_df.to_numpy(dtype=float)          # df-aligned
    seq = seq_df.to_numpy(dtype=float)              # df-aligned
    colors = aux["colors"]

    frame98 = _pick2_sup(df, sup)
    pick2_finite = np.isfinite(frame98.to_numpy(dtype=float)).all(axis=1)

    atr_price = (sup["atr_pct_raw"] * sup["close"]).to_numpy(dtype=float)
    labels, lstats = ev.triple_barrier_labels(
        sup["open"].to_numpy(dtype=float),
        sup["high"].to_numpy(dtype=float),
        sup["low"].to_numpy(dtype=float),
        sup["close"].to_numpy(dtype=float), atr_price)
    rows = hd.common_training_rows(frame_df.iloc[df_pos],
                                   pick2_finite, labels,
                                   sup["target"].to_numpy(dtype=float),
                                   split)
    # Exam feasibility: TB labels defined at all 768 exam rows.
    exam_labels = labels[split:split + N_EXAM]
    assert (exam_labels != -1).all(), "TB label undefined on an exam row"
    # Duel feasibility: HA frame finite at all 792 holdout rows and
    # the sequence frame finite over their trailing windows.
    hold_df_pos = df_pos[split:]
    assert np.isfinite(frame[hold_df_pos]).all(), \
        "non-finite HA frame values on holdout rows"
    assert np.isfinite(seq[df_pos[split] - 23:]).all(), \
        "non-finite sequence values in the holdout window region"
    train_lab = labels[rows]
    assert set(np.unique(train_lab)) == {0, 1}

    LAB.mkdir(parents=True, exist_ok=True)
    np.save(FRAME_PATH, frame)
    np.save(SEQ_PATH, seq)
    np.save(LABELS_PATH, labels)
    np.save(ROWS_PATH, rows)
    hold_sup.to_pickle(HOLD_SUP_PATH)
    save_result("frames", {
        "rows_sup": len(sup), "split": split,
        "df_pos_first": int(df_pos[0]),
        "train_rows_common": int(len(rows)),
        "train_base_rate_tb": float(np.mean(train_lab == 1)),
        "train_base_rate_nc": float(np.mean(
            sup["target"].to_numpy(dtype=int)[rows] == 1)),
        "holdout_first": str(hold_sup.index[0]),
        "holdout_last": str(hold_sup.index[-1]),
        "exam_rows": N_EXAM,
        "label_stats_full_series": lstats,
        "label_base_rate_exam_tb": float(np.mean(exam_labels == 1)),
        "label_base_rate_exam_nc": float(np.mean(
            hold_sup["target"].to_numpy(dtype=int)[:N_EXAM] == 1)),
        "confirmed_shifts_l2_full": int((hd.shift_events(colors, 2)
                                         != 0).sum()),
        "confirmed_shifts_l3_full": int((hd.shift_events(colors, 3)
                                         != 0).sum()),
        "guard_before": hd.guard_hashes(),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"frames: {len(rows)} common training rows "
          f"(TB base rate {np.mean(train_lab == 1):.3f}), holdout "
          f"{hold_sup.index[0]} -> {hold_sup.index[-1]}", flush=True)


# ---------------------------------------------------------------- zoo
def _fit_flat_cols(arm: str, X: np.ndarray, y: np.ndarray,
                   cols: list):
    """Flat arms on the HA frame: (a)/(b) the stacker recipe
    (fit on a named DataFrame), (d) LightGBM (run-82 config)."""
    if arm in ("a", "b"):
        model = build_pick_pipeline(42)
        model.fit(pd.DataFrame(X, columns=cols), y)
        return ("stacker", model)
    if arm == "d":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X, y)
        return ("lgbm", model)
    raise ValueError(arm)


def _flat_proba(fitted, X: np.ndarray, cols: list) -> np.ndarray:
    kind, model = fitted
    if kind == "stacker":
        return model.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
    return model.predict_proba(X)[:, 1]


def _lag_matrix(seq: np.ndarray, df_rows: np.ndarray) -> np.ndarray:
    """Flattened-lag fallback inputs: the 6 sequence series at
    lags 0..23, lag-major (24 x 6 = 144 inputs)."""
    out = np.empty((len(df_rows), 24 * seq.shape[1]), dtype=float)
    for k, t in enumerate(df_rows):
        out[k] = seq[t - 23:t + 1][::-1].reshape(-1)
    return out


def phase_oof() -> None:
    if results().get("oof"):
        print("oof: checkpoint exists, skipping", flush=True)
        return
    from sklearn.model_selection import TimeSeriesSplit

    frame = np.load(FRAME_PATH)
    seq = np.load(SEQ_PATH)
    labels = np.load(LABELS_PATH)
    rows = np.load(ROWS_PATH)                       # sup positions
    fr = results()["frames"]
    df_pos0 = fr["df_pos_first"]
    rows_df = df_pos0 + rows                        # df positions
    sup = make_supervised(_df())
    y_tb = labels[rows].astype(int)
    y_nc = sup["target"].to_numpy(dtype=int)[rows]
    X = frame[rows_df]
    cols = list(hd.FEATURE_COLUMNS_HA)
    tss = list(TimeSeriesSplit(n_splits=4).split(X))
    n = len(rows)
    probas: dict[str, np.ndarray] = {}
    table: dict[str, dict] = {}

    for arm, y_fit in (("a", y_tb), ("b", y_nc), ("d", y_tb)):
        t0 = time.time()
        proba = np.full(n, np.nan)
        for tr, te in tss:
            fitted = _fit_flat_cols(arm, X[tr], y_fit[tr], cols)
            proba[te] = _flat_proba(fitted, X[te], cols)
        y_own = y_tb if arm in ("a", "d") else y_nc
        table[arm] = _oof_metrics(proba, np.isfinite(proba), y_own)
        probas[arm] = proba
        print(f"oof arm {arm}: {table[arm]} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Arm (c) — the HA sequence LSTM; (c') fallback on failure.
    fallback = {"fired": False, "reason": None}
    t0 = time.time()
    try:
        proba = np.full(n, np.nan)
        seqs_all = ev.sequence_tensor(seq, rows_df)
        assert seqs_all.shape[1:] == (24, 6), seqs_all.shape
        for tr, te in tss:
            model = ev.train_lstm_eyes(seqs_all[tr], y_tb[tr], seed=42)
            p = model.predict_proba(seqs_all[te])
            if not np.isfinite(p).all():
                raise RuntimeError("non-finite LSTM OOF probabilities")
            proba[te] = p
        table["c"] = _oof_metrics(proba, np.isfinite(proba), y_tb)
        probas["c"] = proba
        print(f"oof arm c (LSTM): {table['c']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        fallback["fired"] = True
        fallback["reason"] = f"{type(e).__name__}: {e}"
        print(f"oof arm c FAILED ({fallback['reason']}) — "
              f"substituting the flattened-lag variant (c')",
              flush=True)
        from lightgbm import LGBMClassifier
        t0 = time.time()
        proba = np.full(n, np.nan)
        lags_all = _lag_matrix(seq, rows_df)
        for tr, te in tss:
            m = LGBMClassifier(**LGBM_PARAMS)
            m.fit(lags_all[tr], y_tb[tr])
            proba[te] = m.predict_proba(lags_all[te])[:, 1]
        table["c"] = _oof_metrics(proba, np.isfinite(proba), y_tb)
        probas["c"] = proba
        print(f"oof arm c' (lag-LGBM): {table['c']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    selected = ev.select_arm(table)
    # Judge sub-selection: better TB flat arm of (a)/(d).
    ja, jd = table["a"], table["d"]
    if abs(ja["balanced_accuracy"] - jd["balanced_accuracy"]) < 1e-12:
        judge_arm = "a" if ja["auc"] >= jd["auc"] else "d"
    else:
        judge_arm = "a" if ja["balanced_accuracy"] > \
            jd["balanced_accuracy"] else "d"

    # Block ablation (diagnostic only): arm (a)'s configuration
    # refit OOF on the seven frozen subsets, TB labels.
    ablation: dict[str, dict] = {}
    for name, idxs in BLOCK_IDX.items():
        t0 = time.time()
        Xs = X[:, idxs]
        sub_cols = [cols[i] for i in idxs]
        proba = np.full(n, np.nan)
        for tr, te in tss:
            fitted = _fit_flat_cols("a", Xs[tr], y_tb[tr], sub_cols)
            proba[te] = _flat_proba(fitted, Xs[te], sub_cols)
        ablation[name] = _oof_metrics(proba, np.isfinite(proba), y_tb)
        print(f"ablation {name}: BA "
              f"{ablation[name]['balanced_accuracy']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Importance shares from arm (a)'s final full-frame fit.
    fitted_a = _fit_flat_cols("a", X, y_tb, cols)
    joblib.dump(fitted_a[1], LAB / "arm_a_full_fit.pkl")
    shares = _importance_shares(fitted_a[1], cols)

    np.savez(OOF_PATH, **{f"proba_{k}": v for k, v in probas.items()})
    save_result("oof", {"table": table, "selected": selected,
                        "fallback": fallback, "judge_arm": judge_arm,
                        "ablation": ablation,
                        "importance_shares": shares,
                        "arm_names": ARM_NAMES})
    print(f"oof: selected arm {selected}; judge arm {judge_arm}; "
          f"importance shares {shares}", flush=True)


def _importance_shares(model, cols: list) -> dict:
    """Mean normalized impurity importance across the stack's
    fitted base learners, summed per feature block."""
    stack = model.named_steps["stack"]
    imps = []
    for est in stack.estimators_:
        v = np.asarray(est.feature_importances_, dtype=float)
        if v.sum() > 0:
            imps.append(v / v.sum())
    mean_imp = np.mean(imps, axis=0)
    shares = {1: 0.0, 2: 0.0, 3: 0.0}
    for name, v in zip(cols, mean_imp):
        shares[BLOCK_OF[name]] += float(v)
    return {"block1": shares[1], "block2": shares[2],
            "block3": shares[3],
            "per_feature": {c: float(v) for c, v in zip(cols, mean_imp)}}


# ---------------------------------------------------------------- exam
def _selected_votes(selected: str, fallback: dict, frame, seq,
                    labels, rows, rows_df, sup, hold_df_pos):
    """Fit the selected arm on ALL common rows; return (votes over
    the 792 holdout positions, fit info)."""
    y_tb = labels[rows].astype(int)
    y_nc = sup["target"].to_numpy(dtype=int)[rows]
    cols = list(hd.FEATURE_COLUMNS_HA)
    X = frame[rows_df]
    Xh = frame[hold_df_pos]
    if selected in ("a", "b", "d"):
        y_fit = y_nc if selected == "b" else y_tb
        fitted = _fit_flat_cols(selected, X, y_fit, cols)
        votes = _flat_proba(fitted, Xh, cols)
        joblib.dump(fitted[1], MODEL_PKL)
        return votes, {"artifact": str(MODEL_PKL)}
    # sequence arm (c)
    if not fallback["fired"]:
        seqs_tr = ev.sequence_tensor(seq, rows_df)
        model = ev.train_lstm_eyes(seqs_tr, y_tb, seed=42)
        votes = model.predict_proba(ev.sequence_tensor(seq, hold_df_pos))
        import torch
        torch.save({"state_dict": model.net.state_dict(),
                    "mean": model.mean_, "std": model.std_,
                    "param_count": model.param_count,
                    "history": model.history}, MODEL_PT)
        return votes, {"artifact": str(MODEL_PT),
                       "param_count": model.param_count,
                       "history": model.history,
                       "epochs_run": model.epochs_run}
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(_lag_matrix(seq, rows_df), y_tb)
    votes = model.predict_proba(_lag_matrix(seq, hold_df_pos))[:, 1]
    joblib.dump(model, MODEL_PKL)
    return votes, {"artifact": str(MODEL_PKL)}


def _wf_corroboration(selected: str, fallback: dict, frame, seq,
                      frame98, labels, rows, sup, split: int) -> list:
    """Part I grading bar 3: per fold, the selected recipe refit
    (purged) vs the incumbent pick recipe on pick2_98, same labels,
    same fit rows, same graded rows."""
    from sklearn.metrics import (accuracy_score,
                                 balanced_accuracy_score)
    cols = list(hd.FEATURE_COLUMNS_HA)
    cols98 = list(frame98.columns)
    df_pos0 = int(results()["frames"]["df_pos_first"])
    y_tb_all = labels.astype(int)
    y_nc_all = sup["target"].to_numpy(dtype=int)
    y_sel = y_nc_all if selected == "b" else y_tb_all
    segments = _segments(rows, split)
    out = []
    for k in range(1, 5):
        seg = segments[k]
        start = int(seg[0])
        fit_rows = rows[(rows < start) & (rows + 24 < start)]
        graded = rows[(rows >= seg[0]) & (rows <= seg[-1])]
        assert len(fit_rows) > 0 and len(graded) > 0
        y_fit = y_sel[fit_rows]
        y_gr = y_sel[graded]
        # selected recipe
        if selected in ("a", "b", "d"):
            fitted = _fit_flat_cols(selected, frame[df_pos0 + fit_rows],
                                    y_fit, cols)
            p_new = _flat_proba(fitted, frame[df_pos0 + graded], cols)
        elif not fallback["fired"]:
            m = ev.train_lstm_eyes(
                ev.sequence_tensor(seq, df_pos0 + fit_rows),
                y_tb_all[fit_rows], seed=42)
            p_new = m.predict_proba(
                ev.sequence_tensor(seq, df_pos0 + graded))
        else:
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(**LGBM_PARAMS)
            m.fit(_lag_matrix(seq, df_pos0 + fit_rows),
                  y_tb_all[fit_rows])
            p_new = m.predict_proba(
                _lag_matrix(seq, df_pos0 + graded))[:, 1]
        # comparator: incumbent recipe on pick2_98
        comp = build_pick_pipeline(42)
        comp.fit(frame98.iloc[fit_rows], y_fit)
        p_cmp = comp.predict_proba(frame98.iloc[graded])[:, 1]
        out.append({
            "fold": k, "n_graded": int(len(graded)),
            "base_rate": float(np.mean(y_gr == 1)),
            "new_acc": float(accuracy_score(y_gr, p_new >= 0.5)),
            "new_balanced_acc": float(balanced_accuracy_score(
                y_gr, p_new >= 0.5)),
            "champ_acc": float(accuracy_score(y_gr, p_cmp >= 0.5)),
            "champ_balanced_acc": float(balanced_accuracy_score(
                y_gr, p_cmp >= 0.5)),
        })
        print(f"wf fold {k}: new acc {out[-1]['new_acc']:.4f} / BA "
              f"{out[-1]['new_balanced_acc']:.4f} vs comparator acc "
              f"{out[-1]['champ_acc']:.4f} / BA "
              f"{out[-1]['champ_balanced_acc']:.4f} "
              f"(n={len(graded)})", flush=True)
    return out


def phase_fit_exam() -> None:
    if results().get("fit_exam"):
        print("fit_exam: checkpoint exists, skipping", flush=True)
        return
    res = results()
    selected = res["oof"]["selected"]
    fallback = res["oof"]["fallback"]
    frame = np.load(FRAME_PATH)
    seq = np.load(SEQ_PATH)
    labels = np.load(LABELS_PATH)
    rows = np.load(ROWS_PATH)
    df = _df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    df_pos0 = res["frames"]["df_pos_first"]
    rows_df = df_pos0 + rows
    hold_df_pos = df_pos0 + np.arange(split, split + HOLDOUT)
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    frame98 = _pick2_sup(df, sup)

    t0 = time.time()
    votes_new, fit_info = _selected_votes(selected, fallback, frame,
                                          seq, labels, rows, rows_df,
                                          sup, hold_df_pos)
    assert len(votes_new) == HOLDOUT and np.isfinite(votes_new).all()

    meta = nightly._champion_meta()
    champ_model = joblib.load(STACKER_CHAMPION)
    champ_frame, _tag = nightly.champion_holdout_frame(df, hold_sup,
                                                       meta)
    votes_73 = champ_model.predict_proba(champ_frame)[:, 1]
    assert len(votes_73) == HOLDOUT and np.isfinite(votes_73).all()

    off = np.arange(N_EXAM)
    y_tb_exam = labels[split:split + N_EXAM].astype(int)
    y_nc_exam = hold_sup["target"].to_numpy(dtype=int)[:N_EXAM]
    if selected == "b":
        y_primary, y_secondary = y_nc_exam, y_tb_exam
        primary_def = "next-candle"
    else:
        y_primary, y_secondary = y_tb_exam, y_nc_exam
        primary_def = "triple-barrier"
    exam_new = ev.exam_metrics(votes_new[off], y_primary)
    exam_73 = ev.exam_metrics(votes_73[off], y_primary)
    sec_new = ev.exam_metrics(votes_new[off], y_secondary)
    sec_73 = ev.exam_metrics(votes_73[off], y_secondary)

    print("walk-forward corroboration (primary corroboration) ...",
          flush=True)
    wf = _wf_corroboration(selected, fallback, frame, seq, frame98,
                           labels, rows, sup, split)

    np.save(VOTES_NEW_PATH, votes_new)
    np.save(VOTES_73_PATH, votes_73)
    save_result("fit_exam", {
        "selected": selected, "fit_info": fit_info,
        "primary_label_def": primary_def,
        "exam_new": exam_new, "exam_73": exam_73,
        "secondary_new": sec_new, "secondary_73": sec_73,
        "spread_new_792": ev.vote_spread(votes_new),
        "spread_73_792": ev.vote_spread(votes_73),
        "walk_forward": wf,
        "wf_segments": [[int(s[0]), int(s[-1])]
                        for s in _segments(rows, split)],
        "exam_win": bool(exam_new["accuracy"] > exam_73["accuracy"]),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"fit_exam ({primary_def} primary): new acc "
          f"{exam_new['accuracy']:.4f} (prec {exam_new['precision']:.4f}, "
          f"up {exam_new['up_calls']}) vs run73 {exam_73['accuracy']:.4f} "
          f"(prec {exam_73['precision']:.4f}, up {exam_73['up_calls']}); "
          f"secondary: new {sec_new['accuracy']:.4f} vs 73 "
          f"{sec_73['accuracy']:.4f}", flush=True)


def phase_duel_eyes() -> None:
    if results().get("duel_eyes"):
        print("duel_eyes: checkpoint exists, skipping", flush=True)
        return
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    votes_new = np.load(VOTES_NEW_PATH)
    votes_73 = np.load(VOTES_73_PATH)

    print("grading champion arm (44+73, production evaluator) ...",
          flush=True)
    champ_res = evaluate_windows(PPO_CHAMPION, windows,
                                 [votes_73[k * 264:(k + 1) * 264]
                                  for k in range(3)])
    assert champ_res is not None, "champion pair failed to run"
    champ = _agg(champ_res)
    if abs(champ["score"] - CHAMP_AGG_PIN) > 0.011 or \
            abs(champ["total_pnl"] - CHAMP_PNL_PIN) > 0.011:
        raise RuntimeError(
            f"TWO-WAY PIN FAIL: fresh champion arm agg "
            f"{champ['score']:.2f} / P&L {champ['total_pnl']:.2f} != "
            f"recorded {CHAMP_AGG_PIN} / {CHAMP_PNL_PIN} — study "
            f"aborts, nothing registers")
    print(f"  champion pin OK: agg {champ['score']:.2f}, "
          f"P&L {champ['total_pnl']:.2f}", flush=True)

    print("grading challenger arm (44 + HA-Deep eyes) ...", flush=True)
    chall_res = evaluate_windows(PPO_CHAMPION, windows,
                                 [votes_new[k * 264:(k + 1) * 264]
                                  for k in range(3)])
    assert chall_res is not None, "challenger pair failed to run"
    chall = _agg(chall_res)
    save_result("duel_eyes", {
        "champion": champ, "challenger": chall,
        "money_win": bool(chall["score"] > champ["score"]),
        "champ_pin": {"agg": CHAMP_AGG_PIN, "pnl": CHAMP_PNL_PIN,
                      "reproduced": True},
    })
    print(f"duel_eyes: challenger agg {chall['score']:.2f} / P&L "
          f"{chall['total_pnl']:+.2f} vs champion {champ['score']:.2f} / "
          f"{champ['total_pnl']:+.2f} — money win "
          f"{chall['score'] > champ['score']}", flush=True)


# ---------------------------------------------------------------- shift
def _span_arrays(df: pd.DataFrame, sup: pd.DataFrame,
                 d_lo: int, d_hi: int) -> dict:
    """df-aligned candle arrays for the inclusive df range
    [d_lo, d_hi] plus the sup-aligned atr_pct for the same rows.
    sup rows are the contiguous df tail starting at df_pos_first,
    so sup position = df position - df_pos_first."""
    fr = results()["frames"]
    off = fr["df_pos_first"]
    sub = df.iloc[d_lo:d_hi + 1]
    sup_rows = sup.iloc[d_lo - off:d_hi - off + 1]
    assert list(sub.index) == list(sup_rows.index)
    return {"open": sub["open"].to_numpy(dtype=float),
            "high": sub["high"].to_numpy(dtype=float),
            "low": sub["low"].to_numpy(dtype=float),
            "close": sub["close"].to_numpy(dtype=float),
            "atr_pct": sup_rows["atr_pct_raw"].to_numpy(dtype=float),
            "stamps": sub.index}


def _run_form(tag: str, arrays: dict, events_span: np.ndarray,
              lot_span: np.ndarray, judged: bool,
              final_close: float) -> dict:
    """Run one span of one form; persist its ledger; two-way
    re-sum check against the runner's P&L (abort on mismatch)."""
    LEDGERS.mkdir(parents=True, exist_ok=True)
    db = LEDGERS / f"{tag}.db"
    rec = hd.run_shift_span(arrays["open"], arrays["high"],
                            arrays["low"], arrays["close"],
                            arrays["atr_pct"], events_span, lot_span,
                            arrays["stamps"], db, judged)
    resumed = ml.resum_episode_pnl(db, final_close)
    if abs(resumed - rec["pnl"]) >= 0.01:
        raise RuntimeError(
            f"TWO-WAY RE-SUM FAIL on {tag}: ledger re-sum "
            f"{resumed:.2f} != runner P&L {rec['pnl']:.2f} — study "
            f"aborts")
    rec["resum_pnl"] = resumed
    rec["resum_ok"] = True
    rec["fills"] = [{k: f[k] for k in ("ts", "side", "price", "fee",
                                       "reason")}
                    for f in rec["fills"]]
    return rec


def _form_duel_record(window_recs: list, champ_windows: list) -> dict:
    """The _agg-shaped duel record for one Shift Trader form."""
    wins = sum(1 for r, cw in zip(window_recs, champ_windows)
               if r["score"] > cw["score"])
    last = window_recs[-1]["action_counts"]
    tot = sum(last.values())
    mix = {k: (v / tot if tot else 0.0) for k, v in last.items()}
    return {
        "score": float(np.mean([r["score"] for r in window_recs])),
        "total_pnl": float(sum(r["pnl"] for r in window_recs)),
        "max_drawdown": float(max(r["max_drawdown"]
                                  for r in window_recs)),
        "bench_pnl": float(sum(r["bench"] for r in window_recs)),
        "windows": [{"score": r["score"], "pnl": r["pnl"],
                     "bench": r["bench"], "dd": r["max_drawdown"],
                     "trades": r["trades"], "fees": r["fees"],
                     "time_in_position": r["time_in_position"]}
                    for r in window_recs],
        "action_mix": mix,
        "windows_won": int(wins),
        "trades": int(sum(r["trades"] for r in window_recs)),
        "fees": float(sum(r["fees"] for r in window_recs)),
    }


def phase_shift() -> None:
    if results().get("shift"):
        print("shift: checkpoint exists, skipping", flush=True)
        return
    res = results()
    judge_arm = res["oof"]["judge_arm"]
    frame = np.load(FRAME_PATH)
    labels = np.load(LABELS_PATH)
    rows = np.load(ROWS_PATH)
    df = _df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    df_pos0 = res["frames"]["df_pos_first"]
    cols = list(hd.FEATURE_COLUMNS_HA)
    y_tb_rows = labels[rows].astype(int)

    ha = hd.ha_candles(df)
    colors = hd.ha_colors(ha)
    events2 = hd.shift_events(colors, 2)
    events3 = hd.shift_events(colors, 3)
    df_close = df["close"].to_numpy(dtype=float)
    segments = _segments(rows, split)
    judge_log_all: list[dict] = []

    def _fit_judge(fit_rows_sup: np.ndarray):
        X = frame[df_pos0 + fit_rows_sup]
        y = labels[fit_rows_sup].astype(int)
        return _fit_flat_cols(judge_arm, X, y, cols)

    def _lot_for(fitted, span_lo: int, span_hi: int,
                 events_full: np.ndarray) -> np.ndarray:
        lot = np.full(span_hi - span_lo + 1, np.nan)
        idx = np.where(events_full[span_lo:span_hi + 1] != 0)[0]
        if len(idx):
            proba = _flat_proba(fitted, frame[span_lo + idx], cols)
            for j, p in zip(idx, proba):
                lot[j] = hd.judge_lot(float(p),
                                      int(events_full[span_lo + j]))
        return lot

    forms = ("pure", "judged", "pure_l3")
    wf: dict[str, dict] = {f: {"folds": []} for f in forms}
    duel_spans: dict[str, list] = {f: [] for f in forms}

    for k in range(1, 5):
        seg = segments[k]
        d_lo = df_pos0 + int(seg[0])
        d_hi = df_pos0 + int(seg[-1])
        arrays = _span_arrays(df, sup, d_lo, d_hi)
        bench = 100_000.0 * (df_close[d_hi] / df_close[d_lo - 1] - 1.0)
        fit_rows = rows[(rows < seg[0]) & (rows + 24 < seg[0])]
        fitted_j = _fit_judge(fit_rows)
        specs = (("pure", events2, False),
                 ("judged", events2, True),
                 ("pure_l3", events3, False))
        for form, ev_full, judged in specs:
            lot = (_lot_for(fitted_j, d_lo, d_hi, ev_full) if judged
                   else np.full(d_hi - d_lo + 1, np.nan))
            rec = _run_form(f"wf_f{k}_{form}", arrays,
                            ev_full[d_lo:d_hi + 1], lot, judged,
                            float(df_close[d_hi]))
            rec["bench"] = float(bench)
            rec["fold"] = k
            for entry in rec["judge_log"]:
                entry["span"] = f"wf_fold_{k}"
                judge_log_all.append(entry)
            rec["judge_log"] = []
            wf[form]["folds"].append(rec)
        print(f"shift wf fold {k}: " + "; ".join(
            f"{f} P&L {wf[f]['folds'][-1]['pnl']:+,.2f} "
            f"({wf[f]['folds'][-1]['trades']} fills)"
            for f in forms), flush=True)

    for form in forms:
        fs = wf[form]["folds"]
        wf[form]["total"] = {
            "pnl": float(sum(r["pnl"] for r in fs)),
            "worst_drawdown": float(max(r["max_drawdown"] for r in fs)),
            "score": float(sum(r["pnl"] for r in fs)
                           - 0.5 * max(r["max_drawdown"] for r in fs)),
            "trades": int(sum(r["trades"] for r in fs)),
            "fees": float(sum(r["fees"] for r in fs)),
            "bench_pnl": float(sum(r["bench"] for r in fs)),
        }

    # Duel fit: judge on ALL common training rows.
    fitted_duel = _fit_judge(rows)
    joblib.dump(fitted_duel[1], JUDGE_PKL)
    champ_windows = res["duel_eyes"]["champion"]["windows"]
    for k in range(3):
        w_lo = split + k * 264
        w_hi = split + (k + 1) * 264 - 1
        d_lo, d_hi = df_pos0 + w_lo, df_pos0 + w_hi
        arrays = _span_arrays(df, sup, d_lo, d_hi)
        bench = 100_000.0 * (df_close[d_hi] / df_close[d_lo - 1] - 1.0)
        for form, ev_full, judged in (("pure", events2, False),
                                      ("judged", events2, True),
                                      ("pure_l3", events3, False)):
            lot = (_lot_for(fitted_duel, d_lo, d_hi, ev_full) if judged
                   else np.full(d_hi - d_lo + 1, np.nan))
            rec = _run_form(f"duel_w{k + 1}_{form}", arrays,
                            ev_full[d_lo:d_hi + 1], lot, judged,
                            float(df_close[d_hi]))
            rec["bench"] = float(bench)
            for entry in rec["judge_log"]:
                entry["span"] = f"duel_window_{k + 1}"
                judge_log_all.append(entry)
            rec["judge_log"] = []
            duel_spans[form].append(rec)
        print(f"shift duel window {k + 1}: " + "; ".join(
            f"{f} P&L {duel_spans[f][-1]['pnl']:+,.2f}"
            for f in forms), flush=True)

    duel_records = {f: _form_duel_record(duel_spans[f], champ_windows)
                    for f in forms}
    # Verdicts: formula + gate floor per form (pending only from
    # the judged form, applied at registration).
    champ = res["duel_eyes"]["champion"]
    verdicts = {}
    for form in forms:
        rec = duel_records[form]
        overall_counts = {"hold": 0, "buy": 0, "short": 0}
        for r in duel_spans[form]:
            for kk, vv in r["action_counts"].items():
                overall_counts[kk] += vv
        tot = sum(overall_counts.values())
        overall_mix = {kk: vv / tot for kk, vv in overall_counts.items()}
        formula = bool(rec["score"] > champ["score"]
                       and rec["windows_won"] >= 2)
        promotable, floor_fail = gate_decision(
            {"total_pnl": rec["total_pnl"],
             "bench_pnl": rec["bench_pnl"],
             "action_mix": overall_mix}, formula)
        verdicts[form] = {"formula_pass": formula,
                          "floor_failure": floor_fail,
                          "promotable": bool(promotable),
                          "overall_action_mix": overall_mix}
    taken = [e for e in judge_log_all if e["verdict"] == "taken"]
    skipped = [e for e in judge_log_all if e["verdict"] == "skipped"]
    JUDGE_LOG_PATH.write_text(json.dumps(judge_log_all, indent=1))
    save_result("shift", {
        "judge_arm": judge_arm,
        "judge_oof": res["oof"]["table"][judge_arm],
        "wf_segments": [[int(s[0]), int(s[-1])] for s in segments],
        "walk_forward": wf,
        "duel_records": duel_records,
        "verdicts": verdicts,
        "judge_log_summary": {
            "shifts_judged": len(judge_log_all),
            "taken": len(taken), "skipped": len(skipped),
            "noop": len(judge_log_all) - len(taken) - len(skipped),
        },
        "resum_all_ok": True,
    })
    print(f"shift: judged WF total P&L "
          f"{wf['judged']['total']['pnl']:+,.2f} vs pure "
          f"{wf['pure']['total']['pnl']:+,.2f} vs pure L3 "
          f"{wf['pure_l3']['total']['pnl']:+,.2f}; duel judged agg "
          f"{duel_records['judged']['score']:.2f} vs champion "
          f"{champ['score']:.2f}; verdicts {verdicts}; judge log: "
          f"{len(judge_log_all)} shifts, {len(taken)} taken, "
          f"{len(skipped)} skipped", flush=True)


# ---------------------------------------------------------------- register
def phase_register() -> None:
    res = results()
    if res.get("registered_eyes") and res.get("registered_trader"):
        print("register: already done, skipping", flush=True)
        return
    oof, fe, duel = res["oof"], res["fit_exam"], res["duel_eyes"]
    shift = res["shift"]
    selected = oof["selected"]
    fallback = oof["fallback"]
    arm_label = ARM_NAMES["c_fallback" if selected == "c"
                          and fallback["fired"] else selected]

    # ---- run 87: the eyes
    if not res.get("registered_eyes"):
        exam_win = bool(fe["exam_win"])
        money_win = bool(duel["money_win"])
        both = exam_win and money_win
        en, e7 = fe["exam_new"], fe["exam_73"]
        chall, champ = duel["challenger"], duel["champion"]
        rel = (f"exam acc {en['accuracy']:.4f} vs run 73 re-scored "
               f"{e7['accuracy']:.4f} (prec {en['precision']:.4f} vs "
               f"{e7['precision']:.4f}, up-calls {en['up_calls']} vs "
               f"{e7['up_calls']}); duel agg {chall['score']:.0f} vs "
               f"champion {champ['score']:.0f}, P&L "
               f"{chall['total_pnl']:+,.0f} vs "
               f"{champ['total_pnl']:+,.0f}")
        if both:
            verdict = "WOULD PROMOTE — PENDING KASS"
            reason = (f"HA-Deep Eyes ({arm_label}): {rel} — WOULD "
                      f"PROMOTE: held as PENDING KASS APPROVAL under "
                      f"the promotion lock; champion kept, nothing "
                      f"installed")
        elif exam_win:
            verdict = "FAIL"
            reason = (f"HA-Deep Eyes ({arm_label}): exam win only — "
                      f"{rel}; the money test did not follow, so no "
                      f"pending approval (the eyes-duel lesson); "
                      f"champion kept")
        elif money_win:
            verdict = "FAIL"
            reason = (f"HA-Deep Eyes ({arm_label}): money win only "
                      f"— {rel}; both bars are required for a "
                      f"pending record; champion kept")
        else:
            verdict = "FAIL"
            reason = (f"HA-Deep Eyes ({arm_label}): champion kept — "
                      f"{rel} — failed eyes attempt (component "
                      f"study, not a numbered null)")
        wf = fe["walk_forward"]
        metrics = {
            "family": hd.FAMILY,
            "challenger_acc": en["accuracy"],
            "champion_acc": e7["accuracy"],
            "precision": en["precision"],
            "champion_precision": e7["precision"],
            "challenger_wf_acc": oof["table"][selected]["accuracy"],
            "folds": 4,
            "deep": False,
            "pending_approval": bool(both),
            "reason": reason,
            "ha_deep": {
                "label": hd.LABEL,
                "verdict": verdict,
                "design": "58-feature Heikin-Ashi frame (anatomy/"
                          "streaks 22 + main-chart on HA 21 + "
                          "sub-chart on HA 15), 4-arm zoo, OOF "
                          "selection, triple-barrier/next-candle "
                          "labels",
                "features": [{"name": c, "block": BLOCK_OF[c]}
                             for c in hd.FEATURE_COLUMNS_HA],
                "selected_arm": selected,
                "arm_name": arm_label,
                "fallback": fallback,
                "zoo_oof": oof["table"],
                "judge_arm": oof["judge_arm"],
                "ablation": oof["ablation"],
                "importance_shares": oof["importance_shares"],
                "train_rows": res["frames"]["train_rows_common"],
                "train_base_rate_tb": res["frames"]
                ["train_base_rate_tb"],
                "primary_label_def": fe["primary_label_def"],
                "exam": {"new": en, "champ73": e7,
                         "secondary_new": fe["secondary_new"],
                         "secondary_73": fe["secondary_73"],
                         "exam_rows": N_EXAM},
                "spread": {"new_792": fe["spread_new_792"],
                           "champ73_792": fe["spread_73_792"]},
                "walk_forward": wf,
                "duel": {"challenger": {k: chall[k] for k in
                                        ("score", "total_pnl",
                                         "max_drawdown", "bench_pnl",
                                         "action_mix")},
                         "champion": {k: champ[k] for k in
                                      ("score", "total_pnl",
                                       "max_drawdown", "bench_pnl",
                                       "action_mix")}},
                "exam_win": exam_win, "money_win": money_win,
            },
        }
        nightly.log_run("stacker", metrics, False)
        import sqlite3
        with sqlite3.connect(nightly.registry_db()) as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        nightly.add_lab_note(
            "HA-Deep Eyes — PENDING KASS APPROVAL" if both
            else "HA-Deep Eyes", reason)
        save_result("registered_eyes", {"run_id": int(run_id),
                                        "verdict": verdict,
                                        "reason": reason})
        print(f"registered eyes run {run_id}: {verdict}", flush=True)

    # ---- run 88: Kass's Shift Trader (judged form)
    if not res.get("registered_trader"):
        jv = shift["verdicts"]["judged"]
        judged_rec = shift["duel_records"]["judged"]
        champ = duel["champion"]
        wf_j = shift["walk_forward"]["judged"]["total"]
        pending = bool(jv["promotable"])
        if pending:
            verdict = "WOULD PROMOTE — PENDING KASS"
            reason = (f"HA Shift Trader (judged form): walk-forward "
                      f"P&L {wf_j['pnl']:+,.0f} across 4 folds; duel "
                      f"agg {judged_rec['score']:.0f} vs champion "
                      f"{champ['score']:.0f}, P&L "
                      f"{judged_rec['total_pnl']:+,.0f} vs "
                      f"{champ['total_pnl']:+,.0f} — WOULD PROMOTE: "
                      f"held as PENDING KASS APPROVAL under the "
                      f"promotion lock; nothing installed")
        else:
            verdict = "FAIL"
            fail_half = ("formula (agg/windows)" if not
                         jv["formula_pass"] else
                         (jv["floor_failure"] or "floor"))
            reason = (f"HA Shift Trader (judged form): walk-forward "
                      f"P&L {wf_j['pnl']:+,.0f} across 4 folds; duel "
                      f"agg {judged_rec['score']:.0f} vs champion "
                      f"{champ['score']:.0f}, P&L "
                      f"{judged_rec['total_pnl']:+,.0f} vs "
                      f"{champ['total_pnl']:+,.0f} — FAIL on "
                      f"{fail_half}; champion kept (component "
                      f"study, not a numbered null)")
        metrics = {
            "family": hd.SHIFT_FAMILY,
            "challenger": judged_rec,
            "champion": {k: champ[k] for k in
                         ("score", "total_pnl", "max_drawdown",
                          "bench_pnl", "action_mix", "windows")},
            "reason": reason,
            "deep": False,
            "timesteps": 0,
            "pending_approval": pending,
            "ha_shift": {
                "label": hd.SHIFT_LABEL,
                "verdict": verdict,
                "design": "confirmed HA color shift (run length 2) "
                          "+ judge at 0.55 on the full HA frame; "
                          "BUY/SHORT/HOLD only; production broker "
                          "chassis (ATR stop/trailing, 25% tranche, "
                          "75% cap)",
                "judge": {"arm": shift["judge_arm"],
                          "oof": shift["judge_oof"],
                          "threshold": hd.JUDGE_THRESHOLD},
                "walk_forward": shift["walk_forward"],
                "duel_records": shift["duel_records"],
                "verdicts": shift["verdicts"],
                "judge_log_summary": shift["judge_log_summary"],
                "two_way_resum_ok": shift["resum_all_ok"],
            },
        }
        nightly.log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(nightly.registry_db()) as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        nightly.add_lab_note(
            "HA Shift Trader — PENDING KASS APPROVAL" if pending
            else "HA Shift Trader", reason)
        save_result("registered_trader", {"run_id": int(run_id),
                                          "verdict": verdict,
                                          "reason": reason})
        print(f"registered trader run {run_id}: {verdict}", flush=True)

    guard_after = hd.guard_hashes()
    before = res["frames"]["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue  # the books' own loops write these; info only
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("guards_ok", True)
    print("guards: all strict hashes identical before/after",
          flush=True)


PHASES = {"frames": phase_frames, "oof": phase_oof,
          "fit_exam": phase_fit_exam, "duel_eyes": phase_duel_eyes,
          "shift": phase_shift, "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== ha-deep study runner done ===", flush=True)


if __name__ == "__main__":
    main()
