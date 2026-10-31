#!/usr/bin/env python
"""Eyes v2 — the tradeable-move stacker (run 85): study runner
(2026-10-30).

Kass's order: "What can we change to make the eyes better —
do whatever and change whatever to get us results." Design
frozen in refs/eyes-v2-study-2026-10-30.md BEFORE any of this
code ran (arm (d), the 24-candle sequence eyes, is Kass's own
amendment, folded in before the freeze). In one paragraph:
the eyes are retrained on triple-barrier "tradeable move"
labels (1.0 x ATR(14), 24-candle horizon) over pick2_98 + 8
causal context features; a four-arm zoo (pick recipe / run-82
LightGBM / old-label ablation / LSTM sequence eyes, with a
flattened-lag fallback) is selected on training-region
TimeSeriesSplit(4) OOF balanced accuracy against the
triple-barrier labels ONLY; the winner faces two frozen bars
— the 768-row triple-barrier exam vs run 73 re-scored, and
the standard duel paired with PPO run 44 vs the 44+73
champion pair (whose fresh grade must first reproduce runs
82–84's recorded −210.17 / −$305.28 to the cent or the study
aborts).

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded
before/after, incl. Book #3's gated files; the paper books
are never opened). Writes live only under data/eyes_v2/,
plus the registry run and the lab note at the end. Both bars
pass -> WOULD PROMOTE — PENDING KASS APPROVAL
(pending_approval=1); exam-only -> recorded exam win, no
pending; anything less is a failed eyes attempt (component
study, not numbered among the program nulls).

Phases checkpoint under data/eyes_v2/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  frames -> oof -> fit_exam -> duel -> register

Usage: .venv/bin/python scripts/run_eyes_v2_study.py [phase]
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
from training import nightly  # noqa: E402
from training.meta_label import LGBM_PARAMS  # noqa: E402
from training.nightly import _agg, evaluate_windows  # noqa: E402

LAB82 = ROOT / "data" / "meta_label"
LAB = ROOT / "data" / "eyes_v2"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB82 / "df.pkl"
HOLD_SUP82_PATH = LAB82 / "hold_sup.pkl"
REFS_PATH = ROOT / "refs" / "eyes-v2-study-2026-10-30.md"
FRAME_PATH = LAB / "frame_all.npy"
COLS_PATH = LAB / "columns.json"
LABELS_PATH = LAB / "labels.npy"
ROWS_PATH = LAB / "rows_train.npy"
HOLD_SUP_PATH = LAB / "hold_sup.pkl"
OOF_PATH = LAB / "oof.npz"
VOTES_NEW_PATH = LAB / "votes_new.npy"
VOTES_73_PATH = LAB / "votes_73.npy"
MODEL_PKL = LAB / "eyes_v2_model.pkl"
MODEL_PT = LAB / "eyes_v2_sequence.pt"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"
HOLDOUT = 792
N_EXAM = 768
# Runs 82–84's recorded champion pair on this byte-identical
# holdout — the two-way pin for the money test.
CHAMP_AGG_PIN = -210.17
CHAMP_PNL_PIN = -305.28

ARM_NAMES = {
    "a": "pick recipe (run 73's) on triple-barrier labels",
    "b": "LightGBM (run 82 config) on triple-barrier labels",
    "c": "pick recipe on OLD next-candle labels, new features",
    "d": "LSTM sequence eyes (Kass's 24-candle lookback)",
    "d_fallback": "flattened-lag LightGBM (substituted for d)",
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


# ---------------------------------------------------------------- phases
def phase_frames() -> None:
    if results().get("frames"):
        print("frames: checkpoint exists, skipping", flush=True)
        return
    assert REFS_PATH.exists(), "pre-registration missing — design not frozen"
    meta = nightly._champion_meta()
    assert meta.get("run_id") == 73, f"lab stacker champion is not run 73: {meta.get('run_id')}"
    assert meta.get("feature_set") == "pick2_98", meta.get("feature_set")
    champ_cols = list(meta.get("feature_columns") or [])
    assert len(champ_cols) == 98, len(champ_cols)

    df = _df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    hold_sup = sup.iloc[split:]
    hold82 = pd.read_pickle(HOLD_SUP82_PATH)
    assert list(hold_sup.index) == list(hold82.index), \
        "holdout differs from run 82's persisted holdout"
    assert len(hold_sup) == HOLDOUT

    print("building pick2 frame + context (106 features) ...", flush=True)
    t0 = time.time()
    frame, columns, evidence = ev.build_feature_frame(df, sup, champ_cols)
    assert frame.shape[1] == 106 and len(columns) == 106
    finite_pre = int(np.isfinite(frame[:split]).all(axis=1).sum())
    assert finite_pre == 12079, \
        f"pick2-finite training rows {finite_pre} != run 82's 12,079"

    atr_price = (sup["atr_pct_raw"] * sup["close"]).to_numpy(dtype=float)
    labels, lstats = ev.triple_barrier_labels(
        sup["open"].to_numpy(dtype=float), sup["high"].to_numpy(dtype=float),
        sup["low"].to_numpy(dtype=float), sup["close"].to_numpy(dtype=float),
        atr_price)
    rows = ev.common_training_rows(frame, labels, split)
    # Duel/exam feasibility: the frame must be finite from 23 rows
    # before the holdout to the end (sequence windows for holdout
    # rows reach back into the training region's tail).
    assert np.isfinite(frame[split - ev.SEQ_LEN + 1:]).all(), \
        "non-finite frame values in the holdout sequence region"
    train_lab = labels[rows]
    assert set(np.unique(train_lab)) == {0, 1}
    exam_pos = ev.exam_positions(labels, split, N_EXAM)

    LAB.mkdir(parents=True, exist_ok=True)
    np.save(FRAME_PATH, frame)
    COLS_PATH.write_text(json.dumps(columns))
    np.save(LABELS_PATH, labels)
    np.save(ROWS_PATH, rows)
    hold_sup.to_pickle(HOLD_SUP_PATH)
    save_result("frames", {
        "rows_sup": len(sup), "split": split,
        "train_rows_common": int(len(rows)),
        "train_base_rate": float(np.mean(train_lab == 1)),
        "holdout_first": str(hold_sup.index[0]),
        "holdout_last": str(hold_sup.index[-1]),
        "exam_rows": int(len(exam_pos)),
        "label_stats_full_series": lstats,
        "label_base_rate_exam": float(np.mean(labels[exam_pos] == 1)),
        "feature_evidence": evidence,
        "guard_before": ev.guard_hashes(),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"frames: {len(rows)} common training rows "
          f"(base rate {np.mean(train_lab == 1):.3f}), holdout "
          f"{hold_sup.index[0]} -> {hold_sup.index[-1]}, "
          f"exam rows {len(exam_pos)}, labels {lstats}", flush=True)


def _oof_metrics(proba: np.ndarray, mask: np.ndarray,
                 y: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)
    p, t = proba[mask], y[mask]
    return {"balanced_accuracy": float(balanced_accuracy_score(t, p >= 0.5)),
            "auc": float(roc_auc_score(t, p)),
            "accuracy": float(accuracy_score(t, p >= 0.5)),
            "n_oof": int(mask.sum())}


def _fit_flat(arm: str, X: pd.DataFrame, y: np.ndarray):
    if arm in ("a", "c"):
        model = build_pick_pipeline(42)
        model.fit(X, y)
        return model
    if arm == "b":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X.to_numpy(dtype=float), y)
        return model
    raise ValueError(arm)


def _flat_proba(arm: str, model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(
        X if arm in ("a", "c") else X.to_numpy(dtype=float))[:, 1]


def phase_oof() -> None:
    if results().get("oof"):
        print("oof: checkpoint exists, skipping", flush=True)
        return
    from sklearn.model_selection import TimeSeriesSplit

    frame = np.load(FRAME_PATH)
    labels = np.load(LABELS_PATH)
    rows = np.load(ROWS_PATH)
    columns = json.loads(COLS_PATH.read_text())
    sup = make_supervised(_df())
    y_tb = labels[rows].astype(int)
    y_old = sup["target"].to_numpy(dtype=int)[rows]
    X = pd.DataFrame(frame[rows], columns=columns)
    tss = list(TimeSeriesSplit(n_splits=4).split(X))
    n = len(rows)
    probas: dict[str, np.ndarray] = {}
    table: dict[str, dict] = {}

    for arm, y_fit in (("a", y_tb), ("c", y_old), ("b", y_tb)):
        t0 = time.time()
        proba = np.full(n, np.nan)
        for tr, te in tss:
            model = _fit_flat(arm, X.iloc[tr], y_fit[tr])
            proba[te] = _flat_proba(arm, model, X.iloc[te])
        mask = np.isfinite(proba)
        table[arm] = _oof_metrics(proba, mask, y_tb)
        probas[arm] = proba
        print(f"oof arm {arm}: {table[arm]} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Arm (d) — Kass's sequence eyes; (d') fallback on failure.
    fallback = {"fired": False, "reason": None, "subset_cols": None}
    t0 = time.time()
    try:
        proba = np.full(n, np.nan)
        seqs_all = ev.sequence_tensor(frame, rows)
        for tr, te in tss:
            model = ev.train_lstm_eyes(seqs_all[tr], y_tb[tr], seed=42)
            p = model.predict_proba(seqs_all[te])
            if not np.isfinite(p).all():
                raise RuntimeError("non-finite LSTM OOF probabilities")
            proba[te] = p
        mask = np.isfinite(proba)
        table["d"] = _oof_metrics(proba, mask, y_tb)
        probas["d"] = proba
        print(f"oof arm d (LSTM): {table['d']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        fallback["fired"] = True
        fallback["reason"] = f"{type(e).__name__}: {e}"
        print(f"oof arm d FAILED ({fallback['reason']}) — "
              f"substituting the flattened-lag variant (d')", flush=True)
        from lightgbm import LGBMClassifier
        t0 = time.time()
        full_a = _fit_flat("a", X, y_tb)
        imp = ev.pick2_importance_ranking(full_a)
        top24 = np.argsort(-imp[:ev.N_PICK2])[:24].tolist()
        subset = list(range(ev.N_PICK2, ev.N_PICK2 + 8)) + \
            sorted(int(i) for i in top24)
        fallback["subset_cols"] = subset
        proba = np.full(n, np.nan)
        for tr, te in tss:
            m = LGBMClassifier(**LGBM_PARAMS)
            m.fit(ev.lag_matrix(frame, rows[tr], subset), y_tb[tr])
            proba[te] = m.predict_proba(
                ev.lag_matrix(frame, rows[te], subset))[:, 1]
        mask = np.isfinite(proba)
        table["d"] = _oof_metrics(proba, mask, y_tb)
        probas["d"] = proba
        print(f"oof arm d' (lag-LGBM): {table['d']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    selected = ev.select_arm(table)
    np.savez(OOF_PATH, **{f"proba_{k}": v for k, v in probas.items()})
    save_result("oof", {"table": table, "selected": selected,
                        "fallback": fallback,
                        "arm_names": ARM_NAMES})
    print(f"oof: selected arm {selected} "
          f"({ARM_NAMES['d_fallback' if selected == 'd' and fallback['fired'] else selected]})",
          flush=True)


def _selected_votes(frame, labels, rows, columns, sup,
                    selected: str, fallback: dict):
    """Fit the selected arm on ALL common training rows; return
    (votes over the 792 holdout positions, fit info)."""
    split = len(sup) - HOLDOUT
    hold_pos = np.arange(split, split + HOLDOUT)
    y_tb = labels[rows].astype(int)
    if selected in ("a", "b", "c"):
        y_fit = y_tb if selected in ("a", "b") else \
            sup["target"].to_numpy(dtype=int)[rows]
        X = pd.DataFrame(frame[rows], columns=columns)
        model = _fit_flat(selected, X, y_fit)
        Xh = pd.DataFrame(frame[hold_pos], columns=columns)
        votes = _flat_proba(selected, model, Xh)
        joblib.dump(model, MODEL_PKL)
        return votes, {"artifact": str(MODEL_PKL)}
    # sequence arms
    if not fallback["fired"]:
        seqs_tr = ev.sequence_tensor(frame, rows)
        model = ev.train_lstm_eyes(seqs_tr, y_tb, seed=42)
        seqs_hold = ev.sequence_tensor(frame, hold_pos)
        votes = model.predict_proba(seqs_hold)
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
    subset = fallback["subset_cols"]
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(ev.lag_matrix(frame, rows, subset), y_tb)
    votes = model.predict_proba(ev.lag_matrix(frame, hold_pos, subset))[:, 1]
    joblib.dump(model, MODEL_PKL)
    return votes, {"artifact": str(MODEL_PKL), "subset_cols": subset}


def phase_fit_exam() -> None:
    if results().get("fit_exam"):
        print("fit_exam: checkpoint exists, skipping", flush=True)
        return
    res = results()
    selected = res["oof"]["selected"]
    fallback = res["oof"]["fallback"]
    frame = np.load(FRAME_PATH)
    labels = np.load(LABELS_PATH)
    rows = np.load(ROWS_PATH)
    columns = json.loads(COLS_PATH.read_text())
    df = _df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)

    t0 = time.time()
    votes_new, fit_info = _selected_votes(frame, labels, rows, columns,
                                          sup, selected, fallback)
    assert len(votes_new) == HOLDOUT and np.isfinite(votes_new).all()

    meta = nightly._champion_meta()
    champ_model = joblib.load(STACKER_CHAMPION)
    champ_frame, _tag = nightly.champion_holdout_frame(df, hold_sup, meta)
    votes_73 = champ_model.predict_proba(champ_frame)[:, 1]
    assert len(votes_73) == HOLDOUT and np.isfinite(votes_73).all()

    exam_pos = ev.exam_positions(labels, split, N_EXAM)
    y_exam = labels[exam_pos].astype(int)
    off = exam_pos - split
    exam_new = ev.exam_metrics(votes_new[off], y_exam)
    exam_73 = ev.exam_metrics(votes_73[off], y_exam)
    # Continuity diagnostics: run 73 on its own next-candle exam.
    y_nc = hold_sup["target"].to_numpy(dtype=int)
    diag_73 = {"next_candle_exam_rows": ev.exam_metrics(votes_73[off],
                                                       y_nc[off]),
               "next_candle_full_holdout": ev.exam_metrics(votes_73, y_nc),
               "recorded_holdout_acc": meta.get("accuracy")}
    np.save(VOTES_NEW_PATH, votes_new)
    np.save(VOTES_73_PATH, votes_73)
    save_result("fit_exam", {
        "selected": selected, "fit_info": fit_info,
        "exam_new": exam_new, "exam_73": exam_73,
        "spread_new_792": ev.vote_spread(votes_new),
        "spread_73_792": ev.vote_spread(votes_73),
        "spread_new_exam": ev.vote_spread(votes_new[off]),
        "spread_73_exam": ev.vote_spread(votes_73[off]),
        "diag_73": diag_73,
        "exam_win": bool(exam_new["accuracy"] > exam_73["accuracy"]),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"fit_exam: new acc {exam_new['accuracy']:.4f} "
          f"(prec {exam_new['precision']:.4f}, up {exam_new['up_calls']}) vs "
          f"run73 acc {exam_73['accuracy']:.4f} "
          f"(prec {exam_73['precision']:.4f}, up {exam_73['up_calls']}); "
          f"spread new [{votes_new.min():.3f}, {votes_new.max():.3f}] vs "
          f"73 [{votes_73.min():.3f}, {votes_73.max():.3f}]", flush=True)


def phase_duel() -> None:
    if results().get("duel"):
        print("duel: checkpoint exists, skipping", flush=True)
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
            f"TWO-WAY PIN FAIL: fresh champion arm agg {champ['score']:.2f} "
            f"/ P&L {champ['total_pnl']:.2f} != recorded "
            f"{CHAMP_AGG_PIN} / {CHAMP_PNL_PIN} — study aborts, "
            f"nothing registers")
    print(f"  champion pin OK: agg {champ['score']:.2f}, "
          f"P&L {champ['total_pnl']:.2f}", flush=True)

    print("grading challenger arm (44 + Eyes v2) ...", flush=True)
    chall_res = evaluate_windows(PPO_CHAMPION, windows,
                                 [votes_new[k * 264:(k + 1) * 264]
                                  for k in range(3)])
    assert chall_res is not None, "challenger pair failed to run"
    chall = _agg(chall_res)
    save_result("duel", {
        "champion": champ, "challenger": chall,
        "money_win": bool(chall["score"] > champ["score"]),
        "champ_pin": {"agg": CHAMP_AGG_PIN, "pnl": CHAMP_PNL_PIN,
                      "reproduced": True},
    })
    print(f"duel: challenger agg {chall['score']:.2f} / P&L "
          f"{chall['total_pnl']:+.2f} vs champion {champ['score']:.2f} / "
          f"{champ['total_pnl']:+.2f} — money win "
          f"{chall['score'] > champ['score']}", flush=True)


def phase_register() -> None:
    res = results()
    if res.get("registered"):
        print("register: already done, skipping", flush=True)
        return
    oof, fe, duel = res["oof"], res["fit_exam"], res["duel"]
    selected = oof["selected"]
    fallback = oof["fallback"]
    exam_win = bool(fe["exam_win"])
    money_win = bool(duel["money_win"])
    both = exam_win and money_win
    en, e7 = fe["exam_new"], fe["exam_73"]
    chall, champ = duel["challenger"], duel["champion"]
    arm_label = ARM_NAMES["d_fallback" if selected == "d"
                          and fallback["fired"] else selected]
    rel = (f"exam acc {en['accuracy']:.4f} vs run 73 re-scored "
           f"{e7['accuracy']:.4f} (prec {en['precision']:.4f} vs "
           f"{e7['precision']:.4f}, up-calls {en['up_calls']} vs "
           f"{e7['up_calls']}); duel agg {chall['score']:.0f} vs "
           f"champion {champ['score']:.0f}, P&L "
           f"{chall['total_pnl']:+,.0f} vs {champ['total_pnl']:+,.0f}")
    if both:
        verdict = "PASS"
        reason = (f"Eyes v2 ({arm_label}): {rel} — WOULD PROMOTE: "
                  f"held as PENDING KASS APPROVAL under the promotion "
                  f"lock; champion kept, nothing installed")
    elif exam_win:
        verdict = "EXAM ONLY"
        reason = (f"Eyes v2 ({arm_label}): exam win only — {rel}; "
                  f"the money test did not follow, so no pending "
                  f"approval (the eyes-duel lesson); champion kept")
    elif money_win:
        verdict = "MONEY ONLY"
        reason = (f"Eyes v2 ({arm_label}): money win only — {rel}; "
                  f"both bars are required for a pending record; "
                  f"champion kept")
    else:
        verdict = "FAIL"
        reason = (f"Eyes v2 ({arm_label}): champion kept — {rel} — "
                  f"failed eyes attempt (component study, not a "
                  f"numbered null)")
    metrics = {
        "family": ev.FAMILY,
        "challenger_acc": en["accuracy"],
        "champion_acc": e7["accuracy"],
        "precision": en["precision"],
        "champion_precision": e7["precision"],
        "challenger_wf_acc": oof["table"][selected]["accuracy"],
        "folds": 4,
        "deep": False,
        "pending_approval": bool(both),
        "reason": reason,
        "eyes_v2": {
            "label": ev.LABEL,
            "verdict": verdict,
            "design": "triple-barrier 1.0xATR/24h labels, pick2_98 "
                      "+ 8 ctx (106 feats), 4-arm zoo, OOF selection",
            "selected_arm": selected,
            "arm_name": arm_label,
            "fallback": fallback,
            "zoo_oof": oof["table"],
            "train_rows": res["frames"]["train_rows_common"],
            "train_base_rate": res["frames"]["train_base_rate"],
            "exam": {"new": en, "champ73": e7,
                     "exam_rows": res["frames"]["exam_rows"],
                     "label_base_rate": res["frames"]
                     ["label_base_rate_exam"]},
            "spread": {"new_792": fe["spread_new_792"],
                       "champ73_792": fe["spread_73_792"]},
            "duel": {"challenger": {k: chall[k] for k in
                                    ("score", "total_pnl", "max_drawdown",
                                     "bench_pnl", "action_mix")},
                     "champion": {k: champ[k] for k in
                                  ("score", "total_pnl", "max_drawdown",
                                   "bench_pnl", "action_mix")}},
            "exam_win": exam_win, "money_win": money_win,
        },
    }
    nightly.log_run("stacker", metrics, False)
    import sqlite3
    with sqlite3.connect(nightly.registry_db()) as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    nightly.add_lab_note(
        "Eyes v2 — PENDING KASS APPROVAL" if both
        else "Eyes v2 — Tradeable-Move Stacker", reason)
    guard_after = ev.guard_hashes()
    before = res["frames"]["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue  # the books' own loops write these; info only
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("registered", {"run_id": int(run_id), "verdict": verdict,
                               "reason": reason, "guard_after_ok": True})
    print(f"registered run {run_id}: verdict {verdict} — {reason}",
          flush=True)


PHASES = {"frames": phase_frames, "oof": phase_oof,
          "fit_exam": phase_fit_exam, "duel": phase_duel,
          "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== eyes v2 study runner done ===", flush=True)


if __name__ == "__main__":
    main()
