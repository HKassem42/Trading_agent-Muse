#!/usr/bin/env python
"""Regime-Swing System — run 86: study runner (2026-10-30).

Kass's full delegation for the last build before tonight's
retrain: "build whatever you want to get us the best
results — if you gotta rebuild the brain or the eyes or how
it trades, do that with everything we've learned." Design
frozen in refs/regime-swing-study-2026-10-30.md BEFORE any
of this code ran. In one paragraph: 4h buckets (END-stamped,
values-based) over run 82's persisted 1h fetch; the eyes are
a 3-state regime classifier (UP/DOWN/CHOP over the forward
4 days, band 0.75 x ATR4h) chosen from a two-arm zoo
(stacker recipe / LightGBM) on training-region OOF 3-class
balanced accuracy; the trader is deterministic rules (enter
with the regime, one add, exit on flip/CHOP) on the
production PaperBroker chassis. PRIMARY validation:
walk-forward, 4 folds, per-fold refits with a full label
purge. SECONDARY: the standard holdout duel vs the 44+73
champion pair (FIFTH use of that holdout), verdict by the
formula + gate_decision() floor.

LAB ONLY under the promotion lock: nothing installs, nothing
is crowned, no seat or book is touched (hash-guarded
before/after, incl. Book #3's gated files; the paper books
are never opened). Writes live only under data/regime_swing/,
plus the registry run and the lab note at the end. A pass
records WOULD PROMOTE — PENDING KASS APPROVAL
(pending_approval=1, promoted=0).

Phases checkpoint under data/regime_swing/; reruns skip done
phases (delete a phase's checkpoint to redo it):
  frames -> oof -> walkforward -> duel -> register

Usage: .venv/bin/python scripts/run_regime_swing_study.py [phase]
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
from training import nightly  # noqa: E402
from training import regime_swing as rs  # noqa: E402
from training.nightly import _agg, evaluate_windows, gate_decision  # noqa: E402

LAB82 = ROOT / "data" / "meta_label"
LAB = ROOT / "data" / "regime_swing"
LEDGERS = LAB / "ledgers"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB82 / "df.pkl"
HOLD_SUP82_PATH = LAB82 / "hold_sup.pkl"
REFS_PATH = ROOT / "refs" / "regime-swing-study-2026-10-30.md"
X_PATH = LAB / "features.npy"
LABELS_PATH = LAB / "labels.npy"
H4_PATH = LAB / "h4.npz"
ROWS_PATH = LAB / "rows.json"
COLS_PATH = LAB / "columns.json"
HOLD_SUP_PATH = LAB / "hold_sup.pkl"
OOF_PATH = LAB / "oof.npz"
MODEL_PKL = LAB / "regime_model.pkl"
PPO_CHAMPION = ROOT / "models" / "ppo_champion.zip"
STACKER_CHAMPION = ROOT / "models" / "stacker_champion.pkl"
HOLDOUT = 792
CHAMP_AGG_PIN = -210.17
CHAMP_PNL_PIN = -305.28
ARMS = ("stacker", "lgbm")


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


def _load_frame():
    X = np.load(X_PATH)
    labels = np.load(LABELS_PATH)
    h4 = np.load(H4_PATH)
    rows = json.loads(ROWS_PATH.read_text())
    return X, labels, h4, rows


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
    boundary = hold_sup.index[0]

    print("building 4h frame + features (115) ...", flush=True)
    t0 = time.time()
    h4, fstats = rs.build_4h_frame(df)
    atr_pct = rs.atr_pct_series(h4)
    labels = rs.regime_labels(h4["close"].to_numpy(dtype=float), atr_pct)
    A, cols_a = rs.block_a_features(h4, atr_pct)
    from features import pick2_features
    frame98, cols98, ev98 = pick2_features.build_pick2_frame(df)
    assert list(cols98) == champ_cols, \
        "pick2 columns differ from the champion's recorded set"
    pick2_vals = frame98.reindex(df.index)[champ_cols].to_numpy(dtype=float)
    B = rs.block_b_features(pick2_vals, df.index, h4.index)
    X = np.hstack([A, B])
    columns = list(cols_a) + [f"p98__{c}" for c in champ_cols]
    assert X.shape[1] == 115 and len(columns) == 115

    ends = h4.index
    finite = np.isfinite(X).all(axis=1)
    train_mask = np.asarray(ends <= boundary)
    hold_mask = np.asarray((ends > boundary)
                           & (ends <= boundary
                              + pd.Timedelta(hours=HOLDOUT)))
    train_rows = np.nonzero(train_mask & finite)[0]
    holdout_rows = np.nonzero(hold_mask & finite)[0]
    assert len(holdout_rows) == 198, \
        f"holdout decision buckets {len(holdout_rows)} != 198"
    for name, rr in (("train", train_rows), ("holdout", holdout_rows)):
        assert np.all(np.diff(rr) == 1), f"{name} decision rows not contiguous"
    boundary_pos = int(holdout_rows[0])
    elig = set(rs.eligible_fit_rows(labels, boundary_pos).tolist())
    fit_rows = np.array([t for t in train_rows if t in elig], dtype=int)
    y_fit = labels[fit_rows]
    assert set(np.unique(y_fit)) == {0, 1, 2}
    lab_counts = {rs.CLASS_NAMES[c]: int((labels[train_rows] == c).sum())
                  for c in (0, 1, 2)}

    LAB.mkdir(parents=True, exist_ok=True)
    np.save(X_PATH, X)
    np.save(LABELS_PATH, labels)
    np.savez(H4_PATH,
             end_ns=ends.values.astype("datetime64[ns]").astype(np.int64),
             open=h4["open"].to_numpy(dtype=float),
             high=h4["high"].to_numpy(dtype=float),
             low=h4["low"].to_numpy(dtype=float),
             close=h4["close"].to_numpy(dtype=float),
             atr_pct=atr_pct,
             first_pos=h4["first_pos"].to_numpy(dtype=int),
             last_pos=h4["last_pos"].to_numpy(dtype=int))
    ROWS_PATH.write_text(json.dumps({
        "train_rows": train_rows.tolist(),
        "fit_rows": fit_rows.tolist(),
        "holdout_rows": holdout_rows.tolist(),
        "boundary_pos": boundary_pos}))
    COLS_PATH.write_text(json.dumps(columns))
    hold_sup.to_pickle(HOLD_SUP_PATH)
    save_result("frames", {
        "rows_sup": len(sup), "split": split,
        "boundary": str(boundary),
        "frame_stats": fstats,
        "n_buckets": int(len(h4)),
        "train_rows": int(len(train_rows)),
        "fit_rows": int(len(fit_rows)),
        "holdout_rows": int(len(holdout_rows)),
        "train_first_end": str(ends[train_rows[0]]),
        "train_last_end": str(ends[train_rows[-1]]),
        "label_counts_train_rows": lab_counts,
        "fit_base_rate": {rs.CLASS_NAMES[c]: float(np.mean(y_fit == c))
                          for c in (0, 1, 2)},
        "pick2_evidence": ev98,
        "guard_before": rs.guard_hashes(),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"frames: {len(h4)} buckets, {len(train_rows)} train decision "
          f"rows, {len(fit_rows)} fit rows, 198 holdout buckets; "
          f"labels {lab_counts}", flush=True)


def phase_oof() -> None:
    if results().get("oof"):
        print("oof: checkpoint exists, skipping", flush=True)
        return
    from sklearn.model_selection import TimeSeriesSplit

    X, labels, _h4, rows = _load_frame()
    fit_rows = np.array(rows["fit_rows"], dtype=int)
    Xf, yf = X[fit_rows], labels[fit_rows].astype(int)
    tss = list(TimeSeriesSplit(n_splits=4).split(Xf))
    table: dict = {}
    probas: dict = {}
    for arm in ARMS:
        t0 = time.time()
        proba = np.full((len(fit_rows), 3), np.nan)
        for tr, te in tss:
            model = rs.fit_eyes(Xf[tr], yf[tr], arm)
            proba[te] = rs.predict_proba(model, arm, Xf[te])
        covered = np.isfinite(proba).all(axis=1)
        table[arm] = rs.oof_metrics(proba[covered], yf[covered])
        probas[arm] = proba
        print(f"oof arm {arm}: BA {table[arm]['balanced_accuracy']:.4f} "
              f"AUC {table[arm]['macro_auc']:.4f} acc "
              f"{table[arm]['accuracy']:.4f} recalls "
              f"{table[arm]['per_class_recall']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    selected = rs.select_arm(table)
    np.savez(OOF_PATH, **{f"proba_{k}": v for k, v in probas.items()})
    save_result("oof", {"table": table, "selected": selected})
    print(f"oof: selected arm {selected}", flush=True)


def _fit_rows_for(train_rows: np.ndarray, labels: np.ndarray,
                  boundary_pos: int) -> np.ndarray:
    elig = set(rs.eligible_fit_rows(labels, boundary_pos).tolist())
    return np.array([t for t in train_rows if t in elig], dtype=int)


def _span_run(df, h4, buckets, regimes, name: str) -> dict:
    buckets = np.asarray(buckets, dtype=int)
    return rs.run_span(df, h4["first_pos"], h4["last_pos"],
                       list(buckets), np.asarray(regimes),
                       h4["atr_pct"][buckets], LEDGERS / f"{name}.db")


def phase_walkforward() -> None:
    if results().get("walkforward"):
        print("walkforward: checkpoint exists, skipping", flush=True)
        return
    selected = results()["oof"]["selected"]
    X, labels, h4, rows = _load_frame()
    df = _df()
    train_rows = np.array(rows["train_rows"], dtype=int)
    # Amendment (refs doc, recorded before any WF grading):
    # 5 segments — segment 0 is the burn-in (fit-only), the
    # 4 graded folds are segments 1-4, expanding-window fits.
    segments = np.array_split(train_rows, 5)
    folds = segments[1:]
    burn_in = int(len(segments[0]))
    h4close = pd.Series(h4["close"], index=pd.to_datetime(h4["end_ns"]))
    fold_recs = []
    for k, fold in enumerate(folds):
        t0 = time.time()
        fstart = int(fold[0])
        frows = _fit_rows_for(train_rows, labels, fstart)
        assert len(frows) > 100, f"fold {k}: only {len(frows)} fit rows"
        model = rs.fit_eyes(X[frows], labels[frows].astype(int), selected)
        regimes = rs.predict_class(model, selected, X[fold])
        rec = _span_run(df, h4, fold, regimes, f"wf_fold{k}")
        final_close = float(df["close"].iloc[int(h4["last_pos"][fold[-1]])])
        resumed = rs.resum_span_pnl(LEDGERS / f"wf_fold{k}.db", final_close)
        assert abs(resumed - rec["pnl"]) < 0.01, \
            f"fold {k} two-way check: re-sum {resumed:.2f} vs {rec['pnl']:.2f}"
        bench = 100_000.0 * (float(h4close.iloc[int(fold[-1])])
                             / float(h4close.iloc[fstart - 1]) - 1.0)
        pred_counts = {rs.CLASS_NAMES[c]: int((regimes == c).sum())
                       for c in (0, 1, 2)}
        fold_recs.append({
            "fold": k, "first_end": str(h4close.index[int(fold[0])]),
            "last_end": str(h4close.index[int(fold[-1])]),
            "n_buckets": int(len(fold)), "fit_rows": int(len(frows)),
            "pred_counts": pred_counts,
            "pnl": rec["pnl"], "max_drawdown": rec["max_drawdown"],
            "score": rec["score"], "trades": rec["trades"],
            "fees": rec["fees"], "round_trips": rec["round_trips"],
            "time_in_position": rec["time_in_position"],
            "bench_pnl": bench, "resum_diff": abs(resumed - rec["pnl"]),
        })
        print(f"wf fold {k}: fit {len(frows)} rows, P&L "
              f"{rec['pnl']:+,.2f} (B&H {bench:+,.2f}), DD "
              f"{rec['max_drawdown']:,.2f}, trades {rec['trades']}, "
              f"fees ${rec['fees']:.2f}, TiP "
              f"{rec['time_in_position']:.1%} ({time.time() - t0:.0f}s)",
              flush=True)
    tot_pnl = float(sum(f["pnl"] for f in fold_recs))
    worst_dd = float(max(f["max_drawdown"] for f in fold_recs))
    n_candles = sum(f["n_buckets"] for f in fold_recs)
    totals = {
        "pnl": tot_pnl, "max_drawdown": worst_dd,
        "score": tot_pnl - 0.5 * worst_dd,
        "trades": int(sum(f["trades"] for f in fold_recs)),
        "fees": float(sum(f["fees"] for f in fold_recs)),
        "round_trips": int(sum(f["round_trips"] for f in fold_recs)),
        "time_in_position": float(
            sum(f["time_in_position"] * f["n_buckets"]
                for f in fold_recs) / n_candles),
        "bench_pnl": float(sum(f["bench_pnl"] for f in fold_recs)),
    }
    save_result("walkforward", {"folds": fold_recs, "totals": totals,
                                "selected": selected,
                                "burn_in_rows": burn_in})
    print(f"walkforward totals: P&L {tot_pnl:+,.2f} vs B&H "
          f"{totals['bench_pnl']:+,.2f}, score {totals['score']:+,.2f}, "
          f"trades {totals['trades']}, fees ${totals['fees']:.2f}",
          flush=True)


def phase_duel() -> None:
    if results().get("duel"):
        print("duel: checkpoint exists, skipping", flush=True)
        return
    res = results()
    selected = res["oof"]["selected"]
    X, labels, h4, rows = _load_frame()
    df = _df()
    fit_rows = np.array(rows["fit_rows"], dtype=int)
    holdout_rows = np.array(rows["holdout_rows"], dtype=int)
    hold_sup = pd.read_pickle(HOLD_SUP_PATH)
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]

    t0 = time.time()
    model = rs.fit_eyes(X[fit_rows], labels[fit_rows].astype(int), selected)
    joblib.dump(model, MODEL_PKL)
    regimes_hold = rs.predict_class(model, selected, X[holdout_rows])
    # Block importance shares on the final fit (separability record).
    imp_share = None
    try:
        if selected == "stacker":
            from training.eyes_v2 import pick2_importance_ranking
            imp = pick2_importance_ranking(model)
        else:
            imp = np.asarray(model.feature_importances_, dtype=float)
        if len(imp) == X.shape[1] and imp.sum() > 0:
            imp = imp / imp.sum()
            imp_share = {"block_a": float(imp[:17].sum()),
                         "block_b": float(imp[17:].sum())}
    except Exception as e:
        imp_share = {"error": f"{type(e).__name__}: {e}"}
    print(f"duel: final fit on {len(fit_rows)} rows ({selected}); "
          f"holdout regime counts "
          f"{ {rs.CLASS_NAMES[c]: int((regimes_hold == c).sum()) for c in (0, 1, 2)} }",
          flush=True)

    # Champion arm — production machinery, two-way pin.
    meta = nightly._champion_meta()
    champ_model = joblib.load(STACKER_CHAMPION)
    champ_frame, _tag = nightly.champion_holdout_frame(df, hold_sup, meta)
    votes73 = champ_model.predict_proba(champ_frame)[:, 1]
    champ_res = evaluate_windows(
        PPO_CHAMPION, windows,
        [votes73[k * 264:(k + 1) * 264] for k in range(3)])
    assert champ_res is not None, "champion pair failed to run"
    champ = _agg(champ_res)
    if abs(champ["score"] - CHAMP_AGG_PIN) > 0.011 or \
            abs(champ["total_pnl"] - CHAMP_PNL_PIN) > 0.011:
        raise RuntimeError(
            f"TWO-WAY PIN FAIL: fresh champion arm agg {champ['score']:.2f} "
            f"/ P&L {champ['total_pnl']:.2f} != recorded "
            f"{CHAMP_AGG_PIN} / {CHAMP_PNL_PIN} — study aborts, "
            f"nothing registers")
    print(f"  champion pin OK: agg {champ['score']:.2f}, P&L "
          f"{champ['total_pnl']:.2f}", flush=True)

    # Challenger arm — the swing trader per window.
    pos_of = {ts: i for i, ts in enumerate(df.index)}
    win_recs, win_objs = [], []
    for w in range(3):
        buckets = holdout_rows[w * 66:(w + 1) * 66]
        assert int(h4["first_pos"][buckets[0]]) == \
            pos_of[hold_sup.index[w * 264]], f"window {w} start misaligned"
        assert int(h4["last_pos"][buckets[-1]]) == \
            pos_of[hold_sup.index[(w + 1) * 264 - 1]], \
            f"window {w} end misaligned"
        rec = _span_run(df, h4, buckets, regimes_hold[w * 66:(w + 1) * 66],
                        f"duel_w{w}")
        closes = windows[w]["close"].to_numpy(dtype=float)
        bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0))
        final_close = float(df["close"].iloc[int(h4["last_pos"][buckets[-1]])])
        resumed = rs.resum_span_pnl(LEDGERS / f"duel_w{w}.db", final_close)
        assert abs(resumed - rec["pnl"]) < 0.01, \
            f"window {w} two-way check: re-sum {resumed:.2f} vs {rec['pnl']:.2f}"
        mix = {k: rec["action_counts"][k] / max(
            sum(rec["action_counts"].values()), 1)
            for k in ("hold", "buy", "sell", "short")}
        win_recs.append({
            "score": rec["score"], "pnl": rec["pnl"],
            "max_drawdown": rec["max_drawdown"], "bench_pnl": bench,
            "trades": rec["trades"], "fees": rec["fees"],
            "round_trips": rec["round_trips"],
            "time_in_position": rec["time_in_position"],
            "action_mix": mix, "resum_diff": abs(resumed - rec["pnl"])})
        win_objs.append(rec)
        print(f"  window {w}: score {rec['score']:+.2f}, P&L "
              f"{rec['pnl']:+,.2f} (B&H {bench:+,.2f}), trades "
              f"{rec['trades']}, fees ${rec['fees']:.2f}, TiP "
              f"{rec['time_in_position']:.1%}", flush=True)
    chall = {
        "score": float(np.mean([w["score"] for w in win_recs])),
        "total_pnl": float(sum(w["pnl"] for w in win_recs)),
        "mean_reward": 0.0,
        "max_drawdown": float(max(w["max_drawdown"] for w in win_recs)),
        "mean_drawdown": float(np.mean([w["max_drawdown"]
                                        for w in win_recs])),
        "bench_pnl": float(sum(w["bench_pnl"] for w in win_recs)),
        "episodes": 30,
        "round_trips_per_day": float(np.mean(
            [w["round_trips"] / 11.0 for w in win_recs])),
        "action_mix": {k: round(v, 3)
                       for k, v in win_recs[-1]["action_mix"].items()},
        "windows": [{"score": round(w["score"], 2),
                     "pnl": round(w["pnl"], 2),
                     "bench": round(w["bench_pnl"], 2),
                     "dd": round(w["max_drawdown"], 2),
                     "mdd": round(w["max_drawdown"], 2),
                     "mr": 0.0, "episodes": 10} for w in win_recs],
        "runnable": True,
    }
    wins = sum(1 for cw, hw in zip(champ_res["windows"], win_recs)
               if hw["score"] > cw["score"])
    formula = bool(chall["score"] > champ["score"] and wins >= 2)
    promotable, floor_fail = gate_decision(chall, formula)
    save_result("duel", {
        "champion": champ, "challenger": chall, "wins": wins,
        "formula_pass": formula, "floor_fail": floor_fail,
        "promotable": bool(promotable),
        "windows_detail": win_recs,
        "imp_share": imp_share,
        "holdout_pred_counts": {rs.CLASS_NAMES[c]:
                                int((regimes_hold == c).sum())
                                for c in (0, 1, 2)},
        "champ_pin": {"agg": CHAMP_AGG_PIN, "pnl": CHAMP_PNL_PIN,
                      "reproduced": True},
        "seconds": round(time.time() - t0, 1),
    })
    print(f"duel: challenger agg {chall['score']:.2f} / P&L "
          f"{chall['total_pnl']:+,.2f} vs champion {champ['score']:.2f} / "
          f"{champ['total_pnl']:+,.2f}; wins {wins}/3, formula={formula}, "
          f"floor_fail={floor_fail}, promotable={promotable}", flush=True)


def phase_register() -> None:
    res = results()
    if res.get("registered"):
        print("register: already done, skipping", flush=True)
        return
    frames, oof = res["frames"], res["oof"]
    wf, duel = res["walkforward"], res["duel"]
    selected = oof["selected"]
    chall, champ = duel["challenger"], duel["champion"]
    wins = duel["wins"]
    formula = duel["formula_pass"]
    floor_fail = duel["floor_fail"]
    promotable = duel["promotable"]
    wft = wf["totals"]
    oof_sel = oof["table"][selected]
    rel = (f"walk-forward P&L {wft['pnl']:+,.0f} over 4 folds (B&H "
           f"{wft['bench_pnl']:+,.0f}, {wft['trades']} trades, fees "
           f"${wft['fees']:.0f}); holdout duel aggregate "
           f"{chall['score']:.0f} vs champion {champ['score']:.0f}, won "
           f"{wins}/3 windows, P&L {chall['total_pnl']:+,.0f} vs "
           f"champion {champ['total_pnl']:+,.0f} (buy-and-hold "
           f"{chall['bench_pnl']:+,.0f}); eyes OOF balanced acc "
           f"{oof_sel['balanced_accuracy']:.3f} ({selected} arm)")
    if promotable:
        verdict = "PASS"
        reason = (f"Regime-Swing System: {rel} (gate pass incl. "
                  f"two-part floor) — WOULD PROMOTE: held as PENDING "
                  f"KASS APPROVAL under the promotion lock; champion "
                  f"kept, nothing installed")
    elif formula and floor_fail is not None:
        verdict = "FAIL"
        reason = (f"Regime-Swing System: passed the relative gate "
                  f"({rel}) but was not promoted — {floor_fail}; "
                  f"champion kept — failed system attempt")
    else:
        verdict = "FAIL"
        reason = (f"Regime-Swing System: champion kept — {rel} — "
                  f"failed system attempt")
    metrics = {
        "family": rs.FAMILY,
        "challenger": chall,
        "champion": champ,
        "reason": reason,
        "deep": False,
        "timesteps": 0,
        "pending_approval": bool(promotable),
        "reward_curve": None,
        "telemetry": None,
        "regime_swing": {
            "label": "4h Swing",
            "verdict": verdict,
            "design": "4h regime classifier (115 feats: 17 native "
                      "+ pick2_98 block) + deterministic swing rules "
                      "on the production broker; walk-forward primary, "
                      "holdout duel fifth use",
            "selected_arm": selected,
            "eyes_oof": oof["table"],
            "walk_forward": wf,
            "duel": {"wins": wins, "formula_pass": formula,
                     "floor_fail": floor_fail,
                     "windows_detail": duel["windows_detail"],
                     "holdout_pred_counts": duel["holdout_pred_counts"],
                     "imp_share": duel["imp_share"],
                     "champ_pin": duel["champ_pin"]},
            "frames": {k: frames[k] for k in
                       ("train_rows", "fit_rows", "holdout_rows",
                        "label_counts_train_rows", "fit_base_rate",
                        "frame_stats")},
        },
    }
    nightly.log_run("ppo", metrics, False)
    import sqlite3
    with sqlite3.connect(nightly.registry_db()) as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    nightly.add_lab_note(
        "Regime-Swing System — PENDING KASS APPROVAL" if promotable
        else "Regime-Swing System", reason)
    guard_after = rs.guard_hashes()
    before = frames["guard_before"]
    for k, v in before.items():
        if k.endswith("_info_only"):
            continue  # the books' own loops write these; info only
        assert guard_after[k] == v, f"GUARD TRIPPED on {k}"
    save_result("registered", {"run_id": int(run_id), "verdict": verdict,
                               "reason": reason, "guard_after_ok": True})
    print(f"registered run {run_id}: verdict {verdict} — {reason}",
          flush=True)


PHASES = {"frames": phase_frames, "oof": phase_oof,
          "walkforward": phase_walkforward, "duel": phase_duel,
          "register": phase_register}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[which]()
    print("=== regime-swing study runner done ===", flush=True)


if __name__ == "__main__":
    main()
