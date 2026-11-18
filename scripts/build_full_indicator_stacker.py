"""Full-indicator stacker LAB study (2026-10-25, Kass's direction).

Kass: "incorporate all those sub indicators and main indicators
into the stacker and let it make those decisions based on those
indicators." So: the PRODUCTION stacker recipe (RandomForest +
XGBoost + LightGBM -> LogisticRegression, scaler + SMOTE) trained on
the full computable indicator universe, graded against the current
21-feature champion stacker on the same exam.

Two arms, both rebuilt FRESH on current BTC/USD 1h data (730d fetch,
same source as the nightly run):
- Arm A "Raw 98": the matrix-v2 column set -- 59 raw AZ indicator
  series + 31 main-chart overlays = 90 feature columns (the v2 file's
  98 columns include Date/Ticker/Target/OHLCV).
- Arm B "Scored 102": the matrix-v3 column set -- the same 90 series
  as trailing-4,320-bar percentile scores (0-100, oriented per the
  bullish/bearish guide) + the 4 family composites = 94 features.

Feature construction reuses the matrix builders' own code paths
(scripts/build_timeseries_matrix_v3.py's build_series /
ranked_quantity / rolling_scores, which wrap the v1 AZ builders and
src/features/main_chart.py) -- no re-implementation. Labels are the
PRODUCTION stacker target exactly (make_supervised: 1 if the next
candle closes higher), NOT the matrices' band label. Split identical
to nightly: holdout = last 792 supervised rows, training = everything
before (each arm drops its own warmup-NaN rows, as production does),
disjointness asserted. Verdict per arm = strict holdout-accuracy win
vs the champion re-scored on the identical holdout timestamps (the
production stacker rule).

Causality is re-verified on the FRESH build under the v5.1 standard:
truncation rebuilds (features recomputed on a truncated frame must
match the full-frame values on shared rows) for sampled columns of
each arm -- incl. a main-chart overlay and scored columns -- plus a
brute-force second method for the percentile scores and range
audits. A column that failed would be dropped and documented, never
silently fixed forward (none did -- see results.json).

LAB ONLY: never runs the live loop, never writes the paper ledger,
never installs anything. Registrations are promoted=0; a strict win
is recorded pending_approval under the promotion lock (Kass,
2026-10-25). Champion files are hash-guarded before/after;
src/features/technical_indicators.py is untouched (productionizing a
winner is a separate Kass-approved step). PPO is NOT retrained.

Usage: .venv/bin/python scripts/build_full_indicator_stacker.py
Phases checkpoint to data/full_indicator_stacker/; reruns skip done
phases.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.market_data import fetch_klines  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, make_supervised)
from models import stacker  # noqa: E402
from training.nightly import add_lab_note, log_run  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "matrix_v3_builder", ROOT / "scripts" / "build_timeseries_matrix_v3.py")
b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b3)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "full_indicator_stacker"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
FEAT_PATH = LAB / "raw90.pkl"
ARMA_PATH = LAB / "arm_a.pkl"
ARMB_PATH = LAB / "arm_b.pkl"
V2_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v2.csv"

HOLDOUT = 792
WINDOW = b3.WINDOW  # 4320 trailing bars for the percentile scores

# Truncation-check samples. Arm A: AZ oscillator, AZ flipped-source,
# AZ cumulative, main-chart overlay distance, main-chart side,
# main-chart fractal. Arm B reuses the first two + SuperTrend Side
# through the full scoring path.
SAMPLE_A = ["Relative Strength Index (RSI)", "Connors RSI (CRSI)",
            "On-Balance Volume (OBV)", "Distance to SMA 200 (%)",
            "SuperTrend Side (+1/-1)",
            "Distance to Fractal Resistance (%)"]
SAMPLE_B = ["Relative Strength Index (RSI)", "Connors RSI (CRSI)",
            "SuperTrend Side (+1/-1)"]


# ---------------------------------------------------------------- state
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    _save_json(PROGRESS_PATH, {"done": sorted(progress() | {phase})})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard() -> dict:
    return {name: _sha(MODELS / name) for name in
            ("ppo_champion.zip", "stacker_champion.pkl",
             "stacker_champion.json")} | {
            "paper.db": _sha(ROOT / "data" / "paper.db")}


def v2_feature_columns() -> list[str]:
    """The canonical 90 feature columns of the matrix-v2 file, in
    file order (59 AZ raw + 31 main-chart), metadata excluded."""
    import csv
    with open(V2_CSV) as fh:
        header = next(csv.reader(fh))
    cols = [c for c in header if c in b3.SPEC]
    assert len(cols) == 90, len(cols)
    return cols


def get_df() -> pd.DataFrame:
    if DF_PATH.exists():
        return pd.read_pickle(DF_PATH)
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(DF_PATH)
    return df


def get_frames():
    """(df, prod_sup, split, holdout_start, hold_start_pos)."""
    df = get_df()
    sup = make_supervised(df)
    split = len(sup) - HOLDOUT
    holdout_start = sup.index[split]
    hold_start_pos = int(df.index.get_indexer([holdout_start])[0])
    assert df.index[hold_start_pos] == holdout_start
    assert sup.index[:split].max() < holdout_start
    return df, sup, split, holdout_start, hold_start_pos


# ---------------------------------------------------------------- phases
def phase_data() -> None:
    df, sup, split, holdout_start, hold_start_pos = get_frames()
    save_result("meta", {
        "rows_raw": len(df), "rows_sup": len(sup), "split": split,
        "holdout_start": str(holdout_start),
        "hold_start_pos": hold_start_pos,
        "data_first": str(df.index[0]), "data_last": str(df.index[-1]),
        "guard_before": _guard(),
        "champion_ppo_sha": _sha(MODELS / "ppo_champion.zip"),
    })
    print(f"data: {len(df)} raw rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(sup)}, holdout from {holdout_start}", flush=True)


def _truncation_check(df: pd.DataFrame, feat: pd.DataFrame,
                      cuts: list[int]) -> dict:
    """Rebuild the raw series on truncated frames; sampled columns
    must match the full-frame values on every shared row."""
    out = {}
    for K in cuts:
        feat_t = b3.build_series(df.iloc[:K + 1])
        per = {}
        for col in SAMPLE_A:
            a = feat[col].iloc[:K + 1].to_numpy(float)
            b = feat_t[col].to_numpy(float)
            d = np.abs(a - b)
            d = d[~np.isnan(d)]
            mx = float(d.max()) if len(d) else 0.0
            ok = bool(np.allclose(a, b, rtol=0, atol=1e-9,
                                  equal_nan=True))
            per[col] = {"max_abs_diff": mx, "pass": ok}
            assert ok, f"causality FAIL (raw, K={K}): {col} max {mx}"
        out[str(K)] = per
        print(f"causality raw truncation K={K}: PASS "
              f"({len(SAMPLE_A)} sampled columns, all rows)",
              flush=True)
    return out


def phase_build() -> None:
    df, sup, split, holdout_start, hold_start_pos = get_frames()
    cols90 = v2_feature_columns()

    print("building raw 90 series on the fresh frame ...", flush=True)
    t0 = time.time()
    feat = b3.build_series(df)
    assert set(feat.columns) == set(cols90), (
        f"column mismatch: {set(cols90) ^ set(feat.columns)}")
    feat = feat[cols90]
    feat.to_pickle(FEAT_PATH)
    print(f"  raw90 built in {time.time() - t0:.0f}s", flush=True)

    # ---- causality, Arm A (raw): truncation rebuilds ----
    cuts = [9000, hold_start_pos + 300]
    trunc_a = _truncation_check(df, feat, cuts)

    # ---- Arm B: percentile scores on trailing 4320-bar windows ----
    close = df["close"]
    qa_map: dict[str, np.ndarray] = {}
    p0 = WINDOW - 1
    warmup = {}
    for name in cols90:
        q = b3.ranked_quantity(name, feat, close)
        qa = q.to_numpy(float)
        isn = np.isnan(qa) | np.isinf(qa)
        last_bad = int(np.max(np.nonzero(isn))) if isn.any() else -1
        warmup[name] = last_bad
        # first position whose full trailing window is finite
        p0 = max(p0, last_bad + WINDOW)
        qa_map[name] = qa
    positions = np.arange(p0, len(df))
    assert positions[-1] == len(df) - 1
    assert hold_start_pos >= p0, "holdout precedes the scored region"
    scores: dict[str, np.ndarray] = {}
    for name in cols90:
        s = b3.rolling_scores(qa_map[name], positions)
        if b3.SPEC[name][2] == "FLIP":
            s = 100.0 - s
        assert s.min() >= 0.0 and s.max() <= 100.0, name
        scores[name] = s
    fam_members: dict[str, list[str]] = {"trend": [], "momentum": [],
                                         "volume": []}
    for name in cols90:
        fam = b3.SPEC[name][3]
        if fam:
            fam_members[fam].append(name)
    fam_scores = {f: np.mean([scores[n] for n in ms], axis=0)
                  for f, ms in fam_members.items()}
    master = np.mean([fam_scores[f] for f in
                      ("trend", "momentum", "volume")], axis=0)
    b_cols = (["score_master", "score_trend", "score_momentum",
               "score_volume"] + cols90)
    bframe = pd.DataFrame(index=df.index)
    for cname, arr in (("score_master", master),
                       ("score_trend", fam_scores["trend"]),
                       ("score_momentum", fam_scores["momentum"]),
                       ("score_volume", fam_scores["volume"])):
        full = np.full(len(df), np.nan)
        full[positions] = arr
        bframe[cname] = full
    for name in cols90:
        full = np.full(len(df), np.nan)
        full[positions] = scores[name]
        bframe[name] = full
    print(f"arm B scored: p0={p0} ({df.index[p0]}), "
          f"{len(positions)} scored rows, composites "
          f"{ {f: len(m) for f, m in fam_members.items()} }",
          flush=True)

    # ---- causality, Arm B: brute-force second method + truncation
    # through the full scoring path on the K2 truncated rebuild ----
    brute = {}
    rsi_q = qa_map["Relative Strength Index (RSI)"]
    rsi_scores = scores["Relative Strength Index (RSI)"]
    for j in (0, len(positions) // 2, len(positions) - 1):
        p = int(positions[j])
        bf = b3.brute_score(rsi_q, p)
        assert abs(bf - float(rsi_scores[j])) < 1e-9, (p, bf,
                                                       rsi_scores[j])
        brute[str(p)] = {"brute": bf, "rolling": float(rsi_scores[j])}
    K2 = cuts[-1]
    feat_t = b3.build_series(df.iloc[:K2 + 1])
    close_t = df["close"].iloc[:K2 + 1]
    pos_t = positions[positions <= K2]
    trunc_b = {}
    for name in SAMPLE_B:
        q_t = b3.ranked_quantity(name, feat_t, close_t).to_numpy(float)
        s_t = b3.rolling_scores(q_t, pos_t)
        if b3.SPEC[name][2] == "FLIP":
            s_t = 100.0 - s_t
        ref = scores[name][:len(pos_t)]
        d = float(np.abs(s_t - ref).max())
        trunc_b[name] = d
        assert d == 0.0, f"causality FAIL (scored): {name} diff {d}"
    print(f"causality scored: brute-force exact on 3 rows; truncation "
          f"exact on {len(SAMPLE_B)} columns x {len(pos_t)} rows",
          flush=True)

    # ---- assemble arm frames on the supervised rows ----
    hold_idx = sup.index[split:]
    arms = {}
    for arm, frame, cols in (("A", feat, cols90),
                             ("B", bframe, b_cols)):
        af = frame.reindex(sup.index)[cols].copy()
        af["target"] = sup["target"].astype(int)
        hold = af.iloc[split:]
        assert list(hold.index) == list(hold_idx)
        assert np.isfinite(hold[cols].to_numpy(float)).all(), (
            f"arm {arm}: non-finite feature inside the holdout")
        assert (hold["target"].to_numpy()
                == sup.iloc[split:]["target"].to_numpy()).all()
        train = af.iloc[:split].dropna(subset=cols)
        arms[arm] = {"frame": af, "train_rows": len(train),
                     "dropped_warmup_train_rows": int(split
                                                      - len(train))}
        af.to_pickle(ARMA_PATH if arm == "A" else ARMB_PATH)
        print(f"arm {arm}: {len(cols)} features, train rows "
              f"{len(train)} (warmup-dropped "
              f"{split - len(train)}), holdout {len(hold)} rows "
              f"finite + label-identical", flush=True)
    save_result("build", {
        "cols90": cols90, "arm_b_cols": b_cols, "score_p0": int(p0),
        "score_p0_ts": str(df.index[p0]),
        "scored_rows": int(len(positions)),
        "composite_members": fam_members,
        "warmup_last_bad_max": int(max(warmup.values())),
        "causality": {"raw_truncation": trunc_a,
                      "scored_brute_rsi": brute,
                      "scored_truncation_max_abs_diff": trunc_b},
        "dropped_columns": [],
        "arms": {a: {"train_rows": v["train_rows"],
                     "dropped_warmup_train_rows":
                         v["dropped_warmup_train_rows"]}
                 for a, v in arms.items()},
    })


def _redundancy(frame: pd.DataFrame, cols: list[str]) -> dict:
    """Pairwise |r| summary + greedy union-find clusters at 0.90."""
    corr = frame[cols].corr().abs().to_numpy(float)
    n = len(cols)
    iu = np.triu_indices(n, k=1)
    vals = corr[iu]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs90 = int((vals > 0.90).sum())
    pairs95 = int((vals > 0.95).sum())
    idx90 = np.argwhere(np.triu(corr > 0.90, k=1))
    for i, j in idx90:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    from collections import Counter
    sizes = Counter(find(i) for i in range(n))
    clusters = sorted((s for s in sizes.values() if s >= 2),
                      reverse=True)
    max_r = corr.copy()
    np.fill_diagonal(max_r, 0.0)
    return {"n_features": n, "pairs_gt_0.90": pairs90,
            "pairs_gt_0.95": pairs95,
            "clusters_gt_0.90": len(clusters),
            "largest_cluster": clusters[0] if clusters else 1,
            "cols_with_max_r_gt_0.90": int((max_r.max(axis=1)
                                            > 0.90).sum())}


def _train_arm(arm: str, cols: list[str], champ_model,
               holdp: pd.DataFrame, acc_c: float) -> dict:
    from sklearn.metrics import accuracy_score, precision_score

    af = pd.read_pickle(ARMA_PATH if arm == "A" else ARMB_PATH)
    _, sup, split, _, _ = get_frames()
    train = af.iloc[:split].dropna(subset=cols)
    hold = af.iloc[split:]
    Xtr, ytr = train[cols], train["target"].astype(int)
    Xhold, yhold = hold[cols], hold["target"].astype(int)
    assert list(hold.index) == list(holdp.index)
    t0 = time.time()
    score = stacker.train_and_save(Xtr, ytr, LAB / f"stacker_arm{arm}.pkl")
    model = joblib.load(LAB / f"stacker_arm{arm}.pkl")
    pred = model.predict(Xhold)
    acc = float(accuracy_score(yhold, pred))
    prec = float(precision_score(yhold, pred, zero_division=0))
    # Decision anatomy: the fitted base learners' own holdout grades
    # and what each leans on (the stack's scaler feeds them).
    scaler = model.named_steps["scaler"]
    stack = model.named_steps["stack"]
    Xs_hold = scaler.transform(Xhold)
    base_acc, imps = {}, {}
    for name, est in stack.named_estimators_.items():
        base_acc[name] = float(accuracy_score(
            yhold, est.predict(Xs_hold)))
        imp = np.asarray(est.feature_importances_, dtype=float)
        imps[name] = imp / imp.sum() if imp.sum() > 0 else imp
    agg_imp = np.mean([imps[k] for k in ("rf", "xgb", "lgbm")], axis=0)
    order = np.argsort(-agg_imp)
    top15 = [{"feature": cols[i], "importance": float(agg_imp[i]),
              "rf": float(imps["rf"][i]), "xgb": float(imps["xgb"][i]),
              "lgbm": float(imps["lgbm"][i])} for i in order[:15]]
    meta_coef = {name: float(c) for name, c in zip(
        ("rf", "xgb", "lgbm"),
        stack.final_estimator_.coef_[0])}
    win = acc > acc_c
    print(f"arm {arm}: holdout acc={acc:.4f} prec={prec:.4f} "
          f"(wf {score['accuracy']:.4f}) vs champion {acc_c:.4f} "
          f"-> {'WIN' if win else 'no win'} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return {"acc": acc, "precision": prec,
            "wf_acc": score["accuracy"],
            "wf_precision": score["precision"], "folds": score["folds"],
            "win": win, "train_rows": int(len(train)),
            "base_learner_holdout_acc": base_acc,
            "meta_coef": meta_coef, "top15": top15,
            "redundancy": _redundancy(train, cols)}


def phase_train() -> None:
    from sklearn.metrics import accuracy_score, precision_score

    _, sup, split, _, _ = get_frames()
    holdp = sup.iloc[split:]
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    pred_c = champ_model.predict(holdp[FEATURE_COLUMNS])
    acc_c = float(accuracy_score(holdp["target"], pred_c))
    prec_c = float(precision_score(holdp["target"], pred_c,
                                   zero_division=0))
    print(f"champion re-scored on identical holdout: acc={acc_c:.4f} "
          f"prec={prec_c:.4f}", flush=True)
    b = results()["build"]
    out = {"champion": {"acc": acc_c, "precision": prec_c,
                        "features": FEATURE_COLUMNS}}
    for arm, cols in (("A", b["cols90"]), ("B", b["arm_b_cols"])):
        out[arm] = _train_arm(arm, cols, champ_model, holdp, acc_c)
    train_sup = sup.iloc[:split]
    out["redundancy_production21"] = _redundancy(
        train_sup[FEATURE_COLUMNS], FEATURE_COLUMNS)
    save_result("train", out)


def _already_registered() -> int:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM runs WHERE kind='stacker' AND "
            "metrics LIKE '%full_indicator_stacker%'").fetchone()[0]
    finally:
        con.close()
    return int(n)


def phase_register() -> None:
    res = results()
    if _already_registered() >= 2:
        print("register: full_indicator_stacker runs already present "
              "— skipping duplicate registration", flush=True)
        guard_after = _guard()
        champs = ("ppo_champion.zip", "stacker_champion.pkl",
                  "stacker_champion.json")
        assert all(guard_after[c] == res["meta"]["guard_before"][c]
                   for c in champs), "CHAMPION GUARD TRIPPED"
        save_result("registered", {"skipped_duplicate": True,
                                   "guard_after": guard_after})
        return
    tr = res["train"]
    acc_c = tr["champion"]["acc"]
    labels = {"A": ("Raw 98", "Arm A — Raw 98: production stacker "
                             "recipe on the matrix-v2 set (59 AZ raw "
                             "+ 31 main-chart = 90 features), "
                             "production labels"),
              "B": ("Scored 102", "Arm B — Scored 102: production "
                                  "stacker recipe on the matrix-v3 "
                                  "set (90 percentile scores + 4 "
                                  "composites = 94 features), "
                                  "production labels")}
    details = []
    for arm in ("A", "B"):
        a = tr[arm]
        label, design = labels[arm]
        cols = (res["build"]["cols90"] if arm == "A"
                else res["build"]["arm_b_cols"])
        win = bool(a["win"])
        reason = (
            f"full-indicator stacker {label} ({len(cols)} features) "
            f"holdout acc {a['acc']:.4f} "
            f"{'beat' if win else 'did not beat'} champion "
            f"{acc_c:.4f} on the shared holdout timestamps "
            f"(full indicator set vs production 21)")
        if win:
            reason += (" — WOULD PROMOTE: held as PENDING KASS "
                       "APPROVAL under the promotion lock; champion "
                       "kept (study verdict only)")
        metrics = {
            "challenger_acc": a["acc"], "champion_acc": acc_c,
            "challenger_wf_acc": a["wf_acc"],
            "challenger_wf_precision": a["wf_precision"],
            "precision": a["precision"],
            "champion_precision": tr["champion"]["precision"],
            "folds": a["folds"], "pending_approval": win,
            "reason": reason, "deep": False,
            "family": "full_indicator_stacker",
            "full_indicator_stacker": {
                "label": label, "arm": arm,
                "verdict": "PASS" if win else "FAIL",
                "design": design, "n_features": len(cols),
                "features": cols, "margin": a["acc"] - acc_c,
                "train_rows": a["train_rows"],
                "base_learner_holdout_acc":
                    a["base_learner_holdout_acc"],
                "top_features": [t["feature"] for t in a["top15"]],
            }}
        log_run("stacker", metrics, False)
        details.append(f"{label}: acc {a['acc']:.4f} vs champion "
                       f"{acc_c:.4f} ({'WIN' if win else 'no win'})")
        print(f"register: arm {arm} logged (promoted=0, "
              f"pending={win})", flush=True)
    add_lab_note(
        "Full-indicator stacker study (family=full_indicator_stacker)",
        "Kass's direction: the production stacker recipe retrained "
        "on the full computable indicator universe, graded vs the "
        "21-feature champion on the shared holdout. "
        + "; ".join(details)
        + f". Champion holdout precision "
          f"{tr['champion']['precision']:.4f}. Study verdict only — "
          f"nothing installed (promotion lock).")
    guard_after = _guard()
    champs = ("ppo_champion.zip", "stacker_champion.pkl",
              "stacker_champion.json")
    assert all(guard_after[c] == res["meta"]["guard_before"][c]
               for c in champs), (
        f"CHAMPION GUARD TRIPPED: {guard_after} vs "
        f"{res['meta']['guard_before']}")
    save_result("registered", {
        "arm_a_win": bool(tr["A"]["win"]),
        "arm_b_win": bool(tr["B"]["win"]),
        "guard_after": guard_after,
        "paper_note": "paper.db is written by the live loop on hourly "
                      "candle closes throughout the study window; "
                      "the study itself never opens it (hash read "
                      "only). Champion files are hash-identical "
                      "before/after."})
    print("register: 2 runs logged (promoted=0); champion guard "
          "intact", flush=True)


PHASES = [("data", phase_data), ("build", phase_build),
          ("train", phase_train), ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== full-indicator stacker study (phases done: "
          f"{sorted(done)}) ===", flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== full-indicator stacker study complete ===", flush=True)


if __name__ == "__main__":
    main()
