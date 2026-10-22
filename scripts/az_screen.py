"""A-Z indicator screening program (2026-10-22, Kass's directive:
"include all these features to test").

Trading is PARKED: this script is lab-only. It never runs the live
loop, never touches the paper ledger, never promotes or registers
anything as a challenger. SCREENING ONLY: it measures whether any
single indicator from the A-Z encyclopedia carries (i) solo signal,
(ii) marginal value on top of the production 21 features, and
(iii) non-redundant information -- under multiple-comparison
discipline, so that the graduates list (possibly empty) is defensible
from the table alone.

Method mirrors refs/diagnosis-2026-10-21.md exactly: same cached
supervised frame (data/diagnosis/sup.pkl, 17,049 rows), same
production split (train = all but the last 792 rows), same stacker
pipeline (stacker.build_pipeline, seed 42), same walk-forward grading
(TimeSeriesSplit, 5 folds, on the train frame).

Per indicator:
- solo WF: single-feature stacker; per-fold accuracies plus POOLED
  out-of-fold predictions -> pooled accuracy and a binomial p-value
  against the OOF majority-class rate (the honest coin-flip null).
- Spearman IC vs next-candle return (and direction) per fold and
  pooled, as a model-free second angle.
- marginal WF: full-21 stacker vs full-21 + indicator, computed on
  the SAME train rows (paired folds -> delta with SE). The baseline
  is recomputed per distinct row set and cached.
- redundancy: max |Pearson r| of the scaled series vs each of the 21
  production features and vs every other screened series (train
  rows). > 0.95 = duplicate-class (the obv_slope/netvol_rel disease,
  r = 0.999).

Multiple comparisons: with N indicators screened, the expected number
of false positives at a raw alpha of 0.05 is 0.05 * N by construction.
Graduation therefore requires the solo binomial p-value to beat the
Bonferroni threshold 0.05 / N (BH q-values are reported alongside),
a walk-forward mean accuracy >= 0.51 with pooled OOF accuracy above
the majority rate, max |r| vs the 21 <= 0.95, and a marginal paired
delta >= 0.

Resumable: phases and per-indicator results are checkpointed to
data/az_screen/{progress,results}.json; the raw and scaled indicator
frames are cached as pickles so a restart never recomputes or
shifts anything. Safe to relaunch.

Usage: .venv/bin/python scripts/az_screen.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.az_indicators import AZ_COLUMNS, REGISTRY, compute_az  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models import stacker  # noqa: E402

LAB = ROOT / "data" / "az_screen"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
SUP_PATH = ROOT / "data" / "diagnosis" / "sup.pkl"
RAW_PATH = LAB / "raw.pkl"
SCALED_PATH = LAB / "scaled.pkl"
SEED = 42
HOLDOUT = 792


# ---------------------------------------------------------------- state
def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
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


# ---------------------------------------------------------------- data
def get_sup() -> pd.DataFrame:
    sup = pd.read_pickle(SUP_PATH)
    assert len(sup) == 17049, f"sup rows {len(sup)} != 17049 (diagnosis)"
    return sup


def split_frames(sup: pd.DataFrame):
    split = len(sup) - HOLDOUT
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min()
    return train_sup, hold_sup


def get_frames():
    """(raw, scaled) AZ frames over the sup OHLCV, scalers fitted on
    the train rows only; cached so restarts never shift values."""
    if RAW_PATH.exists() and SCALED_PATH.exists():
        return pd.read_pickle(RAW_PATH), pd.read_pickle(SCALED_PATH)
    sup = get_sup()
    train_sup, _ = split_frames(sup)
    ohlcv = sup[["open", "high", "low", "close", "volume"]]
    scaled, raw, scalers = compute_az(ohlcv, len(train_sup))
    LAB.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(RAW_PATH)
    scaled.to_pickle(SCALED_PATH)
    save_result("scalers", scalers)
    return raw, scaled


# ------------------------------------------------------- grading helpers
def wf_folds(X: pd.DataFrame, y: pd.Series):
    """Mirror of stacker.walk_forward_score, but returns per-fold
    accuracies AND the pooled out-of-fold predictions (positions are
    positional within X)."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    tss = TimeSeriesSplit(n_splits=5)
    fold_accs, oof_pos, oof_pred = [], [], []
    for tr_idx, te_idx in tss.split(X):
        pipe = stacker.build_pipeline(SEED)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        fold_accs.append(float(accuracy_score(y.iloc[te_idx], pred)))
        oof_pos.append(te_idx)
        oof_pred.append(pred)
    return fold_accs, np.concatenate(oof_pos), np.concatenate(oof_pred)


def _mean_se(xs) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs)))


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    import sklearn
    import scipy

    import pandas_ta  # noqa: F401  (route documentation)

    sup = get_sup()
    train_sup, hold_sup = split_frames(sup)
    raw, scaled = get_frames()
    train_scaled = scaled.iloc[: len(train_sup)]
    checks = {
        "sup_rows_match_diagnosis": len(sup) == 17049,
        "train_holdout_disjoint": bool(
            train_sup.index.max() < hold_sup.index.min()),
        "scaled_within_contract": bool(
            np.nanmin(train_scaled.to_numpy()) >= -1.0
            and np.nanmax(train_scaled.to_numpy()) <= 1.0),
        "n_indicators_ge_100": len(AZ_COLUMNS) >= 100,
        "all_series_usable_on_train": bool(
            (train_scaled.notna().mean() > 0.5).all()),
    }
    save_result("meta", {
        "route": "pandas_ta 0.4.71b0 wrappers + hand implementations "
                 "(TA-Lib not installed; see az_indicators docstring)",
        "versions": {"sklearn": sklearn.__version__,
                     "scipy": scipy.__version__,
                     "pandas": pd.__version__,
                     "numpy": np.__version__},
        "sup_rows": len(sup), "train_rows": len(train_sup),
        "holdout_rows": len(hold_sup),
        "sup_start": str(sup.index.min()), "sup_end": str(sup.index.max()),
        "n_indicators": len(AZ_COLUMNS),
        "train_finite_frac_min": float(train_scaled.notna().mean().min()),
        "checks": checks,
    })
    print(f"setup: {len(AZ_COLUMNS)} indicators, checks={checks}",
          flush=True)
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_corr() -> None:
    sup = get_sup()
    train_sup, _ = split_frames(sup)
    _, scaled = get_frames()
    n = len(train_sup)
    comb = pd.concat(
        [train_sup[FEATURE_COLUMNS].reset_index(drop=True),
         scaled.iloc[:n].reset_index(drop=True)], axis=1)
    corr = comb.corr(method="pearson")
    out = {}
    az = list(AZ_COLUMNS)
    for name in az:
        vs21 = corr.loc[name, FEATURE_COLUMNS].abs()
        vsaz = corr.loc[name, az].abs().drop(name)
        out[name] = {
            "max_corr_21": float(vs21.max()),
            "argmax_21": str(vs21.idxmax()),
            "max_corr_az": float(vsaz.max()),
            "argmax_az": str(vsaz.idxmax()),
        }
    save_result("corr", out)
    save_result("corr_matrix_az", {
        "cols": az,
        "values": corr.loc[az, az].round(4).to_numpy().tolist(),
    })
    top = sorted(out.items(), key=lambda kv: -kv[1]["max_corr_21"])[:5]
    print("corr: top vs-21 overlaps: " + ", ".join(
        f"{k}~{v['argmax_21']}({v['max_corr_21']:.3f})" for k, v in top),
        flush=True)


def _baseline_for(rows_key: str, X21: pd.DataFrame, y: pd.Series) -> dict:
    cache = results().get("baselines", {})
    if rows_key in cache:
        return cache[rows_key]
    fold_accs, _, _ = wf_folds(X21, y)
    mean, se = _mean_se(fold_accs)
    rec = {"fold_accs": fold_accs, "mean": mean, "se": se,
           "n_rows": int(len(y))}
    cache[rows_key] = rec
    save_result("baselines", cache)
    return rec


def phase_screen_one(name: str) -> None:
    from scipy.stats import binomtest, spearmanr

    sup = get_sup()
    train_sup, _ = split_frames(sup)
    _, scaled = get_frames()
    y_all = train_sup["target"]
    ind_all = scaled[name].iloc[: len(train_sup)]
    mask = ind_all.notna().to_numpy()
    rows = np.where(mask)[0]
    rows_key = hashlib.sha1(rows.tobytes()).hexdigest()[:16]
    sub_idx = train_sup.index[rows]
    y = y_all.loc[sub_idx]
    ind = ind_all.loc[sub_idx]
    assert (ind.notna()).all()
    next_ret = (train_sup["close"].shift(-1) / train_sup["close"]
                 - 1.0).loc[sub_idx]

    t0 = time.time()
    # (i) solo WF + pooled OOF + binomial vs the OOF majority rate
    fold_accs, oof_pos, oof_pred = wf_folds(ind.to_frame(), y)
    y_oof = y.iloc[oof_pos].to_numpy()
    oof_acc = float((oof_pred == y_oof).mean())
    p0 = float(y_oof.mean())
    p0 = max(p0, 1.0 - p0)  # majority-class rate on the OOF segment
    correct = int((oof_pred == y_oof).sum())
    binom_p = float(binomtest(correct, len(y_oof), p0,
                              alternative="two-sided").pvalue)
    wf_mean, wf_se = _mean_se(fold_accs)
    # (i-bis) Spearman IC per fold (same splits) + pooled
    from sklearn.model_selection import TimeSeriesSplit
    ic_folds, ic_dir_folds = [], []
    for _, te_idx in TimeSeriesSplit(n_splits=5).split(ind):
        te = np.asarray(te_idx)
        ok = np.isfinite(next_ret.iloc[te].to_numpy())
        r_ret = spearmanr(ind.iloc[te][ok],
                          next_ret.iloc[te][ok]).statistic
        r_dir = spearmanr(ind.iloc[te], y.iloc[te]).statistic
        ic_folds.append(float(r_ret))
        ic_dir_folds.append(float(r_dir))
    ic_mean, ic_se = _mean_se(ic_folds)
    ok_all = np.isfinite(next_ret.to_numpy())
    pooled = spearmanr(ind[ok_all], next_ret[ok_all])
    # (ii) marginal: baseline (cached per row set) vs 21 + indicator
    X21 = train_sup[FEATURE_COLUMNS].loc[sub_idx]
    base = _baseline_for(rows_key, X21, y)
    plus_accs, _, _ = wf_folds(
        pd.concat([X21, ind], axis=1), y)
    deltas = [p - b for p, b in zip(plus_accs, base["fold_accs"])]
    d_mean, d_se = _mean_se(deltas)
    corr = results().get("corr", {}).get(name, {})
    rec = {
        "family": REGISTRY[name]["family"],
        "source": REGISTRY[name]["source"],
        "n_train_rows": int(len(rows)),
        "finite_frac_train": float(len(rows) / len(train_sup)),
        "solo_fold_accs": fold_accs, "solo_wf_acc": wf_mean,
        "solo_wf_se": wf_se,
        "oof_n": int(len(y_oof)), "oof_acc": oof_acc,
        "oof_majority_rate": p0, "binom_p": binom_p,
        "ic_folds": ic_folds, "ic_mean": ic_mean, "ic_se": ic_se,
        "ic_dir_mean": float(np.mean(ic_dir_folds)),
        "ic_pooled": float(pooled.statistic),
        "ic_pooled_p": float(pooled.pvalue),
        "baseline_wf_acc": base["mean"],
        "plus_wf_acc": float(np.mean(plus_accs)),
        "plus_fold_accs": plus_accs,
        "marginal_delta": d_mean, "marginal_delta_se": d_se,
        "marginal_pos_folds": int(sum(1 for d in deltas if d > 0)),
        "max_corr_21": corr.get("max_corr_21"),
        "argmax_21": corr.get("argmax_21"),
        "max_corr_az": corr.get("max_corr_az"),
        "argmax_az": corr.get("argmax_az"),
        "seconds": round(time.time() - t0, 1),
    }
    scr = results().get("screen", {})
    scr[name] = rec
    save_result("screen", scr)
    print(f"screen {name}: solo_wf={wf_mean:.4f} oof={oof_acc:.4f} "
          f"p={binom_p:.2e} ic={ic_mean:+.4f} delta={d_mean:+.4f} "
          f"maxr21={rec['max_corr_21']:.3f} ({rec['seconds']:.0f}s)",
          flush=True)


def _bh_qvalues(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values (step-up), plain implementation."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def phase_analysis() -> None:
    res = results()
    scr = res["screen"]
    names = sorted(scr)
    n = len(names)
    bonf = 0.05 / n
    qvals = _bh_qvalues([scr[k]["binom_p"] for k in names])
    ic_q = _bh_qvalues([scr[k]["ic_pooled_p"] for k in names])
    graduates, table = [], {}
    for i, name in enumerate(names):
        r = scr[name]
        sig = r["binom_p"] < bonf
        dup = r["max_corr_21"] is not None and r["max_corr_21"] > 0.95
        grad = bool(sig and r["solo_wf_acc"] >= 0.51
                    and r["oof_acc"] > r["oof_majority_rate"]
                    and not dup and r["marginal_delta"] >= 0)
        verdict = ("GRADUATE" if grad else
                   "duplicate of production feature" if dup else
                   "not significant after Bonferroni" if not sig else
                   "significant but fails effect/marginal rule")
        table[name] = {**r, "binom_q_bh": qvals[i],
                       "ic_q_bh": ic_q[i], "bonferroni_sig": bool(sig),
                       "duplicate_21": bool(dup), "verdict": verdict}
        if grad:
            graduates.append(name)
    # clusters among the screened set at |r| > 0.90 (union-find)
    mat = res["corr_matrix_az"]
    cols, vals = mat["cols"], np.asarray(mat["values"], dtype=float)
    parent = list(range(len(cols)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(vals[i, j]) > 0.90:
                parent[find(i)] = find(j)
    clusters: dict[int, list[str]] = {}
    for i, cname in enumerate(cols):
        clusters.setdefault(find(i), []).append(cname)
    big = sorted([sorted(v) for v in clusters.values() if len(v) > 1],
                 key=len, reverse=True)
    save_result("table", table)
    save_result("analysis", {
        "n_tested": n,
        "bonferroni_alpha": bonf,
        "expected_false_positives_raw_0.05": 0.05 * n,
        "n_raw_significant": int(sum(
            1 for k in names if scr[k]["binom_p"] < 0.05)),
        "n_bonferroni_significant": int(sum(
            1 for k in names if scr[k]["binom_p"] < bonf)),
        "graduates": graduates,
        "clusters_abs_r_gt_0.90": big,
    })
    print(f"analysis: n={n} raw_sig="
          f"{sum(1 for k in names if scr[k]['binom_p'] < 0.05)} "
          f"bonf_sig={sum(1 for k in names if scr[k]['binom_p'] < bonf)} "
          f"graduates={graduates}", flush=True)


PHASES = [("setup", phase_setup), ("corr", phase_corr)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== az screen (phases done: {sorted(done) or 'none'}) ===",
          flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done -- skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"--- phase {name} done in {time.time() - t0:.0f}s ---",
              flush=True)
    scr_done = set(results().get("screen", {}))
    for name in AZ_COLUMNS:
        key = f"screen:{name}"
        if key in progress():
            continue
        if name in scr_done:
            mark(key)
            continue
        phase_screen_one(name)
        mark(key)
    if "analysis" not in progress():
        phase_analysis()
        mark("analysis")
    print("=== az screen complete ===", flush=True)


if __name__ == "__main__":
    main()
