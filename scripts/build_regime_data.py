"""Drawing-Board Option B -- daily regime data (2026-10-24).

Executes refs/regime-data-plan-2026-10-24.md EXACTLY as
pre-registered -- the plan is the contract; any deviation would be
reported as a deviation, never silently absorbed. Lab only: never
runs the live loop, never writes the paper ledger or live state,
never promotes (every registration is promoted=0; even a full pass
is a report to Kass, not a promotion).

Arms: GBM-Focus+Regime (focus-11 + 10 daily-regime features) is
graded walk-forward AND in the duel (standard 3x264 windows + the
older robustness set); GBM-Regime-Only is a walk-forward diagnostic;
the F&G Filter (Focus-B-style signal, convictions vetoed by usable
Fear & Greed: LONG zeroed < 25, SHORT zeroed > 75) is duel-only.

Lineage notes (also in the report): the filter arm's standard-window
base signal is run 22's OWN cached probabilities
(data/gbm_focus/probs_B.npz -- bit-identical to its recorded
grading), not a refit; its robustness signal is refit on rows
strictly before the robustness region, per the plan. The
unfiltered-base duel graded here is an integrity reference (not an
arm, not registered): under current production code it must land on
the size study's reproduction of Focus B (agg -3,329.80), NOT run
22's pre-stop-anchor-fix record (-3,476.80).

Usage: .venv/bin/python scripts/build_regime_data.py
Phases checkpoint to data/regime_data/; reruns skip done phases.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models import composite_screen as cs  # noqa: E402
from models import gbm_focus as gf  # noqa: E402
from models import regime_data as rd  # noqa: E402
from models.gbm_policy import build_rule_model, conviction_np  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "regime_data"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
MATRIX_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"
FOCUS_PROBS_B = ROOT / "data" / "gbm_focus" / "probs_B.npz"
HA_LAB = ROOT / "data" / "ha_seq"
RULE_ZIP = LAB / "rule_model"

HOLDOUT = 792
WINDOW = 264

# Run 22 (Focus B) as recorded, and as the size study reproduced it
# under current (post-stop-anchor-fix) production code.
FOCUS_B_RECORDED = {"agg": -3476.80, "pnl": -8432.41, "bench": 717.54}
FOCUS_B_CURRENT_CODE = {"agg": -3329.80}
# Focus B's recorded walk-forward (run 22) -- arm 1's comparison.
FOCUS_B_WF = {"accuracy": 0.5508, "balanced_accuracy": 0.4145,
              "majority_baseline": 0.6690}


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


def _mtimes() -> dict:
    out = {}
    for p in (MODELS / "ppo_champion.zip", MODELS / "stacker_champion.pkl",
              MODELS / "stacker_champion.json", ROOT / "data" / "paper.db",
              ROOT / "data" / "live_state.json"):
        out[p.name] = p.stat().st_mtime if p.exists() else None
    return out


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data
_FRAME = None


def get_frame() -> pd.DataFrame:
    """The v1 matrix joined with the 10 causal regime features
    (cached to a pickle inside the lab dir after first build)."""
    global _FRAME
    if _FRAME is not None:
        return _FRAME
    cache = LAB / "frame.pkl"
    if cache.exists():
        _FRAME = pd.read_pickle(cache)
        return _FRAME
    matrix = gf.load_matrix(MATRIX_PATH)
    frame = rd.make_feature_frame(matrix, rd.build_daily_features())
    LAB.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache)
    _FRAME = frame
    return _FRAME


def get_sup() -> pd.DataFrame:
    return pd.read_pickle(HA_LAB / "sup.pkl")


def windows_std(sup) -> list:
    split = len(sup) - HOLDOUT
    return [sup.iloc[split + k * WINDOW: split + (k + 1) * WINDOW]
            for k in range(3)]


def windows_robust(sup) -> list:
    split = len(sup) - 2 * HOLDOUT
    return [sup.iloc[split + k * WINDOW: split + (k + 1) * WINDOW]
            for k in range(3)]


def duel_start_ts(sup) -> pd.Timestamp:
    return sup.index[len(sup) - HOLDOUT]


def robust_start_ts(sup) -> pd.Timestamp:
    return sup.index[len(sup) - 2 * HOLDOUT]


def stacker_p_ups(windows) -> list:
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    return [stacker_model.predict_proba(w[FEATURE_COLUMNS])[:, 1]
            for w in windows]


def _flat(res: dict) -> dict:
    ws = res["windows"]
    return {"score": res["agg_score"], "total_pnl": res["total_pnl"],
            "mean_reward": float(np.mean([w["mean_reward"] for w in ws])),
            "max_drawdown": res["max_drawdown"],
            "bench_pnl": res["bench_pnl"],
            "action_mix": res["action_mix"],
            "windows": [{"score": round(w["score"], 2),
                         "pnl": round(w["total_pnl"], 2),
                         "bench": round(w["bench_pnl"], 2)} for w in ws],
            "runnable": True}


def _rule_zip() -> Path:
    zip_file = LAB / "rule_model.zip"
    if not zip_file.exists():
        build_rule_model().save(RULE_ZIP)  # SB3 appends .zip
    assert zip_file.exists()
    return zip_file


def _load_rule_model():
    from stable_baselines3 import PPO
    return PPO.load(_rule_zip())


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    frame, sup = get_frame(), get_sup()
    raw_fng = rd.load_daily_raw("fng")
    last = frame.index[-1]
    checks = {
        "matrix_rows_16735": len(frame) == 16_735,
        "regime_cols_exact": list(
            frame.columns[-len(rd.REGIME_FEATURES):]) == rd.REGIME_FEATURES,
        "regime_zero_nan": bool(
            frame[rd.REGIME_FEATURES].notna().all().all()),
        "fng_level_in_0_100": bool(
            frame["fng_level"].between(0, 100).all()),
        "pre_duel_rows_15961":
            int((frame.index < duel_start_ts(sup)).sum()) == 15_961,
        "duel_start_matches_plan":
            str(duel_start_ts(sup)) == "2026-09-19 11:00:00",
        "robust_start_matches_plan":
            str(robust_start_ts(sup)) == "2026-08-17 11:00:00",
        # D+1: the final matrix day reads the Oct 20 stamp, not 21st.
        "fng_tail_is_d_plus_1": bool(
            frame.loc[last, "fng_level"]
            == raw_fng.loc[pd.Timestamp("2026-10-20")]),
    }
    coverage = {}
    for key in rd.SERIES:
        raw = rd.load_daily_raw(key)
        in_span = raw[(raw.index >= pd.Timestamp("2024-11-23"))
                      & (raw.index <= pd.Timestamp("2026-10-21"))]
        coverage[key] = {
            "fetched_span": f"{raw.index.min().date()} -> "
                            f"{raw.index.max().date()}",
            "fetched_rows": int(len(raw)),
            "stamps_in_matrix_span": int(len(in_span)),
            "last_stamp_le_span_end":
                str(raw[raw.index <= pd.Timestamp("2026-10-21")]
                    .index.max().date()),
            "lag_days": rd.SERIES[key][1],
        }
    champ_sha = hashlib.sha256(
        (MODELS / "ppo_champion.zip").read_bytes()).hexdigest()
    save_result("meta", {
        "matrix_rows": int(len(frame)),
        "pre_duel_rows":
            int((frame.index < duel_start_ts(sup)).sum()),
        "duel_start": str(duel_start_ts(sup)),
        "robust_start": str(robust_start_ts(sup)),
        "lgbm_params": gf.LGBM_PARAMS,
        "regime_features": rd.REGIME_FEATURES,
        "coverage": coverage,
        "champion_file": "models/ppo_champion.zip",
        "champion_sha256": champ_sha,
        "champion_identity": "run 44's PPO brain, installed "
                             "2026-10-24 by Kass's manual override",
        "checks": checks, "integrity_start": _mtimes(),
    })
    log(f"setup: frame {len(frame)} rows, coverage={coverage}, "
        f"checks={checks}")
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_wf(arm: str) -> None:
    frame, sup = get_frame(), get_sup()
    pre = frame.loc[frame.index < duel_start_ts(sup)]
    cols = (rd.FOCUS_FEATURES + rd.REGIME_FEATURES
            if arm == "focus_regime" else rd.REGIME_FEATURES)
    rec = rd.walk_forward_features(pre, cols)
    rec["arm"] = arm
    rec["features"] = cols
    save_result(f"wf_{arm}", rec)
    p = rec["pooled"]
    folds = ", ".join(f"{f['balanced_accuracy']:.3f}"
                      for f in rec["folds"])
    log(f"wf_{arm}: acc {p['accuracy']:.4f} vs majority "
        f"{p['majority_baseline']:.4f} | balanced "
        f"{p['balanced_accuracy']:.4f} | folds BA [{folds}]")


def phase_duel_champion() -> None:
    from training.nightly import evaluate_windows

    sup = get_sup()
    windows = windows_std(sup)
    p_ups = stacker_p_ups(windows)
    from stable_baselines3 import PPO
    champ_model = PPO.load(MODELS / "ppo_champion.zip")
    mine = cs.evaluate_windows_composite(windows, p_ups, champ_model)
    theirs = evaluate_windows(MODELS / "ppo_champion.zip", windows, p_ups)
    assert theirs is not None, "champion failed to grade (nightly)"
    cross = {
        "my_agg": mine["agg_score"], "nightly_agg": theirs["agg_score"],
        "my_pnl": mine["total_pnl"], "nightly_pnl": theirs["total_pnl"],
        "my_bench": mine["bench_pnl"],
        "nightly_bench": float(sum(w["bench_pnl"]
                                    for w in theirs["windows"])),
    }
    cross["agg_diff"] = abs(cross["my_agg"] - cross["nightly_agg"])
    cross["pnl_diff"] = abs(cross["my_pnl"] - cross["nightly_pnl"])
    save_result("duel_champion", {"champion": _flat(mine),
                                  "detail": mine, "crosscheck": cross})
    log(f"duel_champion: agg={mine['agg_score']:,.2f} "
        f"pnl={mine['total_pnl']:+,.2f} gross={mine['gross_pnl']:+,.2f} "
        f"fees={mine['fees_total']:,.2f} bench={mine['bench_pnl']:+,.2f} "
        f"| mirror crosscheck agg_diff={cross['agg_diff']:.4f} "
        f"pnl_diff={cross['pnl_diff']:.4f}")
    assert cross["agg_diff"] < 0.01 and cross["pnl_diff"] < 0.01, \
        f"mirror disagrees with nightly: {cross}"


def phase_duel_champion_robust() -> None:
    from stable_baselines3 import PPO

    sup = get_sup()
    windows = windows_robust(sup)
    p_ups = stacker_p_ups(windows)
    champ_model = PPO.load(MODELS / "ppo_champion.zip")
    res = cs.evaluate_windows_composite(windows, p_ups, champ_model)
    save_result("duel_champion_robust", {"champion": _flat(res),
                                         "detail": res})
    log(f"duel_champion_robust: agg={res['agg_score']:,.2f} "
        f"pnl={res['total_pnl']:+,.2f} bench={res['bench_pnl']:+,.2f}")


def _fit_conv(frame: pd.DataFrame, feature_cols: list[str],
              fit_mask, region_index) -> tuple[dict, dict]:
    """Fit arm-B on the fit-masked rows; conviction over the region
    rows. Returns (conv_by_ts, audit)."""
    fit_rows = frame.loc[fit_mask]
    model, audit = gf.fit_arm_final(
        fit_rows[feature_cols].to_numpy(dtype=float),
        fit_rows[gf.TARGET_COL].to_numpy(), "B")
    region = frame.loc[region_index]
    proba = model.predict_proba(
        region[feature_cols].to_numpy(dtype=float))
    conv = conviction_np(proba[:, 0], proba[:, 2])
    return cs.conviction_by_ts(region.index, conv), audit


def _duel_from_conv(windows, conv_by_ts) -> tuple[dict, int]:
    model = _load_rule_model()
    p_ups, missing = cs.p_ups_from_conviction(windows, conv_by_ts)
    return cs.evaluate_windows_composite(windows, p_ups, model), missing


def phase_duel_base_check() -> None:
    """Integrity reference (NOT an arm): run 22's own cached
    probabilities, unfiltered, through today's mirror on the
    standard windows. Must land on the size study's current-code
    reproduction of Focus B (agg -3,329.80), not the recorded
    pre-anchor-fix -3,476.80."""
    frame, sup = get_frame(), get_sup()
    start = duel_start_ts(sup)
    z = np.load(FOCUS_PROBS_B)
    prob_ts = pd.DatetimeIndex(z["ts"])
    proba = z["proba"]
    region = frame.loc[frame.index >= start]
    assert prob_ts.equals(region.index), \
        "cached probs_B timestamps != matrix duel region"
    conv = conviction_np(proba[:, 0], proba[:, 2])
    res, missing = _duel_from_conv(
        windows_std(sup), cs.conviction_by_ts(prob_ts, conv))
    rec = {"detail": res, "missing_conviction_rows": missing,
           "recorded_run22": FOCUS_B_RECORDED,
           "expected_current_code_agg": FOCUS_B_CURRENT_CODE["agg"],
           "agg_diff_vs_current_code":
               abs(res["agg_score"] - FOCUS_B_CURRENT_CODE["agg"])}
    save_result("duel_base_check", rec)
    log(f"duel_base_check: agg={res['agg_score']:,.2f} (current-code "
        f"expectation {FOCUS_B_CURRENT_CODE['agg']:,.2f}, recorded "
        f"run 22 {FOCUS_B_RECORDED['agg']:,.2f}) pnl="
        f"{res['total_pnl']:+,.2f} bench={res['bench_pnl']:+,.2f}")
    assert rec["agg_diff_vs_current_code"] < 1.0, \
        f"base lineage check failed: {rec['agg_diff_vs_current_code']}"


def phase_duel_focus_regime() -> None:
    frame, sup = get_frame(), get_sup()
    cols = rd.FOCUS_FEATURES + rd.REGIME_FEATURES
    start, rstart = duel_start_ts(sup), robust_start_ts(sup)
    conv_std, audit = _fit_conv(
        frame, cols, frame.index < start, frame.index[frame.index >= start])
    res, missing = _duel_from_conv(windows_std(sup), conv_std)
    region_idx = frame.index[(frame.index >= rstart)
                             & (frame.index < start)]
    conv_r, audit_r = _fit_conv(
        frame, cols, frame.index < rstart, region_idx)
    res_r, missing_r = _duel_from_conv(windows_robust(sup), conv_r)
    save_result("duel_focus_regime", {
        "arm": _flat(res), "detail": res,
        "robust": {"arm": _flat(res_r), "detail": res_r},
        "fit_audit": audit, "robust_fit_audit": audit_r,
        "missing_conviction_rows": missing,
        "missing_conviction_rows_robust": missing_r})
    log(f"duel_focus_regime: agg={res['agg_score']:,.2f} pnl="
        f"{res['total_pnl']:+,.2f} gross={res['gross_pnl']:+,.2f} "
        f"fees={res['fees_total']:,.2f} trips={res['total_round_trips']} "
        f"non-hold={res['non_hold_share']:.1%} | robust pnl="
        f"{res_r['total_pnl']:+,.2f}")


def phase_duel_filter() -> None:
    frame, sup = get_frame(), get_sup()
    start, rstart = duel_start_ts(sup), robust_start_ts(sup)
    # Standard: base = run 22's cached probabilities (bit-identical).
    z = np.load(FOCUS_PROBS_B)
    prob_ts = pd.DatetimeIndex(z["ts"])
    conv_base = conviction_np(z["proba"][:, 0], z["proba"][:, 2])
    assert prob_ts.equals(frame.index[frame.index >= start])
    fng_std = frame.loc[prob_ts, "fng_level"].to_numpy(dtype=float)
    conv_f, vetoes = rd.apply_fng_filter(conv_base, fng_std)
    res, missing = _duel_from_conv(
        windows_std(sup), cs.conviction_by_ts(prob_ts, conv_f))
    # Robustness: base refit on rows strictly before the region.
    region_idx = frame.index[(frame.index >= rstart)
                             & (frame.index < start)]
    conv_base_r_map, audit_r = _fit_conv(
        frame, rd.FOCUS_FEATURES, frame.index < rstart, region_idx)
    conv_base_r = np.array([conv_base_r_map[t] for t in region_idx])
    fng_r = frame.loc[region_idx, "fng_level"].to_numpy(dtype=float)
    conv_fr, vetoes_r = rd.apply_fng_filter(conv_base_r, fng_r)
    res_r, missing_r = _duel_from_conv(
        windows_robust(sup), cs.conviction_by_ts(region_idx, conv_fr))
    save_result("duel_filter", {
        "arm": _flat(res), "detail": res,
        "robust": {"arm": _flat(res_r), "detail": res_r},
        "vetoes_std": vetoes, "vetoes_robust": vetoes_r,
        "base_active_std": int((conv_base != 0).sum()),
        "base_active_robust": int((conv_base_r != 0).sum()),
        "robust_fit_audit": audit_r,
        "missing_conviction_rows": missing,
        "missing_conviction_rows_robust": missing_r})
    log(f"duel_filter: agg={res['agg_score']:,.2f} pnl="
        f"{res['total_pnl']:+,.2f} gross={res['gross_pnl']:+,.2f} "
        f"fees={res['fees_total']:,.2f} trips={res['total_round_trips']} "
        f"non-hold={res['non_hold_share']:.1%} vetoes={vetoes} | "
        f"robust pnl={res_r['total_pnl']:+,.2f} vetoes={vetoes_r}")


def phase_verdicts() -> None:
    res = results()
    v1 = rd.apply_verdict_arm1(res["wf_focus_regime"],
                               res["duel_focus_regime"]["detail"],
                               res["duel_focus_regime"]["robust"]
                               ["detail"])
    r1_only = rd.rule1_walk_forward(res["wf_regime_only"])
    v2 = "WF ONLY" if r1_only else "FAIL"
    v3 = rd.apply_verdict_arm3(res["duel_filter"]["detail"],
                               res["duel_filter"]["robust"]["detail"])
    out = {"focus_regime": v1,
           "regime_only": {"verdict": v2, "rule1_walk_forward": r1_only,
                           "rule2_duel": None, "rule3_robustness": None,
                           "fold_balanced_accuracies":
                               [f["balanced_accuracy"] for f in
                                res["wf_regime_only"]["folds"]]},
           "filter": v3}
    save_result("verdicts", out)
    for arm, v in out.items():
        log(f"verdict_{arm}: {v['verdict']} (rule1={v['rule1_walk_forward']}"
            f", rule2={v['rule2_duel']}, rule3={v['rule3_robustness']})")


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run

    res = results()
    champ = res["duel_champion"]["champion"]
    verdicts = res["verdicts"]
    run_ids = dict(res.get("registered", {}))
    if not run_ids:
        # Resume guard (tf4h pattern): seed from training.db so a
        # resume never double-registers.
        import sqlite3
        con = sqlite3.connect(ROOT / "data" / "training.db")
        for rid, mtxt in con.execute("SELECT id, metrics FROM runs"):
            try:
                m = json.loads(mtxt)
            except Exception:
                continue
            if m.get("family") == "regime_data":
                run_ids[m["regime_data"]["arm"]] = rid
        con.close()
        if run_ids:
            log(f"register: seeded existing runs from db: {run_ids}")

    summaries = []

    def _register(arm: str, metrics: dict, summary: str) -> None:
        if arm in run_ids:
            log(f"register {arm}: already run {run_ids[arm]}, skip")
            summaries.append(summary)
            return
        log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            rid = int(con.execute("SELECT MAX(id) FROM runs")
                      .fetchone()[0])
        run_ids[arm] = rid
        res["registered"] = run_ids
        save_result("registered", run_ids)
        summaries.append(summary)
        log(f"registered {arm} as run {rid}: "
            f"{metrics['regime_data']['verdict']}")

    # -- arm 1: Focus+Regime (WF + duel) --
    arm = "focus_regime"
    wf, duel, v = (res["wf_focus_regime"], res["duel_focus_regime"],
                   verdicts[arm])
    a, d = duel["arm"], duel["detail"]
    rob = duel["robust"]["detail"]
    p = wf["pooled"]
    reason = (
        f"Regime data test (plan 2026-10-24): Focus+Regime on "
        f"focus-11 + 10 daily-regime features (F&G D+1, FRED D+2 "
        f"usable lags). WF acc {p['accuracy']:.4f} vs majority "
        f"{p['majority_baseline']:.4f} (Focus B recorded "
        f"{FOCUS_B_WF['accuracy']:.4f}), balanced "
        f"{p['balanced_accuracy']:.4f} (Focus B "
        f"{FOCUS_B_WF['balanced_accuracy']:.4f}) -> rule 1 "
        f"{'PASS' if v['rule1_walk_forward'] else 'FAIL'}; duel P&L "
        f"{a['total_pnl']:+,.2f} (gross {d['gross_pnl']:+,.2f}, fees "
        f"{d['fees_total']:,.2f}) vs B&H {a['bench_pnl']:+,.2f}, "
        f"non-hold {d['non_hold_share']:.1%} -> rule 2 "
        f"{'PASS' if v['rule2_duel'] else 'FAIL'}; robustness P&L "
        f"{rob['total_pnl']:+,.2f} -> rule 3 "
        f"{'PASS' if v['rule3_robustness'] else 'FAIL'}. Verdict: "
        f"{v['verdict']} (champion yardstick, run 44 brain: agg "
        f"{champ['score']:.0f}, P&L {champ['total_pnl']:+,.2f}; "
        f"promoted=0 always -- a pass is a report to Kass).")
    metrics = {
        "challenger": a, "champion": champ, "reason": reason,
        "family": "regime_data", "deep": False, "timesteps": 0,
        "reward_curve": None,
        "regime_data": {
            "arm": arm, "label": rd.ARM_LABELS[arm],
            "design": rd.ARM_DESIGNS[arm], "verdict": v["verdict"],
            "verdict_rules": {
                "rule1_walk_forward": v["rule1_walk_forward"],
                "rule2_duel": v["rule2_duel"],
                "rule3_robustness": v["rule3_robustness"]},
            "gross_pnl": d["gross_pnl"], "fees_total": d["fees_total"],
            "non_hold_share": d["non_hold_share"],
            "robust_pnl": rob["total_pnl"],
            "robust_gross_pnl": rob["gross_pnl"],
            "robust_fees": rob["fees_total"],
            "wf_accuracy": p["accuracy"],
            "wf_balanced_accuracy": p["balanced_accuracy"],
            "wf_majority_baseline": p["majority_baseline"],
            "wf_fold_balanced": v["fold_balanced_accuracies"],
            "focus_b_wf_recorded": FOCUS_B_WF,
        },
    }
    _register(arm, metrics,
              f"Focus+Regime: WF acc {p['accuracy']:.4f} / bal "
              f"{p['balanced_accuracy']:.4f}; duel P&L "
              f"{a['total_pnl']:+,.0f} (gross {d['gross_pnl']:+,.0f}); "
              f"robust {rob['total_pnl']:+,.0f} — **{v['verdict']}**.")

    # -- arm 2: Regime Only (WF diagnostic) --
    arm = "regime_only"
    wf, v = res["wf_regime_only"], verdicts[arm]
    p = wf["pooled"]
    reason = (
        f"Regime data diagnostic (plan 2026-10-24): Regime Only on "
        f"the 10 daily-regime features alone -- walk-forward ONLY "
        f"by pre-registered design (never duel-graded, so a lucky "
        f"diagnostic cannot be cherry-picked). WF acc "
        f"{p['accuracy']:.4f} vs majority {p['majority_baseline']:.4f}, "
        f"balanced {p['balanced_accuracy']:.4f} (fold spread "
        f"{min(f['balanced_accuracy'] for f in wf['folds']):.3f}-"
        f"{max(f['balanced_accuracy'] for f in wf['folds']):.3f}) -> "
        f"walk-forward rule "
        f"{'PASS (WF-only, not a tradable claim)' if v['rule1_walk_forward'] else 'FAIL'}; "
        f"lab only, promoted=0.")
    metrics = {
        "challenger": {}, "champion": champ, "reason": reason,
        "family": "regime_data", "deep": False, "timesteps": 0,
        "reward_curve": None,
        "regime_data": {
            "arm": arm, "label": rd.ARM_LABELS[arm],
            "design": rd.ARM_DESIGNS[arm], "verdict": v["verdict"],
            "verdict_rules": {
                "rule1_walk_forward": v["rule1_walk_forward"],
                "rule2_duel": None, "rule3_robustness": None},
            "wf_accuracy": p["accuracy"],
            "wf_balanced_accuracy": p["balanced_accuracy"],
            "wf_majority_baseline": p["majority_baseline"],
            "wf_fold_balanced": v["fold_balanced_accuracies"],
        },
    }
    _register(arm, metrics,
              f"Regime Only (WF diagnostic): acc {p['accuracy']:.4f} "
              f"/ bal {p['balanced_accuracy']:.4f} — **{v['verdict']}**.")

    # -- arm 3: F&G Filter (duel only) --
    arm = "filter"
    duel, v = res["duel_filter"], verdicts[arm]
    a, d = duel["arm"], duel["detail"]
    rob = duel["robust"]["detail"]
    base = res["duel_base_check"]["detail"]
    reason = (
        f"Regime data test (plan 2026-10-24): F&G Filter -- Focus-B "
        f"signal (standard windows: run 22's cached probabilities, "
        f"bit-identical) with convictions vetoed by usable F&G "
        f"(LONG zeroed < 25, SHORT zeroed > 75; "
        f"{duel['vetoes_std']} vetoes standard / "
        f"{duel['vetoes_robust']} robustness). Duel P&L "
        f"{a['total_pnl']:+,.2f} (gross {d['gross_pnl']:+,.2f}, fees "
        f"{d['fees_total']:,.2f}) vs B&H {a['bench_pnl']:+,.2f}, "
        f"non-hold {d['non_hold_share']:.1%} -> rule 2 "
        f"{'PASS' if v['rule2_duel'] else 'FAIL'}; robustness P&L "
        f"{rob['total_pnl']:+,.2f} -> rule 3 "
        f"{'PASS' if v['rule3_robustness'] else 'FAIL'}. Verdict: "
        f"{v['verdict']} (unfiltered base under current code: agg "
        f"{base['agg_score']:.0f}, P&L {base['total_pnl']:+,.2f}; "
        f"promoted=0 always).")
    metrics = {
        "challenger": a, "champion": champ, "reason": reason,
        "family": "regime_data", "deep": False, "timesteps": 0,
        "reward_curve": None,
        "regime_data": {
            "arm": arm, "label": rd.ARM_LABELS[arm],
            "design": rd.ARM_DESIGNS[arm], "verdict": v["verdict"],
            "verdict_rules": {
                "rule1_walk_forward": None,
                "rule2_duel": v["rule2_duel"],
                "rule3_robustness": v["rule3_robustness"]},
            "gross_pnl": d["gross_pnl"], "fees_total": d["fees_total"],
            "non_hold_share": d["non_hold_share"],
            "robust_pnl": rob["total_pnl"],
            "robust_gross_pnl": rob["gross_pnl"],
            "robust_fees": rob["fees_total"],
            "vetoes_std": duel["vetoes_std"],
            "vetoes_robust": duel["vetoes_robust"],
            "base_unfiltered_agg_current_code": base["agg_score"],
            "base_unfiltered_pnl_current_code": base["total_pnl"],
        },
    }
    _register(arm, metrics,
              f"F&G Filter: duel P&L {a['total_pnl']:+,.0f} (gross "
              f"{d['gross_pnl']:+,.0f}); robust {rob['total_pnl']:+,.0f}"
              f" — **{v['verdict']}**.")

    if res.get("lab_note_done"):
        log("lab note already added, skip")
        return
    lines = ["**Daily regime data test (Drawing-Board B, 2026-10-24)**"
             " — daily F&G (D+1) + FRED VIX/USD/T10Y2Y/DFF (D+2) as "
             "context on the 1h frame (plan "
             "refs/regime-data-plan-2026-10-24.md):"]
    for s in summaries:
        lines.append(f"- {s}")
    lines.append("Full report: refs/regime-data-2026-10-24.md.")
    add_lab_note("Daily regime data test (family=regime_data)",
                 "\n".join(lines))
    res["lab_note_done"] = True
    _save_json(RESULTS_PATH, res)
    log("register: lab note added")


def phase_verify() -> None:
    start = results()["meta"]["integrity_start"]
    now = _mtimes()
    champ_same = all(start[k] == now[k] for k in
                     ("ppo_champion.zip", "stacker_champion.pkl",
                      "stacker_champion.json"))
    save_result("verify", {"champion_mtimes_unchanged": bool(champ_same),
                           "start": start, "end": now,
                           "note": "paper.db / live_state.json are "
                                   "written by the LIVE loop (bot "
                                   "un-parked 2026-10-24); this lab "
                                   "study never opened them for "
                                   "writing -- all duel brokers used "
                                   "scratch DBs in /tmp."})
    log(f"verify: champion mtimes unchanged = {champ_same}")
    assert champ_same, f"INTEGRITY: champion mtimes changed {start} -> {now}"


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    phases = [
        ("setup", phase_setup),
        ("wf_focus_regime", lambda: phase_wf("focus_regime")),
        ("wf_regime_only", lambda: phase_wf("regime_only")),
        ("duel_champion", phase_duel_champion),
        ("duel_champion_robust", phase_duel_champion_robust),
        ("duel_base_check", phase_duel_base_check),
        ("duel_focus_regime", phase_duel_focus_regime),
        ("duel_filter", phase_duel_filter),
        ("verdicts", phase_verdicts),
        ("register", phase_register),
        ("verify", phase_verify),
    ]
    done = progress()
    log(f"=== regime data build (phases done: {sorted(done) or 'none'}) ===")
    for name, fn in phases:
        if name in done:
            log(f"{name}: already done -- skipped")
            continue
        t0 = time.time()
        fn()
        mark(name)
        log(f"--- phase {name} done in {time.time() - t0:.0f}s ---")
    log("=== regime data build complete ===")


if __name__ == "__main__":
    main()
