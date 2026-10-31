"""Drawing-Board Option C -- the two-step hierarchy (2026-10-24).

Executes refs/hierarchy-plan-2026-10-24.md EXACTLY as
pre-registered -- the plan is the contract; any deviation would be
reported as a deviation, never silently absorbed. Lab only: never
runs the live loop, never writes the paper ledger or live state,
never promotes (every registration is promoted=0; even a full pass
is a report to Kass, not a promotion).

Arms: Hierarchy-Focus (both stages on Kass's focus-11) and
Hierarchy-All (both stages on the focus test's all-60 feature set).
Each arm: stage-1 binary gate (TRADE vs NO-TRADE) + stage-2 binary
direction (UP/DOWN, trained only on true TRADE rows), combined by
the fixed rule (gate fires at P(TRADE) >= 0.50, direction by
stage-2 argmax, tie -> UP). Walk-forward (TimeSeriesSplit 5 folds)
with stage diagnostics, then the duel (standard 3x264h windows +
the older robustness set, refit out-of-sample there) through the
shared signal-policy harness, vs buy-and-hold and the current
champion graded fresh (run 44's PPO brain).

Usage: .venv/bin/python scripts/build_hierarchy.py
Phases checkpoint to data/hierarchy/; reruns skip done phases.
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
from models import hierarchy as hier  # noqa: E402
from models.gbm_policy import build_rule_model  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "hierarchy"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
MATRIX_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"
HA_LAB = ROOT / "data" / "ha_seq"
RULE_ZIP = LAB / "rule_model"

HOLDOUT = 792
WINDOW = 264

# The flat models on identical features, from their own records
# (provenance in refs/hierarchy-plan-2026-10-24.md): Focus B's duel
# figure is the size study's current-code re-grade; Focus D's is
# its recorded duel (no current-code re-grade exists).
FLAT_COMPARISON = {
    "focus": {"label": "Focus B (run 22, flat, focus-11)",
              "wf_accuracy": 0.5508, "wf_balanced_accuracy": 0.4145,
              "wf_majority_baseline": 0.6690, "duel_pnl": -8138.37,
              "duel_pnl_provenance":
                  "size-study re-grade under current production code "
                  "(recorded at the time: -8,432.41, pre-anchor-fix)"},
    "all": {"label": "Focus D (run 24, flat, all-60)",
            "wf_accuracy": 0.5668, "wf_balanced_accuracy": 0.4163,
            "wf_majority_baseline": 0.6690, "duel_pnl": -3178.41,
            "duel_pnl_provenance":
                  "as recorded in run 24 (no current-code re-grade)"},
}


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
def get_matrix() -> pd.DataFrame:
    return gf.load_matrix(MATRIX_PATH)


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
    matrix, sup = get_matrix(), get_sup()
    checks = {
        "matrix_rows_16735": len(matrix) == 16_735,
        "focus_cols_present":
            all(c in matrix.columns for c in gf.FOCUS_FEATURES),
        "all_feature_count_60":
            len(gf.matrix_feature_columns(list(matrix.columns))) == 60,
        "label_counts": bool(
            (matrix[gf.TARGET_COL].value_counts().to_dict()
             == {"HOLD": 10_976, "BUY": 2_915, "SHORT": 2_844})),
        "pre_duel_rows_15961":
            int((matrix.index < duel_start_ts(sup)).sum()) == 15_961,
        "duel_start_matches_plan":
            str(duel_start_ts(sup)) == "2026-09-19 11:00:00",
        "robust_start_matches_plan":
            str(robust_start_ts(sup)) == "2026-08-17 11:00:00",
        "features_no_nan_pre_duel": bool(
            matrix.loc[matrix.index < duel_start_ts(sup),
                       gf.matrix_feature_columns(list(matrix.columns))]
            .notna().all().all()),
    }
    champ_sha = hashlib.sha256(
        (MODELS / "ppo_champion.zip").read_bytes()).hexdigest()
    save_result("meta", {
        "matrix_rows": int(len(matrix)),
        "pre_duel_rows":
            int((matrix.index < duel_start_ts(sup)).sum()),
        "duel_start": str(duel_start_ts(sup)),
        "robust_start": str(robust_start_ts(sup)),
        "lgbm_binary_params": hier.BINARY_PARAMS,
        "gate_threshold": hier.GATE_THRESHOLD,
        "focus_features": gf.FOCUS_FEATURES,
        "champion_file": "models/ppo_champion.zip",
        "champion_sha256": champ_sha,
        "champion_identity": "run 44's PPO brain, installed "
                             "2026-10-24 by Kass's manual override",
        "checks": checks, "integrity_start": _mtimes(),
    })
    log(f"setup: matrix {len(matrix)} rows, checks={checks}")
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_wf(arm: str) -> None:
    matrix, sup = get_matrix(), get_sup()
    pre = matrix.loc[matrix.index < duel_start_ts(sup)]
    rec = hier.walk_forward_arm(pre, arm)
    save_result(f"wf_{arm}", rec)
    p = rec["pooled"]
    d = p["diagnostics"]
    folds = ", ".join(f"{f['balanced_accuracy']:.3f}"
                      for f in rec["folds"])
    log(f"wf_{arm}: acc {p['accuracy']:.4f} vs majority "
        f"{p['majority_baseline']:.4f} | balanced "
        f"{p['balanced_accuracy']:.4f} | folds BA [{folds}] | "
        f"gate BA {d['gate_balanced_accuracy']:.4f} | dir@trade "
        f"{d['stage2_acc_on_true_trade']:.4f} | dir@fired "
        f"{d['stage2_acc_on_fired_true_trade']:.4f} | activity "
        f"{d['activity_share']:.1%}")


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


def _fit_conv(frame: pd.DataFrame, arm: str, fit_mask,
              region_index) -> tuple[dict, dict]:
    """Fit both stages on the fit-masked rows (stage 2 on their
    TRADE subset); conviction over the region rows."""
    cols = hier.arm_features(frame, arm)
    fit_rows = frame.loc[fit_mask]
    model, audit = hier.fit_hierarchy(
        fit_rows[cols].to_numpy(dtype=float),
        fit_rows[gf.TARGET_COL].to_numpy(), ts=fit_rows.index)
    region = frame.loc[region_index]
    conv = hier.hierarchy_conviction(
        model, region[cols].to_numpy(dtype=float))
    return cs.conviction_by_ts(region.index, conv), audit


def _duel_from_conv(windows, conv_by_ts) -> tuple[dict, int]:
    model = _load_rule_model()
    p_ups, missing = cs.p_ups_from_conviction(windows, conv_by_ts)
    return cs.evaluate_windows_composite(windows, p_ups, model), missing


def phase_duel_arm(arm: str) -> None:
    frame, sup = get_matrix(), get_sup()
    start, rstart = duel_start_ts(sup), robust_start_ts(sup)
    conv_std, audit = _fit_conv(
        frame, arm, frame.index < start, frame.index[frame.index >= start])
    res, missing = _duel_from_conv(windows_std(sup), conv_std)
    region_idx = frame.index[(frame.index >= rstart)
                              & (frame.index < start)]
    conv_r, audit_r = _fit_conv(
        frame, arm, frame.index < rstart, region_idx)
    res_r, missing_r = _duel_from_conv(windows_robust(sup), conv_r)
    champ = results()["duel_champion"]["detail"]
    wins = sum(1 for cw, aw in zip(champ["windows"], res["windows"])
               if aw["score"] > cw["score"])
    save_result(f"duel_{arm}", {
        "arm": _flat(res), "detail": res, "windows_won": wins,
        "robust": {"arm": _flat(res_r), "detail": res_r},
        "fit_audit": audit, "robust_fit_audit": audit_r,
        "missing_conviction_rows": missing,
        "missing_conviction_rows_robust": missing_r})
    log(f"duel_{arm}: agg={res['agg_score']:,.2f} pnl="
        f"{res['total_pnl']:+,.2f} gross={res['gross_pnl']:+,.2f} "
        f"fees={res['fees_total']:,.2f} trips={res['total_round_trips']} "
        f"non-hold={res['non_hold_share']:.1%} won {wins}/3 vs champ | "
        f"robust pnl={res_r['total_pnl']:+,.2f}")


def phase_verdicts() -> None:
    res = results()
    out = {}
    for arm in hier.ARMS:
        v = hier.apply_verdict(res[f"wf_{arm}"], res[f"duel_{arm}"]["detail"],
                               res[f"duel_{arm}"]["robust"]["detail"])
        out[arm] = v
        log(f"verdict_{arm}: {v['verdict']} (rule1 WF="
            f"{v['rule1_walk_forward']}, rule2 duel={v['rule2_duel']}, "
            f"rule3 robust={v['rule3_robustness']})")
    save_result("verdicts", out)


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run

    res = results()
    champ = res["duel_champion"]["champion"]
    verdicts = res["verdicts"]
    run_ids = dict(res.get("registered", {}))
    if not run_ids:
        import sqlite3
        con = sqlite3.connect(ROOT / "data" / "training.db")
        for rid, mtxt in con.execute("SELECT id, metrics FROM runs"):
            try:
                m = json.loads(mtxt)
            except Exception:
                continue
            if m.get("family") == "hierarchy":
                run_ids[m["hierarchy"]["arm"]] = rid
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
            f"{metrics['hierarchy']['verdict']}")

    for arm in hier.ARMS:
        wf, duel, v = res[f"wf_{arm}"], res[f"duel_{arm}"], verdicts[arm]
        a, d = duel["arm"], duel["detail"]
        rob = duel["robust"]["detail"]
        p = wf["pooled"]
        dg = p["diagnostics"]
        flat = FLAT_COMPARISON[arm]
        reason = (
            f"Two-step hierarchy test (plan 2026-10-24): "
            f"{hier.ARM_LABELS[arm]} -- stage-1 gate + stage-2 "
            f"direction. WF end-to-end acc {p['accuracy']:.4f} vs "
            f"majority {p['majority_baseline']:.4f} (flat comparison "
            f"{flat['label']}: {flat['wf_accuracy']:.4f}), balanced "
            f"{p['balanced_accuracy']:.4f} (flat "
            f"{flat['wf_balanced_accuracy']:.4f}) -> rule 1 "
            f"{'PASS' if v['rule1_walk_forward'] else 'FAIL'}; gate "
            f"balanced acc {dg['gate_balanced_accuracy']:.4f}, "
            f"stage-2 acc {dg['stage2_acc_on_true_trade']:.4f} on "
            f"true-TRADE rows; duel P&L {a['total_pnl']:+,.2f} "
            f"(gross {d['gross_pnl']:+,.2f}, fees "
            f"{d['fees_total']:,.2f}) vs B&H {a['bench_pnl']:+,.2f}, "
            f"non-hold {d['non_hold_share']:.1%} -> rule 2 "
            f"{'PASS' if v['rule2_duel'] else 'FAIL'}; robustness "
            f"P&L {rob['total_pnl']:+,.2f} -> rule 3 "
            f"{'PASS' if v['rule3_robustness'] else 'FAIL'}. "
            f"Verdict: {v['verdict']} (champion yardstick, run 44 "
            f"brain: agg {champ['score']:.0f}, P&L "
            f"{champ['total_pnl']:+,.2f}; promoted=0 always -- a "
            f"pass is a report to Kass).")
        metrics = {
            "challenger": a, "champion": champ, "reason": reason,
            "family": "hierarchy", "deep": False, "timesteps": 0,
            "reward_curve": None,
            "hierarchy": {
                "arm": arm, "label": hier.ARM_LABELS[arm],
                "design": hier.ARM_DESIGNS[arm],
                "verdict": v["verdict"],
                "verdict_rules": {
                    "rule1_walk_forward": v["rule1_walk_forward"],
                    "rule2_duel": v["rule2_duel"],
                    "rule3_robustness": v["rule3_robustness"]},
                "windows_won": duel["windows_won"],
                "gross_pnl": d["gross_pnl"],
                "fees_total": d["fees_total"],
                "non_hold_share": d["non_hold_share"],
                "robust_pnl": rob["total_pnl"],
                "robust_gross_pnl": rob["gross_pnl"],
                "robust_fees": rob["fees_total"],
                "wf_accuracy": p["accuracy"],
                "wf_balanced_accuracy": p["balanced_accuracy"],
                "wf_majority_baseline": p["majority_baseline"],
                "wf_fold_balanced": v["fold_balanced_accuracies"],
                "diagnostics": dg,
                "flat_comparison": flat,
            },
        }
        _register(arm, metrics,
                  f"{hier.ARM_LABELS[arm]}: WF acc {p['accuracy']:.4f} "
                  f"/ bal {p['balanced_accuracy']:.4f} (gate BA "
                  f"{dg['gate_balanced_accuracy']:.4f}, dir@trade "
                  f"{dg['stage2_acc_on_true_trade']:.4f}); duel P&L "
                  f"{a['total_pnl']:+,.0f} (gross "
                  f"{d['gross_pnl']:+,.0f}); robust "
                  f"{rob['total_pnl']:+,.0f} — **{v['verdict']}**.")

    if res.get("lab_note_done"):
        log("lab note already added, skip")
        return
    lines = ["**Two-step hierarchy test (Drawing-Board C, "
             "2026-10-24)** — Kass's two-step decision split "
             "(stage-1 trade/no-trade gate, stage-2 direction on "
             "TRADE rows only) on the 1h matrix (plan "
             "refs/hierarchy-plan-2026-10-24.md):"]
    for s in summaries:
        lines.append(f"- {s}")
    lines.append("Full report: refs/hierarchy-2026-10-24.md.")
    add_lab_note("Two-step hierarchy test (family=hierarchy)",
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
        ("wf_focus", lambda: phase_wf("focus")),
        ("wf_all", lambda: phase_wf("all")),
        ("duel_champion", phase_duel_champion),
        ("duel_champion_robust", phase_duel_champion_robust),
        ("duel_focus", lambda: phase_duel_arm("focus")),
        ("duel_all", lambda: phase_duel_arm("all")),
        ("verdicts", phase_verdicts),
        ("register", phase_register),
        ("verify", phase_verify),
    ]
    done = progress()
    log(f"=== hierarchy build (phases done: {sorted(done) or 'none'}) ===")
    for name, fn in phases:
        if name in done:
            log(f"{name}: already done -- skipped")
            continue
        t0 = time.time()
        fn()
        mark(name)
        log(f"--- phase {name} done in {time.time() - t0:.0f}s ---")
    log("=== hierarchy build complete ===")


if __name__ == "__main__":
    main()
