"""Matrix v3 composite screen (2026-10-24, Kass green-lit).

Executes refs/composite-screen-plan-v3-2026-10-23.md EXACTLY as
pre-registered -- the plan is the contract; any deviation would be
reported as a deviation, never silently absorbed. Lab only: never
runs the live loop, never writes the paper ledger or live state,
never promotes (every registration is promoted=0; even a full pass
is a report to Kass, not a promotion).

Arms: Rule-Master (fixed 60/40 on score_master) and GBM-4
(LightGBM, gbm_focus arm-B treatment, 4 composites) are graded
walk-forward AND in the duel (standard 3x264 windows + the older
robustness set); Rule-Trend / Rule-Momentum / Rule-Volume are
walk-forward diagnostics only, by design.

The duel grades the CURRENT champion fresh through the same
harness: as of this run the champion file is run 44's PPO brain,
installed 2026-10-24 by Kass's manual override. The mirror driver
(models.composite_screen.evaluate_windows_composite) is cross-
checked against training.nightly.evaluate_windows on the champion
and must agree before any arm number is trusted.

Usage: .venv/bin/python scripts/build_composite_screen.py
Phases checkpoint to data/composite_screen/; reruns skip done phases.
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
from models.gbm_policy import build_rule_model, conviction_np  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "composite_screen"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
V3_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v3.csv"
V2_PATH = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v2.csv"
HA_LAB = ROOT / "data" / "ha_seq"
RULE_ZIP = LAB / "rule_model"

HOLDOUT = 792
WINDOW = 264


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


# ---------------------------------------------------------------- data
def get_v3() -> pd.DataFrame:
    return cs.load_v3(V3_PATH)


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
    """Flatten a duel record into the nightly run-record shape."""
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
    v3, sup = get_v3(), get_sup()
    v2 = pd.read_csv(V2_PATH, parse_dates=["Date"],
                     usecols=["Date", cs.TARGET_COL, "Open", "High",
                              "Low", "Close"])
    v2 = v2.set_index("Date").sort_index()
    mask_pre = np.asarray(v3.index < duel_start_ts(sup))
    checks = {
        "v3_rows_16735": len(v3) == 16_735,
        "v3_index_equals_v2": bool(v3.index.equals(v2.index)),
        "labels_identical_to_v2": bool(
            (v3[cs.TARGET_COL].to_numpy()
             == v2[cs.TARGET_COL].to_numpy()).all()),
        "ohlc_identical_to_v2": all(
            np.array_equal(v3[c].to_numpy(dtype=float),
                           v2[c].to_numpy(dtype=float))
            for c in ("Open", "High", "Low", "Close")),
        "composites_complete_in_range": bool(
            v3[cs.COMPOSITE_COLS].notna().all().all()
            and v3[cs.COMPOSITE_COLS].min().min() >= 0.0
            and v3[cs.COMPOSITE_COLS].max().max() <= 100.0),
        "pre_duel_rows_15961": int(mask_pre.sum()) == 15_961,
        "duel_start_matches_plan": str(duel_start_ts(sup))
        == "2026-09-19 11:00:00",
    }
    champ_sha = hashlib.sha256(
        (MODELS / "ppo_champion.zip").read_bytes()).hexdigest()
    save_result("meta", {
        "v3_rows": int(len(v3)),
        "pre_duel_rows": int(mask_pre.sum()),
        "duel_start": str(duel_start_ts(sup)),
        "robust_start": str(robust_start_ts(sup)),
        "lgbm_params": gf.LGBM_PARAMS,
        "champion_file": "models/ppo_champion.zip",
        "champion_sha256": champ_sha,
        "champion_identity": "run 44's PPO brain, installed "
                             "2026-10-24 by Kass's manual override",
        "checks": checks, "integrity_start": _mtimes(),
    })
    print(f"setup: v3 {len(v3)} rows, pre-duel {int(mask_pre.sum())}, "
          f"duel from {duel_start_ts(sup)}, robust from "
          f"{robust_start_ts(sup)}, checks={checks}", flush=True)
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_wf(arm: str) -> None:
    v3, sup = get_v3(), get_sup()
    pre = v3.loc[v3.index < duel_start_ts(sup)]
    rec = cs.walk_forward_arm(pre, arm)
    save_result(f"wf_{arm}", rec)
    p = rec["pooled"]
    folds = ", ".join(f"{f['balanced_accuracy']:.3f}"
                      for f in rec["folds"])
    print(f"wf_{arm}: acc {p['accuracy']:.4f} vs majority "
          f"{p['majority_baseline']:.4f} | balanced "
          f"{p['balanced_accuracy']:.4f} | folds BA [{folds}]",
          flush=True)


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
    rec = {"champion": _flat(mine), "detail": mine, "crosscheck": cross}
    save_result("duel_champion", rec)
    print(f"duel_champion: agg={mine['agg_score']:,.2f} "
          f"pnl={mine['total_pnl']:+,.2f} gross="
          f"{mine['gross_pnl']:+,.2f} fees={mine['fees_total']:,.2f} "
          f"bench={mine['bench_pnl']:+,.2f} | mirror crosscheck "
          f"agg_diff={cross['agg_diff']:.4f} pnl_diff="
          f"{cross['pnl_diff']:.4f}", flush=True)
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
    print(f"duel_champion_robust: agg={res['agg_score']:,.2f} "
          f"pnl={res['total_pnl']:+,.2f} bench={res['bench_pnl']:+,.2f}",
          flush=True)


def _arm_conviction_std(arm: str):
    """(conv_by_ts, fit_audit) for an arm on the standard duel
    region, from the v3 rows at/after the duel start."""
    v3, sup = get_v3(), get_sup()
    start = duel_start_ts(sup)
    if arm == "rule_master":
        duel_rows = v3.loc[v3.index >= start]
        conv = cs.rule_conviction(
            duel_rows["score_master"].to_numpy(dtype=float))
        return cs.conviction_by_ts(duel_rows.index, conv), {"fit": "none"}
    if arm == "gbm4":
        cache = LAB / "probs_gbm4.npz"
        pre = v3.loc[v3.index < start]
        if cache.exists():
            z = np.load(cache)
            prob_ts, proba = z["ts"], z["proba"]
            audit = {"cached": True}
        else:
            model, audit = gf.fit_arm_final(
                pre[cs.COMPOSITE_COLS].to_numpy(dtype=float),
                pre[cs.TARGET_COL].to_numpy(), "B")
            duel_rows = v3.loc[v3.index >= start]
            proba = model.predict_proba(
                duel_rows[cs.COMPOSITE_COLS].to_numpy(dtype=float))
            prob_ts = duel_rows.index.to_numpy()
            np.savez(cache, ts=prob_ts, proba=proba)
        conv = conviction_np(proba[:, 0], proba[:, 2])
        return cs.conviction_by_ts(prob_ts, conv), audit
    raise ValueError(arm)


def _arm_conviction_robust(arm: str):
    """Same, for the robustness region [robust_start, duel_start).
    GBM-4 is REFIT on rows strictly before the region (out-of-sample
    there), mirroring the lag build's robustness discipline."""
    v3, sup = get_v3(), get_sup()
    rstart, dend = robust_start_ts(sup), duel_start_ts(sup)
    region = v3.loc[(v3.index >= rstart) & (v3.index < dend)]
    if arm == "rule_master":
        conv = cs.rule_conviction(
            region["score_master"].to_numpy(dtype=float))
        return cs.conviction_by_ts(region.index, conv), {"fit": "none"}
    if arm == "gbm4":
        cache = LAB / "probs_gbm4_robust.npz"
        if cache.exists():
            z = np.load(cache)
            prob_ts, proba = z["ts"], z["proba"]
            audit = {"cached": True}
        else:
            pre = v3.loc[v3.index < rstart]
            model, audit = gf.fit_arm_final(
                pre[cs.COMPOSITE_COLS].to_numpy(dtype=float),
                pre[cs.TARGET_COL].to_numpy(), "B")
            proba = model.predict_proba(
                region[cs.COMPOSITE_COLS].to_numpy(dtype=float))
            prob_ts = region.index.to_numpy()
            np.savez(cache, ts=prob_ts, proba=proba)
        conv = conviction_np(proba[:, 0], proba[:, 2])
        return cs.conviction_by_ts(prob_ts, conv), audit
    raise ValueError(arm)


def phase_duel_arm(arm: str) -> None:
    sup = get_sup()
    model = _load_rule_model()
    conv_std, audit = _arm_conviction_std(arm)
    p_ups, missing = cs.p_ups_from_conviction(windows_std(sup), conv_std)
    res = cs.evaluate_windows_composite(windows_std(sup), p_ups, model)
    conv_r, audit_r = _arm_conviction_robust(arm)
    p_ups_r, missing_r = cs.p_ups_from_conviction(
        windows_robust(sup), conv_r)
    res_r = cs.evaluate_windows_composite(
        windows_robust(sup), p_ups_r, model)
    champ = results()["duel_champion"]["detail"]
    wins = sum(1 for cw, aw in zip(champ["windows"], res["windows"])
               if aw["score"] > cw["score"])
    rec = {"arm": _flat(res), "detail": res, "windows_won": wins,
           "robust": {"arm": _flat(res_r), "detail": res_r},
           "fit_audit": audit, "robust_fit_audit": audit_r,
           "missing_conviction_rows": missing,
           "missing_conviction_rows_robust": missing_r}
    save_result(f"duel_{arm}", rec)
    print(f"duel_{arm}: agg={res['agg_score']:,.2f} pnl="
          f"{res['total_pnl']:+,.2f} gross={res['gross_pnl']:+,.2f} "
          f"fees={res['fees_total']:,.2f} trips="
          f"{res['total_round_trips']} won {wins}/3 | robust pnl="
          f"{res_r['total_pnl']:+,.2f}", flush=True)


def phase_verdicts() -> None:
    res = results()
    champ = res["duel_champion"]["detail"]
    out = {}
    for arm in cs.ARMS_DUEL:
        v = cs.apply_verdict(res[f"wf_{arm}"], res[f"duel_{arm}"]["detail"],
                             res[f"duel_{arm}"]["robust"]["detail"], champ)
        out[arm] = v
        print(f"verdict_{arm}: {v['verdict']} (rule1 WF="
              f"{v['rule1_walk_forward']}, rule2 duel="
              f"{v['rule2_duel_gate']}, rule3 robust="
              f"{v['rule3_robustness']}; formula={v['formula_pass']}, "
              f"floor_fail={v['floor_fail']})", flush=True)
    save_result("verdicts", out)


def _rule1(wf: dict) -> bool:
    p = wf["pooled"]
    return bool(p["balanced_accuracy"] > 1.0 / 3.0
                and sum(1 for f in wf["folds"]
                        if f["balanced_accuracy"] > 1.0 / 3.0) >= 4
                and p["accuracy"] >= p["majority_baseline"])


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run

    res = results()
    champ = res["duel_champion"]["champion"]
    champ_detail = res["duel_champion"]["detail"]
    verdicts = res["verdicts"]
    run_ids = {}
    summaries = []
    for arm in ("rule_master", "gbm4"):
        wf, duel, v = res[f"wf_{arm}"], res[f"duel_{arm}"], verdicts[arm]
        a, d = duel["arm"], duel["detail"]
        rob = duel["robust"]["detail"]
        rules = (f"verdict rules -- walk-forward "
                 f"{'PASS' if v['rule1_walk_forward'] else 'FAIL'}, "
                 f"duel+floor "
                 f"{'PASS' if v['rule2_duel_gate'] else 'FAIL'}"
                 f"{'' if v['floor_fail'] is None else ' (' + v['floor_fail'] + ')'}, "
                 f"robustness "
                 f"{'PASS' if v['rule3_robustness'] else 'FAIL'}")
        reason = (
            f"Composite screen challenger (family=composite_screen, "
            f"arm {cs.ARM_LABELS[arm]}: {cs.ARM_DESIGNS[arm]}): "
            f"screen {v['verdict']} -- duel won {duel['windows_won']}/3 "
            f"windows, aggregate {a['score']:.0f} vs champion "
            f"{champ['score']:.0f} (run 44 brain, manual override), "
            f"P&L ${a['total_pnl']:+,.0f} (gross "
            f"${d['gross_pnl']:+,.0f}, fees ${d['fees_total']:,.0f}), "
            f"{d['total_round_trips']} round trips, non-hold "
            f"{d['non_hold_share']:.1%} (buy-and-hold same windows: "
            f"${champ['bench_pnl']:+,.0f}); robustness P&L "
            f"${rob['total_pnl']:+,.0f}; walk-forward acc "
            f"{wf['pooled']['accuracy']:.4f} vs majority baseline "
            f"{wf['pooled']['majority_baseline']:.4f}, balanced acc "
            f"{wf['pooled']['balanced_accuracy']:.4f}; {rules}; "
            f"lab only, promoted=0")
        metrics = {
            "challenger": a, "champion": champ, "reason": reason,
            "family": "composite_screen", "deep": False, "timesteps": 0,
            "reward_curve": None,
            "composite_screen": {
                "arm": arm, "label": cs.ARM_LABELS[arm],
                "design": cs.ARM_DESIGNS[arm],
                "verdict": v["verdict"],
                "verdict_rules": {
                    "rule1_walk_forward": v["rule1_walk_forward"],
                    "rule2_duel_gate": v["rule2_duel_gate"],
                    "rule3_robustness": v["rule3_robustness"]},
                "windows_won": duel["windows_won"],
                "formula_pass": v["formula_pass"],
                "floor_fail": v["floor_fail"],
                "gross_pnl": d["gross_pnl"],
                "fees_total": d["fees_total"],
                "non_hold_share": d["non_hold_share"],
                "robust_pnl": rob["total_pnl"],
                "robust_gross_pnl": rob["gross_pnl"],
                "wf_accuracy": wf["pooled"]["accuracy"],
                "wf_majority_baseline": wf["pooled"]["majority_baseline"],
                "wf_balanced_accuracy": wf["pooled"]["balanced_accuracy"],
                "wf_fold_balanced": v["fold_balanced_accuracies"],
            },
        }
        log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        run_ids[arm] = int(run_id)
        summaries.append(
            f"{cs.ARM_LABELS[arm]}: {v['verdict']} -- agg "
            f"{a['score']:.0f} vs champ {champ['score']:.0f}, P&L "
            f"${a['total_pnl']:+,.0f} (gross ${d['gross_pnl']:+,.0f}), "
            f"robust P&L ${rob['total_pnl']:+,.0f}, WF acc "
            f"{wf['pooled']['accuracy']:.4f} vs "
            f"{wf['pooled']['majority_baseline']:.4f} (run {run_id}).")
        print(f"register: {arm} -> run {run_id} (promoted=0)", flush=True)
    for arm in ("rule_trend", "rule_momentum", "rule_volume"):
        wf = res[f"wf_{arm}"]
        p = wf["pooled"]
        r1 = _rule1(wf)
        reason = (
            f"Composite screen diagnostic (family=composite_screen, "
            f"arm {cs.ARM_LABELS[arm]}: {cs.ARM_DESIGNS[arm]}): "
            f"walk-forward ONLY by pre-registered design (never "
            f"duel-graded, so a lucky family cannot be cherry-picked) "
            f"-- acc {p['accuracy']:.4f} vs majority baseline "
            f"{p['majority_baseline']:.4f}, balanced acc "
            f"{p['balanced_accuracy']:.4f} (fold spread "
            f"{min(f['balanced_accuracy'] for f in wf['folds']):.3f}-"
            f"{max(f['balanced_accuracy'] for f in wf['folds']):.3f}), "
            f"walk-forward rule "
            f"{'would pass' if r1 else 'FAIL'}; lab only, promoted=0")
        metrics = {
            "challenger": {}, "champion": champ, "reason": reason,
            "family": "composite_screen", "deep": False, "timesteps": 0,
            "reward_curve": None,
            "composite_screen": {
                "arm": arm, "label": cs.ARM_LABELS[arm],
                "design": cs.ARM_DESIGNS[arm],
                "verdict": "WF ONLY",
                "wf_accuracy": p["accuracy"],
                "wf_majority_baseline": p["majority_baseline"],
                "wf_balanced_accuracy": p["balanced_accuracy"],
                "wf_fold_balanced": [f["balanced_accuracy"]
                                     for f in wf["folds"]],
                "wf_rule1_would_pass": r1,
            },
        }
        log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        run_ids[arm] = int(run_id)
        summaries.append(
            f"{cs.ARM_LABELS[arm]} (WF diagnostic): acc "
            f"{p['accuracy']:.4f} vs {p['majority_baseline']:.4f}, "
            f"balanced {p['balanced_accuracy']:.4f} (run {run_id}).")
        print(f"register: {arm} -> run {run_id} (promoted=0)", flush=True)
    note = ("Matrix v3 composite screen (pre-registered plan "
            "refs/composite-screen-plan-v3-2026-10-23.md; features = "
            "the 4 v3 composites only): graded walk-forward + duel "
            f"(standard + robustness windows) vs the current champion "
            f"(run 44 brain; agg {champ['score']:.0f}, P&L "
            f"${champ['total_pnl']:+,.0f}, buy-and-hold "
            f"${champ['bench_pnl']:+,.0f}). " + " ".join(summaries) +
            " Full report: refs/composite-screen-2026-10-24.md")
    add_lab_note("Matrix v3 composite screen (family=composite_screen)",
                 note)
    save_result("register", {"run_ids": run_ids, "lab_note": note})
    print("register: lab note added", flush=True)


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
                                   "screen never opened them for "
                                   "writing -- all duel brokers used "
                                   "scratch DBs in /tmp."})
    print(f"verify: champion mtimes unchanged = {champ_same}", flush=True)
    assert champ_same, f"INTEGRITY: champion mtimes changed {start} -> {now}"


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    phases = [("setup", phase_setup)]
    phases += [(f"wf_{a}", (lambda a: lambda: phase_wf(a))(a))
               for a in cs.ARMS_WF]
    phases += [("duel_champion", phase_duel_champion),
               ("duel_champion_robust", phase_duel_champion_robust)]
    phases += [(f"duel_{a}", (lambda a: lambda: phase_duel_arm(a))(a))
               for a in cs.ARMS_DUEL]
    phases += [("verdicts", phase_verdicts),
               ("register", phase_register),
               ("verify", phase_verify)]
    done = progress()
    print(f"=== composite screen build (phases done: "
          f"{sorted(done) or 'none'}) ===", flush=True)
    for name, fn in phases:
        if name in done:
            print(f"{name}: already done -- skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"--- phase {name} done in {time.time() - t0:.0f}s ---",
              flush=True)
    print("=== composite screen build complete ===", flush=True)


if __name__ == "__main__":
    main()
