"""Exit & transition study (2026-10-22, Kass's request).

Hold the SIGNAL fixed (GBM Lag F, training.db run 26) and vary ONLY
the exit + transition machinery. See src/models/exit_study.py for the
mechanics and refs/exit-study-2026-10-22.md for the write-up.

Grid (pre-registered):
  Stage 1 (exits, T0 through-flat): stop_mult {1.5, 2.0, 3.0} x
    take_profit {off, +4%, +8%, 2R} x trailing {current, off, tight}
    = 36 cells.
  Stage 2 (transitions): T0/T1/T2 x {default recipe, top-3 Stage-1
    configs by standard-window duel aggregate} -- T0 cells repeat
    Stage-1 numbers as controls; the 8 T1/T2 cells are new.
Every cell is graded on the standard duel windows (last 792h,
3x264h) AND a robustness set (the preceding 792h, same layout; Lag F
probabilities there come from the same recipe fit only on earlier
rows -- see exit_study.load_lagf_signal).

Lab only: promoted=0 everywhere; production artifacts untouched.

Usage: .venv/bin/python scripts/build_exit_study.py
Phases checkpoint to data/exit_study/; reruns skip completed cells.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.exit_study import (ExitConfig, evaluate_windows_study,  # noqa: E402
                               load_lagf_signal)

LAB = ROOT / "data" / "exit_study"
RESULTS_PATH = LAB / "results.json"
MODELS = ROOT / "models"

PROTECTED = [MODELS / "ppo_champion.zip", MODELS / "stacker_champion.pkl",
             MODELS / "stacker_champion.json", ROOT / "data" / "paper.db",
             ROOT / "data" / "live_state.json",
             ROOT / "src" / "trading" / "broker.py",
             ROOT / "src" / "environment" / "trading_env.py",
             ROOT / "src" / "training" / "nightly.py"]


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
    return obj


def results() -> dict:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_result(key, value) -> None:
    res = results()
    res[key] = value
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(res), indent=1))
    tmp.replace(RESULTS_PATH)


def _mtimes() -> dict:
    return {p.name: (p.stat().st_mtime if p.exists() else None)
            for p in PROTECTED}


# ------------------------------------------------------------------ grid
def stage1_configs() -> list[ExitConfig]:
    tps = [("off", None, 0.0), ("pct4", "pct", 0.04),
           ("pct8", "pct", 0.08), ("R2", "R", 2.0)]
    out = []
    for stop in (1.5, 2.0, 3.0):
        for _name, mode, val in tps:
            for trail in ("current", "off", "tight"):
                out.append(ExitConfig(stop_mult=stop, tp_mode=mode,
                                       tp_value=val, trail=trail,
                                       transition="T0"))
    return out


def cell_key(cfg: ExitConfig) -> str:
    return cfg.label()


def run_cell(cfg: ExitConfig, signals: dict) -> dict:
    rec = {"config": {"stop_mult": cfg.stop_mult, "tp_mode": cfg.tp_mode,
                      "tp_value": cfg.tp_value, "trail": cfg.trail,
                      "transition": cfg.transition},
           "label": cfg.label(), "short_label": cfg.short_label()}
    for name in ("standard", "robustness"):
        windows, p_ups = signals[name]
        res = evaluate_windows_study(windows, p_ups, cfg)
        rec[name] = res
        print(f"  {name}: agg {res['agg_score']:,.2f} pnl "
              f"{res['total_pnl']:+,.2f} (B&H {res['bench_pnl']:+,.2f}) "
              f"trips {res['total_round_trips']} closed "
              f"{res['closed_trips']} exits {res['exit_counts']}",
              flush=True)
    rec["mean_agg"] = float(np.mean([rec["standard"]["agg_score"],
                                      rec["robustness"]["agg_score"]]))
    return rec


def get_signals() -> dict:
    sig = {}
    for name in ("standard", "robustness"):
        windows, p_ups = load_lagf_signal(name)
        sig[name] = (windows, p_ups)
        print(f"signal[{name}]: {len(windows)} windows x "
              f"{len(windows[0])} rows", flush=True)
    return sig


# ---------------------------------------------------------------- phases
def phase_setup(signals) -> None:
    if "meta" in results():
        return
    save_result("meta", {"integrity_start": _mtimes(),
                         "signal": "GBM Lag F (training.db run 26)",
                         "windows": "standard = last 792h (3x264h); "
                                    "robustness = preceding 792h"})
    print("setup: integrity baseline recorded", flush=True)


def phase_stage1(signals) -> None:
    res = results()
    done = res.get("stage1", {})
    for cfg in stage1_configs():
        key = cell_key(cfg)
        if key in done:
            print(f"stage1 {key}: cached", flush=True)
            continue
        print(f"stage1 {key}: running...", flush=True)
        t0 = time.time()
        done[key] = run_cell(cfg, signals)
        res = results()
        res["stage1"] = done
        save_result("stage1", done)
        print(f"--- {key} in {time.time() - t0:.0f}s ---", flush=True)


def _cfg_from_rec(rec: dict, transition: str) -> ExitConfig:
    c = rec["config"]
    return ExitConfig(stop_mult=c["stop_mult"], tp_mode=c["tp_mode"],
                      tp_value=c["tp_value"], trail=c["trail"],
                      transition=transition)


def phase_stage2(signals) -> None:
    res = results()
    if "stage2" in res:
        print("stage2: cached", flush=True)
        return
    stage1 = res["stage1"]
    ranked = sorted(stage1.values(),
                    key=lambda r: r["standard"]["agg_score"],
                    reverse=True)
    top3 = ranked[:3]
    default_key = cell_key(ExitConfig())
    bases, seen = [], set()
    for b in [stage1[default_key]] + top3:
        if b["label"] not in seen:
            bases.append(b)
            seen.add(b["label"])
    print("stage2 bases (default + top-3 stage-1 by standard agg):",
          [b["label"] for b in bases], flush=True)
    cells = {}
    for base in bases:
        for t in ("T0", "T1", "T2"):
            cfg = _cfg_from_rec(base, t)
            key = cell_key(cfg)
            if t == "T0":
                cells[key] = stage1[cell_key(_cfg_from_rec(base, "T0"))]
                cells[key] = dict(cells[key])
                cells[key]["control_from_stage1"] = True
                print(f"stage2 {key}: control (stage-1 numbers)",
                      flush=True)
                continue
            print(f"stage2 {key}: running...", flush=True)
            cells[key] = run_cell(cfg, signals)
    save_result("stage2", {"cells": cells,
                           "base_labels": [b["label"] for b in bases]})
    print("stage2 done", flush=True)


def phase_recommend() -> dict:
    res = results()
    pool = {}
    for key, rec in res["stage1"].items():
        pool[key] = rec
    for key, rec in res["stage2"]["cells"].items():
        pool[key] = rec
    ranked = sorted(pool.values(), key=lambda r: r["mean_agg"],
                    reverse=True)
    default = pool[cell_key(ExitConfig())]
    best = ranked[0]
    rec = {"criterion": "highest mean of (standard agg, robustness agg) "
                        "across all 44 evaluated cells",
           "recommended_label": best["label"],
           "recommended": best,
           "default": default,
           "top10": [{"label": r["label"], "mean_agg": r["mean_agg"],
                      "std_agg": r["standard"]["agg_score"],
                      "rob_agg": r["robustness"]["agg_score"],
                      "std_pnl": r["standard"]["total_pnl"],
                      "rob_pnl": r["robustness"]["total_pnl"]}
                     for r in ranked[:10]]}
    save_result("recommendation", rec)
    print(f"recommendation: {best['label']} (mean agg "
          f"{best['mean_agg']:,.2f} vs default {default['mean_agg']:,.2f})",
          flush=True)
    return rec


def _flatten(res: dict) -> dict:
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


def phase_register() -> None:
    res = results()
    if "register" in res:
        print("register: cached", flush=True)
        return
    from training.nightly import add_lab_note, log_run

    rec = res["recommendation"]
    best_label = rec["recommended_label"]
    default_std = rec["default"]["standard"]
    lag = json.loads((ROOT / "data" / "gbm_lag" / "results.json")
                     .read_text())
    reference = lag["duel_F"]["arm"]  # run 26's recorded record

    finalists = []
    for key, cell in res["stage2"]["cells"].items():
        if cell["config"]["transition"] in ("T1", "T2"):
            finalists.append(cell)
    to_register = list(finalists)
    if all(c["label"] != best_label for c in to_register):
        pool = dict(res["stage1"])
        pool.update(res["stage2"]["cells"])
        to_register.append(pool[best_label])

    run_ids = {}
    for cell in to_register:
        std, rob = cell["standard"], cell["robustness"]
        is_best = cell["label"] == best_label
        verdict = "BEST RECIPE" if is_best else "FINALIST"
        d_agg = std["agg_score"] - default_std["agg_score"]
        d_pnl = std["total_pnl"] - default_std["total_pnl"]
        reason = (
            f"Exit & transition study (family=exit_study; signal fixed "
            f"= GBM Lag F, run 26): config {cell['label']} -- standard "
            f"windows agg {std['agg_score']:.0f} / P&L "
            f"${std['total_pnl']:+,.0f} vs production-exit default agg "
            f"{default_std['agg_score']:.0f} / P&L "
            f"${default_std['total_pnl']:+,.0f} (delta agg {d_agg:+.0f}, "
            f"P&L {d_pnl:+,.0f}); robustness windows agg "
            f"{rob['agg_score']:.0f} / P&L ${rob['total_pnl']:+,.0f} "
            f"(B&H {rob['bench_pnl']:+,.0f}); {verdict}; lab only, "
            f"promoted=0 -- recipe adoption is Kass's decision")
        metrics = {
            "challenger": _flatten(std), "champion": reference,
            "reason": reason, "family": "exit_study", "deep": False,
            "timesteps": 0, "reward_curve": None,
            "exit_study": {
                "label": cell["short_label"], "verdict": verdict,
                "config": cell["config"],
                "std_agg": std["agg_score"], "std_pnl": std["total_pnl"],
                "rob_agg": rob["agg_score"], "rob_pnl": rob["total_pnl"],
                "mean_agg": cell["mean_agg"],
                "closed_trips": std["closed_trips"],
                "exit_counts": std["exit_counts"],
            },
        }
        log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        run_ids[cell["label"]] = int(run_id)
        print(f"register: {cell['label']} -> run {run_id}", flush=True)

    best = rec["recommended"]
    note = (
        "Exit & transition study (Kass: 'play with the stop loss and "
        "take profit metrics... pick the best way for it to close or "
        "open positions'): signal fixed = GBM Lag F (run 26); 36 "
        "exit cells (stop 1.5/2/3xATR x TP off/+4%/+8%/2R x trailing "
        "current/off/tight) under through-flat, then T1 one-motion "
        "reversal and T2 staged flips on the default + top-3 exit "
        "configs; every cell graded on the standard windows AND the "
        "preceding 792h. Default-config cell reproduces run 26 "
        "exactly (agg -1,871.18 / P&L -$4,416.98 / 40 trips). "
        f"Recommended recipe: {best_label} -- standard agg "
        f"{best['standard']['agg_score']:.0f} / P&L "
        f"${best['standard']['total_pnl']:+,.0f}, robustness agg "
        f"{best['robustness']['agg_score']:.0f} / P&L "
        f"${best['robustness']['total_pnl']:+,.0f}, vs production "
        f"default mean agg {rec['default']['mean_agg']:.0f}. "
        f"Registered runs: {run_ids}. Full grid: "
        "refs/exit-study-2026-10-22.md")
    add_lab_note("Exit & transition study (family=exit_study)", note)
    save_result("register", {"run_ids": run_ids, "lab_note": note})
    print("register: lab note added", flush=True)


def phase_verify() -> None:
    start = results()["meta"]["integrity_start"]
    now = _mtimes()
    same = all(start[k] == now[k] for k in start)
    save_result("verify", {"protected_unchanged": bool(same),
                           "start": start, "end": now})
    print(f"verify: protected files unchanged = {same}", flush=True)
    assert same, f"INTEGRITY: protected files changed {start} -> {now}"


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    signals = get_signals()
    phase_setup(signals)
    phase_stage1(signals)
    phase_stage2(signals)
    phase_recommend()
    phase_register()
    phase_verify()
    print("=== exit study build complete ===", flush=True)


if __name__ == "__main__":
    main()
