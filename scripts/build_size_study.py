"""Tranche-size study (2026-10-22, Kass's request: "lower the 25
percent of total portfolio value to 5 and then re run the numbers").

Re-grades the reference signals at tranche = 5% of equity vs 25%,
everything else frozen (see src/models/size_study.py). Signals:
GBM Lag F (run 26, standard + robustness windows), the PPO champion
(run 17's brain, standard), GBM Focus B (run 22, standard).

The 25% cells are asserted against the CURRENT-CODE baselines the
tests pin (the recorded duels for PPO / Focus B / Lag F-robustness
predate the 2026-10-22 stop-anchor fix and are reproduced exactly
only by the tests' pre-fix emulation broker -- see
tests/test_size_study.py). If production code drifts, this build
stops instead of registering a wrong comparison.

Lab only: promoted=0 everywhere; the production default
(max_position_pct=0.25) is NOT changed; champion files, paper.db
and live state are verified untouched at the end.

Usage: .venv/bin/python scripts/build_size_study.py
Phases checkpoint to data/size_study/results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.size_study import (evaluate_windows_size,  # noqa: E402
                               load_focusb_signal, load_lagf_signal,
                               load_model, load_ppo_signal)

LAB = ROOT / "data" / "size_study"
RESULTS_PATH = LAB / "results.json"
MODELS = ROOT / "models"

PROTECTED = [MODELS / "ppo_champion.zip", MODELS / "stacker_champion.pkl",
             MODELS / "stacker_champion.json", ROOT / "data" / "paper.db",
             ROOT / "data" / "live_state.json",
             ROOT / "src" / "trading" / "broker.py",
             ROOT / "src" / "environment" / "trading_env.py",
             ROOT / "src" / "training" / "nightly.py"]

# Current-code 25% baselines (pinned identically in
# tests/test_size_study.py). agg / total_pnl / round_trips.
BASELINES_25 = {
    ("lagf", "standard"): (-1871.1787, -4416.9757, 40),
    ("lagf", "robustness"): (-8587.8592, -21288.1523, 75),
    ("ppo", "standard"): (-2037.2361, -4420.6411, None),
    ("focusb", "standard"): (-3329.8011, -8138.3720, 66),
}

LABELS = {"lagf": "Lag F", "ppo": "PPO Champion", "focusb": "Focus B"}
SIGNALS = {"lagf": ("standard", "robustness"),
           "ppo": ("standard",),
           "focusb": ("standard",)}


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


# ------------------------------------------------------------------ phases
def phase_setup() -> None:
    if "meta" in results():
        print("setup: cached", flush=True)
        return
    save_result("meta", {"integrity_start": _mtimes(),
                         "tranches": [0.25, 0.05],
                         "baselines_25": {
                             f"{s}|{w}": v for (s, w), v
                             in BASELINES_25.items()}})
    print("setup: integrity baseline recorded", flush=True)


def _load_signal(sig: str, wset: str):
    if sig == "lagf":
        return load_lagf_signal(wset)
    if sig == "ppo":
        return load_ppo_signal()
    return load_focusb_signal()


def phase_cells() -> None:
    cells = results().get("cells", {})
    for sig, wsets in SIGNALS.items():
        model = load_model(sig)
        for wset in wsets:
            windows, p_ups = _load_signal(sig, wset)
            for tranche in (0.25, 0.05):
                key = f"{sig}|{wset}|{tranche}"
                if key in cells:
                    print(f"cell {key}: cached", flush=True)
                    continue
                res = evaluate_windows_size(windows, p_ups, model,
                                            tranche)
                if tranche == 0.25:
                    agg, pnl, trips = BASELINES_25[(sig, wset)]
                    assert abs(res["agg_score"] - agg) < 0.01, (
                        f"{key}: 25% agg drifted {res['agg_score']} "
                        f"vs pinned {agg}")
                    assert abs(res["total_pnl"] - pnl) < 0.01, (
                        f"{key}: 25% P&L drifted {res['total_pnl']} "
                        f"vs pinned {pnl}")
                    if trips is not None:
                        assert res["total_round_trips"] == trips, (
                            f"{key}: 25% trips drifted "
                            f"{res['total_round_trips']} vs {trips}")
                cells[key] = res
                save_result("cells", cells)
                print(f"cell {key}: agg {res['agg_score']:,.2f} pnl "
                      f"{res['total_pnl']:+,.2f} (B&H "
                      f"{res['bench_pnl']:+,.2f}) trips "
                      f"{res['total_round_trips']} closed "
                      f"{res['closed_trips']} fees "
                      f"${res['fees_total']:,.2f} avgdep "
                      f"${res['avg_deployed_all']:,.0f}", flush=True)


def phase_compare() -> None:
    if "comparison" in results():
        print("compare: cached", flush=True)
        return
    cells = results()["cells"]
    comp = {}
    for sig, wsets in SIGNALS.items():
        comp[sig] = {}
        for wset in wsets:
            a = cells[f"{sig}|{wset}|0.25"]
            b = cells[f"{sig}|{wset}|0.05"]
            comp[sig][wset] = {
                "agg_25": a["agg_score"], "agg_5": b["agg_score"],
                "d_agg": b["agg_score"] - a["agg_score"],
                "pnl_25": a["total_pnl"], "pnl_5": b["total_pnl"],
                "d_pnl": b["total_pnl"] - a["total_pnl"],
                "bench": a["bench_pnl"],
                "trips_25": a["total_round_trips"],
                "trips_5": b["total_round_trips"],
                "fees_25": a["fees_total"], "fees_5": b["fees_total"],
                "avgdep_all_25": a["avg_deployed_all"],
                "avgdep_all_5": b["avg_deployed_all"],
                "pnlpct_dep_25": a["pnl_pct_deployed_all"],
                "pnlpct_dep_5": b["pnl_pct_deployed_all"],
                "pnlpct_inpos_25": a["pnl_pct_deployed_inpos"],
                "pnlpct_inpos_5": b["pnl_pct_deployed_inpos"],
                "maxdd_25": a["max_drawdown"],
                "maxdd_5": b["max_drawdown"],
                "cap_limited_25": a["tele"]["cap_limited"],
                "cap_limited_5": b["tele"]["cap_limited"],
                "cap_blocked_25": a["tele"]["cap_blocked"],
                "cap_blocked_5": b["tele"]["cap_blocked"],
                "adds_mean_25": a["adds_mean"],
                "adds_mean_5": b["adds_mean"],
            }
    save_result("comparison", comp)
    print("compare: deltas computed", flush=True)


def _reference_record(sig: str) -> dict:
    if sig == "lagf":
        lag = json.loads((ROOT / "data" / "gbm_lag" / "results.json")
                         .read_text())
        return lag["duel_F"]["arm"]  # run 26's recorded record
    foc = json.loads((ROOT / "data" / "gbm_focus" / "results.json")
                     .read_text())
    if sig == "focusb":
        return foc["duel_B"]["arm"]  # run 22's recorded record
    return foc["duel_champion"]["champion"]  # run 17 champion re-grade


def phase_register() -> None:
    res = results()
    if "register" in res:
        print("register: cached", flush=True)
        return
    from training.nightly import add_lab_note, log_run

    cells = res["cells"]
    run_ids = {}
    for sig in SIGNALS:
        label = LABELS[sig]
        std25 = cells[f"{sig}|standard|0.25"]
        std5 = cells[f"{sig}|standard|0.05"]
        reference = _reference_record(sig)
        rob_txt = ""
        rob_block = {}
        if sig == "lagf":
            rob25 = cells["lagf|robustness|0.25"]
            rob5 = cells["lagf|robustness|0.05"]
            rob_txt = (f"; robustness windows 5% agg "
                       f"{rob5['agg_score']:.0f} / P&L "
                       f"${rob5['total_pnl']:+,.0f} vs 25% agg "
                       f"{rob25['agg_score']:.0f} / P&L "
                       f"${rob25['total_pnl']:+,.0f}")
            rob_block = {"rob25_agg": rob25["agg_score"],
                         "rob25_pnl": rob25["total_pnl"],
                         "rob5_agg": rob5["agg_score"],
                         "rob5_pnl": rob5["total_pnl"]}
        reason = (
            f"Tranche-size study (family=size_study; signal = "
            f"{label}): the duel re-graded at 5% tranche size "
            f"(production remains 25% -- sizing adoption is Kass's "
            f"decision) -- standard windows agg {std5['agg_score']:.0f}"
            f" / P&L ${std5['total_pnl']:+,.0f} vs 25% same-code agg "
            f"{std25['agg_score']:.0f} / P&L "
            f"${std25['total_pnl']:+,.0f} (B&H "
            f"${std25['bench_pnl']:+,.0f}){rob_txt}; fees 5% "
            f"${std5['fees_total']:,.0f} vs 25% "
            f"${std25['fees_total']:,.0f}; lab only, promoted=0 -- "
            f"5% tranche study -- NOT a production change")
        metrics = {
            "challenger": _flatten(std5), "champion": reference,
            "reason": reason, "family": "size_study", "deep": False,
            "timesteps": 0, "reward_curve": None,
            "size_study": {
                "label": label, "verdict": "SIZE STUDY",
                "tranche_pct": 0.05,
                "std25_recorded_agg": reference.get("score"),
                "std25_recorded_pnl": reference.get("total_pnl"),
                "std25_agg": std25["agg_score"],
                "std25_pnl": std25["total_pnl"],
                "std5_agg": std5["agg_score"],
                "std5_pnl": std5["total_pnl"],
                "fees25": std25["fees_total"],
                "fees5": std5["fees_total"],
                "avgdep_all_25": std25["avg_deployed_all"],
                "avgdep_all_5": std5["avg_deployed_all"],
                "pnlpct_dep_25": std25["pnl_pct_deployed_all"],
                "pnlpct_dep_5": std5["pnl_pct_deployed_all"],
                **rob_block,
            },
        }
        log_run("ppo", metrics, False)
        import sqlite3
        with sqlite3.connect(ROOT / "data" / "training.db") as con:
            run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
        run_ids[sig] = int(run_id)
        print(f"register: {label} -> run {run_id}", flush=True)

    comp = res["comparison"]
    parts = []
    for sig in SIGNALS:
        c = comp[sig]["standard"]
        parts.append(
            f"{LABELS[sig]}: 5% P&L ${c['pnl_5']:+,.0f} vs 25% "
            f"${c['pnl_25']:+,.0f} (agg {c['agg_5']:.0f} vs "
            f"{c['agg_25']:.0f}); P&L as % of avg deployed capital "
            f"{c['pnlpct_dep_5']:.1f}% vs {c['pnlpct_dep_25']:.1f}%")
    note = (
        "Tranche-size study (Kass: 'lower the 25 percent of total "
        "portfolio value to 5 and then re run the numbers'): the "
        "reference signals re-graded at 5% tranche size, everything "
        "else frozen (same signals, production exits incl. the "
        "stop-anchor ratchet, 75% cap, 0.15% fees, next-candle "
        "fills). Standard windows -- " + "; ".join(parts) +
        f". Registered runs: {run_ids}. Production sizing is "
        "unchanged (25%); adoption is Kass's decision. Full tables: "
        "refs/size-study-5pct-2026-10-22.md")
    add_lab_note("Tranche-size study 5% (family=size_study)", note)
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
    phase_setup()
    phase_cells()
    phase_compare()
    phase_register()
    phase_verify()
    print("=== size study build complete ===", flush=True)


if __name__ == "__main__":
    main()
