#!/usr/bin/env python
"""Build driver: Drawing-Board Option A, the 4h timeframe test
(2026-10-24, refs/timeframe-4h-plan-2026-10-24.md).

Checkpointed phases in data/tf4h/progress.json; results in
data/tf4h/results.json. LAB ONLY: nothing here imports
src.trading.live; champions, paper.db and live_state are never
written (champion mtimes are snapshotted in setup and re-verified
in verify).

  setup     frame/slice censuses + integrity snapshot
  sup       make_supervised on the long 4h bars; calendar windows
  wf:<arm>  TimeSeriesSplit(5) walk-forward per arm
  duel:<arm> final fits + standard duel + robustness duel per arm
  register  one training.db run per arm (family tf4h, promoted=0)
            + one lab note
  verify    champion mtimes unchanged; prints the summary
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
sys.path.insert(0, str(ROOT / "scripts"))

from features.technical_indicators import make_supervised  # noqa: E402
from models import gbm_focus as gf  # noqa: E402
from models import tf4h  # noqa: E402
from models.gbm_policy import build_rule_model  # noqa: E402

OUT = ROOT / "data" / "tf4h"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
RESULTS = OUT / "results.json"
TARGET = gf.TARGET_COL

CHAMPIONS = [ROOT / "models" / "ppo_champion.zip",
             ROOT / "models" / "stacker_champion.pkl"]


def load(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def save(p, obj):
    p.write_text(json.dumps(obj, indent=1, default=str))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mtimes():
    return {str(p.name): p.stat().st_mtime_ns for p in CHAMPIONS}


def matrix_df():
    """The sliced 4h matrix as a working frame: index = bar open,
    TARGET_COL = the label."""
    frame = tf4h.load_frame4h()
    sl = tf4h.slice_span(frame)
    df = sl.copy()
    df[TARGET] = df["__label"]
    return df


def std_start(df):
    return df.index[-tf4h.HOLDOUT_BARS]


def rob_start(df):
    return df.index[-2 * tf4h.HOLDOUT_BARS]


# ---------------------------------------------------------------- phases
def phase_setup(res):
    frame = tf4h.load_frame4h()
    sl = tf4h.slice_span(frame)
    feats = tf4h.all_feature_columns(frame)
    split = sl["__label"].value_counts().to_dict()
    df = matrix_df()
    s0, r0 = std_start(df), rob_start(df)
    pre = df[df.index < s0]
    checks = {
        "long_bars": int(len(frame)),
        "sliced_bars": int(len(sl)),
        "features": int(len(feats)),
        "nan_in_slice_features": int(sl[feats].isna().sum().sum()),
        "class_split": {k: int(v) for k, v in split.items()},
        "first_bar": str(sl.index[0]), "last_bar": str(sl.index[-1]),
        "std_start": str(s0), "rob_start": str(r0),
        "pre_duel_rows": int(len(pre)),
        "pre_duel_classes": {k: int(v) for k, v in
                             pre[TARGET].value_counts().items()},
    }
    assert checks["sliced_bars"] == 4183, checks
    assert checks["features"] == 90, checks
    assert checks["nan_in_slice_features"] == 0, checks
    assert checks["pre_duel_rows"] == 3985, checks
    assert str(s0) == "2026-09-18 20:00:00", checks
    assert str(r0) == "2026-08-16 20:00:00", checks
    res["setup"] = {"checks": checks, "champion_mtimes": mtimes()}
    log(f"setup ok: {checks}")


def phase_sup(res):
    frame = tf4h.load_frame4h()
    bars = frame[["open", "high", "low", "close", "volume"]]
    sup = make_supervised(bars)
    sl_idx = tf4h.slice_span(frame).index
    sup = sup.loc[sup.index.isin(sl_idx)].sort_index()
    assert len(sup) == 4183, len(sup)
    std = sup.iloc[-tf4h.HOLDOUT_BARS:]
    rob = sup.iloc[-2 * tf4h.HOLDOUT_BARS:-tf4h.HOLDOUT_BARS]
    assert str(std.index[0]) == res["setup"]["checks"]["std_start"]
    assert str(rob.index[0]) == res["setup"]["checks"]["rob_start"]
    sup.to_pickle(OUT / "sup4h.pkl")
    res["sup"] = {"rows": int(len(sup)),
                  "std": [str(std.index[0]), str(std.index[-1])],
                  "rob": [str(rob.index[0]), str(rob.index[-1])]}
    log(f"sup ok: {res['sup']}")


def _windows(sup, which):
    if which == "std":
        rows = sup.iloc[-tf4h.HOLDOUT_BARS:]
    else:
        rows = sup.iloc[-2 * tf4h.HOLDOUT_BARS:-tf4h.HOLDOUT_BARS]
    return [rows.iloc[k * tf4h.WINDOW_BARS:(k + 1) * tf4h.WINDOW_BARS]
            for k in range(3)]


def _conviction_series(model, df_region, feats):
    proba = model.predict_proba(df_region[feats].to_numpy(dtype=float))
    # conviction = P(BUY) - P(SHORT), the composite screen's def.
    conv = np.asarray(proba[:, 0], dtype=float) \
        - np.asarray(proba[:, 2], dtype=float)
    return pd.Series(conv, index=df_region.index)


def phase_wf(arm, res):
    df = matrix_df()
    pre = df[df.index < std_start(df)]
    out = tf4h.walk_forward_arm(pre, arm)
    res.setdefault("wf", {})[arm] = out
    log(f"wf {arm}: pooled acc {out['pooled']['accuracy']:.4f} "
        f"bal {out['pooled']['balanced_accuracy']:.4f} "
        f"base {out['pooled']['majority_baseline']:.4f}")


def phase_duel(arm, res):
    rule_path = OUT / "rule_model"
    if not (rule_path.with_suffix(".zip")).exists():
        build_rule_model().save(str(rule_path))
    from stable_baselines3 import PPO
    rule = PPO.load(str(rule_path))
    df = matrix_df()
    sup = pd.read_pickle(OUT / "sup4h.pkl")
    feats = tf4h.arm_features(df, arm)
    s0, r0 = std_start(df), rob_start(df)
    out = {}
    probs = {}
    for which, cutoff, wins in (("std", s0, _windows(sup, "std")),
                                ("rob", r0, _windows(sup, "rob"))):
        fit_rows = df[df.index < cutoff]
        model, audit = gf.fit_arm_final(
            fit_rows[feats].to_numpy(dtype=float),
            fit_rows[TARGET].to_numpy(), "B")
        region = (df[(df.index >= cutoff) & (df.index < s0)]
                  if which == "rob" else df[df.index >= s0])
        conv = _conviction_series(model, region, feats)
        by_ts = tf4h.conviction_by_ts(conv.index, conv.to_numpy())
        p_ups, missing = tf4h.p_ups_from_conviction(wins, by_ts)
        assert missing == 0, (arm, which, missing)
        ev = tf4h.evaluate_windows_4h(wins, p_ups, rule)
        out[which] = ev
        probs[which] = np.concatenate(p_ups)
        log(f"duel {arm} {which}: pnl {ev['total_pnl']:+.2f} "
            f"gross {ev['gross_pnl']:+.2f} fees {ev['fees_total']:.2f} "
            f"bench {ev['bench_pnl']:+.2f} trips "
            f"{ev['total_round_trips']} nonhold "
            f"{ev['non_hold_share']:.3f} (fit n={len(fit_rows)})")
    np.savez(OUT / f"probs_{arm}.npz", **probs)
    res.setdefault("duel", {})[arm] = out


def _flat(ev):
    return {
        "score": float(ev["agg_score"]),
        "total_pnl": float(ev["total_pnl"]),
        "mean_reward": float(np.mean([w["mean_reward"]
                                      for w in ev["windows"]])),
        "max_drawdown": float(ev["max_drawdown"]),
        "bench_pnl": float(ev["bench_pnl"]),
        "action_mix": dict(ev["action_mix"]),
        "windows": [{"score": float(w["score"]),
                     "total_pnl": float(w["total_pnl"]),
                     "bench_pnl": float(w["bench_pnl"])}
                    for w in ev["windows"]],
        "runnable": True,
    }


def phase_register(res):
    from training.nightly import add_lab_note, log_run
    df = matrix_df()
    run_ids = dict(res.get("registered", {}))
    if not run_ids:
        # Resume guard: an earlier attempt may have registered the
        # runs before crashing at the lab note (results save happens
        # after the phase). Seed from training.db so a resume never
        # double-registers.
        import sqlite3
        con = sqlite3.connect(ROOT / "data" / "training.db")
        for rid, mtxt in con.execute(
                "SELECT id, metrics FROM runs"):
            try:
                m = json.loads(mtxt)
            except Exception:
                continue
            if m.get("family") == "tf4h":
                run_ids[m["tf4h"]["arm"]] = rid
        con.close()
        if run_ids:
            log(f"register: seeded existing runs from db: {run_ids}")
    for arm in tf4h.ARMS:
        wf = res["wf"][arm]
        duel = res["duel"][arm]["std"]
        robust = res["duel"][arm]["rob"]
        verdict = tf4h.apply_verdict(wf, duel, robust)
        res.setdefault("verdict", {})[arm] = verdict
        if arm in run_ids:
            log(f"register {arm}: already run {run_ids[arm]}, skip")
            continue
        pooled = wf["pooled"]
        v = verdict
        reason = (
            f"4H timeframe test (plan 2026-10-24): {arm} on 4,183 4h "
            f"bars. WF acc {pooled['accuracy']:.4f} vs majority "
            f"{pooled['majority_baseline']:.4f}, balanced "
            f"{pooled['balanced_accuracy']:.4f} -> rule 1 "
            f"{'PASS' if v['rule1_walk_forward'] else 'FAIL'}; "
            f"duel P&L {duel['total_pnl']:+,.2f} (gross "
            f"{duel['gross_pnl']:+,.2f}, fees {duel['fees_total']:,.2f}) "
            f"vs B&H {duel['bench_pnl']:+,.2f}, non-hold "
            f"{duel['non_hold_share']:.1%} -> rule 2 "
            f"{'PASS' if v['rule2_duel_floor'] else 'FAIL'}; "
            f"robustness P&L {robust['total_pnl']:+,.2f} -> rule 3 "
            f"{'PASS' if v['rule3_robustness'] else 'FAIL'}. "
            f"Verdict: {v['verdict']} (no PPO champion party at 4h; "
            f"promoted=0 always -- a pass is a report to Kass).")
        metrics = {
            "challenger": _flat(duel),
            "champion": {"score": None, "total_pnl": None,
                         "note": "no PPO champion party at 4h (1h "
                                 "brain); yardsticks are buy-and-hold "
                                 "and the absolute floor"},
            "reason": reason,
            "family": "tf4h",
            "tf4h": {
                "arm": arm,
                "label": tf4h.ARM_LABELS[arm],
                "design": tf4h.ARM_DESIGNS[arm],
                "verdict": v["verdict"],
                "verdict_rules": {
                    "rule1_walk_forward": v["rule1_walk_forward"],
                    "rule2_duel_floor": v["rule2_duel_floor"],
                    "rule3_robustness": v["rule3_robustness"]},
                "gross_pnl": float(duel["gross_pnl"]),
                "fees_total": float(duel["fees_total"]),
                "non_hold_share": float(duel["non_hold_share"]),
                "robust_pnl": float(robust["total_pnl"]),
                "robust_gross_pnl": float(robust["gross_pnl"]),
                "robust_fees": float(robust["fees_total"]),
                "wf_accuracy": float(pooled["accuracy"]),
                "wf_balanced_accuracy":
                    float(pooled["balanced_accuracy"]),
                "wf_majority_baseline":
                    float(pooled["majority_baseline"]),
                "wf_fold_balanced": v["fold_balanced_accuracies"],
            },
        }
        rid = log_run("ppo", metrics, False)
        run_ids[arm] = rid
        res["registered"] = run_ids
        save(RESULTS, res)  # persist before the lab note: a crash
        # here must never re-register a run on resume
        log(f"registered {arm} as run {rid}: {v['verdict']}")
    if res.get("lab_note_done"):
        log("lab note already added, skip")
        return
    lines = ["**4H timeframe test (Drawing-Board A, 2026-10-24)** — "
             "the same 90 indicators + Kass's focus-11 on 4h bars "
             "(4,183 bars, same calendar span/duel windows as 1h):"]
    for arm in tf4h.ARMS:
        v = res["verdict"][arm]
        d = res["duel"][arm]["std"]
        lines.append(
            f"- {tf4h.ARM_LABELS[arm]}: WF balanced "
            f"{res['wf'][arm]['pooled']['balanced_accuracy']:.4f}; "
            f"duel P&L {d['total_pnl']:+,.0f} (gross "
            f"{d['gross_pnl']:+,.0f}, fees {d['fees_total']:,.0f}) vs "
            f"B&H {d['bench_pnl']:+,.0f}; robustness "
            f"{res['duel'][arm]['rob']['total_pnl']:+,.0f} — "
            f"**{v['verdict']}** "
            f"(rules {int(v['rule1_walk_forward'])}/"
            f"{int(v['rule2_duel_floor'])}/"
            f"{int(v['rule3_robustness'])}).")
    lines.append("Full report: refs/timeframe-4h-2026-10-24.md.")
    add_lab_note("4H timeframe test (family=tf4h)", "\n".join(lines))
    res["lab_note_done"] = True
    save(RESULTS, res)
    log(f"lab note added; runs {run_ids}")


def phase_verify(res):
    now = mtimes()
    base = res["setup"]["champion_mtimes"]
    assert now == {k: int(v) for k, v in base.items()} or \
        all(now[k] == int(base[k]) for k in base), (base, now)
    res["verify"] = {"champion_mtimes_unchanged": True}
    log("verify ok: champion mtimes unchanged")


PHASES = ["setup", "sup", "wf:all4h", "wf:focus4h", "duel:all4h",
          "duel:focus4h", "register", "verify"]


def main():
    prog = load(PROGRESS, {"done": []})
    res = load(RESULTS, {})
    for ph in PHASES:
        if ph in prog["done"]:
            log(f"skip {ph}")
            continue
        t0 = time.time()
        if ph == "setup":
            phase_setup(res)
        elif ph == "sup":
            phase_sup(res)
        elif ph.startswith("wf:"):
            phase_wf(ph.split(":")[1], res)
        elif ph.startswith("duel:"):
            phase_duel(ph.split(":")[1], res)
        elif ph == "register":
            phase_register(res)
        elif ph == "verify":
            phase_verify(res)
        prog["done"].append(ph)
        save(PROGRESS, prog)
        save(RESULTS, res)
        log(f"phase {ph} done in {time.time() - t0:.1f}s")
    log("ALL PHASES DONE")


if __name__ == "__main__":
    main()
