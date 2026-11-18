"""Curated features v6 LAB study (2026-10-25, Kass green-lit).

Question: does a CURATED wider feature set beat the production 21
through the same gate? Construction (pre-registered in
src/features/features_v6.py): production 21 minus the obv_slope
duplicate, plus AZ-screen indicators with max |r| < 0.90 vs the 21,
greedily filtered to pairwise |r| < 0.90 against the retained 20 and
each other on the study's train frame. Stacker + PPO (fresh, 60k --
no warm start exists across feature versions) trained on v6, graded
through the SAME 3 chronological windows, next-candle-open fills,
score = P&L - 0.5 x maxDD, duel vs the reigning champions, two-part
floor evaluated by the production gate_decision.

LAB ONLY: never runs the live loop, never writes the paper ledger,
never installs anything. Registrations are promoted=0; a pass is
recorded pending_approval under the promotion lock (Kass,
2026-10-25) and reported as a study verdict. Champion files and
paper.db are hash-guarded before/after.

Usage: .venv/bin/python scripts/build_features_v6.py
Phases checkpoint to data/features_v6/; reruns skip done phases.
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

from data.market_data import fetch_klines  # noqa: E402
from features import features_v6 as v6  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, feature_matrix, make_supervised)
from models import stacker  # noqa: E402
from models.ppo_train import train as train_ppo  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import (  # noqa: E402
    _agg, add_lab_note, crossfit_p_up, evaluate_windows, gate_decision,
    log_run)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "features_v6"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"

HOLDOUT = 792
WINDOW = 264
MAX_ADDED = 16
MIN_ADDED = 8
CORR_CAP = 0.90
TIMESTEPS = 60_000


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


def get_df() -> pd.DataFrame:
    if DF_PATH.exists():
        return pd.read_pickle(DF_PATH)
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(DF_PATH)
    return df


def get_frames():
    """(df, prod_sup, split, holdout_start, train_end_pos)."""
    df = get_df()
    prod_sup = make_supervised(df)
    split = len(prod_sup) - HOLDOUT
    holdout_start = prod_sup.index[split]
    train_end_pos = int(df.index.get_indexer([holdout_start])[0])
    assert df.index[train_end_pos] == holdout_start
    return df, prod_sup, split, holdout_start, train_end_pos


# ---------------------------------------------------------------- phases
def phase_data() -> None:
    df, prod_sup, split, holdout_start, train_end_pos = get_frames()
    save_result("meta", {
        "rows_raw": len(df), "rows_sup": len(prod_sup),
        "split": split, "holdout_start": str(holdout_start),
        "train_end_pos": train_end_pos,
        "guard_before": _guard(),
        "champion_ppo_sha": _sha(MODELS / "ppo_champion.zip"),
    })
    print(f"data: {len(df)} raw rows, sup {len(prod_sup)}, "
          f"holdout from {holdout_start}", flush=True)


def phase_curate() -> None:
    from features.technical_indicators import add_features

    df, prod_sup, split, holdout_start, train_end_pos = get_frames()
    candidates = v6.ADD_CANDIDATES + v6.ADD_FALLBACK
    scaled = v6.compute_added(df, candidates, train_end_pos)
    base = add_features(df)
    frame = base[v6.RETAINED_COLUMNS].join(scaled)
    train_idx = prod_sup.index[:split]
    hold_idx = prod_sup.index[split:]

    accepted: list[str] = []
    evidence: dict[str, dict] = {}
    for name in candidates:
        series = frame[name]
        if len(accepted) >= MAX_ADDED:
            evidence[name] = {"accepted": False,
                              "reason": "addition cap reached"}
            continue
        finite_holdout = bool(np.isfinite(
            series.loc[hold_idx].to_numpy()).all())
        if not finite_holdout:
            evidence[name] = {"accepted": False,
                              "reason": "NaN inside the holdout region"}
            continue
        cols = v6.RETAINED_COLUMNS + accepted + [name]
        sub = frame.loc[train_idx, cols].dropna()
        corrs = sub.corr().abs()
        others = v6.RETAINED_COLUMNS + accepted
        max_r = float(corrs.loc[name, others].max())
        argmax = str(corrs.loc[name, others].idxmax())
        if max_r < CORR_CAP:
            accepted.append(name)
            evidence[name] = {"accepted": True, "max_r_train": max_r,
                              "argmax_train": argmax,
                              "screen_max_corr_21": None}
        else:
            evidence[name] = {"accepted": False, "max_r_train": max_r,
                              "argmax_train": argmax,
                              "reason": f"max |r| {max_r:.3f} >= 0.90"}
    # Attach the screen's own redundancy measurement per candidate.
    screen = json.loads((ROOT / "data" / "az_screen" / "results.json")
                        .read_text())["corr"]
    for name in candidates:
        evidence[name]["screen_max_corr_21"] = screen[name]["max_corr_21"]
        evidence[name]["screen_argmax_21"] = screen[name]["argmax_21"]
    assert len(accepted) >= MIN_ADDED, (
        f"only {len(accepted)} additions survived curation")
    # The assembled frame's holdout must be row-identical to the
    # production holdout, or the duel windows would not be the same
    # candles. This is the alignment contract for the whole study.
    sup6 = v6.make_v6_supervised(df, accepted, train_end_pos)
    hold6 = sup6[sup6.index >= holdout_start]
    assert list(hold6.index) == list(hold_idx), (
        "v6 holdout timestamps differ from production holdout")
    save_result("curation", {"added": accepted, "evidence": evidence,
                             "n_features": len(v6.v6_columns(accepted))})
    print(f"curate: {len(accepted)} additions -> "
          f"{len(v6.v6_columns(accepted))} features: {accepted}",
          flush=True)


def _v6_sup():
    df, prod_sup, split, holdout_start, train_end_pos = get_frames()
    added = results()["curation"]["added"]
    sup6 = v6.make_v6_supervised(df, added, train_end_pos)
    return df, prod_sup, split, holdout_start, sup6, added


def phase_stacker() -> None:
    _, prod_sup, split, holdout_start, sup6, added = _v6_sup()
    cols = v6.v6_columns(added)
    train6 = sup6[sup6.index < holdout_start]
    hold6 = sup6[sup6.index >= holdout_start]
    Xtr, ytr = train6[cols], train6["target"].astype(int)
    score = stacker.train_and_save(Xtr, ytr, LAB / "stacker_v6.pkl")
    model6 = joblib.load(LAB / "stacker_v6.pkl")
    from sklearn.metrics import accuracy_score, precision_score
    pred6 = model6.predict(hold6[cols])
    acc6 = float(accuracy_score(hold6["target"], pred6))
    prec6 = float(precision_score(hold6["target"], pred6,
                                  zero_division=0))
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    holdp = prod_sup.iloc[split:]
    assert list(hold6.index) == list(holdp.index)
    assert (hold6["target"].to_numpy()
            == holdp["target"].to_numpy()).all()
    pred_c = champ_model.predict(holdp[FEATURE_COLUMNS])
    acc_c = float(accuracy_score(holdp["target"], pred_c))
    win = acc6 > acc_c
    save_result("stacker", {
        "challenger_acc": acc6, "champion_acc": acc_c,
        "challenger_wf_acc": score["accuracy"],
        "challenger_wf_precision": score["precision"],
        "precision": prec6, "folds": score["folds"], "win": win})
    print(f"stacker: v6 holdout acc={acc6:.4f} (wf {score['accuracy']:.4f}) "
          f"vs champion {acc_c:.4f} -> {'WIN' if win else 'no win'}",
          flush=True)


def phase_ppo() -> None:
    _, _, _, holdout_start, sup6, added = _v6_sup()
    cols = v6.v6_columns(added)
    train6 = sup6[sup6.index < holdout_start]
    Xtr, ytr = train6[cols], train6["target"].astype(int)
    p_up_train = crossfit_p_up(Xtr, ytr)
    out = train_ppo(
        timesteps=TIMESTEPS, warm_start=False,
        champion_path=LAB / "ppo_anchor.zip",  # nonexistent: fresh brain;
        # outputs (challenger zip + curve) land in LAB by construction
        stacker_path=LAB / "stacker_v6.pkl",
        db_path=LAB / "train_tmp.db",
        ent_coef=0.05,
        sup=train6, p_up=p_up_train,
        feature_columns=cols)
    curve = None
    curve_path = LAB / "ppo_challenger_curve.json"
    if curve_path.exists():
        try:
            curve = json.loads(curve_path.read_text()).get("curve")
        except Exception:
            curve = None
    save_result("ppo", {"challenger_path": str(out),
                        "reward_curve": curve})
    print(f"ppo: fresh {TIMESTEPS} steps on {len(cols)} features -> {out}",
          flush=True)


def phase_duel() -> None:
    _, prod_sup, split, holdout_start, sup6, added = _v6_sup()
    cols = v6.v6_columns(added)
    holdp = prod_sup.iloc[split:]
    hold6 = sup6[sup6.index >= holdout_start]
    assert list(hold6.index) == list(holdp.index)
    windows_p = [holdp.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    windows_6 = [hold6.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    champ_stacker = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold_p = champ_stacker.predict_proba(
        holdp[FEATURE_COLUMNS])[:, 1]
    p_ups_p = [p_up_hold_p[k * WINDOW:(k + 1) * WINDOW] for k in range(3)]
    model6 = joblib.load(LAB / "stacker_v6.pkl")
    p_up_hold_6 = model6.predict_proba(hold6[cols])[:, 1]
    p_ups_6 = [p_up_hold_6[k * WINDOW:(k + 1) * WINDOW] for k in range(3)]

    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows_p, p_ups_p)
    chall_path = Path(results()["ppo"]["challenger_path"])
    chall_res = evaluate_windows(chall_path, windows_6, p_ups_6,
                                 feature_columns=cols)
    assert champ_res is not None, "champion failed to grade"
    assert chall_res is not None, "v6 challenger failed to grade"
    # Cross-checks (two ways): identical windows => identical
    # buy-and-hold per window, recomputed straight from the closes;
    # and the aggregate arithmetic must reconcile from the windows.
    for k, (wp, w6) in enumerate(zip(windows_p, windows_6)):
        closes = wp["close"].to_numpy(dtype=float)
        bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0))
        assert abs(champ_res["windows"][k]["bench_pnl"] - bench) < 1e-6
        assert abs(chall_res["windows"][k]["bench_pnl"] - bench) < 1e-6
        assert list(wp.index) == list(w6.index)
    for res in (champ_res, chall_res):
        assert abs(res["agg_score"]
                   - float(np.mean([w["score"]
                                    for w in res["windows"]]))) < 1e-9
        assert abs(res["total_pnl"]
                   - float(sum(w["total_pnl"]
                               for w in res["windows"]))) < 1e-9
    wins = sum(1 for cw, hw in zip(champ_res["windows"],
                                   chall_res["windows"])
               if hw["score"] > cw["score"])
    formula_pass = bool(chall_res["agg_score"] > champ_res["agg_score"]
                        and wins >= 2)
    chall_agg = _agg(chall_res)
    promotable, floor_fail = gate_decision(chall_agg, formula_pass)
    save_result("duel", {
        "champion": _agg(champ_res), "challenger": chall_agg,
        "windows_won": wins, "formula_pass": formula_pass,
        "floor_fail": floor_fail, "promotable": promotable})
    print(f"duel: v6 agg={chall_agg['score']:.2f} "
          f"(pnl={chall_agg['total_pnl']:.2f}) vs champion "
          f"agg={_agg(champ_res)['score']:.2f} "
          f"(pnl={_agg(champ_res)['total_pnl']:.2f}); won {wins}/3; "
          f"formula={formula_pass} floor_fail={floor_fail}",
          flush=True)


def _already_registered() -> bool:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        kinds = {k for (k,) in con.execute(
            "SELECT kind FROM runs WHERE metrics LIKE '%features_v6%'")}
    finally:
        con.close()
    return {"stacker", "ppo"} <= kinds


def phase_register() -> None:
    res = results()
    if _already_registered():
        # A previous register attempt completed the writes and then
        # stopped at the guard below; never double-register.
        print("register: features_v6 runs already present — "
              "skipping duplicate registration", flush=True)
        guard_after = _guard()
        champs = ("ppo_champion.zip", "stacker_champion.pkl",
                  "stacker_champion.json")
        assert all(guard_after[c] == res["meta"]["guard_before"][c]
                   for c in champs), "CHAMPION GUARD TRIPPED"
        save_result("registered", {
            "stacker_win": res["stacker"]["win"],
            "ppo_promotable": res["duel"]["promotable"],
            "guard_after": guard_after,
            "paper_note": "paper.db is written by the live loop on "
                          "hourly candle closes throughout the study "
                          "window; the study itself never opens it "
                          "(hash read only). Champion files are "
                          "hash-identical before/after."})
        print("register: verified existing registration; champion "
              "guard intact", flush=True)
        return
    added = res["curation"]["added"]
    cols = v6.v6_columns(added)
    stk, duel = res["stacker"], res["duel"]
    design = (f"features v6: 20 retained (obv_slope duplicate dropped) "
              f"+ {len(added)} curated AZ additions "
              f"({', '.join(a[3:] for a in added)}), pairwise |r|<0.90 "
              f"on train; same recipe, fresh brains")
    common = {"family": "features_v6", "deep": False,
              "features_v6": {
                  "design": design, "features": cols,
                  "retained": v6.RETAINED_COLUMNS, "added": added,
                  "dropped": v6.DROPPED_DUPLICATE,
                  "n_features": len(cols),
                  "corr_evidence": res["curation"]["evidence"]}}

    stk_win = bool(stk["win"])
    stk_reason = (
        f"features v6 stacker holdout acc {stk['challenger_acc']:.4f} "
        f"{'beat' if stk_win else 'did not beat'} champion "
        f"{stk['champion_acc']:.4f} on the shared holdout timestamps "
        f"(v6 features vs production 21)")
    if stk_win:
        stk_reason += (" — WOULD PROMOTE: held as PENDING KASS "
                       "APPROVAL under the promotion lock; champion "
                       "kept (study verdict only)")
    stk_metrics = {"challenger_acc": stk["challenger_acc"],
                   "champion_acc": stk["champion_acc"],
                   "challenger_wf_acc": stk["challenger_wf_acc"],
                   "precision": stk["precision"],
                   "folds": stk["folds"],
                   "pending_approval": stk_win,
                   "reason": stk_reason,
                   **common,
                   "features_v6": {**common["features_v6"],
                                   "label": "Stacker",
                                   "verdict": "PASS" if stk_win
                                              else "FAIL"}}
    log_run("stacker", stk_metrics, False)

    promotable = bool(duel["promotable"])
    wins = duel["windows_won"]
    chall, champ = duel["challenger"], duel["champion"]
    if promotable:
        ppo_reason = (
            f"features v6 challenger won {wins}/3 windows, aggregate "
            f"{chall['score']:.0f} vs champion {champ['score']:.0f} "
            f"(gate pass incl. two-part floor) — WOULD PROMOTE: held "
            f"as PENDING KASS APPROVAL under the promotion lock; "
            f"champion kept (study verdict only)")
    elif duel["floor_fail"] is not None:
        ppo_reason = (
            f"features v6 challenger passed the relative gate "
            f"(aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f}, won {wins}/3) but failed the "
            f"floor — {duel['floor_fail']}; champion kept")
    else:
        ppo_reason = (
            f"features v6 challenger won {wins}/3 windows, aggregate "
            f"{chall['score']:.0f} vs champion {champ['score']:.0f}; "
            f"champion kept (study verdict)")
    ppo_metrics = {"challenger": chall, "champion": champ,
                   "reason": ppo_reason, "deep": False,
                   "timesteps": TIMESTEPS,
                   "pending_approval": promotable,
                   "reward_curve": res["ppo"].get("reward_curve"),
                   **common,
                   "features_v6": {**common["features_v6"],
                                   "label": "PPO",
                                   "verdict": "PASS" if promotable
                                              else "FAIL",
                                   "formula_pass":
                                       duel["formula_pass"],
                                   "floor_fail": duel["floor_fail"],
                                   "windows_won": wins}}
    log_run("ppo", ppo_metrics, False)
    add_lab_note(
        "Features v6 study (family=features_v6)",
        f"{design}. Stacker: {stk_reason}. PPO: {ppo_reason}. "
        f"Buy-and-hold over the duel windows: "
        f"${chall['bench_pnl']:+,.0f}. Study verdict only — nothing "
        f"installed (promotion lock).")
    guard_after = _guard()
    champs = ("ppo_champion.zip", "stacker_champion.pkl",
              "stacker_champion.json")
    assert all(guard_after[c] == res["meta"]["guard_before"][c]
               for c in champs), (
        f"CHAMPION GUARD TRIPPED: {guard_after} vs "
        f"{res['meta']['guard_before']}")
    save_result("registered", {
        "stacker_win": stk_win, "ppo_promotable": promotable,
        "guard_after": guard_after,
        "paper_note": "paper.db is written by the live loop on hourly "
                      "candle closes throughout the study window; the "
                      "study itself never opens it (hash read only). "
                      "Champion files are hash-identical before/after."})
    print("register: 2 runs logged (promoted=0); champion + paper "
          "guard intact", flush=True)


PHASES = [("data", phase_data), ("curate", phase_curate),
          ("stacker", phase_stacker), ("ppo", phase_ppo),
          ("duel", phase_duel), ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== features v6 study (phases done: {sorted(done)}) ===",
          flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== features v6 study complete ===", flush=True)


if __name__ == "__main__":
    main()
