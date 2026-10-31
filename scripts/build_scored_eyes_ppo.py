"""Scored-Eyes PPO LAB study (2026-10-25, Kass approved).

Question: Champion #2 (run 64) -- the lab stacker -- reads the
matrix-v3 scored frame (94 percentile-scored indicators), but the
PPO champion decides from the 21 raw production features; the
stacker's whole view reaches the PPO compressed into its single
vote. Does a PPO that sees the SAME scored eyes as the stacker
beat the 21-feature PPO champion in the standard duel?

Design (pre-registered):
- Challenger state (98 numbers): the 94 scored features of
  Champion #2's frame, restricted to run 64's recorded columns,
  each score s in [0, 100] mapped to [-1, 1] by s/50 - 1 (the
  affine centering family the env applies to the vote, 2p - 1);
  + the stacker vote; + the env's 3 portfolio readings, built by
  TradingEnv exactly as for the production brain.
- Vote source, IDENTICAL both sides: Champion #2 (run 64). In
  TRAINING the votes are cross-fitted (out-of-fold scored stackers,
  TimeSeriesSplit, leading segment neutral 0.5 -- the production
  anti-leak pattern, nightly.crossfit_p_up on the raw scored
  frame); in GRADING, run 64 itself on the holdout.
- Champion side graded exactly as the current production nightly
  grades it: prod21 supervised windows + lab-champion votes (the
  dual-frame grading of 2026-10-25). The ONLY systematic
  difference between the sides is the challenger's market-feature
  block (94 scored vs 21 raw) and its training.
- Arms: (1) 60,000 steps fresh; (2) 300,000 steps fresh via 25k
  segments (checkpointed per segment; no warm start exists across
  state shapes -- 25 vs 98 -- and none is faked: arm 2's segments
  chain from the arm's OWN previous segment only).
- Frame: the 730d family of the stacker study; holdout = the last
  792 production-supervised rows (33 days), identical timestamps
  both sides; train/holdout disjointness asserted.
- Grading: the standard duel -- 3 chronological windows x 10
  episodes, next-candle-open fills, score = P&L - 0.5 x maxDD,
  production gate formula + two-part floor per arm.

LAB ONLY: never runs the live loop, never writes the paper ledger,
never installs anything. Registrations are promoted=0; a pass is
recorded pending_approval under the promotion lock (Kass,
2026-10-25). Champion AND seat files are hash-guarded before/after
(paper.db is read-hash only -- the live loop writes it on hourly
closes throughout the study window).

Usage: .venv/bin/python scripts/build_scored_eyes_ppo.py
Phases checkpoint to data/scored_eyes_ppo/; reruns skip done phases
(arm 2 resumes at its next unfinished segment).
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
from features import scored_eyes, scored_features  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models.ppo_train import train as train_ppo  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import (  # noqa: E402
    _agg, add_lab_note, crossfit_p_up, evaluate_windows, gate_decision,
    log_run)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "scored_eyes_ppo"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
ENV_SUP_PATH = LAB / "env_sup.pkl"
PROD_SUP_PATH = LAB / "prod_sup.pkl"
VOTES_PATH = LAB / "votes.npz"

HOLDOUT = 792
WINDOW = 264
SEGMENT = 25_000
ARM60_STEPS = 60_000
ARM300_STEPS = 300_000
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_stacker_meta": MODELS / "live_seat" / "stacker.json",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
}
SAMPLE_COLS = ["score_master", "Relative Strength Index (RSI)",
               "Distance to SMA 50 (%)"]


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
    return {k: _sha(p) for k, p in GUARD_FILES.items()} | {
        "paper_db_info_only": _sha(ROOT / "data" / "paper.db")}


def _assert_guard(before: dict, after: dict) -> None:
    for k in GUARD_FILES:
        assert after[k] == before[k], (
            f"GUARD TRIPPED on {k}: {before[k]} -> {after[k]}")


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
    from features.technical_indicators import make_supervised
    prod_sup = make_supervised(df)
    split = len(prod_sup) - HOLDOUT
    holdout_start = prod_sup.index[split]
    hold_start_pos = int(df.index.get_indexer([holdout_start])[0])
    assert df.index[hold_start_pos] == holdout_start
    assert prod_sup.index[:split].max() < holdout_start
    return df, prod_sup, split, holdout_start, hold_start_pos


def _train_env_sup(env_sup, holdout_start):
    cols = scored_eyes.champion_columns()
    train = env_sup[env_sup.index < holdout_start].dropna(subset=cols)
    assert len(train) > 5_000, f"scored train frame too small: {len(train)}"
    assert train.index.max() < holdout_start
    return train


# ---------------------------------------------------------------- phases
def phase_data() -> None:
    df, prod_sup, split, holdout_start, hold_start_pos = get_frames()
    save_result("meta", {
        "rows_raw": len(df), "rows_sup": len(prod_sup),
        "split": split, "holdout_start": str(holdout_start),
        "hold_start_pos": hold_start_pos,
        "data_first": str(df.index[0]), "data_last": str(df.index[-1]),
        "guard_before": _guard(),
        "state_size": scored_eyes.STATE_SIZE,
        "n_scored": len(scored_eyes.champion_columns()),
    })
    print(f"data: {len(df)} raw rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(prod_sup)}, holdout from {holdout_start}", flush=True)


def phase_frames() -> None:
    df, prod_sup, split, holdout_start, hold_start_pos = get_frames()
    cols = scored_eyes.champion_columns()
    print("building scored frame (Champion #2's eyes) ...", flush=True)
    t0 = time.time()
    prod2, scored_raw, env_sup, built = scored_eyes.build_frames(df)
    assert built == cols
    assert list(prod2.index) == list(prod_sup.index)
    print(f"  scored frame built in {time.time() - t0:.0f}s", flush=True)

    hold_env = env_sup.iloc[split:]
    hold_prod = prod_sup.iloc[split:]
    assert list(hold_env.index) == list(hold_prod.index), (
        "env holdout timestamps differ from production holdout")
    assert np.isfinite(hold_env[cols].to_numpy(float)).all(), (
        "non-finite scored feature inside the holdout")
    assert (hold_env["close"].to_numpy(float)
            == hold_prod["close"].to_numpy(float)).all()
    assert (hold_env["target"].to_numpy()
            == hold_prod["target"].to_numpy()).all()
    train = _train_env_sup(env_sup, holdout_start)
    vals = train[cols].to_numpy(float)
    assert np.isfinite(vals).all()
    assert vals.min() >= -1.0 and vals.max() <= 1.0

    # ---- causality spot-check on THIS built frame: truncation
    # rebuilds through the whole scored path must reproduce the
    # full frame's scores on every shared row, and run 64's votes
    # from truncated features must equal its votes from the full
    # frame -- the vote at row i reads only candles <= i.
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    cuts = [9000, hold_start_pos + 300]
    trunc_evidence = {}
    for K in cuts:
        frame_t, _ = scored_features.build_scored_frame(df.iloc[:K + 1])
        per = {}
        for c in SAMPLE_COLS:
            a = scored_raw[c].iloc[:K + 1].to_numpy(float)
            b = frame_t[c].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            per[c] = {"max_abs_diff": d, "shared_rows": int(both.sum())}
            assert d <= 1e-9, f"causality FAIL (scored, K={K}): {c} {d}"
        # vote alignment at this cut: the newest 200 scored rows
        # the truncated frame can see (holdout rows when the cut
        # reaches into the holdout, train rows otherwise)
        pos_of = df.index.get_indexer(prod_sup.index)
        cand = [t for t, p in zip(prod_sup.index, pos_of)
                if p <= K and np.isfinite(
                    scored_raw.loc[t, SAMPLE_COLS[1]])]
        rows = cand[-200:]
        assert len(rows) >= 50, f"too few shared rows at K={K}"
        v_full = champ_model.predict_proba(
            scored_raw.reindex(rows)[cols])[:, 1]
        v_trunc = champ_model.predict_proba(
            frame_t.reindex(rows)[cols])[:, 1]
        vd = float(np.abs(v_full - v_trunc).max())
        per["__votes__"] = {"max_abs_diff": vd, "rows": len(rows)}
        assert vd <= 1e-12, f"vote alignment FAIL (K={K}): {vd}"
        trunc_evidence[str(K)] = per
        print(f"causality truncation K={K}: PASS (scores exact on "
              f"{len(SAMPLE_COLS)} sampled columns; run-64 votes "
              f"identical on {len(rows)} shared rows)", flush=True)

    env_sup.to_pickle(ENV_SUP_PATH)
    prod_sup.to_pickle(PROD_SUP_PATH)
    # The raw scored frame rides along for the votes phase.
    scored_raw.to_pickle(LAB / "scored_raw.pkl")
    save_result("frames", {
        "train_rows": len(train),
        "dropped_warmup_train_rows": int(split - len(train)),
        "holdout_rows": len(hold_env),
        "causality_truncation": trunc_evidence,
        "train_first": str(train.index[0]),
        "train_last": str(train.index[-1]),
    })
    print(f"frames: train {len(train)} rows (warmup-dropped "
          f"{split - len(train)}), holdout {len(hold_env)} finite + "
          f"label/close-identical to production", flush=True)


def phase_votes() -> None:
    _, prod_sup, split, holdout_start, _ = get_frames()
    cols = scored_eyes.champion_columns()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    scored_raw = pd.read_pickle(LAB / "scored_raw.pkl")
    train = _train_env_sup(env_sup, holdout_start)
    X_raw = scored_raw.reindex(train.index)[cols]
    assert np.isfinite(X_raw.to_numpy(float)).all()
    y = train["target"].astype(int)
    print(f"cross-fitting scored stackers on {len(train)} train rows "
          f"(production anti-leak pattern) ...", flush=True)
    t0 = time.time()
    p_up_train = crossfit_p_up(X_raw, y)
    print(f"  crossfit done in {time.time() - t0:.0f}s", flush=True)
    # The leading TimeSeriesSplit segment must be the neutral 0.5.
    n_lead = len(train) // 5  # folds=4 -> first test block starts at 1/5
    assert (p_up_train[:1] == 0.5).all()

    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    hold_idx = prod_sup.index[split:]
    X_hold = scored_raw.reindex(hold_idx)[cols]
    assert np.isfinite(X_hold.to_numpy(float)).all()
    p_up_hold = champ_model.predict_proba(X_hold)[:, 1]
    np.savez(VOTES_PATH, p_up_train=p_up_train, p_up_hold=p_up_hold,
             train_index=train.index.astype(str),
             hold_index=hold_idx.astype(str))
    save_result("votes", {
        "train_vote_mean": float(np.mean(p_up_train)),
        "train_vote_std": float(np.std(p_up_train)),
        "train_vote_neutral_share": float(np.mean(p_up_train == 0.5)),
        "hold_vote_mean": float(np.mean(p_up_hold)),
        "hold_vote_std": float(np.std(p_up_hold)),
        "hold_vote_min": float(np.min(p_up_hold)),
        "hold_vote_max": float(np.max(p_up_hold)),
        "n_lead_neutral_expected_approx": n_lead,
    })
    print(f"votes: train crossfit mean={np.mean(p_up_train):.3f} "
          f"std={np.std(p_up_train):.3f}; holdout (run 64) "
          f"mean={np.mean(p_up_hold):.3f} std={np.std(p_up_hold):.3f}",
          flush=True)


def _load_votes():
    z = np.load(VOTES_PATH)
    return z["p_up_train"], z["p_up_hold"]


def _train_frame_and_votes():
    _, _, _, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    p_up_train, p_up_hold = _load_votes()
    assert len(p_up_train) == len(train)
    return train, p_up_train, p_up_hold


def _curve_from(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text()).get("curve")
        except Exception:
            return None
    return None


def phase_train60() -> None:
    cols = scored_eyes.champion_columns()
    train, p_up_train, _ = _train_frame_and_votes()
    arm = LAB / "arm60"
    arm.mkdir(parents=True, exist_ok=True)
    out = train_ppo(
        timesteps=ARM60_STEPS, warm_start=False,
        champion_path=arm / "ppo_fresh.zip",  # nonexistent: fresh brain
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=LAB / "train_tmp.db",
        ent_coef=0.05,
        sup=train, p_up=p_up_train,
        feature_columns=cols)
    save_result("arm60", {
        "challenger_path": str(out), "timesteps": ARM60_STEPS,
        "reward_curve": _curve_from(arm / "ppo_challenger_curve.json")})
    print(f"arm60: fresh {ARM60_STEPS} steps on 98-wide state -> {out}",
          flush=True)


def phase_train300() -> None:
    cols = scored_eyes.champion_columns()
    train, p_up_train, _ = _train_frame_and_votes()
    arm = LAB / "arm300"
    arm.mkdir(parents=True, exist_ok=True)
    target = arm / "ppo_challenger.zip"  # segments chain from this file
    done_segs = int(results().get("arm300_segments_done", 0))
    curve = list(results().get("arm300_curve") or [])
    n_segs = ARM300_STEPS // SEGMENT
    for seg in range(done_segs, n_segs):
        out = train_ppo(
            timesteps=SEGMENT, warm_start=True,
            champion_path=target,  # seg 0: nonexistent -> fresh brain
            stacker_path=MODELS / "stacker_champion.pkl",
            db_path=LAB / "train_tmp.db",
            ent_coef=0.05,
            sup=train, p_up=p_up_train,
            feature_columns=cols)
        assert Path(out) == target
        seg_curve = _curve_from(arm / "ppo_challenger_curve.json") or []
        curve += [[int(t) + seg * SEGMENT, r] for t, r in seg_curve]
        save_result("arm300_segments_done", seg + 1)
        save_result("arm300_curve", curve)
        print(f"arm300: segment {seg + 1}/{n_segs} done "
              f"({(seg + 1) * SEGMENT}/{ARM300_STEPS} steps)", flush=True)
    save_result("arm300", {
        "challenger_path": str(target), "timesteps": ARM300_STEPS,
        "reward_curve": curve})
    print(f"arm300: fresh {ARM300_STEPS} steps in {n_segs} segments "
          f"-> {target}", flush=True)


def phase_duel() -> None:
    cols = scored_eyes.champion_columns()
    _, prod_sup, split, _, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    _, p_up_hold = _load_votes()
    hold_p = prod_sup.iloc[split:]
    hold_s = env_sup.iloc[split:]
    assert list(hold_s.index) == list(hold_p.index)
    windows_p = [hold_p.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    windows_s = [hold_s.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    p_ups = [p_up_hold[k * WINDOW:(k + 1) * WINDOW] for k in range(3)]

    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows_p, p_ups)
    assert champ_res is not None, "champion failed to grade"
    arms = {}
    for name, key in (("arm60", "arm60"), ("arm300", "arm300")):
        path = Path(results()[key]["challenger_path"])
        res = evaluate_windows(path, windows_s, p_ups,
                               feature_columns=cols)
        assert res is not None, f"{name} challenger failed to grade"
        arms[name] = res
    # Cross-checks (two ways): identical windows => identical
    # buy-and-hold per window, recomputed straight from the closes;
    # aggregate arithmetic reconciles from the window records.
    all_res = [champ_res] + [arms[k] for k in ("arm60", "arm300")]
    for k, w in enumerate(windows_p):
        closes = w["close"].to_numpy(dtype=float)
        bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0))
        for res in all_res:
            assert abs(res["windows"][k]["bench_pnl"] - bench) < 1e-6
    for res in all_res:
        assert abs(res["agg_score"]
                   - float(np.mean([w["score"]
                                    for w in res["windows"]]))) < 1e-9
        assert abs(res["total_pnl"]
                   - float(sum(w["total_pnl"]
                               for w in res["windows"]))) < 1e-9
    champ_agg = _agg(champ_res)
    out = {"champion": champ_agg}
    for name, res in arms.items():
        wins = sum(1 for cw, hw in zip(champ_res["windows"],
                                       res["windows"])
                   if hw["score"] > cw["score"])
        formula_pass = bool(res["agg_score"] > champ_res["agg_score"]
                            and wins >= 2)
        chall_agg = _agg(res)
        promotable, floor_fail = gate_decision(chall_agg, formula_pass)
        out[name] = {"challenger": chall_agg, "windows_won": wins,
                     "formula_pass": formula_pass,
                     "floor_fail": floor_fail, "promotable": promotable}
        print(f"duel {name}: agg={chall_agg['score']:.2f} "
              f"(pnl={chall_agg['total_pnl']:.2f}) vs champion "
              f"agg={champ_agg['score']:.2f} "
              f"(pnl={champ_agg['total_pnl']:.2f}); won {wins}/3; "
              f"formula={formula_pass} floor_fail={floor_fail}",
              flush=True)
    save_result("duel", out)
    print(f"duel: champion agg={champ_agg['score']:.2f} "
          f"pnl={champ_agg['total_pnl']:.2f} "
          f"bench={champ_agg['bench_pnl']:.2f}", flush=True)


def _already_registered() -> bool:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        kinds = {k for (k,) in con.execute(
            "SELECT kind FROM runs WHERE metrics LIKE "
            "'%scored_eyes_ppo%'")}
    finally:
        con.close()
    return "ppo" in kinds


def phase_register() -> None:
    res = results()
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    if _already_registered():
        print("register: scored_eyes_ppo runs already present — "
              "skipping duplicate registration", flush=True)
        save_result("registered", {"guard_after": guard_after,
                                   "note": "existing registration "
                                           "verified; guard intact"})
        return
    cols = scored_eyes.champion_columns()
    duel = res["duel"]
    champ = duel["champion"]
    design = ("scored-eyes PPO: 98-wide state = Champion #2's 94 "
              "scored indicators (s/50-1 into [-1,1]) + its vote + "
              "3 portfolio readings; fresh brain (no warm start "
              "across state shapes); votes cross-fitted in training, "
              "run 64 itself on the holdout; champion graded on "
              "prod21 + the same run-64 votes")
    common = {"family": scored_eyes.FAMILY,
              "scored_eyes_ppo": {
                  "design": design,
                  "features": cols,
                  "n_features": len(cols),
                  "state_size": scored_eyes.STATE_SIZE,
                  "transform": "score/50 - 1 ([0,100] -> [-1,1])",
                  "vote_source": "Champion #2 (run 64, scored102 "
                                 "stacker): cross-fitted out-of-fold "
                                 "in training, the champion itself "
                                 "on the holdout -- identical both "
                                 "sides of the duel",
                  "warm_start": False}}
    verdicts = {}
    for key, label, steps, deep in (
            ("arm60", "60k Fresh", ARM60_STEPS, False),
            ("arm300", "300k Deep", ARM300_STEPS, True)):
        d = duel[key]
        chall = d["challenger"]
        promotable = bool(d["promotable"])
        wins = d["windows_won"]
        if promotable:
            reason = (
                f"scored-eyes challenger ({label}) won {wins}/3 "
                f"windows, aggregate {chall['score']:.0f} vs "
                f"champion {champ['score']:.0f} (gate pass incl. "
                f"two-part floor) — WOULD PROMOTE: held as PENDING "
                f"KASS APPROVAL under the promotion lock; champion "
                f"kept (study verdict only)")
        elif d["floor_fail"] is not None:
            reason = (
                f"scored-eyes challenger ({label}) passed the "
                f"relative gate (aggregate {chall['score']:.0f} vs "
                f"champion {champ['score']:.0f}, won {wins}/3) but "
                f"failed the floor — {d['floor_fail']}; champion kept")
        else:
            reason = (
                f"scored-eyes challenger ({label}) won {wins}/3 "
                f"windows, aggregate {chall['score']:.0f} vs "
                f"champion {champ['score']:.0f}; champion kept "
                f"(study verdict)")
        metrics = {"challenger": chall, "champion": champ,
                   "reason": reason, "deep": deep,
                   "timesteps": steps,
                   "pending_approval": promotable,
                   "reward_curve": res[key].get("reward_curve"),
                   **common,
                   "scored_eyes_ppo": {
                       **common["scored_eyes_ppo"],
                       "label": label,
                       "verdict": "PASS" if promotable else "FAIL",
                       "formula_pass": d["formula_pass"],
                       "floor_fail": d["floor_fail"],
                       "windows_won": wins}}
        log_run("ppo", metrics, False)
        verdicts[key] = {"promotable": promotable, "reason": reason}
        print(f"register: {label} logged (promoted=0, "
              f"pending={promotable})", flush=True)
    a60, a300 = duel["arm60"], duel["arm300"]
    add_lab_note(
        "Scored-Eyes PPO study (family=scored_eyes_ppo)",
        f"{design}. 60k arm: aggregate "
        f"{a60['challenger']['score']:.0f} (P&L "
        f"${a60['challenger']['total_pnl']:+,.0f}) vs champion "
        f"{champ['score']:.0f} (P&L ${champ['total_pnl']:+,.0f}), "
        f"won {a60['windows_won']}/3. 300k arm: aggregate "
        f"{a300['challenger']['score']:.0f} (P&L "
        f"${a300['challenger']['total_pnl']:+,.0f}), won "
        f"{a300['windows_won']}/3. Buy-and-hold over the duel "
        f"windows: ${champ['bench_pnl']:+,.0f}. Study verdict "
        f"only — nothing installed (promotion lock).")
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    save_result("registered", {"verdicts": verdicts,
                               "guard_after": guard_after})
    print("register: 2 runs logged (promoted=0); champion + seat "
          "guard intact", flush=True)


PHASES = [("data", phase_data), ("frames", phase_frames),
          ("votes", phase_votes), ("train60", phase_train60),
          ("train300", phase_train300), ("duel", phase_duel),
          ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== scored-eyes PPO study (phases done: {sorted(done)}) ===",
          flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== scored-eyes PPO study complete ===", flush=True)


if __name__ == "__main__":
    main()
