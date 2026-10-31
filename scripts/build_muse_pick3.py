"""Muse's Pick #3 LAB study (2026-10-27, Kass's delegation).

Kass (2026-10-27 ~14:08 ET): "Do whatever you think is best. I
want you to try a completly different tactic for the ppo brain
aswell and go ahead and switch up how it trades and stops and
all that too if you want and then test your next model against
whatever champion you think is best so far".

The tactic (design frozen in refs/muse-pick-3-2026-10-27.md
BEFORE any result): the exposure-manager brain. Direction is
never the brain's choice -- it is run 73's vote (Lab Stacker
Champion #4, crowned earlier today under the same delegation),
sign(vote - 0.5). The brain picks EXPOSURE only: STAND ASIDE
(0), HALF (37.5% of equity), FULL (75%), HOLD (ride). Challenger
exits bind on the challenger side only (stop 1.5xATR [2%, 6%],
trail arm +2%, trail 1.25xATR [1.5%, 5%], 72h time-stop);
production broker.py untouched. Reward = the production dense
log-return reward + terminal +/-1, NO drawdown shaping (closed
twice), NO activity penalty (the frozen deviation). SB3 PPO
fresh, 60,000 steps, seed 42, ent_coef 0.05, 264-candle
training episodes, telemetry on. Training consumes run 73's
CROSS-FITTED votes (pick machinery); the holdout uses the
fitted champion's votes.

Duel: challenger (Pick #3 brain + run 73 eyes) vs the strongest
incumbent pair recomputed fresh -- run 44 PPO on its native
25-wide state with run 73's fitted holdout votes as its vote
input (the Pick #2 champion-side precedent), production broker
params. 3 windows x 10 episodes, next-candle-open fills, score
= P&L - 0.5 x max DD; formula (agg beats champion AND >= 2/3
windows) AND gate_decision()'s two-part floor.

Verdict under the delegation: formula AND floor pass -> LAB
PPO champion in the registry (promoted=1, provenance recorded;
the LIVE SEAT is explicitly unchanged -- run 44 keeps driving
Book #1; seat changes are Kass's own pick only). Either fails
-> recorded LOSS (fourteenth null), champion kept. Guards:
live seat + shadow seat + both book DBs untouched; lab/shadow
stacker hashes fixed by Phase 1; lab PPO hash changes only on
a crown (this script's register phase, archive-first).

Usage: .venv/bin/python scripts/build_muse_pick3.py
Phases checkpoint to data/muse_pick3/; reruns skip done phases.
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
from features import scored_eyes  # noqa: E402
from models import muse_pick3 as mp3  # noqa: E402
from models.muse_pick import crossfit_p_up_pick  # noqa: E402
from models.ppo_train import train as train_ppo  # noqa: E402
import models.ppo_train as ppo_train_mod  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import (  # noqa: E402
    _agg, add_lab_note, evaluate_windows, gate_decision, log_run)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "muse_pick3"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
ENV_SUP_PATH = LAB / "env_sup.pkl"
PROD_SUP_PATH = LAB / "prod_sup.pkl"
RAW98_PATH = LAB / "raw98.pkl"
VOTES_PATH = LAB / "votes.npz"
ARM = LAB / "arm60"

HOLDOUT = 792
WINDOW = 264
STEPS = 60_000
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_stacker_meta": MODELS / "live_seat" / "stacker.json",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
    "shadow_stacker": MODELS / "shadow_seat" / "stacker.pkl",
    "shadow_stacker_meta": MODELS / "shadow_seat" / "stacker.json",
    "shadow_ppo": MODELS / "shadow_seat" / "ppo.zip",
}
DESIGN_PPO = (
    "Muse's Pick #3 PPO (exposure manager): SB3 PPO, 102-wide "
    "scored state + run 73 votes; direction = sign(vote - 0.5) "
    "only; actions STAND ASIDE / HALF 37.5% / FULL 75% / HOLD; "
    "challenger exits stop 1.5xATR [2%,6%], trail +2% / "
    "1.25xATR [1.5%,5%], 72h time-stop; no shaping, no activity "
    "penalty; 264-candle training episodes")


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
        "paper_db_info_only": _sha(ROOT / "data" / "paper.db"),
        "paper_shadow_db_info_only": _sha(ROOT / "data" / "paper_shadow.db")}


def _assert_guard(before: dict, after: dict,
                  allow: set[str] = frozenset()) -> None:
    for k in GUARD_FILES:
        if k in allow:
            continue
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


def cols98() -> list[str]:
    meta = json.loads((MODELS / "stacker_champion.json").read_text())
    assert meta.get("run_id") == 73
    return list(meta["feature_columns"])


def _train_env_sup(env_sup, holdout_start):
    cols = cols98()
    train = env_sup[env_sup.index < holdout_start].dropna(subset=cols)
    assert len(train) > 5_000, f"pick3 train frame too small: {len(train)}"
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
        "state_size": len(cols98()) + 1 + 3,
        "n_features": len(cols98()),
        "eyes_run": 73,
        "design": {"ppo": DESIGN_PPO,
                   "episode_candles": 264,
                   "actions": mp3.ACTION_KEYS,
                   "half_pct": mp3.HALF_PCT, "full_pct": mp3.FULL_PCT,
                   "time_stop_candles": mp3.TIME_STOP_CANDLES,
                   "challenger_exits": {
                       "stop_atr_mult": mp3.CH_STOP_ATR_MULT,
                       "stop_min_pct": mp3.CH_STOP_MIN_PCT,
                       "stop_max_pct": mp3.CH_STOP_MAX_PCT,
                       "trail_activate_pct": mp3.CH_TRAIL_ACTIVATE_PCT,
                       "trail_atr_mult": mp3.CH_TRAIL_ATR_MULT,
                       "trail_min_pct": mp3.CH_TRAIL_MIN_PCT,
                       "trail_max_pct": mp3.CH_TRAIL_MAX_PCT}},
    })
    print(f"data: {len(df)} raw rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(prod_sup)}, holdout from {holdout_start}", flush=True)


def phase_frames() -> None:
    """Run 73's frames (prod / raw98 / env_sup), assembled exactly
    as Pick #2 assembled its own (mp3.build_pick3_frames), plus
    the holdout hygiene asserts."""
    df, prod_sup, split, holdout_start, _ = get_frames()
    cols = cols98()
    print("building pick3 frames (98-col pick2 frame + env sup) ...",
          flush=True)
    t0 = time.time()
    prod2, raw98, env_sup, built = mp3.build_pick3_frames(df)
    assert list(built) == cols
    assert len(prod2) == len(prod_sup)
    print(f"  frames built in {time.time() - t0:.0f}s", flush=True)
    hold_env = env_sup.iloc[split:]
    hold_prod = prod_sup.iloc[split:]
    assert list(hold_env.index) == list(hold_prod.index)
    assert np.isfinite(hold_env[cols].to_numpy(float)).all(), (
        "non-finite pick3 feature inside the holdout")
    assert (hold_env["close"].to_numpy(float)
            == hold_prod["close"].to_numpy(float)).all()
    assert (hold_env["target"].to_numpy()
            == hold_prod["target"].to_numpy()).all()
    train = _train_env_sup(env_sup, holdout_start)
    vals = train[cols].to_numpy(float)
    assert np.isfinite(vals).all()
    assert vals.min() >= -1.0 and vals.max() <= 1.0
    env_sup.to_pickle(ENV_SUP_PATH)
    prod_sup.to_pickle(PROD_SUP_PATH)
    raw98.to_pickle(RAW98_PATH)
    save_result("frames", {
        "train_rows": len(train),
        "dropped_warmup_train_rows": int(split - len(train)),
        "holdout_rows": len(hold_env),
        "train_first": str(train.index[0]),
        "train_last": str(train.index[-1]),
    })
    print(f"frames: train {len(train)} rows (warmup-dropped "
          f"{split - len(train)}), holdout {len(hold_env)} finite + "
          f"label/close-identical to production", flush=True)


def phase_votes() -> None:
    """Run 73's votes: CROSS-FITTED on the train region (the pick
    recipe -- run 73's own pipeline architecture, refit per
    fold) and FITTED (the installed Champion #4 artifact) on the
    holdout. One vote series serves both duel sides."""
    cols = cols98()
    _, prod_sup, split, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    raw98 = pd.read_pickle(RAW98_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    X_raw = raw98.reindex(train.index)[cols]
    assert np.isfinite(X_raw.to_numpy(float)).all()
    y = train["target"].astype(int)
    print(f"cross-fitting run-73-recipe stackers on {len(train)} "
          f"train rows (pick machinery) ...", flush=True)
    t0 = time.time()
    p_up_train = crossfit_p_up_pick(X_raw, y)
    print(f"  crossfit done in {time.time() - t0:.0f}s", flush=True)
    assert (p_up_train[:1] == 0.5).all()

    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    hold_idx = prod_sup.index[split:]
    X_hold = raw98.reindex(hold_idx)[cols]
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
    })
    print(f"votes: train crossfit (run-73 recipe) "
          f"mean={np.mean(p_up_train):.3f} std={np.std(p_up_train):.3f}; "
          f"holdout fitted mean={np.mean(p_up_hold):.3f}", flush=True)


def _load_votes():
    z = np.load(VOTES_PATH)
    return z["p_up_train"], z["p_up_hold"]


def _curve_and_telemetry(arm: Path):
    p = arm / "ppo_challenger_curve.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return d.get("curve"), d.get("telemetry")
        except Exception:
            return None, None
    return None, None


def phase_train() -> None:
    """60k fresh SB3 PPO on the exposure-manager env (264-candle
    episodes). The production TradingEnv is swapped for the study
    subclass inside ppo_train's module namespace for THIS call
    only and restored immediately after (Pick #1/#2 precedent)."""
    cols = cols98()
    _, _, _, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    p_up_train, _ = _load_votes()
    assert len(p_up_train) == len(train)
    ARM.mkdir(parents=True, exist_ok=True)
    original = ppo_train_mod.TradingEnv
    ppo_train_mod.TradingEnv = mp3.ExposureTrainEnv
    try:
        out = train_ppo(
            timesteps=STEPS, warm_start=False,
            champion_path=ARM / "ppo_fresh.zip",  # nonexistent: fresh
            stacker_path=MODELS / "stacker_champion.pkl",
            db_path=LAB / "train_tmp.db",
            ent_coef=0.05,
            sup=train, p_up=p_up_train,
            feature_columns=cols)
    finally:
        ppo_train_mod.TradingEnv = original
    assert ppo_train_mod.TradingEnv is original
    curve, telemetry = _curve_and_telemetry(ARM)
    save_result("arm", {"challenger_path": str(out), "timesteps": STEPS,
                        "episode_candles": 264,
                        "reward_curve": curve, "telemetry": telemetry})
    print(f"arm: fresh {STEPS} steps on 102-wide exposure-manager "
          f"state with 264-candle episodes -> {out}; "
          f"telemetry={'captured' if telemetry else 'MISSING'}",
          flush=True)


def phase_duel() -> None:
    cols = cols98()
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
    path = Path(results()["arm"]["challenger_path"])
    chall_windows = mp3.evaluate_exposure_windows(
        path, windows_s, p_ups, cols)
    res = {"windows": chall_windows,
           "agg_score": float(np.mean([w["score"]
                                       for w in chall_windows])),
           "total_pnl": float(np.sum([w["total_pnl"]
                                      for w in chall_windows])),
           "mean_reward": float(np.mean([w["mean_reward"]
                                         for w in chall_windows]))}
    for k, w in enumerate(windows_p):
        closes = w["close"].to_numpy(dtype=float)
        bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0))
        for r in (champ_res, res):
            assert abs(r["windows"][k]["bench_pnl"] - bench) < 1e-6
    for r in (champ_res, res):
        assert abs(r["agg_score"]
                   - float(np.mean([w["score"]
                                    for w in r["windows"]]))) < 1e-9
        assert abs(r["total_pnl"]
                   - float(sum(w["total_pnl"]
                               for w in r["windows"]))) < 1e-9
    champ_agg = _agg(champ_res)
    chall_agg = _agg(res)
    wins = sum(1 for cw, hw in zip(champ_res["windows"], res["windows"])
               if hw["score"] > cw["score"])
    formula_pass = bool(res["agg_score"] > champ_res["agg_score"]
                        and wins >= 2)
    promotable, floor_fail = gate_decision(chall_agg, formula_pass)
    save_result("duel", {
        "champion": champ_agg,
        "challenger": chall_agg, "windows_won": wins,
        "formula_pass": formula_pass, "floor_fail": floor_fail,
        "promotable": promotable})
    print(f"duel: pick3 agg={chall_agg['score']:.2f} "
          f"(pnl={chall_agg['total_pnl']:.2f}) vs champion "
          f"agg={champ_agg['score']:.2f} "
          f"(pnl={champ_agg['total_pnl']:.2f}); won {wins}/3; "
          f"formula={formula_pass} floor_fail={floor_fail} "
          f"promotable={promotable}", flush=True)


def _already_registered() -> int:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM runs WHERE metrics LIKE "
            "'%\"family\": \"muse_pick3\"%'").fetchone()[0]
    finally:
        con.close()
    return int(n)


def _newest_pick3_run_id() -> int:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        row = con.execute(
            "SELECT id, metrics FROM runs WHERE kind='ppo' "
            "ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        con.close()
    assert json.loads(row[1]).get("family") == "muse_pick3", (
        "newest ppo run is not the pick3 run -- registration drift")
    return int(row[0])


def phase_register() -> None:
    res = results()
    guard_before = res["meta"]["guard_before"]
    duel = res["duel"]
    champ = duel["champion"]
    chall = duel["challenger"]
    wins = duel["windows_won"]
    promotable = bool(duel["promotable"])
    cols = cols98()

    if _already_registered() >= 1:
        print("register: muse_pick3 run already present -- skipping "
              "duplicate registration", flush=True)
        save_result("registered", {
            "ppo_promotable": promotable,
            "note": "existing registration verified"})
        return

    crowned_run_id = None
    if promotable:
        # Verdict under Kass's 2026-10-27 delegation: crown the
        # LAB PPO champion. Archive first; the live seat (Book
        # #1) is NOT touched -- run 44 keeps driving it; only the
        # lab champion file and the roster's ppo_champion_run_id
        # record change.
        nightly.archive_champions("20261027_pre_pick3_ppo")
        import shutil
        shutil.copy2(Path(res["arm"]["challenger_path"]),
                     MODELS / "ppo_champion.zip")
        p_reason = (
            f"Muse's Pick #3 PPO (exposure manager) won {wins}/3 "
            f"windows, aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f}, and cleared the gate's two-part "
            f"floor — CROWNED LAB PPO CHAMPION under Kass's "
            f"2026-10-27 delegation ('do whatever you think is "
            f"best'); decided by Muse on the duel evidence. The "
            f"LIVE SEAT is unchanged: run 44 keeps driving Book #1 "
            f"(seat changes are Kass's own pick only)")
    elif duel["floor_fail"] is not None:
        p_reason = (
            f"Muse's Pick #3 PPO (exposure manager) passed the "
            f"relative gate (aggregate {chall['score']:.0f} vs "
            f"champion {champ['score']:.0f}, won {wins}/3) but "
            f"failed the floor — {duel['floor_fail']}; champion "
            f"kept (fourteenth null)")
    else:
        p_reason = (
            f"Muse's Pick #3 PPO (exposure manager) won {wins}/3 "
            f"windows, aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f}; champion kept (fourteenth null)")

    arm = res["arm"]
    p_metrics = {
        "challenger": chall, "champion": champ,
        "reason": p_reason, "deep": False,
        "timesteps": STEPS,
        "pending_approval": False,
        "reward_curve": arm.get("reward_curve"),
        "telemetry": arm.get("telemetry"),
        "family": mp3.FAMILY,
        "muse_pick3": {
            "label": "PPO",
            "verdict": "PASS" if promotable else "FAIL",
            "design": DESIGN_PPO,
            "features": cols, "n_features": len(cols),
            "state_size": len(cols) + 1 + 3,
            "transform": "score/50 - 1 ([0,100] -> [-1,1])",
            "vote_source": "run 73 (Champion #4) stacker: "
                           "cross-fitted out-of-fold (pick recipe) "
                           "in training, the fitted champion on "
                           "the holdout; direction = sign(vote-0.5)",
            "warm_start": False,
            "formula_pass": duel["formula_pass"],
            "floor_fail": duel["floor_fail"],
            "windows_won": wins,
            "training_episode_candles": 264,
            "actions": mp3.ACTION_KEYS,
            "challenger_exits": res["meta"]["design"][
                "challenger_exits"],
            "time_stop_candles": mp3.TIME_STOP_CANDLES,
            "reward": "production dense log-return % clipped "
                      "[-1,1] + terminal +/-1; NO drawdown "
                      "shaping; NO activity penalty (frozen "
                      "deviation: standing aside is first-class)",
            "crowned_lab_champion": promotable,
            "decided_by": ("Muse (delegated by Kass 2026-10-27)"
                           if promotable else None),
        }}
    if promotable:
        p_metrics["pending_approval"] = False
        p_metrics["pending_decision"] = "approved"
        p_metrics["pending_decided_by"] = (
            "Muse (delegated by Kass 2026-10-27)")
        from datetime import datetime, timezone
        p_metrics["pending_decided_at"] = datetime.now(
            timezone.utc).isoformat(timespec="seconds")
        p_metrics["pending_decision_reason"] = p_reason
    log_run("ppo", p_metrics, promotable)
    crowned_run_id = _newest_pick3_run_id() if promotable else None
    if promotable:
        # The champions roster reads the LAB PPO champion run id
        # from the seat ledger's top-level field; update the
        # record only -- the seat block (run 44 driving Book #1)
        # is untouched.
        seat_path = ROOT / "data" / "live_seat.json"
        seat = json.loads(seat_path.read_text())
        seat["ppo_champion_run_id"] = crowned_run_id
        seat_path.write_text(json.dumps(seat, indent=1))
    add_lab_note(
        "Muse's Pick #3 PPO (family=muse_pick3)",
        f"{DESIGN_PPO}. Fresh 60k steps, seed 42, consuming run "
        f"73's votes. Duel vs champion pair (run 44 PPO + run 73 "
        f"votes): aggregate {chall['score']:.0f} (P&L "
        f"${chall['total_pnl']:+,.0f}) vs champion "
        f"{champ['score']:.0f} (P&L ${champ['total_pnl']:+,.0f}), "
        f"won {wins}/3. Buy-and-hold over the duel windows: "
        f"${champ['bench_pnl']:+,.0f}. "
        f"{'CROWNED LAB PPO CHAMPION under the delegation; live seat unchanged (run 44 drives Book #1).' if promotable else 'No promotion — fourteenth null. Champion kept.'}")
    guard_after = _guard()
    _assert_guard(guard_before, guard_after,
                  allow={"lab_ppo"} if promotable else set())
    if promotable:
        assert guard_after["lab_ppo"] != guard_before["lab_ppo"]
    save_result("registered", {
        "ppo_promotable": promotable,
        "crowned_run_id": crowned_run_id,
        "guard_after": guard_after})
    print(f"register: Muse's Pick #3 PPO logged "
          f"(promoted={promotable}"
          f"{', run ' + str(crowned_run_id) + ' crowned LAB PPO champion' if promotable else ''}); "
          f"guards intact", flush=True)


PHASES = [("data", phase_data), ("frames", phase_frames),
          ("votes", phase_votes), ("train", phase_train),
          ("duel", phase_duel), ("register", phase_register)]


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    for name, fn in PHASES:
        if only and name != only:
            continue
        if name in done and not only:
            print(f"[{name}] already done -- skipping", flush=True)
            continue
        print(f"[{name}] running ...", flush=True)
        fn()
        mark(name)
    print("BUILD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
