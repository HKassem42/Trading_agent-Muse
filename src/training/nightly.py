"""Nightly retrain: the learning loop. Runs at 00:00 ET via cron.

1. Pull fresh BTC-USD data, rebuild features.
2. Train stacker challenger; walk-forward score.
3. Warm-start PPO from champion weights; train challenger.
4. Evaluate champion vs challenger PPO on a time holdout (deterministic
   episodes, mean +/-1 reward). Promote only the winner.
5. Log the named run to the training registry.

Usage: python -m src.training.nightly
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.market_data import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, feature_matrix, make_supervised
from models import stacker
from trading.broker import PaperBroker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"


def registry_db() -> Path:
    p = DATA / "training.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                kind TEXT NOT NULL, metrics TEXT NOT NULL, promoted INTEGER NOT NULL)"""
        )
    return p


def log_run(kind: str, metrics: dict, promoted: bool) -> None:
    with sqlite3.connect(registry_db()) as con:
        con.execute(
            "INSERT INTO runs(ts,kind,metrics,promoted) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             kind, json.dumps(metrics), int(promoted)),
        )


def crossfit_p_up(X: pd.DataFrame, y: pd.Series, folds: int = 4) -> np.ndarray:
    """Out-of-fold stacker votes for the training frame.

    PPO trains on one of the 16 observation dims being the stacker's P(up).
    A stacker predicting on rows it was fit on produces systematically
    sharper votes than the out-of-sample votes the duel and the live loop
    produce (~0.50 coin flip) -- a train/serve distribution mismatch that
    teaches the brain to trust the stacker more than it should (audit
    2026-10-20). Here each vote comes from a pipeline fit ONLY on earlier
    data (time-series expanding windows, no future leakage). The leading
    segment (no past to fit on) gets the neutral 0.5.
    """
    from sklearn.model_selection import TimeSeriesSplit

    p_up = np.full(len(X), 0.5)
    tss = TimeSeriesSplit(n_splits=folds)
    for tr_idx, te_idx in tss.split(X):
        pipe = stacker.build_pipeline(random_state=42)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p_up[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:, 1]
    return p_up


def evaluate_ppo(model_path: Path, sup, p_up, episodes: int = 10) -> dict:
    """Deterministic rollouts on ONE evaluation window: chronological,
    non-overlapping episodes (240 decision candles / 24h = 10 days) with
    NEXT-CANDLE-OPEN fills (v4) -- the duel grades carry the execution lag
    every real fill has instead of the optimistic same-close fill.

    Returns mean episode reward plus a risk-adjusted backtest score:
    score = total day-P&L - 0.5 * worst within-episode drawdown, where the
    drawdown is measured on each episode's own equity curve (audit
    2026-10-20: the old code drew the "drawdown" from the cumulative sum
    of per-episode P&Ls in seed order -- not chronological, not an equity
    curve -- so intra-episode drawdowns were invisible and the same market
    moves were double-counted across overlapping episodes).
    Promotion uses the score, not raw reward -- a challenger must make money
    without wild swings to take the crown. `bench_pnl` is the buy-and-hold
    result over the same window: the yardstick that can't be flattered.
    """
    import os

    from stable_baselines3 import PPO

    n = len(sup)
    assert n >= episodes * 24, f"holdout too short: {n} candles"
    starts = [k * 24 for k in range(episodes)]

    model = PPO.load(model_path)
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    # Unique scratch DB per process (audit 2026-10-20): the fixed /tmp
    # path collided when a manual evaluation overlapped the nightly run.
    tmp = Path("/tmp") / f"tam_eval_{os.getpid()}.db"
    if tmp.exists():
        tmp.unlink()
    try:
        for ep, start in enumerate(starts):
            broker = PaperBroker(tmp, starting_balance=100_000,
                                 max_position_pct=0.25, stop_loss_pct=0.05)
            env = TradingEnv(sup, p_up, broker, fill_mode="next_open")
            obs, _ = env.reset(seed=ep, options={"start": start})
            start_eq = env._day_start_equity
            curve = [start_eq]  # within-episode equity curve
            done, ep_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[["hold", "buy", "sell", "short"][int(action)]] += 1
                obs, r, done, _, _ = env.step(int(action))
                ep_r += r
                curve.append(env._prev_eq)
            rewards.append(ep_r)
            pnls.append(env._prev_eq - start_eq)
            trips.append(env._round_trips)
            c = np.array(curve)
            ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
    finally:
        tmp.unlink(missing_ok=True)
    total = float(np.sum(pnls))
    max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
    total_actions = sum(action_counts.values()) or 1
    closes = sup["close"].to_numpy(dtype=float)
    bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0)) if len(closes) > 1 else 0.0
    return {
        "mean_reward": float(np.mean(rewards)),
        "total_pnl": total,
        "max_drawdown": max_dd,
        "mean_drawdown": float(np.mean(ep_dds)) if ep_dds else 0.0,
        "score": total - 0.5 * max_dd,
        "bench_pnl": bench,
        "episodes": episodes,
        "round_trips_per_day": float(np.mean(trips)) if trips else 0.0,
        "action_mix": {k: round(v / total_actions, 3)
                       for k, v in action_counts.items()},
    }


# -- v4 helpers: champion rollback shelf + lab notebook ----------------

def archive_champions(tag: str) -> None:
    """Copy the current champions onto the rollback shelf before any are
    replaced (v4): if a midnight promotion ever turns out bad, the
    previous brain can be restored in one step instead of retraining our
    way out. Keeps the newest 5 snapshots."""
    import shutil

    shelf = MODELS / "archive"
    shelf.mkdir(parents=True, exist_ok=True)
    dest = shelf / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{tag}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ppo_champion.zip", "stacker_champion.pkl",
                 "stacker_champion.json"):
        src = MODELS / name
        if src.exists():
            shutil.copy2(src, dest / name)
    snaps = sorted(p for p in shelf.iterdir() if p.is_dir())
    for old in snaps[:-5]:
        shutil.rmtree(old, ignore_errors=True)


def add_lab_note(title: str, detail: str) -> None:
    """Append an entry to the lab notebook (data/lab_notes.json): the
    human-readable paper trail of what was tried and what happened,
    surfaced in the Training Lab."""
    path = DATA / "lab_notes.json"
    notes = []
    if path.exists():
        try:
            notes = json.loads(path.read_text())
        except Exception:
            notes = []
    notes.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "title": title, "detail": detail})
    path.write_text(json.dumps(notes[-30:]))


def evaluate_windows(model_path: Path, windows, p_ups) -> dict | None:
    """Grade one brain across all gate windows; None if it cannot run."""
    per = []
    try:
        for w_sup, w_pup in zip(windows, p_ups):
            per.append(evaluate_ppo(model_path, w_sup, w_pup))
    except Exception as e:
        print(f"evaluation failed for {model_path} ({type(e).__name__}): {e}",
              flush=True)
        return None
    agg = float(np.mean([w["score"] for w in per]))
    return {"windows": per, "agg_score": agg,
            "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
            "mean_reward": float(np.mean([w["mean_reward"] for w in per]))}


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d")
    print(f"=== nightly retrain {run_id} ===", flush=True)

    # yfinance hiccups at midnight shouldn't kill the whole run: retry.
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h", period="730d")
            break
        except Exception as e:
            print(f"fetch attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
            import time
            time.sleep(30 * (attempt + 1))
    sup = make_supervised(df)
    X, y = feature_matrix(sup)

    # Time holdout: the last 33 days (792 hourly candles) are evaluation
    # ONLY, split into three chronological 11-day windows (264 candles:
    # 10 duel days + 1 buffer day for next-open fills). Nothing trains on
    # them tonight (audit 2026-10-20: the PPO challenger used to train on
    # the full fetched series -- including these exact candles -- minutes
    # before being graded on them). The 3-window gate (v4) means a
    # challenger can no longer take the crown off one lucky window.
    split = len(sup) - 792
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min(), "train/holdout overlap"
    windows_sup = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]

    # -- stacker challenger --
    # Promotion is judged on the SHARED holdout with a strict >, not by
    # comparing walk-forward accuracies computed on different windows at
    # different times (audit 2026-10-20: the old rule compared tonight's
    # challenger WF accuracy against the champion's stored WF accuracy
    # from its own older window, and promoted on ties).
    from sklearn.metrics import accuracy_score, precision_score

    Xtr, ytr = feature_matrix(train_sup)
    Xhold, yhold = feature_matrix(hold_sup)
    score = stacker.train_and_save(Xtr, ytr, MODELS / "stacker_challenger.pkl")
    challenger_model = joblib.load(MODELS / "stacker_challenger.pkl")
    chall_pred = challenger_model.predict(Xhold)
    chall_acc = float(accuracy_score(yhold, chall_pred))
    chall_prec = float(precision_score(yhold, chall_pred, zero_division=0))
    champ_pkl = MODELS / "stacker_champion.pkl"
    champ_score_path = MODELS / "stacker_champion.json"
    if champ_pkl.exists():
        champ_model = joblib.load(champ_pkl)
        champ_acc = float(accuracy_score(yhold, champ_model.predict(Xhold)))
    else:
        champ_acc = -1.0
    stacker_promoted = chall_acc > champ_acc
    stacker_reason = (
        f"stacker challenger holdout acc {chall_acc:.4f} > champion {champ_acc:.4f}"
        if stacker_promoted else
        f"stacker challenger holdout acc {chall_acc:.4f} did not beat champion {champ_acc:.4f}")
    # Rollback shelf (v4): snapshot the current champions BEFORE any
    # replacement tonight, so a bad promotion can be undone in one step.
    archive_champions(run_id)
    if stacker_promoted:
        (MODELS / "stacker_challenger.pkl").replace(champ_pkl)
        champ_score_path.write_text(json.dumps({
            "accuracy": chall_acc,
            "precision": chall_prec,
            "wf_accuracy": score["accuracy"],
            "wf_precision": score["precision"],
            "folds": score["folds"],
            "model_path": str(champ_pkl),
        }))
    log_run("stacker", {"challenger_acc": chall_acc,
                        "champion_acc": champ_acc,
                        "challenger_wf_acc": score["accuracy"],
                        "precision": chall_prec,
                        "reason": stacker_reason}, stacker_promoted)
    add_lab_note("Stacker duel", stacker_reason)
    print(f"stacker: challenger holdout acc={chall_acc:.4f} (wf {score['accuracy']:.4f}) "
          f"champion holdout acc={champ_acc:.4f} promoted={stacker_promoted}", flush=True)

    # -- PPO challenger (warm-started) --
    # Trains ONLY on train_sup, with cross-fitted (out-of-fold) stacker
    # votes -- never on the windows it is about to be graded on.
    import os

    from models.ppo_train import train as train_ppo
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold_all = stacker_model.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    p_up_windows = [p_up_hold_all[k * 264:(k + 1) * 264] for k in range(3)]
    p_up_train = crossfit_p_up(Xtr, ytr)
    # Weekly deep train (v4): TAM_DEEP=1 -> 300k steps instead of 60k.
    deep = bool(os.environ.get("TAM_DEEP"))
    timesteps = 300_000 if deep else 60_000
    challenger_path = train_ppo(
        timesteps=timesteps, warm_start=True,
        champion_path=MODELS / "ppo_champion.zip",
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=DATA / "train_tmp.db",  # throwaway ledger; live paper.db untouched
        ent_coef=0.05,  # keep exploring; sparse +/-1 rewards need it
        sup=train_sup, p_up=p_up_train,
    )

    def _agg(res) -> dict:
        """Flatten a 3-window evaluation into the run-record shape."""
        if res is None:
            return {"mean_reward": -999.0, "total_pnl": 0.0,
                    "max_drawdown": 0.0, "score": -1e9, "runnable": False}
        ws = res["windows"]
        return {"score": res["agg_score"], "total_pnl": res["total_pnl"],
                "mean_reward": res["mean_reward"],
                "max_drawdown": float(max(w["max_drawdown"] for w in ws)),
                "bench_pnl": float(sum(w["bench_pnl"] for w in ws)),
                "action_mix": ws[-1]["action_mix"],
                "windows": [{"score": round(w["score"], 2),
                             "pnl": round(w["total_pnl"], 2),
                             "bench": round(w["bench_pnl"], 2)} for w in ws],
                "runnable": True}

    if (MODELS / "ppo_champion.zip").exists():
        champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                     windows_sup, p_up_windows)
        chall_res = evaluate_windows(challenger_path, windows_sup, p_up_windows)
        champ, chall = _agg(champ_res), _agg(chall_res)
        wins = 0
        if champ_res and chall_res:
            wins = sum(1 for cw, hw in zip(champ_res["windows"],
                                           chall_res["windows"])
                       if hw["score"] > cw["score"])
        # The gate (v4): aggregate score strictly better AND a majority of
        # windows won. One hot window can no longer crown a brain.
        ppo_promoted = chall_res is not None and (
            champ_res is None or (chall["score"] > champ["score"] and wins >= 2))
        print(f"ppo gate: challenger agg={chall['score']:.2f} "
              f"(pnl={chall['total_pnl']:.2f}) champion agg={champ['score']:.2f} "
              f"(pnl={champ['total_pnl']:.2f}) windows won={wins}/3 "
              f"deep={deep} promoted={ppo_promoted}", flush=True)
        if champ_res is None:
            # e.g. champion trained on an old feature recipe can't even run
            # in the current env; a runnable challenger takes the crown
            # (manual recovery on 2026-10-20; now the standard path).
            ppo_reason = ("champion could not run in the current setup, "
                          "so the challenger takes the crown by default")
        elif chall_res is None:
            ppo_reason = "challenger failed evaluation; champion kept"
        elif ppo_promoted:
            ppo_reason = (f"challenger won {wins}/3 windows, aggregate score "
                          f"{chall['score']:.0f} vs champion {champ['score']:.0f} "
                          f"(next-candle fills; buy-and-hold over the same "
                          f"windows: ${chall['bench_pnl']:+,.0f})")
        else:
            ppo_reason = (f"champion kept: challenger won {wins}/3 windows, "
                          f"aggregate {chall['score']:.0f} vs champion "
                          f"{champ['score']:.0f}")
    else:
        chall = _agg(evaluate_windows(challenger_path, windows_sup, p_up_windows))
        champ = {"mean_reward": -1.0, "total_pnl": 0.0,
                 "max_drawdown": 0.0, "score": -1.0}
        ppo_promoted = True
        ppo_reason = "no champion existed; first challenger promoted"
        print("ppo: no champion yet, promoting first challenger", flush=True)
    if ppo_promoted:
        challenger_path.replace(MODELS / "ppo_champion.zip")
    reward_curve = None
    curve_path = MODELS / "ppo_challenger_curve.json"
    if curve_path.exists():
        try:
            reward_curve = json.loads(curve_path.read_text()).get("curve")
        except Exception:
            reward_curve = None
    log_run("ppo", {"challenger": chall, "champion": champ,
                    "reason": ppo_reason, "deep": deep,
                    "timesteps": timesteps,
                    "reward_curve": reward_curve}, ppo_promoted)
    add_lab_note("PPO promotion gate", ppo_reason)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
