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

from data.yfinance_adapter import fetch_klines
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
    """Deterministic rollouts on the holdout: chronological, non-overlapping
    episodes (240 candles / 24h = 10 days), paired across brains.

    Returns mean episode reward plus a risk-adjusted backtest score:
    score = total day-P&L - 0.5 * worst within-episode drawdown, where the
    drawdown is measured on each episode's own equity curve (audit
    2026-10-20: the old code drew the "drawdown" from the cumulative sum
    of per-episode P&Ls in seed order -- not chronological, not an equity
    curve -- so intra-episode drawdowns were invisible and the same market
    moves were double-counted across overlapping episodes).
    Promotion uses the score, not raw reward -- a challenger must make money
    without wild swings to take the crown.
    """
    import os

    from stable_baselines3 import PPO

    n = len(sup)
    assert n >= episodes * 24, f"holdout too short: {n} candles"
    starts = [k * 24 for k in range(episodes)]

    model = PPO.load(model_path)
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0}
    # Unique scratch DB per process (audit 2026-10-20): the fixed /tmp
    # path collided when a manual evaluation overlapped the nightly run.
    tmp = Path("/tmp") / f"tam_eval_{os.getpid()}.db"
    if tmp.exists():
        tmp.unlink()
    try:
        for ep, start in enumerate(starts):
            broker = PaperBroker(tmp, starting_balance=100_000,
                                 max_position_pct=0.25, stop_loss_pct=0.05)
            env = TradingEnv(sup, p_up, broker)
            obs, _ = env.reset(seed=ep, options={"start": start})
            start_eq = env._day_start_equity
            curve = [start_eq]  # within-episode equity curve
            done, ep_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[["hold", "buy", "sell"][int(action)]] += 1
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
    return {
        "mean_reward": float(np.mean(rewards)),
        "total_pnl": total,
        "max_drawdown": max_dd,
        "mean_drawdown": float(np.mean(ep_dds)) if ep_dds else 0.0,
        "score": total - 0.5 * max_dd,
        "episodes": episodes,
        "round_trips_per_day": float(np.mean(trips)) if trips else 0.0,
        "action_mix": {k: round(v / total_actions, 3)
                       for k, v in action_counts.items()},
    }


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

    # time holdout: last 10 days (240 hourly candles) for evaluation ONLY.
    # Nothing trains on it tonight (audit 2026-10-20: the PPO challenger
    # used to train on the full fetched series -- including these exact
    # candles -- minutes before being graded on them).
    split = len(sup) - 240
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min(), "train/holdout overlap"

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
    print(f"stacker: challenger holdout acc={chall_acc:.4f} (wf {score['accuracy']:.4f}) "
          f"champion holdout acc={champ_acc:.4f} promoted={stacker_promoted}", flush=True)

    # -- PPO challenger (warm-started) --
    # Trains ONLY on train_sup, with cross-fitted (out-of-fold) stacker
    # votes -- never on the holdout it is about to be graded on.
    from models.ppo_train import train as train_ppo
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold = stacker_model.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    p_up_train = crossfit_p_up(Xtr, ytr)
    challenger_path = train_ppo(
        timesteps=60000, warm_start=True,
        champion_path=MODELS / "ppo_champion.zip",
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=DATA / "train_tmp.db",  # throwaway ledger; live paper.db untouched
        ent_coef=0.05,  # keep exploring; sparse +/-1 rewards need it
        sup=train_sup, p_up=p_up_train,
    )
    if (MODELS / "ppo_champion.zip").exists():
        try:
            champ = evaluate_ppo(MODELS / "ppo_champion.zip", hold_sup, p_up_hold)
        except Exception as e:
            # e.g. champion trained on an old feature recipe can't even run
            # in the current env. An unrunnable champion scores worst, so a
            # runnable challenger takes the crown (manual recovery on
            # 2026-10-20 did exactly this; now it's the standard path).
            print(f"champion not runnable ({type(e).__name__}): {e}", flush=True)
            champ = {"mean_reward": -999.0, "total_pnl": 0.0,
                     "max_drawdown": 0.0, "score": -1e9, "runnable": False}
        chall = evaluate_ppo(challenger_path, hold_sup, p_up_hold)
        # Kass's rule: promote ONLY if strictly better.
        ppo_promoted = chall["score"] > champ["score"]
        print(f"ppo: challenger score={chall['score']:.2f} "
              f"(pnl={chall['total_pnl']:.2f} dd={chall['max_drawdown']:.2f}) "
              f"champion score={champ['score']:.2f} "
              f"(pnl={champ['total_pnl']:.2f} dd={champ['max_drawdown']:.2f}) "
              f"promoted={ppo_promoted}", flush=True)
        if champ.get("runnable") is False:
            ppo_reason = ("champion could not run in the current setup, "
                          "so the challenger takes the crown by default")
        elif ppo_promoted:
            ppo_reason = (f"challenger score {chall['score']:.0f} beat champion "
                          f"{champ['score']:.0f} on 10 chronological holdout days")
        else:
            ppo_reason = (f"champion kept: challenger score {chall['score']:.0f} "
                          f"did not beat champion {champ['score']:.0f}")
    else:
        champ = {"mean_reward": -1.0, "total_pnl": 0.0,
                 "max_drawdown": 0.0, "score": -1.0}
        chall = evaluate_ppo(challenger_path, hold_sup, p_up_hold)
        ppo_promoted = True
        ppo_reason = "no champion existed; first challenger promoted"
        print("ppo: no champion yet, promoting first challenger", flush=True)
    if ppo_promoted:
        challenger_path.replace(MODELS / "ppo_champion.zip")
    log_run("ppo", {"challenger": chall, "champion": champ,
                    "reason": ppo_reason}, ppo_promoted)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
