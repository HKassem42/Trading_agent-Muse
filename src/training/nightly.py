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


def evaluate_ppo(model_path: Path, sup, p_up, episodes: int = 10) -> dict:
    """Deterministic rollouts on the holdout.

    Returns mean episode reward plus a risk-adjusted backtest score:
    score = total day-P&L - 0.5 * max drawdown of the cumulative P&L curve.
    Promotion uses the score, not raw reward — a challenger must make money
    without wild swings to take the crown.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    rewards, pnls, trips = [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0}
    tmp = Path("/tmp") / "tam_eval.db"
    if tmp.exists():
        tmp.unlink()
    for ep in range(episodes):
        broker = PaperBroker(tmp, starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = TradingEnv(sup, p_up, broker)
        obs, _ = env.reset(seed=ep)
        start_eq = env._day_start_equity
        done, ep_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[["hold", "buy", "sell"][int(action)]] += 1
            obs, r, done, _, _ = env.step(int(action))
            ep_r += r
        rewards.append(ep_r)
        pnls.append(env._prev_eq - start_eq)
        trips.append(env._round_trips)
    pnls = np.array(pnls)
    curve = np.cumsum(pnls)
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max(peak - curve)) if len(curve) else 0.0
    total = float(np.sum(pnls))
    total_actions = sum(action_counts.values()) or 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "total_pnl": total,
        "max_drawdown": max_dd,
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

    # time holdout: last 10 days (240 hourly candles) for PPO evaluation
    split = len(sup) - 240
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]

    # -- stacker challenger --
    Xtr, ytr = feature_matrix(train_sup)
    score = stacker.train_and_save(Xtr, ytr, MODELS / "stacker_challenger.pkl")
    champ_score_path = MODELS / "stacker_champion.json"
    champ_acc = json.loads(champ_score_path.read_text())["accuracy"] if champ_score_path.exists() else -1
    stacker_promoted = score["accuracy"] >= champ_acc
    if stacker_promoted:
        (MODELS / "stacker_challenger.pkl").replace(MODELS / "stacker_champion.pkl")
        champ_score_path.write_text(json.dumps(score))
    log_run("stacker", {**score, "challenger_acc": score["accuracy"],
                        "champion_acc": champ_acc}, stacker_promoted)
    print(f"stacker: challenger acc={score['accuracy']:.4f} "
          f"champion acc={champ_acc:.4f} promoted={stacker_promoted}", flush=True)

    # -- PPO challenger (warm-started) --
    from models.ppo_train import train as train_ppo
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold = stacker_model.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    challenger_path = train_ppo(
        timesteps=60000, warm_start=True,
        champion_path=MODELS / "ppo_champion.zip",
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=DATA / "train_tmp.db",  # throwaway ledger; live paper.db untouched
        ent_coef=0.05,  # keep exploring; sparse +/-1 rewards need it
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
    else:
        champ = {"mean_reward": -1.0, "total_pnl": 0.0,
                 "max_drawdown": 0.0, "score": -1.0}
        chall = evaluate_ppo(challenger_path, hold_sup, p_up_hold)
        ppo_promoted = True
        print("ppo: no champion yet, promoting first challenger", flush=True)
    if ppo_promoted:
        challenger_path.replace(MODELS / "ppo_champion.zip")
    log_run("ppo", {"challenger": chall, "champion": champ}, ppo_promoted)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
