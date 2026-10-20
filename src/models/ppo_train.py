"""PPO agent training.

Trains (or warm-starts) the PPO policy on the TradingEnv. The nightly job
calls this with warm_start=True so the agent keeps learning instead of
starting over — continual learning, per the design.

Usage:
    python -m src.models.ppo_train --timesteps 20000
    python -m src.models.ppo_train --timesteps 20000 --warm-start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.yfinance_adapter import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, make_supervised
from models.stacker import load as load_stacker
from trading.broker import PaperBroker


def train(timesteps: int, warm_start: bool, champion_path: Path, stacker_path: Path,
          db_path: Path, ent_coef: float = 0.01) -> Path:
    from stable_baselines3 import PPO

    df = fetch_klines("BTC-USD", interval="15m", period="60d")
    sup = make_supervised(df)
    stacker = load_stacker(stacker_path)
    p_up = stacker.predict_proba(sup[FEATURE_COLUMNS])[:, 1]

    # Unique scratch ledger per invocation: two trainings sharing one SQLite
    # file collide with "database is locked" (bit us 2026-10-20 when a manual
    # run and the nightly run overlapped). PID-suffix + cleanup prevents it.
    import os
    work_db = db_path.with_name(f"{db_path.stem}_{os.getpid()}{db_path.suffix}")
    broker = PaperBroker(work_db, starting_balance=10_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, p_up, broker)

    if warm_start and champion_path.exists():
        try:
            model = PPO.load(champion_path, env=env)
            print(f"warm-starting from {champion_path}")
        except Exception as e:
            # e.g. observation-dim mismatch after a feature-recipe change:
            # fall back to a fresh brain rather than crashing the nightly run
            print(f"warm-start failed ({type(e).__name__}), training fresh: {e}")
            model = None
    else:
        model = None
    if model is None:
        model = PPO("MlpPolicy", env, verbose=0, seed=42,
                    learning_rate=3e-4, n_steps=512, batch_size=64,
                    ent_coef=ent_coef)
    model.learn(total_timesteps=timesteps)

    out = champion_path.parent / "ppo_challenger.zip"
    model.save(out)
    print(f"saved challenger -> {out}")
    try:
        work_db.unlink(missing_ok=True)
    except OSError:
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--warm-start", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    # NEVER train against the live paper.db ledger -- scratch file only.
    train(args.timesteps, args.warm_start,
          root / "models" / "ppo_champion.zip",
          root / "models" / "stacker_champion.pkl",
          root / "data" / "train_tmp.db")
