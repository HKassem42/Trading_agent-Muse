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
          db_path: Path) -> Path:
    from stable_baselines3 import PPO

    df = fetch_klines("BTC-USD", interval="15m", period="60d")
    sup = make_supervised(df)
    stacker = load_stacker(stacker_path)
    p_up = stacker.predict_proba(sup[FEATURE_COLUMNS])[:, 1]

    broker = PaperBroker(db_path, starting_balance=10_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, p_up, broker)

    if warm_start and champion_path.exists():
        model = PPO.load(champion_path, env=env)
        print(f"warm-starting from {champion_path}")
    else:
        model = PPO("MlpPolicy", env, verbose=0, seed=42,
                    learning_rate=3e-4, n_steps=512, batch_size=64)
    model.learn(total_timesteps=timesteps)

    out = champion_path.parent / "ppo_challenger.zip"
    model.save(out)
    print(f"saved challenger -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--warm-start", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    train(args.timesteps, args.warm_start,
          root / "models" / "ppo_champion.zip",
          root / "models" / "stacker_champion.pkl",
          root / "data" / "paper.db")
