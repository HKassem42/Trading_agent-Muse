"""One-off PPO training for the 1h / features-v3 migration (2026-10-20).

Fresh brain, 200k timesteps in 4 chunks of 50k. After each chunk the model
is checkpointed to models/ppo_1h_ckpt.zip and progress recorded in
models/ppo_1h_progress.json; re-running the script resumes from the
checkpoint instead of starting over (runtime restarts must not zero
progress). Training data: 1h BTC-USD, all but the last 240 candles
(10 days), which are reserved as the duel holdout. On completion the final
model is copied to models/ppo_challenger.zip.

Usage: .venv/bin/python scripts/train_ppo_1h.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from data.yfinance_adapter import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, make_supervised
from models.stacker import load as load_stacker
from trading.broker import PaperBroker

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data"
CKPT = MODELS / "ppo_1h_ckpt.zip"
PROGRESS = MODELS / "ppo_1h_progress.json"
TOTAL_STEPS = 200_000
CHUNK = 50_000
HOLDOUT_ROWS = 240


def build_env():
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    sup = make_supervised(df).iloc[:-HOLDOUT_ROWS]
    stacker = load_stacker(MODELS / "stacker_champion.pkl")
    p_up = stacker.predict_proba(sup[FEATURE_COLUMNS])[:, 1]
    work_db = DATA / f"ppo_1h_train_{os.getpid()}.db"
    broker = PaperBroker(work_db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    return TradingEnv(sup, p_up, broker), work_db


def main() -> None:
    from stable_baselines3 import PPO

    done_steps = 0
    if PROGRESS.exists():
        done_steps = int(json.loads(PROGRESS.read_text())["steps_done"])
    print(f"starting at {done_steps}/{TOTAL_STEPS} steps", flush=True)

    env, work_db = build_env()
    try:
        if done_steps > 0 and CKPT.exists():
            model = PPO.load(CKPT, env=env)
            print(f"resumed from checkpoint {CKPT}", flush=True)
        else:
            model = PPO("MlpPolicy", env, verbose=0, seed=42,
                        learning_rate=3e-4, n_steps=512, batch_size=64,
                        ent_coef=0.05)  # explore; matches nightly setting
        while done_steps < TOTAL_STEPS:
            t0 = time.time()
            model.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
            done_steps += CHUNK
            model.save(CKPT)
            PROGRESS.write_text(json.dumps(
                {"steps_done": done_steps, "total": TOTAL_STEPS}))
            print(f"chunk done: {done_steps}/{TOTAL_STEPS} "
                  f"in {time.time() - t0:.0f}s", flush=True)
        out = MODELS / "ppo_challenger.zip"
        model.save(out)
        print(f"training complete -> {out}", flush=True)
    finally:
        try:
            work_db.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
