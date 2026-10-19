"""Live paper-trading loop (cron-driven, 24/7).

Designed to run every 15 minutes via cron. Each invocation:
1. Loads the champion stacker + champion PPO.
2. Fetches the latest closed 15m candles; processes any not yet seen
   (tracked in data/live_state.json, so restarts are safe).
3. For each new candle: compute features -> stacker P(up) -> PPO action
   (0 hold / 1 buy / 2 sell) -> broker executes with 25% cap + 5% stop-loss.
4. Snapshots equity to the ledger after every candle.

Usage: python -m src.trading.live
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.yfinance_adapter import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, add_features
from models.stacker import load as load_stacker
from trading.broker import PaperBroker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"
STATE_FILE = DATA / "live_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_ts": None}


def save_state(state: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def run_once() -> dict:
    stacker = load_stacker(MODELS / "stacker_champion.pkl")
    from stable_baselines3 import PPO
    ppo = PPO.load(MODELS / "ppo_champion.zip")

    broker = PaperBroker(DATA / "paper.db", starting_balance=10_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    state = load_state()

    df = fetch_klines("BTC-USD", interval="15m", period="5d")
    # drop the still-forming current candle
    df = df.iloc[:-1]
    if state["last_ts"]:
        df = df[df.index > pd.Timestamp(state["last_ts"])]
    if df.empty:
        return {"processed": 0}

    # need enough history for indicators: refetch a wider window for features
    full = fetch_klines("BTC-USD", interval="15m", period="60d").iloc[:-1]
    feat = add_features(full)
    processed = 0
    last_price = 0.0
    for ts, row in df.iterrows():
        frow = feat.loc[ts]
        if frow[FEATURE_COLUMNS].isna().any():
            continue
        p_up = float(stacker.predict_proba(frow[FEATURE_COLUMNS].to_frame().T)[0, 1])
        price = float(row["close"])
        last_price = price
        market = [float(frow[c]) for c in FEATURE_COLUMNS] + [p_up]
        eq = broker.equity(price)
        obs = np.array(market + [
            broker.cash / max(eq, 1e-9),
            broker.btc * price / max(eq, 1e-9),
            broker.unrealized_pct(price),
        ], dtype=np.float32)
        action, _ = ppo.predict(obs, deterministic=True)
        ts_s = ts.isoformat()
        broker.check_stop(price, ts=ts_s)
        if int(action) == 1:
            fill = broker.buy(price, reason="ppo_live", ts=ts_s)
            if fill and broker.entry_price:
                state["last_entry_price"] = broker.entry_price
        elif int(action) == 2:
            broker.sell(price, reason="ppo_live", ts=ts_s)
            if broker.btc == 0:
                state["last_entry_price"] = None
        broker.snapshot(ts_s, price)
        state["last_ts"] = ts_s
        state["last_stance"] = ["hold", "buy", "sell"][int(action)]
        state["last_p_up"] = round(p_up, 4)
        processed += 1

    save_state(state)
    return {"processed": processed, "equity": broker.equity(last_price),
            "cash": broker.cash, "btc": broker.btc}


if __name__ == "__main__":
    print(json.dumps(run_once()), flush=True)
