"""Live paper-trading loop (cron-driven, 24/7).

Designed to run every hour via cron (the parent owns the schedule). Each
invocation:
1. Loads the champion stacker + champion PPO.
2. Fetches the latest closed 1h candles; processes any not yet seen
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
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt state must not kill the loop (audit 2026-10-20):
            # rebuild the account from the ledger, which is durable.
            print("live_state.json unreadable; rebuilding from ledger", flush=True)
            return rebuild_state_from_ledger()
    return {"last_ts": None}


def rebuild_state_from_ledger() -> dict:
    """Reconstruct cash/BTC/entry/last_ts from paper.db.

    Every fill row records cash_after/btc_after, and entry is the
    volume-weighted average of the buys since the position was last flat
    -- the same math the broker itself maintains. Used only when the JSON
    state file is missing or corrupt; the JSON remains the fast path.
    """
    import sqlite3

    state: dict = {"last_ts": None}
    db = DATA / "paper.db"
    if not db.exists():
        return state
    con = sqlite3.connect(db)
    try:
        row = con.execute(
            "SELECT cash_after, btc_after FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
        eq = con.execute(
            "SELECT ts FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
        if row is None:
            return state
        cash, btc = float(row[0]), float(row[1])
        state.update({"cash": cash, "btc": btc,
                      "last_ts": eq[0] if eq else None})
        entry = None
        if btc > 0:
            fills = con.execute(
                "SELECT side, price, qty FROM trades ORDER BY id ASC").fetchall()
            held, cost = 0.0, 0.0
            for side, price, qty in fills:
                if side == "BUY":
                    cost += price * qty
                    held += qty
                else:
                    held -= qty
                    if held <= 1e-10:
                        held, cost = 0.0, 0.0
            entry = cost / held if held > 1e-10 else None
        state["entry_price"] = entry
        state["last_entry_price"] = entry
        return state
    finally:
        con.close()


def save_state(state: dict) -> None:
    """Atomic save (temp file + rename): a crash mid-write can never leave
    a half-written JSON for the next run to choke on."""
    import os

    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE_FILE)


def run_once() -> dict:
    stacker = load_stacker(MODELS / "stacker_champion.pkl")
    from stable_baselines3 import PPO
    ppo = PPO.load(MODELS / "ppo_champion.zip")

    # Feature-recipe upgrades change the observation dimension. If the
    # champion was trained on the old recipe, skip trading (rather than
    # crash) until a new champion is promoted by the training loop.
    expected_dim = len(FEATURE_COLUMNS) + 1 + 3  # features + p_up + portfolio
    model_dim = int(ppo.observation_space.shape[0])
    if model_dim != expected_dim:
        return {"processed": 0,
                "skipped": f"champion expects {model_dim}-dim obs, "
                           f"recipe now builds {expected_dim}-dim; "
                           f"awaiting new champion promotion"}

    broker = PaperBroker(DATA / "paper.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    state = load_state()
    # Restore the persisted paper account: without this, every hourly run
    # would start back at $100k/0 BTC and positions would never carry.
    # (Audit 2026-10-20: this restore was missing in the 15m era -- live
    # positions were silently dropped between runs until it was fixed.)
    if state.get("cash") is not None:
        broker.cash = float(state["cash"])
        broker.btc = float(state.get("btc", 0.0))
        broker.entry_price = state.get("entry_price")

    # ONE fetch (audit 2026-10-20): the old code fetched 5d to decide what
    # to process and 60d for features -- two snapshots of the market that
    # could disagree (KeyError surface), and the 5d processing window meant
    # an outage longer than 5 days silently skipped candles forever, with
    # the position unmanaged across the hole. Now the 60d frame serves
    # both, so catch-up covers any outage up to 60 days.
    full = fetch_klines("BTC-USD", interval="1h", period="60d").iloc[:-1]
    if full.empty:
        raise RuntimeError("market fetch returned no closed candles")
    # Staleness alarm: if the newest closed candle is old, the feed is
    # stale/cached -- fail LOUDLY instead of the old silent {"processed": 0}
    # success, so a dead feed can't masquerade as a quiet bot for days.
    newest = full.index[-1]
    age_h = (pd.Timestamp.now(tz=newest.tz) - newest).total_seconds() / 3600
    if age_h > 2.5:
        raise RuntimeError(
            f"stale market data: newest closed candle is {age_h:.1f}h old "
            f"({newest}); refusing to trade on it")
    if state["last_ts"] and pd.Timestamp(state["last_ts"]) < full.index[0]:
        print(f"WARNING: last processed candle {state['last_ts']} predates "
              f"the 60d fetch window (starts {full.index[0]}); any candles "
              f"in between are unrecoverable from this feed", flush=True)
    feat = add_features(full)
    df = full
    if state["last_ts"]:
        df = df[df.index > pd.Timestamp(state["last_ts"])]
    if df.empty:
        return {"processed": 0}

    processed = 0
    last_price = 0.0
    for ts, row in df.iterrows():
        frow = feat.loc[ts]
        if frow[FEATURE_COLUMNS].isna().any():
            # Do NOT skip-and-continue (audit 2026-10-20): later candles
            # would advance last_ts past this one and it would never be
            # processed. Stop here; the next run retries with fresher data.
            print(f"candle {ts} has NaN features; stopping run here", flush=True)
            break
        p_up = float(stacker.predict_proba(frow[FEATURE_COLUMNS].to_frame().T)[0, 1])
        price = float(row["close"])
        last_price = price
        # Build the observation exactly as TradingEnv._state does
        # (features v3 contract): features already in [-1, 1], stacker
        # probability centered to [-1, 1], portfolio values clipped.
        market = [float(frow[c]) for c in FEATURE_COLUMNS] + [2.0 * p_up - 1.0]
        eq = broker.equity(price)
        obs = np.array(market + [
            broker.cash / max(eq, 1e-9),
            broker.btc * price / max(eq, 1e-9),
            broker.unrealized_pct(price),
        ], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        action, _ = ppo.predict(obs, deterministic=True)
        ts_s = ts.isoformat()
        broker.check_stop(price, ts=ts_s, low=float(row["low"]))
        if int(action) == 1:
            broker.buy(price, reason="ppo_live", ts=ts_s)
        elif int(action) == 2:
            broker.sell(price, reason="ppo_live", ts=ts_s)
        broker.snapshot(ts_s, price)
        # Persist the account after EVERY candle (audit 2026-10-20): the
        # old end-of-run-only save meant a mid-loop crash left durable
        # fills in the ledger next to stale account state, and the next
        # run would reprocess the same candles into duplicate fills. The
        # dashboard reads last_entry_price; it stays in sync with the
        # broker's true entry (single source of truth -- also correct
        # after a stop-loss, which the old in-loop writes missed).
        state["last_ts"] = ts_s
        state["last_stance"] = ["hold", "buy", "sell"][int(action)]
        state["last_p_up"] = round(p_up, 4)
        state["cash"] = broker.cash
        state["btc"] = broker.btc
        state["entry_price"] = broker.entry_price
        state["last_entry_price"] = broker.entry_price
        save_state(state)
        processed += 1

    return {"processed": processed, "equity": broker.equity(last_price),
            "cash": broker.cash, "btc": broker.btc}


if __name__ == "__main__":
    print(json.dumps(run_once()), flush=True)
