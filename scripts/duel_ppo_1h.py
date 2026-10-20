"""Duel + promotion for the 1h / features-v3 PPO brain (2026-10-20).

Evaluates models/ppo_challenger.zip on the holdout (last 240 hourly
candles = 10 days) with the standard risk-adjusted score
(score = total P&L - 0.5 * max drawdown) and duels it against the
do-nothing baseline (score exactly 0.0 -- the old 15m champion cannot run
in the 1h/v3 env at all). The new brain becomes the live champion for the
1h era regardless of the duel outcome, for the same documented reason as
the 2026-10-20 recovery promotion: it is the only runnable brain. The old
champion stays on disk as models/ppo_champion_15m_backup.zip.

Usage: .venv/bin/python scripts/duel_ppo_1h.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from data.yfinance_adapter import fetch_klines
from features.technical_indicators import FEATURE_COLUMNS, make_supervised
from src.training.nightly import evaluate_ppo, log_run

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HOLDOUT_ROWS = 240


def main() -> None:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    sup = make_supervised(df)
    hold_sup = sup.iloc[-HOLDOUT_ROWS:]
    stacker = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold = stacker.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]

    chall = evaluate_ppo(MODELS / "ppo_challenger.zip", hold_sup, p_up_hold,
                         episodes=10)
    baseline = {"score": 0.0, "total_pnl": 0.0, "max_drawdown": 0.0,
                "policy": "do_nothing (old 15m champion unrunnable on "
                          "1h/features-v3; kept as backup file)"}
    print(f"challenger: score={chall['score']:.2f} pnl={chall['total_pnl']:.2f} "
          f"dd={chall['max_drawdown']:.2f} mean_reward={chall['mean_reward']:.3f} "
          f"trips/day={chall['round_trips_per_day']:.2f} "
          f"mix={chall['action_mix']} episodes={chall['episodes']}", flush=True)
    print(f"baseline do-nothing: score=0.00  "
          f"beats_baseline={chall['score'] > 0.0}", flush=True)

    reason = ("1h timeframe + features v3 migration: old champion is a 15m/v2 "
              "brain and cannot run in the new env; new brain promoted as the "
              "only runnable champion for the 1h era regardless of duel "
              "outcome (mirrors the 2026-10-20 recovery promotion)")
    (MODELS / "ppo_challenger.zip").replace(MODELS / "ppo_champion.zip")
    log_run("ppo", {"challenger": chall, "champion": baseline,
                    "timeframe": "1h", "features": "v3",
                    "promotion_reason": reason,
                    "holdout_rows": HOLDOUT_ROWS,
                    "duel_vs": "do_nothing baseline (score 0.0)"}, True)
    print("promoted challenger -> ppo_champion.zip; run logged", flush=True)


if __name__ == "__main__":
    main()
