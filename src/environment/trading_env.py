"""PPO trading environment (gymnasium).

From the Notion blueprint:
- State: market data + technical indicators + portfolio status (+ stacker vote).
- Actions: 0 = hold, 1 = buy, 2 = sell.
- Reward: exactly as specified — per episode (one trading day),
  +1 if the day's net paper P&L > 0, else -1. Sparse, honest, paper-only.

The env wraps PaperBroker so fills, fees, stop-loss and the ledger are real.
One step = one 15m candle. Episode = 96 steps (one day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from features.technical_indicators import FEATURE_COLUMNS
from trading.broker import PaperBroker

STATE_FEATURES = FEATURE_COLUMNS + ["stacker_p_up"]


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        stacker_p_up: np.ndarray | pd.Series,
        broker: PaperBroker,
        episode_days: int = 1,
        steps_per_day: int = 96,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.stacker_p_up = np.asarray(stacker_p_up, dtype=float)
        assert len(self.stacker_p_up) == len(self.df), "stacker votes must align with df"
        self.broker = broker
        self.episode_len = episode_days * steps_per_day

        # state: 10 market features + cash_frac + btc_frac + unrealized_pct
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(STATE_FEATURES) + 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold / buy / sell
        self._cursor = 0
        self._episode_start = 0
        self._day_start_equity = 0.0

    # -- helpers ------------------------------------------------------
    def _state(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        price = float(row["close"])
        market = [float(row[c]) for c in FEATURE_COLUMNS] + [float(self.stacker_p_up[i])]
        eq = self.broker.equity(price)
        port = [
            self.broker.cash / max(eq, 1e-9),
            self.broker.btc * price / max(eq, 1e-9),
            self.broker.unrealized_pct(price),
        ]
        return np.array(market + port, dtype=np.float32)

    # -- gym API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.df) - self.episode_len - 1
        self._episode_start = int(self.np_random.integers(0, max_start))
        self._cursor = self._episode_start
        price = float(self.df.iloc[self._cursor]["close"])
        self._day_start_equity = self.broker.equity(price)
        return self._state(self._cursor), {}

    def step(self, action: int):
        i = self._cursor
        row = self.df.iloc[i]
        price = float(row["close"])
        ts = str(row.name)

        # stop-loss enforced by the broker on every price, like live
        self.broker.check_stop(price, ts=ts)
        if action == 1:
            self.broker.buy(price, reason="ppo", ts=ts)
        elif action == 2:
            self.broker.sell(price, reason="ppo", ts=ts)

        self._cursor += 1
        done = (self._cursor - self._episode_start) >= self.episode_len
        done = bool(done) or self._cursor >= len(self.df) - 1

        reward = 0.0
        if done:
            day_pnl = self.broker.equity(price) - self._day_start_equity
            reward = 1.0 if day_pnl > 0 else -1.0
        obs = self._state(min(self._cursor, len(self.df) - 1))
        return obs, reward, done, False, {}
