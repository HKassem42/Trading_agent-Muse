"""PPO trading environment (gymnasium).

From the Notion blueprint:
- State: market data + technical indicators + portfolio status (+ stacker vote).
- Actions: 0 = hold, 1 = buy, 2 = sell, 3 = short (v5: the broker's
  position-aware semantics -- buy covers a short, short closes a long
  through flat, sell reduces a long; one action never flips directly).
- Reward: exactly as specified — per episode (one trading day),
  +1 if the day's net paper P&L > 0, else -1. Sparse, honest, paper-only.

The env wraps PaperBroker so fills, fees, stop-loss and the ledger are real.
One step = one 1h candle. Episode = 24 steps (one day).

Scoring (reward) design, per Kass 2026-10-19 ("however you think the scoring
should go so we get results"):
- Dense per-step signal: mark-to-market percent return of the paper account
  each 1h step, in percentage points, clipped to [-1, 1]. Tells the agent
  immediately whether its position is paying -- price moves, fees and the
  stop-loss all flow through equity, so overtrading is punished by fees and
  bag-holding losers is punished by the market.
- Sparse terminal signal (the original spec, kept): +1 if the day's net
  paper P&L > 0, else -1. The official daily verdict.
Total episode return ~= sum of step returns + the daily verdict.
"""

from __future__ import annotations

import json

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
        steps_per_day: int = 24,
        min_round_trips: int = 2,
        fill_mode: str = "close",
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.stacker_p_up = np.asarray(stacker_p_up, dtype=float)
        assert len(self.stacker_p_up) == len(self.df), "stacker votes must align with df"
        self.broker = broker
        self.episode_len = episode_days * steps_per_day
        # Kass 2026-10-19: the agent must actually trade to learn -- at least
        # `min_round_trips` completed buy->sell round trips per episode,
        # else a shortfall penalty at episode end (-0.2 per missing trip).
        # Scaled 2026-10-20 for the 1h migration: 5 trips per 96 15m candles
        # -> 2 trips per 24 1h candles keeps roughly the same trading
        # density (one trip per ~12-19 candles) without forcing churn on
        # a timeframe where each candle carries 4x the information.
        self.min_round_trips = min_round_trips
        self._round_trips = 0
        # Fill timing (v4): "close" = orders fill at the decision candle's
        # close (training default; matches live, where the decision happens
        # the moment the candle closes). "next_open" = orders fill at the
        # NEXT candle's open -- used by duel evaluation so grades carry the
        # execution lag every real fill has, instead of the optimistic
        # same-close fill.
        assert fill_mode in ("close", "next_open")
        self.fill_mode = fill_mode

        # Features v3 contract (2026-10-20): every observation component
        # lives in [-1, 1] -- features by construction, the rest by the
        # transforms in _state plus a final clip guard.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(STATE_FEATURES) + 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # hold / buy / sell / short
        self._cursor = 0
        self._episode_start = 0
        self._day_start_equity = 0.0

    # -- helpers ------------------------------------------------------
    def _state(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        price = float(row["close"])
        # stacker p_up is a probability in [0, 1]; center it to [-1, 1].
        p_up_c = 2.0 * float(self.stacker_p_up[i]) - 1.0
        market = [float(row[c]) for c in FEATURE_COLUMNS] + [p_up_c]
        eq = self.broker.equity(price)
        # cash/btc fractions already live in [0, 1]; unrealized_pct is an
        # unbounded return, so clip it. Final clip is a belt-and-braces
        # guard for the whole vector (features v3 contract).
        port = [
            self.broker.cash / max(eq, 1e-9),
            self.broker.btc * price / max(eq, 1e-9),
            self.broker.unrealized_pct(price),
        ]
        state = np.array(market + port, dtype=np.float32)
        return np.clip(state, -1.0, 1.0).astype(np.float32)

    # -- gym API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Every episode starts from the same clean account (audit
        # 2026-10-20: episodes used to share one broker, so positions and
        # equity carried across "days" and a carried position was revalued
        # at a random other date's price on each reset).
        self.broker.reset_state()
        max_start = len(self.df) - self.episode_len
        if options and "start" in options:
            # Pinned start (evaluation): chronological, non-overlapping
            # episodes instead of random overlapping windows.
            self._episode_start = min(int(options["start"]), max_start)
        else:
            self._episode_start = int(self.np_random.integers(0, max_start))
        self._cursor = self._episode_start
        price = float(self.df.iloc[self._cursor]["close"])
        # Equity at the previous step's price: the dense reward measures the
        # market move over each interval on the position we actually held.
        self._prev_eq = self.broker.equity(price)
        self._day_start_equity = self._prev_eq
        self._round_trips = 0
        return self._state(self._cursor), {}

    def _decision_context(self, i: int) -> str:
        """JSON snapshot stored on fills made from this decision row."""
        row = self.df.iloc[i]
        feats = [round(float(row[c]), 4) for c in FEATURE_COLUMNS]
        return json.dumps({"f": feats, "p": round(float(self.stacker_p_up[i]), 4)})

    def _atr_pct(self, i: int) -> float | None:
        row = self.df.iloc[i]
        if "atr_pct_raw" not in row:
            return None
        val = float(row["atr_pct_raw"])
        return val if np.isfinite(val) and val > 0 else None

    def step(self, action: int):
        i = self._cursor
        row = self.df.iloc[i]
        ctx = self._decision_context(i)
        atr = self._atr_pct(i)
        held_before = abs(self.broker.btc) > 1e-10

        if self.fill_mode == "next_open" and i + 1 < len(self.df):
            # Duel/evaluation timing: the decision made on candle i fills at
            # candle i+1's OPEN, then the broker's exit rules watch candle
            # i+1. Equity is marked at candle i+1's close.
            erow = self.df.iloc[i + 1]
            open_px = float(erow["open"])
            close_px = float(erow["close"])
            low_px = float(erow["low"]) if "low" in erow else close_px
            high_px = float(erow["high"]) if "high" in erow else close_px
            ets = str(erow.name)
            if action == 1:
                self.broker.buy(open_px, reason="ppo", ts=ets,
                                atr_pct=atr, context=ctx)
            elif action == 2:
                self.broker.sell(open_px, reason="ppo", ts=ets, context=ctx)
            elif action == 3:
                self.broker.short_open(open_px, reason="ppo", ts=ets,
                                       atr_pct=atr, context=ctx)
            stop_fill = self.broker.check_stop(
                close_px, ts=ets, low=low_px, high=high_px, context=ctx)
            mark_price = close_px
        else:
            price = float(row["close"])
            low = float(row["low"]) if "low" in row else price
            high = float(row["high"]) if "high" in row else price
            ts = str(row.name)
            # Exit rules enforced by the broker on every candle
            # (wick-throughs count), like live.
            stop_fill = self.broker.check_stop(price, ts=ts, low=low,
                                               high=high, context=ctx)
            if action == 1:
                self.broker.buy(price, reason="ppo", ts=ts,
                                atr_pct=atr, context=ctx)
            elif action == 2:
                self.broker.sell(price, reason="ppo", ts=ts, context=ctx)
            elif action == 3:
                self.broker.short_open(price, reason="ppo", ts=ts,
                                       atr_pct=atr, context=ctx)
            mark_price = price

        # A stop-out is the broker's exit, not the agent completing a
        # trade: it does not satisfy the activity quota (audit 2026-10-20).
        # v5: a completed round trip in EITHER direction counts (long
        # buy->sell, short open->cover, or a through-flat action close).
        if held_before and abs(self.broker.btc) <= 1e-10 and stop_fill is None:
            self._round_trips += 1

        # Dense signal: per-step log return on equity, in percentage points,
        # clipped to [-1, 1] -- computed AFTER the action executes (audit
        # 2026-10-20: the old code measured equity pre-trade and then reset
        # the baseline to the post-trade equity at the same price, so the
        # 0.15% fee landed between baselines and cancelled out of the
        # learning signal entirely; churn looked free). Post-trade equity
        # at the mark price captures both the market move on the held
        # position and the fee the trade just paid. Log form keeps the
        # signal stationary across compounding (a +1% then -1% move nets
        # to ~zero, unlike raw percent changes).
        eq_after = self.broker.equity(mark_price)
        prev = max(self._prev_eq, 1e-9)
        step_r = 100.0 * float(np.log(eq_after / prev))
        reward = float(np.clip(step_r, -1.0, 1.0))
        self._prev_eq = eq_after

        self._cursor += 1
        done = (self._cursor - self._episode_start) >= self.episode_len
        done = bool(done) or self._cursor >= len(self.df) - 1

        if done:
            # The daily verdict (original spec): +1 profitable day, else -1.
            day_pnl = self._prev_eq - self._day_start_equity
            reward += 1.0 if day_pnl > 0 else -1.0
            # Activity rule: trading is mandatory while learning. A flat day
            # (no round trips) can no longer tie a well-traded day.
            reward -= 0.2 * max(0, self.min_round_trips - self._round_trips)
        obs = self._state(min(self._cursor, len(self.df) - 1))
        return obs, reward, done, False, {}
