"""Muse's Pick #3 -- the exposure-manager brain (2026-10-27).

Kass's delegation (2026-10-27 ~14:08 ET): "try a completely
different tactic for the ppo brain... switch up how it trades and
stops and all that too... test your next model against whatever
champion you think is best."

THE TACTIC. Thirteen clean nulls say direction prediction from
these candles is a coin flip, while the one axis that ever
predicted anything was TRADEABILITY (hierarchy study, runs
55/56: balanced accuracy ~0.61 for "worth trading at all" vs
~0.50 for direction). So this brain never chooses direction at
all: direction is the crowned stacker's vote (run 73, Champion
#4), taken as sign(vote - 0.5) with the vote's strength in the
state. The brain's only job is EXPOSURE -- how much of the book
rides on the vote, and when to stand aside:

    0 STAND ASIDE -- target exposure 0 (close any position)
    1 HALF        -- target 37.5% of equity in the vote's direction
    2 FULL        -- target 75% of equity (the program cap) in the
                     vote's direction
    3 HOLD        -- keep the current position untouched this
                     step, even across a vote flip; exits only
                     via the stop or the time-stop

EXITS (challenger side only -- production broker.py untouched;
implemented by shadowing the broker constants on this module's
broker subclass, the StudyBroker precedent):
    stop = 1.5 x ATR (min 2%, max 6%)          [production: 2.0x, 2.5-8%]
    trailing activates at +2%                  [production: +3%]
    trail = 1.25 x ATR (min 1.5%, max 5%)      [production: 1.5x, 2-6%]
    plus a 72-hour max-hold TIME-STOP: a position held 72
    candles is flattened at the next candle open.

MECHANICS (frozen in refs/muse-pick-3-2026-10-27.md before any
result): rebalances execute at the step's execution price
(next-candle open in duel timing, the decision close in training
timing -- the production TradingEnv timing and step order either
way, mirrored in step() below). A flip closes through flat, then
opens, in the same step. Rebalances drive the production
broker's own tranche math: for a buy, the position/exposure caps
are set for that one fill so the broker's spend (grossed up by
1/(1 - fee), since the broker nets the fee out of the quantity)
lands the RESULTING position value exactly on the target;
sells and covers use the broker's pct parameter (exact by
construction); short opens land on exact notional the same cap
way. Deltas under the $10 minimum ticket are skipped
(production dust rule); full closes always execute. A vote of
exactly 0.5 is direction 0 -- HALF/FULL then target flat.
Time-stop: position age counts candles from the entry fill's
candle (adds keep the original entry); at a step's execution
point, age >= 72 flattens at the execution price BEFORE the
step's action applies (so HOLD cannot dodge it, and a flattened
position may be re-opened by the same step's action).

REWARD: the standard dense mark-to-market % return + the
terminal +/-1 on episode P&L sign. NO drawdown shaping (tested
twice, closed) and -- a deliberate, recorded deviation from
TradingEnv -- NO round-trip activity penalty: standing aside is
this brain's first-class decision, and a quota that pays for
churn would fight the thesis. Round trips are still counted and
reported.

State layout (102, identical to Pick #2 for comparability):
    [98 scored features (s/50 - 1)] + [vote (2p - 1)]
    + [0, position/equity, equity/100000 - 1]
The features are run 73's own frame (Pick #2's 98 columns,
assembled exactly as Pick #2 assembled its env frame); the vote
column is run 73's -- cross-fitted in training, fitted on the
holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from environment.trading_env import TradingEnv
from trading.broker import FEE_RATE as FEE
from trading.broker import PaperBroker

# The broker's minimum ticket (a literal inside PaperBroker's
# fill methods): partial rebalances under it are dust.
MIN_NOTIONAL = 10.0

ROOT = Path(__file__).resolve().parents[2]
FAMILY = "muse_pick3"

HALF_PCT = 0.375          # HALF target: 37.5% of equity
FULL_PCT = 0.75           # FULL target: 75% of equity (program cap)
TIME_STOP_CANDLES = 72    # 72-hour max hold

# The challenger's exit configuration (instance-shadowed onto
# ExposureBroker; production values in trading/broker.py).
CH_STOP_ATR_MULT = 1.5
CH_STOP_MIN_PCT = 0.02
CH_STOP_MAX_PCT = 0.06
CH_TRAIL_ACTIVATE_PCT = 0.02
CH_TRAIL_ATR_MULT = 1.25
CH_TRAIL_MIN_PCT = 0.015
CH_TRAIL_MAX_PCT = 0.05

# Action ids.
ACT_STAND_ASIDE, ACT_HALF, ACT_FULL, ACT_HOLD = 0, 1, 2, 3
ACTION_KEYS = {0: "stand_aside", 1: "half", 2: "full", 3: "hold"}


def build_pick3_frames(df: pd.DataFrame):
    """(prod_sup, raw98, env_sup, cols98) over a raw OHLCV frame,
    assembled exactly as Pick #2 assembled its frames:
    prod_sup = the production supervised frame; raw98 = Pick
    #2's 98-column raw frame (the stacker's own inputs, for
    votes); env_sup = assemble_env_sup(prod_sup, raw98, cols98)
    -- all 98 columns transformed s/50 - 1, Pick #2's env
    convention, so this brain's state is byte-comparable to
    Pick #2's."""
    from features import pick2_features, scored_eyes
    from features.technical_indicators import make_supervised

    meta = json.loads(
        (ROOT / "models" / "stacker_champion.json").read_text())
    assert meta.get("feature_set") == "pick2_98" and \
        meta.get("run_id") == 73, (
            f"Pick #3 reads Champion #4's frame; lab champion is "
            f"run {meta.get('run_id')} ({meta.get('feature_set')})")
    cols98 = list(meta["feature_columns"])
    prod_sup = make_supervised(df)
    raw98, built, _ev = pick2_features.build_pick2_frame(df)
    assert list(built) == cols98, (
        "pick2 frame columns differ from Champion #4's recorded set")
    env_sup = scored_eyes.assemble_env_sup(prod_sup, raw98, cols98)
    return prod_sup, raw98, env_sup, cols98


class ExposureBroker(PaperBroker):
    """PaperBroker with the Pick #3 challenger's exit config
    shadowed onto the instance (StudyBroker precedent) and the
    position/exposure caps lifted -- the env's exposure targets
    ARE the risk control here (max 75% of equity by construction).
    Production broker.py is never modified; the champion side of
    the duel trades a plain PaperBroker."""

    def __init__(self, db_path, starting_balance: float = 100_000.0):
        super().__init__(db_path, starting_balance=starting_balance)
        # The broker reads the UPPERCASE class constants via self;
        # shadow them on the instance (StudyBroker precedent).
        self.STOP_ATR_MULT = CH_STOP_ATR_MULT
        self.STOP_MIN_PCT = CH_STOP_MIN_PCT
        self.STOP_MAX_PCT = CH_STOP_MAX_PCT
        self.TRAIL_ACTIVATE_PCT = CH_TRAIL_ACTIVATE_PCT
        self.TRAIL_ATR_MULT = CH_TRAIL_ATR_MULT
        self.TRAIL_MIN_PCT = CH_TRAIL_MIN_PCT
        self.TRAIL_MAX_PCT = CH_TRAIL_MAX_PCT
        self.max_position_pct = 1.0
        self.max_exposure_pct = 1.0


class ExposureManagerEnv(TradingEnv):
    """The exposure-manager environment. Inherits TradingEnv's
    bookkeeping (state assembly, dense reward, terminal bonus,
    stop machinery, episode handling) and replaces only the
    action semantics and the time-stop. The inherited _execute is
    never called (step is overridden); targets are fractions of
    current equity, and min_round_trips is forced to 0 -- no
    activity penalty (the recorded deviation)."""

    def __init__(self, df: pd.DataFrame, stacker_p_up, broker,
                 episode_days: int = 1,
                 steps_per_day: int = 24,
                 fill_mode: str = "close",
                 feature_columns=None):
        super().__init__(df, stacker_p_up, broker,
                         episode_days=episode_days,
                         steps_per_day=steps_per_day,
                         min_round_trips=0,
                         fill_mode=fill_mode,
                         feature_columns=feature_columns)
        self.action_space = spaces.Discrete(4)
        self._entry_idx: int | None = None
        self.action_counts = np.zeros(4, dtype=int)

    # -- episode bookkeeping ------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._entry_idx = None
        self.action_counts = np.zeros(4, dtype=int)
        return obs, info

    # -- exposure machinery --------------------------------------
    def _capped_fill(self, spend: float, pos_value: float,
                     equity: float, fn) -> object:
        """Run one broker open/add (buy or short_open) with the
        position/exposure caps set for this fill alone, so the
        broker's own tranche math spends exactly `spend`:
        budget = equity x max_position_pct = spend and headroom =
        equity x max_exposure_pct - |pos| = spend. Caps are
        restored to the lifted 1.0 defaults afterwards."""
        b = self.broker
        b.max_position_pct = spend / equity
        b.max_exposure_pct = (abs(pos_value) + spend) / equity
        try:
            return fn()
        finally:
            b.max_position_pct = 1.0
            b.max_exposure_pct = 1.0

    def _rebalance_to(self, target_value: float, price: float,
                      ts: str, ctx: str,
                      atr_pct: float | None) -> list:
        """Move the position to target_value (signed dollars) at
        price, through the production broker's own fill methods.
        A flip closes through flat first, then opens. Returns the
        Fill objects made. The resulting position value lands
        EXACTLY on target in every branch (module docstring)."""
        fills: list = []
        b = self.broker
        pos_value = b.btc * price
        delta = target_value - pos_value
        if abs(delta) < 1e-9:
            return fills
        if pos_value != 0.0 and target_value != 0.0 \
                and np.sign(target_value) != np.sign(pos_value):
            f = self._close(price, ts, ctx, "exposure")
            if f is not None:
                fills.append(f)
            pos_value = 0.0
            delta = target_value
        if target_value != 0.0 and abs(delta) < MIN_NOTIONAL:
            return fills  # production dust rule
        equity = b.equity(price)
        if delta > 0:
            if b.btc < 0:  # partial cover: exact by pct
                f = b.cover(price, pct=delta / abs(pos_value),
                            reason="exposure", ts=ts, context=ctx)
            else:  # open/add long: spend grossed up for the fee
                spend = delta / (1 - FEE)
                f = self._capped_fill(
                    spend, pos_value, equity,
                    lambda: b.buy(price, reason="exposure", ts=ts,
                                  atr_pct=atr_pct, context=ctx))
        else:
            if b.btc > 0:  # partial sell: exact by pct
                f = b.sell(price, pct=abs(delta) / pos_value,
                           reason="exposure", ts=ts, context=ctx)
            else:  # open/add short: exact notional
                f = self._capped_fill(
                    abs(delta), pos_value, equity,
                    lambda: b.short_open(price, reason="exposure",
                                         ts=ts, atr_pct=atr_pct,
                                         context=ctx))
        if f is not None:
            fills.append(f)
        return fills

    def _close(self, price: float, ts: str, ctx: str,
               reason: str):
        """Full close at price via the broker's pct=1 exits (never
        tranche-capped). Returns the Fill or None."""
        if self.broker.btc > 0:
            return self.broker.sell(price, pct=1.0, reason=reason,
                                    ts=ts, context=ctx)
        if self.broker.btc < 0:
            return self.broker.cover(price, pct=1.0, reason=reason,
                                     ts=ts, context=ctx)
        return None

    def target_for(self, action: int, vote: float,
                   equity: float) -> float:
        """The signed dollar target an action implies right now."""
        direction = float(np.sign(vote - 0.5))
        if action == ACT_HALF:
            return direction * HALF_PCT * equity
        if action == ACT_FULL:
            return direction * FULL_PCT * equity
        return 0.0

    # -- the step --------------------------------------------------
    def step(self, action: int):
        """Mirror TradingEnv.step's ordering and reward exactly,
        with exposure actions in place of buy/sell/short and the
        time-stop inserted before the action in both timings:

        next_open: [time-stop at candle i+1's open] -> action
        fills at candle i+1's open -> broker check_stop watches
        candle i+1 -> mark at candle i+1's close.
        close: broker check_stop watches candle i -> [time-stop
        at candle i's close] -> action fills at candle i's close
        -> mark at candle i's close.

        Reward: TradingEnv's own -- 100 x log equity return in
        percentage points, clipped to [-1, 1], plus the terminal
        +/-1 on episode P&L sign. The activity-penalty term is
        identically zero here (min_round_trips=0): the recorded
        deviation. Round-trip counting mirrors production (a step
        that starts holding and ends flat counts) except that a
        close by a broker exit -- stop or time-stop -- does not
        count as the brain completing a trip."""
        action = int(action)
        i = self._cursor
        row = self.df.iloc[i]
        ctx = self._decision_context(i)
        atr = self._atr_pct(i)
        pos_at_start = self.broker.btc
        held_before = abs(pos_at_start) > 1e-10
        self.action_counts[action] += 1
        fills: list = []
        exit_fired = False
        vote = float(self.stacker_p_up[i])

        def _time_stop(price: float, ts: str, exec_i: int) -> None:
            nonlocal exit_fired
            if (abs(self.broker.btc) > 1e-10
                    and self._entry_idx is not None
                    and exec_i - self._entry_idx >= TIME_STOP_CANDLES):
                f = self._close(price, ts, ctx, "time_stop")
                if f is not None:
                    fills.append(f)
                    exit_fired = True

        def _act(price: float, ts: str) -> None:
            if action in (ACT_STAND_ASIDE, ACT_HALF, ACT_FULL):
                equity = self.broker.equity(price)
                target = self.target_for(action, vote, equity)
                fills.extend(self._rebalance_to(target, price, ts,
                                                ctx, atr))
            # ACT_HOLD: nothing -- the position rides.

        if self.fill_mode == "next_open" and i + 1 < len(self.df):
            erow = self.df.iloc[i + 1]
            open_px = float(erow["open"])
            close_px = float(erow["close"])
            low_px = float(erow["low"]) if "low" in erow else close_px
            high_px = float(erow["high"]) if "high" in erow else close_px
            ets = str(erow.name)
            exec_i = i + 1
            _time_stop(open_px, ets, exec_i)
            _act(open_px, ets)
            stop_fill = self.broker.check_stop(
                close_px, ts=ets, low=low_px, high=high_px, context=ctx)
            if stop_fill is not None:
                fills.append(stop_fill)
                exit_fired = True
            mark_price = close_px
        else:
            price = float(row["close"])
            low = float(row["low"]) if "low" in row else price
            high = float(row["high"]) if "high" in row else price
            ts = str(row.name)
            exec_i = i
            stop_fill = self.broker.check_stop(
                price, ts=ts, low=low, high=high, context=ctx)
            if stop_fill is not None:
                fills.append(stop_fill)
                exit_fired = True
            _time_stop(price, ts, exec_i)
            _act(price, ts)
            mark_price = price

        # Entry-candle bookkeeping: a fresh open or a flip starts
        # a new position age at this step's execution candle; an
        # add to a surviving position keeps the original entry.
        pos_after = self.broker.btc
        if abs(pos_after) <= 1e-10:
            self._entry_idx = None
        elif abs(pos_at_start) <= 1e-10 \
                or np.sign(pos_after) != np.sign(pos_at_start):
            self._entry_idx = exec_i
        elif self._entry_idx is None:
            self._entry_idx = exec_i

        # Round trips, production semantics (see docstring).
        if held_before and abs(pos_after) <= 1e-10 and not exit_fired:
            self._round_trips += 1

        # Reward + advance, TradingEnv's own lines.
        eq_after = self.broker.equity(mark_price)
        prev = max(self._prev_eq, 1e-9)
        step_r = 100.0 * float(np.log(eq_after / prev))
        reward = float(np.clip(step_r, -1.0, 1.0))
        self._prev_eq = eq_after
        self._cursor += 1
        done = (self._cursor - self._episode_start) >= self.episode_len
        done = bool(done) or self._cursor >= len(self.df) - 1
        if done:
            day_pnl = self._prev_eq - self._day_start_equity
            reward += 1.0 if day_pnl > 0 else -1.0
        info = {"action": action, "fills": fills,
                "equity": eq_after, "round_trips": self._round_trips}
        return self._state(self._cursor), float(reward), done, False, info


class ExposureTrainEnv(ExposureManagerEnv):
    """Training variant: duel-length episodes (264 candles = the
    11-day duel window), the DuelLengthShapedTradingEnv precedent
    -- so the 72h time-stop binds in training exactly as designed
    and the brain learns whole-window exposure management.

    train_ppo wires every env a PRODUCTION PaperBroker on a
    scratch ledger; the exposure manager must train under its own
    exit config, so this subclass substitutes an ExposureBroker
    on the SAME scratch db path (the ledger is simulation-
    internal and discarded by train_ppo after the run either
    way; no code reads training fills)."""

    def __init__(self, df, stacker_p_up, broker, **kwargs):
        kwargs["episode_days"] = 11
        sub = ExposureBroker(broker.db_path,
                             starting_balance=broker.starting_balance)
        super().__init__(df, stacker_p_up, sub, **kwargs)
        self.steps_per_day = 24


def evaluate_exposure_windows(model_path, windows, votes_by_window,
                              feature_columns):
    """Grade an exposure-manager model EXACTLY the way
    nightly.evaluate_ppo / evaluate_windows grade production
    brains -- this function mirrors evaluate_ppo line for line
    (episode starts k*24, window P&L = SUM of episode P&Ls,
    window drawdown = MAX episode drawdown, score = P&L -
    0.5 x max DD, pooled action mix, bench from window closes),
    swapping in ExposureManagerEnv + ExposureBroker on a scratch
    ledger. Returns evaluate_ppo-shaped per-window dicts, so
    nightly._agg and gate_decision apply unchanged. Action-mix
    keys are the exposure actions; "hold" is the HOLD-action
    share -- the env's only literal no-op, and the share the gate
    floor's non-hold test reads (STAND ASIDE is an active
    exposure decision, not a hold)."""
    import os

    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    tmp = Path("/tmp") / f"tam_eval_pick3_{os.getpid()}.db"
    if tmp.exists():
        tmp.unlink()
    results = []
    try:
        for w, sup in enumerate(windows):
            p_up = np.asarray(votes_by_window[w], dtype=float)
            rewards, pnls, trips, ep_dds = [], [], [], []
            counts = {k: 0 for k in ACTION_KEYS.values()}
            for ep, start in enumerate([k * 24 for k in range(10)]):
                broker = ExposureBroker(tmp, starting_balance=100_000)
                env = ExposureManagerEnv(
                    sup, p_up, broker, fill_mode="next_open",
                    feature_columns=feature_columns)
                obs, _ = env.reset(seed=ep, options={"start": start})
                start_eq = env._day_start_equity
                curve = [start_eq]
                done, ep_r = False, 0.0
                while not done:
                    act, _ = model.predict(obs, deterministic=True)
                    counts[ACTION_KEYS[int(act)]] += 1
                    obs, r, done, _, _ = env.step(int(act))
                    ep_r += r
                    curve.append(env._prev_eq)
                rewards.append(ep_r)
                pnls.append(env._prev_eq - start_eq)
                trips.append(env._round_trips)
                c = np.array(curve)
                ep_dds.append(
                    float(np.max(np.maximum.accumulate(c) - c)))
            total = float(np.sum(pnls))
            max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
            closes = sup["close"].to_numpy(dtype=float)
            bench = (float(100_000.0 * (closes[-1] / closes[0] - 1.0))
                     if len(closes) > 1 else 0.0)
            total_actions = sum(counts.values()) or 1
            results.append({
                "window": w + 1,
                "mean_reward": float(np.mean(rewards)),
                "total_pnl": total,
                "max_drawdown": max_dd,
                "mean_drawdown": float(np.mean(ep_dds)) if ep_dds else 0.0,
                "score": total - 0.5 * max_dd,
                "bench_pnl": bench,
                "episodes": 10,
                "round_trips_per_day": float(np.mean(trips)),
                "action_mix": {k: round(v / total_actions, 3)
                               for k, v in counts.items()},
            })
    finally:
        tmp.unlink(missing_ok=True)
    return results


__all__ = [
    "ExposureBroker", "ExposureManagerEnv", "ExposureTrainEnv",
    "evaluate_exposure_windows", "build_pick3_frames",
    "ACTION_KEYS", "ACT_STAND_ASIDE", "ACT_HALF", "ACT_FULL",
    "ACT_HOLD", "HALF_PCT", "FULL_PCT", "TIME_STOP_CANDLES",
]
