"""Lab-only PPO buy-collapse fix (Phase 2, 2026-11-23).

TradingEnv gains three OPT-IN flags -- symmetric_exit, boundary_penalty,
realized_bonus -- all default-off, so the production env (nightly train,
duel grading, live loop) is byte-identical to before. Written test-first:
the flag tests fail on the old code (TypeError: unexpected keyword) and
the parity test pins pre-change rewards so a default-constructed env
cannot drift.

Action masking is deliberately NOT in this bundle. Phase 1's masked
rollout proved it cannot create buys (bull-window buy rate 0.4% ->
0.4%; freed mass flows to HOLD/SHORT), sb3-contrib/MaskablePPO is not
installed, and evaluate_ppo loads via plain PPO.load. The last test
below proves why an env-level SELL->HOLD remap would be a no-op: SELL
while flat is already behaviorally identical to HOLD in the production
env, so there is nothing for a remap to fix.

Run: .venv/bin/python -m pytest tests/test_ppo_buy_fix.py -q
"""
import inspect
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


# -- fixtures ---------------------------------------------------------
def _synthetic_df(n=600, seed=11):
    """The test_v5_shorts random-walk frame (proven non-degenerate)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = close * np.exp(rng.normal(0, 0.001, n))
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    df = pd.DataFrame({
        "open": open_,
        "high": top * (1 + rng.uniform(0, 0.003, n)),
        "low": bot * (1 - rng.uniform(0, 0.003, n)),
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    return add_features(df)


def _walk_df(n=1500, drift=0.0001, vol=0.001, seed=11):
    """Seeded random walk with a slight drift. Deliberately NOT a flat
    or monotonic series: constant prices make momentum features NaN
    (the whole frame drops), and a pure ramp keeps zero rows after
    dropna. Vol is small enough that no stop (floor 2.5%) can fire
    over a 24-step episode."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.0005, "low": close * 0.9995,
        "close": close, "volume": np.full(n, 100.0)}, index=idx)
    return add_features(df)


def _crash_df(n=1500, crash_at=1200, seed=3):
    """Random-walk prices, then one -10% candle at `crash_at`: a long's
    stop (entry ~100, ATR ~0.1% -> 2.5% floor) must fire on that candle.
    Pre-crash prices vary, so the momentum features stay defined."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    close[crash_at:] = 90.0
    df = pd.DataFrame({
        "open": close, "high": close * 1.0005, "low": close * 0.9995,
        "close": close, "volume": np.full(n, 100.0)}, index=idx)
    return add_features(df)


def _env(df, tmp_path, name="e.db", **kw):
    sup = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    p = np.full(len(sup), 0.5)
    broker = PaperBroker(tmp_path / name)
    env = TradingEnv(sup, p, broker, **kw)
    return sup, env, broker


def _ledger_trip_pnl(db_path):
    """Realized P&L of the (single) round trip in the ledger, computed
    from the fills alone -- independent of the env's equity tracking.

    Definition matches the env's realized term: market P&L net of the
    EXIT fee only, i.e. (sell proceeds net of fee) minus the position's
    mark value at open (qty * buy price). The ENTRY fee is deliberately
    excluded: it was already scored in the open step's dense reward
    (equity drops by the fee the moment the buy fills), so including
    it here would charge it twice."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT side, price, qty, fee, cash_after FROM trades ORDER BY id"
    ).fetchall()
    con.close()
    assert len(rows) == 2, rows
    (s0, p0, q0, f0, _), (s1, p1, q1, f1, _) = rows
    assert s0 == "BUY" and s1 == "SELL"
    assert abs(q0 - q1) < 1e-9
    received = q1 * p1 - f1  # sell proceeds net of the exit fee
    return received - q0 * p0  # minus the position's mark value at open


# -- 1. production defaults are unchanged ------------------------------
# Pinned 2026-11-23 BEFORE the fix (scripted episode on the rng-11
# frame, start=30, 24 steps = one full episode). If the default path
# drifts by even a fee, this fails.
PARITY_ACTIONS = [3, 1, 2, 0, 3, 1, 1, 2, 3, 0, 1, 2, 3, 3,
                  1, 0, 2, 1, 3, 0, 1, 1, 2, 3]
PARITY_REWARDS = [-0.037507033, -0.141655089, 0.000160478, 6.3066e-05,
                  -0.037771081, -0.034082611, -0.000679078, 0.0,
                  -0.037507033, 0.078004031, -0.020939095, 0.0,
                  -0.037507033, -0.165346851, -0.253216056, -0.085255887,
                  0.159009895, 0.041897547, -0.036896968, 0.094519035,
                  -0.151085956, -0.000812749, 0.0, -1.037507033]


def test_default_env_rewards_pinned(tmp_path):
    sup = _synthetic_df().dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    env = TradingEnv(sup, np.full(len(sup), 0.5),
                     PaperBroker(tmp_path / "e.db"))
    env.reset(seed=0, options={"start": 30})
    rews = []
    for a in PARITY_ACTIONS:
        _, r, done, _, _ = env.step(a)
        rews.append(round(float(r), 9))
    assert done is True
    assert rews == PARITY_REWARDS
    assert env._round_trips == 3


def test_flag_defaults_are_production():
    sig = inspect.signature(TradingEnv.__init__)
    assert sig.parameters["symmetric_exit"].default is False
    assert sig.parameters["boundary_penalty"].default == 0.0
    assert sig.parameters["realized_bonus"].default == 0.0


def test_ppo_train_accepts_env_kwargs():
    from models import ppo_train
    sig = inspect.signature(ppo_train.train)
    assert "env_kwargs" in sig.parameters
    assert sig.parameters["env_kwargs"].default is None


def test_nightly_train_call_passes_no_env_kwargs():
    src = (Path(__file__).resolve().parents[1]
           / "src" / "training" / "nightly.py").read_text()
    call = src[src.index("challenger_path = train_ppo("):]
    call = call[:call.index(")\n")]
    assert "env_kwargs" not in call


# -- 2. symmetric exit -------------------------------------------------
@pytest.mark.parametrize("fill_mode", ["close", "next_open"])
def test_symmetric_exit_covers_full_short_in_one_buy(tmp_path, fill_mode):
    # Build a ~75% short (three 25%-of-equity tranches), then ONE BUY.
    sup, env, broker = _env(_walk_df(), tmp_path, fill_mode=fill_mode,
                            symmetric_exit=True)
    env.reset(seed=0, options={"start": 60})
    for _ in range(3):
        env.step(3)  # SHORT
    assert broker.btc < 0
    short_before = broker.btc
    assert abs(short_before) * 100.0 > 60_000  # near the 75% cap
    env.step(1)  # BUY: one decision must exit the whole short
    assert broker.btc == 0


@pytest.mark.parametrize("fill_mode", ["close", "next_open"])
def test_default_env_still_tranche_caps_covers(tmp_path, fill_mode):
    # Control: the production env needs up to 3 BUYs for the same short.
    sup, env, broker = _env(_walk_df(), tmp_path, fill_mode=fill_mode)
    env.reset(seed=0, options={"start": 60})
    for _ in range(3):
        env.step(3)
    assert broker.btc < 0
    env.step(1)  # BUY: one tranche only
    assert broker.btc < 0
    env.step(1)
    env.step(1)
    assert broker.btc == 0


def test_symmetric_cover_uses_cover_reason_and_counts_trip(tmp_path):
    sup, env, broker = _env(_walk_df(), tmp_path, symmetric_exit=True)
    env.reset(seed=0, options={"start": 60})
    for _ in range(3):
        env.step(3)
    env.step(1)
    con = sqlite3.connect(tmp_path / "e.db")
    reasons = [r[0] for r in con.execute(
        "SELECT reason FROM trades ORDER BY id").fetchall()]
    con.close()
    assert reasons[-1] == "cover"
    assert env._round_trips == 1  # a decision-close, like a long exit


# -- 3. boundary penalty ----------------------------------------------
def test_boundary_penalty_charges_open_position_at_episode_end(tmp_path):
    # Identical episodes, one env with the penalty: the ONLY difference
    # in total reward must be exactly -0.5 (episode ends short).
    totals = {}
    for tag, kw in (("control", {}), ("penalized", {"boundary_penalty": 0.5})):
        sup, env, broker = _env(_walk_df(), tmp_path, name=f"{tag}.db", **kw)
        env.reset(seed=0, options={"start": 60})
        total = 0.0
        for i in range(24):
            _, r, done, _, _ = env.step(3 if i == 0 else 0)
            total += float(r)
        assert done is True and broker.btc < 0
        totals[tag] = total
    assert totals["penalized"] == pytest.approx(totals["control"] - 0.5,
                                                abs=1e-9)


def test_boundary_penalty_free_when_flat_at_episode_end(tmp_path):
    totals = {}
    for tag, kw in (("control", {}), ("penalized", {"boundary_penalty": 0.5})):
        sup, env, broker = _env(_walk_df(), tmp_path, name=f"{tag}.db", **kw)
        env.reset(seed=0, options={"start": 60})
        total = 0.0
        for _ in range(24):
            _, r, done, _, _ = env.step(0)
            total += float(r)
        assert done is True and broker.btc == 0
        totals[tag] = total
    assert totals["penalized"] == pytest.approx(totals["control"], abs=1e-9)


# -- 4. realized bonus -------------------------------------------------
def test_realized_bonus_credits_decision_close(tmp_path):
    # BUY at p0, SELL one step later at p1 > p0. The step is found by
    # scanning for a clear up-move (a pure up-drift walk makes rsi_c
    # NaN everywhere -- prices must actually fall sometimes). The SELL
    # step's reward must exceed the control env's by exactly
    # 0.5 * clip(realized_pct, -1, 1), with realized_pct computed from
    # the LEDGER fills (independent of the env's equity tracking).
    rews = {}
    for tag, kw in (("control", {}), ("fixed", {"realized_bonus": 0.5})):
        sup, env, broker = _env(_walk_df(drift=0.002, vol=0.004, seed=5),
                                tmp_path, name=f"{tag}.db", **kw)
        rets = sup["close"].to_numpy()[1:] / sup["close"].to_numpy()[:-1] - 1
        start = int(np.where(rets > 0.004)[0][0])
        env.reset(seed=0, options={"start": start})
        rs = []
        for a in (1, 2, 0, 0):
            _, r, done, _, _ = env.step(a)
            rs.append(float(r))
        rews[tag] = rs
        assert broker.btc == 0
    assert rews["fixed"][0] == pytest.approx(rews["control"][0], abs=1e-9)
    assert rews["fixed"][2:] == pytest.approx(rews["control"][2:], abs=1e-9)
    trip = _ledger_trip_pnl(tmp_path / "fixed.db")
    expected = 0.5 * float(np.clip(100.0 * trip / 100_000.0, -1.0, 1.0))
    assert trip > 0  # a winning trip on the scanned up-step
    assert rews["fixed"][1] - rews["control"][1] == pytest.approx(
        expected, abs=1e-9)


def test_realized_bonus_absent_on_stop_close(tmp_path):
    # Same long, but the -10% crash candle stops it out: the exit is
    # the broker's, not a decision, so no realized term applies and
    # the fix env's rewards must equal the control's exactly.
    rews = {}
    for tag, kw in (("control", {}), ("fixed", {"realized_bonus": 0.5,
                                                "boundary_penalty": 0.5,
                                                "symmetric_exit": True})):
        sup, env, broker = _env(_crash_df(), tmp_path, name=f"{tag}.db", **kw)
        # Locate the crash candle in the RESET frame (dropna shifts it).
        crash_pos = int(np.where(sup["close"].to_numpy() < 95.0)[0][0])
        env.reset(seed=0, options={"start": crash_pos - 4})
        rs = []
        # BUY 4 candles before the crash, HOLD through it: the stop
        # fires at the START of the crash-candle step, before the HOLD.
        for i in range(6):
            _, r, done, _, _ = env.step(1 if i == 0 else 0)
            rs.append(float(r))
            if broker.btc == 0 and i >= 4:
                break
        rews[tag] = rs
        assert broker.btc == 0  # stopped out, not decision-closed
    assert rews["fixed"] == pytest.approx(rews["control"], abs=1e-9)


# -- 5. masking: why it is NOT in the bundle ---------------------------
def test_sell_while_flat_is_hold(tmp_path):
    """SELL while flat is already a no-op identical to HOLD in the
    production env (same reward, same equity, no fill, same obs) --
    so an env-level SELL->HOLD remap could change nothing, and only
    policy-level masking (MaskablePPO, not installed) could redirect
    the freed mass. Phase 1 measured that redirection and it does not
    create buys (bull buy rate 0.4% -> 0.4%)."""
    outs = {}
    for tag, first in (("sell", 2), ("hold", 0)):
        sup, env, broker = _env(_synthetic_df(), tmp_path, name=f"{tag}.db")
        env.reset(seed=0, options={"start": 30})
        obs1, r1, _, _, _ = env.step(first)
        obs2, r2, _, _, _ = env.step(0)
        outs[tag] = (float(r1), float(r2), broker.equity(100.0),
                     obs1.tobytes(), obs2.tobytes())
        con = sqlite3.connect(tmp_path / f"{tag}.db")
        n_fills = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        con.close()
        assert n_fills == 0
    assert outs["sell"] == outs["hold"]
