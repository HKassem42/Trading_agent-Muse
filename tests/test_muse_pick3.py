"""Muse's Pick #3 -- the exposure-manager brain (2026-10-27).

Written BEFORE the study ran, against the design frozen in
refs/muse-pick-3-2026-10-27.md. The brain never picks direction
(run 73's vote does); it picks exposure: STAND ASIDE (0), HALF
(37.5% of equity), FULL (75%), HOLD (ride). Pins:

- the challenger exit config binds on ExposureBroker (stop
  1.5xATR [2%, 6%], trail arm +2%, trail 1.25xATR [1.5%, 5%])
  and production PaperBroker constants/instances are untouched;
- exposure targets land EXACTLY (hand-computed ledgers: the
  grossed-up buy, the pct sell/cover, the exact-notional short,
  and a full flip through flat);
- STAND ASIDE closes fully and counts a round trip; HOLD
  ignores a vote flip; a vote of exactly 0.5 targets flat;
- the 72-hour time-stop flattens before the step's action, and
  HOLD cannot dodge it;
- NO activity penalty (the frozen deviation): a never-trading
  episode scores exactly the terminal -1 where the production
  env scores -1.4 on identical data;
- the training subclass substitutes the challenger broker for
  the production broker train_ppo hands it, with duel-length
  264-candle episodes;
- the grading evaluator returns evaluate_ppo-shaped window
  dicts (score = P&L - 0.5 x max DD, bench hand-checked).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from models import muse_pick3 as mp3  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

COLS = [f"f{k}" for k in range(98)]
FEE = 0.0015


def _df(n=300, price=100.0, atr=0.01):
    df = pd.DataFrame(0.0, index=range(n), columns=COLS)
    df["open"] = price
    df["high"] = price
    df["low"] = price
    df["close"] = price
    df["volume"] = 1.0
    df["atr_pct_raw"] = atr
    return df


def _env(tmp_path, votes=0.9, balance=100_000.0, episode_days=10,
         fill_mode="close", df=None, name="t.db"):
    df = _df() if df is None else df
    v = (np.full(len(df), votes) if np.isscalar(votes)
         else np.asarray(votes, dtype=float))
    broker = mp3.ExposureBroker(tmp_path / name,
                                starting_balance=balance)
    env = mp3.ExposureManagerEnv(df, v, broker,
                                 episode_days=episode_days,
                                 fill_mode=fill_mode,
                                 feature_columns=COLS)
    env.reset(options={"start": 0})
    return env


# -- the challenger exit config -----------------------------------

def test_challenger_broker_params_bind(tmp_path):
    b = mp3.ExposureBroker(tmp_path / "b.db")
    assert b.STOP_ATR_MULT == 1.5 and b.STOP_MIN_PCT == 0.02
    assert b.STOP_MAX_PCT == 0.06
    assert b.TRAIL_ACTIVATE_PCT == 0.02 and b.TRAIL_ATR_MULT == 1.25
    assert b.TRAIL_MIN_PCT == 0.015 and b.TRAIL_MAX_PCT == 0.05
    assert b.max_position_pct == 1.0 and b.max_exposure_pct == 1.0


def test_production_broker_untouched(tmp_path):
    # Class constants are production's.
    assert PaperBroker.STOP_ATR_MULT == 2.0
    assert PaperBroker.STOP_MIN_PCT == 0.025
    assert PaperBroker.STOP_MAX_PCT == 0.08
    assert PaperBroker.TRAIL_ACTIVATE_PCT == 0.02  # Kass 2026-11-02 (was 0.03)
    assert PaperBroker.TRAIL_ATR_MULT == 1.5
    assert PaperBroker.TRAIL_MIN_PCT == 0.02
    assert PaperBroker.TRAIL_MAX_PCT == 0.06
    # A fresh production instance carries no instance shadows.
    p = PaperBroker(tmp_path / "p.db", starting_balance=100_000,
                    max_position_pct=0.25, stop_loss_pct=0.05)
    assert p.STOP_ATR_MULT == 2.0 and p.max_position_pct == 0.25
    assert p.max_exposure_pct == 0.75


def test_challenger_stop_anchor_is_tighter(tmp_path):
    # atr 1%: challenger stop distance = max(2%, 1.5x1%) = 2%
    # (floor binds); production = max(2.5%, 2x1%) = 2.5%.
    env = _env(tmp_path, votes=0.9)
    env.step(mp3.ACT_FULL)
    assert env.broker.stop_anchor == pytest.approx(98.0)
    prod = PaperBroker(tmp_path / "prod.db", starting_balance=100_000,
                       max_position_pct=0.25, stop_loss_pct=0.05)
    prod.buy(100.0, reason="signal", ts="t", atr_pct=0.01)
    assert prod.stop_anchor == pytest.approx(97.5)


def test_challenger_stop_fires_where_production_would_not(tmp_path):
    df = _df()
    df.loc[1, "low"] = 97.9   # through the 98.0 challenger stop,
    df.loc[1, "close"] = 99.0  # above production's 97.5
    env = _env(tmp_path, votes=0.9, df=df)
    env.step(mp3.ACT_FULL)
    assert env.broker.btc > 0
    _o, _r, _d, _t, info = env.step(mp3.ACT_HOLD)
    assert env.broker.btc == 0.0
    assert [f.reason for f in info["fills"]] == ["stop_loss"]
    assert info["fills"][0].price == pytest.approx(98.0)


# -- exposure targets, hand-computed -------------------------------

def test_full_opens_exactly_75_percent(tmp_path):
    env = _env(tmp_path, votes=0.9)
    _o, _r, _d, _t, info = env.step(mp3.ACT_FULL)
    b = env.broker
    spend = 75_000.0 / (1 - FEE)
    assert b.btc * 100.0 == pytest.approx(75_000.0, abs=1e-6)
    assert b.btc == pytest.approx(750.0, abs=1e-6)
    assert b.cash == pytest.approx(100_000.0 - spend, abs=1e-6)
    assert len(info["fills"]) == 1
    f = info["fills"][0]
    assert f.side == "BUY" and f.reason == "exposure"
    assert f.fee == pytest.approx(spend * FEE, abs=1e-6)


def test_half_opens_exactly_37_5_percent(tmp_path):
    env = _env(tmp_path, votes=0.2)  # bearish vote: HALF is short
    _o, _r, _d, _t, info = env.step(mp3.ACT_HALF)
    b = env.broker
    assert b.btc * 100.0 == pytest.approx(-37_500.0, abs=1e-6)
    # Short open: exact notional, fee out of the proceeds.
    assert b.cash == pytest.approx(
        100_000.0 + 37_500.0 * (1 - FEE), abs=1e-6)
    assert info["fills"][0].reason == "short_open"


def test_stand_aside_closes_fully_and_counts_the_trip(tmp_path):
    env = _env(tmp_path, votes=0.9)
    env.step(mp3.ACT_FULL)
    _o, _r, _d, _t, info = env.step(mp3.ACT_STAND_ASIDE)
    b = env.broker
    assert b.btc == 0.0
    spend = 75_000.0 / (1 - FEE)
    expect_cash = (100_000.0 - spend) + 75_000.0 * (1 - FEE)
    assert b.cash == pytest.approx(expect_cash, abs=1e-6)
    assert info["round_trips"] == 1
    assert [f.reason for f in info["fills"]] == ["exposure"]


def test_hold_ignores_a_vote_flip(tmp_path):
    votes = np.full(300, 0.9)
    votes[1:] = 0.1  # the vote flips bearish after the open
    env = _env(tmp_path, votes=votes)
    env.step(mp3.ACT_FULL)
    held = env.broker.btc
    _o, _r, _d, _t, info = env.step(mp3.ACT_HOLD)
    assert info["fills"] == []
    assert env.broker.btc == held


def test_flip_closes_through_flat_then_opens_exact(tmp_path):
    votes = np.full(300, 0.9)
    votes[1] = 0.1
    env = _env(tmp_path, votes=votes)
    env.step(mp3.ACT_FULL)
    equity_at_step = env.broker.equity(100.0)  # 99,887.33 after fees
    _o, _r, _d, _t, info = env.step(mp3.ACT_FULL)  # now bearish FULL
    b = env.broker
    target = -0.75 * equity_at_step
    assert b.btc * 100.0 == pytest.approx(target, abs=1e-6)
    assert [f.reason for f in info["fills"]] == [
        "exposure", "short_open"]
    assert [f.side for f in info["fills"]] == ["SELL", "SELL"]
    # Hand ledger: close proceeds net of fee, then short proceeds
    # net of fee land in cash.
    spend = 75_000.0 / (1 - FEE)
    cash_after_close = (100_000.0 - spend) + 75_000.0 * (1 - FEE)
    expect_cash = cash_after_close + abs(target) * (1 - FEE)
    assert b.cash == pytest.approx(expect_cash, abs=1e-6)


def test_vote_exactly_half_targets_flat(tmp_path):
    env = _env(tmp_path, votes=0.5)
    _o, _r, _d, _t, info = env.step(mp3.ACT_FULL)
    assert info["fills"] == [] and env.broker.btc == 0.0
    # And from a position: the 0.5 vote's FULL closes to flat.
    votes = np.full(300, 0.9)
    votes[1] = 0.5
    env2 = _env(tmp_path, votes=votes, name="t2.db")
    env2.step(mp3.ACT_FULL)
    assert env2.broker.btc > 0
    env2.step(mp3.ACT_FULL)
    assert env2.broker.btc == 0.0


def test_dust_delta_is_skipped(tmp_path):
    # $1,000 book: the fee drag on a repeated HALF is ~$0.21,
    # under the $10 ticket -- no fill, position untouched.
    env = _env(tmp_path, votes=0.9, balance=1_000.0)
    env.step(mp3.ACT_HALF)
    assert env.broker.btc == pytest.approx(3.75)
    _o, _r, _d, _t, info = env.step(mp3.ACT_HALF)
    assert info["fills"] == []
    assert env.broker.btc == pytest.approx(3.75)


# -- the time-stop --------------------------------------------------

def test_time_stop_fires_at_72h_and_hold_cannot_dodge_it(tmp_path):
    env = _env(tmp_path, votes=0.9, episode_days=10)
    env.step(mp3.ACT_FULL)  # entry at candle 0 (close timing)
    for k in range(1, 72):
        _o, _r, _d, _t, info = env.step(mp3.ACT_HOLD)
        assert info["fills"] == [], f"step {k} filled early"
        assert env.broker.btc > 0
    _o, _r, _d, _t, info = env.step(mp3.ACT_HOLD)  # candle 72
    assert env.broker.btc == 0.0
    assert [f.reason for f in info["fills"]] == ["time_stop"]
    # A time-stop close is a broker exit, not the brain's trip.
    assert info["round_trips"] == 0


# -- the reward deviation -------------------------------------------

def test_no_activity_penalty(tmp_path):
    # Never trade, flat prices: Pick #3's episode reward is the
    # terminal -1 alone. The production env on identical data
    # charges its 2-trip quota: -1 - 0.2 x 2 = -1.4.
    df = _df(n=100)
    env = _env(tmp_path, votes=0.9, episode_days=1, df=df)
    total, done = 0.0, False
    while not done:
        _o, r, done, _t, _i = env.step(mp3.ACT_STAND_ASIDE)
        total += r
    assert total == pytest.approx(-1.0)
    broker = PaperBroker(tmp_path / "prod2.db",
                         starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    prod = TradingEnv(df, np.full(100, 0.9), broker,
                      fill_mode="close", feature_columns=COLS)
    prod.reset(options={"start": 0})
    total_p, done = 0.0, False
    while not done:
        _o, r, done, _t, _i = prod.step(0)  # production HOLD
        total_p += r
    assert total_p == pytest.approx(-1.4)


# -- training wiring --------------------------------------------------

def test_train_env_substitutes_the_challenger_broker(tmp_path):
    prod = PaperBroker(tmp_path / "scratch.db",
                       starting_balance=100_000,
                       max_position_pct=0.25, stop_loss_pct=0.05)
    env = mp3.ExposureTrainEnv(_df(n=600), np.full(600, 0.9), prod,
                               fill_mode="close",
                               feature_columns=COLS)
    assert isinstance(env.broker, mp3.ExposureBroker)
    assert env.broker.STOP_ATR_MULT == 1.5
    assert env.broker.db_path == prod.db_path  # same scratch ledger
    assert env.episode_len == 264  # duel-length episodes
    assert env.min_round_trips == 0


# -- the feed contract: BOTH action vocabularies --------------------
# Run 75's mix uses the exposure vocabulary; every older brain
# uses the directional one. The training feed must serialize
# each mix with its own keys intact and numeric (the 2026-10-27
# watch defect: a fixed buy/sell read of an exposure mix yields
# NaN dashboard-side).

def test_mix_serializes_both_vocabularies_intact():
    from dashboards import compact

    expo = {"stand_aside": 0.196, "half": 0.05, "full": 0.008,
            "hold": 0.746}
    out = compact._mix(expo)
    assert out == {"stand_aside": 0.2, "half": 0.05, "full": 0.01,
                   "hold": 0.75}
    dire = {"hold": 0.571, "buy": 0.0, "sell": 0.421, "short": 0.008}
    assert compact._mix(dire) == {"hold": 0.57, "buy": 0, "sell": 0.42,
                                  "short": 0.01}


def test_duel_run_carries_the_runs_own_vocabulary():
    from dashboards import compact

    base = {"id": 999, "ts": "2026-10-27T00:00:00+00:00",
            "kind": "ppo", "promoted": 0}
    expo_run = dict(base, metrics={
        "challenger": {"score": -415.89, "total_pnl": -453.23,
                       "max_drawdown": 675.8, "episodes": 30,
                       "bench_pnl": -5152.5,
                       "action_mix": {"stand_aside": 0.196,
                                      "half": 0.05, "full": 0.008,
                                      "hold": 0.746}},
        "champion": {"score": -35.59, "total_pnl": 212.21,
                     "action_mix": {"hold": 0.571, "buy": 0.0,
                                    "sell": 0.421, "short": 0.008}}})
    out = compact._duel_run(expo_run)
    assert set(out["mix"]) == {"stand_aside", "half", "full", "hold"}
    assert set(out["cmix"]) == {"hold", "buy", "sell", "short"}
    import math
    assert all(isinstance(v, (int, float)) and not math.isnan(v)
               for v in list(out["mix"].values())
               + list(out["cmix"].values()))


def test_pick3_run_is_named_and_verdicted_in_the_index():
    """Run 75 in the all-models index: Kass's ordered label, family
    code 16, the FAIL verdict -- without the muse_pick3 cases the
    index fell back to a nightly name and mislabeled the run
    ("Stacker v5.1", caught on the feed before its first push)."""
    from dashboards import compact

    # The index ships in push_models_page pages since 2026-11-17;
    # assemble it the way the desk does.
    entry = next(e for p in compact.build_models_pages()
                 for e in p["rows"] if e[0] == 75)
    assert entry[2] == "Muse's Pick #3 PPO"
    assert entry[3] == 16
    assert entry[5] == pytest.approx(-415.9)
    assert entry[6] == -453
    assert entry[7] == "FAIL"
    assert compact._FAMILY_CODES["muse_pick3"] == 16


def test_training_feed_run75_mix_is_exposure_shaped_and_numeric():
    import math

    from dashboards import compact

    # RE-PINNED 2026-11-21: runs 74/75 rolled out of the detail
    # window as the registry grew (the window is the newest runs),
    # so their mixes are asserted on the same row builder over
    # their full-registry records -- the mix shape the builder
    # emits is what this test pins, not window membership (same
    # pattern as the run-45 naming assertion above).
    by_id = {r["id"]: r for r in compact._all_runs_desc()}
    r75 = compact._duel_run(by_id[75])
    r74 = compact._duel_run(by_id[74])
    assert set(r75["mix"]) == {"stand_aside", "half", "full",
                               "hold"}
    assert set(r74["mix"]) == {"hold", "buy", "sell", "short"}
    feed = compact.build_compact_training()
    for row in feed["runs"]:
        for key in ("mix", "cmix"):
            if key in row:
                assert all(
                    isinstance(v, (int, float)) and not math.isnan(v)
                    for v in row[key].values()), (row["i"], key)


# -- the grading evaluator's contract --------------------------------

def test_evaluate_exposure_windows_shape(tmp_path):
    from stable_baselines3 import PPO

    df = _df(n=264)
    env = _env(tmp_path, votes=0.9, df=df, name="ev.db")
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    path = tmp_path / "untrained.zip"
    model.save(path)
    # evaluate_ppo consumes 264-candle windows; give it three.
    windows = [df.copy() for _ in range(3)]
    votes = [np.full(264, 0.9) for _ in range(3)]
    res = mp3.evaluate_exposure_windows(path, windows, votes, COLS)
    assert len(res) == 3
    for w, r in enumerate(res):
        assert r["window"] == w + 1
        assert r["episodes"] == 10
        assert r["score"] == pytest.approx(
            r["total_pnl"] - 0.5 * r["max_drawdown"])
        assert r["bench_pnl"] == pytest.approx(0.0)  # flat prices
        assert set(r["action_mix"]) == {
            "stand_aside", "half", "full", "hold"}
        assert sum(r["action_mix"].values()) == pytest.approx(
            1.0, abs=0.01)
