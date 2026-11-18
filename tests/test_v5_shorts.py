"""Tests for the v5 upgrade (2026-10-20): SHORT support + Kass's indicators.

Covers: short open->cover round-trip P&L signs, the short stop triggering
on the candle HIGH, the short trailing exit, the exposure cap on shorts,
the through-flat rule (one action never flips a position), snapshot trip
accounting for short trips, the env's 4th action, and the 21-feature
contract/causality on synthetic data. The real-data contract check lives
in the v5 verification run (730d Coinbase) alongside scripts/derive_v5_scales.py.

Run: .venv/bin/python -m pytest tests/ -q
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboards.snapshot import closed_trades, trade_stats  # noqa: E402
from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

V5_FEATURES = ["ha_dir", "obv_slope", "adl_slope", "netvol_rel", "ewo"]


def test_short_round_trip_pnl_signs(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    f = b.short_open(100.0, ts="t0", atr_pct=0.005)
    assert f is not None and f.side == "SELL" and f.reason == "short_open"
    assert b.btc < 0 and b.entry_price == 100.0
    # Price falls: the short is in profit mark-to-market, and covering
    # with a BUY (tranche covers the whole tranche-sized short) realizes it.
    assert b.equity(90.0) > 100_000
    f2 = b.buy(90.0, ts="t1")
    assert f2 is not None and f2.side == "BUY" and f2.reason == "cover"
    assert b.btc == 0 and b.cash > 100_000  # profit banked, net of fees

    # Price rises instead: covering realizes a loss.
    b2 = PaperBroker(tmp_path / "t2.db")
    b2.short_open(100.0, ts="t0", atr_pct=0.005)
    assert b2.equity(110.0) < 100_000
    b2.cover(110.0, ts="t1")  # protective-style full cover, uncapped
    assert b2.btc == 0 and b2.cash < 100_000


def test_short_stop_triggers_on_candle_high(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.short_open(100.0, ts="t0", atr_pct=0.005)
    lvl, trailing = b.stop_level()
    assert abs(lvl - 102.5) < 1e-9 and not trailing  # entry*(1+max(2*.005,.025))
    # Close AND high both below the stop: no exit.
    assert b.check_stop(101.0, ts="t1", high=102.0) is None
    # Close still below the stop but the HIGH wicks through it: stopped
    # out (a close-only check would have held), filled AT the level
    # because the close stayed on the safe side.
    f = b.check_stop(101.0, ts="t2", high=103.0)
    assert f is not None and f.reason == "stop_loss"
    assert abs(f.price - 102.5) < 1e-9 and f.side == "BUY" and b.btc == 0


def test_short_trailing_exit(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.short_open(100.0, ts="t0", atr_pct=0.005)
    # Down 4%: trailing arms off the lowest close (96).
    assert b.check_stop(96.0, ts="t1", high=97.0) is None
    lvl, trailing = b.stop_level()
    assert trailing and abs(lvl - 97.92) < 1e-9  # 96 * (1 + max(1.5*.005, .02))
    # Bounce whose high crosses the trailing level: exit at the level.
    f = b.check_stop(97.0, ts="t2", high=98.5)
    assert f is not None and f.reason == "trailing_stop"
    assert abs(f.price - 97.92) < 1e-9 and b.btc == 0


def test_short_exposure_cap(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    fills = [b.short_open(100.0, ts=f"t{i}", atr_pct=0.005) for i in range(6)]
    assert abs(b.btc) * 100 <= 75_000 + 1  # 75% cap, never pyramided past it
    assert sum(f is not None for f in fills) < 6


def test_through_flat_rule(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=0.005)
    assert b.btc > 0
    f = b.short_open(100.0, ts="t1", atr_pct=0.005)  # SHORT while long
    assert f is not None and f.side == "SELL"
    assert b.btc == 0  # closed the long only -- did NOT flip short
    f2 = b.short_open(100.0, ts="t2", atr_pct=0.005)
    assert f2.reason == "short_open" and b.btc < 0
    f3 = b.buy(100.0, ts="t3")  # BUY while short covers, never flips long
    assert f3.reason == "cover" and b.btc <= 0
    b.buy(100.0, ts="t4")
    assert b.btc == 0


def test_snapshot_counts_short_trips(tmp_path):
    db = tmp_path / "paper.db"
    b = PaperBroker(db)
    b.buy(100.0, ts="2026-01-01T00:00:00", atr_pct=0.005)
    b.sell(110.0, ts="2026-01-01T05:00:00")        # long trip, wins
    b.short_open(100.0, ts="2026-01-01T06:00:00", atr_pct=0.005)
    b.cover(90.0, ts="2026-01-01T09:00:00")        # short trip, wins
    con = sqlite3.connect(db)
    closed = closed_trades(con)
    assert len(closed) == 2
    assert all(t["pnl"] > 0 for t in closed)
    stats = trade_stats(con)
    assert stats["total_trades"] == 2
    assert stats["win_pct"] == 100.0
    assert stats["net_pnl"] == round(sum(t["pnl"] for t in closed), 2)
    assert stats["profit_factor"] is None  # no losses yet -> undefined, not 0


def _synthetic_df(n=600):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = close * np.exp(rng.normal(0, 0.001, n))
    # Wick sizes vary per bar: with fixed +/-0.2% wicks the close would sit
    # exactly mid-range every bar and the ADL money-flow multiplier would
    # be identically zero -- a degenerate frame for the volume features.
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    df = pd.DataFrame({
        "open": open_,
        "high": top * (1 + rng.uniform(0, 0.003, n)),
        "low": bot * (1 - rng.uniform(0, 0.003, n)),
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    return add_features(df)


def test_env_short_action_and_obs_dim(tmp_path):
    df = _synthetic_df(600).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    p = np.full(len(df), 0.5)
    broker = PaperBroker(tmp_path / "e.db")
    env = TradingEnv(df, p, broker)
    assert env.observation_space.shape == (25,)  # 21 feats + p_up + 3 portfolio
    assert env.action_space.n == 4
    env.reset(seed=0, options={"start": 30})
    env.step(3)  # SHORT
    assert broker.btc < 0
    env.step(1)  # BUY covers (tranche-capped): can reach flat, never long
    assert broker.btc <= 0


def test_v5_feature_contract_and_causality():
    df = _synthetic_df(3000)
    assert len(FEATURE_COLUMNS) == 21
    for c in V5_FEATURES:
        assert c in FEATURE_COLUMNS
    sup = df.dropna(subset=FEATURE_COLUMNS)
    assert len(sup) > 2000
    assert sup[FEATURE_COLUMNS].min().min() >= -1.0001
    assert sup[FEATURE_COLUMNS].max().max() <= 1.0001
    # The volume trio must actually move on data with bar direction
    # (guards against a silently all-zero implementation).
    for c in ("obv_slope", "adl_slope", "netvol_rel"):
        assert sup[c].std() > 0.01, c
    cut = df.index[2000]
    trunc = add_features(df.loc[:cut + pd.Timedelta(hours=48)])
    a = df.loc[:cut, FEATURE_COLUMNS]
    bb = trunc.loc[:cut, FEATURE_COLUMNS]
    assert (a - bb).abs().max().max() < 1e-9  # no future leakage
