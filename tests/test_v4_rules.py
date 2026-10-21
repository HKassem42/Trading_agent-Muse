"""Regression tests for the v4 upgrade rules (2026-10-20).

Covers: exposure cap, ATR-sized stops, trailing take-profit exits,
per-fill context persistence, next-candle-open duel fills, and the
multi-timeframe feature contract/causality. Run: .venv/bin/python -m pytest tests/ -q
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


def test_exposure_cap(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    fills = [b.buy(100.0, ts=f"t{i}", atr_pct=0.005) for i in range(6)]
    assert b.btc * 100 <= 75_000 + 1  # 75% cap, never pyramided past it
    assert sum(f is not None for f in fills) < 6


def test_atr_stop_and_trailing_exit(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=0.005)
    lvl, trailing = b.stop_level()
    assert abs(lvl - 97.5) < 1e-9 and not trailing  # floor: max(2*.005, .025)
    assert b.check_stop(105.0, ts="t1", low=104.0) is None
    lvl, trailing = b.stop_level()
    assert trailing and abs(lvl - 102.9) < 1e-9  # armed: hw*(1-.02)
    f = b.check_stop(103.5, ts="t2", low=102.0)
    assert f is not None and f.reason == "trailing_stop" and b.btc == 0


def test_flat_fallback_stop(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0")  # no ATR -> legacy flat 5%
    lvl, _ = b.stop_level()
    assert abs(lvl - 95.0) < 1e-9


def test_fill_context_persisted(tmp_path):
    db = tmp_path / "t.db"
    b = PaperBroker(db)
    b.buy(100.0, ts="t0", context=json.dumps({"f": [0.1] * 16, "p": 0.6}))
    row = sqlite3.connect(db).execute("SELECT context FROM trades").fetchone()
    assert json.loads(row[0])["p"] == 0.6


def _synthetic_df(n=80):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    return add_features(df)


def test_mtf_features_causal_and_in_contract():
    df = _synthetic_df(3000)
    sup = df.dropna(subset=FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 21  # 16 since v4, +5 in v5 (Kass's indicators)
    assert sup[FEATURE_COLUMNS].min().min() >= -1.0001
    assert sup[FEATURE_COLUMNS].max().max() <= 1.0001
    cut = df.index[2000]
    trunc = add_features(df.loc[:cut + pd.Timedelta(hours=48)])
    a = df.loc[:cut, FEATURE_COLUMNS]
    b = trunc.loc[:cut, FEATURE_COLUMNS]
    assert (a - b).abs().max().max() < 1e-9  # no future leakage


def test_next_open_fill(tmp_path):
    # 600 rows: the daily-context features need ~480h of warmup.
    df = _synthetic_df(600).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    p = np.full(len(df), 0.5)
    for mode, expect in (("close", float(df.iloc[30]["close"])),
                         ("next_open", float(df.iloc[31]["open"]))):
        broker = PaperBroker(tmp_path / f"{mode}.db")
        env = TradingEnv(df, p, broker, fill_mode=mode)
        env.reset(seed=0, options={"start": 30})
        env.step(1)  # BUY
        row = sqlite3.connect(tmp_path / f"{mode}.db").execute(
            "SELECT price FROM trades WHERE side='BUY'").fetchone()
        assert row is not None and abs(row[0] - expect) < 1e-6, mode
