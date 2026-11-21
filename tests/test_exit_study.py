"""Exit & transition study mechanics (2026-10-22, Kass's request).

Pins: (1) the default ExitConfig IS the production v4 recipe, (2) the
new take-profit mechanism (wick triggers, long + short, % and R
forms, stop-wins ordering), (3) the three transition rules at the
broker seam (T0 pass-through, T1 one-motion reversal, T2 staged full
close), (4) trailing modes, and (5) the load-bearing identity check:
the default config driven through the study harness reproduces GBM
Lag F's recorded duel (training.db run 26: agg -1,871.18, P&L
-$4,416.98, 40 round trips) on the standard windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.exit_study import ExitConfig, StudyBroker, StudyEnv  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


def _broker(tmp_path, cfg=None, name="b.db"):
    return StudyBroker(tmp_path / name, starting_balance=100_000,
                       max_position_pct=0.25, stop_loss_pct=0.05,
                       cfg=cfg or ExitConfig())


def _fills(broker):
    import sqlite3
    with sqlite3.connect(broker.db_path) as con:
        return con.execute(
            "SELECT side, price, qty, fee, reason FROM trades ORDER BY id"
        ).fetchall()


# ---------------------------------------------------------------- config
def test_default_config_is_the_production_recipe():
    cfg = ExitConfig()
    assert cfg.stop_mult == PaperBroker.STOP_ATR_MULT == 2.0
    assert cfg.stop_min_pct == PaperBroker.STOP_MIN_PCT == 0.025
    assert cfg.stop_max_pct == PaperBroker.STOP_MAX_PCT == 0.08
    assert cfg.trail_activate_pct == PaperBroker.TRAIL_ACTIVATE_PCT == 0.02
    assert cfg.trail_atr_mult == PaperBroker.TRAIL_ATR_MULT == 1.5
    assert cfg.trail_min_pct == PaperBroker.TRAIL_MIN_PCT == 0.02
    assert cfg.trail_max_pct == PaperBroker.TRAIL_MAX_PCT == 0.06
    assert cfg.tp_mode is None and cfg.transition == "T0"


def test_default_broker_stop_level_matches_production(tmp_path):
    """Same entry, same ATR -> the study broker's stop level equals a
    plain PaperBroker's, bit for bit (defaults change nothing)."""
    prod = PaperBroker(tmp_path / "p.db", starting_balance=100_000,
                       max_position_pct=0.25, stop_loss_pct=0.05)
    study = _broker(tmp_path)
    for b in (prod, study):
        b.buy(100.0, ts="t0", atr_pct=0.02)
        b.check_stop(110.0, ts="t1", low=109.0, high=111.0)
    assert study.stop_level() == prod.stop_level()


# ------------------------------------------------------------- take-profit
def test_tp_long_triggers_on_wick_fills_at_level(tmp_path):
    b = _broker(tmp_path, ExitConfig(tp_mode="pct", tp_value=0.04))
    b.buy(100.0, ts="t0", atr_pct=0.01)  # stop clips to 2.5% -> 97.5
    fill = b.check_stop(102.0, ts="t1", low=101.0, high=104.5)
    assert fill is not None and fill.reason == "take_profit"
    assert fill.price == pytest.approx(104.0)  # wicked up, closed below
    assert b.btc == 0


def test_tp_long_close_above_target_fills_at_close(tmp_path):
    b = _broker(tmp_path, ExitConfig(tp_mode="pct", tp_value=0.04))
    b.buy(100.0, ts="t0", atr_pct=0.01)
    # Low stays above the armed trailing level (105 x (1-2%) = 102.9),
    # so only the TP can fire on this candle.
    fill = b.check_stop(105.0, ts="t1", low=103.5, high=106.0)
    assert fill is not None and fill.reason == "take_profit"
    assert fill.price == pytest.approx(105.0)  # closed through: the close


def test_tp_short_mirror(tmp_path):
    b = _broker(tmp_path, ExitConfig(tp_mode="pct", tp_value=0.04))
    b.short_open(100.0, ts="t0", atr_pct=0.01)  # stop clips to 102.5
    fill = b.check_stop(98.0, ts="t1", low=95.5, high=99.0)
    assert fill is not None and fill.reason == "take_profit"
    assert fill.price == pytest.approx(96.0)
    assert b.btc == 0


def test_tp_r_multiple_uses_initial_stop_distance(tmp_path):
    # atr 2% -> base stop pct = 2.0 * 2% = 4% -> 2R target = entry + 8%.
    b = _broker(tmp_path, ExitConfig(tp_mode="R", tp_value=2.0))
    b.buy(100.0, ts="t0", atr_pct=0.02)
    assert b.tp_level() == pytest.approx(108.0)
    fill = b.check_stop(103.0, ts="t1", low=102.0, high=108.2)
    assert fill is not None and fill.reason == "take_profit"
    assert fill.price == pytest.approx(108.0)


def test_stop_wins_when_one_candle_spans_stop_and_tp(tmp_path):
    b = _broker(tmp_path, ExitConfig(tp_mode="pct", tp_value=0.04))
    b.buy(100.0, ts="t0", atr_pct=0.02)  # stop 96.0, TP 104.0
    fill = b.check_stop(100.0, ts="t1", low=95.0, high=105.0)
    assert fill is not None and fill.reason == "stop_loss"
    assert fill.price == pytest.approx(96.0)
    assert len(_fills(b)) == 2  # open + stop only, no TP fill


def test_tp_off_by_default_no_fill_above_target(tmp_path):
    b = _broker(tmp_path)  # production: no take-profit exists
    b.buy(100.0, ts="t0", atr_pct=0.01)
    # +20% candle whose low (118) stays above the armed trailing level
    # (120 x (1-2%) = 117.6): production holds -- no TP mechanism fires.
    fill = b.check_stop(120.0, ts="t1", low=118.0, high=121.0)
    assert fill is None and b.btc > 0


# --------------------------------------------------------------- trailing
def test_trail_tight_trails_closer_than_current(tmp_path):
    # RENAMED 2026-11-21 (was test_trail_tight_arms_earlier_than_current):
    # Kass's 2026-11-02 directive moved the production arm from 3% to
    # 2%, colliding with tight's 2% arm -- at +2.5% both presets arm
    # now, so "arms earlier" no longer distinguishes them. What still
    # distinguishes tight is the closer trail (1.0x ATR vs 1.5x ATR):
    # both arm, and tight's level sits higher (closer to price).
    cur = _broker(tmp_path, ExitConfig(), "cur.db")
    tight = _broker(tmp_path, ExitConfig(trail="tight"), "tight.db")
    for b in (cur, tight):
        b.buy(100.0, ts="t0", atr_pct=0.02)
        b.check_stop(102.5, ts="t1", low=102.0, high=102.6)
    clvl, carmed = cur.stop_level()
    tlvl, tarmed = tight.stop_level()
    assert carmed is True and tarmed is True
    assert tlvl > clvl  # tight trails closer
    assert clvl == pytest.approx(102.5 * (1 - 0.03))
    assert tlvl == pytest.approx(102.5 * (1 - 0.02))


def test_trail_off_never_arms(tmp_path):
    b = _broker(tmp_path, ExitConfig(trail="off"))
    b.buy(100.0, ts="t0", atr_pct=0.02)
    b.check_stop(130.0, ts="t1", low=125.0, high=131.0)
    lvl, armed = b.stop_level()
    assert armed is False and lvl == pytest.approx(96.0)  # initial stop


# ------------------------------------------------------------- transitions
def _open_short(b, price=100.0):
    b.short_open(price, ts="t0", atr_pct=0.02)
    assert b.btc < 0


def _open_long(b, price=100.0):
    b.buy(price, ts="t0", atr_pct=0.02)
    assert b.btc > 0


def test_t0_short_open_while_long_closes_only(tmp_path):
    b = _broker(tmp_path)  # T0 = production pass-through
    _open_long(b)
    b.signal_dir = -1
    b.short_open(100.0, ts="t1", atr_pct=0.02)
    assert b.btc == 0
    assert len(_fills(b)) == 2  # buy + closing sell, no short opened


def test_t0_buy_while_short_covers_one_tranche_only(tmp_path):
    b = _broker(tmp_path)
    _open_short(b)
    _open_short(b)  # second tranche: short is now ~2 tranches deep
    before = b.btc
    b.signal_dir = 1
    b.buy(100.0, ts="t1")
    assert b.btc < 0 and b.btc > before  # reduced, not closed, no long


def test_t1_long_to_short_reverses_in_one_step(tmp_path):
    b = _broker(tmp_path, ExitConfig(transition="T1"))
    _open_long(b)
    held = b.btc
    b.signal_dir = -1
    b.short_open(100.0, ts="t1", atr_pct=0.02)
    fills = _fills(b)
    assert b.btc < 0  # flipped in the same step
    assert len(fills) == 3  # buy, closing sell, short_open
    assert fills[1][0] == "SELL" and fills[1][2] == pytest.approx(held)
    assert fills[2][4] == "short_open"
    # fees charged on BOTH legs' actual volume
    assert fills[1][3] == pytest.approx(fills[1][1] * fills[1][2] * 0.0015)
    assert fills[2][3] > 0


def test_t1_short_to_long_reverses_in_one_step(tmp_path):
    b = _broker(tmp_path, ExitConfig(transition="T1"))
    _open_short(b)
    held = -b.btc
    b.signal_dir = 1
    b.buy(100.0, ts="t1", atr_pct=0.02)
    fills = _fills(b)
    assert b.btc > 0
    assert len(fills) == 3  # short_open, cover, buy
    assert fills[1][4] == "cover" and fills[1][2] == pytest.approx(held)
    assert fills[2][0] == "BUY"


def test_t1_band_buy_while_short_is_not_a_flip(tmp_path):
    """A band BUY (signal gone, decide() -> BUY to flatten) must not
    open a long under T1: only an active opposite signal flips."""
    b = _broker(tmp_path, ExitConfig(transition="T1"))
    _open_short(b)
    b.signal_dir = 0  # band: no active long signal
    b.buy(100.0, ts="t1")
    assert b.btc <= 0  # covered (partially or fully), never went long


def test_t2_short_flip_closes_fully_and_opens_nothing(tmp_path):
    b = _broker(tmp_path, ExitConfig(transition="T2"))
    _open_short(b)
    _open_short(b)  # two tranches deep
    b.signal_dir = 1
    b.buy(100.0, ts="t1")
    assert b.btc == 0  # fully flat in one step (T0 would still be short)
    fills = _fills(b)
    assert fills[-1][4] == "cover"
    assert len(fills) == 3  # 2 opens + 1 full cover; NO long opened
    # Next candle: signal persists -> a plain buy opens the long.
    b.buy(100.0, ts="t2", atr_pct=0.02)
    assert b.btc > 0


def test_t2_long_flip_closes_only(tmp_path):
    b = _broker(tmp_path, ExitConfig(transition="T2"))
    _open_long(b)
    b.signal_dir = -1
    b.short_open(100.0, ts="t1", atr_pct=0.02)
    assert b.btc == 0 and len(_fills(b)) == 2


def test_env_sets_signal_dir_from_conviction(tmp_path):
    """Through the real env: a short opened on a bearish candle, then
    a bullish flip candle under T1 must end the step LONG (T0 control
    ends it still short after one tranche cover)."""
    from features.technical_indicators import FEATURE_COLUMNS
    from models.gbm_policy import decide

    n = 30  # >= one 24-step episode + next-open buffer
    rows = {c: np.zeros(n) for c in FEATURE_COLUMNS}
    rows.update({"open": np.full(n, 100.0), "high": np.full(n, 100.5),
                 "low": np.full(n, 99.5), "close": np.full(n, 100.0),
                 "atr_pct_raw": np.full(n, 0.02)})
    sup = pd.DataFrame(rows)
    # Conviction by row (obs[21] = 2*p_up - 1): row 0 bearish (open
    # short), row 1 bullish (the flip candle), flat band afterwards.
    p_up = np.array([0.2, 0.8, 0.8] + [0.5] * (n - 3))

    def run(cfg):
        broker = StudyBroker(tmp_path / f"e{cfg.transition}.db",
                             starting_balance=100_000, max_position_pct=0.25,
                             stop_loss_pct=0.05, cfg=cfg)
        env = StudyEnv(sup, p_up, broker, fill_mode="next_open")
        obs, _ = env.reset(seed=0, options={"start": 0})
        obs, *_ = env.step(decide(obs[21], obs[23]))   # SHORT fires
        assert broker.btc < 0
        obs, *_ = env.step(decide(obs[21], obs[23]))   # BUY flip fires
        return broker.btc

    assert run(ExitConfig(transition="T1")) > 0   # reversed in one step
    assert run(ExitConfig(transition="T0")) < 0   # one tranche covered


# ---------------------------------------------------------------- identity
@pytest.mark.slow
def test_default_config_reproduces_run26_duel():
    """Load-bearing: GBM Lag F (run 26) recorded agg -1,871.18 / P&L
    -$4,416.98 / 40 round trips on the standard windows with the
    production exits. The study harness at default config (which IS
    the production recipe) must reproduce those numbers."""
    from models.exit_study import (evaluate_windows_study,
                                   load_lagf_signal)
    windows, p_ups = load_lagf_signal("standard")
    res = evaluate_windows_study(windows, p_ups, ExitConfig())
    assert res["total_round_trips"] == 40
    assert res["agg_score"] == pytest.approx(-1871.1787, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-4416.9757, abs=0.01)
    assert res["bench_pnl"] == pytest.approx(717.5366, abs=0.01)
