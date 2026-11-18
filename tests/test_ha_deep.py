"""HA-Deep (runs 87/88) — adversarial tests, written BEFORE the
grading implementation per the house quality standard. Design
frozen in refs/ha-deep-study-2026-10-30.md.

Pins: the HA recursion on hand-computed candles (incl. the
zero-body neutral rule), streak/shift state machine on crafted
color runs (shift = color change vs the previous NONZERO run;
same color after neutral candles is not a shift; one event
per run), causality of all 58 features + the 6 sequence series
under future garble, bucket-END stamping of the multi-timeframe
HA state (values may change only at bucket-boundary decision
rows), the common training row set (purge + trailing-window
finiteness), the judge direction mapping, the Shift Trader
state machine (BUY/SHORT/HOLD only; flips as close+open at one
next-open; judged skips; stop-out -> flat until the next
confirmed shift; last-candle decisions counted, never filled),
next-open fill timing, ledger re-sum parity, and the compact
family registry for ha_deep / ha_shift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import ha_deep as hd  # noqa: E402
from training import meta_label as ml  # noqa: E402


def _df(rows, start="2025-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                       "volume"], index=idx,
                        dtype=float)


# ------------------------------------------------------- HA recursion
def test_ha_recursion_hand_computed():
    df = _df([
        (100, 104, 96, 102, 10),
        (102, 108, 100, 106, 10),
        (106, 107, 101, 102, 10),
    ])
    ha = hd.ha_candles(df)
    # HA open(0) = (O+C)/2 = 101; HA close(0) = mean(100,104,96,102).
    assert ha["ha_open"].iloc[0] == pytest.approx(101.0)
    assert ha["ha_close"].iloc[0] == pytest.approx(100.5)
    assert ha["ha_high"].iloc[0] == pytest.approx(104.0)
    assert ha["ha_low"].iloc[0] == pytest.approx(96.0)
    # HA open(1) = (101 + 100.5)/2; HA close(1) = mean(102,108,100,106).
    assert ha["ha_open"].iloc[1] == pytest.approx(100.75)
    assert ha["ha_close"].iloc[1] == pytest.approx(104.0)
    assert ha["ha_open"].iloc[2] == pytest.approx(102.375)
    assert ha["ha_close"].iloc[2] == pytest.approx(104.0)
    colors = hd.ha_colors(ha)
    assert list(colors) == [-1, 1, 1]  # body -0.5, +3.25, +1.625


def test_ha_zero_body_is_neutral_and_ends_the_run():
    df = _df([
        (100, 104, 96, 102, 10),
        (102, 108, 100, 106, 10),
        (102, 104, 100, 103.5, 10),  # HA close == HA open == 102.375
    ])
    ha = hd.ha_candles(df)
    assert float(ha["ha_close"].iloc[2] - ha["ha_open"].iloc[2]) == 0.0
    colors = hd.ha_colors(ha)
    assert list(colors) == [-1, 1, 0]
    st = hd.run_state(colors)
    assert list(st["streak"]) == [-1, 1, 0]
    assert st["run_id"][2] == -1
    # the neutral row remembers the last completed nonzero run (+1)
    assert st["prev_run_color"][2] == 1


# ------------------------------------------------------- runs & shifts
def test_run_state_crafted():
    colors = np.array([1, 1, 1, -1, -1, 0, 1, -1, -1, -1])
    st = hd.run_state(colors)
    assert list(st["streak"]) == [1, 2, 3, -1, -2, 0, 1, -1, -2, -3]
    assert list(st["run_id"]) == [0, 0, 0, 1, 1, -1, 2, 3, 3, 3]
    # prev nonzero run color: run 1 (red) follows green -> prev +1;
    # the neutral row's last completed nonzero run is the RED run
    # -> -1; run 2 (green) follows red -> prev -1; run 3 (red)
    # follows green -> prev +1.
    assert list(st["prev_run_color"]) == [0, 0, 0, 1, 1, -1, -1, 1, 1, 1]


def test_shift_events_l2_and_l3():
    colors = np.array([1, 1, -1, -1, 1, 1])
    ev2 = hd.shift_events(colors, 2)
    # first run is never a shift; red run confirms at idx 3 (shift
    # -1); green run confirms at idx 5 (shift +1).
    assert list(ev2) == [0, 0, 0, -1, 0, 1]
    colors3 = np.array([1, 1, 1, -1, -1, -1])
    assert list(hd.shift_events(colors3, 3)) == [0, 0, 0, 0, 0, -1]
    # L=2: the green run is first (never a shift); the red run
    # reaches length 2 at index 4.
    assert list(hd.shift_events(colors3, 2)) == [0, 0, 0, 0, -1, 0]


def test_shift_same_color_after_neutral_is_not_a_shift():
    colors = np.array([1, 1, 0, 1, 1, -1, -1])
    ev = hd.shift_events(colors, 2)
    # the second green run repeats the previous nonzero color:
    # no event. The red run after it IS a shift.
    assert list(ev) == [0, 0, 0, 0, 0, 0, -1]


def test_shift_events_fire_once_per_run():
    colors = np.array([-1, -1, 1, 1, 1, 1])
    ev = hd.shift_events(colors, 2)
    assert list(ev) == [0, 0, 0, 1, 0, 0]


# ------------------------------------------------------- frame causality
def _random_df(n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    steps = rng.normal(0, 0.004, n).cumsum()
    close = 100.0 * np.exp(steps)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_features_causal_under_future_garble():
    df = _random_df()
    frame, seq, _aux = hd.build_ha_frame(df)
    g = 600  # a midnight row: 4h and 1d buckets align here
    assert df.index[g].hour == 0
    garbled = df.copy()
    garbled.iloc[g:] = garbled.iloc[g:] * 1.7
    garbled.iloc[g:, garbled.columns.get_loc("volume")] *= 3.0
    frame2, seq2, _ = hd.build_ha_frame(garbled)
    pd.testing.assert_frame_equal(frame.iloc[:g], frame2.iloc[:g],
                                  check_exact=True)
    pd.testing.assert_frame_equal(seq.iloc[:g], seq2.iloc[:g],
                                  check_exact=True)
    # the garble must actually bite somewhere later (else the
    # test could pass vacuously)
    later = frame.iloc[g + 48:]
    later2 = frame2.iloc[g + 48:]
    assert not later.equals(later2)


def test_mtf_state_changes_only_at_bucket_boundaries():
    df = _random_df()
    frame, _seq, _aux = hd.build_ha_frame(df)
    hours = df.index.hour.to_numpy()
    for col, ok_hours in (("ha4_color", {3, 7, 11, 15, 19, 23}),
                          ("ha4_streak", {3, 7, 11, 15, 19, 23}),
                          ("ha1d_color", {23}),
                          ("ha1d_streak", {23})):
        v = frame[col].to_numpy(dtype=float)
        prev = np.concatenate([[np.nan], v[:-1]])
        changed = np.isfinite(v) & np.isfinite(prev) & (v != prev)
        bad = [int(h) for h, ch in zip(hours, changed) if ch]
        assert all(h in ok_hours for h in bad), (col, bad[:5])


def test_frame_columns_frozen_order():
    assert hd.FEATURE_COLUMNS_HA == (
        hd.BLOCK1_COLUMNS + hd.BLOCK2_COLUMNS + hd.BLOCK3_COLUMNS)
    assert len(hd.FEATURE_COLUMNS_HA) == 58
    assert len(hd.BLOCK1_COLUMNS) == 22
    assert len(hd.BLOCK2_COLUMNS) == 21
    assert len(hd.BLOCK3_COLUMNS) == 15


# ------------------------------------------------------- common rows
def test_common_training_rows_purge_and_windows():
    n, split = 100, 80
    frame = pd.DataFrame(np.ones((n, 58)), columns=hd.FEATURE_COLUMNS_HA)
    frame.iloc[30, 0] = np.nan          # poisons windows t in [30, 54]
    pick2_finite = np.ones(n, dtype=bool)
    pick2_finite[60] = False
    tb = np.ones(n)
    tb[50] = np.nan
    nc = np.ones(n)
    rows = hd.common_training_rows(frame, pick2_finite, tb, nc, split)
    expected = [t for t in range(24, split)
                if t + 24 < split and t != 50 and t != 60
                and not (30 <= t <= 54)]
    assert list(rows) == expected


# ------------------------------------------------------- judge mapping
def test_judge_lot_direction_mapping():
    assert hd.judge_lot(0.8, 1) == pytest.approx(0.8)
    assert hd.judge_lot(0.8, -1) == pytest.approx(0.2)
    assert hd.judge_lot(0.3, -1) == pytest.approx(0.7)


# ------------------------------------------------------- shift trader
def _span(n=10, price=100.0):
    o = np.full(n, price)
    return o.copy(), o.copy(), o.copy(), o.copy()


def _run_span(tmp_path, events, lot=None, judged=False, o=None, h=None,
              lo=None, c=None, atr=0.01):
    n = len(events)
    if o is None:
        o, h, lo, c = _span(n)
    stamps = pd.date_range("2026-01-01", periods=n, freq="1h")
    if lot is None:
        lot = np.full(n, np.nan)
    db = tmp_path / "span.db"
    rec = hd.run_shift_span(o, h, lo, c, np.full(n, atr),
                            np.asarray(events), np.asarray(lot, dtype=float),
                            stamps, db, judged)
    return rec, db


def test_trader_entry_hold_flip(tmp_path):
    events = [0, 0, 1, 0, 0, 0, -1, 0, 0, 0]
    rec, db = _run_span(tmp_path, events)
    # green shift at candle 2 -> BUY fills at candle 3's open;
    # red shift at candle 6 -> flip fills at candle 7's open
    # (SELL close + SELL short_open). End: short.
    assert rec["trades"] == 3
    assert rec["action_counts"] == {"hold": 8, "buy": 1, "short": 1}
    resumed = ml.resum_episode_pnl(db, 100.0)
    assert resumed == pytest.approx(rec["pnl"], abs=1e-6)
    # flat prices: P&L is exactly the fee drag
    assert rec["pnl"] == pytest.approx(-rec["fees"], abs=1e-6)
    assert rec["time_in_position"] == pytest.approx(0.7)


def test_trader_next_open_fill_price(tmp_path):
    n = 6
    o = np.array([100.0, 100.0, 111.0, 111.0, 111.0, 111.0])
    h = o + 1.0
    lo = o - 1.0
    c = o.copy()
    events = [0, 1, 0, 0, 0, 0]
    rec, _db = _run_span(tmp_path, events, o=o, h=h, lo=lo, c=c)
    assert rec["trades"] == 1
    assert rec["fills"][0]["price"] == pytest.approx(111.0)  # candle 2 open


def test_trader_same_direction_event_while_positioned_is_noop(tmp_path):
    events = [0, 0, 1, 0, 1, 0, 0]
    rec, _db = _run_span(tmp_path, events)
    assert rec["trades"] == 1  # only the first entry
    assert rec["action_counts"]["buy"] == 1


def test_trader_last_candle_decision_counted_not_filled(tmp_path):
    events = [0, 0, 0, 0, 1]
    rec, _db = _run_span(tmp_path, events)
    assert rec["trades"] == 0
    assert rec["action_counts"] == {"hold": 4, "buy": 1, "short": 0}


def test_trader_judged_skip_and_take(tmp_path):
    events = [0, 0, 1, 0, 0, 0, -1, 0, 0, 0]
    lot = np.full(10, np.nan)
    lot[2] = 0.40   # green shift judged "a little" -> skipped
    lot[6] = 0.90   # red shift judged "a lot" -> SHORT from flat
    rec, _db = _run_span(tmp_path, events, lot=lot, judged=True)
    assert rec["trades"] == 1
    assert rec["fills"][0]["reason"] == "short_open"
    log = rec["judge_log"]
    assert len(log) == 2
    assert log[0]["verdict"] == "skipped" and log[0]["direction"] == 1
    assert log[1]["verdict"] == "taken" and log[1]["direction"] == -1
    assert log[1]["context"] == "entry"


def test_trader_judged_little_opposing_shift_does_not_close(tmp_path):
    events = [0, 0, 1, 0, 0, 0, -1, 0, 0, 0]
    lot = np.full(10, np.nan)
    lot[2] = 0.95   # enter long
    lot[6] = 0.30   # opposing red shift judged little -> hold position
    rec, _db = _run_span(tmp_path, events, lot=lot, judged=True)
    assert rec["trades"] == 1  # the entry only; never closed
    assert rec["judge_log"][1]["verdict"] == "skipped"
    assert rec["judge_log"][1]["context"] == "flip"


def test_trader_stop_out_then_flat_until_next_shift(tmp_path):
    n = 14
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    c = np.full(n, 100.0)
    # entry at candle 3 open (100); candle 5 wicks to 95, through
    # the stop (atr 2% -> stop pct 4% -> level 96): stop_loss.
    lo[5] = 95.0
    c[5] = 97.0
    h[5] = 100.5
    o[6:] = 97.0
    h[6:] = 97.5
    lo[6:] = 96.5
    c[6:] = 97.0
    events = np.zeros(n, dtype=int)
    events[2] = 1    # enter long at candle 3
    events[8] = -1   # next confirmed shift: enter short at candle 9
    rec, db = _run_span(tmp_path, events, o=o, h=h, lo=lo, c=c, atr=0.02)
    reasons = [f["reason"] for f in rec["fills"]]
    assert "stop_loss" in reasons
    # fills: BUY entry, SELL stop_loss, SELL short_open
    assert rec["trades"] == 3
    resumed = ml.resum_episode_pnl(db, 97.0)
    assert resumed == pytest.approx(rec["pnl"], abs=1e-6)


def test_trader_stopped_streak_never_refires(tmp_path):
    # A span whose only event precedes a stop: after the stop the
    # book stays flat (the event is spent; nothing re-fires).
    n = 8
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    c = np.full(n, 100.0)
    lo[4] = 90.0
    c[4] = 95.0
    events = np.zeros(n, dtype=int)
    events[1] = 1
    rec, _db = _run_span(tmp_path, events, o=o, h=h, lo=lo, c=c, atr=0.02)
    assert rec["trades"] == 2  # entry + stop exit only
    assert rec["time_in_position"] < 0.5


# ------------------------------------------------------- compact registry
def test_compact_family_registry_ha():
    sys.path.insert(0, str(ROOT / "src" / "dashboards"))
    import compact  # noqa: E402
    assert compact._FAMILY_CODES["ha_deep"] == 20
    assert compact._FAMILY_CODES["ha_shift"] == 21
    run_e = {"id": 87, "ts": "2026-10-30T21:00:00+00:00",
             "kind": "stacker", "promoted": 0,
             "metrics": {"family": "ha_deep", "challenger_acc": 0.55,
                         "champion_acc": 0.51,
                         "ha_deep": {"label": "HA-Deep Eyes",
                                     "verdict": "FAIL"}}}
    name_e = compact._display_name(run_e, None)
    assert name_e == "HA-Deep Eyes"
    row_e = compact._model_entry(run_e, name_e)
    assert row_e[3] == 20 and row_e[5] == 0.55 and row_e[7] == "FAIL"
    run_t = {"id": 88, "ts": "2026-10-30T21:05:00+00:00",
             "kind": "ppo", "promoted": 0,
             "metrics": {"family": "ha_shift",
                         "challenger": {"score": 12.0, "total_pnl": 34.0},
                         "ha_shift": {"label": "HA Shift Trader",
                                      "verdict": "FAIL"}}}
    name_t = compact._display_name(run_t, None)
    assert name_t == "HA Shift Trader"
    row_t = compact._model_entry(run_t, name_t)
    assert row_t[3] == 21 and row_t[5] == 12.0 and row_t[6] == 34
