"""Trades-tape (Time & Sales) extension of the L2 collector: pins.

Adversarial pins for the market_trades side of src/data/l2_collector.py
+ src/data/l2_features.py (2026-11-18). The load-bearing detail is the
SIDE CONVENTION: Coinbase Advanced Trade documents the market_trades
`side` field as the MAKER (resting) order's side ("Each market trade
belongs to a side, which refers to the makers side" -- Advanced Trade
WebSocket Channels docs). The aggressor is the opposite side: a print
of side=BUY means a resting buy was hit by a seller, so the volume is
SELL-initiated. All buy/sell accounting uses the TAKER side.

Fixture trades for the minute aggregation (printed side -> taker):
  1. BUY  @ 100.00 x 0.25  -> taker SELL 0.25
  2. SELL @ 100.10 x 0.50  -> taker BUY  0.50  (large, >= 0.5 BTC)
  3. SELL @ 100.20 x 0.10  -> taker BUY  0.10
  4. BUY  @  99.90 x 1.00  -> taker SELL 1.00  (large)
  buy_vol = 0.60, sell_vol = 1.25, total = 1.85, n = 4 (2 buy / 2 sell)
  sum(price*qty) = 25.00 + 50.05 + 10.02 + 99.90 = 184.97
  vwap = 184.97 / 1.85, max_qty = 1.00, large_vol = 1.50

Hourly rollup divergence fixture: minute A buy 9 / sell 1 (imbalance
+0.8), minute B buy 1 / sell 2 (imbalance -1/3). The mean of the minute
imbalances is +0.2333; the imbalance of the SUMMED volumes is
(10-3)/13 = +0.5385. The rollup must produce the from-sums value.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.l2_collector import (
    CollectorState,
    TapeAccumulator,
    _handle_message,
    tape_minute_row,
    taker_side,
)
from data.l2_features import compute_trades_hour_rollup

H = 1699999200  # an exact hour boundary (2023-11-14T22:00:00Z)


# --------------------------------------------------------- side convention

def test_taker_side_inverts_maker_side():
    # printed side is the MAKER's; the aggressor is the other side
    assert taker_side("BUY") == "SELL"
    assert taker_side("SELL") == "BUY"
    assert taker_side("buy") is None      # venue prints upper-case only
    assert taker_side("") is None
    assert taker_side(None) is None


def test_printed_buy_lands_in_sell_vol_and_vice_versa():
    acc = TapeAccumulator()
    acc.add("BUY", 100.0, 0.25)           # resting buy hit -> taker sold
    assert acc.sell_vol == pytest.approx(0.25)
    assert acc.buy_vol == 0.0
    assert acc.n_sell == 1 and acc.n_buy == 0
    acc.add("SELL", 100.0, 0.25)          # resting sell lifted -> taker bought
    assert acc.buy_vol == pytest.approx(0.25)
    assert acc.n_buy == 1


# ------------------------------------------------------ minute aggregation

def _fixture_acc():
    acc = TapeAccumulator()
    acc.add("BUY", 100.00, 0.25)
    acc.add("SELL", 100.10, 0.50)
    acc.add("SELL", 100.20, 0.10)
    acc.add("BUY", 99.90, 1.00)
    return acc


def test_minute_aggregation_hand_computed():
    acc = _fixture_acc()
    assert acc.buy_vol == pytest.approx(0.60)
    assert acc.sell_vol == pytest.approx(1.25)
    assert acc.total_vol == pytest.approx(1.85)
    assert acc.n_trades == 4
    assert acc.n_buy == 2 and acc.n_sell == 2
    assert acc.vwap == pytest.approx(184.97 / 1.85)
    assert acc.max_qty == pytest.approx(1.00)
    assert acc.large_vol == pytest.approx(1.50)   # 0.50 + 1.00, both >= 0.5


def test_large_threshold_is_inclusive_at_half_btc():
    acc = TapeAccumulator()
    acc.add("SELL", 100.0, 0.4999)
    assert acc.large_vol == 0.0
    acc.add("SELL", 100.0, 0.5)
    assert acc.large_vol == pytest.approx(0.5)


def test_minute_row_zero_trade_live_minute_is_zeros_not_nulls():
    row = tape_minute_row(H, None, coverage=60.0, sampled_at=H + 60)
    assert row["buy_vol"] == 0.0 and row["sell_vol"] == 0.0
    assert row["total_vol"] == 0.0 and row["n_trades"] == 0
    assert row["max_qty"] == 0.0 and row["large_vol"] == 0.0
    assert row["vwap"] is None            # 0/0 is undefined, not 0
    assert row["coverage"] == 60.0


def test_minute_row_down_minute_is_nulls_with_true_coverage():
    row = tape_minute_row(H, None, coverage=0.0, sampled_at=H + 60)
    for col in ("buy_vol", "sell_vol", "total_vol", "n_trades", "n_buy",
                "n_sell", "vwap", "max_qty", "large_vol"):
        assert row[col] is None, col
    assert row["coverage"] == 0.0
    # partial coverage, still no trades seen: stream was up, tape quiet
    row = tape_minute_row(H, None, coverage=17.5, sampled_at=H + 60)
    assert row["total_vol"] == 0.0 and row["coverage"] == 17.5


def test_minute_row_with_trades_carries_values_and_coverage():
    row = tape_minute_row(H, _fixture_acc(), coverage=23.5, sampled_at=H + 60)
    assert row["buy_vol"] == pytest.approx(0.60)
    assert row["total_vol"] == pytest.approx(1.85)
    assert row["coverage"] == 23.5        # partial minute stays partial


# ------------------------------------------------------------ message path

def _mt_msg(trades, seq=1, ev_type="update"):
    return json.dumps({
        "channel": "market_trades",
        "sequence_num": seq,
        "events": [{"type": ev_type, "trades": trades}],
    })


def _trade(tid, side, price, size, iso_time):
    return {"trade_id": str(tid), "product_id": "BTC-USD",
            "price": str(price), "size": str(size), "side": side,
            "time": iso_time}


def test_handler_buckets_by_trade_time_and_dedupes():
    st = CollectorState()
    # a trade executed at 22:00:10Z but received in the next minute's
    # batch still belongs to minute H (bucketing is by TRADE time)
    t1 = _trade(1, "SELL", 100.10, 0.50, "2023-11-14T22:00:10.000000Z")
    _handle_message(st, _mt_msg([t1], seq=1), recv_ts=H + 75.0)
    assert H in st.tape_acc
    assert st.tape_acc[H].buy_vol == pytest.approx(0.50)   # SELL print -> buy
    assert st.trades_total == 1
    assert st.last_trade_ts == pytest.approx(H + 10.0)
    # the same trade_id arriving again (snapshot/update overlap) counts once
    _handle_message(st, _mt_msg([t1], seq=2, ev_type="snapshot"),
                    recv_ts=H + 76.0)
    assert st.tape_acc[H].n_trades == 1
    assert st.trades_total == 1
    # a different trade in the same minute accumulates
    t2 = _trade(2, "BUY", 100.00, 0.25, "2023-11-14T22:00:20.000000Z")
    _handle_message(st, _mt_msg([t2], seq=3), recv_ts=H + 77.0)
    assert st.tape_acc[H].sell_vol == pytest.approx(0.25)
    assert st.tape_acc[H].n_trades == 2


def test_handler_drops_trades_for_finalized_minutes():
    st = CollectorState()
    st.tape_open_from = H + 60            # minute H is already written
    old = _trade(9, "SELL", 100.0, 1.0, "2023-11-14T22:00:30.000000Z")
    _handle_message(st, _mt_msg([old], seq=1), recv_ts=H + 130.0)
    assert H not in st.tape_acc           # never merged into a written row
    assert st.trades_late_dropped == 1
    assert st.trades_total == 0


# ------------------------------------------------------------------ rollup

def _tminute(ts, buy=None, sell=None, n=0, vwap=None, maxq=None, large=None,
             coverage=60.0):
    if buy is None:                       # a down minute: NULL aggregates
        return {"ts": ts, "buy_vol": None, "sell_vol": None,
                "total_vol": None, "n_trades": None, "n_buy": None,
                "n_sell": None, "vwap": None, "max_qty": None,
                "large_vol": None, "coverage": coverage}
    total = buy + sell
    return {"ts": ts, "buy_vol": buy, "sell_vol": sell, "total_vol": total,
            "n_trades": n, "n_buy": None, "n_sell": None, "vwap": vwap,
            "max_qty": maxq, "large_vol": large, "coverage": coverage}


def test_trades_rollup_imbalance_from_sums_not_mean_of_minutes():
    rows = [
        _tminute(H, buy=9.0, sell=1.0, n=10, vwap=100.0, maxq=0.7,
                 large=2.0),
        _tminute(H + 60, buy=1.0, sell=2.0, n=5, vwap=110.0, maxq=0.4,
                 large=0.0),
        _tminute(H + 120, coverage=0.0),                    # down minute
        _tminute(H + 180, buy=0.0, sell=0.0, n=0, vwap=None, maxq=0.0,
                 large=0.0),                                # quiet minute
    ]
    r = compute_trades_hour_rollup(rows, H)
    assert r["buy_vol"] == pytest.approx(10.0)
    assert r["sell_vol"] == pytest.approx(3.0)
    # from the SUMS: (10-3)/13 = 0.5385 -- NOT the mean of the minute
    # imbalances, which would be (0.8 + (-1/3)) / 2 = 0.2333
    assert r["signed_imbalance"] == pytest.approx(7.0 / 13.0)
    assert r["signed_imbalance"] != pytest.approx((0.8 - 1.0 / 3.0) / 2.0,
                                                  abs=0.05)
    assert r["n_trades"] == 15
    # hour vwap is volume-weighted across minutes: (100*10 + 110*3) / 13
    assert r["vwap"] == pytest.approx(1330.0 / 13.0)
    assert r["max_qty"] == pytest.approx(0.7)
    assert r["large_share"] == pytest.approx(2.0 / 13.0)
    # minutes_present counts live minutes (incl. the quiet one), not the
    # down minute; coverage_mean covers every row inside the hour
    assert r["minutes_present"] == 3
    assert r["coverage_mean"] == pytest.approx((60.0 + 60.0 + 0.0 + 60.0) / 4)
    assert r["usable"] == 0


def test_trades_rollup_causality_boundary():
    rows = [
        _tminute(H - 60, buy=500.0, sell=0.0, n=99, vwap=1.0, maxq=9.0,
                 large=500.0),
        _tminute(H, buy=1.0, sell=1.0, n=2, vwap=100.0, maxq=0.6,
                 large=1.0),
        _tminute(H + 3600, buy=500.0, sell=0.0, n=99, vwap=1.0, maxq=9.0,
                 large=500.0),
    ]
    r = compute_trades_hour_rollup(rows, H)
    assert r["buy_vol"] == pytest.approx(1.0)     # poison rows excluded
    assert r["sell_vol"] == pytest.approx(1.0)
    assert r["signed_imbalance"] == pytest.approx(0.0)
    assert r["n_trades"] == 2
    assert r["minutes_present"] == 1


def test_trades_rollup_zero_volume_hour_has_null_ratios():
    rows = [_tminute(H + i * 60, buy=0.0, sell=0.0, n=0, vwap=None,
                     maxq=0.0, large=0.0) for i in range(60)]
    r = compute_trades_hour_rollup(rows, H)
    assert r["buy_vol"] == 0.0 and r["sell_vol"] == 0.0
    assert r["signed_imbalance"] is None          # 0/0, not 0
    assert r["vwap"] is None
    assert r["large_share"] is None
    assert r["max_qty"] == 0.0
    assert r["minutes_present"] == 60
    assert r["usable"] == 1                       # a quiet hour is usable


def test_trades_rollup_all_down_hour_is_unusable_not_missing():
    rows = [_tminute(H + i * 60, coverage=0.0) for i in range(60)]
    r = compute_trades_hour_rollup(rows, H)
    assert r["minutes_present"] == 0
    assert r["buy_vol"] is None and r["signed_imbalance"] is None
    assert r["coverage_mean"] == pytest.approx(0.0)
    assert r["usable"] == 0


def test_trades_rollup_usable_thresholds_are_exact():
    def usable_for(n_valid, coverage):
        rows = [_tminute(H + i * 60, buy=0.1, sell=0.1, n=1, vwap=100.0,
                         maxq=0.1, large=0.0, coverage=coverage)
                for i in range(n_valid)]
        return compute_trades_hour_rollup(rows, H)["usable"]

    assert usable_for(49, 60.0) == 0
    assert usable_for(50, 55.0) == 1
    assert usable_for(50, 54.9) == 0
