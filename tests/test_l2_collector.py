"""Coinbase Level 2 order-book collector: load-bearing math pins.

Adversarial pins for src/data/l2_collector.py + src/data/l2_features.py
(2026-11-18). The collector's ONLY product is numbers a future study
will trust, so the feature math is pinned against a fixture book whose
expected values are computed by hand here -- never by calling the code
under test:

Fixture book (prices -> BTC qty):
  bids: 100.00:2.0, 99.90:1.0, 99.50:4.0, 99.00:1.0, 98.00:10.0
  asks: 100.10:1.0, 100.20:3.0, 100.50:2.0, 101.00:5.0, 102.50:1.0
  best bid 100.00 (2.0), best ask 100.10 (1.0), mid = 100.05.

Band membership around mid=100.05 (each band edge excludes exactly one
level, so an off-by-one in the band comparison flips a pinned value):
  +-0.1%: bid >= 99.94995 -> {100.00} = 2.0 ; ask <= 100.15005 -> {100.10} = 1.0
  +-0.5%: bid >= 99.54975 -> 2+1 = 3.0      ; ask <= 100.55025 -> 1+3+2 = 6.0
  +-1%:   bid >= 99.04950 -> 2+1+4 = 7.0    ; ask <= 101.05050 -> 1+3+2+5 = 11.0
  +-2%:   bid >= 98.04900 -> 2+1+4+1 = 8.0  ; ask <= 102.05100 -> 11.0 (102.50 out)

Hourly rollup pins: means over VALID minutes only, "end" = last valid
minute (trailing NULL minutes must not poison it), the usable flag at
its exact thresholds (50 minutes / 55s coverage), and the causality
boundary -- a minute at H+3600 or H-60 must never enter hour H's row.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.l2_collector import CoverageTracker, OrderBook, book_features
from data.l2_features import compute_hour_rollup

FIXTURE_BIDS = {100.00: 2.0, 99.90: 1.0, 99.50: 4.0, 99.00: 1.0, 98.00: 10.0}
FIXTURE_ASKS = {100.10: 1.0, 100.20: 3.0, 100.50: 2.0, 101.00: 5.0, 102.50: 1.0}

H = 1699999200  # an exact hour boundary (H % 3600 == 0)


# ---------------------------------------------------------------- book math

def test_book_features_hand_computed():
    f = book_features(dict(FIXTURE_BIDS), dict(FIXTURE_ASKS))
    assert f is not None
    assert f["mid"] == pytest.approx(100.05)
    # spread = (100.10 - 100.00) / 100.05 * 1e4
    assert f["spread_bps"] == pytest.approx(0.10 / 100.05 * 1e4)
    assert f["spread_bps"] == pytest.approx(9.9950, abs=1e-3)
    # microprice = (best_ask*bid_qty + best_bid*ask_qty) / (bid_qty + ask_qty)
    expected_micro = (100.10 * 2.0 + 100.00 * 1.0) / (2.0 + 1.0)
    assert f["micro_dev_bps"] == pytest.approx(
        (expected_micro - 100.05) / 100.05 * 1e4)
    assert f["micro_dev_bps"] == pytest.approx(1.6658, abs=1e-3)
    # imbalances from the band sums worked out in the module docstring
    assert f["imb_10bps"] == pytest.approx((2.0 - 1.0) / (2.0 + 1.0))
    assert f["imb_50bps"] == pytest.approx((3.0 - 6.0) / (3.0 + 6.0))
    assert f["imb_100bps"] == pytest.approx((7.0 - 11.0) / (7.0 + 11.0))
    assert f["imb_200bps"] == pytest.approx((8.0 - 11.0) / (8.0 + 11.0))
    assert f["depth_1pct_btc"] == pytest.approx(7.0 + 11.0)
    assert f["best_bid_qty"] == 2.0
    assert f["best_ask_qty"] == 1.0
    assert f["n_bid_levels"] == 5
    assert f["n_ask_levels"] == 5


def test_book_features_empty_or_crossed_side_is_none():
    assert book_features({}, dict(FIXTURE_ASKS)) is None
    assert book_features(dict(FIXTURE_BIDS), {}) is None
    assert book_features({}, {}) is None
    # crossed book (best bid >= best ask) is not a real book: refuse
    assert book_features({100.20: 1.0}, {100.10: 1.0}) is None


# ------------------------------------------------------------- update path

def _u(side, price, qty):
    return {"side": side, "price_level": str(price), "new_quantity": str(qty)}


def test_snapshot_then_updates_add_change_remove():
    book = OrderBook()
    book.apply_snapshot([
        _u("bid", "100.00", "2.0"),
        _u("bid", "99.90", "1.0"),
        _u("offer", "100.10", "1.0"),
    ])
    assert book.best_bid() == (100.00, 2.0)
    assert book.best_ask() == (100.10, 1.0)

    # change quantity at the best bid
    book.apply_update([_u("bid", "100.00", "0.5")])
    assert book.best_bid() == (100.00, 0.5)

    # quantity 0 removes the level entirely
    book.apply_update([_u("bid", "100.00", "0")])
    assert 100.00 not in book.bids
    assert book.best_bid() == (99.90, 1.0)

    # a brand-new level inside the spread becomes the best ask
    book.apply_update([_u("offer", "100.05", "0.25")])
    assert book.best_ask() == (100.05, 0.25)

    # a fresh snapshot REPLACES the book, it does not merge into it
    book.apply_snapshot([_u("bid", "50.00", "3.0")])
    assert book.bids == {50.00: 3.0}
    assert book.asks == {}
    assert book_features(book.bids, book.asks) is None


# --------------------------------------------------------------- coverage

def test_coverage_tracker_partial_minute():
    tr = CoverageTracker()
    tr.set_live(True, 1000.0)          # live from 1000 (inside minute 960)
    tr.set_live(False, 1030.0)         # dies at 1030 (inside minute 1020)
    # the live interval [1000, 1030] splits across the two minutes:
    # minute 960 gets [1000,1020] = 20s, minute 1020 gets [1020,1030] = 10s
    assert tr.coverage_for(960, 1030.0) == pytest.approx(20.0)
    assert tr.coverage_for(1020, 1030.0) == pytest.approx(10.0)


def test_coverage_tracker_live_across_finalize():
    tr = CoverageTracker()
    tr.set_live(True, 1010.0)
    # finalizing minute 960 while still live banks the tail and re-anchors
    assert tr.coverage_for(960, 1020.0) == pytest.approx(10.0)
    assert tr.is_live
    tr.set_live(False, 1130.0)
    assert tr.coverage_for(1020, 1130.0) == pytest.approx(60.0)
    assert tr.coverage_for(1080, 1130.0) == pytest.approx(50.0)
    assert not tr.is_live


# ---------------------------------------------------------------- rollup

def _minute(ts, spread=10.0, micro=2.0, imb=0.1, depth=20.0, coverage=60.0,
            valid=True):
    if not valid:
        return {"ts": ts, "mid": None, "spread_bps": None,
                "micro_dev_bps": None, "imb_10bps": None, "imb_50bps": None,
                "imb_100bps": None, "imb_200bps": None,
                "depth_1pct_btc": None, "coverage": coverage}
    return {"ts": ts, "mid": 100.0, "spread_bps": spread,
            "micro_dev_bps": micro, "imb_10bps": imb, "imb_50bps": imb,
            "imb_100bps": imb, "imb_200bps": imb,
            "depth_1pct_btc": depth, "coverage": coverage}


def test_rollup_means_end_usable_and_window():
    rows = []
    for i in range(55):  # minutes 0..54 valid
        if i == 54:      # last valid minute carries distinct values
            rows.append(_minute(H + i * 60, spread=21.0, micro=3.0,
                                imb=0.5, depth=26.0))
        else:
            rows.append(_minute(H + i * 60))
    for i in (55, 56, 57):  # dead-at-sample minutes: NULL features
        rows.append(_minute(H + i * 60, coverage=30.0, valid=False))
    # minutes 58, 59: no rows at all (process down)
    # poison rows just outside the hour must be ignored
    rows.append(_minute(H - 60, spread=999.0))
    rows.append(_minute(H + 3600, spread=999.0))

    r = compute_hour_rollup(rows, H)
    assert r["ts"] == H
    assert r["minutes_present"] == 55
    # means over the 55 valid minutes only
    assert r["spread_bps_mean"] == pytest.approx((54 * 10.0 + 21.0) / 55)
    assert r["micro_dev_bps_mean"] == pytest.approx((54 * 2.0 + 3.0) / 55)
    assert r["imb_10bps_mean"] == pytest.approx((54 * 0.1 + 0.5) / 55)
    assert r["depth_1pct_mean"] == pytest.approx((54 * 20.0 + 26.0) / 55)
    # end = last VALID minute (i=54), not the trailing NULL minutes
    assert r["spread_bps_end"] == pytest.approx(21.0)
    assert r["imb_200bps_end"] == pytest.approx(0.5)
    # coverage mean over every row inside the hour (55 valid + 3 NULL)
    assert r["coverage_mean"] == pytest.approx((55 * 60.0 + 3 * 30.0) / 58)
    assert r["usable"] == 1


def test_rollup_end_skips_trailing_null_minutes():
    rows = [_minute(H, spread=5.0), _minute(H + 30 * 60, spread=7.0)]
    rows += [_minute(H + i * 60, valid=False, coverage=0.0)
             for i in range(31, 60)]
    r = compute_hour_rollup(rows, H)
    assert r["minutes_present"] == 2
    assert r["spread_bps_end"] == pytest.approx(7.0)
    assert r["usable"] == 0


def test_rollup_usable_thresholds_are_exact():
    def usable_for(n_valid, coverage):
        rows = [_minute(H + i * 60, coverage=coverage)
                for i in range(n_valid)]
        return compute_hour_rollup(rows, H)["usable"]

    assert usable_for(49, 60.0) == 0     # one minute short
    assert usable_for(50, 55.0) == 1     # exactly at both thresholds
    assert usable_for(50, 54.9) == 0     # coverage just under
    assert usable_for(60, 60.0) == 1     # a perfect hour


def test_rollup_all_null_hour_is_unusable_not_missing():
    rows = [_minute(H + i * 60, valid=False, coverage=12.0)
            for i in range(60)]
    r = compute_hour_rollup(rows, H)
    assert r["minutes_present"] == 0
    assert r["spread_bps_mean"] is None
    assert r["spread_bps_end"] is None
    assert r["coverage_mean"] == pytest.approx(12.0)
    assert r["usable"] == 0
