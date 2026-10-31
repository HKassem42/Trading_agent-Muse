"""Tests for src/features/main_chart.py — hand-computed fixtures
(pivots, Donchian, Bollinger, MA distances, Ichimoku warmups,
fractal swings, round numbers), a SAR direction-flip fixture, and
the causality truncation self-check on real data.

Conventions under test: every Distance (%) column =
(Close - Level) / Close * 100, signed; daily-stamped columns use
the PRIOR completed UTC day only; Ichimoku spans use the value
plotted at t (fixed at t-26).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import main_chart as mc  # noqa: E402


def frame(opens, highs, lows, closes, start="2026-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame(
        {"open": np.asarray(opens, float), "high": np.asarray(highs, float),
         "low": np.asarray(lows, float), "close": np.asarray(closes, float),
         "volume": np.ones(len(closes))}, index=idx)


def flat(n, price=100.0, start="2026-01-01", freq="1h"):
    return frame([price] * n, [price + 1] * n, [price - 1] * n,
                 [price] * n, start=start, freq=freq)


# ------------------------------------------------------------------
# moving-average distances
# ------------------------------------------------------------------

def test_sma50_distance_hand_computed():
    closes = np.arange(1, 61, dtype=float)  # 1..60
    df = frame(closes, closes + 1, closes - 1, closes)
    out = mc.build_all(df)
    col = out["Distance to SMA 50 (%)"]
    assert np.isnan(col.iloc[48])
    # at t=49: SMA50 = mean(1..50) = 25.5, close = 50
    assert col.iloc[49] == pytest.approx((50 - 25.5) / 50 * 100)
    # at t=59: SMA50 = mean(11..60) = 35.5, close = 60
    assert col.iloc[59] == pytest.approx((60 - 35.5) / 60 * 100)


def test_ema_warmup_discipline():
    df = flat(60)
    out = mc.build_all(df)
    col = out["Distance to EMA 21 (%)"]
    assert col.iloc[:20].isna().all()
    assert np.isfinite(col.iloc[20])
    # constant price -> EMA == price -> distance 0
    assert col.iloc[20] == pytest.approx(0.0)


# ------------------------------------------------------------------
# channels: Donchian + Bollinger fixtures
# ------------------------------------------------------------------

def test_donchian_hand_computed():
    n = 30
    highs = 100 + np.arange(n, dtype=float)      # rising highs
    lows = 90 + np.arange(n, dtype=float)
    closes = highs - 5.0                          # 5 below the high
    df = frame(closes, highs, lows, closes)
    out = mc.build_all(df)
    up = out["Distance to Donchian Upper (20) (%)"]
    lo = out["Distance to Donchian Lower (20) (%)"]
    t = 25
    hh = highs[t]            # rising series: HH20 = current high
    ll = lows[t - 19]        # LL20 = low 19 bars back
    assert up.iloc[t] == pytest.approx((closes[t] - hh) / closes[t] * 100)
    assert lo.iloc[t] == pytest.approx((closes[t] - ll) / closes[t] * 100)
    assert np.isnan(up.iloc[18]) and np.isfinite(up.iloc[19])


def test_bollinger_hand_computed():
    closes = np.arange(1, 41, dtype=float)
    df = frame(closes, closes + 0.5, closes - 0.5, closes)
    out = mc.build_all(df)
    t = 30
    w = closes[t - 19:t + 1]
    mid = w.mean()
    sd = w.std(ddof=1)
    exp_up = (closes[t] - (mid + 2 * sd)) / closes[t] * 100
    exp_lo = (closes[t] - (mid - 2 * sd)) / closes[t] * 100
    assert out["Distance to Bollinger Upper Band (20, 2.0) (%)"].iloc[t] \
        == pytest.approx(exp_up)
    assert out["Distance to Bollinger Lower Band (20, 2.0) (%)"].iloc[t] \
        == pytest.approx(exp_lo)
    # constant price: bands collapse onto price -> both distances 0
    out2 = mc.build_all(flat(40))
    assert out2["Distance to Bollinger Upper Band (20, 2.0) (%)"].iloc[30] \
        == pytest.approx(0.0)


# ------------------------------------------------------------------
# pivots + prior-day levels (daily stamping)
# ------------------------------------------------------------------

def _two_day_frame():
    # day 1 (2026-01-01): 24 hourly bars, day H = 110, L = 90,
    # last close 100. day 2: 24 bars flat at close 105.
    n1 = 24
    o1 = np.full(n1, 100.0); h1 = np.full(n1, 101.0)
    l1 = np.full(n1, 99.0); c1 = np.full(n1, 100.0)
    h1[5] = 110.0; l1[7] = 90.0
    o2 = np.full(n1, 105.0); h2 = np.full(n1, 106.0)
    l2 = np.full(n1, 104.0); c2 = np.full(n1, 105.0)
    return frame(np.concatenate([o1, o2]), np.concatenate([h1, h2]),
                 np.concatenate([l1, l2]), np.concatenate([c1, c2]),
                 start="2026-01-01 00:00")


def test_pivots_hand_computed():
    df = _two_day_frame()
    out = mc.build_all(df)
    # prior day: H=110, L=90, C=100 -> P = 100, R1 = 110, S1 = 90
    # day-2 close 105 on every stamped row
    row = 30  # a day-2 row
    assert out["Distance to Prior-Day Pivot Point (%)"].iloc[row] \
        == pytest.approx((105 - 100) / 105 * 100)
    assert out["Distance to Prior-Day Pivot R1 (%)"].iloc[row] \
        == pytest.approx((105 - 110) / 105 * 100)
    assert out["Distance to Prior-Day Pivot S1 (%)"].iloc[row] \
        == pytest.approx((105 - 90) / 105 * 100)
    assert out["Distance to Prior-Day High (%)"].iloc[row] \
        == pytest.approx((105 - 110) / 105 * 100)
    assert out["Distance to Prior-Day Low (%)"].iloc[row] \
        == pytest.approx((105 - 90) / 105 * 100)
    # day-1 rows have no prior day -> NaN
    assert np.isnan(out["Distance to Prior-Day Pivot Point (%)"].iloc[10])


# ------------------------------------------------------------------
# Ichimoku: warmup pattern + constant-price collapse + Chikou
# ------------------------------------------------------------------

def test_ichimoku_warmups_and_flat():
    out = mc.build_all(flat(120))
    ten = out["Ichimoku Tenkan-sen Distance (%)"]
    spa = out["Ichimoku Senkou Span A Distance (%)"]
    spb = out["Ichimoku Senkou Span B Distance (%)"]
    assert np.isnan(ten.iloc[7]) and np.isfinite(ten.iloc[8])
    assert np.isnan(spa.iloc[50]) and np.isfinite(spa.iloc[51])
    assert np.isnan(spb.iloc[76]) and np.isfinite(spb.iloc[77])
    # flat price (H=101, L=99): every midpoint level is 100 = close
    assert ten.iloc[60] == pytest.approx(0.0)
    assert spa.iloc[80] == pytest.approx(0.0)
    assert spb.iloc[100] == pytest.approx(0.0)
    assert out["Ichimoku Chikou Distance (%)"].iloc[60] \
        == pytest.approx(0.0)


def test_chikou_arithmetic_series():
    closes = 100 + np.arange(60, dtype=float)
    df = frame(closes, closes + 1, closes - 1, closes)
    out = mc.build_all(df)
    t = 40
    assert out["Ichimoku Chikou Distance (%)"].iloc[t] == \
        pytest.approx((closes[t] - closes[t - 26]) / closes[t] * 100)


# ------------------------------------------------------------------
# Parabolic SAR direction flip on a crafted series
# ------------------------------------------------------------------

def test_psar_side_flip():
    up = np.arange(100, 130, dtype=float)
    dn = np.arange(128, 98, -1, dtype=float)
    closes = np.concatenate([up, dn])
    df = frame(closes, closes + 1, closes - 1, closes)
    out = mc.build_all(df)
    side = out["Parabolic SAR Side (+1/-1)"]
    dist = out["Parabolic SAR Distance (%)"]
    assert (side.iloc[10:30] == 1.0).all()       # uptrend segment
    assert (side.iloc[40:60] == -1.0).all()      # downtrend segment
    # one regime change after the seed bars (pandas_ta seeds PSAR
    # with a 1-bar initial direction, ignored here)
    flips = int((side.iloc[5:].diff().abs() == 2.0).sum())
    assert flips == 1
    # away from seed/flip bars, side and distance sign agree
    seg = slice(10, 30)
    assert (np.sign(dist.iloc[seg]) == side.iloc[seg]).all()


def test_supertrend_side():
    up = np.arange(100, 140, dtype=float)
    dn = np.arange(138, 98, -1, dtype=float)
    closes = np.concatenate([up, dn])
    df = frame(closes, closes + 1, closes - 1, closes)
    out = mc.build_all(df)
    side = out["SuperTrend Side (+1/-1)"]
    assert (side.iloc[20:40] == 1.0).all()
    assert (side.iloc[60:80] == -1.0).all()


# ------------------------------------------------------------------
# fractal swings + round numbers
# ------------------------------------------------------------------

def test_fractal_levels_fixture():
    n = 40
    closes = np.full(n, 100.0)
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)
    # unique swing high 120 at bar 10 (confirmed at bar 12)
    highs[8:13] = [105, 112, 120, 112, 105]
    # unique swing low 80 at bar 24 (confirmed at bar 26)
    lows[22:27] = [95, 88, 80, 88, 95]
    df = frame(closes, highs, lows, closes)
    res, sup = mc.fractal_levels(df)
    assert np.isnan(res.iloc[11])                # not yet confirmed
    assert res.iloc[12] == pytest.approx(120.0)  # confirmed at 12
    assert sup.iloc[25] != sup.iloc[25] or True   # (NaN check below)
    assert np.isnan(sup.iloc[25])
    assert sup.iloc[26] == pytest.approx(80.0)
    out = mc.build_all(df)
    assert out["Distance to Fractal Resistance (%)"].iloc[20] \
        == pytest.approx((100 - 120) / 100 * 100)
    assert out["Distance to Fractal Support (%)"].iloc[30] \
        == pytest.approx((100 - 80) / 100 * 100)


def test_round_number_distance():
    closes = np.array([102300.0, 103000.0, 97500.0])
    df = frame(closes, closes, closes, closes)
    out = mc._round_number_dist(df)
    assert out.iloc[0] == pytest.approx((102300 - 100000) / 102300 * 100)
    assert out.iloc[1] == pytest.approx((103000 - 105000) / 103000 * 100)
    assert out.iloc[2] == pytest.approx((97500 - 100000) / 97500 * 100)


# ------------------------------------------------------------------
# causality: truncation self-check on real data (sup frame sample)
# ------------------------------------------------------------------

def test_causality_truncation_real_frame():
    sup = pd.read_pickle(ROOT / "data" / "diagnosis" / "sup.pkl")
    df = sup[["open", "high", "low", "close", "volume"]].iloc[:2600]
    full = mc.build_all(df)
    trunc = mc.build_all(df.iloc[:1800])
    cols = ["Distance to SMA 200 (%)",
            "Distance to Donchian Upper (55) (%)",
            "Ichimoku Senkou Span A Distance (%)",
            "Ichimoku Senkou Span B Distance (%)",
            "Parabolic SAR Distance (%)",
            "SuperTrend Distance (%)",
            "Distance to Prior-Day Pivot Point (%)",
            "Distance to Fractal Resistance (%)",
            "Distance to Fractal Support (%)",
            "Distance to $5,000 Round Number (%)"]
    for colname in cols:
        a = full[colname].iloc[:1800].to_numpy()
        b = trunc[colname].to_numpy()
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), \
            f"causality mismatch in {colname}"


def test_registry_shape():
    names = [n for n, _b, _d in mc.INDICATORS]
    assert len(names) == len(set(names)) == 31
    out = mc.build_all(flat(300))
    assert list(out.columns) == names
