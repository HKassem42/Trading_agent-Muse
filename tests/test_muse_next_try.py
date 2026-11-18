"""Adversarial tests for Muse's Next Try features (LAB only).

Written BEFORE the implementation (Kass's quality standard,
2026-10-21): the alignment tests use hand-computed references on
synthetic series, and the causality test rebuilds the raw external
series on truncated inputs and requires every shared row to match
exactly. The funding/metrics rules being pinned:

- a feature at row t (hour-start index) may use only external
  observations timestamped < t + 1h (through candle t's close --
  the same footing as the spot features, which read candle t);
- funding level at t = last settlement with calc_time <= t;
- funding 24h mean at t = mean of settlements in (t - 24h, t];
- OI at t = last metrics observation < t + 1h; OI changes compare
  that observation to the last observation < t + 1h - 1h / - 24h;
- ratio features at t = last metrics observation < t + 1h, except
  the taker L/S ratio, which is the MEAN of the observations
  inside candle t ([t, t + 1h)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import next_try_features as ntf  # noqa: E402


def _index(n=72, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="1h")


# ------------------------------------------------------------ funding
def test_funding_alignment_hand_computed():
    idx = _index(72)
    # Settlements every 8h starting 2023-12-31 16:00 at 0.001,
    # increasing by 0.001 each settlement.
    times = pd.date_range("2023-12-31 16:00", periods=12, freq="8h")
    rates = np.arange(1, 13) * 0.001
    level, mean24 = ntf.funding_series(idx, times, rates)
    # Row 0 is 2024-01-01 00:00: last settlement <= t is the
    # 2023-12-31 16:00 one (0.001)... wait, settlements at 16:00 then
    # 00:00 -- the 12 settlements from 16:00 step 8h are 16:00, 00:00,
    # 08:00, ... so at row 0 (00:00) a settlement lands exactly at t
    # and counts (<= t): rate index 1 -> 0.002.
    assert level[0] == pytest.approx(0.002)
    # Row 7 (07:00): still the 00:00 settlement.
    assert level[7] == pytest.approx(0.002)
    # Row 8 (08:00): the 08:00 settlement counts.
    assert level[8] == pytest.approx(0.003)
    # mean24 at row 0: settlements in (2023-12-31 00:00, 2024-01-01
    # 00:00] = 08:00 (none -- series starts 16:00), 16:00 (0.001),
    # 00:00 (0.002) -> mean 0.0015.
    assert mean24[0] == pytest.approx(0.0015)
    # mean24 at row 24 (2024-01-02 00:00): settlements in the last
    # 24h are 08:00 (0.003), 16:00 (0.004), 00:00 (0.005) -> 0.004.
    assert mean24[24] == pytest.approx(0.004)


def test_funding_future_settlement_never_leaks():
    idx = _index(24)
    times = pd.date_range("2024-01-01 08:00", periods=2, freq="8h")
    rates = np.array([0.005, 0.009])
    level, _ = ntf.funding_series(idx, times, rates)
    # Before the first settlement (rows 0..7) there is NO settled
    # funding: NaN, never the future 0.005.
    assert np.isnan(level[:8]).all()
    assert level[8] == pytest.approx(0.005)


# ------------------------------------------------------------ metrics
def _metrics(idx):
    """5-minute metrics over the index span: OI grows linearly
    from 100 by +1 per 5 minutes; ratios are simple ramps."""
    times = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=1),
                          freq="5min", inclusive="left")
    k = np.arange(len(times), dtype=float)
    return pd.DataFrame({
        "create_time": times,
        "sum_open_interest": 100.0 + k,
        "count_toptrader_long_short_ratio": 1.0 + k * 0.001,
        "sum_toptrader_long_short_ratio": 2.0 + k * 0.001,
        "count_long_short_ratio": 3.0 + k * 0.001,
        "sum_taker_long_short_vol_ratio": 0.5 + (k % 12) * 0.1,
    })


def test_metrics_alignment_hand_computed():
    idx = _index(72)
    m = _metrics(idx)
    raw = ntf.metrics_series(idx, m)
    # OI at row t: last obs < t+1h is the obs at t:55 -> its k is
    # (t_hours * 12) + 11. Row 30 -> k = 371 -> OI = 471.
    assert raw["oi_level_raw"][30] == pytest.approx(471.0)
    # OI change 1h at row 30: obs at 30:55 (k=371, 471) vs last obs
    # < 30:00 (k=359, 459): 471/459 - 1.
    assert raw["oi_chg1_raw"][30] == pytest.approx(471.0 / 459.0 - 1.0)
    # OI change 24h at row 30: vs last obs < 07:00 next... t+1h-24h
    # = row 30 - 23h -> obs k = 371 - 288 = 83 -> 183.
    assert raw["oi_chg24_raw"][30] == pytest.approx(471.0 / 183.0 - 1.0)
    # Ratio at row 30 = value at obs 30:55: 3 + 371*0.001.
    assert raw["ls_all_raw"][30] == pytest.approx(3.371)
    assert raw["ls_top_pos_raw"][30] == pytest.approx(2.371)
    # Taker ratio at row 30 = mean of the 12 obs inside hour 30:
    # (k % 12) cycles 0..11 -> values 0.5..1.6 step 0.1 -> mean 1.05.
    assert raw["taker_ls_raw"][30] == pytest.approx(1.05)


# ------------------------------------------------------------ perp / eth
def test_perp_and_eth_hand_computed():
    idx = _index(72)
    spot_close = pd.Series(100.0 + np.arange(72), index=idx)
    spot_vol = pd.Series(10.0, index=idx)
    perp = pd.DataFrame({
        "close": spot_close * 1.001,
        "volume": 30.0,
        "taker_buy_volume": 18.0,
    }, index=idx)
    raw = ntf.perp_series(idx, perp, spot_close, spot_vol)
    assert np.allclose(raw["basis_raw"], 0.001)
    assert np.allclose(raw["taker_buy_share_raw"], 0.6)
    assert np.allclose(raw["perp_vol_ratio_raw"], 3.0)

    eth_close = pd.Series(50.0 * (1.01 ** np.arange(72)), index=idx)
    raw_e = ntf.eth_series(idx, eth_close, spot_close)
    # ETH return at row t (t>=1) is exactly +1%.
    assert raw_e["eth_ret_raw"][1] == pytest.approx(0.01)
    assert np.isnan(raw_e["eth_ret_raw"][0])


# ------------------------------------------------------------ calendar
def test_calendar_hand_computed():
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="1h")
    cal = ntf.calendar_series(idx)  # values on the 0..100 scale
    # 2024-01-01 is a Monday (weekday 0). Hour 0: sin(0)=0 -> 50,
    # cos(0)=1 -> 100.
    assert cal["hour_sin"].iloc[0] == pytest.approx(50.0)
    assert cal["hour_cos"].iloc[0] == pytest.approx(100.0)
    # Hour 6: sin(pi/2)=1 -> 100.
    assert cal["hour_sin"].iloc[6] == pytest.approx(100.0)
    assert cal["dow_sin"].iloc[0] == pytest.approx(50.0)
    assert cal["dow_cos"].iloc[0] == pytest.approx(100.0)
    assert cal.min().min() >= 0.0 and cal.max().max() <= 100.0


# ------------------------------------------------------------ causality
def test_raw_external_truncation_causality_all_rows():
    """Build the raw external series on the full history, then on
    inputs truncated at K (external observations limited to
    < idx[K] + 1h, exactly what a builder at time K could see) and
    require EVERY shared row to match exactly."""
    idx = _index(200, start="2024-03-01 00:00")
    rng = np.random.default_rng(7)
    spot_close = pd.Series(100.0 * np.exp(
        np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    spot_vol = pd.Series(rng.uniform(5, 15, len(idx)), index=idx)
    ftimes = pd.date_range(idx[0] - pd.Timedelta(hours=8),
                           idx[-1], freq="8h")
    frates = rng.normal(0.0001, 0.00005, len(ftimes))
    m = _metrics(idx)
    m["sum_open_interest"] = 1000.0 * np.exp(
        np.cumsum(rng.normal(0, 0.0005, len(m))))
    perp = pd.DataFrame({
        "close": spot_close.to_numpy() * (1 + rng.normal(0, 0.0005, len(idx))),
        "volume": rng.uniform(20, 40, len(idx)),
        "taker_buy_volume": rng.uniform(8, 25, len(idx)),
    }, index=idx)
    eth_close = pd.Series(50.0 * np.exp(
        np.cumsum(rng.normal(0, 0.0012, len(idx)))), index=idx)

    full = ntf.raw_external_series(idx, (ftimes, frates), m, perp,
                                   spot_close, spot_vol, eth_close)
    for K in (60, 120, 199):
        end = idx[K] + pd.Timedelta(hours=1)
        m_t = m[m["create_time"] < end]
        perp_t = perp[perp.index < end]
        eth_t = eth_close[eth_close.index < end]
        trunc = ntf.raw_external_series(
            idx[:K + 1], (ftimes, frates), m_t, perp_t,
            spot_close.iloc[:K + 1], spot_vol.iloc[:K + 1], eth_t)
        for col in full.columns:
            a = full[col].to_numpy()[:K + 1]
            b = trunc[col].to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            assert both.sum() > 0, (col, K)
            assert np.array_equal(np.isnan(a), np.isnan(b)), (col, K)
            assert np.max(np.abs(a[both] - b[both])) <= 1e-12, (col, K)


def test_new_columns_contract():
    assert len(ntf.NEW_COLUMNS) == 22
    assert len(ntf.INFO_RAW) == 14
    assert len(set(ntf.NEW_COLUMNS)) == 22


def test_family_code_registered_for_dashboard_feed():
    # Run 101 registered family "muse_next_try" on 2026-11-04; the
    # training feed generator crashed with KeyError until the family
    # was wired into compact.py (caught by the live loop's watch).
    from src.dashboards import compact
    assert compact._FAMILY_CODES["muse_next_try"] == 22


def test_pick_lineage_display_renames():
    # Kass's order 2026-11-04: runs 101/102/103 display as Muse's
    # Pick #4/#5/#6 (continuing Pick #1 run 69, #2 run 73, #3 run 75).
    # Display-only: families and verdicts must stay exactly as
    # registered, and unrenamed runs keep their ordinary names.
    from src.dashboards import compact
    r101 = {"id": 101, "kind": "stacker",
            "metrics": {"family": "muse_next_try",
                        "muse_next_try": {"label": "Muse's Next Try"}}}
    r102 = {"id": 102, "kind": "stacker",
            "metrics": {"family": "muse_next_try2",
                        "muse_next_try2": {"label": "Muse's Next Try #2"}}}
    r103 = {"id": 103, "kind": "ppo", "metrics": {}}
    r104 = {"id": 104, "kind": "ppo", "metrics": {}}
    assert compact._display_name(r101, None) == "Muse's Pick #4"
    assert compact._display_name(r102, None) == "Muse's Pick #5"
    assert compact._display_name(r103, "5.1") == "Muse's Pick #6"
    assert compact._display_name(r104, "5.1") == "PPO v5.1"
