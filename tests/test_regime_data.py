"""Regime-data (Option B) tests: the availability convention is the
study's central causality risk, so it is pinned FIRST, on synthetic
ordinal series where the correct answer is known by construction,
plus the filter/verdict mechanics and a real-data census.

Convention (refs/regime-data-plan-2026-10-24.md): a daily value
stamped day D is usable only from D+lag 00:00 UTC (lag 1 for Fear &
Greed, 2 for FRED); usable(d) = freshest stamp s with s + lag <= d;
an hourly row takes the usable value of its UTC date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import regime_data as rd  # noqa: E402

EPOCH = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2025-03-31")


def _ordinal_series(dates, skip=()) -> pd.Series:
    """Synthetic daily series whose value IS its stamp's day ordinal
    -- any look-ahead shows up as a too-large value, exactly."""
    idx = pd.DatetimeIndex([d for d in dates if d not in skip])
    return pd.Series([(d - EPOCH).days for d in idx], index=idx,
                     dtype=float)


def _daily_with(raw: pd.Series, test_key: str, test_lag: int,
                end: pd.Timestamp = END) -> dict:
    """A full daily-features dict where `test_key`'s frame is built
    from the synthetic raw series at `test_lag` and every other
    series reuses the same raw values at its own registered lag
    (their values are irrelevant to the assertions)."""
    daily = {}
    for key, (_f, klag) in rd.SERIES.items():
        daily[key] = rd.daily_feature_frame(
            raw, test_lag if key == test_key else klag, end)
    return daily


def _hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h")


# ------------------------------------------------------- stamp exactness
def test_fng_stamp_exact_and_midnight_boundary():
    dates = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    raw = _ordinal_series(dates)
    daily = _daily_with(raw, "fng", 1)
    hours = _hours("2025-01-10 00:00", "2025-01-12 23:00")
    feats = rd.hourly_regime_features(hours, daily)["fng_level"]
    for t in hours:
        expected = ((t.normalize() - pd.Timedelta(days=1))
                    - EPOCH).days
        assert feats.loc[t] == expected, f"{t}: {feats.loc[t]}"
    # The boundary itself: 23:00 still reads stamp Jan 9 (ordinal for
    # Jan 9), 00:00 flips to stamp Jan 10 -- never earlier.
    assert feats.loc[pd.Timestamp("2025-01-10 23:00")] == \
        (pd.Timestamp("2025-01-09") - EPOCH).days
    assert feats.loc[pd.Timestamp("2025-01-11 00:00")] == \
        (pd.Timestamp("2025-01-10") - EPOCH).days


def test_fred_stamp_exact_lag2_with_gaps():
    dates = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    skip = {d for d in dates if d.weekday() >= 5} \
        | {pd.Timestamp("2025-01-20")}  # weekend + a holiday hole
    raw = _ordinal_series(dates, skip=skip)
    daily = _daily_with(raw, "vix", 2)
    hours = _hours("2025-01-08 00:00", "2025-01-25 23:00")
    feats = rd.hourly_regime_features(hours, daily)["vix_level"]
    stamps, vals = raw.index, raw.to_numpy()
    for t in hours:
        d = t.normalize()
        ok = [i for i, s in enumerate(stamps)
              if s + pd.Timedelta(days=2) <= d]
        assert ok, f"no usable stamp for {t}"
        assert feats.loc[t] == vals[ok[-1]], f"{t}"


# ------------------------------------------------- truncation invariance
@pytest.mark.parametrize("key,lag", [("fng", 1), ("vix", 2)])
def test_truncation_invariance(key, lag):
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    skip = {d for d in dates if d.weekday() >= 5}
    raw = _ordinal_series(dates, skip=skip)
    t0 = pd.Timestamp("2025-02-14")
    hours = _hours("2025-01-10 00:00", "2025-03-20 23:00")
    full = rd.hourly_regime_features(hours, _daily_with(raw, key, lag))
    trunc = rd.hourly_regime_features(
        hours, _daily_with(raw[raw.index <= t0], key, lag))
    prefix = hours[hours.normalize() <= t0 + pd.Timedelta(days=lag)]
    assert len(prefix) > 500
    for sub in ("level", "chg5"):
        a = full.loc[prefix, f"{key}_{sub}"].to_numpy()
        b = trunc.loc[prefix, f"{key}_{sub}"].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), sub


# --------------------------------------------- extra lag only stales
def test_extra_lag_only_stales():
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    raw = _ordinal_series(dates)
    hours = _hours("2025-01-10 00:00", "2025-03-20 23:00")
    base = rd.hourly_regime_features(hours, _daily_with(raw, "fng", 1))
    plus = rd.hourly_regime_features(hours, _daily_with(raw, "fng", 2))
    for sub in ("level", "chg5"):
        a = plus[f"fng_{sub}"].to_numpy()[24:]
        b = base[f"fng_{sub}"].to_numpy()[:-24]
        assert np.array_equal(a, b, equal_nan=True), sub


def test_chg5_is_five_calendar_days():
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    raw = _ordinal_series(dates)
    frame = rd.daily_feature_frame(raw, 1, END)
    chg = frame["chg5"].dropna()
    assert len(chg) > 80
    assert (chg == 5.0).all()


# ------------------------------------------------------------ the filter
def test_filter_veto_boundaries():
    conv = np.array([0.5, 0.5, -0.6, -0.6, 0.0, -0.9, 0.9])
    fng = np.array([24.9, 25.0, 75.0, 75.1, 5.0, 90.0, 3.0])
    out, n = rd.apply_fng_filter(conv, fng)
    # long vetoed < 25 (idx 0, 6); short vetoed > 75 (idx 3, 5);
    # boundaries inclusive-allowed: 25.0 long passes, 75.0 short
    # passes; a zero conviction is never a veto.
    assert out.tolist() == [0.0, 0.5, -0.6, 0.0, 0.0, 0.0, 0.0]
    assert n == 4
    # F&G pinned mid-range: the filter is an exact identity.
    same, n0 = rd.apply_fng_filter(conv, np.full(7, 50.0))
    assert n0 == 0 and np.array_equal(same, conv)


# -------------------------------------------------------------- verdicts
def _wf(acc=0.70, bal=0.40, base=0.669, folds=(0.40,) * 5):
    return {"pooled": {"accuracy": acc, "balanced_accuracy": bal,
                       "majority_baseline": base},
            "folds": [{"balanced_accuracy": b} for b in folds]}


def _duel(pnl=1.0, bench=0.5, non_hold=0.5):
    return {"total_pnl": pnl, "bench_pnl": bench,
            "non_hold_share": non_hold}


def test_verdict_rules_mechanics():
    rob_ok, rob_bad = {"total_pnl": 0.0}, {"total_pnl": -0.01}
    v = rd.apply_verdict_arm1(_wf(), _duel(), rob_ok)
    assert v["verdict"] == "PASS"
    # Rule 1 legs: accuracy below baseline fails even with BA fine;
    # only 3/5 folds above chance fails; pooled BA at chance fails.
    assert not rd.rule1_walk_forward(_wf(acc=0.60))
    assert not rd.rule1_walk_forward(
        _wf(folds=(0.40, 0.40, 0.40, 0.30, 0.30)))
    assert not rd.rule1_walk_forward(_wf(bal=1.0 / 3.0))
    assert rd.rule1_walk_forward(_wf(acc=0.669))  # == baseline passes
    # Rule 2 legs: strict > on both money legs; activity floor incl.
    assert not rd.rule2_duel(_duel(pnl=0.0))
    assert not rd.rule2_duel(_duel(pnl=0.5, bench=0.5))
    assert not rd.rule2_duel(_duel(non_hold=0.099))
    assert rd.rule2_duel(_duel(non_hold=0.10))
    # Rule 3: zero is a pass, a cent under is not.
    assert rd.rule3_robustness(rob_ok)
    assert not rd.rule3_robustness(rob_bad)
    # Arm 3 needs rules 2+3 only.
    assert rd.apply_verdict_arm3(_duel(), rob_ok)["verdict"] == "PASS"
    assert rd.apply_verdict_arm3(_duel(), rob_bad)["verdict"] == "FAIL"
    assert rd.apply_verdict_arm1(_wf(), _duel(), rob_bad)["verdict"] \
        == "FAIL"


# --------------------------------------------------------- real-data census
def test_real_regime_census():
    from models import gbm_focus as gf
    matrix = gf.load_matrix(
        "data/timeseries_matrix/btc_usd_1h_matrix.csv")
    frame = rd.make_feature_frame(matrix, rd.build_daily_features())
    assert len(frame) == 16_735
    sub = frame[rd.REGIME_FEATURES]
    assert sub.notna().all().all()
    assert frame["fng_level"].between(0, 100).all()
    # The last matrix day can only read F&G stamped <= Oct 20 (D+1)
    # and FRED stamped <= Oct 19 (D+2): spot-check staleness bounds
    # via the raw series' own tails.
    raw_fng = rd.load_daily_raw("fng")
    last = frame.index[-1]
    assert frame.loc[last, "fng_level"] == \
        raw_fng.loc[pd.Timestamp("2026-10-20")]
