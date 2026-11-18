"""Regression: the candle that closes a 4h bucket must get finite h4
features on the FIRST attempt (live diagnosed 2026-10-21).

Live used to drop the still-forming candle BEFORE building features;
the 4h resample only emits the final completed bucket once a candle
from the next bucket exists, so bucket-closing candles (03/07/11/15/
19/23 UTC) computed NaN h4 features and halted the run until the next
hour. The fix builds features on the full frame and only excludes the
forming candle from processing. These tests pin both halves of the
contract: finite at first attempt, and value-stable vs a longer frame
(the values training saw).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.technical_indicators import FEATURE_COLUMNS, add_features


def _frame(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-01", periods=n, freq="1h")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol}, index=idx)


def test_bucket_closing_candle_finite_with_forming_candle_present():
    df = _frame()
    # a bucket-closing row: hour % 4 == 3, deep enough for warmup
    t_idx = next(i for i in range(900, len(df) - 2)
                 if df.index[i].hour % 4 == 3)
    ts = df.index[t_idx]
    # frame as live now builds it: through the forming candle (t_idx+1)
    feat = add_features(df.iloc[: t_idx + 2])
    row = feat.loc[ts]
    assert np.isfinite(row[FEATURE_COLUMNS]).all(), (
        f"{ts} has NaN features: "
        f"{[c for c in FEATURE_COLUMNS if not np.isfinite(row[c])]}")


def test_bucket_closing_values_stable_vs_longer_frame():
    df = _frame()
    t_idx = next(i for i in range(900, len(df) - 6)
                 if df.index[i].hour % 4 == 3)
    ts = df.index[t_idx]
    short = add_features(df.iloc[: t_idx + 2]).loc[ts, FEATURE_COLUMNS]
    long = add_features(df.iloc[: t_idx + 6]).loc[ts, FEATURE_COLUMNS]
    assert np.allclose(short, long, atol=1e-9)


def test_h4_features_come_from_last_completed_bucket():
    """Alignment: a row's h4_rsi_c must equal the RSI of the last 4h
    bucket that had FULLY COMPLETED by the row's decision time -- not
    the bucket that starts at decision time (the 2026-10-21 look-ahead
    bug: Series + shifted index re-aligned by label, nullifying the
    bucket-end stamp)."""
    from features.technical_indicators import _rsi
    df = _frame()
    feat = add_features(df)
    # a row whose decision time is exactly a bucket end
    t_idx = next(i for i in range(900, len(df) - 2)
                 if df.index[i].hour % 4 == 3)
    ts = df.index[t_idx]
    decision = ts + pd.Timedelta(hours=1)
    h4 = df.resample("4h", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    completed = h4[h4.index + pd.Timedelta(hours=4) <= decision]
    expected = ((_rsi(completed["close"], 14).iloc[-1] - 50.0)
                / 50.0)
    expected = float(np.clip(expected, -1.0, 1.0))
    assert abs(float(feat.loc[ts, "h4_rsi_c"]) - expected) < 1e-9


def test_truncation_causality_all_rows():
    """Strict causality: features computed on a frame truncated at K
    must match the full-frame features for EVERY row whose inputs lie
    inside the truncated frame (rows up to K-2). The buggy alignment
    fails this near bucket boundaries because the mis-stamped bucket
    is partial in the truncated frame and complete in the full one."""
    df = _frame()
    full = add_features(df)
    for k in (950, 1000, 1100):
        trunc = add_features(df.iloc[:k])
        rows = full.index[: k - 1]  # rows 0..k-2 (iloc positions)
        a = full.loc[rows, FEATURE_COLUMNS]
        b = trunc.loc[rows, FEATURE_COLUMNS]
        finite = np.isfinite(a.to_numpy()).all(axis=1)
        assert finite.sum() > 200  # the test must actually cover rows
        assert np.isfinite(b.to_numpy()[finite]).all(), (
            f"truncation at {k}: rows finite in full frame went NaN")
        diff = np.abs(a.to_numpy()[finite] - b.to_numpy()[finite])
        assert diff.max() < 1e-9, (
            f"truncation at {k}: max feature diff {diff.max():.4f} "
            f"(look-ahead contamination)")
