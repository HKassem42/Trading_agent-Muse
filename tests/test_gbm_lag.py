"""Tests for the GBM lag-columns experiment (2026-10-22).

Written BEFORE the build, per Kass's quality standard:

(1) Lag construction is exactly what it claims: hand-computed
    lag/delta/return values on a tiny fixture, and FUTURE-INVARIANCE --
    appending wild future rows must not change any earlier row's
    features (the look-ahead bug class that bit this project in v5.1).
(2) Feature-set assembly is exactly the pre-registered spec: 72
    columns for the focus arms (11 x [now, lag1, lag2, lag3, d1, d3]
    + 6 return lags), 186 for the wide arm (60 x [now, lag1, d1] + 6).
(3) SMOTE discipline mirrors test_gbm_focus.py: arm F's treatment is
    SMOTE-inside-the-fold, proven by the fold audit timestamps and by
    a poison check on validation rows.

Run: .venv/bin/python -m pytest tests/test_gbm_lag.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.gbm_focus import FOCUS_FEATURES, fit_arm_fold  # noqa: E402
from models.gbm_lag import (  # noqa: E402
    ARM_TREATMENT,
    RET_FEATURES,
    build_lag_frame,
    delta_name,
    drop_warmup,
    expected_focus_columns,
    focus_lag_frame,
    lag_name,
    wide_lag_frame,
)


# ------------------------------------------------------------ (1) math
def _fixture():
    idx = pd.date_range("2025-01-01", periods=6, freq="1h")
    df = pd.DataFrame({
        "Close": [100.0, 102.0, 101.0, 105.0, 103.0, 108.0],
        "Force Index": [10.0, 20.0, 15.0, 30.0, 25.0, 40.0],
    }, index=idx)
    return df


def test_lag_delta_return_hand_computed():
    df = _fixture()
    fr = build_lag_frame(df, ["Force Index"], lags=(1, 2, 3),
                         deltas=(1, 3), ret_lags=(1, 2))
    fi = "Force Index"
    row = fr.iloc[3]  # t = 3: FI history 10, 20, 15, 30
    assert row[fi] == 30.0
    assert row[lag_name(fi, 1)] == 15.0    # t-1
    assert row[lag_name(fi, 2)] == 20.0    # t-2
    assert row[lag_name(fi, 3)] == 10.0    # t-3
    assert row[delta_name(fi, 1)] == 30.0 - 15.0
    assert row[delta_name(fi, 3)] == 30.0 - 10.0
    # Simple pct returns from Close.
    assert fr["ret_1h"].iloc[1] == pytest.approx(102.0 / 100.0 - 1.0)
    assert fr["ret_2h"].iloc[3] == pytest.approx(105.0 / 102.0 - 1.0)
    assert fr["ret_1h"].iloc[5] == pytest.approx(108.0 / 103.0 - 1.0)
    # Warmup rows are NaN exactly where history is missing.
    assert np.isnan(fr[lag_name(fi, 3)].iloc[2])
    assert np.isnan(fr["ret_2h"].iloc[1])
    assert not np.isnan(fr[lag_name(fi, 3)].iloc[3])


def test_features_invariant_to_future_rows():
    df = _fixture()
    base = build_lag_frame(df, ["Force Index"], lags=(1, 2, 3),
                           deltas=(1, 3), ret_lags=(1, 2))
    wild = pd.DataFrame({
        "Close": [1e9, -5.0, 0.0],
        "Force Index": [1e12, -1e12, 42.0],
    }, index=pd.date_range("2025-01-01 06:00", periods=3, freq="1h"))
    extended = build_lag_frame(pd.concat([df, wild]),
                               ["Force Index"], lags=(1, 2, 3),
                               deltas=(1, 3), ret_lags=(1, 2))
    pd.testing.assert_frame_equal(base, extended.iloc[:len(df)])


def test_future_invariance_on_focus_frame():
    rng = np.random.default_rng(5)
    n = 60
    idx = pd.date_range("2025-03-01", periods=n, freq="1h")
    data = {c: rng.normal(0, 1, n).cumsum() for c in FOCUS_FEATURES}
    data["Close"] = 100.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(data, index=idx)
    base = focus_lag_frame(df)
    wild = pd.DataFrame(
        {c: [1e9] * 5 for c in FOCUS_FEATURES} | {"Close": [1e9] * 5},
        index=pd.date_range(idx[-1] + pd.Timedelta(hours=1),
                             periods=5, freq="1h"))
    extended = focus_lag_frame(pd.concat([df, wild]))
    pd.testing.assert_frame_equal(base, extended.iloc[:n])


# ------------------------------------------------------- (2) assembly
def test_focus_assembly_exactly_72_specified_columns():
    rng = np.random.default_rng(0)
    n = 40
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    data = {c: rng.normal(size=n) for c in FOCUS_FEATURES}
    data["Close"] = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(data, index=idx)
    fr = focus_lag_frame(df)
    assert list(fr.columns) == expected_focus_columns()
    assert fr.shape[1] == 72
    # 11 current + 33 lags + 22 deltas + 6 return lags.
    kinds = {"now": 0, "lag": 0, "delta": 0, "ret": 0}
    for c in fr.columns:
        if c in RET_FEATURES:
            kinds["ret"] += 1
        elif "__lag" in c:
            kinds["lag"] += 1
        elif "__d" in c:
            kinds["delta"] += 1
        else:
            kinds["now"] += 1
    assert kinds == {"now": 11, "lag": 33, "delta": 22, "ret": 6}


def test_wide_assembly_exactly_186_columns():
    rng = np.random.default_rng(1)
    n = 40
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    base = [f"feat{i}" for i in range(60)]
    data = {c: rng.normal(size=n) for c in base}
    data["Close"] = np.linspace(50.0, 60.0, n)
    df = pd.DataFrame(data, index=idx)
    fr = build_lag_frame(df, base, lags=(1,), deltas=(1,))
    assert fr.shape[1] == 60 * 3 + 6 == 186
    for c in base:
        assert c in fr.columns
        assert lag_name(c, 1) in fr.columns
        assert delta_name(c, 1) in fr.columns
    assert list(fr.columns[-6:]) == RET_FEATURES


def test_warmup_drop_count_is_max_return_lag():
    rng = np.random.default_rng(2)
    n = 40
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    data = {c: rng.normal(size=n) for c in FOCUS_FEATURES}
    data["Close"] = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame(data, index=idx)
    kept = drop_warmup(focus_lag_frame(df))
    assert len(kept) == n - 24  # ret_24h is the longest memory
    assert kept.index[0] == idx[24]
    assert kept.notna().all().all()


# ------------------------------------------------- (3) SMOTE discipline
def _toy_lag_matrix(n: int = 900, seed: int = 7):
    """Matrix-shaped toy: 11 focus columns + Close, 3-class labels
    with a HOLD majority and a faint learnable nudge."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    data = {c: rng.normal(0, 1, n).cumsum() for c in FOCUS_FEATURES}
    data["Close"] = 100.0 + rng.normal(0, 0.5, n).cumsum()
    y = np.array(["HOLD"] * n, dtype=object)
    y[::7] = "BUY"
    y[3::11] = "SHORT"
    data["Force Index"] = data["Force Index"] + np.where(
        y == "BUY", 2.0, np.where(y == "SHORT", -2.0, 0.0))
    df = pd.DataFrame(data, index=idx)
    frame = drop_warmup(focus_lag_frame(df))
    keep = df.index.isin(frame.index)
    return frame, y[keep], frame.index


def test_arm_treatments_match_brief():
    assert ARM_TREATMENT == {"E": "B", "F": "C", "G": "B"}


def test_smote_inside_fold_on_lag_features():
    from sklearn.model_selection import TimeSeriesSplit

    frame, y, ts = _toy_lag_matrix()
    X = frame.to_numpy(dtype=float)
    assert X.shape[1] == 72
    tss = TimeSeriesSplit(n_splits=5)
    tr_idx, va_idx = list(tss.split(X))[2]
    assert tr_idx.max() < va_idx.min()

    model, audit = fit_arm_fold(X, y, ts, tr_idx, va_idx,
                                arm=ARM_TREATMENT["F"])
    assert audit["fed_ts_max"] < ts[va_idx].min()
    assert audit["n_fed_original"] == len(tr_idx)
    counts = audit["resampled_class_counts"]
    assert len(set(counts.values())) == 1, counts  # SMOTE balanced
    assert audit["n_after_resample"] > len(tr_idx)
    proba = model.predict_proba(X[va_idx])
    assert proba.shape == (len(va_idx), 3)


def test_smote_poison_check_on_lag_features():
    from sklearn.model_selection import TimeSeriesSplit

    frame, y, ts = _toy_lag_matrix()
    X = frame.to_numpy(dtype=float)
    tr_idx, va_idx = list(TimeSeriesSplit(n_splits=5).split(X))[2]
    X_poisoned = X.copy()
    X_poisoned[va_idx] = 1e15
    _, audit = fit_arm_fold(X_poisoned, y, ts, tr_idx, va_idx,
                            arm=ARM_TREATMENT["F"])
    assert audit["fit_max_abs"] < 1e14
