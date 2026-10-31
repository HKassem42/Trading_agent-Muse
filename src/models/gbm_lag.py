"""GBM lag-columns experiment core (2026-10-22, Kass green-lit).

Follow-up to the GBM focus-list experiment (models/gbm_focus.py).
Kass's hypothesis: every model so far saw a single-row snapshot; add
TRAJECTORY -- for each focus feature, its values 1/2/3 hours ago and
its changes over 1h and 3h -- plus raw close-return lags, and see if
the gradient booster does better.

Construction rules (all PAST-only by construction):
  lag{k}   = value at t-k            (DataFrame.shift(+k))
  delta{k} = value at t - value at t-k
  ret_{k}h = Close[t] / Close[t-k] - 1   (simple pct return)

Because every derived column is a positive shift or a difference of
positive shifts of the source columns, row t's features depend only
on rows <= t. tests/test_gbm_lag.py pins this two ways: hand-computed
values on a tiny fixture, and invariance of shared rows when future
rows are appended.

Arms (LightGBM via models.gbm_focus fit helpers; same fixed
hyperparameters and seeds as the focus build):
  E  lag-augmented focus set (72 cols), balanced class weights
  F  lag-augmented focus set (72 cols), SMOTE inside training folds
  G  lag-augmented ALL-60 (lag1 + delta1 only per base feature, to
     bound width: 60 x 3 + 6 return lags = 186 cols), class weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.gbm_focus import FOCUS_FEATURES, matrix_feature_columns

# ---------------------------------------------------------------- spec
FOCUS_LAGS = (1, 2, 3)
FOCUS_DELTAS = (1, 3)
WIDE_LAGS = (1,)
WIDE_DELTAS = (1,)
RET_LAGS = (1, 2, 3, 6, 12, 24)

RET_FEATURES = [f"ret_{k}h" for k in RET_LAGS]

# Lag-experiment arm -> fit treatment in models.gbm_focus._fit_core
# ("B" = balanced class weights, "C" = SMOTE inside the fold).
ARM_TREATMENT = {"E": "B", "F": "C", "G": "B"}
ARMS = ("E", "F", "G")

ARM_DESIGNS = {
    "E": "lag focus-72 (11 base x [t, lag1..3, d1, d3] + 6 return "
         "lags), balanced class weights",
    "F": "lag focus-72, SMOTE inside each training fold (scaler->SMOTE)",
    "G": "lag all-60 (60 base x [t, lag1, d1] + 6 return lags = 186), "
         "balanced class weights",
}


def lag_name(base: str, k: int) -> str:
    return f"{base}__lag{k}"


def delta_name(base: str, k: int) -> str:
    return f"{base}__d{k}"


def feature_kind(name: str) -> str:
    """Classify an assembled feature name for importance grouping:
    'ret' (return lag), 'lag', 'delta', or 'now' (current snapshot)."""
    if name in RET_FEATURES:
        return "ret"
    if "__lag" in name:
        return "lag"
    if "__d" in name:
        return "delta"
    return "now"


def build_lag_frame(df: pd.DataFrame, base_features: list[str],
                    lags: tuple[int, ...], deltas: tuple[int, ...],
                    ret_lags: tuple[int, ...] = RET_LAGS
                    ) -> pd.DataFrame:
    """Assemble the lag-augmented feature frame for `base_features`.

    Column order: for each base feature -- current value (under its
    own name), then lag{k} for k in `lags`, then delta{k} for k in
    `deltas`; return lags from Close come last. Everything is a
    positive shift / difference of positive shifts -> past-only.
    """
    out = {}
    for base in base_features:
        s = df[base]
        out[base] = s
        for k in lags:
            out[lag_name(base, k)] = s.shift(k)
        for k in deltas:
            out[delta_name(base, k)] = s - s.shift(k)
    close = df["Close"]
    for k in ret_lags:
        out[f"ret_{k}h"] = close / close.shift(k) - 1.0
    return pd.DataFrame(out, index=df.index)


def focus_lag_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Arm E/F frame: 11 focus features x 6 + 6 return lags = 72."""
    return build_lag_frame(df, list(FOCUS_FEATURES),
                           FOCUS_LAGS, FOCUS_DELTAS)


def wide_lag_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Arm G frame: all 60 matrix features x 3 + 6 return lags = 186."""
    base = matrix_feature_columns(list(df.columns))
    return build_lag_frame(df, base, WIDE_LAGS, WIDE_DELTAS)


def arm_frame(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm in ("E", "F"):
        return focus_lag_frame(df)
    if arm == "G":
        return wide_lag_frame(df)
    raise ValueError(f"unknown arm {arm!r}")


def expected_focus_columns() -> list[str]:
    cols: list[str] = []
    for base in FOCUS_FEATURES:
        cols.append(base)
        cols += [lag_name(base, k) for k in FOCUS_LAGS]
        cols += [delta_name(base, k) for k in FOCUS_DELTAS]
    return cols + list(RET_FEATURES)


def drop_warmup(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any NaN feature (the first max(ret_lags) hours)."""
    return frame.dropna(axis=0, how="any")


def truncation_check(df: pd.DataFrame, n_trunc: int,
                     sample_cols: list[str]) -> dict:
    """Causality self-check: rebuild the focus lag frame on the first
    `n_trunc` rows only and assert the shared rows match the full-frame
    build EXACTLY on the sampled lag/delta/return columns. Any
    look-ahead in construction would break equality."""
    full = focus_lag_frame(df)
    trunc = focus_lag_frame(df.iloc[:n_trunc])
    shared = trunc.index
    ok = True
    worst = 0.0
    for col in sample_cols:
        a = trunc[col].to_numpy(dtype=float)
        b = full.loc[shared, col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        diff = np.where(np.isnan(diff), np.inf, diff)
        worst = max(worst, float(diff.max()))
        ok = ok and bool((diff == 0.0).all())
    return {"pass": ok, "max_abs_diff": worst, "n_trunc": int(n_trunc),
            "sample_columns": list(sample_cols)}
