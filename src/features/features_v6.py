"""Curated features v6 -- LAB ONLY (Kass directive, 2026-10-25).

A candidate successor to the production feature set (v5.1, 21
columns), built to answer one question honestly: does a CURATED
wider set beat the 21 through the same gate? Construction rules,
pre-registered before any training:

- RETAINED: the production 21 minus obv_slope. The diagnosis
  (refs/diagnosis-2026-10-21.md) measured obv_slope == netvol_rel at
  r = 0.999 (sign agreement 99.71%) -- a pure duplicate dimension.
  netvol_rel stays: it is the bounded share by construction, defined
  without the cumulative OBV level.
- ADDED: only indicators from the precomputed universe (the A-Z
  screen's 132, data/az_screen/ -- the same builders the timeseries
  matrices v1-v3 scored) whose screen-measured max |r| against the
  production 21 is < 0.90 (data/az_screen/results.json `corr`). The
  screen found NO graduate among all 132, and 16 of them are
  near-duplicates of the existing 21 at |r| > 0.95 -- so curation is
  by NON-REDUNDANCY and family diversity, in the priority order
  below; the build script additionally enforces pairwise |r| < 0.90
  against the retained 20 and every already-accepted addition,
  computed on the study's own train frame, and records the evidence
  for every accept/reject.
- SCALING: the AZ module's own spec (natural/unit/robust), with
  robust scalers fitted on TRAIN rows only (train_end_pos), exactly
  the screen's discipline -- mutating holdout rows cannot move a
  train value. Every column lands in [-1, 1].
- CAUSALITY: every builder is trailing-only (the AZ registry's
  contract, pinned by tests/test_az_indicators.py), and the assembled
  v6 frame is held to the v5.1 standard by tests/test_features_v6.py:
  truncation causality across ALL rows, a future-perturbation test,
  and a two-way equality check against the AZ module's compute_az.

Production never imports this module; the live loop and the nightly
retrain stay on FEATURE_COLUMNS whatever the study concludes. A v6
win would still need Kass's explicit approval to install (the
promotion lock, 2026-10-25).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.az_indicators import REGISTRY
from features.technical_indicators import FEATURE_COLUMNS, add_features

# The duplicate pair: obv_slope is dropped, netvol_rel kept.
DROPPED_DUPLICATE = "obv_slope"
RETAINED_COLUMNS = [c for c in FEATURE_COLUMNS if c != DROPPED_DUPLICATE]

# Priority order: family-diverse, lowest screen redundancy first
# within each family. Screen max_corr_21 (vs the production 21 --
# a superset of the retained 20, so these bounds stay valid after
# the drop) noted per candidate.
ADD_CANDIDATES = [
    "az_er_10",         # trend efficiency      0.276
    "az_adx_14",        # trend strength        0.500
    "az_supertrend_dir",  # trend direction     0.657
    "az_vhf_28",        # trend/range regime    0.445
    "az_aroon_down",    # trend recency         0.583
    "az_cci_20",        # momentum              0.383
    "az_dpo_20",        # detrended momentum    0.577
    "az_stc",           # momentum cycle        0.502
    "az_fisher_9",      # momentum transform    0.753
    "az_chop_14",       # volatility regime     0.486
    "az_mass_index",    # volatility            0.408
    "az_chaikin_vol",   # volatility            0.429
    "az_ulcer_14",      # drawdown volatility   0.669
    "az_vroc_14",       # volume momentum       0.483
    "az_klinger_diff",  # volume flow           0.691
    "az_mfi_14",        # volume momentum       0.780
    "az_ret_kurt_20",   # return distribution   0.198
    "az_acf_1_50",      # return autocorrelation 0.226
    "az_pv_corr_20",    # price/volume corr     0.619
    "az_ibs",           # bar strength          0.666
]

# Consulted in order only if the primary list yields fewer than the
# study's minimum additions (pairwise rejections can bite).
ADD_FALLBACK = [
    "az_aroon_up",      # 0.591
    "az_dss_10",        # 0.527
    "az_vol_osc",       # 0.518
    "az_bwmfi",         # 0.266
    "az_nvi_ret_20",    # 0.293
    "az_ret_skew_20",   # 0.507
    "az_eom_14",        # 0.759
    "az_force_index",   # 0.727
]


def v6_columns(added: list[str]) -> list[str]:
    """The v6 feature list for a chosen addition set."""
    return list(RETAINED_COLUMNS) + list(added)


def _fit_scalers_subset(raw_train: pd.DataFrame, names) -> dict:
    """The AZ module's fit_scalers, restricted to `names` (identical
    formulas: robust center = train median, scale = (p99.5 - p0.5)/2,
    degenerate guards identical)."""
    scalers = {}
    for name in names:
        if REGISTRY[name]["scale"] != ("robust",):
            continue
        x = raw_train[name].dropna()
        if len(x) < 50:
            scalers[name] = {"center": 0.0, "scale": 1.0,
                             "degenerate": True}
            continue
        center = float(x.median())
        scale = float((x.quantile(0.995) - x.quantile(0.005)) / 2.0)
        scalers[name] = {"center": center,
                         "scale": scale if scale > 1e-12 else 1.0,
                         "degenerate": bool(scale <= 1e-12)}
    return scalers


def compute_added(df: pd.DataFrame, added: list[str],
                  train_end_pos: int) -> pd.DataFrame:
    """The added AZ indicators for an OHLCV frame, scaled to [-1, 1]
    per the registry spec with robust scalers fitted on the first
    `train_end_pos` rows ONLY. Verified column-for-column against
    features.az_indicators.compute_az by the v6 tests."""
    if not added:
        return pd.DataFrame(index=df.index)
    raw = pd.DataFrame(
        {name: pd.Series(np.asarray(REGISTRY[name]["build"](df),
                                    dtype=float), index=df.index)
         for name in added}, index=df.index)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    scalers = _fit_scalers_subset(raw.iloc[:train_end_pos], added)
    out = pd.DataFrame(index=df.index)
    for name in added:
        spec = REGISTRY[name]["scale"]
        x = raw[name]
        if spec == ("unit",):
            y = x.clip(-1.0, 1.0)
        elif spec == ("robust",):
            sc = scalers[name]
            y = ((x - sc["center"]) / sc["scale"]).clip(-1.0, 1.0)
        else:  # ("natural", lo, hi)
            _, lo, hi = spec
            y = (2.0 * (x - lo) / (hi - lo) - 1.0).clip(-1.0, 1.0)
        out[name] = y
    return out


def make_v6_supervised(df: pd.DataFrame, added: list[str],
                       train_end_pos: int) -> pd.DataFrame:
    """Production supervised frame + the scaled additions, with the
    production target (next candle closes higher) and the production
    dropna discipline over the v6 column set."""
    out = add_features(df)
    if added:
        out = out.join(compute_added(df, added, train_end_pos))
    out["target"] = (out["close"].shift(-1) > out["close"]).astype(int)
    cols = v6_columns(added)
    return out.dropna(subset=cols + ["target"]).copy()
