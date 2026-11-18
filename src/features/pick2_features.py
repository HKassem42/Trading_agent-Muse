"""Muse's Pick #2 new features -- LAB ONLY (Kass's order, 2026-10-27).

Pick #2's eyes: Champion #3's scored-102 frame (94 percentile-
scored columns) PLUS four new causal features drawn from the
archive-mining candidates (refs/archive-mining-2026-10-27.md):
a rolling Hurst exponent (regime feature) and the statistical-
moment family (rolling skew / kurtosis / lag-1 autocorrelation
of returns). Every window choice below was pre-registered in
refs/muse-pick-2-challenger-2026-10-27.md BEFORE any grading:

- Hurst: rescaled-range (R/S) exponent of LOG PRICE over a
  trailing 100-bar window. H = log(R/S) / log(100), R the range
  of the mean-adjusted cumulative log-price path inside the
  window, S the window's sample standard deviation (ddof=1) of
  log price. S == 0 (a perfectly flat window) -> NaN.
- Moments: on log returns r_t = log(close_t / close_{t-1}),
  trailing 100-bar windows ending at t: sample skewness and
  excess kurtosis (pandas' unbiased estimators), and the lag-1
  autocorrelation (Pearson correlation of r with its own 1-bar
  lag inside the window).

Each raw series is then percentile-scored through the project's
OWN trailing-4,320-bar machinery (the matrix-v3 builder's
rolling_scores -- the identical function that scores every
champion column), over exactly the champion frame's scored
positions (extended only if a new series' warmup demands it),
with NO orientation flip: the score is the plain trailing
percentile of the raw value. The stacker's trees read the rank
either way; the percentile step exists to put the new features
on the same bounded [0, 100] scale as the rest of the frame.

Causality: every quantity at row t reads only candles <= t --
rolling windows END at t, and the percentile window ends at the
scored row (the v5.1 standard; the Pick #2 builder re-verifies
by truncation rebuild, and tests/test_muse_pick2.py pins the
raw functions against hand-computed references plus a
truncate-and-append invariance check).

Production never imports this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import scored_features

HURST_WINDOW = 100
MOMENT_WINDOW = 100
AUTOCORR_WINDOW = 100

# The champion-metadata feature_set tag for this frame (nightly's
# champion_holdout_frame dispatch): run 73 was crowned Lab Stacker
# Champion #4 on 2026-10-27 under this tag (Muse's ruling under
# Kass's delegation of the same day).
FRAME_TAG = "pick2_98"

RAW_COLUMNS = ["hurst_rs_100_raw", "ret_skew_100_raw",
               "ret_kurt_100_raw", "ret_acf1_100_raw"]
NEW_COLUMNS = ["Rolling Hurst Exponent (100)",
               "Return Skewness (100)",
               "Return Kurtosis (100)",
               "Return Autocorrelation Lag-1 (100)"]


# ------------------------------------------------------------ raw series
def rolling_hurst(close: pd.Series, window: int = HURST_WINDOW
                  ) -> pd.Series:
    """R/S Hurst exponent of log price over a trailing window.

    For the window of log prices x[0..w-1] ending at row t:
    m = mean(x); y = cumsum(x - m); R = max(y) - min(y);
    S = std(x, ddof=1); H = log(R / S) / log(w). NaN until a full
    window exists, and where S == 0 (flat prices carry no range
    information)."""
    logp = np.log(close.to_numpy(dtype=float))
    n = len(logp)
    out = np.full(n, np.nan)
    logw = np.log(window)
    for t in range(window - 1, n):
        x = logp[t - window + 1:t + 1]
        s = x.std(ddof=1)
        if not np.isfinite(s) or s == 0.0:
            continue
        y = np.cumsum(x - x.mean())
        r = float(y.max() - y.min())
        if r <= 0.0:
            continue
        out[t] = np.log(r / s) / logw
    return pd.Series(out, index=close.index, name="hurst_rs_100_raw")


def log_returns(close: pd.Series) -> pd.Series:
    """r_t = log(close_t / close_{t-1}); r_0 is NaN."""
    return np.log(close / close.shift(1))


def rolling_moments(close: pd.Series, window: int = MOMENT_WINDOW
                    ) -> pd.DataFrame:
    """Trailing sample skew / excess kurtosis of log returns
    (pandas' unbiased rolling estimators), windows ending at t."""
    r = log_returns(close)
    return pd.DataFrame({
        "ret_skew_100_raw": r.rolling(window).skew(),
        "ret_kurt_100_raw": r.rolling(window).kurt(),
    }, index=close.index)


def rolling_autocorr1(close: pd.Series, window: int = AUTOCORR_WINDOW
                       ) -> pd.Series:
    """Trailing lag-1 autocorrelation of log returns: the rolling
    Pearson correlation of r_t with r_{t-1} over the window ending
    at t."""
    r = log_returns(close)
    return r.rolling(window).corr(r.shift(1)).rename(
        "ret_acf1_100_raw")


def raw_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """The four raw (unscored) Pick #2 series over a raw OHLCV
    frame, indexed exactly like `df`."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["hurst_rs_100_raw"] = rolling_hurst(close)
    mom = rolling_moments(close)
    out["ret_skew_100_raw"] = mom["ret_skew_100_raw"]
    out["ret_kurt_100_raw"] = mom["ret_kurt_100_raw"]
    out["ret_acf1_100_raw"] = rolling_autocorr1(close)
    return out[RAW_COLUMNS]


# ------------------------------------------------------------ scoring
def build_pick2_frame(df: pd.DataFrame):
    """(frame98, cols98, evidence): Champion #3's scored frame
    (94 columns, from scored_features) joined with the four new
    percentile-scored columns, indexed exactly like `df`.

    Scored positions: the champion frame's own finite positions,
    extended forward only if a new raw series is still warming up
    at the champion's first position (p0 = max(champion p0,
    last_bad_new + WINDOW) per new series, the same warmup rule
    build_scored_frame applies). New columns are NaN outside
    their scored positions; the caller asserts the holdout is
    fully finite on all 98 columns."""
    b3 = scored_features._b3
    window = scored_features.WINDOW
    champ_frame, champ_cols = scored_features.build_scored_frame(df)
    assert champ_cols == scored_features.scored_columns()
    finite = np.isfinite(champ_frame.to_numpy(dtype=float)).all(axis=1)
    champ_positions = np.nonzero(finite)[0]
    assert len(champ_positions) > 0
    raw = raw_new_features(df)
    p0 = int(champ_positions[0])
    qa_map: dict[str, np.ndarray] = {}
    for name in RAW_COLUMNS:
        qa = raw[name].to_numpy(dtype=float)
        isn = ~np.isfinite(qa)
        last_bad = int(np.max(np.nonzero(isn))) if isn.any() else -1
        p0 = max(p0, last_bad + window)
        qa_map[name] = qa
    positions = np.arange(p0, len(df))
    assert len(positions) > 0, "no fully-warmed position for pick2 frame"
    frame = champ_frame.copy()
    for raw_name, col_name in zip(RAW_COLUMNS, NEW_COLUMNS):
        s = b3.rolling_scores(qa_map[raw_name], positions)
        assert s.min() >= 0.0 and s.max() <= 100.0, col_name
        full = np.full(len(df), np.nan)
        full[positions] = s
        frame[col_name] = full
    cols = list(champ_cols) + list(NEW_COLUMNS)
    evidence = {
        "champion_p0": int(champ_positions[0]),
        "pick2_p0": int(p0),
        "scored_positions": int(len(positions)),
        "raw_warmup_last_bad": {
            name: int(np.max(np.nonzero(~np.isfinite(qa_map[name]))))
            if (~np.isfinite(qa_map[name])).any() else -1
            for name in RAW_COLUMNS},
    }
    return frame[cols], cols, evidence


def build_pick2_frame_pair(df: pd.DataFrame):
    """(frame98, cols98) -- build_pick2_frame without the
    evidence, in the (frame, columns) shape the other scored
    frame builders return (the shadow loop's and the dashboard
    feeds' builder slot)."""
    frame, cols, _evidence = build_pick2_frame(df)
    return frame, cols


def builder_for_tag(tag: str | None):
    """The scored-frame builder for a seat/champion feature_set
    tag: scored102 -> the matrix-v3 builder; pick2_98 -> this
    module's 98-column builder. Unknown tags raise -- a silent
    fallback would read the wrong eyes (the nightly dispatch
    doctrine, applied to the shadow seat)."""
    if tag == FRAME_TAG:
        return build_pick2_frame_pair
    if tag == "scored102":
        return scored_features.build_scored_frame
    raise ValueError(f"unknown scored frame tag: {tag!r}")
