"""Population Stability Index (PSI) drift math (2026-10-26,
research incorporation from Kass's "Automated Trading Dashboard
Design" doc, Part III: Data Drift and Regime Shift Detection).

PSI compares a live sample against a baseline (reference) sample of
the same variable. Bins are the BASELINE's quantiles, identical for
both samples, so the statistic measures how the live mass moved
across the baseline's own shape:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

where e_i / a_i are the baseline / live shares in bin i. Shares of
exactly zero are floored at `eps` (standard practice) so an emptied
or newly filled bin yields a large finite penalty instead of an
exception. Bands (the research doc's table): < 0.1 stable,
0.1-0.25 moderate shift, >= 0.25 significant shift.

A feature whose baseline is (near-)constant has no quantile shape
to compare against: psi returns None -- "no baseline" -- rather
than a fabricated number. The same holds for empty samples and
samples too short to bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAND_STABLE = 0.1
BAND_MODERATE = 0.25


def psi_band(value: float | None) -> str:
    """The research doc's alert band for one PSI value."""
    if value is None:
        return "none"
    if value < BAND_STABLE:
        return "stable"
    if value < BAND_MODERATE:
        return "moderate"
    return "significant"


def psi_for_samples(baseline, live, n_bins: int = 10,
                    eps: float = 1e-4) -> float | None:
    """PSI of `live` against `baseline`, or None when no honest
    comparison exists (empty/short samples, or a baseline with
    fewer than 2 distinct quantile edges -- a constant baseline
    has no distribution shape to drift from)."""
    e = np.asarray(baseline, dtype=float)
    a = np.asarray(live, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) < n_bins * 2 or len(a) < 2:
        return None
    edges = np.unique(np.quantile(e, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:  # need at least 2 bins with real width
        return None
    cuts = edges[1:-1]
    e_bin = np.searchsorted(cuts, e, side="right")
    a_bin = np.searchsorted(cuts, a, side="right")
    n = len(edges) - 1
    e_share = np.bincount(e_bin, minlength=n) / len(e)
    a_share = np.bincount(a_bin, minlength=n) / len(a)
    e_share = np.maximum(e_share, eps)
    a_share = np.maximum(a_share, eps)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_frame(baseline: pd.DataFrame, live: pd.DataFrame,
              columns) -> dict[str, float | None]:
    """PSI per column for two frames sharing a feature pipeline.
    Columns missing from either frame map to None (no baseline)."""
    out: dict[str, float | None] = {}
    for c in columns:
        if c not in baseline.columns or c not in live.columns:
            out[c] = None
            continue
        out[c] = psi_for_samples(baseline[c].to_numpy(),
                                 live[c].to_numpy())
    return out
