"""PSI drift math (2026-10-26 incorporation, item 4).

The expected values are HAND-COMPUTED from the PSI definition on
toy binned distributions -- the implementation must reproduce the
arithmetic a person can do on paper, not just agree with itself.
Bins are baseline quantiles, identical for both samples.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.drift import psi_band, psi_for_samples, psi_frame  # noqa: E402


def test_identical_distributions_have_zero_psi():
    x = np.linspace(-1.0, 1.0, 200)
    assert psi_for_samples(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_hand_computed_two_bin_shift():
    # Baseline: 4 zeros + 4 ones -> quantile edges [0, 0.5, 1] at
    # n_bins=2, shares e = [0.5, 0.5]. Live: 3 zeros + 1 one ->
    # shares a = [0.75, 0.25].
    # PSI = 0.25*ln(1.5) + (-0.25)*ln(0.5) = 0.27465...
    baseline = np.array([0.0] * 4 + [1.0] * 4)
    live = np.array([0.0] * 3 + [1.0])
    expected = 0.25 * math.log(1.5) - 0.25 * math.log(0.5)
    got = psi_for_samples(baseline, live, n_bins=2)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(0.2747, abs=1e-3)


def test_hand_computed_emptied_bin_uses_eps_floor():
    # Live entirely in bin 0: a = [1.0, eps] after the 1e-4 floor.
    baseline = np.array([0.0] * 4 + [1.0] * 4)
    live = np.array([0.0] * 8)
    eps = 1e-4
    expected = ((1.0 - 0.5) * math.log(1.0 / 0.5)
                + (eps - 0.5) * math.log(eps / 0.5))
    got = psi_for_samples(baseline, live, n_bins=2)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got > 0.25  # an emptied bin is a significant shift


def test_bins_come_from_baseline_not_live():
    # If bins were (wrongly) taken from the live sample, this pair
    # would read ~0: the live sample IS uniform in its own bins.
    # Against the baseline's shape the shift is real and positive.
    rng = np.random.default_rng(3)
    baseline = rng.normal(0.0, 1.0, 2000)
    live = rng.normal(1.5, 1.0, 2000)
    got = psi_for_samples(baseline, live)
    assert got is not None and got > 0.25


def test_degenerate_baseline_is_no_baseline():
    assert psi_for_samples(np.full(100, 0.5), np.linspace(0, 1, 50)) is None
    near_const = np.full(100, 0.5)
    near_const[0] = 0.6  # still < 2 usable quantile edges of width
    assert psi_for_samples(near_const, np.linspace(0, 1, 50)) is None


def test_short_or_empty_samples_are_none():
    assert psi_for_samples(np.array([]), np.array([1.0, 2.0])) is None
    assert psi_for_samples(np.linspace(0, 1, 100), np.array([])) is None
    assert psi_for_samples(np.linspace(0, 1, 5), np.linspace(0, 1, 50),
                           n_bins=10) is None  # baseline too short to bin


def test_nan_and_inf_are_dropped_not_counted():
    baseline = np.linspace(-1, 1, 200)
    live = np.concatenate([baseline, [np.nan] * 40, [np.inf] * 10])
    assert psi_for_samples(baseline, live) == pytest.approx(0.0, abs=1e-9)


def test_bands_match_the_research_doc():
    assert psi_band(None) == "none"
    assert psi_band(0.0) == "stable"
    assert psi_band(0.0999) == "stable"
    assert psi_band(0.1) == "moderate"
    assert psi_band(0.2499) == "moderate"
    assert psi_band(0.25) == "significant"
    assert psi_band(2.0) == "significant"


def test_psi_frame_maps_columns_and_missing_to_none():
    rng = np.random.default_rng(11)
    base = pd.DataFrame({"a": rng.normal(0, 1, 500),
                         "b": rng.normal(5, 1, 500)})
    live = pd.DataFrame({"a": rng.normal(0, 1, 200),
                         "b": rng.normal(5, 1, 200)})
    out = psi_frame(base, live, ["a", "b", "missing_col"])
    assert out["a"] is not None and out["a"] < 0.1
    assert out["b"] is not None and out["b"] < 0.1
    assert out["missing_col"] is None
