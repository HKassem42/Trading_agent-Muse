"""Tests for the A-Z indicator library (screening program, 2026-10-22).

The load-bearing properties, in the house style (see
tests/test_live_frame.py and the causality-checks skill):

(a) truncation causality -- every registered indicator, computed on a
    frame truncated at K, must reproduce the full-frame values for
    ALL rows whose inputs lie inside the truncated frame;
(b) train-only scaling statistics -- mutating rows at/after the split
    must not change any scaled train value (no global-stats leak);
(c) the [-1, 1] range contract on scaled outputs, synthetic + real
    data, wherever the value is finite;
(d) reference values computed a second, independent way: RSI and ATR
    against the project's own pre-existing implementations in
    features.technical_indicators, and Stochastic %K against
    arithmetic written out in the test from the definition;
(e) warmup NaNs form a prefix per indicator (explicit warmup, no
    interior gaps appearing silently);
(f) registry sanity.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.az_indicators import (  # noqa: E402
    AZ_COLUMNS, REGISTRY, apply_scalers, compute_az, compute_raw,
    fit_scalers)

ROOT = Path(__file__).resolve().parents[1]


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


# (f) ----------------------------------------------------------------
def test_registry_sane():
    assert len(AZ_COLUMNS) >= 100
    assert AZ_COLUMNS == sorted(REGISTRY)
    assert len(set(AZ_COLUMNS)) == len(AZ_COLUMNS)
    for name, spec in REGISTRY.items():
        assert callable(spec["build"]), name
        kind = spec["scale"][0]
        assert kind in ("natural", "unit", "robust"), name
        if kind == "natural":
            _, lo, hi = spec["scale"]
            assert lo < hi, name


# (a) ----------------------------------------------------------------
def test_truncation_causality_all_indicators_all_rows():
    df = _frame()
    full = compute_raw(df)
    assert list(full.columns) == AZ_COLUMNS
    for k in (500, 800, 1100):
        trunc = compute_raw(df.iloc[:k])
        rows = full.index[: k - 1]  # rows whose inputs sit inside [:k]
        a = full.loc[rows].to_numpy(dtype=float)
        b = trunc.loc[rows].to_numpy(dtype=float)
        finite = np.isfinite(a)
        # coverage: the test must actually exercise most of the frame
        assert finite.mean() > 0.5
        assert np.isfinite(b[finite]).all(), (
            f"truncation at {k}: rows finite in the full frame went "
            "NaN in the truncated frame")
        diff = np.abs(a[finite] - b[finite])
        bad = int((diff >= 1e-6).sum())
        assert bad == 0, (
            f"truncation at {k}: {bad} values differ (max "
            f"{diff.max():.2e}) -- look-ahead contamination")


# (b) ----------------------------------------------------------------
def test_scaling_statistics_train_only():
    df = _frame()
    split = 900
    scaled_a, _, scalers_a = compute_az(df, split)
    mutated = df.copy()
    mutated.iloc[split:, :] = mutated.iloc[split:, :] * 1.0
    mutated.iloc[split:, mutated.columns.get_loc("close")] *= 1.37
    mutated.iloc[split:, mutated.columns.get_loc("open")] *= 1.37
    mutated.iloc[split:, mutated.columns.get_loc("high")] *= 1.41
    mutated.iloc[split:, mutated.columns.get_loc("low")] *= 1.33
    mutated.iloc[split:, mutated.columns.get_loc("volume")] *= 2.9
    scaled_b, _, scalers_b = compute_az(mutated, split)
    assert scalers_a == scalers_b, "train-fitted scalers changed"
    a = scaled_a.iloc[:split].to_numpy(dtype=float)
    b = scaled_b.iloc[:split].to_numpy(dtype=float)
    assert (np.isfinite(a) == np.isfinite(b)).all()
    fin = np.isfinite(a)
    assert np.allclose(a[fin], b[fin], atol=1e-12), (
        "a scaled TRAIN value changed when holdout rows were mutated "
        "-- scaling statistics leaked across the split")


# (c) ----------------------------------------------------------------
def test_scaled_range_contract_synthetic():
    df = _frame()
    scaled, _, _ = compute_az(df, 900)
    v = scaled.to_numpy(dtype=float)
    fin = v[np.isfinite(v)]
    assert fin.min() >= -1.0 - 1e-12 and fin.max() <= 1.0 + 1e-12


def test_scaled_range_contract_real_frame():
    cache = ROOT / "data" / "diagnosis" / "sup.pkl"
    if not cache.exists():
        pytest.skip("diagnosis sup cache not present")
    sup = pd.read_pickle(cache)
    ohlcv = sup[["open", "high", "low", "close", "volume"]]
    scaled, _, _ = compute_az(ohlcv, len(sup) - 792)
    v = scaled.to_numpy(dtype=float)
    fin = v[np.isfinite(v)]
    assert fin.min() >= -1.0 - 1e-12 and fin.max() <= 1.0 + 1e-12
    # every series must be usable on the train split
    train = scaled.iloc[: len(sup) - 792]
    frac = train.notna().mean()
    assert (frac > 0.5).all(), (
        f"series mostly NaN on train: {frac[frac <= 0.5].to_dict()}")


# (d) ----------------------------------------------------------------
def test_rsi_matches_project_implementation():
    from features.technical_indicators import _rsi
    df = _frame(1500, seed=11)
    raw = compute_raw(df)
    ref = _rsi(df["close"], 14)
    diff = (raw["az_rsi_14"] - ref).abs().iloc[300:]
    assert diff.max() < 1e-8, f"RSI disagreement: {diff.max():.2e}"


def test_atr_matches_project_implementation():
    from features.technical_indicators import _atr
    df = _frame(1500, seed=11)
    raw = compute_raw(df)
    ref = _atr(df, 14)
    got = raw["az_atr_pct"] * df["close"]
    diff = (got - ref).abs().iloc[300:]
    assert diff.max() < 1e-6, f"ATR disagreement: {diff.max():.2e}"


def test_stochastic_k_hand_computed():
    """%K from the definition, arithmetic written out here: raw %K is
    (close - LL14)/(HH14 - LL14)*100; pandas_ta smooths %K with an
    SMA(3) (smooth_k) and %D is SMA(3) of %K."""
    rng = np.random.default_rng(5)
    n = 300  # long enough for every builder's warmup (SMA200, PVI)
    idx = pd.date_range("2026-09-01", periods=n, freq="1h")
    close = 50.0 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 0.2, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.2, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": np.full(n, 10.0)},
                      index=idx)
    raw = compute_raw(df)
    raws = []
    for t in range(n):
        if t < 13:
            raws.append(np.nan)
            continue
        hh = high[t - 13: t + 1].max()
        ll = low[t - 13: t + 1].min()
        raws.append(100.0 * (close[t] - ll) / (hh - ll))
    raws = np.array(raws)
    k_smooth = pd.Series(raws).rolling(3).mean().to_numpy()
    d_smooth = pd.Series(k_smooth).rolling(3).mean().to_numpy()
    got_k = raw["az_stoch_k"].to_numpy()
    got_d = raw["az_stoch_d"].to_numpy()
    m = np.isfinite(k_smooth)
    assert np.allclose(got_k[m], k_smooth[m], atol=1e-8)
    m2 = np.isfinite(d_smooth)
    assert np.allclose(got_d[m2], d_smooth[m2], atol=1e-8)


# (e) ----------------------------------------------------------------
def test_warmup_nans_are_prefix_only():
    df = _frame()
    raw = compute_raw(df)
    for name in AZ_COLUMNS:
        isn = raw[name].isna().to_numpy()
        assert not isn.all(), f"{name}: all NaN"
        if isn.any():
            first_finite = int(np.argmax(~isn))
            assert not isn[first_finite:].any(), (
                f"{name}: interior NaN after warmup (first at "
                f"{int(np.where(isn[first_finite:])[0][0]) + first_finite})")


# scaling machinery round-trip ---------------------------------------
def test_scalers_roundtrip_bounds():
    df = _frame()
    raw = compute_raw(df)
    scalers = fit_scalers(raw.iloc[:900])
    scaled = apply_scalers(raw, scalers)
    assert list(scaled.columns) == AZ_COLUMNS
    for name, sc in scalers.items():
        assert sc["scale"] > 0, name
