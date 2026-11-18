"""Tests for the GBM focus-list challenger experiment (2026-10-22).

Written BEFORE the build, per Kass's quality standard:

(1) Feature extraction pulls exactly the 11 named focus columns from
    the real matrix CSV header (and the arm-D derivation is every
    remaining feature column, Volume included).
(2) No leakage: TimeSeriesSplit fold training indices strictly precede
    validation indices, and arm C's SMOTE is fitted INSIDE the fold --
    proven two ways: the fold trainer's audit shows every original row
    it fed to SMOTE/model carries a training-fold timestamp, and
    poisoning the validation rows with absurd values cannot move the
    trainer's recorded fit-input range (if fitting touched validation
    rows, the poison would show up).
(3) Wrapper accounting through the real TradingEnv + PaperBroker,
    mirroring test_ha_seq.py: a scripted probability stub that always
    says BUY opens a long at the next candle's open, 25% tranche,
    0.15% fees, and a wick through the stop fills at the stop level.

Run: .venv/bin/python -m pytest tests/test_gbm_focus.py -q
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models.gbm_focus import (  # noqa: E402
    ALL_FEATURE_EXCLUDE,
    FOCUS_FEATURES,
    fit_arm_fold,
    matrix_feature_columns,
)
from models.gbm_policy import conviction, decide  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"


# ---------------------------------------------------------------- (1)
def test_focus_features_exact_from_real_csv_header():
    header = pd.read_csv(MATRIX, nrows=0).columns.tolist()
    for name in FOCUS_FEATURES:
        assert name in header, f"focus column missing from CSV: {name}"
    assert len(FOCUS_FEATURES) == 11
    assert len(set(FOCUS_FEATURES)) == 11
    df = pd.read_csv(MATRIX, nrows=64)
    X = df[FOCUS_FEATURES]
    assert X.shape == (64, 11)
    assert list(X.columns) == FOCUS_FEATURES
    assert X.notna().all().all()


def test_arm_d_feature_derivation():
    header = pd.read_csv(MATRIX, nrows=0).columns.tolist()
    feats = matrix_feature_columns(header)
    assert "Volume" in feats
    for banned in ALL_FEATURE_EXCLUDE:
        assert banned not in feats
    assert feats == [c for c in header if c not in ALL_FEATURE_EXCLUDE]
    assert len(feats) == len(header) - len(ALL_FEATURE_EXCLUDE)


# ---------------------------------------------------------------- rule
def test_conviction_thresholds_and_ties():
    # Neither side clears 0.40 -> flat, however lopsided the split.
    assert conviction(0.39, 0.10) == 0.0
    assert conviction(0.10, 0.39) == 0.0
    assert conviction(0.34, 0.33) == 0.0
    # One side clears -> signed conviction carries that probability.
    assert conviction(0.40, 0.20) == pytest.approx(0.40)
    assert conviction(0.20, 0.40) == pytest.approx(-0.40)
    assert conviction(0.90, 0.05) == pytest.approx(0.90)
    # Both clear -> the larger probability wins.
    assert conviction(0.45, 0.42) == pytest.approx(0.45)
    assert conviction(0.42, 0.45) == pytest.approx(-0.45)
    # Exact tie at/above the bar -> LONG (the rule lists LONG first).
    assert conviction(0.45, 0.45) == pytest.approx(0.45)


def test_decide_mirrors_ha_action_mapping():
    assert decide(0.40, 0.0) == 1     # conviction boundary -> long
    assert decide(-0.40, 0.0) == 3    # -> short
    assert decide(0.399, 0.0) == 0    # inside the band, flat book -> hold
    assert decide(0.0, 0.4) == 2      # band + long -> sell (exit)
    assert decide(0.0, -0.4) == 1    # band + short -> buy (cover)
    assert decide(0.85, -0.4) == 1    # conviction overrides position
    assert decide(-0.85, 0.4) == 3


# ---------------------------------------------------------------- (2)
def _toy_matrix(n: int = 900, seed: int = 3):
    """Small matrix-shaped frame: hourly timestamps, the 11 focus
    columns (deterministic wiggle), 3-class labels with HOLD majority."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    data = {c: rng.normal(0, 1, n).cumsum() for c in FOCUS_FEATURES}
    # Give BUY/SHORT rows a faint, learnable nudge on one column.
    y = np.array(["HOLD"] * n, dtype=object)
    y[::7] = "BUY"
    y[3::11] = "SHORT"
    data["Force Index"] = data["Force Index"] + np.where(
        y == "BUY", 2.0, np.where(y == "SHORT", -2.0, 0.0))
    df = pd.DataFrame(data, index=idx)
    return df[FOCUS_FEATURES].to_numpy(dtype=float), y, idx


def test_fold_chronology_and_smote_inside_fold():
    from sklearn.model_selection import TimeSeriesSplit

    X, y, ts = _toy_matrix()
    tss = TimeSeriesSplit(n_splits=5)
    folds = list(tss.split(X))
    tr_idx, va_idx = folds[2]
    assert tr_idx.max() < va_idx.min(), "fold indices overlap in time"

    model, audit = fit_arm_fold(X, y, ts, tr_idx, va_idx, arm="C")
    # Every ORIGINAL row the trainer fed to SMOTE/model is a train row.
    assert audit["fed_ts_min"] >= ts[tr_idx].min()
    assert audit["fed_ts_max"] <= ts[tr_idx].max()
    assert audit["fed_ts_max"] < ts[va_idx].min()
    assert audit["n_fed_original"] == len(tr_idx)
    # SMOTE actually ran: classes balanced after resampling.
    counts = audit["resampled_class_counts"]
    assert len(set(counts.values())) == 1, counts
    assert audit["n_after_resample"] > len(tr_idx)
    # And the fitted model still predicts the validation fold's shape.
    proba = model.predict_proba(X[va_idx])
    assert proba.shape == (len(va_idx), 3)


def test_smote_arm_never_sees_validation_rows_poison_check():
    from sklearn.model_selection import TimeSeriesSplit

    X, y, ts = _toy_matrix()
    tss = TimeSeriesSplit(n_splits=5)
    tr_idx, va_idx = list(tss.split(X))[2]
    X_poisoned = X.copy()
    X_poisoned[va_idx] = 1e15  # absurd values in validation rows only
    _, audit = fit_arm_fold(X_poisoned, y, ts, tr_idx, va_idx, arm="C")
    # If SMOTE (or the scaler, or the model) had been fit on anything
    # beyond the training fold, the poison would appear in fit inputs.
    assert audit["fit_max_abs"] < 1e14
    _, audit_a = fit_arm_fold(X_poisoned, y, ts, tr_idx, va_idx, arm="A")
    assert audit_a["fit_max_abs"] < 1e14


# ---------------------------------------------------------------- (3)
class _AlwaysBuyStub:
    """Scripted probabilities (p_buy=0.90, p_short=0.05) read through
    the env's conviction channel exactly as the SB3 rule policy does:
    obs[21] carries the signed conviction, obs[23] the position."""

    def predict(self, obs, deterministic=True):
        return decide(float(obs[21]), float(obs[23])), None


def _synthetic_sup(closes, opens, highs, lows, atr=0.01):
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [10.0] * n, "atr_pct_raw": [atr] * n,
    }, index=idx)
    for c in FEATURE_COLUMNS:
        df[c] = 0.0
    return df


def test_wrapper_accounting_next_open_fees_and_wick_stop(tmp_path):
    # Same fixture shape as test_ha_seq.py (d): decision candle closes
    # at 111, fill must print at row 1's OPEN (100); row 1 wicks to 96.5
    # through the 2.5% clipped stop (97.5) but closes at 99, so the stop
    # fill must print at the level itself.
    closes = [111.0, 99.0] + [99.0] * 28
    opens = [110.0, 100.0] + [99.0] * 28
    highs = [112.0, 100.5] + [99.5] * 28
    lows = [109.0, 96.5] + [98.5] * 28
    sup = _synthetic_sup(closes, opens, highs, lows)
    # Conviction channel: always-BUY stub -> c = +0.90 on row 0, encoded
    # as p_up = (c + 1) / 2 so obs[21] == c; neutral 0.5 elsewhere.
    p_up = np.full(len(sup), 0.5)
    p_up[0] = (conviction(0.90, 0.05) + 1.0) / 2.0
    db = tmp_path / "gbm_acct.db"
    broker = PaperBroker(db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, p_up, broker, fill_mode="next_open")
    policy = _AlwaysBuyStub()
    obs, _ = env.reset(seed=0, options={"start": 0})
    done, steps = False, 0
    while not done and steps < 30:
        action, _ = policy.predict(obs)
        obs, _, done, _, _ = env.step(int(action))
        steps += 1
    con = sqlite3.connect(db)
    trades = con.execute(
        "SELECT side, price, qty, fee, reason FROM trades ORDER BY id"
    ).fetchall()
    con.close()
    assert len(trades) == 2, trades
    (side, price, qty, fee, _), (side2, price2, qty2, fee2, reason2) = trades
    assert side == "BUY" and price == pytest.approx(100.0)   # next open
    assert qty == pytest.approx(250.0 * (1 - 0.0015), rel=1e-9)
    assert fee == pytest.approx(25_000.0 * 0.0015)          # 0.15% of spend
    assert side2 == "SELL" and reason2 == "stop_loss"
    assert price2 == pytest.approx(97.5)                    # stop level
    assert qty2 == pytest.approx(qty)                       # full exit
    assert fee2 == pytest.approx(qty * 97.5 * 0.0015)       # fee on exit too
