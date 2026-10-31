"""GBM focus-list experiment core (2026-10-22, Kass green-lit).

Data + training machinery for the challenger built on Kass's pasted
top-10 feature ranking, using the real BTC matrix
(data/timeseries_matrix/btc_usd_1h_matrix.csv). Pure machinery here --
the phased build/grade/register driver is scripts/build_gbm_focus.py.

Arms (all LightGBM multiclass, fixed hyperparameters, RAW values --
trees need no scaling):
  A  focus-11, plain
  B  focus-11, balanced class weights (inverse class frequency)
  C  focus-11, SMOTE inside each training fold only. This arm follows
     the project's own stacker recipe (StandardScaler -> SMOTE ->
     model): SMOTE's k-nearest-neighbour geometry is meaningless on
     raw scales (OBV ~1e8 next to ratios ~1), so the scaler is fit on
     the training fold only, SMOTE resamples in scaled space, and the
     wrapped predictor applies the same scaler at predict time.
  D  all matrix feature columns (Volume included), class-weighted.

Leakage discipline: every fit helper receives ONLY the training rows
of a chronological fold (or the full pre-duel region for final fits)
and returns an audit dict proving what it consumed -- the tests poison
the validation rows and assert the poison never reaches a fit input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "Target_Action (BUY/SELL/SHORT/HOLD)"
CLASSES = ["BUY", "HOLD", "SHORT"]

# Kass's pasted top-10 (item 1's name was cut off in his paste; its
# description is a raw-volume move-confirmation feature -> covered by
# the two volume columns below, which share his rank 1).
FOCUS_FEATURES = [
    "Volume",
    "On-Balance Volume (OBV)",
    "Force Index",
    "Elder Force Index (EFI)",
    "Historical Volatility (HV)",
    "Mass Index (MI)",
    "Bollinger BandWidth",
    "Connors RSI (CRSI)",
    "Accumulative Swing Index (ASI)",
    "Ultimate Oscillator (UO)",
    "Choppiness Index (CHOP)",
]

KASS_CLAIMED = {
    "Volume": {"rank": 1, "pct": None},          # % cut off in paste
    "On-Balance Volume (OBV)": {"rank": 1, "pct": None},
    "Force Index": {"rank": 2, "pct": 2.9},
    "Historical Volatility (HV)": {"rank": 3, "pct": 2.6},
    "Mass Index (MI)": {"rank": 4, "pct": 2.3},
    "Bollinger BandWidth": {"rank": 5, "pct": 2.1},
    "Connors RSI (CRSI)": {"rank": 6, "pct": 2.0},
    "Elder Force Index (EFI)": {"rank": 7, "pct": 1.9},
    "Accumulative Swing Index (ASI)": {"rank": 8, "pct": 1.9},
    "Ultimate Oscillator (UO)": {"rank": 9, "pct": 1.9},
    "Choppiness Index (CHOP)": {"rank": 10, "pct": 1.8},
}

ALL_FEATURE_EXCLUDE = ["Date", "Ticker", TARGET_COL,
                       "Open", "High", "Low", "Close"]

ARMS = ("A", "B", "C", "D")

# Fixed hyperparameters -- chosen once, documented, never tuned
# against the duel windows (that would be grading on the answer key).
LGBM_PARAMS = dict(
    objective="multiclass", num_class=3,
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=40, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=4, verbose=-1,
)

SEED = 42


def matrix_feature_columns(header) -> list[str]:
    """Arm-D feature set: every column except id/target/OHLC (Volume
    and all 59 indicator columns stay in)."""
    return [c for c in header if c not in ALL_FEATURE_EXCLUDE]


def load_matrix(path) -> pd.DataFrame:
    """Load the BTC matrix CSV with Date as a sorted DatetimeIndex."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    return df


def arm_features(df: pd.DataFrame, arm: str) -> list[str]:
    if arm in ("A", "B", "C"):
        return list(FOCUS_FEATURES)
    if arm == "D":
        return matrix_feature_columns(list(df.columns))
    raise ValueError(f"unknown arm {arm!r}")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-class-frequency weights: n / (n_classes * count_c)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y], dtype=float)


class _Predictor:
    """Uniform predict_proba surface for every arm: optional train-fold
    scaler (arm C) in front of the LightGBM model, probability columns
    always reordered to CLASSES."""

    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self._order = [list(model.classes_).index(c) for c in CLASSES]

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, self._order]

    def predict(self, X) -> np.ndarray:
        return np.array(CLASSES)[np.argmax(self.predict_proba(X), axis=1)]


def _fit_core(X_tr: np.ndarray, y_tr: np.ndarray, arm: str):
    """Fit one arm on TRAINING ROWS ONLY. Returns (predictor, audit)
    where audit records the resampling outcome (arms A/B/D: identity)."""
    from lightgbm import LGBMClassifier

    counts = {c: int((y_tr == c).sum()) for c in CLASSES}
    audit = {"resampled_class_counts": dict(counts),
             "n_after_resample": int(len(y_tr))}
    model = LGBMClassifier(**LGBM_PARAMS)
    if arm == "C":
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X_tr)
        X_s = scaler.transform(X_tr)
        X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_s, y_tr)
        model.fit(X_res, y_res)
        audit["resampled_class_counts"] = {
            c: int((y_res == c).sum()) for c in CLASSES}
        audit["n_after_resample"] = int(len(y_res))
        return _Predictor(model, scaler=scaler), audit
    if arm in ("B", "D"):
        model.fit(X_tr, y_tr, sample_weight=balanced_weights(y_tr))
    else:
        model.fit(X_tr, y_tr)
    return _Predictor(model), audit


def fit_arm_fold(X: np.ndarray, y: np.ndarray, ts: pd.DatetimeIndex,
                 tr_idx: np.ndarray, va_idx: np.ndarray, arm: str):
    """Fit `arm` on rows tr_idx of (X, y) -- nothing else is touched.

    The audit proves the consumption boundary: fed_ts_min/max are the
    timestamps of the ORIGINAL rows handed to the fit (synthetic SMOTE
    rows have no timestamps and are never counted as evidence), and
    fit_max_abs is the largest |value| among the raw rows fed in, so a
    caller can poison validation rows and verify the poison stayed
    out. va_idx is accepted (and length-checked) so call sites state
    the fold explicitly; it is never read for fitting.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    tr_idx = np.asarray(tr_idx)
    va_idx = np.asarray(va_idx)
    assert len(set(tr_idx.tolist()) & set(va_idx.tolist())) == 0
    assert tr_idx.max() < va_idx.min(), "fold is not chronological"
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    predictor, audit = _fit_core(X_tr, y_tr, arm)
    fed_ts = ts[tr_idx]
    audit.update({
        "n_fed_original": int(len(tr_idx)),
        "fed_ts_min": fed_ts.min(),
        "fed_ts_max": fed_ts.max(),
        "fit_max_abs": float(np.abs(X_tr).max()),
    })
    return predictor, audit


def fit_arm_final(X: np.ndarray, y: np.ndarray, arm: str):
    """Fit `arm` on a complete (pre-duel) training region."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    predictor, audit = _fit_core(X, y, arm)
    audit.update({"n_fed_original": int(len(y)),
                  "fit_max_abs": float(np.abs(X).max())})
    return predictor, audit
