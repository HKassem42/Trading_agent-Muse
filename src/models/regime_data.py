"""Drawing-Board Option B -- daily regime data (2026-10-24).

Executes refs/regime-data-plan-2026-10-24.md EXACTLY as
pre-registered. The plan is the contract; this module is machinery
only. Every prior study rearranged the same 1h candle information;
this one adds a genuinely new information TYPE: daily regime /
context series (Alternative.me Fear & Greed; FRED VIXCLS, DTWEXBGS,
T10Y2Y, DFF), used as context on the 1h frame -- as features and as
a filter, never as standalone entry signals.

CAUSALITY -- the central trap of this study. A daily value stamped
day D becomes usable only from D+1 00:00 UTC (Fear & Greed) or
D+2 00:00 UTC (FRED: publication lag + a safety day). Formally,
usable(d) = value of the freshest stamp s with s + lag <= d; an
hourly row takes the usable value of its UTC calendar date.
Forward-fill only, across calendar days of the usable series;
never backward. The convention is pinned by
tests/test_regime_data.py (stamp exactness incl. the midnight
boundary, truncation invariance, the extra-lag staleness identity).

Disclosed limitation (in the plan, repeated in the report):
fredgraph serves current REVISED values; first-published vintages
are not retrievable from it. The D+2 lag mitigates timing, not
revision. Fear & Greed values are published once, unrevised.

Arms (all LightGBM, gbm_focus.LGBM_PARAMS verbatim, balanced class
weights via gbm_focus's arm-B helpers):
  focus_regime  focus-11 + the 10 regime features -- WF + duel.
  regime_only   the 10 regime features alone -- WF diagnostic only.
  filter        Focus-B-style signal (focus-11, arm-B fit) with the
                conviction vetoed by source-native F&G bands: LONG
                zeroed when usable F&G < 25, SHORT zeroed when
                usable F&G > 75 -- duel only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models import gbm_focus as gf

ROOT = Path(__file__).resolve().parents[2]
REGIME_DIR = ROOT / "data" / "regime"

TARGET_COL = gf.TARGET_COL
CLASSES = gf.CLASSES
FOCUS_FEATURES = list(gf.FOCUS_FEATURES)

# key -> (raw csv under data/regime/, availability lag in days).
# F&G is stamped with the day it describes and published after that
# day closes -> D+1. FRED series get D+2 (publication lag + safety).
SERIES = {
    "fng": ("fear_greed_raw.csv", 1),
    "vix": ("fred_VIXCLS_raw.csv", 2),
    "usd": ("fred_DTWEXBGS_raw.csv", 2),
    "t10y2y": ("fred_T10Y2Y_raw.csv", 2),
    "dff": ("fred_DFF_raw.csv", 2),
}
REGIME_FEATURES = [f"{k}_{s}" for k in SERIES for s in ("level", "chg5")]

# Source-native F&G veto lines (Alternative.me published bands:
# 0-24 Extreme Fear, 76-100 Extreme Greed), fixed in the plan.
FNG_LONG_MIN = 25.0    # longs allowed only when usable F&G >= 25
FNG_SHORT_MAX = 75.0   # shorts allowed only when usable F&G <= 75

ARMS = ("focus_regime", "regime_only", "filter")
ARM_LABELS = {
    "focus_regime": "Focus+Regime",
    "regime_only": "Regime Only",
    "filter": "F&G Filter",
}
ARM_DESIGNS = {
    "focus_regime": "LightGBM class-weighted on Kass's focus-11 plus "
                    "10 daily-regime features (usable level + 5d "
                    "change for F&G / VIX / broad USD / T10Y2Y / DFF; "
                    "F&G usable D+1, FRED D+2)",
    "regime_only": "LightGBM class-weighted on the 10 regime "
                   "features alone (walk-forward diagnostic only)",
    "filter": "Focus-B-style signal (focus-11, class-weighted) with "
              "convictions vetoed by usable Fear & Greed: LONG zeroed "
              "when F&G < 25, SHORT zeroed when F&G > 75 (duel only)",
}

__all__ = ["SERIES", "REGIME_FEATURES", "FOCUS_FEATURES",
           "FNG_LONG_MIN", "FNG_SHORT_MAX", "ARMS", "ARM_LABELS",
           "ARM_DESIGNS", "load_daily_raw", "usable_series",
           "daily_feature_frame", "build_daily_features",
           "hourly_regime_features", "make_feature_frame",
           "apply_fng_filter", "walk_forward_features",
           "rule1_walk_forward", "rule2_duel", "rule3_robustness",
           "apply_verdict_arm1", "apply_verdict_arm3"]


# ------------------------------------------------------------------ data
def load_daily_raw(key: str, base: Path | None = None) -> pd.Series:
    """One raw daily series from data/regime/: a float Series indexed
    by stamp date (midnight, tz-naive UTC), sorted, de-duplicated.
    FRED '.' missing markers are dropped (none occur in-span)."""
    fname, _lag = SERIES[key]
    path = (base or REGIME_DIR) / fname
    raw = pd.read_csv(path)
    date_col, val_col = raw.columns[0], raw.columns[1]
    s = pd.Series(
        pd.to_numeric(raw[val_col].replace(".", np.nan),
                      errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(raw[date_col]).dt.normalize(),
        name=key)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def usable_series(raw: pd.Series, lag: int,
                  end_date: pd.Timestamp) -> pd.Series:
    """The causal usable series on a full calendar: value(d) = the
    freshest stamp s with s + lag days <= d. Forward-fill only."""
    shifted = pd.Series(raw.to_numpy(dtype=float),
                          index=raw.index + pd.Timedelta(days=lag),
                          name=raw.name).sort_index()
    calendar = pd.date_range(shifted.index.min(), end_date, freq="D")
    return shifted.reindex(calendar, method="ffill")


def daily_feature_frame(raw: pd.Series, lag: int,
                        end_date: pd.Timestamp) -> pd.DataFrame:
    """Per-series daily features on the usable calendar: the usable
    level and its 5-calendar-day change (regime direction)."""
    u = usable_series(raw, lag, end_date)
    return pd.DataFrame({"level": u, "chg5": u - u.shift(5)})


def build_daily_features(
        end_date: pd.Timestamp = pd.Timestamp("2026-10-24"),
        base: Path | None = None) -> dict[str, pd.DataFrame]:
    """{series key: daily feature frame} for every kept series."""
    out = {}
    for key, (_fname, lag) in SERIES.items():
        out[key] = daily_feature_frame(load_daily_raw(key, base),
                                       lag, end_date)
    return out


def hourly_regime_features(hour_index: pd.DatetimeIndex,
                           daily: dict[str, pd.DataFrame]
                           ) -> pd.DataFrame:
    """Map the daily usable features onto hourly rows: each hour
    takes the usable value of its own UTC calendar date."""
    days = hour_index.normalize()
    cols = {}
    for key, frame in daily.items():
        for sub in ("level", "chg5"):
            cols[f"{key}_{sub}"] = frame[sub].reindex(days).to_numpy(
                dtype=float)
    return pd.DataFrame(cols, index=hour_index)[REGIME_FEATURES]


def make_feature_frame(matrix: pd.DataFrame,
                       daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The v1 matrix joined with the 10 regime features on its own
    hourly index. Aborts on any NaN inside the matrix span."""
    feats = hourly_regime_features(matrix.index, daily)
    out = matrix.join(feats, how="left")
    n_nan = int(out[REGIME_FEATURES].isna().sum().sum())
    assert n_nan == 0, f"regime features have {n_nan} NaNs in-span"
    return out


# ------------------------------------------------------------------ filter
def apply_fng_filter(conv: np.ndarray, fng_level: np.ndarray):
    """The pre-registered regime veto on signed convictions: a LONG
    conviction (c > 0) is zeroed when the usable F&G level is < 25;
    a SHORT conviction (c < 0) is zeroed when it is > 75; everything
    else passes through with magnitude unchanged. Returns
    (filtered, n_vetoed)."""
    conv = np.asarray(conv, dtype=float)
    fng = np.asarray(fng_level, dtype=float)
    assert conv.shape == fng.shape
    veto = ((conv > 0) & (fng < FNG_LONG_MIN)) \
        | ((conv < 0) & (fng > FNG_SHORT_MAX))
    out = np.where(veto, 0.0, conv)
    return out, int(veto.sum())


# ------------------------------------------------------------- walk-forward
def _fold_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pred_distribution": {c: float((pred == c).mean())
                              for c in CLASSES},
    }


def walk_forward_features(df_pre: pd.DataFrame,
                          feature_cols: list[str]) -> dict:
    """TimeSeriesSplit(5) walk-forward over the pre-duel frame with
    gbm_focus's arm-B fold helper (LGBM_PARAMS verbatim, balanced
    class weights) on an arbitrary feature set. Identical fold
    indices for every caller; pooled metrics over the graded rows
    (the union of the 5 validation blocks)."""
    from sklearn.model_selection import TimeSeriesSplit

    X = df_pre[feature_cols].to_numpy(dtype=float)
    y = df_pre[TARGET_COL].to_numpy()
    ts = df_pre.index
    tss = TimeSeriesSplit(n_splits=5)
    folds, audits = [], []
    oof_pred = np.full(len(df_pre), None, dtype=object)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, audit = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, "B")
        pred = model.predict(X[va_idx])
        oof_pred[va_idx] = pred
        audits.append(audit)
        m = _fold_metrics(y[va_idx], pred)
        m["fold"] = k
        folds.append(m)
    graded = oof_pred != None  # noqa: E711 -- object array sentinel
    pooled = _fold_metrics(y[graded], oof_pred[graded])
    pooled["majority_baseline"] = float(
        max((y[graded] == c).mean() for c in CLASSES))
    pooled["n_graded"] = int(graded.sum())
    return {"folds": folds, "pooled": pooled, "fold_audits": audits}


# ---------------------------------------------------------------- verdict
def rule1_walk_forward(wf: dict) -> bool:
    """Pooled balanced accuracy > 1/3 AND >= 4/5 folds above 1/3 AND
    pooled accuracy >= the majority baseline of the graded rows --
    the operationalization the composite and 4h plans fixed."""
    p = wf["pooled"]
    fold_bals = [f["balanced_accuracy"] for f in wf["folds"]]
    return bool(p["balanced_accuracy"] > 1.0 / 3.0
                and sum(1 for b in fold_bals if b > 1.0 / 3.0) >= 4
                and p["accuracy"] >= p["majority_baseline"])


def rule2_duel(duel: dict) -> bool:
    """Absolute duel floor (fixed in the tasking): total P&L > $0
    AND total P&L > buy-and-hold AND non-hold activity >= 10%."""
    return bool(duel["total_pnl"] > 0.0
                and duel["total_pnl"] > duel["bench_pnl"]
                and duel["non_hold_share"] >= 0.10)


def rule3_robustness(robust: dict) -> bool:
    return bool(robust["total_pnl"] >= 0.0)


def apply_verdict_arm1(wf: dict, duel: dict, robust: dict) -> dict:
    """Arm 1 (Focus+Regime): all three rules or NULL."""
    r1 = rule1_walk_forward(wf)
    r2 = rule2_duel(duel)
    r3 = rule3_robustness(robust)
    return {
        "verdict": "PASS" if (r1 and r2 and r3) else "FAIL",
        "rule1_walk_forward": r1,
        "rule2_duel": r2,
        "rule3_robustness": r3,
        "fold_balanced_accuracies":
            [f["balanced_accuracy"] for f in wf["folds"]],
    }


def apply_verdict_arm3(duel: dict, robust: dict) -> dict:
    """Arm 3 (F&G Filter): duel + robustness rules only (the plan
    gives it no walk-forward of its own)."""
    r2 = rule2_duel(duel)
    r3 = rule3_robustness(robust)
    return {
        "verdict": "PASS" if (r2 and r3) else "FAIL",
        "rule1_walk_forward": None,
        "rule2_duel": r2,
        "rule3_robustness": r3,
        "fold_balanced_accuracies": [],
    }
