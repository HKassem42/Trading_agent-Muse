"""Muse's Pick #4 / #5 / #6 machinery — LAB ONLY (Kass's order,
2026-11-04). Pre-registered in
refs/muse-picks-456-preregistration-2026-11-04.md.

Three different levers on the same 98 scored eyes and the same
792-candle holdout the Next Try study used:

- #4 "Three Clocks": Pick-recipe pipelines fitted to 1h / 4h /
  24h direction labels; the vote is their equal-weight average.
- #5 "Fresh Eyes": the champion four tree learners x 5 seeds,
  fitted with exponential recency sample weights (half-life
  120 days), no SMOTE, no meta-learner; plain probability vote.
- #6 "Wide Net, Clean Labels": a 7-learner stacking classifier
  (champion four + kNN + MLP + LR) trained only on rows whose
  |next-candle return| clears the train rows' bottom quartile.

Nothing here touches the production stacker, the nightly, or any
seat. Horizon labels are LABELS ONLY: the label at row t reads
close[t] and close[t+h], which is the definition of the target —
features at row t never see past t (the eyes frame's own
causality was truncation-tested in the Next Try study, and the
label construction is pinned by tests/test_muse_picks456.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

HALF_LIFE_DAYS = 120.0
BAG_SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 4, 24)


def horizon_labels(close: pd.Series, h: int) -> pd.Series:
    """1 where close[t+h] > close[t], 0 otherwise, NaN where the
    endpoint does not exist. Label-only by construction."""
    if h < 1:
        raise ValueError("horizon must be >= 1")
    fut = close.shift(-h)
    return (fut > close).astype(float).where(fut.notna())


def recency_weights(index: pd.DatetimeIndex,
                    half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential decay weights anchored at the newest row:
    w = 0.5 ** (age_days / half_life). Newest row weighs 1.0."""
    idx = pd.DatetimeIndex(index)
    age_days = (idx[-1] - idx).total_seconds() / 86400.0
    return np.power(0.5, age_days / float(half_life_days))


def quartile_threshold(abs_values: pd.Series) -> float:
    """The bottom-quartile cut of the values handed in — the
    caller is responsible for handing it TRAIN rows only."""
    return float(np.quantile(np.asarray(abs_values, dtype=float), 0.25))


def clean_keep_mask(abs_values: pd.Series, threshold: float) -> np.ndarray:
    """Rows whose |move| strictly clears the threshold survive."""
    return (np.asarray(abs_values, dtype=float) > threshold)


def build_bag_learners(random_state: int):
    """The Pick recipe's four tree learners, champion
    hyperparameters, as bare estimators (Pick #5 fits them
    directly — no SMOTE, no stacking meta-learner)."""
    return [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1,
            random_state=random_state)),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=random_state, eval_metric="logloss")),
        ("lgbm", LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=random_state, verbose=-1)),
        ("et", ExtraTreesClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1,
            random_state=random_state)),
    ]


def build_wide_stack(random_state: int = 42) -> StackingClassifier:
    """Pick #6's seven-learner stack: the champion four trees
    plus kNN, an MLP, and logistic regression for inductive-bias
    diversity; LR meta-learner, 3-fold internal CV (the Pick
    recipe's own stacking depth). Expects pre-scaled inputs."""
    base = build_bag_learners(random_state) + [
        ("knn", KNeighborsClassifier(n_neighbors=40, n_jobs=-1)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(48, 24), max_iter=250,
            random_state=random_state)),
        ("lr", LogisticRegression(max_iter=1000)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )
