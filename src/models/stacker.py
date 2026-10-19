"""Stacker v2: retrainable next-candle direction classifier.

Rebuilds the Project-Bot recipe (best_trading_model.pkl) as a nightly-
retrainable pipeline: StandardScaler -> SMOTE -> StackingClassifier
(RandomForest + XGBoost + LightGBM -> LogisticRegression).

Outputs P(up) for the next candle, which also feeds the PPO agent's state.
Walk-forward (TimeSeriesSplit) scoring keeps us honest about edge.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def build_pipeline(random_state: int = 42) -> ImbPipeline:
    base = [
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8, n_jobs=-1, random_state=random_state)),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=random_state, eval_metric="logloss")),
        ("lgbm", LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=random_state, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
        ("stack", stack),
    ])


def walk_forward_score(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                       random_state: int = 42) -> dict:
    """TimeSeriesSplit CV. Returns mean accuracy/precision across folds."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    accs, precs = [], []
    for train_idx, test_idx in tss.split(X):
        pipe = build_pipeline(random_state)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], pred))
        precs.append(precision_score(y.iloc[test_idx], pred, zero_division=0))
    return {
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "folds": n_splits,
    }


def train_and_save(X: pd.DataFrame, y: pd.Series, out_path: str | Path,
                   random_state: int = 42) -> dict:
    """Fit on all data, save with joblib, return walk-forward score."""
    score = walk_forward_score(X, y, random_state=random_state)
    pipe = build_pipeline(random_state)
    pipe.fit(X, y)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    score["model_path"] = str(out_path)
    return score


def load(path: str | Path):
    return joblib.load(path)
