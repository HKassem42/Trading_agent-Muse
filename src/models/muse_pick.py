"""Muse's Pick challenger -- LAB ONLY (Kass's order, 2026-10-26).

Kass: "go through [the RL/ML doc], pick something and create your
own next challenger -- a stacker and the brain for the Challenger.
Make sure to label it that it's just your pic." The doc is his
Drive spreadsheet "Machine Learning and Reinforcement Learning
Kinds, Definitions, and Use Cases" (Table 1). The pick, drawn from
its rows:

- Supervised Learning (ML Paradigm) -- the stacker: the production
  recipe (StandardScaler -> SMOTE -> RF + XGBoost + LightGBM ->
  LogisticRegression) on Champion #2's scored-102 frame, with ONE
  added base learner, ExtraTrees, for ensemble diversity.
- Reinforcement Learning (RL Paradigm) + Reward Shaping (RL
  Paradigm/Technique: "augments sparse rewards with additional
  intermediate signals to guide the learning process") via Stable
  Baselines3 (RL Library) -- the brain: an SB3 PPO trained exactly
  like the scored-eyes study's 60k arm (fresh brain, 98-wide
  scored state, the stacker's votes cross-fitted in training), but
  consuming THIS pick stacker's votes and with a shaped reward:

      shaped step reward = standard dense reward
                           - 0.5 x (the step's increase in the
                             current episode's drawdown, in
                             percentage points, floored at 0)

  The standard dense reward is TradingEnv's per-step mark-to-market
  log return in percentage points, clipped to [-1, 1]; the standard
  terminal +1/-1 on the episode's P&L sign and the standard
  activity penalty ride along unchanged, so reward shaping is the
  ONLY varied lever. Rationale: the promotion gate already grades
  score = P&L - 0.5 x max drawdown, yet every prior PPO run trained
  on raw mark-to-market alone -- this challenger trains on (a
  per-step decomposition of) the objective it is graded by.

The shaping lives in ShapedTradingEnv, a TradingEnv SUBCLASS used
only by the study builder (which swaps it into ppo_train's env
slot for the study's train call only). The production env, the
nightly retrain, and both seats are never touched. A study win
installs nothing (the promotion lock, 2026-10-25): it is recorded
pending_approval for Kass.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from environment.trading_env import TradingEnv

FAMILY = "muse_pick"
SHAPING_COEF = 0.5


# ------------------------------------------------------------- stacker
def build_pick_pipeline(random_state: int = 42) -> ImbPipeline:
    """The production stacker recipe (models/stacker.build_pipeline)
    with ONE added base learner: ExtraTrees, mirroring the
    RandomForest's hyperparameters (200 trees, depth 8) -- same
    scaler, same SMOTE, same LogisticRegression meta-learner, same
    internal 3-fold stacking CV. The single varied lever on the
    stacker side is ensemble diversity."""
    base = [
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


def walk_forward_score_pick(X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5,
                            random_state: int = 42) -> dict:
    """stacker.walk_forward_score with the pick pipeline: TimeSeriesSplit
    CV, mean accuracy/precision across folds."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    accs, precs = [], []
    for train_idx, test_idx in tss.split(X):
        pipe = build_pick_pipeline(random_state)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], pred))
        precs.append(precision_score(y.iloc[test_idx], pred,
                                     zero_division=0))
    return {"accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "folds": n_splits}


def train_pick_and_save(X: pd.DataFrame, y: pd.Series,
                        out_path: str | Path,
                        random_state: int = 42) -> dict:
    """stacker.train_and_save with the pick pipeline: walk-forward
    score, fit on all data, save with joblib."""
    score = walk_forward_score_pick(X, y, random_state=random_state)
    pipe = build_pick_pipeline(random_state)
    pipe.fit(X, y)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    score["model_path"] = str(out_path)
    return score


def crossfit_p_up_pick(X: pd.DataFrame, y: pd.Series,
                       folds: int = 4) -> np.ndarray:
    """nightly.crossfit_p_up with the pick pipeline: out-of-fold
    votes from pick stackers fit ONLY on earlier data (expanding
    TimeSeriesSplit windows); the leading segment (no past to fit
    on) gets the neutral 0.5 -- the production anti-leak pattern,
    so the PPO's training votes carry the same distribution as the
    fitted pick stacker's out-of-sample votes in grading."""
    p_up = np.full(len(X), 0.5)
    tss = TimeSeriesSplit(n_splits=folds)
    for tr_idx, te_idx in tss.split(X):
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p_up[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:, 1]
    return p_up


# ------------------------------------------------------------- shaping
def drawdown_increases(equities) -> list[float]:
    """Per-step increases in drawdown, in percentage points, for an
    episode equity path whose FIRST element is the episode-start
    equity: returns one value per subsequent step, each
    max(0, dd_t - dd_{t-1}) with dd measured against the running
    peak. A step that recovers part of a drawdown (or sets a new
    peak) contributes 0 -- each step is charged only for the
    drawdown it newly creates relative to the previous step, so a
    partial recovery re-arms the charge for a later fresh leg
    down. Along any single uninterrupted descent the increases
    telescope to that descent's total drawdown."""
    eq = np.asarray(equities, dtype=float)
    if len(eq) < 2:
        return []
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, 100.0 * (peak - eq) / np.maximum(peak, 1e-9),
                  0.0)
    inc = np.diff(dd)
    return [float(max(0.0, v)) for v in inc]


def shaping_penalties(equities, coef: float = SHAPING_COEF) -> list[float]:
    """The per-step shaping penalties for an equity path:
    coef x drawdown increase, one per step after the first."""
    return [coef * v for v in drawdown_increases(equities)]


class ShapedTradingEnv(TradingEnv):
    """TradingEnv with the pre-registered reward shaping (study
    use only). Everything -- state, fills, broker, terminal +1/-1,
    activity penalty -- is the production env's; the subclass only
    subtracts SHAPING_COEF x the step's drawdown increase from the
    step reward, tracking the episode's running peak equity and
    current drawdown across steps. `shaping_log` records the exact
    penalty applied at each step of the current episode (audit)."""

    SHAPING_COEF = SHAPING_COEF

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # super().reset set _prev_eq to the episode-start equity.
        self._peak_eq = float(self._prev_eq)
        self._cur_dd = 0.0
        self.shaping_log: list[float] = []
        return obs, info

    def step(self, action: int):
        obs, reward, done, truncated, info = super().step(action)
        eq = float(self._prev_eq)  # post-step equity (super updated it)
        if eq > self._peak_eq:
            self._peak_eq = eq
        peak = max(self._peak_eq, 1e-9)
        dd = 100.0 * (self._peak_eq - eq) / peak if self._peak_eq > 0 else 0.0
        inc = max(0.0, dd - self._cur_dd)
        self._cur_dd = dd
        penalty = self.SHAPING_COEF * inc
        self.shaping_log.append(penalty)
        return obs, reward - penalty, done, truncated, info
