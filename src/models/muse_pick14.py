"""Muse's Pick #14 — "The Committee" — LAB ONLY (Kass's
dealer's-choice order, 2026-11-18).

Pre-registered in refs/muse-pick-14-preregistration-2026-11-18.md
BEFORE any candidate was fitted.

The ONE change vs the championship line (Picks #11+ / Champions
#6-#13): every one of those stackers is the flat Pick recipe
(models.muse_pick.build_pick_pipeline) at a SINGLE seed (42) on
the ha10 frame, and nightly challengers of that recipe win or
lose their exams by 2-4 candles -- seed noise at the margin.
Pick #14 runs the SAME recipe on the SAME ha10 matrix
(models.muse_pick10.build_ha_matrix, unchanged) at SEVEN fixed
seeds and averages the seeds' meta-probabilities. No new
features, no new hyperparameters, no new look-ahead surface:
the committee is a variance reduction over the recipe's own
randomness, nothing else.

    SEEDS = [42, 7, 123, 2024, 555, 9001, 31337]

Cross-fitting mirrors training.nightly.crossfit_p_up exactly
(the production anti-leak pattern the PPO phase consumes):
TimeSeriesSplit expanding windows, each fold's votes from a
committee fit ONLY on earlier rows, the leading segment (no
past to fit on) gets the neutral 0.5. A seed's model never sees
its vote rows in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models.muse_pick import build_pick_pipeline

SEEDS = [42, 7, 123, 2024, 555, 9001, 31337]
N_SEEDS = len(SEEDS)


class CommitteeStacker:
    """The 7-seed Pick-recipe committee: a plain average of the
    member pipelines' P(up). Deliberately a thin wrapper -- each
    member is the unmodified production Pick pipeline at one
    seed, so a member's vote means exactly what the single-seed
    championship recipe's vote means; the committee's vote is
    their mean. Picklable (joblib) so the fitted committee can
    be saved as the study artifact and re-loaded for grading."""

    def __init__(self, pipelines, seeds=None,
                 feature_columns=None):
        self.pipelines = list(pipelines)
        self.seeds = list(seeds) if seeds is not None else list(SEEDS)
        self.feature_columns_ = (list(feature_columns)
                                 if feature_columns is not None
                                 else None)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        cols = [pipe.predict_proba(X)[:, 1] for pipe in self.pipelines]
        p_up = np.mean(cols, axis=0)
        return np.column_stack([1.0 - p_up, p_up])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_seed_pipeline(X: pd.DataFrame, y, seed: int):
    """One committee member: the full Pick recipe at `seed`,
    fit on exactly the rows given (the caller owns the split)."""
    pipe = build_pick_pipeline(random_state=int(seed))
    pipe.fit(X, y)
    return pipe


def fit_committee(X: pd.DataFrame, y, seeds=SEEDS) -> CommitteeStacker:
    """Fit every seed's pipeline on (X, y) and bundle them as a
    CommitteeStacker. Deterministic in `seeds`: the same seeds
    on the same rows reproduce the same averaged probabilities."""
    pipes = [fit_seed_pipeline(X, y, s) for s in seeds]
    return CommitteeStacker(pipes, seeds=list(seeds),
                            feature_columns=list(X.columns))


def member_probas(committee: CommitteeStacker, X) -> np.ndarray:
    """Per-seed P(up) matrix, shape (n_seeds, n_rows) -- the
    committee's average is its row mean; kept separate so the
    study can report each seed's own exam accuracy (the noise
    band the committee averages over)."""
    return np.vstack([pipe.predict_proba(X)[:, 1]
                      for pipe in committee.pipelines])


def crossfit_committee_p_up(X: pd.DataFrame, y, folds: int = 4,
                            seeds=SEEDS) -> np.ndarray:
    """Out-of-fold committee votes for a training frame --
    nightly.crossfit_p_up with the committee as the pipeline
    factory. Each vote comes from a committee fit ONLY on
    earlier data (time-series expanding windows, no future
    leakage); the leading segment (no past to fit on) gets the
    neutral 0.5, exactly the production pattern, so the PPO's
    training votes carry the same distribution as the fitted
    committee's out-of-sample votes in grading."""
    p_up = np.full(len(X), 0.5)
    tss = TimeSeriesSplit(n_splits=folds)
    for tr_idx, te_idx in tss.split(X):
        committee = fit_committee(X.iloc[tr_idx],
                                  np.asarray(y)[tr_idx], seeds=seeds)
        p_up[te_idx] = committee.predict_proba(X.iloc[te_idx])[:, 1]
    return p_up
