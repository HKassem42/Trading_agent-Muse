"""Muse's Pick #7 — "Slow Clock" — LAB ONLY (Kass's order,
2026-11-05: "try muses pick the next one... Pick whatever u
want"). Pre-registered in
refs/muse-pick-7-preregistration-2026-11-05.md.

The lever: the derivatives-positioning columns that failed
against next-1h direction (runs 101/102) are slow variables;
Pick #7 grades them at a 24h horizon on the same causal
120-column frame. This module holds ONLY the split/scoring
machinery whose correctness the study stands on — positional,
purge-aware, and pinned by tests/test_muse_pick7.py:

- exam_split: the train/exam partition with a horizon-row purge
  at the boundary, so no train label reads a close inside the
  exam period;
- purged_wf_splits: TimeSeriesSplit folds with a horizon-row
  embargo on both sides of every fold boundary, so no fold's
  24h labels straddle the boundary;
- anatomy / is_clean: the run-63 degenerate-anatomy rule.

Labels themselves come from models.muse_picks456.horizon_labels
(label-only by construction). Production never imports this.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

HORIZON = 24


def exam_split(index, y, holdout_pos: int, horizon: int = HORIZON):
    """(train_pos, exam_pos) positional split with boundary purge.

    exam_pos: positions >= holdout_pos whose label is defined.
    train_pos: positions p with a defined label and
    p + horizon < holdout_pos — the purge drops exactly the
    `horizon` rows whose labels would read closes at or after
    the first exam position.
    """
    yv = np.asarray(y, dtype=float)
    finite = np.isfinite(yv)
    pos = np.arange(len(yv))
    exam_pos = pos[(pos >= holdout_pos) & finite]
    train_pos = pos[finite & (pos + horizon < holdout_pos)]
    return train_pos, exam_pos


def purged_wf_splits(n: int, horizon: int = HORIZON,
                     n_splits: int = 3):
    """TimeSeriesSplit folds with a horizon embargo: each fold's
    train part loses its last `horizon` rows and its validation
    part loses its first `horizon` rows, so
    max(train) + horizon < min(validation) holds for every fold.
    """
    folds = []
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tss.split(np.zeros((n, 1))):
        tr = tr[: max(0, len(tr) - horizon)]
        va = va[horizon:]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def anatomy(proba, y, thr: float = 0.5) -> dict:
    """Accuracy / precision / up-calls of a probability vector at
    a fixed threshold (0.50 — no threshold tuning, ever)."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = (proba >= thr).astype(int)
    return {
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred,
                                           zero_division=0)),
        "up_calls": int(pred.sum()),
    }


def is_clean(m: dict) -> bool:
    """The run-63 rule: a win needs real two-sided anatomy —
    at least 8 up-calls and precision above 0.05."""
    return bool(m["up_calls"] >= 8 and m["precision"] > 0.05)
