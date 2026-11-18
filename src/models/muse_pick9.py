"""Muse's Pick #9 — LSTM sequence stacker machinery.

Pre-registered in refs/muse-pick-9-preregistration-2026-11-05.md.
The candidate reads the ordered last-WINDOW rows of the 1h
feature matrix (the RL/ML combinations document's LSTM entry:
"time-windowed matrices, e.g. 24-hour lookbacks") instead of the
single-candle row every previous stacker consumed.

Contracts (pinned by tests/test_muse_pick9.py):
- the sequence for supervised row t is X[t-WINDOW+1 .. t] —
  past and present only, never a future row;
- the scaler is fit on train rows only;
- the candidate probability is the mean of SEEDS fixed-seed
  LSTM fits (declared seed-ensembling).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

WINDOW = 24
SEEDS = (42, 7, 123)
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 40
PATIENCE = 5
VAL_TAIL = 792


def sequence_rows(n: int, window: int = WINDOW) -> list[int]:
    """Supervised rows that own a full window: window-1 .. n-1."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return list(range(window - 1, n))


def make_sequences(X: np.ndarray, window: int = WINDOW):
    """Return (seqs, rows): seqs[k] = X[rows[k]-window+1 ..
    rows[k]+1] with rows = sequence_rows(len(X), window)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    X = np.asarray(X, dtype=np.float32)
    rows = sequence_rows(len(X), window)
    if not rows:
        return (np.empty((0, window, X.shape[1]), dtype=np.float32),
                np.array([], dtype=int))
    idx = (np.arange(window)[None, :]
           + np.array(rows)[:, None] - (window - 1))
    return X[idx], np.array(rows, dtype=int)


def fit_scaler(X_train: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def transform(X: np.ndarray, mean, std) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


class _LSTMNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.lstm = nn.LSTM(d_in, HIDDEN, num_layers=1,
                            batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def _fit_one(seqs: np.ndarray, y: np.ndarray, seed: int,
             val_seqs: np.ndarray, val_y: np.ndarray) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = seqs.shape[2]
    model = _LSTMNet(d_in)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pw = torch.tensor(neg / max(pos, 1.0))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.from_numpy(seqs)
    Y = torch.from_numpy(y.astype(np.float32))
    XV = torch.from_numpy(val_seqs)
    YV = torch.from_numpy(val_y.astype(np.float32))
    best_state, best_loss, bad = None, float("inf"), 0
    n = len(Y)
    for _epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(X[j])
            loss = lossf(out, Y[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(XV), YV))
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model: nn.Module, seqs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = []
        X = torch.from_numpy(seqs.astype(np.float32))
        for i in range(0, len(X), 4096):
            out.append(torch.sigmoid(model(X[i:i + 4096])).numpy())
    return np.concatenate(out) if out else np.array([])


def fit_predict_ensemble(X_flat: np.ndarray, y: np.ndarray,
                         train_rows: np.ndarray,
                         pred_rows: np.ndarray,
                         window: int = WINDOW) -> np.ndarray:
    """Fit the seed ensemble on train_rows of the flat supervised
    matrix and return mean P(up) for pred_rows. Scaler and
    sequences derive from train_rows only (plus each row's own
    causal window, which may reach back before the span — past
    candles only). Validation tail = the last VAL_TAIL train
    rows, used for early stopping only."""
    mean, std = fit_scaler(X_flat[train_rows])
    Z = transform(X_flat, mean, std)
    seqs_all, rows_all = make_sequences(Z, window)
    pos_of = {int(r): k for k, r in enumerate(rows_all)}
    tr = np.array([r for r in train_rows if int(r) in pos_of],
                  dtype=int)
    assert len(tr) >= VAL_TAIL + 100, "train span too small"
    fit_rows, val_rows = tr[:-VAL_TAIL], tr[-VAL_TAIL:]
    fit_seqs = seqs_all[[pos_of[int(r)] for r in fit_rows]]
    val_seqs = seqs_all[[pos_of[int(r)] for r in val_rows]]
    fit_y = y[fit_rows]
    val_y = y[val_rows]
    pred_seqs = seqs_all[[pos_of[int(r)] for r in pred_rows]]
    probas = []
    for seed in SEEDS:
        model = _fit_one(fit_seqs, fit_y, seed, val_seqs, val_y)
        probas.append(_predict(model, pred_seqs))
    return np.mean(probas, axis=0)
