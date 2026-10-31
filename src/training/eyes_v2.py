"""Eyes v2 — the tradeable-move stacker (run 85, LAB ONLY).

Kass's order (2026-10-30): "What can we change to make the
eyes better — do whatever and change whatever to get us
results." Design frozen in refs/eyes-v2-study-2026-10-30.md
BEFORE any grading: the eyes stop answering "up or down next
candle?" (a coin flip across fourteen nulls) and answer
"which way does the next TRADEABLE move go?" — a
triple-barrier label (1.0 x ATR(14), 24-candle horizon, first
touch wins, neither-touch falls back to the 24-candle net
move) over the pick2_98 frame plus eight causal context
features. The zoo: (a) run 73's pick recipe on the new
labels, (b) run 82's LightGBM on the new labels, (c) the pick
recipe on the OLD labels with the new features (ablation),
(d) Kass's arm — a compact LSTM over the last 24 candles of
the frame (flattened-lag LightGBM fallback if the neural arm
cannot train). Selection is training-region OOF only; the
winner faces two frozen bars (the triple-barrier exam vs run
73 re-scored, and the standard duel paired with PPO run 44).

Production never imports this module. Labels read future
prices for the OUTCOME only (the meta-labeling convention);
every feature and every sequence reads candles <= t only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FAMILY = "eyes_v2"
LABEL = "Tradeable-Move Stacker"
HORIZON = 24
SEQ_LEN = 24
BARRIER_MULT = 1.0
N_PICK2 = 98
CONTEXT_COLUMNS = ["ctx_rv24", "ctx_rv96", "ctx_rv_ratio",
                   "ctx_ret4", "ctx_ret24", "ctx_dd100",
                   "ctx_hour_sin", "ctx_hour_cos"]

ROOT = Path(__file__).resolve().parents[2]

# Guard set (run 84's, extended by nothing — Book #3's gated
# files included): strict entries must be byte-identical
# before/after the study; the paper books are information-only
# because their own loops write them on candle closes.
GUARD_FILES = {
    "live_seat_ppo": ROOT / "models" / "live_seat" / "ppo.zip",
    "live_seat_stacker": ROOT / "models" / "live_seat" / "stacker.pkl",
    "live_seat_meta": ROOT / "models" / "live_seat" / "stacker.json",
    "shadow_seat_ppo": ROOT / "models" / "shadow_seat" / "ppo.zip",
    "shadow_seat_stacker": ROOT / "models" / "shadow_seat" / "stacker.pkl",
    "shadow_seat_meta": ROOT / "models" / "shadow_seat" / "stacker.json",
    "gated_seat_ppo": ROOT / "models" / "gated_seat" / "ppo.zip",
    "gated_seat_stacker": ROOT / "models" / "gated_seat" / "stacker.pkl",
    "gated_seat_meta": ROOT / "models" / "gated_seat" / "stacker.json",
    "gated_seat_gate_model": ROOT / "models" / "gated_seat" / "gate_model.pkl",
    "gated_seat_manifest": ROOT / "data" / "gated_seat.json",
    "lab_ppo_champion": ROOT / "models" / "ppo_champion.zip",
    "lab_stacker_champion": ROOT / "models" / "stacker_champion.pkl",
    "lab_stacker_meta": ROOT / "models" / "stacker_champion.json",
    "live_state": ROOT / "data" / "live_state.json",
    "live_state_shadow": ROOT / "data" / "live_state_shadow.json",
    "live_state_gated": ROOT / "data" / "live_state_gated.json",
    "paper_db_info_only": ROOT / "data" / "paper.db",
    "paper_shadow_db_info_only": ROOT / "data" / "paper_shadow.db",
    "paper_gated_db_info_only": ROOT / "data" / "paper_gated.db",
}


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def guard_hashes() -> dict:
    return {k: _sha(p) for k, p in GUARD_FILES.items()}


# ---------------------------------------------------------- context
def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """The eight causal context features over a raw OHLCV frame,
    indexed exactly like `df`. Every value at row t reads only
    candles <= t (trailing windows END at t)."""
    close = df["close"].astype(float)
    logret = np.log(close / close.shift(1))
    rv24 = logret.rolling(24).std()
    rv96 = logret.rolling(96).std()
    out = pd.DataFrame(index=df.index)
    out["ctx_rv24"] = rv24
    out["ctx_rv96"] = rv96
    out["ctx_rv_ratio"] = (rv24 / rv96.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan)
    out["ctx_ret4"] = close / close.shift(4) - 1.0
    out["ctx_ret24"] = close / close.shift(24) - 1.0
    out["ctx_dd100"] = close / close.rolling(100).max() - 1.0
    hour = df.index.hour.to_numpy(dtype=float)
    out["ctx_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["ctx_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return out[CONTEXT_COLUMNS]


# ---------------------------------------------------------- labels
def triple_barrier_labels(open_, high, low, close, atr,
                          horizon: int = HORIZON,
                          mult: float = BARRIER_MULT):
    """Triple-barrier tradeable-move labels (frozen design).

    At decision position t: U = close[t] + mult*atr[t],
    D = close[t] - mult*atr[t]. Scan j = t+1 .. t+horizon on
    HIGH/LOW: first touch decides (1 = up, 0 = down); a candle
    spanning BOTH barriers resolves by its own direction
    (close >= open -> up first). No touch: the horizon net
    move decides (close[t+horizon] vs close[t]); an exact tie
    drops the row. Rows without a full horizon and no touch,
    and rows with a non-positive/non-finite ATR, carry -1.

    Returns (labels int8 with -1 = unlabeled, stats dict).
    Every row is counted exactly once in the stats."""
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr, dtype=float)
    n = len(c)
    labels = np.full(n, -1, dtype=np.int8)
    stats = {"up_first": 0, "down_first": 0, "neither_up": 0,
             "neither_down": 0, "tie_dropped": 0, "no_label": 0,
             "no_atr": 0}
    for t in range(n):
        b = a[t] * mult if np.isfinite(a[t]) else np.nan
        if not np.isfinite(b) or b <= 0.0:
            stats["no_atr"] += 1
            continue
        up, dn = c[t] + b, c[t] - b
        lab = None
        last = min(t + horizon, n - 1)
        for j in range(t + 1, last + 1):
            hit_up = h[j] >= up
            hit_dn = lo[j] <= dn
            if hit_up and hit_dn:
                lab = 1 if c[j] >= o[j] else 0
                break
            if hit_up:
                lab = 1
                break
            if hit_dn:
                lab = 0
                break
        if lab is not None:
            labels[t] = lab
            stats["up_first" if lab == 1 else "down_first"] += 1
            continue
        if t + horizon <= n - 1:
            net = c[t + horizon] - c[t]
            if net > 0:
                labels[t] = 1
                stats["neither_up"] += 1
            elif net < 0:
                labels[t] = 0
                stats["neither_down"] += 1
            else:
                stats["tie_dropped"] += 1
        else:
            stats["no_label"] += 1
    return labels, stats


# ---------------------------------------------------------- frames
def build_feature_frame(df: pd.DataFrame, sup: pd.DataFrame,
                        champ_columns: list[str]):
    """(frame106, columns, evidence): the pick2_98 frame (the
    champion's 98 columns in recorded order) reindexed to the
    supervised rows, joined with the 8 context features aligned
    to the same timestamps. NaN outside the scored region —
    row-set selection handles finiteness."""
    from features import pick2_features

    frame98, _cols, ev = pick2_features.build_pick2_frame(df)
    core = frame98.reindex(sup.index)[champ_columns]
    ctx = build_context_features(df).reindex(sup.index)
    full = np.hstack([core.to_numpy(dtype=float),
                      ctx[CONTEXT_COLUMNS].to_numpy(dtype=float)])
    columns = list(champ_columns) + list(CONTEXT_COLUMNS)
    evidence = {"pick2": ev,
                "finite_rows_full_series": int(
                    np.isfinite(full).all(axis=1).sum()),
                "rows": int(len(full))}
    return full, columns, evidence


def common_training_rows(frame: np.ndarray, labels: np.ndarray,
                         split: int) -> np.ndarray:
    """The zoo's shared training row set (frozen): positions t
    in the training region with a defined label, all 106
    features finite at t AND over the trailing 24-row window
    t-23..t, and t + 24 < split (purge — the label horizon
    lies entirely inside the training region)."""
    frame = np.asarray(frame, dtype=float)
    labels = np.asarray(labels)
    n = len(frame)
    finite_row = np.isfinite(frame).all(axis=1)
    csum = np.concatenate([[0], np.cumsum(finite_row.astype(int))])
    rows = []
    for t in range(split):
        if labels[t] < 0 or t + HORIZON >= split or t < SEQ_LEN - 1:
            continue
        if not finite_row[t]:
            continue
        if csum[t + 1] - csum[t + 1 - SEQ_LEN] != SEQ_LEN:
            continue
        rows.append(t)
    assert len(rows) <= n
    return np.array(rows, dtype=int)


def exam_positions(labels: np.ndarray, split: int,
                   n_exam: int = 768) -> np.ndarray:
    """The exam rows: the first n_exam holdout positions whose
    triple-barrier label is defined (identical timestamps for
    both eyes)."""
    labels = np.asarray(labels)
    return np.array([t for t in range(split, split + n_exam)
                     if labels[t] >= 0], dtype=int)


def sequence_tensor(frame: np.ndarray, positions: np.ndarray,
                    seq_len: int = SEQ_LEN) -> np.ndarray:
    """(n, seq_len, F): the frame rows t-seq_len+1 .. t for each
    position t — every sequence ENDS at its decision row."""
    frame = np.asarray(frame, dtype=float)
    positions = np.asarray(positions, dtype=int)
    return np.stack([frame[t - seq_len + 1:t + 1] for t in positions])


def lag_matrix(frame: np.ndarray, positions: np.ndarray,
               col_idx: list[int], seq_len: int = SEQ_LEN) -> np.ndarray:
    """The flattened-lag fallback inputs: for each position,
    the chosen columns at lags 0..seq_len-1, lag-major (all
    columns at lag 0, then lag 1, ...)."""
    frame = np.asarray(frame, dtype=float)
    positions = np.asarray(positions, dtype=int)
    blocks = [frame[positions - lag][:, list(col_idx)]
              for lag in range(seq_len)]
    return np.hstack(blocks)


# ---------------------------------------------------------- metrics
def vote_spread(votes: np.ndarray) -> dict:
    """How strongly an eye expresses views over a vote set."""
    v = np.asarray(votes, dtype=float)
    return {
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()),
        "pct_out_45_55": float(np.mean((v < 0.45) | (v > 0.55))),
        "pct_out_40_60": float(np.mean((v < 0.40) | (v > 0.60))),
        "n": int(len(v)),
    }


def exam_metrics(votes: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy / precision / up-calls at the 0.5 cut."""
    v = np.asarray(votes, dtype=float)
    y = np.asarray(labels, dtype=int)
    pred = (v >= 0.5).astype(int)
    up = int(pred.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "accuracy": float(np.mean(pred == y)),
        "precision": float(tp / up) if up else 0.0,
        "up_calls": up,
        "n": int(len(y)),
        "base_rate": float(np.mean(y == 1)),
    }


def select_arm(oof: dict) -> str:
    """The frozen zoo selection rule: highest OOF balanced
    accuracy; ties (|delta| < 1e-12) broken by higher AUC,
    then by arm order a, b, c, d for determinism."""
    best_ba = max(v["balanced_accuracy"] for v in oof.values())
    cands = [k for k, v in oof.items()
             if abs(v["balanced_accuracy"] - best_ba) < 1e-12]
    if len(cands) == 1:
        return cands[0]
    best_auc = max(oof[k]["auc"] for k in cands)
    cands = [k for k in cands if abs(oof[k]["auc"] - best_auc) < 1e-12]
    return sorted(cands)[0]


def pick2_importance_ranking(fitted_pipeline) -> np.ndarray:
    """Mean normalized impurity importance across the fitted
    pick stack's base learners (RF / XGB / LGBM / ET), aligned
    to the model's input columns. Used only by the (d')
    flattened-lag fallback to rank the pick2 columns."""
    stack = fitted_pipeline.named_steps["stack"]
    imps = []
    for est in stack.estimators_:
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            continue
        imp = np.asarray(imp, dtype=float)
        s = imp.sum()
        if s > 0:
            imps.append(imp / s)
    if not imps:
        return np.zeros(len(stack.estimators_[0].feature_importances_)
                        if hasattr(stack.estimators_[0],
                                   "feature_importances_") else 0)
    return np.mean(imps, axis=0)


# ---------------------------------------------------------- LSTM arm
class _LSTMNet:
    """Lazy torch module factory (keeps torch out of import
    time for the rest of the module)."""

    @staticmethod
    def build(n_features: int, hidden: int = 48, dropout: float = 0.3):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=1,
                                    batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :])).squeeze(-1)

        return Net()


class LSTMEyes:
    """Kass's sequence arm: a compact LSTM over the last
    SEQ_LEN candles of the study frame, ending AT the decision
    candle. Standardization statistics live on the object and
    come from the fit subset only."""

    HIDDEN = 48
    DROPOUT = 0.3

    def __init__(self):
        self.net = None
        self.mean_ = None
        self.std_ = None
        self.param_count = 0
        self.history: dict = {}
        self.epochs_run = 0

    # -- internals ------------------------------------------------
    def _standardize(self, seqs: np.ndarray) -> np.ndarray:
        return (seqs - self.mean_) / self.std_

    def fit(self, seqs: np.ndarray, y: np.ndarray, seed: int = 42,
            max_epochs: int = 40, patience: int = 6,
            val_frac: float = 0.15, purge: int = SEQ_LEN,
            batch_size: int = 256, lr: float = 1e-3,
            weight_decay: float = 1e-2) -> "LSTMEyes":
        import copy

        import torch
        from torch import nn

        seqs = np.asarray(seqs, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        n, _seq, n_feat = seqs.shape
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        n_val = max(1, int(round(val_frac * n)))
        fit_end = n - n_val
        purge_eff = min(purge, max(0, fit_end - 64))
        fit_idx = np.arange(0, fit_end - purge_eff)
        val_idx = np.arange(fit_end, n)
        assert len(fit_idx) >= 32, "fit subset too small"

        fit_seqs = seqs[fit_idx]
        self.mean_ = fit_seqs.reshape(-1, n_feat).mean(axis=0)
        std = fit_seqs.reshape(-1, n_feat).std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        self.mean_ = self.mean_.astype(np.float32)

        X_fit = torch.tensor(self._standardize(fit_seqs))
        y_fit = torch.tensor(y[fit_idx], dtype=torch.float32)
        X_val = torch.tensor(self._standardize(seqs[val_idx]))
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)

        n_pos = float((y[fit_idx] == 1).sum())
        n_neg = float((y[fit_idx] == 0).sum())
        pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))

        self.net = _LSTMNet.build(n_feat, self.HIDDEN, self.DROPOUT)
        self.param_count = int(sum(p.numel()
                                   for p in self.net.parameters()))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr,
                                weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_fn = nn.BCEWithLogitsLoss()

        best_val, best_state, bad = float("inf"), None, 0
        val_hist: list[float] = []
        for epoch in range(max_epochs):
            self.net.train()
            perm = rng.permutation(len(fit_idx))
            for s in range(0, len(perm), batch_size):
                idx = perm[s:s + batch_size]
                opt.zero_grad()
                out = self.net(X_fit[idx])
                loss = loss_fn(out, y_fit[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = float(val_fn(self.net(X_val), y_val))
            val_hist.append(vloss)
            self.epochs_run = epoch + 1
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = copy.deepcopy(self.net.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.history = {"val_bce": val_hist, "best_val_bce": best_val,
                        "n_fit": int(len(fit_idx)),
                        "n_val": int(len(val_idx)),
                        "purge": int(purge_eff),
                        "pos_weight": float(pos_weight)}
        return self

    def predict_proba(self, seqs: np.ndarray) -> np.ndarray:
        import torch

        assert self.net is not None, "LSTMEyes is not fitted"
        x = torch.tensor(
            self._standardize(np.asarray(seqs, dtype=np.float32)))
        self.net.eval()
        with torch.no_grad():
            logits = self.net(x)
            p = torch.sigmoid(logits).numpy()
        return np.asarray(p, dtype=float).reshape(-1)


def train_lstm_eyes(seqs: np.ndarray, y: np.ndarray, seed: int = 42,
                    **kw) -> LSTMEyes:
    return LSTMEyes().fit(seqs, y, seed=seed, **kw)
