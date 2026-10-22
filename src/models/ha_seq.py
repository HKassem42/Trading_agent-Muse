"""HA-sequence challenger family (v1, 2026-10-22): a compact CNN+LSTM
classifier that reads the last 90 hourly candles as Heikin-Ashi shapes
and outputs P(next candle closes up). NO hand-crafted TA features: the
per-candle inputs are derived only from HA-OHLC and volume, expressed
relatively so they are stationary, and every standardization statistic
is fit on the TRAIN split only.

Causality contract (pinned by tests/test_ha_seq.py):
- The HA recursion at row t uses raw candles <= t only, seeded from the
  first candle of the frame -- so recomputing on a truncated frame
  reproduces identical values on the shared rows.
- Per-candle channels at row t use raw/HA values at rows <= t only
  (the volume channel is a trailing 24h mean, current candle included).
- Standardization statistics are fit on rows [0, upto) of the TRAIN
  region and applied unchanged everywhere else: mutating holdout rows
  cannot change any train sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_LEN = 90
VOL_WINDOW = 24

# Per-candle channels (order is the model's input contract):
#   ha_ret      log HA-close return (stationary price motion)
#   body_ratio  signed HA body as a share of the HA range, [-1, 1]
#   upper_wick  upper wick share of the HA range, [0, 1]
#   lower_wick  lower wick share of the HA range, [0, 1]
#   range_rel   HA range relative to HA close (>= 0)
#   vol_log_rel log(volume / trailing-24h mean volume)
CHANNELS = ["ha_ret", "body_ratio", "upper_wick", "lower_wick",
            "range_rel", "vol_log_rel"]


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi OHLC from raw OHLC. Causal recursion:
    hc = (o+h+l+c)/4; ho[0] = (o0+c0)/2; ho[t] = (ho[t-1]+hc[t-1])/2;
    hh = max(h, ho, hc); hl = min(l, ho, hc)."""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    hc = (o + h + lo + c) / 4.0
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2.0
    for t in range(1, len(hc)):
        ho[t] = (ho[t - 1] + hc[t - 1]) / 2.0
    hh = np.maximum(h, np.maximum(ho, hc))
    hl = np.minimum(lo, np.minimum(ho, hc))
    return pd.DataFrame({"ha_open": ho, "ha_high": hh, "ha_low": hl,
                         "ha_close": hc}, index=df.index)


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-candle stationary channels from HA-OHLC + volume.

    Row t depends only on raw candles <= t. First row has NaN ha_ret;
    the first VOL_WINDOW-1 rows have NaN vol_log_rel (full trailing
    window required). Zero-range candles (ho==hc==hh==hl) get ratio
    channels 0 -- they carry no shape."""
    ha = heikin_ashi(df)
    ho, hh, hl, hc = ha["ha_open"], ha["ha_high"], ha["ha_low"], ha["ha_close"]
    rng = (hh - hl).to_numpy(dtype=float)
    denom = np.where(rng > 0, rng, 1.0)
    body = (hc - ho).to_numpy(dtype=float) / denom
    upper = (hh - np.maximum(ho, hc)).to_numpy(dtype=float) / denom
    lower = (np.minimum(ho, hc) - hl).to_numpy(dtype=float) / denom
    ha_ret = np.log(hc / hc.shift(1))
    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    vol_rel = np.log(np.maximum(vol, 1e-12)
                     / np.maximum(vol_mean, 1e-12))
    out = pd.DataFrame({
        "ha_ret": ha_ret,
        "body_ratio": body,
        "upper_wick": upper,
        "lower_wick": lower,
        "range_rel": rng / hc.to_numpy(dtype=float),
        "vol_log_rel": vol_rel,
    }, index=df.index)
    return out[CHANNELS]


def fit_stats(feat: pd.DataFrame, upto: int) -> dict:
    """Channel mean/std (population, ddof=0, NaN-skipping) over the
    positional rows [0, upto) ONLY -- the train region. Anything fit on
    a wider frame leaks the holdout into every sequence (see test (c))."""
    sub = feat.iloc[:upto]
    mean, std = {}, {}
    for c in CHANNELS:
        vals = sub[c].to_numpy(dtype=float)
        mean[c] = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        std[c] = s if s > 1e-12 else 1.0
    return {"mean": mean, "std": std, "upto": int(upto)}


def _standardize(feat: pd.DataFrame, stats: dict) -> np.ndarray:
    arr = feat[CHANNELS].to_numpy(dtype=float)
    mu = np.array([stats["mean"][c] for c in CHANNELS])
    sd = np.array([stats["std"][c] for c in CHANNELS])
    return (arr - mu) / sd


def build_sequence(feat: pd.DataFrame, stats: dict, end_row: int,
                   seq_len: int = SEQ_LEN) -> np.ndarray | None:
    """Standardized (seq_len, C) float32 sequence ENDING at positional
    row `end_row` (inclusive), or None when the window is short or
    contains a NaN. Pure function of rows <= end_row plus `stats`."""
    start = end_row - seq_len + 1
    if start < 0:
        return None
    window = feat.iloc[start:end_row + 1]
    if len(window) < seq_len or window[CHANNELS].isna().any().any():
        return None
    return _standardize(window, stats).astype(np.float32)


def build_all_sequences(feat: pd.DataFrame, stats: dict,
                        seq_len: int = SEQ_LEN):
    """Vectorized build_sequence over every row: returns (X, valid)
    with X[r] the sequence ending at row r (zero-filled when invalid)
    and valid[r] marking rows that have a full, NaN-free window."""
    from numpy.lib.stride_tricks import sliding_window_view
    z = _standardize(feat, stats)
    n, c = z.shape
    X = np.zeros((n, seq_len, c), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n >= seq_len:
        win = sliding_window_view(z, window_shape=seq_len, axis=0)
        win = np.transpose(win, (0, 2, 1))  # (start, seq, C)
        ok = ~np.isnan(win).any(axis=(1, 2))
        ends = np.arange(seq_len - 1, n)
        good = ends[ok]
        X[good] = win[ok].astype(np.float32)
        valid[good] = True
    return X, valid


# ---------------------------------------------------------------- model
def build_model(seq_len: int = SEQ_LEN, n_channels: int = len(CHANNELS)):
    """Compact CNN -> LSTM -> dense -> logit. ~129k params (CPU box).

    Conv front-end (kernel 7 then 5, maxpool 2x twice: 90 -> 45 -> 22
    steps) reads local candle-shape motifs; the LSTM integrates the
    compressed sequence; dropout on the head only."""
    import torch
    from torch import nn

    class HaSeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 48),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(48, 1),
            )

        def forward(self, x):  # x: (batch, seq, C)
            z = self.conv(x.transpose(1, 2))       # (batch, 64, steps)
            z = z.transpose(1, 2)                 # (batch, steps, 64)
            out, _ = self.lstm(z)
            return self.head(out[:, -1, :]).squeeze(-1)  # (batch,) logits

    return HaSeqNet()


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
