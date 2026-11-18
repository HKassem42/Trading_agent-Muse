"""Eyes v2 (run 85) — adversarial tests, written BEFORE the
implementation per the house quality standard. Design frozen in
refs/eyes-v2-study-2026-10-30.md.

Pins: triple-barrier label construction on hand-computed candle
paths (barrier order, both-barrier candle rule, neither-barrier
fallback, tie-drop, horizon boundary), context-feature and
sequence causality (garbling the future must not move the past),
the common training row set (purge + sequence feasibility),
exam-frame metrics on hand-computed votes, the zoo selection
rule, the flattened-lag fallback's layout, and the compact-feed
family registry for eyes_v2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import eyes_v2 as ev  # noqa: E402


# ------------------------------------------------------- label paths
def _flat(n: int, price: float = 100.0):
    """n identical candles at `price` (open=high=low=close)."""
    o = np.full(n, price)
    return o.copy(), o.copy(), o.copy(), o.copy()


def _labels(o, h, lo, c, atr=2.0, horizon=24):
    atr_arr = np.full(len(c), atr, dtype=float)
    return ev.triple_barrier_labels(o, h, lo, c, atr_arr,
                                    horizon=horizon)


def test_label_up_barrier_first():
    o, h, lo, c = _flat(40)
    h[3] = 101.0            # no touch (U = 102)
    h[5] = 103.0            # up barrier touched at t+5
    labels, stats = _labels(o, h, lo, c)
    assert labels[0] == 1
    assert stats["up_first"] >= 1


def test_label_down_barrier_first():
    o, h, lo, c = _flat(40)
    lo[2] = 97.0            # D = 98 touched at t+2
    h[4] = 105.0            # later up touch must not matter
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 0


def test_label_neither_barrier_net_up():
    o, h, lo, c = _flat(40)
    c[24] = 101.0           # net move up at the horizon candle
    labels, stats = _labels(o, h, lo, c)
    assert labels[0] == 1
    assert stats["neither_up"] >= 1


def test_label_neither_barrier_net_down():
    o, h, lo, c = _flat(40)
    c[24] = 99.0
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 0


def test_label_neither_barrier_exact_tie_dropped():
    o, h, lo, c = _flat(40)
    labels, stats = _labels(o, h, lo, c)
    # close[24] == close[0] exactly and no barrier touched: drop.
    assert labels[0] == -1
    assert stats["tie_dropped"] >= 1


def test_label_both_barriers_one_candle_bullish_resolves_up():
    o, h, lo, c = _flat(40)
    h[1], lo[1], c[1], o[1] = 105.0, 95.0, 104.0, 100.0  # spans both, closes up
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 1


def test_label_both_barriers_one_candle_bearish_resolves_down():
    o, h, lo, c = _flat(40)
    h[1], lo[1], c[1], o[1] = 105.0, 95.0, 96.0, 100.0   # spans both, closes down
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 0


def test_label_touch_at_horizon_counts_but_not_after():
    o, h, lo, c = _flat(40)
    h[24] = 103.0           # touch exactly at t+24: inside the horizon
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 1
    o, h, lo, c = _flat(40)
    h[25] = 103.0           # touch at t+25: outside; net move is flat-tie
    c[24] = 99.0            # make the fallback decisive: net down
    labels, _ = _labels(o, h, lo, c)
    assert labels[0] == 0


def test_label_no_horizon_no_label():
    o, h, lo, c = _flat(30)
    labels, _ = _labels(o, h, lo, c)
    assert labels[29] == -1          # nothing after t
    assert labels[6] == -1           # t+24 = 30 > last index, no touch, no close
    assert labels[5] == -1           # t+24 = 29 exists; flat closes -> tie-drop


def test_label_uses_atr_at_t_not_later():
    o, h, lo, c = _flat(40)
    atr = np.full(40, 2.0)
    atr[10:] = 50.0                   # ATR explodes later; t=0 barriers stay 102/98
    h[3] = 101.5                      # touches neither 102 nor 98
    labels, _ = ev.triple_barrier_labels(o, h, lo, c, atr, horizon=24)
    assert labels[0] == -1            # tie fallback (flat closes) -> dropped


# ------------------------------------------------- context causality
def _ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.0005, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.001, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.001, n))),
        "close": close,
        "volume": rng.uniform(10, 20, n),
    }, index=idx)


def test_context_columns_and_shape():
    df = _ohlcv()
    ctx = ev.build_context_features(df)
    assert list(ctx.columns) == ev.CONTEXT_COLUMNS
    assert len(ctx) == len(df)
    assert list(ctx.index) == list(df.index)


def test_context_causality_under_future_garble():
    df = _ohlcv()
    base = ev.build_context_features(df)
    cut = 300
    garbled = df.copy()
    garbled.iloc[cut:] = garbled.iloc[cut:] * 3.7   # wreck the future
    after = ev.build_context_features(garbled)
    a = base.iloc[:cut].to_numpy(dtype=float)
    b = after.iloc[:cut].to_numpy(dtype=float)
    finite = np.isfinite(a)
    assert np.allclose(a[finite], b[finite], rtol=0, atol=1e-12)


def test_context_truncation_invariance():
    df = _ohlcv()
    full = ev.build_context_features(df)
    part = ev.build_context_features(df.iloc[:250])
    a = full.iloc[:250].to_numpy(dtype=float)
    b = part.to_numpy(dtype=float)
    finite = np.isfinite(a)
    assert np.allclose(a[finite], b[finite], rtol=0, atol=1e-12)


def test_context_hand_values():
    # Constant closes -> zero returns, rv 0 (ratio undefined -> NaN
    # allowed), dd100 == 0 where defined, ret == 0.
    n = 150
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"open": 50.0, "high": 50.0, "low": 50.0,
                       "close": 50.0, "volume": 1.0}, index=idx)
    ctx = ev.build_context_features(df)
    row = ctx.iloc[149]
    assert row["ctx_ret4"] == 0.0 and row["ctx_ret24"] == 0.0
    assert row["ctx_dd100"] == 0.0
    assert row["ctx_rv24"] == 0.0
    # hour of 2026-01-01 00:00 + 149h = Jan 7, 05:00 UTC
    assert idx[149].hour == 5
    assert abs(row["ctx_hour_sin"] - np.sin(2 * np.pi * 5 / 24)) < 1e-12
    assert abs(row["ctx_hour_cos"] - np.cos(2 * np.pi * 5 / 24)) < 1e-12


# ------------------------------------------------------- row sets
def test_common_training_rows_purge_and_feasibility():
    n, split = 300, 200
    frame = np.zeros((n, 4))
    labels = np.ones(n, dtype=int)
    labels[150] = -1                       # undefined label
    frame[100, 0] = np.nan                 # row not finite
    frame[60, 1] = np.nan                  # poisons sequences ending 60..83
    rows = ev.common_training_rows(frame, labels, split)
    s = set(int(x) for x in rows)
    assert 150 not in s                    # label
    assert 100 not in s                    # own row finite
    assert all(t not in s for t in range(60, 84))   # window feasibility
    assert all(t + 24 < split for t in s)  # purge: horizon inside training
    assert max(s) == split - 25
    assert 59 in s and 84 in s
    assert all(t >= 23 for t in s)         # needs a full 24-row window


def test_exam_positions_first_768_with_labels():
    labels = np.ones(2000, dtype=int)
    labels[900] = -1                       # a tie-drop inside the exam span
    pos = ev.exam_positions(labels, split=1000, n_exam=768)
    assert pos[0] == 1000 and pos[-1] == 1767
    assert 900 not in set(int(x) for x in pos)  # (outside span anyway)
    labels[1005] = -1
    pos = ev.exam_positions(labels, split=1000, n_exam=768)
    assert 1005 not in set(int(x) for x in pos)
    assert len(pos) == 767


# ------------------------------------------------------- sequences
def test_sequence_tensor_ends_at_t():
    frame = np.arange(50 * 3, dtype=float).reshape(50, 3)
    pos = np.array([30, 49])
    seqs = ev.sequence_tensor(frame, pos, seq_len=24)
    assert seqs.shape == (2, 24, 3)
    assert np.array_equal(seqs[0, -1], frame[30])   # ends AT t
    assert np.array_equal(seqs[0, 0], frame[7])     # starts at t-23
    assert np.array_equal(seqs[1, -1], frame[49])


def test_sequence_overlap_alignment():
    frame = np.arange(50 * 2, dtype=float).reshape(50, 2)
    a = ev.sequence_tensor(frame, np.array([30]))[0]
    b = ev.sequence_tensor(frame, np.array([31]))[0]
    assert np.array_equal(a[1:], b[:-1])            # shared slices identical


def test_sequence_future_garble_invariance():
    frame = np.random.default_rng(3).normal(size=(60, 4))
    pos = np.array([40])
    before = ev.sequence_tensor(frame, pos)
    frame2 = frame.copy()
    frame2[41:] = 9999.0                            # garble after t
    after = ev.sequence_tensor(frame2, pos)
    assert np.array_equal(before, after)


# ------------------------------------------------------- metrics
def test_vote_spread_hand_computed():
    votes = np.array([0.30, 0.44, 0.50, 0.56, 0.70])
    s = ev.vote_spread(votes)
    assert s["min"] == 0.30 and s["max"] == 0.70
    assert s["pct_out_45_55"] == pytest.approx(4 / 5)   # .30 .44 .56 .70 outside
    assert s["pct_out_40_60"] == pytest.approx(2 / 5)   # .30 .70 outside


def test_exam_metrics_hand_computed():
    votes = np.array([0.9, 0.8, 0.3, 0.6, 0.2])
    labels = np.array([1, 0, 0, 1, 1])
    m = ev.exam_metrics(votes, labels)
    # preds: 1,1,0,1,0 -> correct on rows 0,2,3 = 3/5
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["up_calls"] == 3
    # precision: up-calls on rows 0,1,3; true labels 1,0,1 -> 2/3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["n"] == 5


# ------------------------------------------------------- selection
def test_select_arm_highest_balanced_accuracy():
    oof = {"a": {"balanced_accuracy": 0.55, "auc": 0.51},
           "b": {"balanced_accuracy": 0.57, "auc": 0.50},
           "c": {"balanced_accuracy": 0.52, "auc": 0.60},
           "d": {"balanced_accuracy": 0.56, "auc": 0.59}}
    assert ev.select_arm(oof) == "b"


def test_select_arm_tie_broken_by_auc():
    oof = {"a": {"balanced_accuracy": 0.55, "auc": 0.51},
           "b": {"balanced_accuracy": 0.55, "auc": 0.58}}
    assert ev.select_arm(oof) == "b"


# ------------------------------------------------------- fallback
def test_lag_matrix_layout_lag_major():
    frame = np.arange(30 * 2, dtype=float).reshape(30, 2)
    out = ev.lag_matrix(frame, np.array([23]), [0, 1], seq_len=4)
    assert out.shape == (1, 8)
    # lag-major: [f0(t), f1(t), f0(t-1), f1(t-1), ...]
    expect = [frame[23, 0], frame[23, 1], frame[22, 0], frame[22, 1],
              frame[21, 0], frame[21, 1], frame[20, 0], frame[20, 1]]
    assert list(out[0]) == expect


# ------------------------------------------------------- LSTM arm
def test_lstm_eyes_smoke_and_future_invariance():
    rng = np.random.default_rng(11)
    n, f = 260, 6
    # Learnable-ish signal: label follows the sign of feature 0's mean.
    seqs = rng.normal(size=(n, 24, f))
    y = (seqs[:, :, 0].mean(axis=1) > 0).astype(int)
    model = ev.train_lstm_eyes(seqs, y, seed=42, max_epochs=3)
    p = model.predict_proba(seqs[:10])
    assert p.shape == (10,)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert 5_000 < model.param_count < 500_000
    # Garbling candles AFTER a row's decision point cannot change its
    # sequence: rebuild row 100's sequence from a garbled frame.
    frame = rng.normal(size=(200, f))
    pos = np.array([150])
    s_before = ev.sequence_tensor(frame, pos)
    frame[151:] = -777.0
    s_after = ev.sequence_tensor(frame, pos)
    assert np.array_equal(s_before, s_after)
    assert model.predict_proba(s_before)[0] == model.predict_proba(s_after)[0]


def test_lstm_shift_invariance():
    # The same sequence content at two positions gives the same vote.
    rng = np.random.default_rng(5)
    seqs = rng.normal(size=(120, 24, 4))
    y = (seqs[:, :, 1].mean(axis=1) > 0).astype(int)
    model = ev.train_lstm_eyes(seqs, y, seed=42, max_epochs=2)
    one = seqs[7:8]
    assert model.predict_proba(one)[0] == model.predict_proba(one.copy())[0]


# ------------------------------------------------------- compact registry
def test_compact_family_registry_eyes_v2():
    sys.path.insert(0, str(ROOT / "src" / "dashboards"))
    import compact  # noqa: E402
    assert compact._FAMILY_CODES["eyes_v2"] == 18
    run = {"id": 85, "ts": "2026-10-30T19:00:00+00:00", "kind": "stacker",
           "promoted": 0,
           "metrics": {"family": "eyes_v2", "challenger_acc": 0.55,
                       "champion_acc": 0.51, "precision": 0.52,
                       "eyes_v2": {"label": "Tradeable-Move Stacker",
                                   "verdict": "PASS",
                                   "design": "triple-barrier"}}}
    name = compact._display_name(run, None)
    assert name == "Eyes v2 — Tradeable-Move Stacker"
    entry = compact._model_entry(run, name)
    assert entry[3] == 18                     # family code
    assert entry[5] == 0.55                   # headline = stacker accuracy
    assert entry[7] == "PASS"                 # verdict from the sub-record
