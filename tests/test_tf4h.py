"""Tests for Drawing-Board Option A — the 4h timeframe test
(2026-10-24, refs/timeframe-4h-plan-2026-10-24.md).

Pins the new mechanics the pre-registered plan depends on:

(1) Resample correctness, hand-checked: 4h bars aggregate their 4
    hourly bars exactly (open=first, high=max, low=min, close=last,
    volume=sum), are open-time stamped, and incomplete bins (the
    source frame's gap holes) are dropped, never fabricated.
(2) Labels follow the v1 convention (>= +0.003 BUY / <= -0.003
    SHORT on the NEXT bar's close) — checked against an independent
    shift computation, boundaries included.
(3) Truncation causality at 4h: features from the production
    builders recomputed on a truncated frame match the full-frame
    values on sampled columns — the v1/v2 self-check, at 4h.
(4) The sliced span is exactly the v1/v2 matrix span: 4,183 bars,
    2024-11-23 00:00 -> 2026-10-21 16:00, all 90 feature columns
    present and NaN-free (the plan's no-exclusions census).
(5) Duel geometry at 4h: a flat (zero-conviction) signal through
    the 66-bar window harness produces zero fills and P&L $0, and
    the reported benchmark equals the harness formula recomputed
    from the window's closes.
(6) Verdict mechanics: the plan's three rules applied to fabricated
    records — PASS only when all three hold; rule 2's three
    sub-conditions each independently fatal.

Run: .venv/bin/python -m pytest tests/test_tf4h.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_timeseries_matrix as btm  # noqa: E402
from features import main_chart as mc  # noqa: E402
from features.technical_indicators import make_supervised  # noqa: E402
from models import tf4h  # noqa: E402
from models.gbm_policy import build_rule_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRAME_PKL = ROOT / "data" / "tf4h" / "frame4h.pkl"


# ---------------------------------------------------------------- (1)
def test_resample_4h_hand_checked():
    idx = pd.date_range("2026-01-01 00:00", periods=11, freq="1h")
    h = pd.DataFrame({
        "open":   [100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122],
        "high":   [105, 106, 107, 108, 115, 116, 117, 118, 125, 126, 127],
        "low":    [95, 96, 97, 98, 105, 106, 107, 108, 115, 116, 117],
        "close":  [101, 102, 103, 104, 111, 112, 113, 114, 121, 122, 123],
        "volume": [10, 20, 30, 40, 1, 2, 3, 4, 5, 6, 7],
    }, index=idx)
    bars = tf4h.resample_4h(h)
    # Third bin holds only 3 hourly bars -> dropped, not fabricated.
    assert list(bars.index) == [pd.Timestamp("2026-01-01 00:00"),
                                pd.Timestamp("2026-01-01 04:00")]
    b0 = bars.iloc[0]
    assert (b0["open"], b0["high"], b0["low"], b0["close"],
            b0["volume"]) == (100, 108, 95, 104, 100)
    b1 = bars.iloc[1]
    assert (b1["open"], b1["high"], b1["low"], b1["close"],
            b1["volume"]) == (110, 118, 105, 114, 10)


# ---------------------------------------------------------------- (2)
def test_labels_v1_convention_second_path():
    closes = [1000.0, 1003.0, 1000.0, 997.0, 1000.0, 1001.0, 999.0]
    df = pd.DataFrame({"close": closes},
                      index=pd.date_range("2026-01-01", periods=7,
                                          freq="4h"))
    lab = btm.compute_labels(df)
    r = df["close"].shift(-1) / df["close"] - 1.0
    want = pd.Series(np.where(r >= 0.003, "BUY",
                              np.where(r <= -0.003, "SHORT", "HOLD")),
                     index=df.index, dtype=object)
    # The builder and an independent shift computation agree on
    # every row (the default band is the production 0.003).
    assert btm.FEE_BAND == 0.003
    assert list(lab.iloc[:-1]) == list(want.iloc[:-1])
    assert pd.isna(lab.iloc[-1])


def test_labels_boundary_is_inclusive():
    # The >= / <= semantics, pinned in EXACT binary arithmetic:
    # with the band patched to 1/256, a next-bar return of exactly
    # +1/256 is BUY and exactly -1/256 is SHORT (a strict-> rule
    # would call both HOLD). 0.003 itself is not a binary fraction,
    # so the boundary is pinned this way rather than with decimals
    # whose float quotient lands a hair inside the band.
    closes = [1024.0, 1028.0, 1023.984375, 1023.984375]
    df = pd.DataFrame({"close": closes},
                      index=pd.date_range("2026-01-01", periods=4,
                                          freq="4h"))
    old = btm.FEE_BAND
    btm.FEE_BAND = 1.0 / 256.0
    try:
        lab = btm.compute_labels(df)
    finally:
        btm.FEE_BAND = old
    assert list(lab.iloc[:-1]) == ["BUY", "SHORT", "HOLD"]


# ---------------------------------------------------------------- (3)
def test_truncation_causality_at_4h():
    frame = tf4h.load_frame4h(FRAME_PKL)
    bars = frame[["open", "high", "low", "close", "volume"]]
    full59 = btm.build_features(bars.iloc[:1500])
    full31 = mc.build_all(bars.iloc[:1500])
    trunc59 = btm.build_features(bars.iloc[:1200])
    trunc31 = mc.build_all(bars.iloc[:1200])
    rows = bars.index[:1100]  # strictly inside the truncation
    for col in ["Relative Strength Index (RSI)",
                "Exponential Moving Average (EMA)",
                "On-Balance Volume (OBV)"]:
        a = full59[col].reindex(rows).to_numpy()
        b = trunc59[col].reindex(rows).to_numpy()
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), col
    for col in ["Distance to SMA 200 (%)",
                "Ichimoku Senkou Span A Distance (%)",
                "Distance to Prior-Day Pivot Point (%)"]:
        a = full31[col].reindex(rows).to_numpy()
        b = trunc31[col].reindex(rows).to_numpy()
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), col


# ---------------------------------------------------------------- (4)
def test_slice_span_and_feature_census():
    frame = tf4h.load_frame4h(FRAME_PKL)
    sl = tf4h.slice_span(frame)
    assert len(sl) == 4183
    assert sl.index[0] == tf4h.SPAN_START
    assert sl.index[-1] == tf4h.SPAN_END
    feats = tf4h.all_feature_columns(frame)
    assert len(feats) == 90
    focus = tf4h.arm_features(frame, "focus4h")
    assert len(focus) == 11 and all(f in frame.columns for f in focus)
    assert int(sl[feats].isna().sum().sum()) == 0
    assert int(sl["__label"].isna().sum()) == 0


# ---------------------------------------------------------------- (5)
def _one_real_window():
    frame = tf4h.load_frame4h(FRAME_PKL)
    bars = frame[["open", "high", "low", "close", "volume"]]
    sup = make_supervised(bars.iloc[:1500])
    return sup.iloc[-66:]


def test_duel_zero_signal_is_flat_and_bench_formula():
    window = _one_real_window()
    model = build_rule_model()
    p_up = np.full(len(window), 0.5)  # conviction 0 everywhere
    res = tf4h.evaluate_windows_4h([window], [p_up], model)
    assert res["total_round_trips"] == 0 and res["closed_trips"] == 0
    assert res["total_pnl"] == 0.0 and res["fees_total"] == 0.0
    assert res["non_hold_share"] == 0.0
    closes = window["close"].to_numpy(dtype=float)
    bench = 100_000.0 * (closes[-1] / closes[0] - 1.0)
    assert abs(res["bench_pnl"] - bench) < 1e-6


def test_duel_always_long_trades_and_fee_identity():
    window = _one_real_window()
    model = build_rule_model()
    p_up = np.full(len(window), 1.0)  # conviction +1: always BUY
    res = tf4h.evaluate_windows_4h([window], [p_up], model)
    assert res["total_round_trips"] >= 1 or res["open_left"] >= 1
    assert res["fees_total"] > 0.0
    # The plan's fee-transparency identity, as reported.
    assert res["gross_pnl"] == res["total_pnl"] + res["fees_total"]
    assert res["non_hold_share"] > 0.0


# ---------------------------------------------------------------- (6)
def _wf(bal=0.40, acc=0.45, base=0.40, folds=(0.40, 0.41, 0.39, 0.42, 0.38)):
    return {"pooled": {"balanced_accuracy": bal, "accuracy": acc,
                       "majority_baseline": base},
            "folds": [{"balanced_accuracy": b} for b in folds]}


def test_verdict_requires_all_three_rules():
    duel_ok = {"total_pnl": 100.0, "bench_pnl": 50.0,
               "non_hold_share": 0.5}
    robust_ok = {"total_pnl": 0.0}
    v = tf4h.apply_verdict(_wf(), duel_ok, robust_ok)
    assert v["verdict"] == "PASS"
    # Rule 1 failures: pooled BA at chance; fold spread 3/5; acc
    # below the majority baseline.
    assert not tf4h.apply_verdict(
        _wf(bal=0.333), duel_ok, robust_ok)["rule1_walk_forward"]
    assert not tf4h.apply_verdict(
        _wf(folds=(0.40, 0.40, 0.40, 0.30, 0.30)),
        duel_ok, robust_ok)["rule1_walk_forward"]
    assert not tf4h.apply_verdict(
        _wf(acc=0.39), duel_ok, robust_ok)["rule1_walk_forward"]
    # Rule 2 failures: each sub-condition independently fatal.
    for bad in ({"total_pnl": 0.0, "bench_pnl": 50.0,
                 "non_hold_share": 0.5},
                {"total_pnl": 100.0, "bench_pnl": 100.0,
                 "non_hold_share": 0.5},
                {"total_pnl": 100.0, "bench_pnl": 50.0,
                 "non_hold_share": 0.099}):
        out = tf4h.apply_verdict(_wf(), bad, robust_ok)
        assert not out["rule2_duel_floor"] and out["verdict"] == "FAIL"
    # Rule 3 failure: negative robustness.
    out = tf4h.apply_verdict(_wf(), duel_ok, {"total_pnl": -0.01})
    assert not out["rule3_robustness"] and out["verdict"] == "FAIL"


def test_arm_features():
    frame = tf4h.load_frame4h(FRAME_PKL)
    assert len(tf4h.arm_features(frame, "all4h")) == 90
    focus = tf4h.arm_features(frame, "focus4h")
    # Matrix-name "Volume" maps to the frame's raw volume column;
    # the other ten are indicator columns, Kass's list verbatim.
    assert focus[0] == "volume"
    assert focus[1:] == list(tf4h.FOCUS_FEATURES)[1:]
