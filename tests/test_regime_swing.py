"""Regime-Swing System (run 86) — adversarial tests, written
BEFORE the implementation per the house quality standard.
Design frozen in refs/regime-swing-study-2026-10-30.md.

Pins: 4h bucket integrity + END stamps and future-garble
causality (the v5.1 trap), regime-label construction on
hand-computed paths (threshold read at t, horizon boundary),
Block B's bucket-end-candle mapping (positional, never the
next bucket's first candle), walk-forward fit isolation
(purge + garble), the trader rule truth table, next-open
fill timing, broker stop override, fee accounting vs the
ledger re-sum, and degenerate-classifier visibility in the
OOF report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import regime_swing as rs  # noqa: E402


def _hourly(n: int, start: str = "2024-01-01", seed: int = 7,
            drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close,
                         "volume": np.full(n, 10.0)}, index=idx)


# ------------------------------------------------------- 4h frame
def test_bucket_end_stamps_and_aggregation():
    df = _hourly(8)
    h4, stats = rs.build_4h_frame(df)
    assert len(h4) == 2
    # END stamps: the first bucket covers candles 0..3 and is
    # stamped at candle 0's start + 4h, not at its start.
    assert h4.index[0] == df.index[0] + pd.Timedelta(hours=4)
    assert h4.index[1] == df.index[0] + pd.Timedelta(hours=8)
    b0 = h4.iloc[0]
    assert b0["open"] == df["open"].iloc[0]
    assert b0["close"] == df["close"].iloc[3]
    assert b0["high"] == df["high"].iloc[:4].max()
    assert b0["low"] == df["low"].iloc[:4].min()
    assert b0["volume"] == df["volume"].iloc[:4].sum()
    assert int(b0["first_pos"]) == 0 and int(b0["last_pos"]) == 3
    assert stats["n_kept"] == 2 and stats["n_dropped_partial"] == 0


def test_partial_bucket_dropped():
    df = _hourly(10)
    df = df.drop(df.index[6])  # hole inside the second bucket
    h4, stats = rs.build_4h_frame(df)
    # bucket [0..3] intact; bucket [4..7] has 3 candles -> dropped;
    # trailing partial bucket dropped too.
    assert len(h4) == 1
    assert h4.index[0] == df.index[0] + pd.Timedelta(hours=4)
    assert stats["n_dropped_partial"] >= 1


def test_bucket_causality_future_garble():
    df = _hourly(1400, seed=11)
    h4, _ = rs.build_4h_frame(df)
    atr = rs.atr_pct_series(h4)
    feats, cols = rs.block_a_features(h4, atr)
    boundary = 260  # a bucket well past every warmup
    end_stamp = h4.index[boundary]
    garbled = df.copy()
    mask = garbled.index > end_stamp - pd.Timedelta(hours=1)
    garbled.loc[mask, ["open", "high", "low", "close"]] *= 3.7
    h4g, _ = rs.build_4h_frame(garbled)
    atrg = rs.atr_pct_series(h4g)
    featsg, _ = rs.block_a_features(h4g, atrg)
    # Buckets at or before the boundary are bit-identical.
    for c in ("open", "high", "low", "close"):
        assert np.allclose(h4[c].to_numpy()[:boundary + 1],
                           h4g[c].to_numpy()[:boundary + 1])
    row, rowg = feats[boundary], featsg[boundary]
    assert np.isfinite(row).all()
    assert np.allclose(row, rowg, equal_nan=True)


# ------------------------------------------------------- labels
def test_regime_labels_hand_computed():
    n = 60
    atr = np.full(n, 0.01)  # thr = 0.75 * 1% = 0.75%
    up = 100.0 * 1.01 ** np.arange(n)          # +1%/bar -> UP
    dn = 100.0 * 0.99 ** np.arange(n)          # -1%/bar -> DOWN
    flat = np.full(n, 100.0)                   # -> CHOP
    assert rs.regime_labels(up, atr)[0] == rs.UP
    assert rs.regime_labels(dn, atr)[0] == rs.DOWN
    assert rs.regime_labels(flat, atr)[0] == rs.CHOP
    # Inside the band: +0.5% over the whole horizon -> CHOP.
    mild = np.linspace(100.0, 100.5, n)
    assert rs.regime_labels(mild, atr)[0] == rs.CHOP
    # The last `forward` rows carry no label.
    lab = rs.regime_labels(up, atr)
    assert (lab[n - 24:] == -1).all()
    assert (lab[:n - 24] >= 0).all()


def test_regime_label_threshold_read_at_t():
    n = 60
    close = np.linspace(100.0, 103.0, n)  # +3% over 24 bars
    atr = np.full(n, 0.10)                # huge band everywhere...
    atr[0] = 0.01                         # ...except at t=0
    lab = rs.regime_labels(close, atr)
    # At t=0 the band is 0.75%: +3% > band -> UP despite the
    # huge ATR later (the threshold is the one at t).
    assert lab[0] == rs.UP
    # At t=1 the band is 7.5%: the remaining move is inside -> CHOP.
    assert lab[1] == rs.CHOP


# ------------------------------------------------------- block B
def test_block_b_takes_bucket_end_candle():
    n = 16
    df = _hourly(n)
    vals = (np.arange(n)[:, None] * 1000.0
            + np.arange(98)[None, :])  # row identity encoded
    h4, _ = rs.build_4h_frame(df)
    b = rs.block_b_features(vals, df.index, h4.index)
    assert b.shape == (4, 98)
    for k in range(4):
        # The bucket's LAST candle is position 4k+3 (stamped
        # end - 1h) — never position 4k+4 (the next bucket).
        assert b[k, 0] == (4 * k + 3) * 1000.0
        assert b[k, 97] == (4 * k + 3) * 1000.0 + 97


# ------------------------------------------------------- eyes plumbing
def test_eligible_fit_rows_purge():
    labels = np.zeros(200, dtype=np.int8)
    labels[::37] = -1
    rows = rs.eligible_fit_rows(labels, boundary_pos=100)
    assert rows.max() < 100 - 24  # horizon ends before boundary
    assert (labels[rows] >= 0).all()
    expect = np.array([t for t in range(200)
                       if labels[t] >= 0 and t + 24 < 100])
    assert np.array_equal(rows, expect)


def test_walkforward_fit_isolation_under_garble():
    rng = np.random.default_rng(3)
    n, f = 220, 12
    X = rng.normal(size=(n, f))
    y = (X[:, 0] + rng.normal(0, 0.1, n) > 0).astype(int)
    y = np.where(y == 1, rs.UP, rs.DOWN)
    boundary = 140
    probe = X[:9]
    rows = rs.eligible_fit_rows(
        np.where(np.arange(n) < n - 24, y, -1), boundary)
    m1 = rs.fit_eyes(X[rows], y[rows], "lgbm")
    p1 = rs.predict_class(m1, "lgbm", probe)
    Xg, yg = X.copy(), y.copy()
    Xg[boundary:] = rng.normal(9.0, 3.0, Xg[boundary:].shape)
    yg[boundary:] = rs.CHOP
    rows_g = rs.eligible_fit_rows(
        np.where(np.arange(n) < n - 24, yg, -1), boundary)
    assert np.array_equal(rows, rows_g)
    m2 = rs.fit_eyes(Xg[rows_g], yg[rows_g], "lgbm")
    p2 = rs.predict_class(m2, "lgbm", probe)
    assert np.array_equal(p1, p2)


def test_class_report_exposes_degenerate():
    y = np.array([0, 1, 2] * 10)
    pred = np.full(30, rs.CHOP)
    rep = rs.class_report(y, pred)
    assert rep["per_class_recall"][rs.DOWN] == 0.0
    assert rep["per_class_recall"][rs.CHOP] == 1.0
    assert rep["per_class_recall"][rs.UP] == 0.0
    assert rep["pred_counts"][rs.CHOP] == 30
    assert abs(rep["balanced_accuracy"] - 1.0 / 3.0) < 1e-12


def test_select_arm_tiebreaks():
    tab = {"stacker": {"balanced_accuracy": 0.40, "macro_auc": 0.55},
           "lgbm": {"balanced_accuracy": 0.40, "macro_auc": 0.61}}
    assert rs.select_arm(tab) == "lgbm"          # tie -> AUC
    tab["lgbm"]["balanced_accuracy"] = 0.40 + 5e-13  # still a tie
    assert rs.select_arm(tab) == "lgbm"
    tab["stacker"]["balanced_accuracy"] = 0.45   # clear BA win
    assert rs.select_arm(tab) == "stacker"


# ------------------------------------------------------- trader rules
def test_decide_truth_table():
    H, B, S, SS = "hold", "buy", "sell", "short"
    cases = {
        (0, 0, rs.UP): B, (0, 0, rs.DOWN): SS, (0, 0, rs.CHOP): H,
        (1, 1, rs.UP): B, (1, 1, rs.DOWN): S, (1, 1, rs.CHOP): S,
        (1, 2, rs.UP): H, (1, 2, rs.DOWN): S, (1, 2, rs.CHOP): S,
        (-1, 1, rs.UP): B, (-1, 1, rs.DOWN): SS, (-1, 1, rs.CHOP): B,
        (-1, 2, rs.UP): B, (-1, 2, rs.DOWN): H, (-1, 2, rs.CHOP): B,
    }
    for (side, tr, reg), want in cases.items():
        got = rs.decide(side, tr, reg)
        assert got == want, f"decide({side},{tr},{reg})={got} != {want}"


def _span_df(prices: list[float]) -> pd.DataFrame:
    """Hourly candles from a close path (open = previous close,
    no wicks beyond 0.1%)."""
    close = np.array(prices, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h")
    return pd.DataFrame({
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * 1.001,
        "low": np.minimum(open_, close) * 0.999,
        "volume": np.full(len(close), 10.0)}, index=idx)


def _run(df, regimes, atr=0.01, tmp_path=None, name="span.db"):
    h4, _ = rs.build_4h_frame(df)
    first = h4["first_pos"].to_numpy(dtype=int)
    last = h4["last_pos"].to_numpy(dtype=int)
    db = (tmp_path or Path("/tmp")) / name
    db.unlink(missing_ok=True)
    return rs.run_span(df, first, last, list(range(len(h4))),
                       np.asarray(regimes), np.full(len(h4), atr), db), db


def test_next_open_fill_timing(tmp_path):
    # 3 buckets, price flat at 100 except bucket 2 opens at 123.45.
    prices = [100.0] * 12
    df = _span_df(prices)
    df.iloc[4, df.columns.get_loc("open")] = 123.45
    res, db = _run(df, [rs.UP, rs.CHOP, rs.CHOP], tmp_path=tmp_path)
    assert res["trades"] >= 1
    fill = res["fills"][0]
    assert fill["price"] == pytest.approx(123.45)
    assert fill["ts"] == str(df.index[4])
    assert fill["side"] == "BUY"


def test_decision_at_last_bucket_close_never_fills(tmp_path):
    df = _span_df([100.0] * 4)  # one bucket only
    res, _db = _run(df, [rs.UP], tmp_path=tmp_path)
    assert res["trades"] == 0
    assert res["pnl"] == 0.0
    assert res["decisions"][0][1] == "buy"  # intent recorded, dropped


def test_stop_override_and_reentry(tmp_path):
    # Entry at bucket1 open (100), then a wick to 97.0 inside
    # bucket 1 stops the long out (stop = 97.5 at atr 1%);
    # regime still UP at bucket1 close -> fresh entry (tranche
    # reset), filled at bucket2 open.
    prices = [100.0] * 12
    df = _span_df(prices)
    df.iloc[5, df.columns.get_loc("low")] = 97.0
    df.iloc[5, df.columns.get_loc("close")] = 98.0
    res, _db = _run(df, [rs.UP, rs.UP, rs.CHOP], tmp_path=tmp_path)
    reasons = [f["reason"] for f in res["fills"]]
    assert "stop_loss" in reasons
    stop_fill = res["fills"][reasons.index("stop_loss")]
    assert stop_fill["price"] == pytest.approx(97.5)
    # 3 fills: entry, stop exit, re-entry.
    assert res["trades"] == 3
    assert res["round_trips"] == 1
    assert res["fills"][-1]["side"] == "BUY"


# ------------------------------------------------------- compact registry
def test_compact_family_registry_regime_swing():
    sys.path.insert(0, str(ROOT / "src" / "dashboards"))
    import compact  # noqa: E402
    assert compact._FAMILY_CODES["regime_swing"] == 19
    run = {"id": 86, "ts": "2026-10-30T20:00:00+00:00", "kind": "ppo",
           "promoted": 0,
           "metrics": {"family": "regime_swing",
                       "challenger": {"score": 12.0, "total_pnl": 34.0},
                       "regime_swing": {"label": "4h Swing",
                                        "verdict": "FAIL"}}}
    assert compact._display_name(run, None) == "Regime-Swing — 4h Swing"
    row = compact._model_entry(run, "Regime-Swing — 4h Swing")
    assert row[3] == 19 and row[7] == "FAIL"
    assert row[5] == 12.0 and row[6] == 34


def test_fee_accounting_matches_ledger_resum(tmp_path):
    prices = [100.0] * 16
    df = _span_df(prices)
    # UP, UP (entry + add), CHOP (exit), CHOP -> 3 fills, flat end.
    res, db = _run(df, [rs.UP, rs.UP, rs.CHOP, rs.CHOP],
                   tmp_path=tmp_path)
    assert res["trades"] == 3
    assert res["fees"] > 0
    final_close = float(df["close"].iloc[-1])
    resumed = rs.resum_span_pnl(db, final_close)
    assert abs(resumed - res["pnl"]) < 1e-6
    # Flat at the end: P&L is exactly the fee drag (prices flat).
    assert res["pnl"] == pytest.approx(-res["fees"], abs=1e-6)
