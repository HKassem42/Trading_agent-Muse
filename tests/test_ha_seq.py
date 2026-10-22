"""Tests for the HA-sequence challenger family (v1, 2026-10-22).

Written BEFORE the implementation, per Kass's quality standard and the
ml_time_series_causality_checks skill:

(a) Heikin-Ashi transform vs hand-computed reference values.
(b) Sequence-builder causality across ALL rows: the sequence ending at
    row r built from the full frame must equal the one built from the
    frame truncated at r.
(c) Normalization isolation: statistics come from the TRAIN region
    only -- mutating holdout rows must not change any train sequence.
    A deliberately leaky global-stats variant is included to prove the
    check has teeth (it MUST change the sequences).
(d) Wrapper accounting through the real TradingEnv + PaperBroker
    (next_open fills): fills at next-candle open, 0.15% fees deducted,
    stops trigger on wicks and fill at the stop level.

Run: .venv/bin/python -m pytest tests/test_ha_seq.py -q
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from models import ha_seq  # noqa: E402
from models.ha_policy import decide  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


def _ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk OHLCV frame (hourly, UTC-naive index)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    vol = rng.uniform(5, 50, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


# ---------------------------------------------------------------- (a)
def test_heikin_ashi_hand_computed():
    df = pd.DataFrame({
        "open":   [100.0, 106.0, 103.0, 107.0],
        "high":   [110.0, 112.0, 108.0, 109.0],
        "low":    [95.0, 101.0, 99.0, 102.0],
        "close":  [105.0, 103.0, 107.0, 104.0],
        "volume": [10.0, 11.0, 9.0, 12.0],
    })
    ha = ha_seq.heikin_ashi(df)
    # Hand-computed: hc=(o+h+l+c)/4; ho[0]=(o+c)/2, ho[t]=(ho+hc)[t-1]/2;
    # hh=max(h,ho,hc); hl=min(l,ho,hc).
    assert ha["ha_close"].tolist() == pytest.approx(
        [102.5, 105.5, 104.25, 105.5], abs=1e-12)
    assert ha["ha_open"].tolist() == pytest.approx(
        [102.5, 102.5, 104.0, 104.125], abs=1e-12)
    assert ha["ha_high"].tolist() == pytest.approx(
        [110.0, 112.0, 108.0, 109.0], abs=1e-12)
    assert ha["ha_low"].tolist() == pytest.approx(
        [95.0, 101.0, 99.0, 102.0], abs=1e-12)


def test_candle_features_hand_computed():
    df = pd.DataFrame({
        "open":   [100.0, 106.0, 103.0, 107.0],
        "high":   [110.0, 112.0, 108.0, 109.0],
        "low":    [95.0, 101.0, 99.0, 102.0],
        "close":  [105.0, 103.0, 107.0, 104.0],
        "volume": [10.0, 11.0, 9.0, 12.0],
    })
    f = ha_seq.candle_features(df)
    # Row 1 (second candle): range 112-101 = 11, body 105.5-102.5 = 3.
    assert f["body_ratio"].iloc[1] == pytest.approx(3.0 / 11.0, abs=1e-12)
    assert f["upper_wick"].iloc[1] == pytest.approx(6.5 / 11.0, abs=1e-12)
    assert f["lower_wick"].iloc[1] == pytest.approx(1.5 / 11.0, abs=1e-12)
    assert f["range_rel"].iloc[1] == pytest.approx(11.0 / 105.5, abs=1e-12)
    assert f["ha_ret"].iloc[1] == pytest.approx(
        np.log(105.5 / 102.5), abs=1e-12)
    # Bounded channels stay in their ranges on random data too.
    fr = ha_seq.candle_features(_ohlcv(500)).dropna()
    assert fr["body_ratio"].between(-1, 1).all()
    assert fr["upper_wick"].between(0, 1).all()
    assert fr["lower_wick"].between(0, 1).all()
    assert (fr["range_rel"] >= 0).all()


# ---------------------------------------------------------------- (b)
def test_sequence_causality_all_rows():
    raw = _ohlcv(160)
    feat = ha_seq.candle_features(raw)
    stats = ha_seq.fit_stats(feat, upto=120)  # fixed stats, both sides
    for seq_len in (10, 90):
        for end in range(seq_len - 1, len(raw)):
            full = ha_seq.build_sequence(feat, stats, end, seq_len)
            trunc_feat = ha_seq.candle_features(raw.iloc[:end + 1])
            trunc = ha_seq.build_sequence(trunc_feat, stats, end, seq_len)
            assert (full is None) == (trunc is None), (seq_len, end)
            if full is not None:
                assert np.array_equal(full, trunc), (seq_len, end)


# ---------------------------------------------------------------- (c)
def test_normalization_isolation_train_only():
    raw = _ohlcv(400)
    fit_upto = 250
    feat = ha_seq.candle_features(raw)
    stats = ha_seq.fit_stats(feat, upto=fit_upto)
    # Stats really are the train region's, not the full frame's.
    manual = feat.iloc[:fit_upto]
    for c in ha_seq.CHANNELS:
        assert stats["mean"][c] == pytest.approx(float(manual[c].mean()))
        assert stats["std"][c] == pytest.approx(float(manual[c].std(ddof=0)))
    base = {e: ha_seq.build_sequence(feat, stats, e, 30)
            for e in range(29, fit_upto)}
    # Mutate the holdout region violently; train sequences must not move.
    raw2 = raw.copy()
    raw2.iloc[300:, raw2.columns.get_loc("close")] *= 3.0
    raw2.iloc[300:, raw2.columns.get_loc("volume")] *= 10.0
    feat2 = ha_seq.candle_features(raw2)
    for e, seq in base.items():
        assert np.array_equal(seq, ha_seq.build_sequence(feat2, stats, e, 30))
    # Teeth: a leaky variant (stats fit on the FULL mutated frame) MUST
    # change the same train sequences -- the check above can catch it.
    leaky = ha_seq.fit_stats(feat2, upto=len(feat2))
    changed = any(
        not np.array_equal(seq, ha_seq.build_sequence(feat2, leaky, e, 30))
        for e, seq in base.items())
    assert changed, "global-stats leak went undetected: test has no teeth"


# ---------------------------------------------------------------- rule
def test_decide_thresholds_and_position():
    # p_c is the centered probability (2p-1) as it appears in obs[21].
    assert decide(0.10, 0.0) == 1    # p = 0.55 boundary -> long
    assert decide(0.50, 0.0) == 1
    assert decide(-0.10, 0.0) == 3   # p = 0.45 boundary -> short
    assert decide(-0.50, 0.0) == 3
    assert decide(0.0, 0.0) == 0     # flat band, flat book -> hold
    assert decide(0.0, 0.4) == 2     # flat band, long -> exit
    assert decide(0.0, -0.4) == 1    # flat band, short -> cover
    assert decide(0.099, 0.0) == 0   # just inside the band
    assert decide(0.50, -0.4) == 1   # conviction overrides: cover toward long
    assert decide(-0.50, 0.4) == 3   # conviction overrides: close toward short


# ---------------------------------------------------------------- (d)
class _StubPolicy:
    """Drives the real env with the HA rule, reading obs[21]/obs[23]
    exactly as the SB3 wrapper policy does at grading time."""

    def predict(self, obs, deterministic=True):
        return decide(float(obs[21]), float(obs[23])), None


def _synthetic_sup(closes, opens, highs, lows, atr=0.01):
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [10.0] * n, "atr_pct_raw": [atr] * n,
    }, index=idx)
    for c in FEATURE_COLUMNS:
        df[c] = 0.0
    return df


def test_wrapper_accounting_next_open_fees_and_wick_stop(tmp_path):
    # Decision candle (row 0) closes at 111, but the fill must happen at
    # row 1's OPEN (100) -- next-candle-open semantics. Row 1 then wicks
    # to 96.5 through the 2.5% stop (entry 100, atr 1% -> base stop is
    # clipped to the 2.5% floor = 97.5) while closing safe at 99, so the
    # stop fill must print at the level itself, 97.5.
    closes = [111.0, 99.0] + [99.0] * 28
    opens = [110.0, 100.0] + [99.0] * 28
    highs = [112.0, 100.5] + [99.5] * 28
    lows = [109.0, 96.5] + [98.5] * 28
    sup = _synthetic_sup(closes, opens, highs, lows)
    p_up = np.full(len(sup), 0.5)
    p_up[0] = 0.9  # conviction long on the first decision only
    db = tmp_path / "acct.db"
    broker = PaperBroker(db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, p_up, broker, fill_mode="next_open")
    policy = _StubPolicy()
    obs, _ = env.reset(seed=0, options={"start": 0})
    done = False
    steps = 0
    while not done and steps < 30:
        action, _ = policy.predict(obs)
        obs, _, done, _, _ = env.step(int(action))
        steps += 1
    con = sqlite3.connect(db)
    trades = con.execute(
        "SELECT side, price, qty, fee, reason FROM trades ORDER BY id"
    ).fetchall()
    con.close()
    assert len(trades) == 2, trades
    (side, price, qty, fee, _), (side2, price2, qty2, fee2, reason2) = trades
    assert side == "BUY" and price == pytest.approx(100.0)   # next open
    assert qty == pytest.approx(250.0 * (1 - 0.0015), rel=1e-9)
    assert fee == pytest.approx(25_000.0 * 0.0015)          # 0.15% of spend
    assert side2 == "SELL" and reason2 == "stop_loss"
    assert price2 == pytest.approx(97.5)                    # stop level
    assert qty2 == pytest.approx(qty)                       # full exit
    assert fee2 == pytest.approx(qty * 97.5 * 0.0015)       # fee on exit too
