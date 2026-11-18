"""Muse's Pick #2 challenger (Kass's order, 2026-10-27): beat
Muse's own crowned champion (Pick #1's stacker, run 69).

Two artifacts under test:
1. The new causal features (src/features/pick2_features.py):
   rolling R/S Hurst exponent (100 bars, log price), rolling
   return skew / excess kurtosis / lag-1 autocorrelation
   (100 bars, log returns) -- each pinned against a HAND-COMPUTED
   or independent-library second method, and against the
   truncate-and-append invariance the v5.1 standard demands:
   a feature value must not change when future data is appended.
2. The duel-length shaped env (src/models/muse_pick2.py):
   Pick #1's ShapedTradingEnv with training episodes of 264
   candles (the duel-window length) instead of 24 -- Pick #1's
   post-mortem confound, given its first fair test. Lockstep
   against the production env at the same episode length: same
   trading, same equity path, rewards differ by exactly the
   shaping penalties.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import pick2_features, scored_features  # noqa: E402
from models.muse_pick import ShapedTradingEnv, shaping_penalties  # noqa: E402
from models.muse_pick2 import DuelLengthShapedTradingEnv  # noqa: E402
from environment.trading_env import TradingEnv  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


# ------------------------------------------------- hand references
def _hurst_manual(logp: list[float]) -> float:
    """R/S Hurst by explicit arithmetic on plain lists -- the
    independent second method for the module's numpy version."""
    w = len(logp)
    m = sum(logp) / w
    y, run = [], 0.0
    for v in logp:
        run += v - m
        y.append(run)
    r = max(y) - min(y)
    var = sum((v - m) ** 2 for v in logp) / (w - 1)
    s = math.sqrt(var)
    return math.log(r / s) / math.log(w)


def _series(closes, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1h",
                        tz="UTC")
    return pd.Series(np.asarray(closes, dtype=float), index=idx)


# ------------------------------------------------------------- Hurst
def test_hurst_matches_hand_computation():
    rng = np.random.default_rng(11)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    s = _series(closes)
    h = pick2_features.rolling_hurst(s, window=100)
    logp = np.log(closes)
    for t in (99, 150, 250, 399):
        expected = _hurst_manual(list(logp[t - 99:t + 1]))
        assert h.iloc[t] == pytest.approx(expected, rel=1e-12), t
    # Warmup: the first 99 values cannot exist.
    assert h.iloc[:99].isna().all()
    assert np.isfinite(h.iloc[99:].to_numpy(float)).all()


def test_hurst_flat_is_nan_and_trend_exceeds_alternating():
    flat = _series([100.0] * 200)
    h_flat = pick2_features.rolling_hurst(flat, window=100)
    assert h_flat.iloc[99:].isna().all()  # S == 0 -> undefined
    trend = _series(100 * np.exp(0.01 * np.arange(200)))
    h_trend = pick2_features.rolling_hurst(trend, window=100)
    alt = _series([100.0, 101.0] * 100)
    h_alt = pick2_features.rolling_hurst(alt, window=100)
    assert h_trend.iloc[-1] > h_alt.iloc[-1]
    assert h_trend.iloc[-1] == pytest.approx(
        _hurst_manual(list(np.log(trend.to_numpy())[-100:])),
        rel=1e-12)


# ----------------------------------------------------------- moments
def test_moments_match_scipy_unbiased_estimators():
    from scipy import stats
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 400)))
    s = _series(closes)
    mom = pick2_features.rolling_moments(s, window=100)
    r = np.log(closes[1:] / closes[:-1])
    for t in (150, 260, 399):
        w = r[t - 100:t]  # returns ending at row t (r index t-1)
        assert mom["ret_skew_100_raw"].iloc[t] == pytest.approx(
            stats.skew(w, bias=False), rel=1e-9), t
        assert mom["ret_kurt_100_raw"].iloc[t] == pytest.approx(
            stats.kurtosis(w, bias=False), rel=1e-9), t


def test_autocorr_matches_manual_pearson():
    rng = np.random.default_rng(9)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    s = _series(closes)
    ac = pick2_features.rolling_autocorr1(s, window=100)
    r = np.log(closes[1:] / closes[:-1])
    for t in (150, 399):
        # pandas pairs (R_j, R_{j-1}) for the 100 rows j in
        # [t-99, t]: in numpy-return indices that is
        # corr(r[t-100:t], r[t-101:t-1]) -- 100 pairs.
        expected = np.corrcoef(r[t - 100:t], r[t - 101:t - 1])[0, 1]
        assert ac.iloc[t] == pytest.approx(expected, rel=1e-9), t


# ---------------------------------------------------------- causality
def test_raw_features_unchanged_when_future_appended():
    """THE v5.1 test, across ALL rows: append wild future bars;
    every already-computed value must be bit-identical."""
    rng = np.random.default_rng(21)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    base = pick2_features.raw_new_features(
        pd.DataFrame({"close": _series(closes)}))
    wild = closes[-1] * np.exp(np.cumsum(
        rng.normal(0.05, 0.2, 300)))  # violent future
    ext = pick2_features.raw_new_features(
        pd.DataFrame({"close": _series(np.concatenate([closes, wild]),
                                       )}))
    for c in pick2_features.RAW_COLUMNS:
        a = base[c].to_numpy(float)
        b = ext[c].iloc[:500].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isnan(a) == np.isnan(b)).all(), c
        assert np.array_equal(a[both], b[both]), c


def test_new_scores_match_brute_force_second_method():
    """The percentile scoring of a new raw series, checked against
    the v3 builder's own brute-force scorer at sampled positions."""
    b3 = scored_features._b3
    rng = np.random.default_rng(4)
    n = 6000
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"close": _series(closes)})
    raw = pick2_features.raw_new_features(df)
    qa = raw["hurst_rs_100_raw"].to_numpy(float)
    # Positions must satisfy the scorer's own precondition -- the
    # full trailing window finite (last_bad 98 + 4320 <= p) -- the
    # same warmup rule build_scored_frame enforces in production.
    positions = np.arange(4500, n, 37)
    fast = b3.rolling_scores(qa, positions)
    for i, p in enumerate(positions):
        assert fast[i] == pytest.approx(
            b3.brute_score(qa, int(p)), rel=1e-9, abs=1e-9), int(p)


def test_build_pick2_frame_shape_and_truncation():
    rng = np.random.default_rng(3)
    n = 6000
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": closes, "high": closes * 1.002,
         "low": closes * 0.998, "close": closes,
         "volume": rng.uniform(1, 10, n)}, index=idx)
    frame, cols, ev = pick2_features.build_pick2_frame(df)
    assert len(cols) == 98
    assert cols[:94] == scored_features.scored_columns()
    assert cols[94:] == pick2_features.NEW_COLUMNS
    vals = frame[pick2_features.NEW_COLUMNS].to_numpy(float)
    p0 = ev["pick2_p0"]
    assert np.isnan(vals[:p0]).all()
    assert np.isfinite(vals[p0:]).all()
    assert vals[p0:].min() >= 0.0 and vals[p0:].max() <= 100.0
    # Truncation: rebuild on a cut frame; the new columns must be
    # identical on every shared finite row.
    K = n - 400
    frame_t, _, _ = pick2_features.build_pick2_frame(df.iloc[:K + 1])
    for c in pick2_features.NEW_COLUMNS:
        a = frame[c].iloc[:K + 1].to_numpy(float)
        b = frame_t[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 500, c
        assert np.allclose(a[both], b[both], rtol=0, atol=1e-9), c


# ---------------------------------------------------------------- env
def _toy_frame(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "open": closes, "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes], "close": closes,
        "f0": np.zeros(n), "f1": np.zeros(n),
    })


def test_pick2_env_episode_is_264_candles(tmp_path):
    frame = _toy_frame([100.0] * 600)
    broker = PaperBroker(tmp_path / "a.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = DuelLengthShapedTradingEnv(frame, np.full(600, 0.5),
                                     broker,
                                     feature_columns=["f0", "f1"])
    assert env.episode_len == 264
    assert issubclass(DuelLengthShapedTradingEnv, ShapedTradingEnv)
    # The production default is untouched: still a 24-candle day.
    assert TradingEnv(frame, np.full(600, 0.5),
                      PaperBroker(tmp_path / "b.db",
                                  starting_balance=100_000,
                                  max_position_pct=0.25,
                                  stop_loss_pct=0.05),
                      feature_columns=["f0", "f1"]).episode_len == 24


def _run_pair264(tmp_path, closes, actions):
    """Production env and Pick #2 env in lockstep, BOTH at 264-step
    episodes: identical actions/prices -> identical equity paths;
    rewards differ by exactly the shaping penalties."""
    frame = _toy_frame(closes)
    votes = np.full(len(frame), 0.5)
    outs = []
    for tag, cls, kw in (
            ("base", TradingEnv,
             {"episode_days": 11, "steps_per_day": 24}),
            ("pick2", DuelLengthShapedTradingEnv, {})):
        broker = PaperBroker(tmp_path / f"{tag}.db",
                             starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = cls(frame, votes, broker,
                  feature_columns=["f0", "f1"], **kw)
        env.reset(seed=0, options={"start": 0})
        eq_path, rewards, steps = [env._prev_eq], [], 0
        for a in actions:
            _, r, done, _, _ = env.step(a)
            rewards.append(r)
            eq_path.append(env._prev_eq)
            steps += 1
            if done:
                break
        outs.append((rewards, eq_path,
                     getattr(env, "shaping_log", None), steps))
    (br, beq, _, bsteps), (sr, seq, slog, ssteps) = outs
    assert beq == seq
    assert bsteps == ssteps
    return br, sr, beq, slog, ssteps


def test_pick2_env_flat_hold_lockstep(tmp_path):
    closes = [100.0] * 400
    br, sr, _, slog, steps = _run_pair264(tmp_path, closes, [0] * 300)
    assert steps == 264  # the episode really spans 264 candles
    assert sr == br
    assert all(p == 0.0 for p in slog)


def test_pick2_env_shaping_lockstep_over_264(tmp_path):
    # Buy at 100, dip to 95 across the long episode, recover late.
    closes = ([100.0] * 10 + [99.0, 97.0, 95.0, 96.0, 98.0, 100.0,
                              103.0] + [103.0] * 300)
    actions = [1] + [0] * 280
    br, sr, eq_path, slog, steps = _run_pair264(
        tmp_path, closes, actions)
    assert steps == 264
    pen = shaping_penalties(np.array(eq_path))
    assert len(pen) == len(br) == 264
    assert any(p > 0 for p in pen), "the dip must engage the shaping"
    assert slog == pytest.approx(pen)
    for b, s, p in zip(br, sr, pen):
        assert s == pytest.approx(b - p, abs=1e-9)


# ------------------------------------------------------------- feeds
def test_muse_pick2_family_renders_in_index():
    """The two Muse's Pick #2 runs must surface in the models
    index with Kass's ordered label, family code 15, the right
    headline per brain, and their recorded verdicts."""
    from dashboards import compact

    assert compact._FAMILY_CODES["muse_pick2"] == 15
    stk = {"id": 999, "ts": "2026-10-27T12:00:00+00:00",
           "kind": "stacker", "promoted": False,
           "metrics": {
               "challenger_acc": 0.53, "champion_acc": 0.5265,
               "challenger_wf_acc": 0.51, "precision": 0.52,
               "folds": 5, "pending_approval": True,
               "reason": "Muse's Pick #2 stacker holdout acc 0.53",
               "family": "muse_pick2",
               "muse_pick2": {"label": "Stacker", "verdict": "PASS",
                              "design": "pick2 design",
                              "n_features": 98}}}
    name = compact._display_name(stk, None)
    assert name == "Muse's Pick #2 Stacker"
    entry = compact._model_entry(stk, name)
    assert entry[2] == name and entry[3] == 15
    assert entry[4] == 0  # never promoted (promotion lock)
    assert entry[5] == 0.53 and entry[6] is None
    assert entry[7] == "PASS"
    detail = compact._duel_run(stk)
    assert detail["fam"] == 15 and detail["cfg"] == "pick2 design"
    assert detail["s"] == 0.53 and detail["cs"] == 0.5265

    ppo = {"id": 1000, "ts": "2026-10-27T12:05:00+00:00",
           "kind": "ppo", "promoted": False,
           "metrics": {
               "challenger": {"score": -3.5, "total_pnl": -120.0,
                              "mean_reward": 0.0,
                              "max_drawdown": 100.0,
                              "bench_pnl": -50.0, "episodes": 30,
                              "action_mix": {"hold": 0.6, "buy": 0.1,
                                             "sell": 0.2,
                                             "short": 0.1}},
               "champion": {"score": 1.0, "total_pnl": 10.0},
               "reason": "Muse's Pick #2 PPO duel",
               "family": "muse_pick2",
               "muse_pick2": {"label": "PPO", "verdict": "FAIL",
                              "design": "pick2 ppo design",
                              "n_features": 98}}}
    name = compact._display_name(ppo, None)
    assert name == "Muse's Pick #2 PPO"
    entry = compact._model_entry(ppo, name)
    assert entry[2] == name and entry[3] == 15
    assert entry[5] == -3.5 and entry[6] == -120
    assert entry[7] == "FAIL"
