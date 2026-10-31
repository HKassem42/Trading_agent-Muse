"""Meta-Label Gate — Pick A (2026-10-30).

Written BEFORE the implementation, against the design frozen in
refs/meta-label-study-2026-10-30.md. The system: the champion
pair (PPO run 44 + stacker run 73 votes) proposes entries; a
LightGBM meta-model predicts "will THIS candidate trade make
money after fees" from the pick2_98 row at the decision candle
+ context, and VETOES entries whose P(profitable) < 0.55.
Pins:

- entry-attempt classification (buy/short are entries only when
  flat-or-same-side; covers and through-flat closes never are);
- label causality: garbling every row after a candidate's exit
  cannot change its label, P&L, fees or exit (truncation form
  too: the label on data cut at the exit candle is identical);
- feature-at-entry purity: the 104-vector at a decision candle
  is unchanged when every later row of every input array is
  replaced with garbage (no post-entry data in features);
- the gate actually vetoes: a deny-all gate leaves the broker
  flat with an empty ledger while attempts are counted; a
  meta-model trained on separable synthetic candidates vetoes
  the always-lose region at the graded 0.55 threshold;
- gated-env equivalence: with no gate, GatedTradingEnv replays
  TradingEnv step-for-step (same rewards, same fills) — the
  property the duel's pass-all mirror assertion relies on;
- mirror-evaluator equivalence: evaluate_gated_ppo with no
  gate reproduces nightly.evaluate_ppo's window dict exactly
  on a tiny real (untrained) PPO;
- training-region isolation: candidate positions validate
  against the region bound, and the extraction+label pipeline
  on a synthetic slice never reads past its end;
- the ledger re-sum recomputes an episode's P&L from fill
  arithmetic alone and audits the broker's recorded
  cash_after/btc_after on every fill;
- the family renders in the dashboards' index (code 17, real
  name, recorded verdict) — the muse_pick3 lesson.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from training import meta_label as ml  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

FEE = 0.0015


# ------------------------------------------------------------ fixtures
def _sup(n=300, closes=None, atr=0.01):
    """A production-shaped supervised frame: OHLCV + the 21
    features (constant) + atr_pct_raw + target."""
    if closes is None:
        closes = np.full(n, 100.0)
    closes = np.asarray(closes, dtype=float)
    df = pd.DataFrame(index=pd.RangeIndex(n))
    df["open"] = closes.copy()
    df["high"] = closes * 1.001
    df["low"] = closes * 0.999
    df["close"] = closes.copy()
    df["volume"] = 1.0
    for c in FEATURE_COLUMNS:
        df[c] = 0.0
    df["atr_pct_raw"] = atr
    df["target"] = 0
    return df


def _arrays(sup):
    return {k: sup[k].to_numpy(dtype=float)
            for k in ("open", "high", "low", "close", "atr_pct_raw")}


def _ctx(n, seed=0):
    """Aligned meta-feature inputs: 98-col frame values, votes,
    and the production context series."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 100, size=(n, 98)),
            rng.uniform(0.3, 0.7, size=n),
            rng.uniform(-0.5, 0.5, size=n),
            rng.uniform(0.0, 0.8, size=n),
            np.full(n, 0.01))


# ------------------------------------------------------- classification
def test_entry_attempt_truth_table():
    assert ml.is_entry_attempt(1, 0.0) == 1     # buy, flat -> long entry
    assert ml.is_entry_attempt(1, 0.25) == 1    # buy, long -> add
    assert ml.is_entry_attempt(1, -0.25) == 0   # buy, short -> cover
    assert ml.is_entry_attempt(3, 0.0) == -1    # short, flat -> short entry
    assert ml.is_entry_attempt(3, -0.25) == -1  # short, short -> add
    assert ml.is_entry_attempt(3, 0.25) == 0    # short, long -> close
    assert ml.is_entry_attempt(2, 0.25) == 0    # sell is management
    assert ml.is_entry_attempt(2, 0.0) == 0
    assert ml.is_entry_attempt(0, 0.0) == 0     # hold is nothing


# ------------------------------------------------------------ labels
def _cliff_arrays(n=120, drop_at=60):
    """Flat 100s, then a cliff to 80 at `drop_at`: a long entered
    before the cliff is stopped out (8% cap) on the cliff candle;
    a short entered before the cliff rides the trailing stop."""
    closes = np.full(n, 100.0)
    closes[drop_at:] = 80.0
    sup = _sup(n, closes)
    return _arrays(sup)


def test_label_long_stopped_on_cliff(tmp_path):
    arr = _cliff_arrays()
    out = ml.simulate_candidate_label(arr, 10, +1, tmp_path / "a.db")
    assert out is not None
    assert out["label"] == 0 and out["pnl"] < 0
    assert out["exit_reason"] == "stop_loss"
    assert out["exit_pos"] == 60  # the cliff candle itself
    # Fees on both sides are inside the P&L: entry ~25% of 100k.
    assert out["fees"] > 30.0


def test_label_short_wins_on_cliff(tmp_path):
    arr = _cliff_arrays()
    out = ml.simulate_candidate_label(arr, 10, -1, tmp_path / "a.db")
    assert out is not None
    assert out["label"] == 1 and out["pnl"] > 0
    assert out["exit_reason"] in ("trailing_stop", "horizon")


def test_label_causality_garble_after_exit(tmp_path):
    arr = _cliff_arrays()
    full = ml.simulate_candidate_label(arr, 10, +1, tmp_path / "a.db")
    garbled = {k: v.copy() for k, v in arr.items()}
    rng = np.random.default_rng(7)
    for k in garbled:
        garbled[k][full["exit_pos"] + 1:] = rng.uniform(
            1.0, 500.0, size=len(garbled[k]) - full["exit_pos"] - 1)
    out = ml.simulate_candidate_label(garbled, 10, +1, tmp_path / "b.db")
    assert out["label"] == full["label"]
    assert out["pnl"] == pytest.approx(full["pnl"], abs=1e-9)
    assert out["fees"] == pytest.approx(full["fees"], abs=1e-9)
    assert out["exit_pos"] == full["exit_pos"]
    assert out["exit_reason"] == full["exit_reason"]


def test_label_causality_truncation(tmp_path):
    arr = _cliff_arrays()
    full = ml.simulate_candidate_label(arr, 10, +1, tmp_path / "a.db")
    cut = full["exit_pos"] + 1
    trunc = {k: v[:cut] for k, v in arr.items()}
    out = ml.simulate_candidate_label(trunc, 10, +1, tmp_path / "b.db")
    assert out["label"] == full["label"]
    assert out["pnl"] == pytest.approx(full["pnl"], abs=1e-9)
    assert out["exit_pos"] == full["exit_pos"]


def test_label_horizon_close_when_no_stop(tmp_path):
    arr = _arrays(_sup(80))  # dead flat: no stop can ever fire
    out = ml.simulate_candidate_label(arr, 5, +1, tmp_path / "a.db")
    assert out["exit_reason"] == "horizon"
    assert out["exit_pos"] == 79
    # A flat round trip loses exactly the two sides' fees.
    assert out["label"] == 0 and out["pnl"] < 0
    assert out["fees"] == pytest.approx(out["pnl"] * -1, rel=0.02)


# ----------------------------------------------------------- features
def test_feature_row_is_entry_local():
    n = 60
    frame, votes, atr_rel, vol21, atr_raw = _ctx(n)
    idx, side = 25, -1
    base = ml.feature_row(frame, votes, atr_rel, vol21, atr_raw,
                          idx, side)
    assert base.shape == (104,)
    assert base[-1] == -1.0
    assert base[98] == pytest.approx(votes[idx])
    assert base[99] == pytest.approx(abs(votes[idx] - 0.5))
    # Garble EVERYTHING after the decision candle: the vector at
    # the decision candle must not move.
    rng = np.random.default_rng(3)
    frame2, votes2 = frame.copy(), votes.copy()
    atr_rel2, vol212, atr_raw2 = atr_rel.copy(), vol21.copy(), atr_raw.copy()
    frame2[idx + 1:] = rng.uniform(-999, 999, frame2[idx + 1:].shape)
    votes2[idx + 1:] = rng.uniform(-9, 9, n - idx - 1)
    atr_rel2[idx + 1:] = 777.0
    vol212[idx + 1:] = 777.0
    atr_raw2[idx + 1:] = 777.0
    out = ml.feature_row(frame2, votes2, atr_rel2, vol212, atr_raw2,
                         idx, side)
    assert np.array_equal(out, base)


# --------------------------------------------------------------- gate
def test_gate_vetoes_everything_when_denied(tmp_path):
    sup = _sup(264)
    votes = np.full(264, 0.6)
    broker = PaperBroker(tmp_path / "g.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = ml.GatedTradingEnv(sup, votes, broker, fill_mode="next_open",
                             allow_fn=lambda i, s: False)
    obs, _ = env.reset(seed=0, options={"start": 0})
    done = False
    while not done:
        # Alternate buy/short every step: 24 entry attempts.
        act = 1 if env._cursor % 2 == 0 else 3
        obs, _r, done, _, _ = env.step(act)
        assert broker.btc == 0.0
    assert env.entries_attempted == 24
    assert env.entries_vetoed == 24
    assert env.entries_filled == 0
    import sqlite3
    with sqlite3.connect(tmp_path / "g.db") as con:
        assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_gate_allows_and_fills_when_approved(tmp_path):
    sup = _sup(264)
    votes = np.full(264, 0.6)
    broker = PaperBroker(tmp_path / "g.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = ml.GatedTradingEnv(sup, votes, broker, fill_mode="next_open",
                             allow_fn=lambda i, s: True)
    obs, _ = env.reset(seed=0, options={"start": 0})
    obs, _r, _d, _, _ = env.step(1)  # buy: entry attempt, approved
    assert broker.btc > 0.0
    assert env.entries_filled == 1 and env.entries_vetoed == 0
    obs, _r, _d, _, _ = env.step(2)  # sell: management, never gated
    assert broker.btc == 0.0


def test_recorder_sees_attempts_but_never_management(tmp_path):
    sup = _sup(264)
    votes = np.full(264, 0.6)
    rec: list = []
    broker = PaperBroker(tmp_path / "g.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = ml.GatedTradingEnv(sup, votes, broker, fill_mode="next_open",
                             recorder=rec)
    env.reset(seed=0, options={"start": 0})
    env.step(1)   # buy flat -> entry (cursor 0)
    env.step(1)   # buy long -> add = entry (cursor 1)
    env.step(2)   # sell long -> management, NOT recorded
    env.step(3)   # short flat -> entry (cursor 3)
    env.step(1)   # buy short -> cover, NOT recorded
    assert rec == [(0, 1), (1, 1), (3, -1)]


def test_meta_model_vetoes_the_always_lose_region():
    rng = np.random.default_rng(11)
    n = 400
    # Winners carry atr_rel context +0.9, losers -0.9; the other
    # 103 columns are noise. idx of atr_rel in the vector: 98 + 2.
    X = rng.normal(0, 1, size=(n, 104))
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    X[: n // 2, 100] = 0.9
    X[n // 2:, 100] = -0.9
    model = ml.fit_meta_model(X, y)
    # Gate arrays: frame 98 cols of zeros; context built per row.
    m = 6
    frame = np.zeros((m, 98))
    votes = np.full(m, 0.55)
    atr_rel = np.array([0.9, 0.9, 0.9, -0.9, -0.9, -0.9])
    vol21 = np.zeros(m)
    atr_raw = np.full(m, 0.01)
    gate = ml.MetaGate(model, frame, votes, atr_rel, vol21, atr_raw,
                       threshold=0.55)
    assert all(gate.allow(i, +1) for i in range(3))       # winners pass
    assert not any(gate.allow(i, +1) for i in range(3, 6))  # losers vetoed
    assert all(gate.proba(i, +1) < 0.55 for i in range(3, 6))


# --------------------------------------------------------- equivalence
def test_gated_env_without_gate_replays_production_env(tmp_path):
    rng = np.random.default_rng(5)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    sup = _sup(300, closes)
    votes = np.full(300, 0.55)
    actions = [int(a) for a in rng.integers(0, 4, 60)]

    def _run(env_cls, db, **kw):
        broker = PaperBroker(db, starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = env_cls(sup, votes, broker, fill_mode="next_open", **kw)
        env.reset(seed=3, options={"start": 100})
        rewards = []
        for a in actions:
            _o, r, done, _, _ = env.step(a)
            rewards.append(r)
            if done:
                break
        return rewards, broker.cash, broker.btc

    r1 = _run(TradingEnv, tmp_path / "p.db")
    r2 = _run(ml.GatedTradingEnv, tmp_path / "g.db")
    assert r1[0] == pytest.approx(r2[0], abs=1e-12)
    assert r1[1] == pytest.approx(r2[1], abs=1e-9)
    assert r1[2] == pytest.approx(r2[2], abs=1e-12)


def test_mirror_evaluator_matches_production(tmp_path):
    """evaluate_gated_ppo(allow=None) == nightly.evaluate_ppo on
    the same tiny (untrained) PPO, same window, same votes."""
    from stable_baselines3 import PPO

    from training import nightly

    rng = np.random.default_rng(9)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, 264)))
    sup = _sup(264, closes)
    votes = rng.uniform(0.35, 0.65, 264)
    broker = PaperBroker(tmp_path / "mk.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, votes, broker, fill_mode="next_open")
    model = PPO("MlpPolicy", env, seed=42)
    path = tmp_path / "tiny.zip"
    model.save(path)

    prod = nightly.evaluate_ppo(path, sup, votes)
    mine = ml.evaluate_gated_ppo(path, sup, votes, allow_fn=None,
                                 ledger_dir=tmp_path / "led", tag="t")
    for k in ("total_pnl", "max_drawdown", "score", "mean_reward",
              "bench_pnl", "round_trips_per_day"):
        assert mine[k] == pytest.approx(prod[k], abs=1e-9), k
    assert mine["action_mix"] == prod["action_mix"]
    assert mine["episodes"] == prod["episodes"] == 10


# ------------------------------------------------------------- re-sum
def test_resum_episode_pnl_audits_the_ledger(tmp_path):
    sup = _sup(264)
    votes = np.full(264, 0.6)
    db = tmp_path / "ep.db"
    broker = PaperBroker(db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, votes, broker, fill_mode="next_open")
    env.reset(seed=0, options={"start": 0})
    env.step(1)   # buy at candle 1 open
    env.step(0)
    env.step(2)   # sell at candle 3 open -> flat round trip
    # env marked equity at candle 3's close (flat, so = cash).
    expected = broker.cash - 100_000.0
    got = ml.resum_episode_pnl(db, final_close=100.0)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got < 0  # flat round trip = fees only


# --------------------------------------------------------- isolation
def test_validate_candidates_bounds():
    ml.validate_candidates([(0, 1), (263, -1)], 264)
    with pytest.raises(AssertionError):
        ml.validate_candidates([(264, 1)], 264)
    with pytest.raises(AssertionError):
        ml.validate_candidates([(-1, 1)], 264)


def test_pipeline_never_reads_past_the_region(tmp_path):
    """Extraction + labels on a synthetic slice: every candidate
    decision and every label exit stays inside the slice."""
    rng = np.random.default_rng(21)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 264)))
    sup = _sup(264, closes)
    votes = np.full(264, 0.6)
    cands = ml.extract_candidates(sup, votes, lambda obs: 1,
                                  tmp_path / "x.db")
    assert cands, "the always-buy policy must produce candidates"
    ml.validate_candidates(cands, 264)
    frame, v, a_rel, v21, a_raw = _ctx(264)
    X, y, recs = ml.build_candidate_dataset(
        cands, frame, v, a_rel, v21, a_raw, _arrays(sup),
        tmp_path / "lab.db")
    assert X.shape[0] == len(recs) == len(y)
    assert all(0 <= r["exit_pos"] < 264 for r in recs)
    assert all(0 <= r["decision_pos"] < 264 for r in recs)


# ------------------------------------------------------------- family
def test_meta_label_family_renders_in_the_index():
    from dashboards import compact

    assert compact._FAMILY_CODES["meta_label"] == 17
    run = {"id": 999, "ts": "2026-10-30T00:00:00+00:00",
           "kind": "ppo", "promoted": 0,
           "metrics": {
               "family": "meta_label",
               "challenger": {"score": 12.3, "total_pnl": 456.7},
               "champion": {"score": 9.9, "total_pnl": 100.0},
               "meta_label": {"label": "Pick A", "verdict": "PASS",
                              "design": "meta-label gate"}}}
    assert compact._display_name(run, None) == "Meta-Label Gate — Pick A"
    entry = compact._model_entry(run, "Meta-Label Gate — Pick A")
    assert entry[3] == 17
    assert entry[5] == pytest.approx(12.3)
    assert entry[6] == 457
    assert entry[7] == "PASS"
