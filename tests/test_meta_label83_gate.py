"""Meta-Label Gate — run 83, the strengthened gate (2026-10-30).

Written BEFORE the implementation, against the design frozen in
refs/meta-label-study-83-2026-10-30.md. Run 83 fixes run 82's
four diagnosed weaknesses: (1) near-chance meta-model eyes,
(2) thin candidate set (one replay), (3) standalone labels
that ignore book context, (4) a single uncalibrated LightGBM.
Pins:

- FIFO tranche attribution (the in-context label): hand-
  computed contributions for adds, partial closes, short
  mirrors, and tranches marked open at episode end (entry fee
  charged, no exit fee — the env's own accounting);
- attempt resolution: an entry attempt whose fill is refused
  opens no tranche and is counted dropped; a filled attempt's
  tranche carries its tag;
- in-context label causality at replay level: garbling every
  candle after a candidate's tranche closed cannot change
  its contribution or label;
- every new context series is causal (trailing windows only):
  truncating or garbling the future leaves past values
  byte-identical — vote slopes, stop distance in ATR units,
  volatility percentile, drawdown-from-high, hour cycle;
- stream-history features read only outcomes closed at or
  before the decision candle (market-time filter), with the
  frozen Laplace smoothing (wins + 2) / (n + 4);
- the 114-vector: run 82's 104 block byte-identical, then the
  10 frozen context values in order, stream id last-but-four
  block exactly as registered;
- the vote-crossing rule policy's truth table;
- training isolation: extracted candidates sit inside the
  region and every outcome closes strictly after its
  decision;
- the model zoo: the selection rule (highest OOF balanced
  accuracy, ties to AUC), calibrated variants produce valid
  probabilities, and the stacker control fits/predicts;
- the 83 mirror evaluator with no gate reproduces
  nightly.evaluate_ppo exactly (the equivalence the duel's
  pass-all pin relies on), and a deny-all gate83 leaves the
  broker flat with attempts counted;
- the family renders run 83's label in the dashboard index.
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
from training import meta_label83 as m83  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


# ------------------------------------------------------------ fixtures
def _sup(n=300, closes=None, atr=0.01):
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


# ------------------------------------------- FIFO tranche attribution
def test_tranche_fifo_long_adds_partial_close():
    tr = m83.TrancheTracker()
    # Entry 1: buy 1.0 @ 100, fee 15. Entry 2 (add): buy 0.5 @ 110, fee 8.25.
    tr.process_step([("BUY", 100.0, 1.0, 15.0, 1)], attempt_tag="a")
    tr.process_step([("BUY", 110.0, 0.5, 8.25, 2)], attempt_tag="b")
    # Sell 1.2 @ 120, fee 21.6: closes tranche a fully, b partially (0.2).
    tr.process_step([("SELL", 120.0, 1.2, 21.6, 3)])
    out = {o["tag"]: o for o in tr.outcomes}
    # a: 1.0*(120-100) - 15 - 21.6*(1.0/1.2) = 20 - 15 - 18 = -13.
    assert out["a"]["contribution"] == pytest.approx(-13.0, abs=1e-9)
    assert out["a"]["label"] == 0
    assert "b" not in out  # still open
    # Episode end marks b's remaining 0.3 @ 130: 0.3*(130-110) - (8.25 - 3.3)
    # - already-realized part: b realized 0.2*(120-110) - 3.3 - 3.6 = -4.9;
    # marked part 0.3*20 - 4.95 = +1.05; total -3.85.
    tr.end_episode(mark_price=130.0, mark_pos=24)
    out = {o["tag"]: o for o in tr.outcomes}
    assert out["b"]["contribution"] == pytest.approx(-3.85, abs=1e-9)
    assert out["b"]["label"] == 0
    assert out["b"]["close_pos"] == 24


def test_tranche_short_mirror_win_and_loss():
    tr = m83.TrancheTracker()
    tr.process_step([("SELL", 100.0, 1.0, 15.0, 1)], attempt_tag="s")
    tr.process_step([("BUY", 90.0, 1.0, 13.5, 5)])
    out = {o["tag"]: o for o in tr.outcomes}
    # Short gains as price falls: 1.0*(90-100)*(-1) - 15 - 13.5 = -18.5.
    assert out["s"]["contribution"] == pytest.approx(-18.5, abs=1e-9)
    assert out["s"]["label"] == 0

    tr2 = m83.TrancheTracker()
    tr2.process_step([("SELL", 100.0, 1.0, 15.0, 1)], attempt_tag="s")
    tr2.process_step([("BUY", 80.0, 1.0, 12.0, 5)])
    out2 = {o["tag"]: o for o in tr2.outcomes}
    # 1.0*(80-100)*(-1) - 15 - 12 = +20 - 27 = ... 20 - 27 = -7? No:
    # contribution = qty*(exit-entry)*sign - fees = 1.0*(80-100)*(-1) - 27 = -7.
    assert out2["s"]["contribution"] == pytest.approx(-7.0, abs=1e-9)


def test_tranche_open_mark_charges_entry_fee_only():
    tr = m83.TrancheTracker()
    tr.process_step([("BUY", 100.0, 1.0, 15.0, 1)], attempt_tag="x")
    tr.end_episode(mark_price=110.0, mark_pos=24)
    out = {o["tag"]: o for o in tr.outcomes}
    # Marked: 1.0*(110-100) - 15 (entry fee) and NO exit fee = -5.
    assert out["x"]["contribution"] == pytest.approx(-5.0, abs=1e-9)
    assert out["x"]["label"] == 0
    tr2 = m83.TrancheTracker()
    tr2.process_step([("BUY", 100.0, 1.0, 15.0, 1)], attempt_tag="x")
    tr2.end_episode(mark_price=120.0, mark_pos=24)
    out2 = {o["tag"]: o for o in tr2.outcomes}
    assert out2["x"]["contribution"] == pytest.approx(5.0, abs=1e-9)
    assert out2["x"]["label"] == 1


def test_attempt_resolution_dropped_and_tagged():
    tr = m83.TrancheTracker()
    # Attempt with NO fill at all (refused): dropped, counted.
    tr.process_step([], attempt_tag="refused")
    assert tr.dropped_attempts == 1
    assert tr.outcomes == []
    # Attempt whose step's fills are only a stop on an older tranche:
    tr.process_step([("BUY", 100.0, 1.0, 15.0, 1)], attempt_tag="first")
    tr.process_step([("SELL", 95.0, 1.0, 14.25, 2)], attempt_tag="second")
    # 'second' produced no tranche of its own -> dropped; 'first' closed.
    assert tr.dropped_attempts == 2
    out = {o["tag"]: o for o in tr.outcomes}
    assert set(out) == {"first"}
    assert out["first"]["contribution"] == pytest.approx(
        1.0 * (95.0 - 100.0) - 15.0 - 14.25, abs=1e-9)


# ------------------------------------------------ stream-history feats
def test_stream_win_features_market_time_filter():
    outcomes = [(10, 1), (12, 0), (20, 1), (30, 1), (40, 0)]
    # At decision 20: outcomes with close_pos <= 20 -> [1, 0, 1].
    w20, wexp = m83.stream_win_features(outcomes, 20)
    assert wexp == pytest.approx((2 + 2) / (3 + 4))
    assert w20 == pytest.approx((2 + 2) / (3 + 4))
    # At decision 9: no history -> the neutral prior 0.5.
    assert m83.stream_win_features(outcomes, 9) == (0.5, 0.5)
    # At decision 100: all five; last-20 == all five here.
    w20, wexp = m83.stream_win_features(outcomes, 100)
    assert wexp == pytest.approx((3 + 2) / (5 + 4))
    # Last-20 window really is a window: 25 outcomes, only last 20 count.
    many = [(i, 1 if i >= 6 else 0) for i in range(1, 26)]
    w20, wexp = m83.stream_win_features(many, 10_000)
    assert w20 == pytest.approx((20 + 2) / (20 + 4))   # last 20 all wins
    assert wexp == pytest.approx((20 + 2) / (25 + 4))  # 20 wins of 25


# ------------------------------------------------- causal context series
def test_extra_context_series_are_causal():
    rng = np.random.default_rng(4)
    n = 300
    votes = rng.uniform(0.3, 0.7, n)
    atr = rng.uniform(0.005, 0.03, n)
    vol = rng.uniform(0.0, 0.8, n)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    hours = (np.arange(n) % 24).astype(float)
    full = m83.build_extra_context(votes, atr, vol, closes, hours)
    T = 200
    cut = m83.build_extra_context(votes[:T], atr[:T], vol[:T],
                                  closes[:T], hours[:T])
    for key, arr in full.items():
        assert np.array_equal(arr[:T], cut[key]), key
    # Garbling the future cannot move the past either.
    v2, a2, vo2, c2 = votes.copy(), atr.copy(), vol.copy(), closes.copy()
    v2[T:] = 0.01
    a2[T:] = 0.5
    vo2[T:] = 99.0
    c2[T:] = 1.0
    garb = m83.build_extra_context(v2, a2, vo2, c2, hours)
    for key, arr in full.items():
        assert np.array_equal(arr[:T], garb[key][:T]), key


def test_context_series_values():
    votes = np.array([0.50, 0.52, 0.55, 0.60])
    slopes6 = m83.vote_slopes(votes, 6)
    assert slopes6[0] == 0.0 and slopes6[3] == 0.0  # before region start
    v2 = np.arange(30, dtype=float) * 0.01 + 0.4
    s6 = m83.vote_slopes(v2, 6)
    assert s6[10] == pytest.approx(v2[10] - v2[4])
    # Stop distance: atr 0.01 -> stop_pct = 2*0.01 floored to 0.025 -> 2.5 units.
    units = m83.stop_distance_atr_units(np.array([0.01, 0.03, 0.0]))
    assert units[0] == pytest.approx(2.5)
    assert units[1] == pytest.approx(2.0)   # 2*0.03 = 0.06, in band
    assert units[2] == 0.0                  # invalid atr -> 0.0
    # Drawdown from the 100-high: monotone fall over exactly
    # one window -> the last value reads the series peak.
    closes = np.linspace(100.0, 50.0, 100)
    dd = m83.drawdown_from_high(closes, 100)
    assert dd[-1] == pytest.approx(50.0 / 100.0 - 1.0)
    assert (dd <= 0).all()
    # Hour cycle: hour 6 -> sin = 1.
    s, c = m83.hour_sincos(np.array([6.0, 0.0]))
    assert s[0] == pytest.approx(1.0) and c[1] == pytest.approx(1.0)


def test_feature_row_83_layout():
    rng = np.random.default_rng(8)
    n = 60
    frame = rng.uniform(0, 100, size=(n, 98))
    votes = rng.uniform(0.3, 0.7, n)
    atr_rel = rng.uniform(-0.5, 0.5, n)
    vol21 = rng.uniform(0.0, 0.8, n)
    atr_raw = rng.uniform(0.005, 0.03, n)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    hours = (np.arange(n) % 24).astype(float)
    extras = m83.build_extra_context(votes, atr_raw, vol21, closes, hours)
    idx, side = 30, -1
    base = ml.feature_row(frame, votes, atr_rel, vol21, atr_raw, idx, side)
    row = m83.feature_row_83(frame, votes, atr_rel, vol21, atr_raw,
                             extras, idx, side, 0.61, 0.57, 1.0)
    assert row.shape == (114,)
    assert np.array_equal(row[:104], base)
    assert row[104] == pytest.approx(extras["slope6"][idx])
    assert row[105] == pytest.approx(extras["slope24"][idx])
    assert row[106] == pytest.approx(0.61)
    assert row[107] == pytest.approx(0.57)
    assert row[108] == pytest.approx(extras["stop_units"][idx])
    assert row[109] == pytest.approx(extras["vol_pct"][idx])
    assert row[110] == pytest.approx(extras["dd100"][idx])
    assert row[111] == pytest.approx(extras["hour_sin"][idx])
    assert row[112] == pytest.approx(extras["hour_cos"][idx])
    assert row[113] == 1.0


# ------------------------------------------------------- rule policy
def test_rule_action_truth_table():
    votes = np.array([0.50, 0.56, 0.58, 0.44, 0.43, 0.57])
    assert m83.rule_action(votes, 1, 0.0) == 1    # cross up, flat -> buy
    assert m83.rule_action(votes, 2, 0.25) == 0   # no cross, long -> hold
    assert m83.rule_action(votes, 3, 0.25) == 2   # cross down, long -> sell
    assert m83.rule_action(votes, 3, 0.0) == 3    # cross down, flat -> short
    assert m83.rule_action(votes, 4, -0.25) == 0  # no cross, short -> hold
    assert m83.rule_action(votes, 5, -0.25) == 1  # cross up, short -> cover
    assert m83.rule_action(votes, 5, 0.0) == 1    # cross up, flat -> buy
    assert m83.rule_action(votes, 0, 0.0) == 0    # no predecessor -> hold


# ------------------------------------------------- replay-level causality
def test_incontext_label_survives_future_garble(tmp_path):
    rng = np.random.default_rng(21)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 264)))
    sup = _sup(264, closes)
    votes = np.full(264, 0.6)

    def run(frame, tag):
        db = tmp_path / f"{tag}.db"
        broker = PaperBroker(db, starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = ml.GatedTradingEnv(frame, votes, broker,
                                 fill_mode="next_open", recorder=[])
        tracker = m83.TrancheTracker()
        reader = m83.LedgerFillReader(db)
        rec = m83.run_policy_episode(
            env, lambda obs: 1, tracker, reader,
            mark_price=float(frame["close"].iloc[24]), mark_pos=24)
        return rec, tracker

    rec1, tr1 = run(sup, "a")
    assert rec1, "always-buy must attempt entries"
    out1 = {o["tag"]: o for o in tr1.outcomes}
    first = rec1[0]
    assert first in out1
    close_pos = out1[first]["close_pos"]
    # Garble everything after that tranche closed; its outcome is final.
    closes2 = closes.copy()
    closes2[close_pos + 1:] = rng.uniform(1.0, 500.0,
                                          len(closes) - close_pos - 1)
    _rec2, tr2 = run(_sup(264, closes2), "b")
    out2 = {o["tag"]: o for o in tr2.outcomes}
    assert out2[first]["contribution"] == pytest.approx(
        out1[first]["contribution"], abs=1e-9)
    assert out2[first]["label"] == out1[first]["label"]


def test_extraction_isolation_on_synthetic_slice(tmp_path):
    rng = np.random.default_rng(33)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 264)))
    sup = _sup(264, closes)
    votes = rng.uniform(0.35, 0.65, 264)
    cands, outcomes, dropped = m83.extract_stream_episodes(
        sup, votes, lambda obs: 1, tmp_path / "iso.db", offsets=(0,))
    assert cands, "always-buy over one block must yield candidates"
    ml.validate_candidates([(c["decision_pos"], c["side"])
                            for c in cands], 264)
    by_tag = {o["tag"]: o for o in outcomes}
    for c in cands:
        o = by_tag[c["tag"]]
        assert o["close_pos"] > c["decision_pos"]
        assert o["close_pos"] <= 264


# ------------------------------------------------------------ model zoo
def test_select_variant_rule():
    table = {
        "lgbm82": {"balanced_accuracy": 0.60, "auc": 0.61},
        "stacker": {"balanced_accuracy": 0.60, "auc": 0.66},
        "lgbm82+platt": {"balanced_accuracy": 0.59, "auc": 0.99},
    }
    # Tie on balanced accuracy -> higher AUC wins (stacker over lgbm82).
    assert m83.select_variant(table) == "stacker"
    table["lgbm82+platt"]["balanced_accuracy"] = 0.61
    assert m83.select_variant(table) == "lgbm82+platt"


def test_oof_and_final_variants_on_separable_data():
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(0, 1, size=(n, 114))
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    X[: n // 2, 108] = 2.5   # stop-units column separates the classes
    X[n // 2:, 108] = 0.5
    proba = m83.oof_proba_base("lgbm82", X, y, folds=4)
    metrics = m83.oof_metrics(proba, y)
    assert metrics["balanced_accuracy"] > 0.85
    for variant in ("lgbm82", "lgbm82+platt", "lgbm82+isotonic"):
        model = m83.fit_final_variant(variant, X, y)
        p = model.predict_proba(X[:5])
        assert p.shape == (5, 2)
        assert (p >= 0).all() and (p <= 1).all()
        assert list(model.classes_) == [0, 1]


def test_stacker_control_fits_and_predicts():
    rng = np.random.default_rng(12)
    n = 400
    X = rng.normal(0, 1, size=(n, 114))
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    X[: n // 2, 100] += 1.5
    X[n // 2:, 100] -= 1.5
    model = m83.fit_base_model("stacker", X, y)
    p = model.predict_proba(X[:4])
    assert p.shape == (4, 2)


# ------------------------------------------------------- gate + mirror
class _StubModel:
    def __init__(self, p):
        self.classes_ = np.array([0, 1])
        self._p = p

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        return np.tile([1 - self._p, self._p], (len(X), 1))


def _gate83(model, n=264, threshold=0.55):
    rng = np.random.default_rng(6)
    frame = rng.uniform(0, 100, size=(n, 98))
    votes = np.full(n, 0.6)
    atr_rel = np.zeros(n)
    vol21 = np.full(n, 0.3)
    atr_raw = np.full(n, 0.01)
    closes = np.full(n, 100.0)
    hours = (np.arange(n) % 24).astype(float)
    extras = m83.build_extra_context(votes, atr_raw, vol21, closes, hours)
    gate = m83.MetaGate83(model, threshold=threshold)
    gate.set_window(0, frame, votes, atr_rel, vol21, atr_raw, extras)
    return gate


def test_gate83_deny_all_leaves_broker_flat(tmp_path):
    sup = _sup(264)
    votes = np.full(264, 0.6)
    gate = _gate83(_StubModel(0.10))
    res = m83.evaluate_gated_ppo_83(
        _tiny_ppo(tmp_path), sup, votes, gate=gate,
        ledger_dir=tmp_path / "led", tag="deny")
    assert res["gate_stats"]["entries_attempted"] > 0
    assert res["gate_stats"]["entries_vetoed"] == \
        res["gate_stats"]["entries_attempted"]
    assert res["gate_stats"]["fills"] == 0
    assert res["total_pnl"] == pytest.approx(0.0, abs=1e-9)


def test_mirror_evaluator_83_matches_production(tmp_path):
    from training import nightly

    rng = np.random.default_rng(9)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, 264)))
    sup = _sup(264, closes)
    votes = rng.uniform(0.35, 0.65, 264)
    path = _tiny_ppo(tmp_path, sup=sup, votes=votes)
    prod = nightly.evaluate_ppo(path, sup, votes)
    mine = m83.evaluate_gated_ppo_83(path, sup, votes, gate=None,
                                     ledger_dir=tmp_path / "led", tag="t")
    for k in ("total_pnl", "max_drawdown", "score", "mean_reward",
              "bench_pnl", "round_trips_per_day"):
        assert mine[k] == pytest.approx(prod[k], abs=1e-9), k
    assert mine["action_mix"] == prod["action_mix"]


def _tiny_ppo(tmp_path, sup=None, votes=None):
    from stable_baselines3 import PPO

    if sup is None:
        sup = _sup(264)
        votes = np.full(264, 0.55)
    broker = PaperBroker(tmp_path / "mk.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, votes, broker, fill_mode="next_open")
    model = PPO("MlpPolicy", env, seed=42)
    path = tmp_path / "tiny.zip"
    model.save(path)
    return path


# ------------------------------------------------------------- family
def test_meta_label_83_family_renders_in_the_index():
    from dashboards import compact

    run = {"id": 999, "ts": "2026-10-30T00:00:00+00:00",
           "kind": "ppo", "promoted": 0,
           "metrics": {
               "family": "meta_label",
               "challenger": {"score": 12.3, "total_pnl": 456.7},
               "champion": {"score": 9.9, "total_pnl": 100.0},
               "meta_label": {"label": "Pick A-83", "verdict": "FAIL",
                              "design": "meta-label gate 83"}}}
    assert compact._display_name(run, None) == "Meta-Label Gate — Pick A-83"
    entry = compact._model_entry(run, "Meta-Label Gate — Pick A-83")
    assert entry[3] == 17
    assert entry[7] == "FAIL"
