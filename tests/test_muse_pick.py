"""Muse's Pick challenger (Kass's order, 2026-10-26): the lab study
drawn from his Drive sheet "Machine Learning and Reinforcement
Learning Kinds, Definitions, and Use Cases" -- Supervised Learning +
Reinforcement Learning, the Reward Shaping technique, Stable
Baselines3.

Two artifacts under test:
1. The pick stacker pipeline: the production stacker recipe
   (scaler -> SMOTE -> RF + XGB + LGBM -> LR) with ONE added base
   learner, ExtraTrees, for diversity.
2. The shaped reward: standard dense MtM reward MINUS 0.5 x the
   step's increase in current-episode drawdown (% points, floored
   at 0), applied by a TradingEnv SUBCLASS used only by the study
   (the production env is never modified).

The shaping math is pinned against a HAND-COMPUTED toy episode:
scripted equity path [100000, 110000, 99000, 105000, 90000] has
running peaks [100000, 110000, 110000, 110000, 110000], drawdowns
[0, 0, 10, 50/11, 200/11] % and per-step increases in the CURRENT
drawdown [0, 10, 0, 200/11 - 50/11] -- penalties are half of those.
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
from models import muse_pick  # noqa: E402
from models.muse_pick import (  # noqa: E402
    ShapedTradingEnv, build_pick_pipeline, crossfit_p_up_pick,
    drawdown_increases, shaping_penalties)
from trading.broker import PaperBroker  # noqa: E402


# ------------------------------------------------------------- stacker
def test_pick_pipeline_is_production_recipe_plus_extratrees():
    pipe = build_pick_pipeline(random_state=42)
    assert [name for name, _ in pipe.steps] == ["scaler", "smote", "stack"]
    stack = pipe.named_steps["stack"]
    names = {name for name, _ in stack.estimators}
    assert names == {"rf", "xgb", "lgbm", "et"}
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    et = dict(stack.estimators)["et"]
    assert isinstance(et, ExtraTreesClassifier)
    assert isinstance(stack.final_estimator, LogisticRegression)
    # The production three keep their production hyperparameters.
    prod = __import__("models.stacker", fromlist=["build_pipeline"])
    prod_stack = prod.build_pipeline(random_state=42).named_steps["stack"]
    for name in ("rf", "xgb", "lgbm"):
        a = dict(stack.estimators)[name].get_params()
        b = dict(prod_stack.estimators)[name].get_params()
        assert a == b, f"{name} hyperparameters drifted from production"


def test_crossfit_pick_votes_leading_neutral_and_bounded():
    rng = np.random.default_rng(7)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    p = crossfit_p_up_pick(X, y, folds=4)
    assert len(p) == n
    assert np.all((p >= 0.0) & (p <= 1.0))
    # TimeSeriesSplit(4) on 240 rows: first test block starts at 48,
    # so the leading 48 votes are the neutral 0.5 (no past to fit on).
    assert (p[:48] == 0.5).all()
    assert (p[48:] != 0.5).any()


# ------------------------------------------------------------- shaping
def test_drawdown_increases_hand_computed():
    eq = [100_000.0, 110_000.0, 99_000.0, 105_000.0, 90_000.0]
    inc = drawdown_increases(eq)
    assert len(inc) == 4
    assert inc[0] == pytest.approx(0.0)
    assert inc[1] == pytest.approx(10.0)
    assert inc[2] == pytest.approx(0.0)  # partial recovery: no NEW dd
    # The final leg falls from dd 50/11 to dd 200/11: the step's
    # increase in current drawdown is the difference, 150/11.
    assert inc[3] == pytest.approx(150.0 / 11.0)
    pen = shaping_penalties(eq)
    assert pen == pytest.approx([0.0, 5.0, 0.0, 0.5 * (150.0 / 11.0)])


def test_shaping_is_zero_without_drawdown():
    rising = [100_000.0, 100_500.0, 101_000.0, 102_000.0]
    flat = [100_000.0] * 5
    assert shaping_penalties(rising) == [0.0, 0.0, 0.0]
    assert shaping_penalties(flat) == [0.0, 0.0, 0.0, 0.0]
    # A dip fully recovered before a new peak adds nothing further.
    dip_recover = [100_000.0, 90_000.0, 100_000.0, 95_000.0]
    assert shaping_penalties(dip_recover) == pytest.approx([5.0, 0.0, 2.5])


# ------------------------------------------------------------- env
def _toy_frame(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "open": closes, "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes], "close": closes,
        "f0": np.zeros(n), "f1": np.zeros(n),
    })


def _run_pair(tmp_path, closes, actions):
    """Run the production env and the shaped subclass in lockstep
    (identical actions, identical prices, separate brokers) and
    return (base_rewards, shaped_rewards, equity_path, penalties)."""
    frame = _toy_frame(closes)
    votes = np.full(len(frame), 0.5)
    outs = []
    for tag, cls in (("base", TradingEnv), ("shaped", ShapedTradingEnv)):
        broker = PaperBroker(tmp_path / f"{tag}.db",
                             starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = cls(frame, votes, broker, feature_columns=["f0", "f1"])
        env.reset(seed=0, options={"start": 0})
        eq_path = [env._prev_eq]
        rewards = []
        for a in actions:
            _, r, done, _, _ = env.step(a)
            rewards.append(r)
            eq_path.append(env._prev_eq)
            if done:
                break
        outs.append((rewards, eq_path,
                     getattr(env, "shaping_log", None)))
    (br, beq, _), (sr, seq, slog) = outs
    assert beq == seq  # identical trading -> identical equity path
    return br, sr, beq, slog


def test_shaped_env_all_hold_reduces_to_unshaped(tmp_path):
    closes = [100.0 - i for i in range(30)]  # falling market, no trades
    br, sr, _, slog = _run_pair(tmp_path, closes, [0] * 24)
    assert sr == br
    assert all(p == 0.0 for p in slog)


def test_shaped_env_lockstep_matches_pure_function(tmp_path):
    # Buy at 100, dip to 96 (drawdown grows), recover through 104.
    closes = ([100.0, 99.0, 97.0, 96.0, 96.5, 98.0, 100.0, 102.0, 104.0]
              + [104.0] * 21)
    actions = [1] + [0] * 23
    br, sr, eq_path, slog = _run_pair(tmp_path, closes, actions)
    pen = shaping_penalties(np.array(eq_path))
    assert len(pen) == len(br) == 24
    assert any(p > 0 for p in pen), "the dip must engage the shaping"
    assert slog == pytest.approx(pen)
    for b, s, p in zip(br, sr, pen):
        assert s == pytest.approx(b - p, abs=1e-9)


def test_shaped_env_is_a_subclass_and_production_env_untouched():
    assert issubclass(ShapedTradingEnv, TradingEnv)
    assert ShapedTradingEnv.SHAPING_COEF == 0.5
    # The production class carries no shaping state of its own.
    assert not hasattr(TradingEnv, "SHAPING_COEF")


# ------------------------------------------------------------- feeds
def test_muse_pick_family_renders_in_index():
    """The two Muse's Pick runs must surface in the models index
    with the label Kass ordered, family code 14, the right headline
    per brain (stacker: holdout accuracy, no P&L; PPO: duel score
    and whole-dollar P&L), and their recorded verdicts."""
    from dashboards import compact

    stk = {"id": 999, "ts": "2026-10-26T22:00:00+00:00",
           "kind": "stacker", "promoted": False,
           "metrics": {
               "challenger_acc": 0.5012, "champion_acc": 0.4975,
               "challenger_wf_acc": 0.51, "precision": 0.49, "folds": 5,
               "pending_approval": False,
               "reason": "Muse's Pick stacker holdout acc 0.5012",
               "family": "muse_pick",
               "muse_pick": {"label": "Stacker", "verdict": "PASS",
                             "design": "pick design",
                             "n_features": 94}}}
    name = compact._display_name(stk, None)
    assert name == "Muse's Pick Stacker"
    entry = compact._model_entry(stk, name)
    assert entry[2] == name and entry[3] == 14
    assert entry[4] == 0  # never promoted (promotion lock)
    assert entry[5] == 0.5012 and entry[6] is None
    assert entry[7] == "PASS"
    detail = compact._duel_run(stk)
    assert detail["fam"] == 14 and detail["cfg"] == "pick design"
    assert detail["s"] == 0.5012 and detail["cs"] == 0.4975

    ppo = {"id": 1000, "ts": "2026-10-26T22:05:00+00:00",
           "kind": "ppo", "promoted": False,
           "metrics": {
               "challenger": {"score": 12.3, "total_pnl": 456.7,
                              "mean_reward": 0.1, "max_drawdown": 100.0,
                              "bench_pnl": -50.0, "episodes": 30,
                              "action_mix": {"hold": 0.5, "buy": 0.2,
                                             "sell": 0.2, "short": 0.1}},
               "champion": {"score": 1.0, "total_pnl": 10.0},
               "reason": "Muse's Pick PPO duel",
               "family": "muse_pick",
               "muse_pick": {"label": "PPO", "verdict": "FAIL",
                             "design": "pick ppo design",
                             "n_features": 94}}}
    name = compact._display_name(ppo, None)
    assert name == "Muse's Pick PPO"
    entry = compact._model_entry(ppo, name)
    assert entry[2] == name and entry[3] == 14
    assert entry[5] == 12.3 and entry[6] == 457
    assert entry[7] == "FAIL"
