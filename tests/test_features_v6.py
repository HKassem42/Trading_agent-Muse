"""Curated features v6 (LAB study, 2026-10-25): the production 21
minus the obv_slope/netvol_rel duplicate, plus additions drawn only
from the precomputed indicator universe (the A-Z screen / timeseries
matrices) whose screen-measured max |r| vs the production 21 is
< 0.90. Every v6 column must meet the v5.1 standard: values-based
causal construction (truncation across ALL rows), scalers fitted on
train rows only, all values in [-1, 1].

Also pins the machinery generalization the study rides on:
TradingEnv / evaluate_ppo / ppo_train accept an explicit feature
column list, with the production default byte-identical in shape.

Run: .venv/bin/python -m pytest tests/test_features_v6.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features import features_v6 as v6  # noqa: E402
from features.az_indicators import REGISTRY, compute_az  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402


def _frame(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


ADDED = ["az_adx_14", "az_cci_20", "az_chop_14"]


# ------------------------------------------------------------ curation

def test_retained_drops_only_the_duplicate():
    # obv_slope == netvol_rel at r=0.999 (diagnosis 2026-10-21): the
    # v6 set keeps netvol_rel (the bounded share, defined without the
    # cumulative OBV level) and drops obv_slope.
    assert "obv_slope" not in v6.RETAINED_COLUMNS
    assert "netvol_rel" in v6.RETAINED_COLUMNS
    assert v6.RETAINED_COLUMNS == [c for c in FEATURE_COLUMNS
                                   if c != "obv_slope"]
    assert len(v6.RETAINED_COLUMNS) == 20


def test_every_candidate_is_screened_and_nonredundant():
    """Curation evidence, tied to the A-Z screen's own data file:
    every candidate (primary + fallback) is a registered AZ indicator
    whose measured max |r| against the production 21 is < 0.90."""
    res = json.loads((ROOT / "data" / "az_screen" / "results.json")
                     .read_text())
    corr = res["corr"]
    for name in v6.ADD_CANDIDATES + v6.ADD_FALLBACK:
        assert name in REGISTRY, f"{name} not in the AZ registry"
        assert name in corr, f"{name} was never screened"
        assert corr[name]["max_corr_21"] < 0.90, (
            f"{name} max_corr_21={corr[name]['max_corr_21']:.3f} "
            f"vs {corr[name]['argmax_21']} -- too redundant to add")


# ------------------------------------------------------------ the frame

def test_v6_frame_contract():
    df = _frame()
    sup = v6.make_v6_supervised(df, ADDED, train_end_pos=800)
    cols = v6.v6_columns(ADDED)
    assert len(cols) == 23
    vals = sup[cols].to_numpy()
    assert np.isfinite(vals).all()
    assert vals.min() >= -1.0 and vals.max() <= 1.0
    assert "atr_pct_raw" in sup.columns  # broker exit rules ride along
    assert set(sup["target"].unique()) <= {0, 1}


def test_added_columns_match_az_module_two_ways():
    """The study's subset scaler must reproduce the AZ module's own
    compute_az values for the same columns exactly (two independent
    implementations of the scaling spec)."""
    df = _frame()
    sup = v6.make_v6_supervised(df, ADDED, train_end_pos=800)
    scaled_all, _, _ = compute_az(df, 800)
    for name in ADDED:
        a = sup[name]
        b = scaled_all.loc[a.index, name]
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12), name


def test_v6_truncation_causality_all_rows():
    """v5.1 standard: features computed on a frame truncated at K
    must match the full-frame values for EVERY row whose inputs lie
    inside the truncated frame. Scalers are fit on the first 800 rows
    in both cases (train region precedes every truncation point), so
    any diff is look-ahead contamination."""
    df = _frame()
    cols = v6.v6_columns(ADDED)
    full = v6.make_v6_supervised(df, ADDED, train_end_pos=800)
    for k in (1000, 1200, 1400):
        trunc = v6.make_v6_supervised(df.iloc[:k], ADDED,
                                      train_end_pos=800)
        # Rows up to K-2 (iloc positions): the production rule -- the
        # final row of a truncated frame can sit inside a partial
        # higher-timeframe bucket.
        edge = df.index[k - 2]
        rows = [t for t in trunc.index if t in full.index and t <= edge]
        assert len(rows) > 150
        a = full.loc[rows, cols].to_numpy()
        b = trunc.loc[rows, cols].to_numpy()
        assert np.isfinite(b).all()
        diff = np.abs(a - b)
        assert diff.max() < 1e-9, (
            f"truncation at {k}: max v6 feature diff {diff.max():.4f}")


def test_v6_future_perturbation_cannot_change_the_past():
    """Adversarial: scrambling every candle after row 1200 must not
    move any v6 value at or before row 1200 (scalers fit on the first
    800 rows, so the fit region is untouched by construction)."""
    df = _frame()
    cols = v6.v6_columns(ADDED)
    base = v6.make_v6_supervised(df, ADDED, train_end_pos=800)
    scrambled = df.copy()
    rng = np.random.default_rng(5)
    tail = scrambled.iloc[1201:].copy()
    for col in ("open", "high", "low", "close", "volume"):
        tail[col] = rng.permutation(tail[col].to_numpy())
    scrambled.iloc[1201:] = tail
    pert = v6.make_v6_supervised(scrambled, ADDED, train_end_pos=800)
    rows = [t for t in base.index if t <= df.index[1200]]
    assert len(rows) > 300
    a = base.loc[rows, cols].to_numpy()
    b = pert.loc[rows, cols].to_numpy()
    assert np.abs(a - b).max() < 1e-9


# ------------------------------------------------------- env machinery

def _tiny_sup(n=1500, seed=2):
    # 1500 rows: the production warmup (daily SMA20 etc.) eats ~500,
    # leaving ~1000 supervised rows -- enough for envs and PPO smokes.
    df = _frame(n=n, seed=seed)
    from features.technical_indicators import make_supervised
    return make_supervised(df)


def test_env_default_shape_is_production_25():
    sup = _tiny_sup()
    broker = PaperBroker(Path("/tmp/tam_v6_env_default.db"),
                         starting_balance=100_000)
    env = TradingEnv(sup, np.full(len(sup), 0.5), broker)
    assert env.observation_space.shape == (len(FEATURE_COLUMNS) + 4,)
    assert env.observation_space.shape == (25,)


def test_env_custom_columns_shape_and_state(tmp_path):
    sup = _tiny_sup()
    cols = ["rsi_c", "atr_rel", "ewo"]
    broker = PaperBroker(tmp_path / "env.db", starting_balance=100_000)
    env = TradingEnv(sup, np.full(len(sup), 0.5), broker,
                     feature_columns=cols)
    assert env.observation_space.shape == (len(cols) + 4,)
    obs, _ = env.reset(seed=0, options={"start": 10})
    row = sup.iloc[10]
    expect = np.array([row[c] for c in cols] + [0.0], dtype=np.float32)
    assert np.allclose(obs[: len(cols) + 1], expect, atol=1e-6)
    # A full episode runs on the custom observation width.
    done = False
    while not done:
        obs, _, done, _, _ = env.step(0)
    assert obs.shape == (len(cols) + 4,)


def test_evaluate_ppo_threads_feature_columns(tmp_path):
    """nightly.evaluate_ppo must grade a brain on the feature set it
    was trained on -- the v6 duel depends on it."""
    from stable_baselines3 import PPO

    from training.nightly import evaluate_ppo

    sup = _tiny_sup()
    cols = ["rsi_c", "atr_rel", "ewo", "ha_dir"]
    broker = PaperBroker(tmp_path / "eval.db", starting_balance=100_000)
    env = TradingEnv(sup, np.full(len(sup), 0.5), broker,
                     feature_columns=cols)
    model = PPO("MlpPolicy", env, verbose=0, seed=1)
    path = tmp_path / "tiny.zip"
    model.save(path)
    res = evaluate_ppo(path, sup.iloc[:100].copy(),
                       np.full(100, 0.5), episodes=2,
                       feature_columns=cols)
    assert res["episodes"] == 2
    assert set(res["action_mix"]) == {"hold", "buy", "sell", "short"}
    # And the production default still grades a production-width brain.
    broker2 = PaperBroker(tmp_path / "eval2.db",
                          starting_balance=100_000)
    env2 = TradingEnv(sup, np.full(len(sup), 0.5), broker2)
    m2 = PPO("MlpPolicy", env2, verbose=0, seed=1)
    p2 = tmp_path / "tiny2.zip"
    m2.save(p2)
    res2 = evaluate_ppo(p2, sup.iloc[:100].copy(), np.full(100, 0.5),
                        episodes=2)
    assert res2["episodes"] == 2


def test_ppo_train_threads_feature_columns(tmp_path):
    """ppo_train.train must build its env on the caller's feature
    set (the v6 challenger trains fresh on the wider state)."""
    from models.ppo_train import train

    sup = _tiny_sup()
    cols = ["rsi_c", "atr_rel"]
    out = train(timesteps=600, warm_start=False,
                champion_path=tmp_path / "champ_anchor.zip",
                stacker_path=tmp_path / "no_stacker.pkl",
                db_path=tmp_path / "scratch.db",
                sup=sup, p_up=np.full(len(sup), 0.5),
                feature_columns=cols)
    from stable_baselines3 import PPO as _PPO
    m = _PPO.load(out)
    assert m.observation_space.shape == (len(cols) + 4,)
