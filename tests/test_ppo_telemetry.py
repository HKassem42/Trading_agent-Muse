"""PPO telemetry capture (2026-10-26 incorporation, item 6).

Two claims under test:
1. A training run's sidecar carries an honest telemetry summary
   (entropy start/end/min within the mathematical bounds for a
   4-action policy, rollout count, end-of-training gauges).
2. The capture is LOGGING-ONLY: a same-seed micro train with the
   telemetry callback produces a bit-identical learning curve and
   a policy whose action distribution matches the control run (no
   telemetry) to float precision. The callback reads
   model.logger.name_to_value and nothing else -- this test is the
   empirical half of that proof; the diff is the other half.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from models import ppo_train  # noqa: E402


def _tiny_sup(n=600):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    sup = add_features(df).dropna(subset=FEATURE_COLUMNS)
    return sup.reset_index(drop=True)


def _train(tmp_path, telemetry):
    sup = _tiny_sup()
    out = ppo_train.train(
        timesteps=1024, warm_start=False,
        champion_path=tmp_path / "ppo_champion.zip",
        stacker_path=tmp_path / "stacker.pkl",
        db_path=tmp_path / "train.db",
        sup=sup, p_up=np.full(len(sup), 0.5), telemetry=telemetry)
    sidecar = json.loads(
        (tmp_path / "ppo_challenger_curve.json").read_text())
    return out, sidecar


def test_telemetry_summary_is_recorded_and_bounded(tmp_path):
    _out, sidecar = _train(tmp_path, telemetry=True)
    t = sidecar.get("telemetry")
    assert isinstance(t, dict), "telemetry summary missing from sidecar"
    assert t["rollouts"] >= 1
    # 4-action discrete policy: entropy lives in [0, ln 4].
    for key in ("ent_start", "ent_end", "ent_min"):
        assert 0.0 <= t[key] <= math.log(4) + 1e-6, (key, t[key])
    assert t["ent_min"] <= t["ent_start"] + 1e-9
    assert t["ent_min"] <= t["ent_end"] + 1e-9
    assert len(t["ent_curve"]) >= 1
    assert all(0.0 <= e <= math.log(4) + 1e-6 for _s, e in t["ent_curve"])


def test_telemetry_off_writes_null_and_trains_identically(tmp_path):
    control_dir = tmp_path / "control"
    treat_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treat_dir.mkdir()
    out_c, side_c = _train(control_dir, telemetry=False)
    out_t, side_t = _train(treat_dir, telemetry=True)

    assert side_c["telemetry"] is None
    # Identical learning curves: episode returns are a pure
    # function of env dynamics + the policy's RNG consumption;
    # a read-only callback consumes no RNG, so they must match
    # EXACTLY, not approximately.
    assert side_c["curve"] == side_t["curve"]

    from stable_baselines3 import PPO
    mc, mt = PPO.load(out_c), PPO.load(out_t)
    space = mc.observation_space
    space.seed(123)
    obs = np.stack([space.sample() for _ in range(32)]).astype(np.float32)
    import torch
    with torch.no_grad():
        pc = mc.policy.get_distribution(
            mc.policy.obs_to_tensor(obs)[0]).distribution.probs.numpy()
        pt = mt.policy.get_distribution(
            mt.policy.obs_to_tensor(obs)[0]).distribution.probs.numpy()
    assert np.allclose(pc, pt, atol=1e-6), (
        f"policy distributions diverged: max diff "
        f"{np.abs(pc - pt).max()}")
    ac, _ = mc.predict(obs, deterministic=True)
    at, _ = mt.predict(obs, deterministic=True)
    assert (ac == at).all()
