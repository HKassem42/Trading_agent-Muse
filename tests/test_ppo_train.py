"""PPO training chunking (pre-flight Finding 2 fix, 2026-10-24).

The weekly deep train (300k steps) used to be ONE uninterrupted
model.learn() with the challenger saved only at the very end -- the
shape refs/diagnosis-2026-10-21.md records was SIGKILLed three times
on this box and only completed as 12 checkpointed 25k chunks. Now:
runs at or above CHUNK_THRESHOLD train in CHUNK_SIZE segments
(reset_num_timesteps=False after the first) and the challenger is
saved after EACH segment, so a kill loses at most one segment.
Below the threshold the call is exactly the old single learn (the
nightly 60k path is untouched).

The tests monkeypatch the chunk constants down and wrap PPO.learn /
PPO.save with counting proxies over a tiny synthetic frame, so the
suite stays fast -- no real 300k training anywhere near here.
"""

from __future__ import annotations

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


@pytest.fixture()
def counted_ppo(monkeypatch):
    """Wrap sb3 PPO.learn/PPO.save with recording proxies (the real
    methods still run -- the models in these tests really train)."""
    from stable_baselines3 import PPO

    calls = {"learn": [], "save_exists": []}
    real_learn, real_save = PPO.learn, PPO.save

    def counting_learn(self, total_timesteps, callback=None,
                       reset_num_timesteps=True, **kw):
        calls["learn"].append({"total": total_timesteps,
                                "reset": reset_num_timesteps})
        return real_learn(self, total_timesteps=total_timesteps,
                          callback=callback,
                          reset_num_timesteps=reset_num_timesteps, **kw)

    def counting_save(self, path, **kw):
        out = real_save(self, path, **kw)
        p = Path(path)
        if p.suffix != ".zip":
            p = p.with_suffix(".zip")
        calls["save_exists"].append(p.exists())
        return out

    monkeypatch.setattr(PPO, "learn", counting_learn)
    monkeypatch.setattr(PPO, "save", counting_save)
    return calls


def _train(tmp_path, timesteps):
    sup = _tiny_sup()
    return ppo_train.train(
        timesteps=timesteps, warm_start=False,
        champion_path=tmp_path / "ppo_champion.zip",
        stacker_path=tmp_path / "stacker.pkl",
        db_path=tmp_path / "train.db",
        sup=sup, p_up=np.full(len(sup), 0.5))


def test_large_run_trains_in_checkpointed_segments(tmp_path, counted_ppo,
                                                   monkeypatch):
    monkeypatch.setattr(ppo_train, "CHUNK_THRESHOLD", 100)
    monkeypatch.setattr(ppo_train, "CHUNK_SIZE", 100)
    out = _train(tmp_path, timesteps=250)

    # (a) 250 steps at a 100-step chunk size = 3 segments; only the
    # first resets the timestep counter.
    assert counted_ppo["learn"] == [
        {"total": 100, "reset": True},
        {"total": 100, "reset": False},
        {"total": 50, "reset": False},
    ]
    # (b) a checkpoint file existed after EVERY segment save (plus
    # the final save at the end of train()).
    assert counted_ppo["save_exists"] == [True, True, True, True]
    # ... and the final artifact is a real, loadable challenger.
    from stable_baselines3 import PPO
    model = PPO.load(out)
    assert model is not None and hasattr(model, "predict")
    assert (tmp_path / "ppo_challenger_curve.json").exists()


def test_small_run_is_one_learn_call(tmp_path, counted_ppo, monkeypatch):
    """(c) Below the threshold (the nightly 60k shape): exactly one
    learn call, exactly one save -- the pre-fix behavior, unchanged."""
    monkeypatch.setattr(ppo_train, "CHUNK_THRESHOLD", 100)
    monkeypatch.setattr(ppo_train, "CHUNK_SIZE", 100)
    _train(tmp_path, timesteps=40)
    assert counted_ppo["learn"] == [{"total": 40, "reset": True}]
    assert counted_ppo["save_exists"] == [True]
