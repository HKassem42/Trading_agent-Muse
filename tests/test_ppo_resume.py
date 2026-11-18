"""Deep-train resume fix (Kass-approved 2026-11-04).

The Nov 1 weekly deep train died three times and every relaunch
restarted the 300k-step PPO train from step 0: train() warm-started
only from the champion and never looked at the challenger's own
segment checkpoints, even though the chunked trainer already saved
the challenger after every 25k segment. ~175k steps of attempt 2
were thrown away.

The fix: a long train (>= CHUNK_THRESHOLD steps) writes a progress
sidecar (ppo_challenger_progress.json) next to each segment
checkpoint -- steps done, target, feature-set tag, warm-start
champion hash, challenger checkpoint hash, start time. On entry, a
long train resumes from its own challenger checkpoint at the
recorded step count ONLY when every recorded field matches the
current invocation and the checkpoint loads cleanly; any mismatch
or load failure falls back to the previous behavior (warm-start
from the champion at step 0) with the reason printed. Short trains
below the chunk threshold never write or read the sidecar. (The
nightly 60k train joined the checkpointed class on 2026-11-17 --
see tests/test_nightly_resume.py.)

Tests monkeypatch the chunk constants down to toy sizes and wrap
PPO.learn / PPO.load with recording proxies, following
tests/test_ppo_train.py -- the models really train, just briefly.
A "kill" is simulated by making the second segment's learn() raise,
which is exactly where a SIGKILLed process leaves the world: one
checkpoint + sidecar on disk, train() never returns.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from models import ppo_train  # noqa: E402

TARGET = 120          # toy "long train" total, with threshold patched to 100
SEGMENT = 40          # toy CHUNK_SIZE
SIDECAR = "ppo_challenger_progress.json"
CHALLENGER = "ppo_challenger.zip"
CHAMPION = "ppo_champion.zip"


def _tiny_sup(n=600):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    sup = add_features(df).dropna(subset=FEATURE_COLUMNS)
    return sup.reset_index(drop=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def toy_chunks(monkeypatch):
    monkeypatch.setattr(ppo_train, "CHUNK_THRESHOLD", 100)
    monkeypatch.setattr(ppo_train, "CHUNK_SIZE", SEGMENT)


@pytest.fixture()
def spied_ppo(monkeypatch):
    """Record PPO.load paths and PPO.learn segment sizes while the
    real SB3 methods run. `kill_on_learn` (1-based) makes that learn
    call raise instead -- the simulated mid-train kill."""
    from stable_baselines3 import PPO

    rec = {"loads": [], "learns": [], "kill_on_learn": None}
    real_load, real_learn = PPO.load, PPO.learn

    def spying_load(path, *a, **kw):
        rec["loads"].append(Path(path).name)
        return real_load(path, *a, **kw)

    def spying_learn(self, total_timesteps, callback=None,
                     reset_num_timesteps=True, **kw):
        rec["learns"].append(total_timesteps)
        if rec["kill_on_learn"] == len(rec["learns"]):
            raise RuntimeError("simulated kill")
        return real_learn(self, total_timesteps=total_timesteps,
                          callback=callback,
                          reset_num_timesteps=reset_num_timesteps, **kw)

    monkeypatch.setattr(PPO, "load", spying_load)
    monkeypatch.setattr(PPO, "learn", spying_learn)
    return rec


def _make_champion(tmp_path, sup, p_up):
    """An untrained but valid champion zip in tmp_path -- warm-start
    loads it exactly like a real champion."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from environment.trading_env import TradingEnv
    from trading.broker import PaperBroker

    broker = PaperBroker(tmp_path / "champ_scratch.db",
                         starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = Monitor(TradingEnv(sup, p_up, broker))
    model = PPO("MlpPolicy", env, verbose=0, seed=42,
                n_steps=512, batch_size=64)
    path = tmp_path / CHAMPION
    model.save(path)
    return path


def _train(tmp_path, sup, p_up, timesteps=TARGET):
    return ppo_train.train(
        timesteps=timesteps, warm_start=True,
        champion_path=tmp_path / CHAMPION,
        stacker_path=tmp_path / "stacker.pkl",
        db_path=tmp_path / "train.db",
        sup=sup, p_up=p_up, telemetry=False)


def _killed_first_attempt(tmp_path, spied_ppo):
    """Run a long train that dies during its second segment; return
    (sup, p_up) for the relaunch. Asserts the kill landed where a
    real kill would leave things: one segment checkpointed."""
    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)
    spied_ppo["kill_on_learn"] = 2
    with pytest.raises(RuntimeError, match="simulated kill"):
        _train(tmp_path, sup, p_up)
    spied_ppo["kill_on_learn"] = None
    spied_ppo["loads"].clear()
    spied_ppo["learns"].clear()
    # The kill must leave a progress sidecar behind -- that is the
    # artifact every relaunch assertion below is about.
    assert (tmp_path / SIDECAR).exists(), "no progress sidecar written"
    return sup, p_up


def _sidecar(tmp_path) -> dict:
    return json.loads((tmp_path / SIDECAR).read_text())


# -- (a) kill after a segment -> relaunch resumes at the record -----

def test_kill_after_segment_relaunch_resumes_at_recorded_steps(
        tmp_path, toy_chunks, spied_ppo):
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)

    # The sidecar is the resume contract: everything a relaunch
    # needs to prove it is continuing THIS train.
    side = _sidecar(tmp_path)
    assert side["steps_done"] == SEGMENT
    assert side["target_steps"] == TARGET
    assert side["champion_sha256"] == _sha256(tmp_path / CHAMPION)
    assert side["challenger_sha256"] == _sha256(tmp_path / CHALLENGER)
    assert side["feature_set"]
    assert side["started"]

    out = _train(tmp_path, sup, p_up)

    # Resumed from the challenger's own checkpoint -- the champion
    # was never loaded -- and only the REMAINING steps were learned.
    assert spied_ppo["loads"] == [CHALLENGER]
    assert sum(spied_ppo["learns"]) == TARGET - SEGMENT
    assert out == tmp_path / CHALLENGER
    # A completed train cleans its sidecar up: nothing leaks into a
    # later, different train.
    assert not (tmp_path / SIDECAR).exists()


# -- (b) any sidecar mismatch -> warm-start from step 0 -------------

@pytest.mark.parametrize("tamper", ["target_steps", "feature_set",
                                    "champion_sha256"])
def test_sidecar_mismatch_falls_back_to_warm_start(
        tmp_path, toy_chunks, spied_ppo, tamper):
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    side_path = tmp_path / SIDECAR
    side = _sidecar(tmp_path)
    side[tamper] = "tampered" if tamper != "target_steps" else 999
    side_path.write_text(json.dumps(side))

    _train(tmp_path, sup, p_up)

    # Fell back to today's behavior: champion warm start, full
    # target learned from step 0, challenger never loaded.
    assert spied_ppo["loads"] == [CHAMPION]
    assert sum(spied_ppo["learns"]) == TARGET
    assert not side_path.exists()


def test_replaced_champion_falls_back_to_warm_start(
        tmp_path, toy_chunks, spied_ppo):
    """The champion file changing between attempts (a promotion
    landed, a seat was swapped) must break the resume match even
    though the sidecar itself is untouched."""
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    # A different champion: retrain-free, but different bytes.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from environment.trading_env import TradingEnv
    from trading.broker import PaperBroker

    broker = PaperBroker(tmp_path / "champ2_scratch.db",
                         starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = Monitor(TradingEnv(sup, p_up, broker))
    PPO("MlpPolicy", env, verbose=0, seed=123,
        n_steps=512, batch_size=64).save(tmp_path / CHAMPION)

    _train(tmp_path, sup, p_up)
    assert spied_ppo["loads"] == [CHAMPION]
    assert sum(spied_ppo["learns"]) == TARGET


# -- (c) corrupt challenger checkpoint -> clean fallback ------------

def test_corrupt_challenger_zip_falls_back_cleanly(
        tmp_path, toy_chunks, spied_ppo):
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    # Corrupt the checkpoint AND re-sign the sidecar for the corrupt
    # bytes, so every recorded field still "matches" -- only the
    # SB3 load can catch this, and it must degrade to a warm start,
    # never crash the relaunch.
    (tmp_path / CHALLENGER).write_bytes(b"this is not a zip file")
    side = _sidecar(tmp_path)
    side["challenger_sha256"] = _sha256(tmp_path / CHALLENGER)
    (tmp_path / SIDECAR).write_text(json.dumps(side))

    out = _train(tmp_path, sup, p_up)

    # The resume load was attempted on the challenger, failed, and
    # the train completed from the champion instead.
    assert spied_ppo["loads"] == [CHALLENGER, CHAMPION]
    assert sum(spied_ppo["learns"]) == TARGET
    from stable_baselines3 import PPO
    assert PPO.load(out) is not None  # a real, loadable challenger


def test_truncated_challenger_zip_falls_back_cleanly(
        tmp_path, toy_chunks, spied_ppo):
    """A half-written checkpoint (kill landed mid-save) fails the
    recorded challenger hash and must not be resumed from."""
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    raw = (tmp_path / CHALLENGER).read_bytes()
    (tmp_path / CHALLENGER).write_bytes(raw[: len(raw) // 2])

    _train(tmp_path, sup, p_up)
    assert spied_ppo["loads"] == [CHAMPION]
    assert sum(spied_ppo["learns"]) == TARGET


# -- (d) short trains: no sidecar, no resume path -------------------

def test_short_train_writes_no_sidecar(tmp_path, spied_ppo):
    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)
    _train(tmp_path, sup, p_up, timesteps=1024)  # << 100k threshold
    assert not (tmp_path / SIDECAR).exists()


def test_short_train_ignores_stale_sidecar(tmp_path, spied_ppo):
    """A leftover deep-train sidecar in the models dir must not
    leak into a short train's path: no resume load, and the
    sidecar is left byte-identical for the deep train that
    owns it."""
    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)
    stale = {"steps_done": 175000, "target_steps": 300000,
             "feature_set": "whatever", "champion_sha256": "x",
             "challenger_sha256": "y", "started": "2026-11-01T08:00:00+00:00"}
    (tmp_path / SIDECAR).write_text(json.dumps(stale))
    (tmp_path / CHALLENGER).write_bytes(b"leftover")

    _train(tmp_path, sup, p_up, timesteps=1024)

    assert spied_ppo["loads"] == [CHAMPION]  # warm start only
    assert json.loads((tmp_path / SIDECAR).read_text()) == stale
