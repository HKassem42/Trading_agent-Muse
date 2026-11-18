"""Nightly-retrain restart survivability (Kass order, 2026-11-17).

The deep-train resume fix (2026-11-04, tests/test_ppo_resume.py +
tests/test_deep_stacker_skip.py) made the WEEKLY deep train
survive runtime restarts, but the NIGHTLY path was left exposed on
both layers:

1. The nightly PPO challenger trains 60,000 steps -- below the
   100k CHUNK_THRESHOLD -- so it trained as ONE uninterrupted
   learn() with no checkpoint and no progress sidecar. The
   midnight retrain's PPO phase was killed by runtime restarts on
   2026-11-14, 2026-11-16 and 2026-11-17 (three of four nights)
   and every step was lost.
2. The stacker re-grade skip keyed only on DEEP-run identity, so
   a relaunched nightly run re-graded and re-registered its
   stacker challenger -- the run 129 duplicate (an interrupted
   re-run's second grading, superseded by run 130).

The port mirrors the deep design exactly:
- CHUNK_THRESHOLD drops below the nightly length, so the 60k
  train is chunked + checkpointed + sidecar-tracked like the
  deep train (resume still exact-field-match + clean load, or
  warm-start from step 0 -- nothing weakened).
- Nightly stacker rows carry their own identity tags (nightly /
  nightly_run_id / champion_stacker_sha256) and a relaunched
  nightly run finds its own row and skips straight to the PPO
  phase. Deep rows and nightly rows never cross-match.
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
from training import nightly  # noqa: E402

NIGHTLY_STEPS = 60_000   # nightly.py's non-deep PPO train length
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


def _train(tmp_path, sup, p_up, timesteps):
    return ppo_train.train(
        timesteps=timesteps, warm_start=True,
        champion_path=tmp_path / CHAMPION,
        stacker_path=tmp_path / "stacker.pkl",
        db_path=tmp_path / "train.db",
        sup=sup, p_up=p_up, telemetry=False)


# -- (a)+(d) the nightly-length train checkpoints, resumes, and
#           cleans its sidecar up -- at the REAL production
#           constants, no patched thresholds --------------------

def test_nightly_length_train_checkpoints_resumes_and_cleans_up(
        tmp_path, spied_ppo):
    # The regression pin: the nightly 60k train must sit INSIDE
    # the checkpointed class. (On the pre-port code this fails:
    # CHUNK_THRESHOLD was 100_000 and 60k trained as one learn.)
    assert ppo_train.CHUNK_THRESHOLD <= NIGHTLY_STEPS

    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)

    # Attempt 1 dies during its second segment -- where the
    # 2026-11-14/16/17 kills left the world: one segment
    # checkpointed, train() never returns.
    spied_ppo["kill_on_learn"] = 2
    with pytest.raises(RuntimeError, match="simulated kill"):
        _train(tmp_path, sup, p_up, NIGHTLY_STEPS)
    spied_ppo["kill_on_learn"] = None
    spied_ppo["loads"].clear()
    spied_ppo["learns"].clear()

    side = json.loads((tmp_path / SIDECAR).read_text())
    assert side["steps_done"] == ppo_train.CHUNK_SIZE
    assert side["target_steps"] == NIGHTLY_STEPS
    assert side["champion_sha256"] == _sha256(tmp_path / CHAMPION)
    assert side["challenger_sha256"] == _sha256(tmp_path / CHALLENGER)

    # Relaunch: resumes from the challenger's own checkpoint (the
    # champion is never loaded) and learns only the remainder.
    out = _train(tmp_path, sup, p_up, NIGHTLY_STEPS)
    assert spied_ppo["loads"] == [CHALLENGER]
    assert sum(spied_ppo["learns"]) == NIGHTLY_STEPS - ppo_train.CHUNK_SIZE
    assert out == tmp_path / CHALLENGER
    # (d) the completed train deleted its sidecar.
    assert not (tmp_path / SIDECAR).exists()


# -- (b) resume still rejects any inexact sidecar -- toy segment
#       sizes (same shape as the nightly class: a multi-segment
#       train below the OLD 100k threshold), real validation ----

TOY_TARGET = 120
TOY_SEGMENT = 40


@pytest.fixture()
def toy_chunks(monkeypatch):
    monkeypatch.setattr(ppo_train, "CHUNK_THRESHOLD", 100)
    monkeypatch.setattr(ppo_train, "CHUNK_SIZE", TOY_SEGMENT)


def _killed_first_attempt(tmp_path, spied_ppo):
    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)
    spied_ppo["kill_on_learn"] = 2
    with pytest.raises(RuntimeError, match="simulated kill"):
        _train(tmp_path, sup, p_up, TOY_TARGET)
    spied_ppo["kill_on_learn"] = None
    spied_ppo["loads"].clear()
    spied_ppo["learns"].clear()
    assert (tmp_path / SIDECAR).exists()
    return sup, p_up


@pytest.mark.parametrize("tamper", ["target_steps", "feature_set",
                                    "champion_sha256", "steps_done"])
def test_nightly_class_sidecar_mismatch_falls_back(
        tmp_path, toy_chunks, spied_ppo, tamper):
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    side_path = tmp_path / SIDECAR
    side = json.loads(side_path.read_text())
    if tamper == "target_steps":
        side["target_steps"] = 999
    elif tamper == "steps_done":
        side["steps_done"] = TOY_TARGET  # a finished train is stale
    else:
        side[tamper] = "tampered"
    side_path.write_text(json.dumps(side))

    _train(tmp_path, sup, p_up, TOY_TARGET)

    # Warm start from the champion, full target from step 0.
    assert spied_ppo["loads"] == [CHAMPION]
    assert sum(spied_ppo["learns"]) == TOY_TARGET
    assert not side_path.exists()


def test_nightly_class_changed_challenger_checkpoint_falls_back(
        tmp_path, toy_chunks, spied_ppo):
    """The checkpoint bytes changing after the sidecar was written
    (a partial save, another train overwrote it) breaks the match
    even though the sidecar itself is untouched."""
    sup, p_up = _killed_first_attempt(tmp_path, spied_ppo)
    raw = (tmp_path / CHALLENGER).read_bytes()
    (tmp_path / CHALLENGER).write_bytes(raw[: len(raw) // 2])

    _train(tmp_path, sup, p_up, TOY_TARGET)
    assert spied_ppo["loads"] == [CHAMPION]
    assert sum(spied_ppo["learns"]) == TOY_TARGET


def test_nightly_class_completed_train_leaves_no_sidecar(
        tmp_path, toy_chunks, spied_ppo):
    """(d) at toy scale: an uninterrupted checkpointed-class train
    writes sidecars per segment but none survives completion."""
    sup = _tiny_sup()
    p_up = np.full(len(sup), 0.5)
    _make_champion(tmp_path, sup, p_up)
    _train(tmp_path, sup, p_up, TOY_TARGET)
    assert not (tmp_path / SIDECAR).exists()
    assert len(spied_ppo["learns"]) == TOY_TARGET // TOY_SEGMENT


# -- (c) nightly stacker identity: a relaunched nightly run finds
#       its own registered stacker row and skips re-grading -----

RUN_ID = "20261117"
CHAMP_SHA = "c" * 64


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "DATA", tmp_path)
    return tmp_path


def _nightly_metrics(**over):
    m = {"challenger_acc": 0.5417, "champion_acc": 0.5467,
         "challenger_wf_acc": 0.5202, "precision": 0.5099,
         "folds": 5, "pending_approval": False,
         "reason": "HA stacker challenger holdout acc 0.5417 did "
                   "not beat champion 0.5467",
         "nightly": True, "nightly_run_id": RUN_ID,
         "champion_stacker_sha256": CHAMP_SHA}
    m.update(over)
    return m


def test_same_night_stacker_is_found(registry):
    nightly.log_run("stacker", _nightly_metrics(), False)
    found = nightly.find_registered_nightly_stacker(RUN_ID, CHAMP_SHA)
    assert found is not None
    assert found["challenger_acc"] == 0.5417
    assert found["nightly_run_id"] == RUN_ID


def test_nightly_finder_rejects_other_night_or_champion(registry):
    nightly.log_run("stacker", _nightly_metrics(), False)
    assert nightly.find_registered_nightly_stacker(
        "20261118", CHAMP_SHA) is None
    assert nightly.find_registered_nightly_stacker(
        RUN_ID, "d" * 64) is None
    assert nightly.find_registered_nightly_stacker(
        RUN_ID, None) is None


def test_nightly_and_deep_rows_never_cross_match(registry):
    """The Sunday sequence is the trap: the nightly run and the
    deep train share a UTC date and a registry. Each finder must
    return ONLY its own run's row (the rows carry different
    accuracies so a cross-match is visible), never the other's."""
    deep_metrics = _nightly_metrics(challenger_acc=0.5152)
    for key in ("nightly", "nightly_run_id"):
        deep_metrics.pop(key)
    deep_metrics.update({"deep": True, "deep_run_id": RUN_ID})
    nightly.log_run("stacker", deep_metrics, False)
    # Only the deep row exists: the nightly finder sees nothing.
    assert nightly.find_registered_nightly_stacker(
        RUN_ID, CHAMP_SHA) is None

    nightly.log_run("stacker", _nightly_metrics(), False)
    # Both rows exist now: each finder returns its OWN row.
    deep_found = nightly.find_registered_deep_stacker(
        RUN_ID, CHAMP_SHA)
    assert deep_found is not None
    assert deep_found["challenger_acc"] == 0.5152
    nightly_found = nightly.find_registered_nightly_stacker(
        RUN_ID, CHAMP_SHA)
    assert nightly_found is not None
    assert nightly_found["challenger_acc"] == 0.5417


def test_nightly_skip_decision_requires_non_deep(registry):
    nightly.log_run("stacker", _nightly_metrics(), False)
    assert nightly.nightly_stacker_skip(False, RUN_ID,
                                        CHAMP_SHA) is not None
    assert nightly.nightly_stacker_skip(True, RUN_ID,
                                        CHAMP_SHA) is None
    assert nightly.nightly_stacker_skip(False, "20261118",
                                        CHAMP_SHA) is None


# -- (c) wiring: main() run twice in one night registers ONE
#       stacker row; the second run skips grading and still
#       reaches the PPO phase -----------------------------------

def _big_sup_frame(n=2200):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)


def _fake_eval(score, pnl, bench):
    win = {"score": score, "total_pnl": pnl / 3.0,
           "max_drawdown": 10.0, "mean_drawdown": 5.0,
           "bench_pnl": bench / 3.0, "episodes": 10,
           "mean_reward": 0.1,
           "round_trips_per_day": 1.0,
           "action_mix": {"hold": 0.5, "buy": 0.2, "sell": 0.1,
                          "short": 0.2}}
    return {"windows": [dict(win) for _ in range(3)],
            "agg_score": score, "total_pnl": pnl,
            "mean_reward": 0.1}


def _stacker_rows(db_path):
    import sqlite3
    with sqlite3.connect(db_path) as con:
        return con.execute(
            "SELECT metrics FROM runs WHERE kind='stacker'"
        ).fetchall()


def test_nightly_rerun_skips_stacker_and_reaches_ppo(
        tmp_path, monkeypatch):
    import joblib
    from sklearn.dummy import DummyClassifier

    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    models_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(nightly, "MODELS", models_dir)
    monkeypatch.setattr(nightly, "DATA", data_dir)
    monkeypatch.setenv("TAM_PROMOTION_LOCK", "1")
    monkeypatch.delenv("TAM_DEEP", raising=False)

    # Lab champions on disk: a real (if trivial) stacker model the
    # grading + PPO sections can load, and a PPO champion file so
    # main() takes the graded-duel branch.
    champ_stacker = DummyClassifier(strategy="prior")
    champ_stacker.fit(np.zeros((8, len(FEATURE_COLUMNS))),
                      np.array([0, 1] * 4))
    joblib.dump(champ_stacker, models_dir / "stacker_champion.pkl")
    (models_dir / "ppo_champion.zip").write_bytes(b"placeholder")

    monkeypatch.setattr(nightly, "fetch_klines",
                        lambda *a, **kw: _big_sup_frame())

    calls = {"ha_train": 0, "ppo_train": 0}

    def fake_train_ha(Htr, ytr, out_path):
        calls["ha_train"] += 1
        m = DummyClassifier(strategy="prior")
        m.fit(Htr, ytr)
        joblib.dump(m, out_path)
        return {"accuracy": 0.5, "precision": 0.5, "folds": 5,
                "model_path": str(out_path)}

    monkeypatch.setattr(nightly, "train_ha_challenger", fake_train_ha)
    monkeypatch.setattr(
        nightly, "crossfit_p_up",
        lambda X, y, folds=4, pipeline_factory=None:
            np.full(len(X), 0.5))

    import models.ppo_train as ppo_train_mod

    def fake_train_ppo(**kwargs):
        calls["ppo_train"] += 1
        out = models_dir / "ppo_challenger.zip"
        out.write_bytes(b"challenger")
        return out

    monkeypatch.setattr(ppo_train_mod, "train", fake_train_ppo)

    def fake_evaluate_windows(model_path, windows, p_ups,
                              feature_columns=None):
        if "challenger" in str(model_path):
            return _fake_eval(50.0, 300.0, -900.0)
        return _fake_eval(10.0, -100.0, -900.0)

    monkeypatch.setattr(nightly, "evaluate_windows",
                        fake_evaluate_windows)

    nightly.main()
    assert calls == {"ha_train": 1, "ppo_train": 1}
    assert len(_stacker_rows(data_dir / "training.db")) == 1

    # The relaunch (same UTC date, same champion stacker bytes):
    # NO second grading, NO duplicate registry row -- and the run
    # proceeds to the PPO phase instead of dying at the stacker.
    nightly.main()
    assert calls["ha_train"] == 1
    assert calls["ppo_train"] == 2
    assert len(_stacker_rows(data_dir / "training.db")) == 1
