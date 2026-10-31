"""PPO agent training.

Trains (or warm-starts) the PPO policy on the TradingEnv. The nightly job
calls this with warm_start=True so the agent keeps learning instead of
starting over — continual learning, per the design.

Usage:
    python -m src.models.ppo_train --timesteps 20000
    python -m src.models.ppo_train --timesteps 20000 --warm-start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.market_data import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, make_supervised
from models.stacker import load as load_stacker
from trading.broker import PaperBroker

# Chunked training for large runs (pre-flight Finding 2 fix,
# 2026-10-24): at or above CHUNK_THRESHOLD total steps, train() learns
# in CHUNK_SIZE segments and saves the challenger after each one.
# The weekly deep train's single uninterrupted 300k learn was the
# shape refs/diagnosis-2026-10-21.md records being SIGKILLed three
# times on this box; the diagnosis battery's 300k arm only completed
# as 12 checkpointed 25k chunks. Below the threshold (the nightly
# 60k run) training is the original single learn call, unchanged.
CHUNK_THRESHOLD = 100_000
CHUNK_SIZE = 25_000


def train(timesteps: int, warm_start: bool, champion_path: Path, stacker_path: Path,
          db_path: Path, ent_coef: float = 0.01, sup=None, p_up=None,
          feature_columns=None, telemetry: bool = True) -> Path:
    """Train (or warm-start) the PPO challenger.

    `sup` / `p_up`: optional pre-built training frame + stacker votes. The
    nightly job passes them so it controls exactly which candles the
    challenger may learn from (audit 2026-10-20: the old path fetched the
    full series here, which silently included the duel's holdout candles
    -- the challenger was graded on data it had just trained on) and so
    the votes can be cross-fitted instead of in-sample. When omitted
    (manual CLI runs), the full series is fetched as before.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (BaseCallback,
                                                    CallbackList)
    from stable_baselines3.common.monitor import Monitor

    class _TelemetryCallback(BaseCallback):
        """Logging-only SB3 training telemetry (2026-10-26 research
        incorporation, Training Lab telemetry panel): policy entropy,
        approx KL, clip fraction, explained variance, value loss --
        read from model.logger.name_to_value at each rollout end.

        The callback NEVER mutates the model, the env, or any RNG
        stream: it only reads values SB3 already computed for its
        own logs. (One-rollout lag: SB3 dumps an iteration's logger
        values after its callbacks fire, so a rollout-end read sees
        the previous iteration's dump; _on_training_end appends the
        final dump so start/end/min are exact.) tests/
        test_ppo_telemetry.py proves training outcomes are bit-
        identical with and without this callback."""

        _KEYS = {"kl": "train/approx_kl", "clip": "train/clip_fraction",
                 "ev": "train/explained_variance",
                 "vl": "train/value_loss"}

        def __init__(self) -> None:
            super().__init__()
            self.rows: list[dict] = []
            self._ent_points: list[tuple[int, float]] = []

        def _read(self) -> None:
            logger = getattr(self.model, "logger", None)
            vals = getattr(logger, "name_to_value", None) or {}
            row: dict = {}
            el = vals.get("train/entropy_loss")
            if isinstance(el, (int, float)):
                # SB3 logs entropy_loss = -entropy; the panel shows
                # entropy itself (nats, >= 0 while exploring).
                row["ent"] = round(-float(el), 4)
            for key, sb3_key in self._KEYS.items():
                v = vals.get(sb3_key)
                if isinstance(v, (int, float)):
                    row[key] = round(float(v), 4)
            if row:
                self.rows.append(row)
                if "ent" in row:
                    self._ent_points.append(
                        (int(self.num_timesteps), row["ent"]))

        def _on_rollout_end(self) -> None:
            self._read()

        def _on_training_end(self) -> None:
            self._read()

        def _on_step(self) -> bool:
            return True

        def summary(self) -> dict | None:
            if not self.rows:
                return None

            def _last(key):
                for r in reversed(self.rows):
                    if key in r:
                        return r[key]
                return None

            ents = [r["ent"] for r in self.rows if "ent" in r]
            out: dict = {"rollouts": len(self.rows),
                         "kl_end": _last("kl"), "clip_end": _last("clip"),
                         "ev_end": _last("ev"), "vl_end": _last("vl")}
            if ents:
                out.update({"ent_start": ents[0], "ent_end": ents[-1],
                            "ent_min": min(ents)})
                pts = self._ent_points
                if len(pts) > 24:
                    idx = np.linspace(0, len(pts) - 1, 24).astype(int)
                    pts = [pts[k] for k in idx]
                out["ent_curve"] = [[t, e] for t, e in pts]
            return out

    class _CurveCallback(BaseCallback):
        """Collects the learning curve (mean episode return vs timesteps)
        for the Training Lab chart (v4). Buffered in blocks of 10 episodes
        to keep the series compact."""

        def __init__(self) -> None:
            super().__init__()
            self.points: list[tuple[int, float]] = []
            self._buf: list[float] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self._buf.append(float(ep["r"]))
                    if len(self._buf) >= 10:
                        self.points.append(
                            (int(self.num_timesteps),
                             round(float(np.mean(self._buf)), 3)))
                        self._buf = []
            return True

    if sup is None:
        df = fetch_klines("BTC-USD", interval="1h", period="730d")
        sup = make_supervised(df)
    if p_up is None:
        stacker = load_stacker(stacker_path)
        p_up = stacker.predict_proba(
            sup[feature_columns or FEATURE_COLUMNS])[:, 1]

    # Unique scratch ledger per invocation: two trainings sharing one SQLite
    # file collide with "database is locked" (bit us 2026-10-20 when a manual
    # run and the nightly run overlapped). PID-suffix + cleanup prevents it.
    import os
    work_db = db_path.with_name(f"{db_path.stem}_{os.getpid()}{db_path.suffix}")
    broker = PaperBroker(work_db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = Monitor(TradingEnv(sup, p_up, broker,
                             feature_columns=feature_columns))

    if warm_start and champion_path.exists():
        try:
            model = PPO.load(champion_path, env=env)
            # PPO.load restores the champion's saved hyperparameters, so
            # the caller's exploration setting used to be silently inert
            # on warm starts (audit 2026-10-20). Apply it explicitly and
            # log the effective values into the run record.
            model.ent_coef = ent_coef
            print(f"warm-starting from {champion_path} "
                  f"(ent_coef set to {ent_coef}, lr={model.learning_rate})")
        except Exception as e:
            # e.g. observation-dim mismatch after a feature-recipe change:
            # fall back to a fresh brain rather than crashing the nightly run
            print(f"warm-start failed ({type(e).__name__}), training fresh: {e}")
            model = None
    else:
        model = None
    if model is None:
        model = PPO("MlpPolicy", env, verbose=0, seed=42,
                    learning_rate=3e-4, n_steps=512, batch_size=64,
                    ent_coef=ent_coef)
    curve_cb = _CurveCallback()
    telem_cb = _TelemetryCallback() if telemetry else None
    # telemetry=False (the equality test's control arm) passes the
    # bare curve callback -- byte-for-byte the pre-2026-10-26 call.
    learn_cb = (CallbackList([curve_cb, telem_cb])
                if telem_cb is not None else curve_cb)
    out = champion_path.parent / "ppo_challenger.zip"
    if timesteps >= CHUNK_THRESHOLD:
        # Segmented + checkpointed (see CHUNK_* note above): each
        # segment continues the same model (reset_num_timesteps=False
        # after the first) and the challenger is saved after EVERY
        # segment, so a kill loses at most one segment of training.
        done_steps, first = 0, True
        while done_steps < timesteps:
            seg = min(CHUNK_SIZE, timesteps - done_steps)
            model.learn(total_timesteps=seg, callback=learn_cb,
                        reset_num_timesteps=first)
            first = False
            done_steps += seg
            model.save(out)
            print(f"checkpoint: {done_steps}/{timesteps} steps -> {out}")
    else:
        model.learn(total_timesteps=timesteps, callback=learn_cb)

    model.save(out)
    print(f"saved challenger -> {out}")
    # Learning curve sidecar (v4): downsampled to <=24 points, read by the
    # nightly job into the run record for the Training Lab chart. The
    # "telemetry" sibling key (2026-10-26) rides the same sidecar: the
    # nightly job lifts it into the run record for the telemetry
    # panel. Logging only -- see _TelemetryCallback.
    pts = curve_cb.points
    if len(pts) > 24:
        idx = np.linspace(0, len(pts) - 1, 24).astype(int)
        pts = [pts[k] for k in idx]
    import json as _json
    (champion_path.parent / "ppo_challenger_curve.json").write_text(
        _json.dumps({"timesteps": timesteps,
                     "curve": [[t, r] for t, r in pts],
                     "telemetry": telem_cb.summary()
                     if telem_cb is not None else None}))
    try:
        work_db.unlink(missing_ok=True)
    except OSError:
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--warm-start", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    # NEVER train against the live paper.db ledger -- scratch file only.
    train(args.timesteps, args.warm_start,
          root / "models" / "ppo_champion.zip",
          root / "models" / "stacker_champion.pkl",
          root / "data" / "train_tmp.db")
