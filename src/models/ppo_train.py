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


def train(timesteps: int, warm_start: bool, champion_path: Path, stacker_path: Path,
          db_path: Path, ent_coef: float = 0.01, sup=None, p_up=None) -> Path:
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
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

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
        p_up = stacker.predict_proba(sup[FEATURE_COLUMNS])[:, 1]

    # Unique scratch ledger per invocation: two trainings sharing one SQLite
    # file collide with "database is locked" (bit us 2026-10-20 when a manual
    # run and the nightly run overlapped). PID-suffix + cleanup prevents it.
    import os
    work_db = db_path.with_name(f"{db_path.stem}_{os.getpid()}{db_path.suffix}")
    broker = PaperBroker(work_db, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = Monitor(TradingEnv(sup, p_up, broker))

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
    model.learn(total_timesteps=timesteps, callback=curve_cb)

    out = champion_path.parent / "ppo_challenger.zip"
    model.save(out)
    print(f"saved challenger -> {out}")
    # Learning curve sidecar (v4): downsampled to <=24 points, read by the
    # nightly job into the run record for the Training Lab chart.
    pts = curve_cb.points
    if len(pts) > 24:
        idx = np.linspace(0, len(pts) - 1, 24).astype(int)
        pts = [pts[k] for k in idx]
    import json as _json
    (champion_path.parent / "ppo_challenger_curve.json").write_text(
        _json.dumps({"timesteps": timesteps,
                     "curve": [[t, r] for t, r in pts]}))
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
