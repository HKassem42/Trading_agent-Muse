"""Drawing-Board Option A — the 4-hour timeframe test (2026-10-24).

Executes refs/timeframe-4h-plan-2026-10-24.md EXACTLY as
pre-registered. The plan is the contract; this module is machinery
only. The one design change versus every prior study is the BAR
SIZE: the same 90 matrix-v2 indicator columns, the same LightGBM
learner and hyperparameters (models.gbm_focus), the same duel
discipline — computed and graded on 4h bars.

Data: data/history/btc_usd_1h_long.csv resampled to COMPLETE 4h
bars (exactly 4 hourly bars per bin; the source frame's 3 known gaps
drop 6 bins), open-time stamped. Features computed on the full long
4h frame (10,524 bars) by the v1 matrix builders
(scripts/build_timeseries_matrix.py) + src/features/main_chart,
then sliced to the v1/v2 matrix span [2024-11-23 00:00,
2026-10-21 16:00] = 4,183 bars. Labels are the v1 convention
(next surviving bar's close return, >= +0.003 BUY / <= -0.003
SHORT / else HOLD).

Arms (both LightGBM, LGBM_PARAMS verbatim, balanced class weights
-- the gbm_focus arm-B/D treatment):
  all4h    all 90 features at 4h
  focus4h  Kass's focus-11 at 4h (the timeframe twin of run 22)

Duel: calendar windows of 66 bars (11 days). Standard = the final
198 sliced bars; robustness = the 198 before. Episodes scale with
the bar: 10 episodes per window starting every 6 bars (one day),
6 steps each — the analog of 1h's 10x24-in-264. There is NO PPO
champion party at 4h (a 1h brain cannot be graded on 4h bars); the
yardsticks are buy-and-hold and the plan's absolute floor.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models import gbm_focus as gf
from models.composite_screen import (conviction_by_ts,
                                     p_ups_from_conviction)
from models.exit_study import (EPISODES, STARTING_BALANCE, ExitConfig,
                               StudyBroker, StudyEnv, _read_new_fills,
                               segment_trips)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

TARGET_COL = gf.TARGET_COL
CLASSES = gf.CLASSES
FOCUS_FEATURES = gf.FOCUS_FEATURES

SPAN_START = pd.Timestamp("2024-11-23 00:00:00")
SPAN_END = pd.Timestamp("2026-10-21 16:00:00")
STEPS_PER_DAY = 6          # 4h bars per day
WINDOW_BARS = 66           # 11 days
HOLDOUT_BARS = 198         # 3 windows

ARMS = ("all4h", "focus4h")
ARM_LABELS = {"all4h": "All Features", "focus4h": "Focus 11"}
ARM_DESIGNS = {
    "all4h": "LightGBM on all 90 matrix-v2 indicator columns at 4h, "
             "class-weighted (gbm_focus arm-B/D treatment, "
             "LGBM_PARAMS unchanged)",
    "focus4h": "LightGBM on Kass's focus-11 features at 4h, "
               "class-weighted (the direct timeframe twin of the 1h "
               "focus test, run 22)",
}

__all__ = ["TARGET_COL", "FOCUS_FEATURES", "SPAN_START", "SPAN_END",
           "STEPS_PER_DAY", "WINDOW_BARS", "ARMS", "ARM_LABELS",
           "ARM_DESIGNS", "resample_4h", "load_frame4h", "slice_span",
           "all_feature_columns", "arm_features", "walk_forward_arm",
           "evaluate_windows_4h", "conviction_by_ts",
           "p_ups_from_conviction", "apply_verdict"]


# ------------------------------------------------------------------ data
def resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an hourly OHLCV frame (lowercase columns, sorted
    UTC index) to COMPLETE 4h bars: a bin survives only if it holds
    exactly 4 hourly bars. Bars are open-time stamped."""
    h = hourly.copy()
    h["__n"] = 1
    agg = h.resample("4h", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "__n": "sum"})
    bars = agg.loc[agg["__n"] == 4].drop(columns="__n")
    return bars


def load_frame4h(path=None) -> pd.DataFrame:
    """The cached long 4h frame: OHLCV + the 90 feature columns +
    `__label`, built by the probe/build data-prep phase
    (data/tf4h/frame4h.pkl)."""
    p = Path(path) if path else DATA / "tf4h" / "frame4h.pkl"
    return pd.read_pickle(p)


def slice_span(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[(frame.index >= SPAN_START)
                      & (frame.index <= SPAN_END)]


def all_feature_columns(frame: pd.DataFrame) -> list[str]:
    """The 90 indicator columns: every frame column that is not
    OHLCV, the label (raw `__label` or the working frame's
    TARGET_COL), or a helper."""
    skip = {"open", "high", "low", "close", "volume", "__label",
            TARGET_COL}
    return [c for c in frame.columns if c not in skip]


def arm_features(frame: pd.DataFrame, arm: str) -> list[str]:
    if arm == "all4h":
        return all_feature_columns(frame)
    if arm == "focus4h":
        # FOCUS_FEATURES uses matrix-CSV names; in this frame the
        # raw volume column is lowercase "volume" (the matrix's
        # "Volume" OHLCV column — Kass's shared rank-1 feature).
        return ["volume" if f == "Volume" else f
                for f in FOCUS_FEATURES]
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------- walk-forward
def _fold_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pred_distribution": {c: float((pred == c).mean())
                              for c in CLASSES},
    }


def walk_forward_arm(df_pre: pd.DataFrame, arm: str) -> dict:
    """TimeSeriesSplit(5) walk-forward over the pre-duel 4h rows.
    Both arms fit with gbm_focus's weighted fold helper (arm 'B'
    treatment); the fold split and graded rows are identical for
    every arm."""
    from sklearn.model_selection import TimeSeriesSplit

    feats = arm_features(df_pre, arm)
    X = df_pre[feats].to_numpy(dtype=float)
    y = df_pre[TARGET_COL].to_numpy()
    ts = df_pre.index
    tss = TimeSeriesSplit(n_splits=5)
    folds, audits = [], []
    oof_pred = np.full(len(df_pre), None, dtype=object)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        model, audit = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, "B")
        pred = model.predict(X[va_idx])
        oof_pred[va_idx] = pred
        audits.append(audit)
        m = _fold_metrics(y[va_idx], pred)
        m["fold"] = k
        folds.append(m)
    graded = oof_pred != None  # noqa: E711 -- object array sentinel
    pooled = _fold_metrics(y[graded], oof_pred[graded])
    pooled["majority_baseline"] = float(
        max((y[graded] == c).mean() for c in CLASSES))
    pooled["n_graded"] = int(graded.sum())
    return {"arm": arm, "features": feats, "folds": folds,
            "pooled": pooled, "fold_audits": audits}


# ------------------------------------------------------------------ duel
def evaluate_windows_4h(windows, p_ups, model) -> dict:
    """The standard duel at 4h: a line-level mirror of
    composite_screen.evaluate_windows_composite (itself cross-checked
    against training.nightly.evaluate_windows at 1h) with ONLY the
    episode geometry changed — StudyEnv(steps_per_day=6), episode
    starts every 6 bars, 10 episodes per 66-bar window. Default
    ExitConfig (the verified production exit recipe), next-bar-open
    fills, 25% tranches, 75% cap, 0.15% fees, scratch DBs. Adds
    exact action counts, total fees and gross (pre-fee) P&L."""
    per = []
    all_closed: list[dict] = []
    total_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    fees_total = 0.0
    starts = [k * STEPS_PER_DAY for k in range(EPISODES)]
    for w_i, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        tmp = Path("/tmp") / f"tam_tf4h_{os.getpid()}_{w_i}.db"
        if tmp.exists():
            tmp.unlink()
        try:
            broker = StudyBroker(tmp, starting_balance=STARTING_BALANCE,
                                 max_position_pct=0.25, stop_loss_pct=0.05,
                                 cfg=ExitConfig())
            env = StudyEnv(w_sup, w_pup, broker, fill_mode="next_open",
                           steps_per_day=STEPS_PER_DAY)
            rewards, pnls, trips_env, ep_dds = [], [], [], []
            counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
            offset = 0
            pnl_long = pnl_short = 0.0
            open_left = 0
            w_fees = 0.0
            for ep, start in enumerate(starts):
                obs, _ = env.reset(seed=ep, options={"start": start})
                start_eq = env._day_start_equity
                curve = [start_eq]
                done, ep_r = False, 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    counts[["hold", "buy", "sell",
                            "short"][int(action)]] += 1
                    obs, r, done, _, _ = env.step(int(action))
                    ep_r += r
                    curve.append(env._prev_eq)
                ep_pnl = env._prev_eq - start_eq
                rewards.append(ep_r)
                pnls.append(ep_pnl)
                trips_env.append(env._round_trips)
                c = np.array(curve)
                ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
                fills, offset = _read_new_fills(tmp, offset)
                w_fees += float(sum(f["fee"] for f in fills))
                segs = segment_trips(fills, STARTING_BALANCE)
                closed_pnl = 0.0
                open_seg = None
                for s in segs:
                    if s["closed"]:
                        closed_pnl += s["pnl"]
                        all_closed.append(s)
                        if s["side"] > 0:
                            pnl_long += s["pnl"]
                        else:
                            pnl_short += s["pnl"]
                    else:
                        open_seg = s
                if open_seg is not None:
                    residual = ep_pnl - closed_pnl
                    if open_seg["side"] > 0:
                        pnl_long += residual
                    else:
                        pnl_short += residual
                    open_left += 1
            total = float(np.sum(pnls))
            max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
            n_actions = sum(counts.values()) or 1
            for k in total_counts:
                total_counts[k] += counts[k]
            fees_total += w_fees
            closes = w_sup["close"].to_numpy(dtype=float)
            bench = (float(STARTING_BALANCE * (closes[-1] / closes[0] - 1.0))
                     if len(closes) > 1 else 0.0)
            per.append({
                "mean_reward": float(np.mean(rewards)),
                "total_pnl": total,
                "gross_pnl": total + w_fees,
                "fees": w_fees,
                "max_drawdown": max_dd,
                "score": total - 0.5 * max_dd,
                "bench_pnl": bench,
                "round_trips_env": int(sum(trips_env)),
                "action_counts": dict(counts),
                "action_mix": {k: round(v / n_actions, 3)
                               for k, v in counts.items()},
                "pnl_long": pnl_long, "pnl_short": pnl_short,
                "open_left": open_left,
            })
        finally:
            tmp.unlink(missing_ok=True)
    wins = [s for s in all_closed if s["pnl"] > 0]
    losses = [s for s in all_closed if s["pnl"] <= 0]
    total_pnl = float(np.sum([w["total_pnl"] for w in per]))
    total_actions = sum(total_counts.values()) or 1
    return {
        "windows": per,
        "agg_score": float(np.mean([w["score"] for w in per])),
        "total_pnl": total_pnl,
        "gross_pnl": total_pnl + fees_total,
        "fees_total": fees_total,
        "bench_pnl": float(np.sum([w["bench_pnl"] for w in per])),
        "max_drawdown": float(max(w["max_drawdown"] for w in per)),
        "total_round_trips": int(sum(w["round_trips_env"] for w in per)),
        "closed_trips": len(all_closed),
        "wins": len(wins), "losses": len(losses),
        "avg_win": float(np.mean([s["pnl"] for s in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([s["pnl"] for s in losses]))
        if losses else 0.0,
        "pnl_long": float(sum(w["pnl_long"] for w in per)),
        "pnl_short": float(sum(w["pnl_short"] for w in per)),
        "open_left": int(sum(w["open_left"] for w in per)),
        "action_counts": total_counts,
        "action_mix": {k: round(v / total_actions, 3)
                       for k, v in total_counts.items()},
        "non_hold_share": float(1.0 - total_counts["hold"] / total_actions),
    }


# ---------------------------------------------------------------- verdict
def apply_verdict(wf: dict, duel: dict, robust: dict) -> dict:
    """The plan's three pre-registered verdict rules, applied
    mechanically. No champion party exists at 4h, so rule 2 is the
    absolute floor: P&L > $0 AND P&L > buy-and-hold AND non-hold
    activity >= 10%."""
    pooled = wf["pooled"]
    fold_bals = [f["balanced_accuracy"] for f in wf["folds"]]
    rule1 = bool(
        pooled["balanced_accuracy"] > 1.0 / 3.0
        and sum(1 for b in fold_bals if b > 1.0 / 3.0) >= 4
        and pooled["accuracy"] >= pooled["majority_baseline"])
    rule2 = bool(
        duel["total_pnl"] > 0.0
        and duel["total_pnl"] > duel["bench_pnl"]
        and duel["non_hold_share"] >= 0.10)
    rule3 = bool(robust["total_pnl"] >= 0.0)
    passed = rule1 and rule2 and rule3
    return {
        "verdict": "PASS" if passed else "FAIL",
        "rule1_walk_forward": rule1,
        "rule2_duel_floor": rule2,
        "rule3_robustness": rule3,
        "fold_balanced_accuracies": fold_bals,
    }
