"""Meta-Label Gate — Pick A (2026-10-30, Kass's delegated pick).

Design frozen in refs/meta-label-study-2026-10-30.md BEFORE any
result was seen. The champion pair (PPO run 44 deciding on
stacker run 73's votes) proposes entries; a LightGBM
meta-model, trained on the pair's own TRAINING-region candidate
trades, predicts P(a candidate entry makes money after fees)
and vetoes entries below the threshold. The gate is a HARD
veto, never an observation (eyes-duel lesson, 2026-10-29).

Everything money-shaped stays production: fills, stops,
trailing exits, rewards and ledgers are PaperBroker's and
TradingEnv's own code. This module only decides, per entry
attempt, whether the production step sees the brain's action
or a hold (a vetoed entry is substituted BEFORE the step, so
the broker's stop check still runs on that candle — exits are
never gated). Candidates are extracted by replaying the pair
duel-style (24-candle episodes, next-candle-open fills); each
candidate is labeled by a STANDALONE broker simulation of that
single trade under production exit rules.

Lab only: nothing in the live loops or the nightly retrain
imports this module, and it never touches a seat or a book.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from environment.trading_env import TradingEnv
from trading.broker import PaperBroker

FAMILY = "meta_label"
GATE_THRESHOLD = 0.55          # the graded threshold (frozen)
SENSITIVITY_THRESHOLDS = (0.50, 0.55, 0.60)
META_CONTEXT_COLUMNS = ["p_up", "abs_p_up_mhalf", "atr_rel",
                        "volatility_21", "atr_pct_raw", "side"]
LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05,
               "num_leaves": 31, "min_child_samples": 20,
               "subsample": 0.9, "colsample_bytree": 0.9,
               "random_state": 42}
ACTION_KEYS = ["hold", "buy", "sell", "short"]
STARTING_BALANCE = 100_000.0
_EPS = 1e-10


# ------------------------------------------------------- classification
def is_entry_attempt(action: int, btc: float) -> int:
    """+1 if `action` opens/adds a LONG from position `btc`,
    -1 if it opens/adds a SHORT, 0 otherwise. Position-aware,
    mirroring the broker's semantics (broker.py docstring):
    a buy while short COVERS (management), a short while long
    CLOSES through flat (management); sells and holds are never
    entries."""
    if action == 1 and btc >= -_EPS:
        return 1
    if action == 3 and btc <= _EPS:
        return -1
    return 0


# ------------------------------------------------------------ features
def feature_row(frame_values, votes, atr_rel, vol21, atr_raw,
                idx: int, side: int) -> np.ndarray:
    """The frozen 104-vector at a decision candle: the 98
    pick2_98 scored values at that candle, then the context
    block [p_up, |p_up - 0.5|, atr_rel, volatility_21,
    atr_pct_raw, side]. Row-local by construction — it reads
    index `idx` of every input and nothing else (pinned by the
    garble test in tests/test_meta_label_gate.py)."""
    frame = np.asarray(frame_values, dtype=float)
    row = np.empty(104, dtype=float)
    row[:98] = frame[idx]
    p = float(np.asarray(votes, dtype=float)[idx])
    row[98] = p
    row[99] = abs(p - 0.5)
    row[100] = float(np.asarray(atr_rel, dtype=float)[idx])
    row[101] = float(np.asarray(vol21, dtype=float)[idx])
    row[102] = float(np.asarray(atr_raw, dtype=float)[idx])
    row[103] = float(side)
    return row


class MetaGate:
    """The fitted meta-model + the aligned feature arrays of one
    region (a duel window, in the runner). `allow(idx, side)` is
    the gate function GatedTradingEnv consumes: True iff the
    model's P(profitable) at that decision candle and side meets
    the threshold."""

    def __init__(self, model, frame_values, votes, atr_rel, vol21,
                 atr_raw, threshold: float = GATE_THRESHOLD):
        self.model = model
        self.frame = np.asarray(frame_values, dtype=float)
        self.votes = np.asarray(votes, dtype=float)
        self.atr_rel = np.asarray(atr_rel, dtype=float)
        self.vol21 = np.asarray(vol21, dtype=float)
        self.atr_raw = np.asarray(atr_raw, dtype=float)
        self.threshold = float(threshold)
        classes = list(model.classes_)
        self._pos = classes.index(1) if 1 in classes else None

    def proba(self, idx: int, side: int) -> float:
        if self._pos is None:
            return 0.0
        x = feature_row(self.frame, self.votes, self.atr_rel,
                        self.vol21, self.atr_raw, idx, side)
        return float(self.model.predict_proba(x.reshape(1, -1))[0,
                                                               self._pos])

    def allow(self, idx: int, side: int) -> bool:
        return self.proba(idx, side) >= self.threshold


# ------------------------------------------------------------ the env
class GatedTradingEnv(TradingEnv):
    """TradingEnv with a hard entry veto. The ONLY behavioral
    change: when the brain's action is an entry attempt and
    `allow_fn(decision_idx, side)` returns False, a hold is
    substituted before the production step runs — fills, the
    stop check, rewards and round-trip counting are the
    production code's own, untouched. With allow_fn=None the
    env is byte-for-byte the production env (pinned by the
    equivalence tests). `recorder`, when given a list, receives
    (decision_idx, side) for EVERY entry attempt (the candidate
    extraction path). Counters are per-env-instance (one
    episode, in every caller): entries_attempted /
    entries_vetoed / entries_filled (allowed attempts whose
    fill actually increased the position)."""

    def __init__(self, df, stacker_p_up, broker, *, allow_fn=None,
                 recorder=None, **kwargs):
        super().__init__(df, stacker_p_up, broker, **kwargs)
        self.allow_fn = allow_fn
        self.recorder = recorder
        self.executed_action = 0
        self.entries_attempted = 0
        self.entries_vetoed = 0
        self.entries_filled = 0

    def step(self, action: int):
        action = int(action)
        side = is_entry_attempt(action, self.broker.btc)
        executed = action
        before = None
        if side != 0:
            self.entries_attempted += 1
            if self.recorder is not None:
                self.recorder.append((self._cursor, side))
            if self.allow_fn is not None and not self.allow_fn(
                    self._cursor, side):
                executed = 0
                self.entries_vetoed += 1
            else:
                before = abs(self.broker.btc)
        out = super().step(executed)
        self.executed_action = executed
        if before is not None and abs(self.broker.btc) > before + 1e-12:
            self.entries_filled += 1
        return out


# -------------------------------------------------------------- labels
def simulate_candidate_label(arrays: dict, decision_pos: int,
                             side: int, db_path) -> dict | None:
    """The meta-label for one candidate: a STANDALONE trade on a
    fresh PaperBroker (duel config), filled at the candle after
    the decision at its OPEN with the decision candle's ATR —
    the duel's fill convention — then held under production
    exit rules (check_stop on every candle from the fill candle
    onward, wick triggers). Exit is the broker's stop; a trade
    still open at the last array candle is closed at its close
    ("horizon"). Returns None when the standalone entry fill is
    refused. Label = 1 iff net P&L (cash delta, both fees in)
    is strictly positive. Post-entry candles decide the OUTCOME
    only; nothing about them reaches the features."""
    n = len(arrays["close"])
    fill_pos = decision_pos + 1
    if fill_pos >= n:
        return None
    broker = PaperBroker(db_path, starting_balance=STARTING_BALANCE,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    atr = float(arrays["atr_pct_raw"][decision_pos])
    atr = atr if np.isfinite(atr) and atr > 0 else None
    open_px = float(arrays["open"][fill_pos])
    if side > 0:
        fill = broker.buy(open_px, reason="meta_label",
                          ts=str(fill_pos), atr_pct=atr)
    else:
        fill = broker.short_open(open_px, reason="meta_label",
                                 ts=str(fill_pos), atr_pct=atr)
    if fill is None:
        return None
    fees = float(fill.fee)
    exit_reason = "horizon"
    exit_pos = n - 1
    exit_price = float(arrays["close"][n - 1])
    stopped = False
    for k in range(fill_pos, n):
        sf = broker.check_stop(float(arrays["close"][k]), ts=str(k),
                               low=float(arrays["low"][k]),
                               high=float(arrays["high"][k]))
        if sf is not None:
            fees += float(sf.fee)
            exit_reason, exit_pos = sf.reason, k
            exit_price = float(sf.price)
            stopped = True
            break
    if not stopped and broker.btc != 0.0:
        if broker.btc > 0:
            sf = broker.sell(exit_price, pct=1.0, reason="horizon",
                             ts=str(n - 1))
        else:
            sf = broker.cover(exit_price, pct=1.0, reason="horizon",
                              ts=str(n - 1))
        if sf is not None:
            fees += float(sf.fee)
    pnl = float(broker.cash - STARTING_BALANCE)
    return {"label": int(pnl > 0.0), "pnl": pnl, "fees": fees,
            "exit_reason": exit_reason, "exit_pos": int(exit_pos),
            "entry_price": float(fill.price), "exit_price": exit_price,
            "fill_pos": int(fill_pos)}


# ---------------------------------------------------------- extraction
def extract_candidates(sup_block, votes_block, policy_fn,
                       db_path) -> list:
    """Replay the pair over one 264-candle block exactly as the
    duel does (10 episodes, starts k*24, 24 steps, next-open
    fills, broker reset per episode) and return every entry
    attempt as (decision_idx, side), decision_idx local to the
    block. `policy_fn(obs) -> action` is the brain (the runner
    passes run 44's deterministic predict)."""
    recorder: list = []
    for ep, start in enumerate([k * 24 for k in range(10)]):
        broker = PaperBroker(db_path,
                             starting_balance=STARTING_BALANCE,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = GatedTradingEnv(sup_block, votes_block, broker,
                              fill_mode="next_open", recorder=recorder)
        obs, _ = env.reset(seed=ep, options={"start": start})
        done = False
        while not done:
            obs, _r, done, _, _ = env.step(int(policy_fn(obs)))
    return recorder


def build_candidate_dataset(candidates, frame_values, votes,
                            atr_rel, vol21, atr_raw, arrays,
                            db_path):
    """Features + labels for a candidate list: X (n x 104), y,
    and the per-candidate records. Candidates whose standalone
    entry is refused are dropped (counted by the caller as
    len(candidates) - len(records))."""
    X, y, recs = [], [], []
    for pos, side in candidates:
        out = simulate_candidate_label(arrays, int(pos), int(side),
                                       db_path)
        if out is None:
            continue
        X.append(feature_row(frame_values, votes, atr_rel, vol21,
                             atr_raw, int(pos), int(side)))
        y.append(out["label"])
        recs.append({"decision_pos": int(pos), "side": int(side),
                     **out})
    X = (np.array(X, dtype=float) if X
         else np.empty((0, 104), dtype=float))
    return X, np.array(y, dtype=int), recs


def validate_candidates(candidates, n_region: int) -> None:
    """Training-region isolation guard (frozen design): every
    candidate decision must sit inside the training region —
    a candidate at or past the holdout boundary would mean the
    meta-model trained on holdout decisions."""
    for pos, side in candidates:
        assert 0 <= int(pos) < n_region, \
            f"candidate decision outside the training region: {pos}"
        assert int(side) in (1, -1), f"bad candidate side: {side}"


# ---------------------------------------------------------- meta-model
def fit_meta_model(X: np.ndarray, y: np.ndarray):
    """The frozen meta-model: a plain LGBMClassifier, seed 42,
    no class weighting, no feature selection."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)
    return model


def oof_diagnostics(X: np.ndarray, y: np.ndarray,
                    folds: int = 4) -> dict:
    """Time-ordered out-of-fold estimates over the candidate set
    (candidates are time-ordered by construction): expanding
    TimeSeriesSplit, the leading segment uncovered — diagnostics
    only, never graded."""
    from sklearn.metrics import (accuracy_score,
                                 balanced_accuracy_score,
                                 roc_auc_score)
    from sklearn.model_selection import TimeSeriesSplit

    proba = np.full(len(X), np.nan)
    tss = TimeSeriesSplit(n_splits=folds)
    for tr, te in tss.split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        m = fit_meta_model(X[tr], y[tr])
        pos = list(m.classes_).index(1)
        proba[te] = m.predict_proba(X[te])[:, pos]
    mask = np.isfinite(proba)
    out: dict = {"n_oof": int(mask.sum())}
    if mask.sum() and len(np.unique(y[mask])) == 2:
        pred = (proba[mask] >= 0.5).astype(int)
        out["accuracy"] = float(accuracy_score(y[mask], pred))
        out["balanced_accuracy"] = float(
            balanced_accuracy_score(y[mask], pred))
        out["auc"] = float(roc_auc_score(y[mask], proba[mask]))
        out["gate_rate_055"] = float(
            np.mean(proba[mask] >= GATE_THRESHOLD))
    return out


# ------------------------------------------------- the mirror evaluator
def evaluate_gated_ppo(model_path, sup, p_up, allow_fn=None,
                       ledger_dir=None, tag: str = "eval") -> dict:
    """nightly.evaluate_ppo mirrored line-for-line — same episode
    starts, broker config, scoring (score = total P&L - 0.5 x
    max episode drawdown), bench — with two deliberate,
    behavior-neutral differences: the env is GatedTradingEnv
    (vetoed entries are substituted with hold before the
    production step; allow_fn=None is exactly production), and
    each EPISODE gets its own ledger file under `ledger_dir`
    (the broker DB is a write-only sink — broker state is
    in-memory and reset per episode, so per-episode files
    change no trading math; they exist so the ledger re-sum can
    audit each episode independently). The action mix counts
    EXECUTED actions (vetoed = hold): the gated system's real
    activity, as frozen in the design. Returns the evaluate_ppo
    dict plus `gate_stats` (attempts / vetoes / filled entries /
    fills / fees, fees read from the persisted ledgers)."""
    from stable_baselines3 import PPO

    n = len(sup)
    assert n >= 10 * 24, f"holdout too short: {n} candles"
    starts = [k * 24 for k in range(10)]
    model = PPO.load(model_path)
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    stats = {"entries_attempted": 0, "entries_vetoed": 0,
             "entries_filled": 0, "fills": 0, "fees": 0.0}
    for ep, start in enumerate(starts):
        if ledger_dir is not None:
            db = Path(ledger_dir) / f"{tag}_ep{ep}.db"
            db.parent.mkdir(parents=True, exist_ok=True)
        else:
            import os
            db = Path("/tmp") / f"tam_meta_eval_{os.getpid()}_{tag}_{ep}.db"
        db.unlink(missing_ok=True)
        broker = PaperBroker(db, starting_balance=STARTING_BALANCE,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = GatedTradingEnv(sup, p_up, broker, fill_mode="next_open",
                              allow_fn=allow_fn)
        obs, _ = env.reset(seed=ep, options={"start": start})
        start_eq = env._day_start_equity
        curve = [start_eq]
        done, ep_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            action_counts[ACTION_KEYS[env.executed_action]] += 1
            ep_r += r
            curve.append(env._prev_eq)
        rewards.append(ep_r)
        pnls.append(env._prev_eq - start_eq)
        trips.append(env._round_trips)
        c = np.array(curve)
        ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
        stats["entries_attempted"] += env.entries_attempted
        stats["entries_vetoed"] += env.entries_vetoed
        stats["entries_filled"] += env.entries_filled
        with sqlite3.connect(db) as con:
            row = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(fee), 0) FROM trades"
            ).fetchone()
        stats["fills"] += int(row[0])
        stats["fees"] += float(row[1])
        if ledger_dir is None:
            db.unlink(missing_ok=True)
    total = float(np.sum(pnls))
    max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
    total_actions = sum(action_counts.values()) or 1
    closes = sup["close"].to_numpy(dtype=float)
    bench = (float(100_000.0 * (closes[-1] / closes[0] - 1.0))
             if len(closes) > 1 else 0.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "total_pnl": total,
        "max_drawdown": max_dd,
        "mean_drawdown": float(np.mean(ep_dds)) if ep_dds else 0.0,
        "score": total - 0.5 * max_dd,
        "bench_pnl": bench,
        "episodes": 10,
        "round_trips_per_day": float(np.mean(trips)) if trips else 0.0,
        "action_mix": {k: round(v / total_actions, 3)
                       for k, v in action_counts.items()},
        "gate_stats": stats,
    }


def evaluate_gated_windows(model_path, windows, p_ups, allow_fns,
                           ledger_dir=None, tag: str = "eval"):
    """evaluate_windows for gated arms: per-window
    evaluate_gated_ppo dicts plus the aggregate shape nightly's
    _agg consumes, and summed gate_stats. None if any window
    fails to run (production's contract)."""
    per = []
    try:
        for w, (w_sup, w_pup, fn) in enumerate(
                zip(windows, p_ups, allow_fns)):
            per.append(evaluate_gated_ppo(
                model_path, w_sup, w_pup, allow_fn=fn,
                ledger_dir=ledger_dir, tag=f"{tag}w{w}"))
    except Exception as e:
        print(f"gated evaluation failed ({type(e).__name__}): {e}",
              flush=True)
        return None
    stats = {k: sum(w["gate_stats"][k] for w in per)
             for k in per[0]["gate_stats"]}
    return {"windows": per,
            "agg_score": float(np.mean([w["score"] for w in per])),
            "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
            "mean_reward": float(np.mean(
                [w["mean_reward"] for w in per])),
            "gate_stats": stats}


# ------------------------------------------------------------- re-sum
def resum_episode_pnl(db_path, final_close: float,
                      start_equity: float = 100_000.0) -> float:
    """Independent episode P&L from the ledger ALONE: replay the
    fills' cash/position arithmetic from (side, price, qty, fee)
    — BUY pays qty*price + fee, SELL receives qty*price - fee
    (the broker's four fill types collapse to exactly this) —
    auditing every step against the broker's recorded
    cash_after / btc_after, then mark any open position at
    `final_close`. Raises AssertionError on any disagreement:
    this is the second computation of the headline P&L, not a
    re-read of the first."""
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT side, price, qty, fee, cash_after, btc_after "
            "FROM trades ORDER BY id").fetchall()
    cash, btc = float(start_equity), 0.0
    for side, price, qty, fee, ca, ba in rows:
        if side == "BUY":
            cash -= qty * price + fee
            btc += qty
        elif side == "SELL":
            cash += qty * price - fee
            btc -= qty
        else:
            raise ValueError(f"unknown fill side: {side!r}")
        assert abs(cash - ca) < 1e-4, (cash, ca)
        assert abs(btc - ba) < 1e-9, (btc, ba)
    return cash + btc * float(final_close) - float(start_equity)


__all__ = [
    "FAMILY", "GATE_THRESHOLD", "SENSITIVITY_THRESHOLDS",
    "META_CONTEXT_COLUMNS", "LGBM_PARAMS", "ACTION_KEYS",
    "is_entry_attempt", "feature_row", "MetaGate",
    "GatedTradingEnv", "simulate_candidate_label",
    "extract_candidates", "build_candidate_dataset",
    "validate_candidates", "fit_meta_model", "oof_diagnostics",
    "evaluate_gated_ppo", "evaluate_gated_windows",
    "resum_episode_pnl",
]
