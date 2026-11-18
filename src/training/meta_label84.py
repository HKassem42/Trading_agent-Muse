"""Meta-Label Gate — run 84, the activity-calibrated gate
(2026-10-30).

Design frozen in refs/meta-label-study-84-2026-10-30.md BEFORE
any run-84 holdout run. Kass's order, after run 83 passed on a
single filled trade: make the models trade ~30% of the time
instead of ~10%, rerun, and get CLEAR results.

Run 84 changes the OPERATING POINT, not the eyes:

- The gate model is run 83's frozen fitted model, byte-
  identical (the stacker-recipe winner of run 83's OOF zoo).
  No refit, no new model class, no new features — the
  114-vector, the FIFO in-context histories, and the gate
  mechanics are meta_label83's, reused.
- The graded threshold tau84 is CALIBRATED on training-region
  evidence only: the 70th percentile of the frozen recipe's
  TimeSeriesSplit(4) out-of-fold probabilities over the
  stream-1 training candidates — the line that would have
  passed ~30% of stream-1 entry attempts in training. It is
  never adjusted after seeing holdout behavior.
- Primary arm (graded): stream 1 (the champion pair's
  entries) gated at tau84 in the standard 3-window duel.
- Secondary arm (reported): BOTH proposal streams gated at
  tau84 in the same windows — the pair's entries plus the
  vote-crossing rule's entries on the steps where the PPO
  holds (PPO precedence; a vetoed stream-1 entry consumes
  the step). It attacks the deeper scarcity Kass named: the
  pair alone attempts only 6-8 entries per 33 days.

Grading is run 82/83's, unchanged: the shared 33-day holdout,
the production formula plus gate_decision()'s two-part floor
on the primary arm, pass-all equivalence and run-83
reproduction pins on the machinery, ledger re-sum two-way
checks on every arm. Lab only, promotion lock: nothing here
installs, crowns, or touches a seat or a book.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from training import meta_label as ml
from training import meta_label83 as m83

TARGET_PASS_RATE = 0.30
STREAM1 = m83.STREAM_PAIR    # 1 — the champion pair's proposals
STREAM2 = m83.STREAM_VOTES   # 0 — the vote-crossing rule's proposals


# ------------------------------------------------------- calibration
def calibrated_threshold(oof_proba, stream_ids,
                         target_pass_rate: float = TARGET_PASS_RATE,
                         stream: int = STREAM1) -> dict:
    """tau = the (1 - target_pass_rate) quantile (numpy linear
    interpolation) of the OOF probabilities of ONE stream's
    OOF-covered candidates. Training-region provenance by
    construction: the function sees only the probability and
    stream arrays it is handed — the caller hands it the
    training region's regenerated OOF probabilities, never
    in-sample fitted probabilities and never holdout rows.
    The implied pass rate uses the gate's own allow rule
    (proba >= tau)."""
    p = np.asarray(oof_proba, dtype=float)
    s = np.asarray(stream_ids)
    sel = p[(s == int(stream)) & np.isfinite(p)]
    if len(sel) == 0:
        raise ValueError(
            f"no OOF-covered candidates for stream {stream}: "
            "calibration impossible")
    q = 1.0 - float(target_pass_rate)
    tau = float(np.quantile(sel, q))
    return {"threshold": tau,
            "quantile": q,
            "n_calibration": int(len(sel)),
            "implied_pass_rate": float(np.mean(sel >= tau))}


# ------------------------------------------------------------ the gate
class Gate84:
    """Run 83's frozen fitted model at an arbitrary threshold,
    with PER-STREAM in-context histories over ONE duel book.

    MetaGate83's contract and timing, generalized: `allow(
    idx, side)` is the function GatedTradingEnv consumes; it
    scores the candidate for `current_stream` (set by the
    evaluator before each step) — that stream's own win-rate
    history and stream-id feature — and counts the attempt /
    veto per stream. One TrancheTracker rides the book; every
    entry step tells after_step which stream's attempt it
    submitted, so a tranche's outcome lands in the history of
    the stream whose entry opened it. Positions are unified
    exactly as run 83's (training outcomes at training
    positions; holdout at HOLDOUT_POS_BASE + window base +
    local index), so the market-time <= filter is causal
    everywhere. Histories are read as of the last after_step
    sync — MetaGate83's decision timing, unchanged."""

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = float(threshold)
        self.tracker = m83.TrancheTracker()
        self.histories: dict[int, list] = {STREAM1: [], STREAM2: []}
        self.attempts: dict[int, int] = {STREAM1: 0, STREAM2: 0}
        self.vetoes: dict[int, int] = {STREAM1: 0, STREAM2: 0}
        self.current_stream = STREAM1
        self._hist_seen = 0
        self.window_base = 0
        self.reader: m83.LedgerFillReader | None = None
        self.frame = self.votes = self.atr_rel = None
        self.vol21 = self.atr_raw = self.extras = None
        classes = list(model.classes_)
        self._pos = classes.index(1) if 1 in classes else None

    def seed_history(self, stream: int, outcomes) -> None:
        """outcomes: iterable of (close_pos, label) on the
        TRAINING timeline for that stream."""
        self.histories[int(stream)] = sorted(
            (int(cp), int(lab)) for cp, lab in outcomes)

    def set_window(self, window_idx: int, frame_values, votes,
                   atr_rel, vol21, atr_raw, extras) -> None:
        """Swap in one duel window's feature arrays (window-
        local indexing, run 82/83's convention)."""
        self.window_base = int(window_idx) * 264
        self.frame = np.asarray(frame_values, dtype=float)
        self.votes = np.asarray(votes, dtype=float)
        self.atr_rel = np.asarray(atr_rel, dtype=float)
        self.vol21 = np.asarray(vol21, dtype=float)
        self.atr_raw = np.asarray(atr_raw, dtype=float)
        self.extras = extras

    def begin_episode(self, reader: m83.LedgerFillReader) -> None:
        reader.pos_offset = m83.HOLDOUT_POS_BASE + self.window_base
        self.reader = reader
        self.current_stream = STREAM1

    def _drain_outcomes(self) -> None:
        outs = self.tracker.outcomes
        for o in outs[self._hist_seen:]:
            tag = o["tag"]
            if tag not in (STREAM1, STREAM2):
                raise AssertionError(
                    f"duel tranche finalized with tag {tag!r}: an "
                    "entry fill reached the book without a stream "
                    "tag — outcome attribution would be wrong")
            self.histories[int(tag)].append(
                (int(o["close_pos"]), int(o["label"])))
        self._hist_seen = len(outs)

    def _sync(self, attempt_stream) -> None:
        if self.reader is None:
            return
        fills = self.reader.new_fills()
        if fills or attempt_stream is not None:
            self.tracker.process_step(fills,
                                      attempt_tag=attempt_stream)
        self._drain_outcomes()

    def after_step(self, attempt_stream=None) -> None:
        """Sync the tracker from the episode ledger after one
        env step. `attempt_stream` is the stream whose entry
        attempt the step submitted (None for hold / management
        / stop-only steps): a tranche the step opened carries
        that stream's tag."""
        self._sync(attempt_stream)

    def end_episode(self, mark_price: float,
                    mark_pos_local: int) -> None:
        self._sync(None)
        self.tracker.end_episode(
            float(mark_price),
            m83.HOLDOUT_POS_BASE + self.window_base
            + int(mark_pos_local))
        self._drain_outcomes()

    def proba(self, idx: int, side: int) -> float:
        if self._pos is None:
            return 0.0
        idx = int(idx)
        cur = int(self.current_stream)
        win20, winexp = m83.stream_win_features(
            self.histories[cur],
            m83.HOLDOUT_POS_BASE + self.window_base + idx)
        row = m83.feature_row_83(
            self.frame, self.votes, self.atr_rel, self.vol21,
            self.atr_raw, self.extras, idx, int(side), win20,
            winexp, float(cur))
        return float(self.model.predict_proba(
            row.reshape(1, -1))[0, self._pos])

    def allow(self, idx: int, side: int) -> bool:
        cur = int(self.current_stream)
        self.attempts[cur] += 1
        ok = self.proba(idx, side) >= self.threshold
        if not ok:
            self.vetoes[cur] += 1
        return ok


# ------------------------------------------------------------ the duel
def evaluate_ppo_84(model_path, sup: pd.DataFrame,
                    p_up: np.ndarray, gate: Gate84 | None = None,
                    mode: str = "pair", ledger_dir=None,
                    tag: str = "eval84", policy_fn=None) -> dict:
    """meta_label83.evaluate_gated_ppo_83 mirrored — same
    episode starts, broker config, scoring, bench, and
    gate_stats read from the persisted ledgers — with three
    behavior-neutral additions and one frozen behavior change:

    - per-step book sampling -> `time_in_position` (share of
      the window's steps after which the book held a non-zero
      position) plus raw `steps` / `positioned_steps`;
    - the gate hooks take the submitted attempt's stream tag
      (pair mode: always stream 1);
    - `policy_fn(obs) -> action` may replace the PPO (tests);
    - mode="combined" ONLY: on steps where the brain HOLDS,
      the vote-crossing rule (meta_label83.rule_action on this
      window's votes, the book's current position) proposes
      instead — its entries are gated as stream 2 via
      gate.current_stream, its exits execute ungated. A
      non-hold brain action always consumes the step (PPO
      precedence, frozen in the design).

    With gate=None and mode="pair" it must reproduce the
    production evaluator exactly (pinned by the study)."""
    from stable_baselines3 import PPO

    from trading.broker import PaperBroker

    n = len(sup)
    assert n >= 10 * 24, f"holdout too short: {n} candles"
    assert mode in ("pair", "combined"), mode
    starts = [k * 24 for k in range(10)]
    model = PPO.load(model_path) if policy_fn is None else None
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    stats = {"entries_attempted": 0, "entries_vetoed": 0,
             "entries_filled": 0, "fills": 0, "fees": 0.0}
    closes = sup["close"].to_numpy(dtype=float)
    steps_total = 0
    steps_positioned = 0
    for ep, start in enumerate(starts):
        if ledger_dir is not None:
            db = Path(ledger_dir) / f"{tag}_ep{ep}.db"
            db.parent.mkdir(parents=True, exist_ok=True)
        else:
            import os
            db = Path("/tmp") / f"tam_meta84_eval_{os.getpid()}_{tag}_{ep}.db"
        db.unlink(missing_ok=True)
        broker = PaperBroker(db, starting_balance=ml.STARTING_BALANCE,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = ml.GatedTradingEnv(
            sup, p_up, broker, fill_mode="next_open",
            allow_fn=(gate.allow if gate is not None else None))
        if gate is not None:
            gate.begin_episode(m83.LedgerFillReader(db))
        obs, _ = env.reset(seed=ep, options={"start": start})
        start_eq = env._day_start_equity
        curve = [start_eq]
        done, ep_r = False, 0.0
        while not done:
            if policy_fn is not None:
                a1 = int(policy_fn(obs))
            else:
                a1 = int(model.predict(obs, deterministic=True)[0])
            submit = a1
            attempt_stream = None
            if gate is not None:
                gate.current_stream = STREAM1
            if ml.is_entry_attempt(a1, env.broker.btc) != 0:
                attempt_stream = STREAM1
            if mode == "combined" and a1 == 0:
                a2 = m83.rule_action(p_up, env._cursor, env.broker.btc)
                if a2 != 0:
                    submit = a2
                    if ml.is_entry_attempt(a2, env.broker.btc) != 0:
                        attempt_stream = STREAM2
                        if gate is not None:
                            gate.current_stream = STREAM2
            obs, r, done, _, _ = env.step(submit)
            action_counts[ml.ACTION_KEYS[env.executed_action]] += 1
            ep_r += r
            curve.append(env._prev_eq)
            steps_total += 1
            if abs(env.broker.btc) > 1e-9:
                steps_positioned += 1
            if gate is not None:
                gate.after_step(attempt_stream)
        if gate is not None:
            gate.end_episode(float(closes[start + 24]), start + 24)
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
        "steps": steps_total,
        "positioned_steps": steps_positioned,
        "time_in_position": (steps_positioned / steps_total
                             if steps_total else 0.0),
    }


def evaluate_windows_84(model_path, windows: list[pd.DataFrame],
                        p_ups: list[np.ndarray], gate_factory=None,
                        mode: str = "pair", ledger_dir=None,
                        tag: str = "eval84", policy_fn=None) -> dict:
    """3-window duel arm, run 82/83's aggregation shape:
    gate_factory(window_idx) -> the arm's Gate84 with that
    window's arrays already set (or None for the pass-all
    control). Per-episode ledgers persist under ledger_dir as
    {tag}w{w}_ep{ep}.db for the re-sum check. Adds the pooled
    time-in-position share and the gate's per-stream counters.
    Exceptions propagate: a broken gate fails the study
    loudly, not as a lost duel."""
    per = []
    last_gate = None
    for w, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        gate = gate_factory(w) if gate_factory is not None else None
        last_gate = gate
        per.append(evaluate_ppo_84(
            model_path, w_sup, w_pup, gate=gate, mode=mode,
            ledger_dir=ledger_dir, tag=f"{tag}w{w}",
            policy_fn=policy_fn))
    stats = {k: sum(win["gate_stats"][k] for win in per)
             for k in per[0]["gate_stats"]}
    steps = sum(win["steps"] for win in per)
    positioned = sum(win["positioned_steps"] for win in per)
    out = {"windows": per,
           "agg_score": float(np.mean([w["score"] for w in per])),
           "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
           "mean_reward": float(np.mean(
               [w["mean_reward"] for w in per])),
           "gate_stats": stats,
           "steps": steps,
           "positioned_steps": positioned,
           "time_in_position": (positioned / steps if steps else 0.0)}
    if last_gate is not None:
        out["gate_by_stream"] = {
            "attempts": {str(k): v
                         for k, v in last_gate.attempts.items()},
            "vetoes": {str(k): v
                       for k, v in last_gate.vetoes.items()}}
    return out
