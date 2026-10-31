"""Meta-Label Gate — run 83, the strengthened gate (2026-10-30).

Design frozen in refs/meta-label-study-83-2026-10-30.md BEFORE
any grading. Run 83 attacks run 82's four diagnosed weaknesses
(src/training/meta_label.py, which stays untouched so run 82's
pipeline remains reproducible):

1. Weak eyes (OOF balanced accuracy 0.518): a model zoo
   compared on training-region OOF only — run 82's LightGBM
   (control), the program's stacker recipe, and each base with
   isotonic / Platt calibration fitted on OOF predictions.
   Selection: highest OOF balanced accuracy, ties to AUC. The
   duel is graded with ONLY the selected model.
2. Thin data (489 candidates): two proposal streams over the
   full training region — the champion pair replayed at
   episode-start offsets 0/8/16 (stream 1, the duel's stream)
   plus a deterministic stacker-vote crossing rule policy
   (stream 2). Every candidate is a real simulated decision
   under production broker rules.
3. Standalone labels (a candidate's fate simulated in
   isolation): FIFO tranche attribution INSIDE each replay
   trajectory — a candidate is its entry fill; the label is
   the sign of that tranche's realized contribution (pro-rata
   entry/exit fees; tranches still open at episode end are
   marked at the episode's final close with the entry fee
   charged and no exit fee, the env's own accounting).
4. Uncalibrated probabilities: calibration variants above, and
   ten causal context features appended to run 82's 104-vector
   (vote slopes 6/24, the candidate stream's own causal win
   rates, implied stop distance in ATR units, volatility
   percentile, drawdown from the 100-candle high, hour cycle,
   stream id) for a 114-vector.

Grading is run 82's, unchanged: the shared 33-day holdout,
the standard 3-window duel, the production formula plus
gate_decision()'s two-part floor, graded threshold 0.55.
"""

from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from training import meta_label as ml

N_FEATURES_83 = 114
STREAM_PAIR = 1    # stream-id feature value for the pair replay
STREAM_VOTES = 0   # ... and for the vote-crossing rule policy
# Holdout positions are mapped far past every training position
# so one (close_pos <= decision_pos) market-time filter serves
# both regions and the gate's seeded history.
HOLDOUT_POS_BASE = 10_000_000
VARIANTS = ("lgbm82", "stacker", "lgbm82+isotonic", "lgbm82+platt",
            "stacker+isotonic", "stacker+platt")


# ------------------------------------------------------- causal series
def vote_slopes(votes: np.ndarray, k: int) -> np.ndarray:
    """votes[t] - votes[t-k]; 0.0 while t-k is before the
    region's start (there is no earlier vote in the region)."""
    votes = np.asarray(votes, dtype=float)
    out = np.zeros(len(votes))
    if len(votes) > k:
        out[k:] = votes[k:] - votes[:-k]
    return out


def stop_distance_atr_units(atr_raw: np.ndarray) -> np.ndarray:
    """The production stop distance in ATR units: mirrors
    PaperBroker._stop_pct_for with the defaults every study
    broker runs (mult 2.0, floor 2.5%, cap 8%). 0.0 where the
    ATR is not finite-positive (the broker's flat-stop case)."""
    atr = np.asarray(atr_raw, dtype=float)
    out = np.zeros(len(atr))
    ok = np.isfinite(atr) & (atr > 0)
    if ok.any():
        pct = np.minimum(np.maximum(2.0 * atr[ok], 0.025), 0.08)
        out[ok] = pct / atr[ok]
    return out


def volatility_percentile(vol: np.ndarray, window: int = 500) -> np.ndarray:
    """vol[t] ranked against the trailing `window` values
    strictly before t: the fraction below it. 0.5 with no
    history. Trailing only — causal by construction."""
    vol = np.asarray(vol, dtype=float)
    out = np.full(len(vol), 0.5)
    for t in range(len(vol)):
        lo = max(0, t - window)
        if t > lo:
            out[t] = float(np.mean(vol[lo:t] < vol[t]))
    return out


def drawdown_from_high(closes: np.ndarray, window: int = 100) -> np.ndarray:
    """close[t] / max(close[t-window+1 .. t]) - 1 (<= 0). The
    window includes t itself and nothing after it — causal."""
    closes = np.asarray(closes, dtype=float)
    out = np.zeros(len(closes))
    for t in range(len(closes)):
        lo = max(0, t - window + 1)
        peak = closes[lo:t + 1].max()
        out[t] = closes[t] / peak - 1.0 if peak > 0 else 0.0
    return out


def hour_sincos(hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ang = 2.0 * np.pi * np.asarray(hours, dtype=float) / 24.0
    return np.sin(ang), np.cos(ang)


def build_extra_context(votes, atr_raw, vol21, closes, hours) -> dict:
    """All causal context series for one region (training or
    holdout), each the region's length, value at t reading only
    candles <= t of that region."""
    sin, cos = hour_sincos(hours)
    return {
        "slope6": vote_slopes(votes, 6),
        "slope24": vote_slopes(votes, 24),
        "stop_units": stop_distance_atr_units(atr_raw),
        "vol_pct": volatility_percentile(vol21, 500),
        "dd100": drawdown_from_high(closes, 100),
        "hour_sin": sin,
        "hour_cos": cos,
    }


# ------------------------------------------------- stream history feats
def stream_win_features(outcomes, decision_pos: int) -> tuple[float, float]:
    """(last-20 win rate, expanding win rate) over one stream's
    outcomes whose close_pos <= decision_pos (market time),
    Laplace-smoothed (wins + 2) / (n + 4); (0.5, 0.5) with no
    history. `outcomes` is an iterable of (close_pos, label)."""
    ordered = [lab for _cp, lab in
               sorted(((cp, lab) for cp, lab in outcomes
                       if cp <= decision_pos), key=lambda z: z[0])]
    if not ordered:
        return 0.5, 0.5
    exp_rate = (sum(ordered) + 2.0) / (len(ordered) + 4.0)
    last = ordered[-20:]
    win20 = (sum(last) + 2.0) / (len(last) + 4.0)
    return float(win20), float(exp_rate)


# ------------------------------------------------------------ features
def feature_row_83(frame_values, votes, atr_rel, vol21, atr_raw,
                   extras: dict, idx: int, side: int,
                   win20: float, winexp: float,
                   stream_id: float) -> np.ndarray:
    """Run 82's 104-vector at (idx, side), then the ten frozen
    context values in registration order -> a 114-vector.
    `idx` is a position in the region whose arrays are passed."""
    base = ml.feature_row(frame_values, votes, atr_rel, vol21,
                          atr_raw, idx, side)
    extra = np.array([
        extras["slope6"][idx], extras["slope24"][idx],
        float(win20), float(winexp),
        extras["stop_units"][idx], extras["vol_pct"][idx],
        extras["dd100"][idx], extras["hour_sin"][idx],
        extras["hour_cos"][idx], float(stream_id),
    ])
    return np.concatenate([base, extra])


# ------------------------------------------- FIFO tranche attribution
class _Tranche:
    __slots__ = ("sign", "qty", "price", "fee", "tag", "realized")

    def __init__(self, sign, qty, price, fee, tag):
        self.sign = sign
        self.qty = qty
        self.price = price
        self.fee = fee
        self.tag = tag
        self.realized = 0.0


class TrancheTracker:
    """Consumes broker ledger fills in id order and attributes
    P&L to entry tranches FIFO — the in-context label of run 83.

    Fill semantics mirror PaperBroker: a fill moving the book
    AWAY from zero opens a tranche (its entry fee attaches to
    it); a fill moving the book TOWARD zero closes open
    tranches of the opposite sign, oldest first. A closing
    slice realizes qty * (exit - entry) * sign minus its
    pro-rata share of the entry fee and of the closing fill's
    fee. Tranches still open at end_episode are marked at the
    given price (remaining entry fee charged, no exit fee).

    Protocol: one process_step call per env step, with that
    step's new fills and the step's entry-attempt tag (or
    None). An attempt whose step opened no tranche was refused
    — counted in dropped_attempts, no candidate.
    """

    def __init__(self):
        self._tranches: list[_Tranche] = []
        self.outcomes: list[dict] = []
        self.dropped_attempts = 0

    @property
    def position(self) -> float:
        return sum(t.sign * t.qty for t in self._tranches)

    def _finalize(self, t: _Tranche, close_pos: int) -> None:
        self.outcomes.append({
            "tag": t.tag,
            "contribution": float(t.realized),
            "label": int(t.realized > 0),
            "close_pos": int(close_pos),
        })

    def _close(self, qty: float, price: float, fee: float,
               fill_pos: int) -> float:
        """Close up to `qty` against open tranches FIFO; returns
        the unclosed remainder (0 under broker semantics)."""
        remaining = qty
        while remaining > 1e-12 and self._tranches:
            t = self._tranches[0]
            cq = min(t.qty, remaining)
            fee_in = t.fee * (cq / t.qty) if t.qty > 0 else 0.0
            fee_out = fee * (cq / qty) if qty > 0 else 0.0
            t.realized += cq * (price - t.price) * t.sign \
                - fee_in - fee_out
            t.qty -= cq
            t.fee -= fee_in
            remaining -= cq
            if t.qty <= 1e-12:
                self._tranches.pop(0)
                self._finalize(t, fill_pos)
        return remaining

    def process_step(self, fills, attempt_tag=None) -> None:
        opened: list[_Tranche] = []
        for side, price, qty, fee, fill_pos in fills:
            qty = float(qty)
            if qty <= 0:
                continue
            if side == "BUY":
                if self.position < -1e-12:
                    qty = self._close(min(qty, -self.position),
                                      float(price), float(fee),
                                      int(fill_pos))
                if qty > 1e-12:
                    t = _Tranche(+1, qty, float(price), float(fee), None)
                    self._tranches.append(t)
                    opened.append(t)
            elif side == "SELL":
                if self.position > 1e-12:
                    qty = self._close(min(qty, self.position),
                                      float(price), float(fee),
                                      int(fill_pos))
                if qty > 1e-12:
                    t = _Tranche(-1, qty, float(price), float(fee), None)
                    self._tranches.append(t)
                    opened.append(t)
        if attempt_tag is not None:
            if opened:
                opened[0].tag = attempt_tag
            else:
                self.dropped_attempts += 1

    def end_episode(self, mark_price: float, mark_pos: int) -> None:
        for t in self._tranches:
            t.realized += t.qty * (float(mark_price) - t.price) \
                * t.sign - t.fee
            self._finalize(t, mark_pos)
        self._tranches = []


class LedgerFillReader:
    """Incremental reader over one PaperBroker ledger DB:
    new_fills() returns fills since the last call as
    (side, price, qty, fee, fill_pos) in id order, where
    fill_pos = int(ts) + pos_offset — int(ts) is the fill
    candle's position in the env's (reset) frame (production
    ledger semantics); pos_offset maps it onto the caller's
    timeline (the block base in extraction, the unified
    holdout base in the duel). Callers set pos_offset before
    the episodes it applies to."""

    def __init__(self, db_path, pos_offset: int = 0):
        self.db_path = str(db_path)
        self.last_id = 0
        self.pos_offset = int(pos_offset)

    def new_fills(self) -> list[tuple]:
        if not Path(self.db_path).exists():
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id, side, price, qty, fee, ts FROM trades "
                "WHERE id > ? ORDER BY id", (self.last_id,)).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
        fills = []
        for rid, side, price, qty, fee, ts in rows:
            self.last_id = max(self.last_id, int(rid))
            fills.append((side, float(price), float(qty),
                          float(fee), int(ts) + self.pos_offset))
        return fills


# ------------------------------------------------------------ replay
def rule_action(votes: np.ndarray, t: int, btc: float) -> int:
    """Stream 2's deterministic policy on the stacker vote at
    candle t (production action encoding): flat + cross up
    through 0.55 -> buy; flat + cross down through 0.45 ->
    short; long + cross down -> sell; short + cross up ->
    cover (buy); otherwise hold. No adds, ever."""
    if t <= 0:
        return 0
    prev, now = float(votes[t - 1]), float(votes[t])
    cross_up = prev < 0.55 <= now
    cross_down = prev > 0.45 >= now
    if btc > 1e-12:
        return 2 if cross_down else 0
    if btc < -1e-12:
        return 1 if cross_up else 0
    if cross_up:
        return 1
    if cross_down:
        return 3
    return 0


def _episode_loop(env, act_fn, tracker, reader, mark_price,
                  mark_pos, tagger, seed, start):
    obs, _ = env.reset(seed=seed, options={"start": start})
    done = False
    while not done:
        n_att = len(env.recorder)
        obs, _r, done, _tr, _info = env.step(int(act_fn(obs)))
        tag = None
        if len(env.recorder) > n_att:
            entry = env.recorder[-1]
            tag = tagger(entry) if tagger is not None else entry
        tracker.process_step(reader.new_fills(), attempt_tag=tag)
    tracker.end_episode(float(mark_price), int(mark_pos))
    return list(env.recorder)


def run_policy_episode(env, policy_fn, tracker, reader, mark_price,
                       mark_pos, tagger=None, seed=0, start=0):
    """Step one GatedTradingEnv episode under `policy_fn`
    (obs -> action), production reset semantics (seed +
    start option, exactly as the duel replays), syncing the
    ledger into `tracker` after every step. Returns the env's
    recorder entries."""
    return _episode_loop(env, lambda _obs: policy_fn(_obs), tracker,
                         reader, mark_price, mark_pos, tagger,
                         seed, start)


def run_rule_episode(env, votes_block, tracker, reader, mark_price,
                     mark_pos, tagger=None, seed=0, start=0):
    """One episode under the stream-2 vote-crossing rule."""
    return _episode_loop(
        env, lambda _obs: rule_action(votes_block, env._cursor,
                                      env.broker.btc),
        tracker, reader, mark_price, mark_pos, tagger, seed, start)


def extract_stream_episodes(sup, votes, policy_fn, db_path,
                            offsets=(0,), stream=STREAM_PAIR,
                            rule=False):
    """Replay episodes over the region and collect tranche-
    attributed candidates, exactly as the duel replays (run
    82's extract_candidates pattern): the region is tiled into
    264-candle blocks; each block's env spans the full block;
    episodes start at offset + k*24 via the reset start option,
    24 steps, next-open fills, a fresh PaperBroker per episode
    on the shared scratch DB at `db_path`. Returns
    (candidates, outcomes, dropped_attempts) where a candidate
    is {tag, decision_pos, side, stream} for every entry
    attempt whose fill opened a tranche, and outcomes are the
    tracker's finalized {tag, contribution, label, close_pos}
    dicts. Candidate/outcome positions are region-global."""
    from trading.broker import PaperBroker

    closes = sup["close"].to_numpy(dtype=float)
    candidates: list[dict] = []
    tracker = TrancheTracker()
    reader = LedgerFillReader(db_path)
    counter = itertools.count()
    meta: dict[int, dict] = {}
    ctx = {"base": 0}

    def tagger(entry):
        tid = next(counter)
        cursor, side = entry
        meta[tid] = {"tag": tid,
                     "decision_pos": ctx["base"] + int(cursor),
                     "side": int(side), "stream": int(stream)}
        return tid

    n = len(sup)
    for bstart in range(0, n - 263, 264):
        block = sup.iloc[bstart:bstart + 264]
        bvotes = np.asarray(votes[bstart:bstart + 264], dtype=float)
        ctx["base"] = bstart
        reader.pos_offset = bstart
        for off in offsets:
            for ep, start in enumerate(
                    [int(off) + k * 24 for k in range(10)]):
                broker = PaperBroker(db_path, starting_balance=100_000,
                                     max_position_pct=0.25,
                                     stop_loss_pct=0.05)
                env = ml.GatedTradingEnv(block, bvotes, broker,
                                         fill_mode="next_open",
                                         recorder=[])
                mark_pos = bstart + start + 24
                if rule:
                    run_rule_episode(env, bvotes, tracker, reader,
                                     closes[mark_pos], mark_pos,
                                     tagger=tagger, seed=ep, start=start)
                else:
                    run_policy_episode(env, policy_fn, tracker, reader,
                                       closes[mark_pos], mark_pos,
                                       tagger=tagger, seed=ep,
                                       start=start)
    seen_tags = set()
    for o in tracker.outcomes:
        if o["tag"] in meta and o["tag"] not in seen_tags:
            seen_tags.add(o["tag"])
            candidates.append(meta[o["tag"]])
    return candidates, list(tracker.outcomes), tracker.dropped_attempts


# ------------------------------------------------------------ model zoo
def _as_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X, dtype=float),
                        columns=[f"f{i}" for i in range(
                            np.asarray(X).shape[1])])


class _FrameModel:
    """Adapter so the program's stacker (fit on DataFrames in
    production) predicts from numpy rows."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.classes_ = pipe.classes_

    def predict_proba(self, X):
        return self.pipe.predict_proba(_as_frame(X))


def fit_base_model(kind: str, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if kind == "lgbm82":
        model = lgb.LGBMClassifier(**ml.LGBM_PARAMS)
        model.fit(X, y)
        return model
    if kind == "stacker":
        from models.stacker import build_pipeline
        pipe = build_pipeline(42)
        pipe.fit(_as_frame(X), y)
        return _FrameModel(pipe)
    raise ValueError(f"unknown base model {kind!r}")


def _class1_proba(model, X) -> np.ndarray:
    raw = model.predict_proba(np.asarray(X, dtype=float))
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(raw))
    return np.asarray(raw, dtype=float)[:, classes.index(1)]


def oof_proba_base(kind: str, X, y, folds: int = 4) -> np.ndarray:
    """Plain TimeSeriesSplit OOF class-1 probabilities (run 82's
    protocol); NaN where no fold covered a row (the leading
    segment). Degenerate single-class folds fall back to the
    fold's training base rate."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    out = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=folds).split(X):
        if len(te) == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            out[te] = float(np.mean(y[tr])) if len(tr) else 0.5
            continue
        model = fit_base_model(kind, X[tr], y[tr])
        out[te] = _class1_proba(model, X[te])
    return out


def _fit_calibrator(method: str, proba, y):
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=int)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return ("isotonic", iso)
    if method == "platt":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p.reshape(-1, 1), y)
        return ("platt", lr)
    raise ValueError(f"unknown calibration {method!r}")


def _apply_calibrator(cal, proba) -> np.ndarray:
    kind, model = cal
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    if kind == "isotonic":
        return np.clip(model.predict(p), 0.0, 1.0)
    if list(model.classes_) == [1]:
        return np.ones(len(p))
    if list(model.classes_) == [0]:
        return np.zeros(len(p))
    return model.predict_proba(p.reshape(-1, 1))[
        :, list(model.classes_).index(1)]


class _CalibratedModel:
    """Base model + a calibrator fitted on the base's OOF
    probabilities. Exposes the sklearn surface the gate uses."""

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = _apply_calibrator(self.calibrator,
                              _class1_proba(self.base, X))
        return np.column_stack([1.0 - p, p])


def calibrated_oof_proba(kind: str, method: str, X, y, folds: int = 4,
                         inner_folds: int = 3) -> np.ndarray:
    """Nested OOF for a calibrated variant: within each outer
    fold the calibrator is fitted on INNER TimeSeriesSplit OOF
    probabilities of the outer-training part (never the outer
    test rows), then applied to the fold's base predictions."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    out = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=folds).split(X):
        if len(te) == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            out[te] = float(np.mean(y[tr])) if len(tr) else 0.5
            continue
        inner = oof_proba_base(kind, X[tr], y[tr], folds=inner_folds)
        mask = ~np.isnan(inner)
        base = fit_base_model(kind, X[tr], y[tr])
        raw = _class1_proba(base, X[te])
        if mask.sum() < 10 or len(np.unique(y[tr][mask])) < 2:
            out[te] = raw
            continue
        cal = _fit_calibrator(method, inner[mask], y[tr][mask])
        out[te] = _apply_calibrator(cal, raw)
    return out


def oof_metrics(proba, y) -> dict:
    """Balanced accuracy at the 0.5 cut and AUC over the OOF-
    covered rows (run 82's reporting)."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    mask = ~np.isnan(proba)
    pc, yc = proba[mask], y[mask]
    auc = None
    if len(np.unique(yc)) == 2 and len(np.unique(pc)) > 1:
        auc = float(roc_auc_score(yc, pc))
    return {
        "n_oof": int(mask.sum()),
        "accuracy": float(np.mean((pc >= 0.5).astype(int) == yc))
        if len(yc) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(
            yc, (pc >= 0.5).astype(int))) if len(yc) else 0.0,
        "auc": auc,
    }


def oof_table(X, y) -> dict:
    """OOF comparison for every registered variant — training
    region only; the holdout is never touched here."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    table = {}
    for variant in VARIANTS:
        if "+" in variant:
            kind, method = variant.split("+")
            proba = calibrated_oof_proba(kind, method, X, y)
        else:
            proba = oof_proba_base(variant, X, y)
        table[variant] = oof_metrics(proba, y)
    return table


def select_variant(table: dict) -> str:
    """Frozen rule: highest OOF balanced accuracy; ties
    (|delta| <= 1e-12) go to the higher AUC; further ties keep
    the earlier variant in VARIANTS order."""
    best, best_key = None, None
    for variant in VARIANTS:
        if variant not in table:
            continue
        m = table[variant]
        key = (round(float(m["balanced_accuracy"]), 12),
               round(float(m["auc"] or 0.0), 12))
        if best_key is None or key > best_key:
            best, best_key = variant, key
    return best


def fit_final_variant(variant: str, X, y):
    """Fit the selected variant on ALL training candidates.
    Calibrated variants: the calibrator is fitted on the base's
    plain TimeSeriesSplit(4) OOF probabilities over the full
    training set."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if "+" in variant:
        kind, method = variant.split("+")
        base = fit_base_model(kind, X, y)
        oof = oof_proba_base(kind, X, y, folds=4)
        mask = ~np.isnan(oof)
        cal = _fit_calibrator(method, oof[mask], y[mask])
        return _CalibratedModel(base, cal)
    return fit_base_model(variant, X, y)


# ------------------------------------------------------------- the gate
class MetaGate83:
    """The run-83 gate inside the duel: run 82's MetaGate
    contract (allow window-local (idx, side); the caller swaps
    the window's feature arrays in via set_window) plus the
    live book context — a TrancheTracker synced from the duel
    ledgers after every step, so the stream win-rate features
    read the duel book's OWN closed tranches (seeded with all
    stream-1 training outcomes) under the same market-time
    filter as training. Positions are unified for the history
    filter: training outcomes keep their training positions;
    holdout positions are HOLDOUT_POS_BASE + holdout index, so
    every training outcome precedes every holdout outcome and
    the <= filter is causal everywhere. One gate object serves
    a whole duel arm (all three windows) so the history and
    tracker persist across windows."""

    def __init__(self, model, threshold: float = ml.GATE_THRESHOLD):
        self.model = model
        self.threshold = float(threshold)
        self.tracker = TrancheTracker()
        self.history: list[tuple[int, int]] = []
        self._hist_seen = 0
        self.window_base = 0
        self.reader: LedgerFillReader | None = None
        self.frame = self.votes = self.atr_rel = None
        self.vol21 = self.atr_raw = self.extras = None
        classes = list(model.classes_)
        self._pos = classes.index(1) if 1 in classes else None

    def seed_history(self, outcomes) -> None:
        """outcomes: iterable of (close_pos, label) on the
        TRAINING timeline (stream 1's finalized outcomes)."""
        self.history = sorted((int(cp), int(lab))
                              for cp, lab in outcomes)

    def set_window(self, window_idx: int, frame_values, votes,
                   atr_rel, vol21, atr_raw, extras) -> None:
        """Swap in one duel window's feature arrays (window-
        local indexing, run 82's MetaGate convention)."""
        self.window_base = int(window_idx) * 264
        self.frame = np.asarray(frame_values, dtype=float)
        self.votes = np.asarray(votes, dtype=float)
        self.atr_rel = np.asarray(atr_rel, dtype=float)
        self.vol21 = np.asarray(vol21, dtype=float)
        self.atr_raw = np.asarray(atr_raw, dtype=float)
        self.extras = extras

    def begin_episode(self, reader: LedgerFillReader) -> None:
        reader.pos_offset = HOLDOUT_POS_BASE + self.window_base
        self.reader = reader

    def _drain_outcomes(self) -> None:
        outs = self.tracker.outcomes
        for o in outs[self._hist_seen:]:
            self.history.append((int(o["close_pos"]), int(o["label"])))
        self._hist_seen = len(outs)

    def _sync(self) -> None:
        if self.reader is None:
            return
        fills = self.reader.new_fills()
        if fills:
            self.tracker.process_step(fills)
        self._drain_outcomes()

    def after_step(self) -> None:
        self._sync()

    def end_episode(self, mark_price: float, mark_pos_local: int) -> None:
        self._sync()
        self.tracker.end_episode(
            float(mark_price),
            HOLDOUT_POS_BASE + self.window_base + int(mark_pos_local))
        self._drain_outcomes()

    def proba(self, idx: int, side: int) -> float:
        if self._pos is None:
            return 0.0
        idx = int(idx)
        win20, winexp = stream_win_features(
            self.history, HOLDOUT_POS_BASE + self.window_base + idx)
        row = feature_row_83(self.frame, self.votes, self.atr_rel,
                             self.vol21, self.atr_raw, self.extras,
                             idx, int(side), win20, winexp,
                             float(STREAM_PAIR))
        return float(self.model.predict_proba(
            row.reshape(1, -1))[0, self._pos])

    def allow(self, idx: int, side: int) -> bool:
        return self.proba(idx, side) >= self.threshold


# ------------------------------------------------------------ the duel
def evaluate_gated_ppo_83(model_path, sup: pd.DataFrame,
                          p_up: np.ndarray, gate: MetaGate83 | None = None,
                          ledger_dir=None, tag: str = "eval83") -> dict:
    """meta_label.evaluate_gated_ppo mirrored line-for-line —
    same episode starts, broker config, scoring, bench, and
    gate_stats read from the persisted ledgers — with the env's
    allow_fn bound to the MetaGate83 and three behavior-neutral
    gate hooks: begin_episode (bind the episode's ledger
    reader) before the step loop, after_step (sync the gate's
    tranche tracker from the ledger) after every step, and
    end_episode (mark open tranches at the episode's final
    close, window-local start + 24) after the loop. With
    gate=None it must reproduce the production evaluator
    exactly (pinned by test)."""
    from stable_baselines3 import PPO

    from trading.broker import PaperBroker

    n = len(sup)
    assert n >= 10 * 24, f"holdout too short: {n} candles"
    starts = [k * 24 for k in range(10)]
    model = PPO.load(model_path)
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    stats = {"entries_attempted": 0, "entries_vetoed": 0,
             "entries_filled": 0, "fills": 0, "fees": 0.0}
    closes = sup["close"].to_numpy(dtype=float)
    for ep, start in enumerate(starts):
        if ledger_dir is not None:
            db = Path(ledger_dir) / f"{tag}_ep{ep}.db"
            db.parent.mkdir(parents=True, exist_ok=True)
        else:
            import os
            db = Path("/tmp") / f"tam_meta83_eval_{os.getpid()}_{tag}_{ep}.db"
        db.unlink(missing_ok=True)
        broker = PaperBroker(db, starting_balance=ml.STARTING_BALANCE,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = ml.GatedTradingEnv(
            sup, p_up, broker, fill_mode="next_open",
            allow_fn=(gate.allow if gate is not None else None))
        if gate is not None:
            gate.begin_episode(LedgerFillReader(db))
        obs, _ = env.reset(seed=ep, options={"start": start})
        start_eq = env._day_start_equity
        curve = [start_eq]
        done, ep_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(int(action))
            action_counts[ml.ACTION_KEYS[env.executed_action]] += 1
            ep_r += r
            curve.append(env._prev_eq)
            if gate is not None:
                gate.after_step()
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
    }


def evaluate_gated_windows_83(model_path, windows: list[pd.DataFrame],
                              p_ups: list[np.ndarray], gate_factory=None,
                              ledger_dir=None, tag: str = "eval83") -> dict:
    """3-window duel arm, run 82's aggregation shape:
    gate_factory(window_idx) -> the arm's MetaGate83 with that
    window's arrays already set (or None for the pass-all
    control). Per-episode ledgers persist under ledger_dir as
    {tag}w{w}_ep{ep}.db for the re-sum check. Exceptions
    propagate (unlike run 82's wrapper): a broken gate must
    fail the study loudly, not read as a lost duel."""
    per = []
    for w, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        gate = gate_factory(w) if gate_factory is not None else None
        per.append(evaluate_gated_ppo_83(
            model_path, w_sup, w_pup, gate=gate,
            ledger_dir=ledger_dir, tag=f"{tag}w{w}"))
    stats = {k: sum(win["gate_stats"][k] for win in per)
             for k in per[0]["gate_stats"]}
    return {"windows": per,
            "agg_score": float(np.mean([w["score"] for w in per])),
            "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
            "mean_reward": float(np.mean(
                [w["mean_reward"] for w in per])),
            "gate_stats": stats}
