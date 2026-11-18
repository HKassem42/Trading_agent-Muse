"""Regime-Swing System — run 86 (LAB ONLY).

Kass's delegation (2026-10-30): "build whatever you want to
get us the best results — if you gotta rebuild the brain or
the eyes or how it trades, do that with everything you've
learned." Design frozen in refs/regime-swing-study-2026-10-30.md
BEFORE any grading: a 4h-clock system whose eyes are a
3-state REGIME classifier (UP / DOWN / CHOP over the forward
4 days) and whose trader is deterministic rules on the
production PaperBroker chassis — no RL brain (three eyes
swaps moved run 44's duel by $0.00; the brain never added
anything over its inputs). Primary validation is
walk-forward on the training region; the standard holdout
duel is secondary (fifth use of that holdout).

Production never imports this module. Labels read future
prices for the OUTCOME only (the meta-label convention);
every feature reads buckets <= t only, and the 4h frame is
built values-based with bucket-END stamps (the v5.1 trap:
pandas label realignment once made higher-timeframe
features describe the bucket STARTING at decision time).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from features.technical_indicators import _atr
from trading.broker import PaperBroker

FAMILY = "regime_swing"
LABEL = "Regime-Swing"
FORWARD_BARS = 24          # 24 x 4h = the forward 4 days
BARRIER_ATR = 0.75         # regime band, in ATR4h units
DOWN, CHOP, UP = 0, 1, 2
CLASS_NAMES = ["DOWN", "CHOP", "UP"]

BLOCK_A_COLUMNS = [
    "rs_ret6", "rs_ret24", "rs_ret42",
    "rs_dist_sma50", "rs_dist_sma200",
    "rs_sma50_slope", "rs_sma200_slope",
    "rs_rv24", "rs_rv96", "rs_rv_ratio", "rs_atr_pct",
    "rs_acf1_42", "rs_kurt_42", "rs_skew_42", "rs_dd100",
    "rs_dow_sin", "rs_dow_cos",
]

ROOT = Path(__file__).resolve().parents[2]


def guard_hashes() -> dict:
    from training.eyes_v2 import guard_hashes as _gh
    return _gh()


# ---------------------------------------------------------- 4h frame
def build_4h_frame(df: pd.DataFrame):
    """(h4, stats): kept 4h buckets stamped at bucket END.

    Aggregation is positional (resample on the raw frame,
    values taken with .to_numpy()); a bucket is kept iff it
    holds exactly 4 hourly candles. h4 columns: open, high,
    low, close, volume, first_pos, last_pos (positions into
    df). Index = bucket END stamps."""
    pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    agg = df.resample("4h", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), n=("close", "size"))
    pres = pos.resample("4h", closed="left", label="left")
    agg["first_pos"] = pres.first().to_numpy()
    agg["last_pos"] = pres.last().to_numpy()
    n_seen = int(len(agg))
    kept = agg[agg["n"] == 4].copy()
    kept.index = kept.index + pd.Timedelta(hours=4)  # bucket END
    end = kept.index
    diffs = (np.diff(end.values) / np.timedelta64(1, "h")
             if len(end) > 1 else np.array([]))
    gaps = sorted({float(g) for g in diffs if g != 4.0})
    stats = {"n_buckets_seen": n_seen,
             "n_kept": int(len(kept)),
             "n_dropped_partial": int(n_seen - len(kept)),
             "end_gap_hours": gaps,
             "n_gaps": int((diffs != 4.0).sum()) if len(diffs) else 0}
    out = kept[["open", "high", "low", "close", "volume",
                "first_pos", "last_pos"]].copy()
    out["first_pos"] = out["first_pos"].astype(int)
    out["last_pos"] = out["last_pos"].astype(int)
    return out, stats


def atr_pct_series(h4: pd.DataFrame) -> np.ndarray:
    """ATR4h(14) / close on the kept-bucket frame (production
    _atr: Wilder true range, EWM alpha=1/14)."""
    atr = _atr(h4, 14)
    return (atr / h4["close"]).replace(
        [np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def regime_labels(close: np.ndarray, atr_pct: np.ndarray,
                  forward: int = FORWARD_BARS,
                  mult: float = BARRIER_ATR) -> np.ndarray:
    """3-state forward-regime labels (int8, -1 = unlabeled).

    net = close[t+forward]/close[t] - 1; thr = mult *
    atr_pct[t]; net > thr -> UP(2), net < -thr -> DOWN(0),
    else CHOP(1). The last `forward` rows and rows with a
    non-finite/non-positive threshold are unlabeled."""
    close = np.asarray(close, dtype=float)
    atr_pct = np.asarray(atr_pct, dtype=float)
    n = len(close)
    labels = np.full(n, -1, dtype=np.int8)
    if n <= forward:
        return labels
    net = close[forward:] / close[:-forward] - 1.0
    thr = mult * atr_pct[:-forward]
    ok = np.isfinite(thr) & (thr > 0) & np.isfinite(net)
    lab = np.where(net > thr, UP, np.where(net < -thr, DOWN, CHOP))
    labels[:-forward] = np.where(ok, lab, -1).astype(np.int8)
    return labels


# ---------------------------------------------------------- features
def block_a_features(h4: pd.DataFrame,
                     atr_pct: np.ndarray):
    """(values (n,17), columns): the 4h-native trend block.
    Every value at bucket t reads buckets <= t only
    (trailing windows END at t; positional throughout)."""
    close = h4["close"].astype(float)
    logret = np.log(close / close.shift(1))
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rv24 = logret.rolling(24).std()
    rv96 = logret.rolling(96).std()
    out = pd.DataFrame(index=h4.index)
    out["rs_ret6"] = close / close.shift(6) - 1.0
    out["rs_ret24"] = close / close.shift(24) - 1.0
    out["rs_ret42"] = close / close.shift(42) - 1.0
    out["rs_dist_sma50"] = close / sma50 - 1.0
    out["rs_dist_sma200"] = close / sma200 - 1.0
    out["rs_sma50_slope"] = sma50 / sma50.shift(6) - 1.0
    out["rs_sma200_slope"] = sma200 / sma200.shift(6) - 1.0
    out["rs_rv24"] = rv24
    out["rs_rv96"] = rv96
    out["rs_rv_ratio"] = (rv24 / rv96.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan)
    out["rs_atr_pct"] = np.asarray(atr_pct, dtype=float)
    out["rs_acf1_42"] = logret.rolling(42).corr(logret.shift(1))
    out["rs_kurt_42"] = logret.rolling(42).kurt()
    out["rs_skew_42"] = logret.rolling(42).skew()
    out["rs_dd100"] = close / close.rolling(100).max() - 1.0
    dow = h4.index.dayofweek.to_numpy(dtype=float)
    out["rs_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["rs_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out[BLOCK_A_COLUMNS].to_numpy(dtype=float), \
        list(BLOCK_A_COLUMNS)


def block_b_features(pick2_values: np.ndarray,
                     df_index: pd.DatetimeIndex,
                     bucket_end_stamps) -> np.ndarray:
    """(n,98): the pick2_98 row at each bucket's LAST 1h
    candle (stamped bucket_end - 1h), taken positionally via
    a stamp->position map — never by label alignment.
    pick2_values is the full pick2 frame's values aligned to
    df_index."""
    pick2_values = np.asarray(pick2_values, dtype=float)
    pos_of = {ts: i for i, ts in enumerate(df_index)}
    n_feat = pick2_values.shape[1]
    rows = np.full((len(bucket_end_stamps), n_feat), np.nan)
    for k, end in enumerate(bucket_end_stamps):
        p = pos_of.get(end - pd.Timedelta(hours=1))
        if p is not None:
            rows[k] = pick2_values[p]
    return rows


# ---------------------------------------------------------- eyes
def fit_eyes(X: np.ndarray, y: np.ndarray, arm: str):
    """Fit one zoo arm ('stacker' = the pick pipeline,
    'lgbm' = run 82's LightGBM config) on encoded labels."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if arm == "stacker":
        from models.muse_pick import build_pick_pipeline
        model = build_pick_pipeline(42)
        model.fit(X, y)
        return model
    if arm == "lgbm":
        from lightgbm import LGBMClassifier

        from training.meta_label import LGBM_PARAMS
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X, y)
        return model
    raise ValueError(f"unknown arm: {arm!r}")


def predict_proba(model, arm: str, X: np.ndarray) -> np.ndarray:
    """(n,3) probabilities aligned to classes [DOWN, CHOP,
    UP] regardless of the fitted model's class order."""
    X = np.asarray(X, dtype=float)
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = [int(c) for c in model.classes_]
    out = np.zeros((len(X), 3))
    for j, c in enumerate(classes):
        out[:, c] = raw[:, j]
    return out


def predict_class(model, arm: str, X: np.ndarray) -> np.ndarray:
    return predict_proba(model, arm, X).argmax(axis=1).astype(int)


def eligible_fit_rows(labels: np.ndarray, boundary_pos: int,
                      forward: int = FORWARD_BARS) -> np.ndarray:
    """Row positions eligible to FIT on for an evaluation
    span starting at bucket position `boundary_pos`: label
    defined AND t + forward < boundary_pos (the label
    horizon ends before the span starts — full purge)."""
    labels = np.asarray(labels)
    return np.array([t for t in range(len(labels))
                     if labels[t] >= 0 and t + forward < boundary_pos],
                    dtype=int)


def class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-class visibility: recall per class, prediction
    counts, base counts, balanced accuracy. A degenerate
    one-class predictor shows up here, never hidden."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    recall, pred_counts, base_counts = {}, {}, {}
    for c in (DOWN, CHOP, UP):
        mask = y_true == c
        base_counts[c] = int(mask.sum())
        pred_counts[c] = int((y_pred == c).sum())
        recall[c] = float(np.mean(y_pred[mask] == c)) if mask.any() \
            else 0.0
    return {"per_class_recall": recall, "pred_counts": pred_counts,
            "base_counts": base_counts,
            "balanced_accuracy": float(np.mean(list(recall.values())))}


def oof_metrics(proba: np.ndarray, y: np.ndarray) -> dict:
    """Pooled-OOF scoring: 3-class balanced accuracy at the
    argmax, macro one-vs-rest AUC, accuracy, class report."""
    from sklearn.metrics import roc_auc_score
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = proba.argmax(axis=1)
    rep = class_report(y, pred)
    try:
        macro_auc = float(roc_auc_score(y, proba, multi_class="ovr",
                                        average="macro"))
    except ValueError:
        macro_auc = 0.5
    return {"balanced_accuracy": rep["balanced_accuracy"],
            "macro_auc": macro_auc,
            "accuracy": float(np.mean(pred == y)),
            "per_class_recall": rep["per_class_recall"],
            "pred_counts": rep["pred_counts"],
            "base_counts": rep["base_counts"],
            "n_oof": int(len(y))}


def select_arm(oof_table: dict) -> str:
    """Frozen zoo selection: highest OOF balanced accuracy;
    ties (|delta| < 1e-12) by higher macro AUC, then arm
    order ('stacker' before 'lgbm')."""
    best_ba = max(v["balanced_accuracy"] for v in oof_table.values())
    cands = [k for k, v in oof_table.items()
             if abs(v["balanced_accuracy"] - best_ba) < 1e-12]
    if len(cands) == 1:
        return cands[0]
    best_auc = max(oof_table[k]["macro_auc"] for k in cands)
    cands = [k for k in cands
             if abs(oof_table[k]["macro_auc"] - best_auc) < 1e-12]
    return sorted(cands)[0]


# ---------------------------------------------------------- trader
def decide(side: int, tranches: int, regime: int) -> str:
    """The frozen rule table. side: +1 long / -1 short / 0
    flat (the broker's actual position). Returns an intent:
    'hold' | 'buy' | 'sell' | 'short' (a 'buy' while short
    is the cover/exit — the env's semantics)."""
    if side == 0:
        if regime == UP:
            return "buy"
        if regime == DOWN:
            return "short"
        return "hold"
    if side > 0:  # long
        if regime == UP:
            return "buy" if tranches < 2 else "hold"
        return "sell"
    # short
    if regime == DOWN:
        return "short" if tranches < 2 else "hold"
    return "buy"


def run_span(df: pd.DataFrame, bucket_first_pos: np.ndarray,
             bucket_last_pos: np.ndarray, span_buckets,
             regimes: np.ndarray, atr_pct: np.ndarray,
             db_path) -> dict:
    """Run the deterministic trader over one span (a
    walk-forward fold or a duel window) on the production
    PaperBroker: flat $100k start, intents fill at the next
    bucket's first candle open with the decision bucket's
    ATR, check_stop on every 1h candle, equity marked at
    every 1h close. Returns the span record (pnl,
    max_drawdown, score, trades, fees, round_trips,
    time_in_position, decisions, action_counts, fills)."""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    stamps = df.index
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    broker = PaperBroker(db_path, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    tranches = 0
    pending = None  # (intent, atr_pct at the decision bucket)
    decisions: list = []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    curve = [100_000.0]
    tip = 0
    total = 0
    round_trips = 0

    def _exec(intent: str, price: float, ts: str, atr) -> None:
        nonlocal tranches, round_trips
        side_before = 0 if broker.btc == 0 else (1 if broker.btc > 0
                                                 else -1)
        fill = None
        if intent == "buy":
            if side_before < 0:
                fill = broker.cover(price, pct=1.0, reason="signal",
                                    ts=ts)
            else:
                fill = broker.buy(price, reason="signal", ts=ts,
                                  atr_pct=atr)
        elif intent == "sell":
            fill = broker.sell(price, pct=1.0, reason="signal", ts=ts)
        elif intent == "short":
            fill = broker.short_open(price, reason="signal", ts=ts,
                                     atr_pct=atr)
        if fill is None:
            return
        if broker.btc == 0:
            if side_before != 0:
                round_trips += 1
            tranches = 0
        elif side_before == 0:
            tranches = 1
        else:
            tranches += 1

    for j, b in enumerate(span_buckets):
        for p in range(int(bucket_first_pos[b]),
                       int(bucket_last_pos[b]) + 1):
            ts = str(stamps[p])
            if p == int(bucket_first_pos[b]) and pending is not None:
                intent, atr_d = pending
                pending = None
                _exec(intent, float(o[p]), ts, atr_d)
            was_in = broker.btc != 0
            broker.check_stop(float(c[p]), ts=ts, low=float(lo[p]),
                              high=float(h[p]))
            if was_in and broker.btc == 0:
                round_trips += 1
            if broker.btc == 0:
                tranches = 0
            curve.append(broker.equity(float(c[p])))
            total += 1
            tip += int(broker.btc != 0)
            if p == int(bucket_last_pos[b]):
                side = 0 if broker.btc == 0 else (1 if broker.btc > 0
                                                  else -1)
                if side == 0:
                    tranches = 0
                intent = decide(side, tranches, int(regimes[j]))
                decisions.append((int(b), intent))
                action_counts[intent] += 1
                pending = (intent, float(atr_pct[j])) \
                    if intent != "hold" else None
    curve_arr = np.asarray(curve, dtype=float)
    pnl = float(curve_arr[-1] - 100_000.0)
    peak = np.maximum.accumulate(curve_arr)
    max_dd = float(np.max(peak - curve_arr)) if len(curve_arr) else 0.0
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT ts, side, price, qty, fee, reason FROM trades "
            "ORDER BY id").fetchall()
    fills = [{"ts": r[0], "side": r[1], "price": float(r[2]),
              "qty": float(r[3]), "fee": float(r[4]), "reason": r[5]}
             for r in rows]
    return {"pnl": pnl, "max_drawdown": max_dd,
            "score": pnl - 0.5 * max_dd,
            "trades": len(fills),
            "fees": float(sum(f["fee"] for f in fills)),
            "round_trips": round_trips,
            "time_in_position": float(tip / total) if total else 0.0,
            "decisions": decisions,
            "action_counts": action_counts,
            "fills": fills,
            "final_equity": float(curve_arr[-1])}


def resum_span_pnl(db_path, final_close: float,
                   start_equity: float = 100_000.0) -> float:
    """Independent span P&L from the ledger ALONE (the
    meta-label re-sum arithmetic, audited per fill)."""
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
