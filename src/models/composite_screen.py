"""Matrix v3 composite screen (2026-10-24, Kass green-lit twice).

Executes refs/composite-screen-plan-v3-2026-10-23.md EXACTLY as
pre-registered. The plan is the contract; this module is machinery
only. Features are the four v3 composite columns and nothing else:

  score_master, score_trend, score_momentum, score_volume

Arms:
  rule_master    fixed 60/40 rule on score_master (>= 60 BUY,
                 <= 40 SHORT, else HOLD) -- graded walk-forward
                 AND in the duel.
  rule_trend / rule_momentum / rule_volume
                 the same 60/40 rule solo on one family score --
                 walk-forward ONLY (diagnostics; never duel-graded,
                 so a lucky family cannot be cherry-picked).
  gbm4           LightGBM on the 4 composites, class-weighted --
                 literally models.gbm_focus's arm-B treatment (same
                 LGBM_PARAMS, same inverse-frequency weights), the
                 style that traded most honestly in the focus test.

Walk-forward: TimeSeriesSplit(5) over the pre-duel rows only
(duel region starts 2026-09-19 11:00 UTC), identical fold indices
for every arm, so all arms are graded on identical rows.

Duel: arms rule_master and gbm4, graded on the standard 3x264
windows AND the older robustness window set (the preceding 792h,
the exit/size studies' set), through a mirror of
training.nightly.evaluate_ppo's loop -- same StudyEnv/StudyBroker
with the default ExitConfig (the verified production exit recipe:
through-flat transitions, 2xATR stop with the 2.5% floor, trailing
as designed, no take-profit), next-candle-open fills, 25% tranches,
75% cap, 0.15% fees, scratch ledger DBs in /tmp. The mirror adds
what the plan demands and nightly's scorer doesn't keep: EXACT
action counts (the two-part floor's activity test), total fees
actually paid, and pre-fee (gross) P&L = net P&L + fees -- an exact
identity in this broker (every fill's cash delta is
+/-(price x qty) - fee), pinned by tests/test_composite_screen.py
against the real broker. The build script cross-checks the mirror
against nightly.evaluate_windows on the champion (must agree).

Signals ride the env's vote channel as signed conviction encoded
p_up = (c + 1) / 2, exactly like runs 21-27: GBM-4 through
models.gbm_policy.conviction_np at the pre-registered 0.40 bars;
Rule-Master as c in {+1, 0, -1} from its own 60/40 rule. Both are
driven by the same saved rule-policy zip (models.gbm_policy), so
the position-aware action mapping (flatten when the signal is gone
or flipped) is identical across arms.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models import gbm_focus as gf
from models.exit_study import (EPISODES, STARTING_BALANCE, ExitConfig,
                               StudyBroker, StudyEnv, _read_new_fills,
                               segment_trips)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

TARGET_COL = gf.TARGET_COL
CLASSES = gf.CLASSES
COMPOSITE_COLS = ["score_master", "score_trend",
                  "score_momentum", "score_volume"]
RULE_HIGH = 60.0
RULE_LOW = 40.0

ARMS_WF = ("rule_master", "rule_trend", "rule_momentum",
           "rule_volume", "gbm4")
ARMS_DUEL = ("rule_master", "gbm4")
ARM_LABELS = {
    "rule_master": "Rule Master",
    "rule_trend": "Rule Trend",
    "rule_momentum": "Rule Momentum",
    "rule_volume": "Rule Volume",
    "gbm4": "GBM-4",
}
ARM_DESIGNS = {
    "rule_master": "fixed 60/40 rule on score_master "
                   "(>=60 BUY, <=40 SHORT, else HOLD), no learning",
    "rule_trend": "fixed 60/40 rule solo on score_trend (WF diagnostic)",
    "rule_momentum": "fixed 60/40 rule solo on score_momentum "
                     "(WF diagnostic)",
    "rule_volume": "fixed 60/40 rule solo on score_volume "
                   "(WF diagnostic)",
    "gbm4": "LightGBM on the 4 composites, class-weighted "
            "(gbm_focus arm-B treatment, LGBM_PARAMS unchanged)",
}
RULE_ARM_COLUMN = {
    "rule_master": "score_master",
    "rule_trend": "score_trend",
    "rule_momentum": "score_momentum",
    "rule_volume": "score_volume",
}

__all__ = ["COMPOSITE_COLS", "RULE_HIGH", "RULE_LOW", "ARMS_WF",
           "ARMS_DUEL", "ARM_LABELS", "ARM_DESIGNS", "RULE_ARM_COLUMN",
           "load_v3", "rule_labels", "rule_conviction",
           "walk_forward_arm", "evaluate_windows_composite",
           "conviction_by_ts", "p_ups_from_conviction",
           "apply_verdict"]


# ------------------------------------------------------------------ data
def load_v3(path) -> pd.DataFrame:
    """Load the v3 matrix CSV with Date as a sorted DatetimeIndex
    (same loader discipline as models.gbm_focus.load_matrix)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    return df


# ------------------------------------------------------------------ rules
def rule_labels(scores) -> np.ndarray:
    """The pre-registered 60/40 rule: >= 60 -> BUY, <= 40 -> SHORT,
    otherwise HOLD. Boundaries are INCLUSIVE on both sides (60.0 is
    BUY, 40.0 is SHORT), exactly as the plan states them."""
    s = np.asarray(scores, dtype=float)
    out = np.full(s.shape, "HOLD", dtype=object)
    out[s >= RULE_HIGH] = "BUY"
    out[s <= RULE_LOW] = "SHORT"
    return out


def rule_conviction(scores) -> np.ndarray:
    """The rule's decision as signed conviction for the duel vote
    channel: +1 BUY, -1 SHORT, 0 HOLD. The duel's rule policy fires
    at |c| >= 0.40, so {+1, 0, -1} reproduces the rule exactly, with
    the wrapper's position-aware flattening in the 0 band."""
    lab = rule_labels(scores)
    c = np.zeros(lab.shape, dtype=float)
    c[lab == "BUY"] = 1.0
    c[lab == "SHORT"] = -1.0
    return c


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
    """TimeSeriesSplit(5) walk-forward for one arm over the pre-duel
    frame. Rule arms predict directly (no fit -- the fold split only
    partitions the graded rows, identically for every arm); gbm4 fits
    on each fold's train rows ONLY via gbm_focus's arm-B fold helper
    (audit included). Returns per-fold metrics + pooled OOF metrics
    over the graded rows (the union of the 5 validation blocks)."""
    from sklearn.model_selection import TimeSeriesSplit

    y = df_pre[TARGET_COL].to_numpy()
    ts = df_pre.index
    tss = TimeSeriesSplit(n_splits=5)
    folds = []
    oof_pred = np.full(len(df_pre), None, dtype=object)
    audits = []
    if arm == "gbm4":
        X = df_pre[COMPOSITE_COLS].to_numpy(dtype=float)
        for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
            model, audit = gf.fit_arm_fold(X, y, ts, tr_idx, va_idx, "B")
            pred = model.predict(X[va_idx])
            oof_pred[va_idx] = pred
            audits.append(audit)
            m = _fold_metrics(y[va_idx], pred)
            m["fold"] = k
            folds.append(m)
    elif arm in RULE_ARM_COLUMN:
        scores = df_pre[RULE_ARM_COLUMN[arm]].to_numpy(dtype=float)
        pred_all = rule_labels(scores)
        for k, (tr_idx, va_idx) in enumerate(tss.split(scores)):
            pred = pred_all[va_idx]
            oof_pred[va_idx] = pred
            m = _fold_metrics(y[va_idx], pred)
            m["fold"] = k
            folds.append(m)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    graded = oof_pred != None  # noqa: E711 -- object array sentinel
    pooled = _fold_metrics(y[graded], oof_pred[graded])
    majority = float(max((y[graded] == c).mean() for c in CLASSES))
    pooled["majority_baseline"] = majority
    pooled["n_graded"] = int(graded.sum())
    return {"arm": arm, "folds": folds, "pooled": pooled,
            "fold_audits": audits}


# ------------------------------------------------------------------ duel
def conviction_by_ts(ts_index, values) -> dict:
    return {pd.Timestamp(t): float(v) for t, v in zip(ts_index, values)}


def p_ups_from_conviction(windows, conv_by_ts: dict):
    """Per-window p_up arrays from a ts->conviction map; timestamps
    with no matrix row (the supervised frame runs a few rows past
    the matrix end) default to 0 conviction (flat band), the same
    treatment runs 21-27 used. Returns (p_ups, n_missing)."""
    p_ups, missing = [], 0
    for w in windows:
        conv = np.zeros(len(w))
        for i, ts in enumerate(w.index):
            c = conv_by_ts.get(ts)
            if c is None:
                missing += 1
            else:
                conv[i] = c
        p_ups.append((conv + 1.0) / 2.0)
    return p_ups, missing


def evaluate_windows_composite(windows, p_ups, model) -> dict:
    """The standard duel, mirrored from nightly.evaluate_ppo /
    size_study.evaluate_windows_size with the default (production)
    ExitConfig: 10 episodes of 24 starts per window, next-candle-open
    fills, window score = total P&L - 0.5 x worst episode drawdown,
    aggregate = mean of window scores. Adds EXACT action counts,
    ledger-segmented trips, total fees and gross (pre-fee) P&L."""
    per = []
    all_closed: list[dict] = []
    total_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    fees_total = 0.0
    for w_i, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        tmp = Path("/tmp") / f"tam_comp_{os.getpid()}_{w_i}.db"
        if tmp.exists():
            tmp.unlink()
        try:
            broker = StudyBroker(tmp, starting_balance=STARTING_BALANCE,
                                 max_position_pct=0.25, stop_loss_pct=0.05,
                                 cfg=ExitConfig())
            env = StudyEnv(w_sup, w_pup, broker, fill_mode="next_open")
            rewards, pnls, trips_env, ep_dds = [], [], [], []
            counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
            offset = 0
            pnl_long = pnl_short = 0.0
            open_left = 0
            w_fees = 0.0
            for ep, start in enumerate([k * 24 for k in range(EPISODES)]):
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
def apply_verdict(wf: dict, duel: dict, robust: dict,
                  champ: dict) -> dict:
    """The plan's three pre-registered verdict rules, applied
    mechanically. Rule 2's floor is evaluated by the PRODUCTION
    gate_decision (src/training/nightly.py) so the screen can never
    drift from the gate it reports to."""
    from training.nightly import gate_decision

    pooled = wf["pooled"]
    fold_bals = [f["balanced_accuracy"] for f in wf["folds"]]
    rule1 = bool(
        pooled["balanced_accuracy"] > 1.0 / 3.0
        and sum(1 for b in fold_bals if b > 1.0 / 3.0) >= 4
        and pooled["accuracy"] >= pooled["majority_baseline"])
    wins = sum(1 for cw, aw in zip(champ["windows"], duel["windows"])
               if aw["score"] > cw["score"])
    formula_pass = bool(duel["agg_score"] > champ["agg_score"]
                        and wins >= 2)
    flat = {"total_pnl": duel["total_pnl"],
            "bench_pnl": duel["bench_pnl"],
            "action_mix": duel["action_mix"]}
    _promotable, floor_fail = gate_decision(flat, formula_pass)
    rule2 = bool(formula_pass and floor_fail is None)
    rule3 = bool(robust["total_pnl"] >= 0.0)
    passed = rule1 and rule2 and rule3
    return {
        "verdict": "PASS" if passed else "FAIL",
        "rule1_walk_forward": rule1,
        "rule2_duel_gate": rule2,
        "rule3_robustness": rule3,
        "windows_won": wins,
        "formula_pass": formula_pass,
        "floor_fail": floor_fail,
        "fold_balanced_accuracies": fold_bals,
    }
