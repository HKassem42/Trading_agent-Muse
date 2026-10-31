"""Nightly retrain: the learning loop. Runs at 00:00 ET via cron.

1. Pull fresh BTC-USD data, rebuild features.
2. Train stacker challenger; walk-forward score.
3. Warm-start PPO from champion weights; train challenger.
4. Evaluate champion vs challenger PPO on a time holdout (deterministic
   episodes, mean +/-1 reward). Grade the winner -- but see the
   PROMOTION LOCK below: grading no longer installs.
5. Log the named run to the training registry.

PROMOTION LOCK (Kass directive, 2026-10-25): no challenger -- PPO or
stacker, nightly or weekly deep -- advances to champion or reaches
the live trading dashboard without Kass's explicit agreement. A
passing verdict is recorded as PENDING KASS APPROVAL (registry flag
+ lab note) and the champion files are NOT touched; Kass wants the
current champion tested further in live trading, undisturbed.
Installs happen only on a run with the lock lifted
(TAM_PROMOTION_LOCK=0), which is Kass's approval made executable.

Usage: python -m src.training.nightly
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.market_data import fetch_klines
from environment.trading_env import TradingEnv
from features.technical_indicators import FEATURE_COLUMNS, feature_matrix, make_supervised
from models import stacker
from trading.broker import PaperBroker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"


def registry_db() -> Path:
    p = DATA / "training.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                kind TEXT NOT NULL, metrics TEXT NOT NULL, promoted INTEGER NOT NULL)"""
        )
    return p


def log_run(kind: str, metrics: dict, promoted: bool) -> None:
    with sqlite3.connect(registry_db()) as con:
        con.execute(
            "INSERT INTO runs(ts,kind,metrics,promoted) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             kind, json.dumps(metrics), int(promoted)),
        )


def vote_histogram(votes) -> dict:
    """Compact summary of a stacker run's holdout up-probability
    votes: a 20-bin histogram over [0, 1] plus n / min / max /
    median and the share of votes inside the two coin-flip bands
    (0.45-0.55, 0.48-0.52) -- the exact shape the compact training
    feed's votes block serves (src/dashboards/compact.py,
    _votes_block). ~200B as JSON, versus the raw per-candle array
    the grader computes and used to discard (Kass-approved
    grading-time vote logging, 2026-10-31). Raises on unusable
    input; grading callers must go through stacker_run_metrics,
    which guards it -- a summary failure must never fail a grade."""
    v = np.asarray(votes, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("no votes to summarize")
    bins, _ = np.histogram(v, bins=20, range=(0.0, 1.0))
    return {
        "bins": [int(x) for x in bins],
        "n": int(v.size),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
        "med": round(float(np.median(v)), 4),
        "mid": round(100.0 * float(
            np.mean((v >= 0.45) & (v <= 0.55))), 1),
        "tight": round(100.0 * float(
            np.mean((v >= 0.48) & (v <= 0.52))), 1),
    }


def stacker_run_metrics(challenger_acc, champion_acc,
                        challenger_wf_acc, precision, folds,
                        pending_approval, reason, votes=None) -> dict:
    """The registry metrics dict for a graded stacker challenger --
    the exact fields the nightly duel has always recorded, plus,
    when the grading path computed the challenger's holdout votes,
    a "votes" summary from vote_histogram (instrumentation only,
    2026-10-31). ANY failure in the summary is logged and swallowed:
    the returned record is then the pre-2026-10-31 record verbatim,
    so vote logging can never change a grade, a verdict, or a
    promotion -- it only persists data the grader already had."""
    metrics = {"challenger_acc": challenger_acc,
               "champion_acc": champion_acc,
               "challenger_wf_acc": challenger_wf_acc,
               "precision": precision,
               "folds": folds,
               "pending_approval": pending_approval,
               "reason": reason}
    if votes is not None:
        try:
            metrics["votes"] = vote_histogram(votes)
        except Exception as e:
            print(f"WARNING: vote histogram persistence failed "
                  f"({type(e).__name__}: {e}); the run record "
                  f"carries no votes summary and grading is "
                  f"unaffected", flush=True)
    return metrics


def crossfit_p_up(X: pd.DataFrame, y: pd.Series, folds: int = 4) -> np.ndarray:
    """Out-of-fold stacker votes for the training frame.

    PPO trains on one of the 16 observation dims being the stacker's P(up).
    A stacker predicting on rows it was fit on produces systematically
    sharper votes than the out-of-sample votes the duel and the live loop
    produce (~0.50 coin flip) -- a train/serve distribution mismatch that
    teaches the brain to trust the stacker more than it should (audit
    2026-10-20). Here each vote comes from a pipeline fit ONLY on earlier
    data (time-series expanding windows, no future leakage). The leading
    segment (no past to fit on) gets the neutral 0.5.
    """
    from sklearn.model_selection import TimeSeriesSplit

    p_up = np.full(len(X), 0.5)
    tss = TimeSeriesSplit(n_splits=folds)
    for tr_idx, te_idx in tss.split(X):
        pipe = stacker.build_pipeline(random_state=42)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p_up[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:, 1]
    return p_up


def evaluate_ppo(model_path: Path, sup, p_up, episodes: int = 10,
                 feature_columns=None) -> dict:
    """Deterministic rollouts on ONE evaluation window: chronological,
    non-overlapping episodes (240 decision candles / 24h = 10 days) with
    NEXT-CANDLE-OPEN fills (v4) -- the duel grades carry the execution lag
    every real fill has instead of the optimistic same-close fill.

    Returns mean episode reward plus a risk-adjusted backtest score:
    score = total day-P&L - 0.5 * worst within-episode drawdown, where the
    drawdown is measured on each episode's own equity curve (audit
    2026-10-20: the old code drew the "drawdown" from the cumulative sum
    of per-episode P&Ls in seed order -- not chronological, not an equity
    curve -- so intra-episode drawdowns were invisible and the same market
    moves were double-counted across overlapping episodes).
    Promotion uses the score, not raw reward -- a challenger must make money
    without wild swings to take the crown. `bench_pnl` is the buy-and-hold
    result over the same window: the yardstick that can't be flattered.
    """
    import os

    from stable_baselines3 import PPO

    n = len(sup)
    assert n >= episodes * 24, f"holdout too short: {n} candles"
    starts = [k * 24 for k in range(episodes)]

    model = PPO.load(model_path)
    rewards, pnls, trips, ep_dds = [], [], [], []
    action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
    # Unique scratch DB per process (audit 2026-10-20): the fixed /tmp
    # path collided when a manual evaluation overlapped the nightly run.
    tmp = Path("/tmp") / f"tam_eval_{os.getpid()}.db"
    if tmp.exists():
        tmp.unlink()
    try:
        for ep, start in enumerate(starts):
            broker = PaperBroker(tmp, starting_balance=100_000,
                                 max_position_pct=0.25, stop_loss_pct=0.05)
            env = TradingEnv(sup, p_up, broker, fill_mode="next_open",
                             feature_columns=feature_columns)
            obs, _ = env.reset(seed=ep, options={"start": start})
            start_eq = env._day_start_equity
            curve = [start_eq]  # within-episode equity curve
            done, ep_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[["hold", "buy", "sell", "short"][int(action)]] += 1
                obs, r, done, _, _ = env.step(int(action))
                ep_r += r
                curve.append(env._prev_eq)
            rewards.append(ep_r)
            pnls.append(env._prev_eq - start_eq)
            trips.append(env._round_trips)
            c = np.array(curve)
            ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
    finally:
        tmp.unlink(missing_ok=True)
    total = float(np.sum(pnls))
    max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
    total_actions = sum(action_counts.values()) or 1
    closes = sup["close"].to_numpy(dtype=float)
    bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0)) if len(closes) > 1 else 0.0
    return {
        "mean_reward": float(np.mean(rewards)),
        "total_pnl": total,
        "max_drawdown": max_dd,
        "mean_drawdown": float(np.mean(ep_dds)) if ep_dds else 0.0,
        "score": total - 0.5 * max_dd,
        "bench_pnl": bench,
        "episodes": episodes,
        "round_trips_per_day": float(np.mean(trips)) if trips else 0.0,
        "action_mix": {k: round(v / total_actions, 3)
                       for k, v in action_counts.items()},
    }


# -- v4 helpers: champion rollback shelf + lab notebook ----------------

def archive_champions(tag: str) -> None:
    """Copy the current champions onto the rollback shelf before any are
    replaced (v4): if a midnight promotion ever turns out bad, the
    previous brain can be restored in one step instead of retraining our
    way out. Keeps the newest 5 snapshots."""
    import shutil

    shelf = MODELS / "archive"
    shelf.mkdir(parents=True, exist_ok=True)
    dest = shelf / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{tag}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ppo_champion.zip", "stacker_champion.pkl",
                 "stacker_champion.json"):
        src = MODELS / name
        if src.exists():
            shutil.copy2(src, dest / name)
    snaps = sorted(p for p in shelf.iterdir() if p.is_dir())
    for old in snaps[:-5]:
        shutil.rmtree(old, ignore_errors=True)


def add_lab_note(title: str, detail: str) -> None:
    """Append an entry to the lab notebook (data/lab_notes.json): the
    human-readable paper trail of what was tried and what happened,
    surfaced in the Training Lab."""
    path = DATA / "lab_notes.json"
    notes = []
    if path.exists():
        try:
            notes = json.loads(path.read_text())
        except Exception:
            notes = []
    notes.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "title": title, "detail": detail})
    # The notebook is the paper trail the dashboards' name/version
    # derivations read by exact timestamp -- it is NEVER truncated.
    # (The old notes[-30:] cap silently evicted the 2026-10-21
    # "Features v5.1 migration" note once the file filled, and run 16
    # lost its v5.1 attestation; display windowing lives in the feed
    # builders, not here.)
    path.write_text(json.dumps(notes))


def evaluate_windows(model_path: Path, windows, p_ups,
                     feature_columns=None) -> dict | None:
    """Grade one brain across all gate windows; None if it cannot run.
    `feature_columns` threads the brain's own feature set into the
    grading env (features-v6 study, 2026-10-25); None is the
    production set, unchanged."""
    per = []
    try:
        for w_sup, w_pup in zip(windows, p_ups):
            per.append(evaluate_ppo(model_path, w_sup, w_pup,
                                    feature_columns=feature_columns))
    except Exception as e:
        print(f"evaluation failed for {model_path} ({type(e).__name__}): {e}",
              flush=True)
        return None
    agg = float(np.mean([w["score"] for w in per]))
    return {"windows": per, "agg_score": agg,
            "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
            "mean_reward": float(np.mean([w["mean_reward"] for w in per]))}


def _agg(res) -> dict:
    """Flatten a 3-window evaluation into the run-record shape.

    Completeness rule (perfection pass, 2026-10-25): EVERYTHING
    evaluate_ppo computes must survive into the registry -- the old
    flattening silently dropped the per-window `episodes` count (so
    the Training Lab rendered "Episodes judged —" for every modern
    duel), the per-window drawdowns, mean_drawdown, and
    round_trips_per_day. `episodes` here is the TOTAL episodes judged
    across the duel (3 windows x episodes-per-window; the per-window
    count rides in each windows[] entry). The one deliberate
    exception is action_mix: only the LAST window's mix is recorded,
    by design -- the dashboard labels it exactly that ("last grading
    window"), and the floor reads the same recorded mix."""
    if res is None:
        return {"mean_reward": -999.0, "total_pnl": 0.0,
                "max_drawdown": 0.0, "mean_drawdown": 0.0,
                "bench_pnl": 0.0, "episodes": 0,
                "round_trips_per_day": 0.0,
                "score": -1e9, "runnable": False}
    ws = res["windows"]
    return {"score": res["agg_score"], "total_pnl": res["total_pnl"],
            "mean_reward": res["mean_reward"],
            "max_drawdown": float(max(w["max_drawdown"] for w in ws)),
            "mean_drawdown": float(np.mean([w["mean_drawdown"]
                                            for w in ws])),
            "bench_pnl": float(sum(w["bench_pnl"] for w in ws)),
            "episodes": int(sum(w["episodes"] for w in ws)),
            "round_trips_per_day": float(np.mean(
                [w["round_trips_per_day"] for w in ws])),
            "action_mix": ws[-1]["action_mix"],
            "windows": [{"score": round(w["score"], 2),
                         "pnl": round(w["total_pnl"], 2),
                         "bench": round(w["bench_pnl"], 2),
                         "dd": round(w["max_drawdown"], 2),
                         "mdd": round(w["mean_drawdown"], 2),
                         "mr": round(w["mean_reward"], 3),
                         "episodes": int(w["episodes"])} for w in ws],
            "runnable": True}


# -- lab champion feature sets (seat separation, 2026-10-25) ---------

def _champion_meta() -> dict:
    """The lab stacker champion's metadata sidecar, written at
    install time. Champions crowned from 2026-10-25 on carry a
    `feature_set` tag (and their feature columns); older champions
    predate the tag and are the production 21-feature recipe."""
    p = MODELS / "stacker_champion.json"
    if p.exists():
        try:
            meta = json.loads(p.read_text())
            return meta if isinstance(meta, dict) else {}
        except Exception:
            return {}
    return {}


def champion_holdout_frame(df: pd.DataFrame, hold_sup: pd.DataFrame,
                           meta: dict) -> tuple[pd.DataFrame, str]:
    """The feature frame the LAB stacker champion must be graded
    on, rows aligned to the holdout (`hold_sup`) timestamps.

    The Training Lab and the live driver's seat are separate
    (Kass, 2026-10-25): the lab holds the crowns, and a lab
    champion may be trained on a feature set the production path
    does not use -- Champion #2 (run 64) reads the matrix-v3
    scored frame. Grading it on the production 21 columns would
    grade a different brain than the one holding the crown, so
    the frame follows the champion metadata's feature_set tag:
      prod21    -> the production feature matrix (the default;
                   every champion before the separation).
      scored102 -> src/features/scored_features.py's frame,
                   restricted to the champion's recorded columns.
      pick2_98  -> src/features/pick2_features.py's frame (the
                   scored102 frame plus Pick #2's four causal
                   features, 98 columns), restricted to the
                   champion's recorded columns. Added 2026-10-27
                   when run 73 was crowned Lab Stacker Champion
                   #4 (Muse's ruling under Kass's delegation).
    An unknown tag raises -- a silent fallback would misgrade
    the champion and corrupt every verdict built on the grade."""
    tag = (meta or {}).get("feature_set", "prod21")
    if tag == "prod21":
        X, _ = feature_matrix(hold_sup)
        return X, tag
    if tag == "scored102":
        from features import scored_features
        frame, cols = scored_features.build_scored_frame(df)
        want = list((meta or {}).get("feature_columns") or cols)
        return frame.reindex(hold_sup.index)[want], tag
    if tag == "pick2_98":
        from features import pick2_features
        frame, cols, _ev = pick2_features.build_pick2_frame(df)
        want = list((meta or {}).get("feature_columns") or cols)
        return frame.reindex(hold_sup.index)[want], tag
    raise ValueError(f"unknown champion feature_set tag: {tag!r}")


# -- the two-part promotion floor (Kass decision 2026-10-22) ----------

def _usd(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def gate_decision(chall: dict, formula_pass: bool) -> tuple[bool, str | None]:
    """Two-part floor on top of the v4 relative gate formula.

    The formula alone is purely relative: against a champion that lost
    $4,485 on the gate windows, a challenger losing $175 formally
    "passes" (lab runs 21/24/26 all formula-passed while losing money,
    and run 20 formula-passed without placing a single trade). Kass's
    rule (decided 2026-10-22, option 1): a challenger is promotable
    only if the formula passes AND it (1) actually traded -- non-hold
    share >= 10% of its recorded action mix, (2) made money -- total
    duel P&L strictly > $0, and (3) beat doing nothing -- total duel
    P&L strictly > buy-and-hold over the same windows.

    `chall` is the flattened run-record dict (total_pnl / bench_pnl /
    action_mix). Returns (promotable, floor_failure): floor_failure is
    None when the floor passes, and also when the formula itself
    failed -- that case keeps the formula's own recorded reason.
    """
    if not formula_pass:
        return False, None
    mix = chall.get("action_mix") or {}
    # Recorded mixes are rounded to 3 decimals in evaluate_ppo; round
    # the complement the same way so hold=0.90 reads as exactly 10%
    # activity instead of the float artifact 1 - 0.9 = 0.09999...
    non_hold = round(1.0 - float(mix.get("hold", 1.0)), 3)
    if non_hold < 0.10:
        return False, f"floor: activity {non_hold:.1%} < 10%"
    pnl = float(chall.get("total_pnl", 0.0))
    if pnl <= 0.0:
        return False, f"floor: P&L {_usd(pnl)} ≤ $0"
    bench = float(chall.get("bench_pnl", 0.0))
    if pnl <= bench:
        return False, f"floor: P&L {_usd(pnl)} ≤ B&H {_usd(bench)}"
    return True, None


# -- the promotion lock (Kass directive 2026-10-25) -------------------

def promotion_locked() -> bool:
    """Kass's rule (2026-10-25): no challenger advances to champion
    without his explicit agreement, so the install step of every
    promotion verdict is held by default. A verdict that would
    promote is recorded as PENDING KASS APPROVAL instead and the
    champion files are not touched. The lock lifts only when
    TAM_PROMOTION_LOCK=0 is set for a run -- Kass's approval made
    executable, never a default."""
    return os.environ.get("TAM_PROMOTION_LOCK", "1") != "0"


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d")
    print(f"=== nightly retrain {run_id} ===", flush=True)

    # yfinance hiccups at midnight shouldn't kill the whole run: retry.
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h", period="730d")
            break
        except Exception as e:
            print(f"fetch attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
            import time
            time.sleep(30 * (attempt + 1))
    sup = make_supervised(df)
    X, y = feature_matrix(sup)

    # Time holdout: the last 33 days (792 hourly candles) are evaluation
    # ONLY, split into three chronological 11-day windows (264 candles:
    # 10 duel days + 1 buffer day for next-open fills). Nothing trains on
    # them tonight (audit 2026-10-20: the PPO challenger used to train on
    # the full fetched series -- including these exact candles -- minutes
    # before being graded on them). The 3-window gate (v4) means a
    # challenger can no longer take the crown off one lucky window.
    split = len(sup) - 792
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min(), "train/holdout overlap"
    windows_sup = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]

    # -- stacker challenger --
    # Promotion is judged on the SHARED holdout with a strict >, not by
    # comparing walk-forward accuracies computed on different windows at
    # different times (audit 2026-10-20: the old rule compared tonight's
    # challenger WF accuracy against the champion's stored WF accuracy
    # from its own older window, and promoted on ties).
    from sklearn.metrics import accuracy_score, precision_score

    Xtr, ytr = feature_matrix(train_sup)
    Xhold, yhold = feature_matrix(hold_sup)
    score = stacker.train_and_save(Xtr, ytr, MODELS / "stacker_challenger.pkl")
    challenger_model = joblib.load(MODELS / "stacker_challenger.pkl")
    chall_pred = challenger_model.predict(Xhold)
    chall_acc = float(accuracy_score(yhold, chall_pred))
    chall_prec = float(precision_score(yhold, chall_pred, zero_division=0))
    champ_pkl = MODELS / "stacker_champion.pkl"
    champ_score_path = MODELS / "stacker_champion.json"
    champ_meta = _champion_meta()
    champ_Xhold = None
    if champ_pkl.exists():
        champ_model = joblib.load(champ_pkl)
        # Dual-frame grading (seat separation, 2026-10-25): the lab
        # champion is scored on ITS OWN feature set -- the frame
        # follows its metadata tag (see champion_holdout_frame);
        # the challenger above stays on the production 21. A frame
        # failure degrades to the no-champion sentinel with a loud
        # print, the same contract as a missing champion file: the
        # anomaly lands in tonight's record instead of misgrading
        # the champion silently.
        try:
            champ_Xhold, _champ_tag = champion_holdout_frame(
                df, hold_sup, champ_meta)
            champ_acc = float(accuracy_score(
                yhold, champ_model.predict(champ_Xhold)))
        except Exception as e:
            print(f"WARNING: lab champion could not be graded on "
                  f"its feature set ({type(e).__name__}: {e}); "
                  f"treating it as no champion for tonight's duel",
                  flush=True)
            champ_Xhold, champ_acc = None, -1.0
    else:
        champ_acc = -1.0
    stacker_promoted = chall_acc > champ_acc
    stacker_reason = (
        f"stacker challenger holdout acc {chall_acc:.4f} > champion {champ_acc:.4f}"
        if stacker_promoted else
        f"stacker challenger holdout acc {chall_acc:.4f} did not beat champion {champ_acc:.4f}")
    # Rollback shelf (v4): snapshot the current champions BEFORE any
    # replacement tonight, so a bad promotion can be undone in one step.
    archive_champions(run_id)
    # Promotion lock (Kass, 2026-10-25): a winning challenger is held
    # as PENDING KASS APPROVAL -- the verdict is graded and recorded
    # exactly as before, but the champion files are NOT replaced and
    # the registry's promoted flag stays 0 until Kass agrees.
    locked = promotion_locked()
    stacker_pending = stacker_promoted and locked
    stacker_install = stacker_promoted and not locked
    if stacker_pending:
        stacker_reason += (
            " — WOULD PROMOTE: held as PENDING KASS APPROVAL under "
            "the promotion lock; champion kept")
    if stacker_install:
        (MODELS / "stacker_challenger.pkl").replace(champ_pkl)
        champ_score_path.write_text(json.dumps({
            "accuracy": chall_acc,
            "precision": chall_prec,
            "wf_accuracy": score["accuracy"],
            "wf_precision": score["precision"],
            "folds": score["folds"],
            "model_path": str(champ_pkl),
        }))
    # Instrumentation (2026-10-31): capture the challenger's holdout
    # up-probability votes so stacker_run_metrics can persist their
    # summary in the run record. predict_proba feeds NOTHING else --
    # chall_acc / chall_prec above come from predict exactly as
    # before, and the verdict/promotion logic never sees the votes.
    # Any failure here leaves chall_votes=None and the grade stands.
    try:
        chall_votes = challenger_model.predict_proba(Xhold)[:, 1]
    except Exception as e:
        print(f"WARNING: challenger vote capture failed "
              f"({type(e).__name__}: {e}); grading unaffected",
              flush=True)
        chall_votes = None
    log_run("stacker", stacker_run_metrics(
        chall_acc, champ_acc, score["accuracy"], chall_prec,
        score["folds"], stacker_pending, stacker_reason,
        votes=chall_votes), stacker_install)
    add_lab_note("Stacker duel — PENDING KASS APPROVAL"
                 if stacker_pending else "Stacker duel", stacker_reason)
    print(f"stacker: challenger holdout acc={chall_acc:.4f} (wf {score['accuracy']:.4f}) "
          f"champion holdout acc={champ_acc:.4f} promoted={stacker_install} "
          f"pending={stacker_pending}", flush=True)

    # -- PPO challenger (warm-started) --
    # Trains ONLY on train_sup, with cross-fitted (out-of-fold) stacker
    # votes -- never on the windows it is about to be graded on.
    import os

    from models.ppo_train import train as train_ppo
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    # The duel's P(up) votes come from the lab champion stacker,
    # read on its own feature frame (dual-frame, above): for a
    # prod21 champion champ_Xhold IS the production matrix; for a
    # scored102 champion it is the scored frame over the holdout.
    # If that frame failed to build, the stacker duel above has
    # already recorded the anomaly -- grade tonight's PPO duel on
    # neutral 0.5 votes rather than feeding the brain a frame it
    # cannot read (or crashing the whole run).
    if champ_Xhold is not None:
        p_up_hold_all = stacker_model.predict_proba(champ_Xhold)[:, 1]
    elif champ_meta.get("feature_set", "prod21") != "prod21":
        print("WARNING: lab champion's feature frame unavailable; "
              "grading tonight's PPO duel on neutral 0.5 stacker "
              "votes", flush=True)
        p_up_hold_all = np.full(len(hold_sup), 0.5)
    else:
        p_up_hold_all = stacker_model.predict_proba(
            hold_sup[FEATURE_COLUMNS])[:, 1]
    p_up_windows = [p_up_hold_all[k * 264:(k + 1) * 264] for k in range(3)]
    p_up_train = crossfit_p_up(Xtr, ytr)
    # Weekly deep train (v4): TAM_DEEP=1 -> 300k steps instead of 60k.
    deep = bool(os.environ.get("TAM_DEEP"))
    timesteps = 300_000 if deep else 60_000
    challenger_path = train_ppo(
        timesteps=timesteps, warm_start=True,
        champion_path=MODELS / "ppo_champion.zip",
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=DATA / "train_tmp.db",  # throwaway ledger; live paper.db untouched
        ent_coef=0.05,  # keep exploring; sparse +/-1 rewards need it
        sup=train_sup, p_up=p_up_train,
    )

    if (MODELS / "ppo_champion.zip").exists():
        champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                     windows_sup, p_up_windows)
        chall_res = evaluate_windows(challenger_path, windows_sup, p_up_windows)
        champ, chall = _agg(champ_res), _agg(chall_res)
        wins = 0
        if champ_res and chall_res:
            wins = sum(1 for cw, hw in zip(champ_res["windows"],
                                           chall_res["windows"])
                       if hw["score"] > cw["score"])
        # The gate (v4): aggregate score strictly better AND a majority of
        # windows won. One hot window can no longer crown a brain.
        formula_pass = chall_res is not None and (
            champ_res is None or (chall["score"] > champ["score"] and wins >= 2))
        # Two-part floor (Kass, 2026-10-22): the relative formula alone
        # crowns "less bad than a losing champion" -- the challenger
        # must also have traded, made money, and beaten buy-and-hold.
        # ppo_verdict is the WOULD-PROMOTE decision; whether it installs
        # is settled by the promotion lock further below.
        ppo_verdict, floor_fail = gate_decision(chall, formula_pass)
        print(f"ppo gate: challenger agg={chall['score']:.2f} "
              f"(pnl={chall['total_pnl']:.2f}) champion agg={champ['score']:.2f} "
              f"(pnl={champ['total_pnl']:.2f}) windows won={wins}/3 "
              f"deep={deep} verdict_promote={ppo_verdict}", flush=True)
        if chall_res is None:
            ppo_reason = "challenger failed evaluation; champion kept"
        elif floor_fail is not None:
            rel = (f"aggregate {chall['score']:.0f} vs champion "
                   f"{champ['score']:.0f}, won {wins}/3 windows"
                   if champ_res is not None else
                   "champion could not run in the current setup")
            ppo_reason = (f"challenger passed the relative gate ({rel}) "
                          f"but was not promoted — {floor_fail}; "
                          f"champion kept")
        elif champ_res is None:
            # e.g. champion trained on an old feature recipe can't even run
            # in the current env; a runnable challenger that also clears
            # the floor takes the crown (manual recovery on 2026-10-20;
            # now the standard path).
            ppo_reason = ("champion could not run in the current setup, "
                          "so the challenger takes the crown by default "
                          "(gate pass incl. two-part floor)")
        elif ppo_verdict:
            ppo_reason = (f"challenger won {wins}/3 windows, aggregate score "
                          f"{chall['score']:.0f} vs champion {champ['score']:.0f} "
                          f"(next-candle fills; buy-and-hold over the same "
                          f"windows: ${chall['bench_pnl']:+,.0f}) "
                          f"(gate pass incl. two-part floor)")
        else:
            ppo_reason = (f"champion kept: challenger won {wins}/3 windows, "
                          f"aggregate {chall['score']:.0f} vs champion "
                          f"{champ['score']:.0f}")
    else:
        chall = _agg(evaluate_windows(challenger_path, windows_sup, p_up_windows))
        champ = {"mean_reward": -1.0, "total_pnl": 0.0,
                 "max_drawdown": 0.0, "bench_pnl": 0.0, "episodes": 0,
                 "score": -1.0}
        ppo_verdict = True
        ppo_reason = "no champion existed; first challenger would promote"
        print("ppo: no champion yet; first challenger verdict = promote",
              flush=True)
    # Promotion lock (Kass, 2026-10-25): the verdict installs ONLY with
    # the lock lifted. Locked, a passing challenger is recorded as
    # PENDING KASS APPROVAL and the champion file is not touched --
    # this also closes the no-champion bootstrap path, which is a
    # promotion like any other under Kass's rule.
    ppo_pending = ppo_verdict and locked
    ppo_promoted = ppo_verdict and not locked
    if ppo_pending:
        ppo_reason += (" — WOULD PROMOTE: held as PENDING KASS APPROVAL "
                       "under the promotion lock; champion kept")
        print("ppo: WOULD PROMOTE — held pending Kass approval "
              "(promotion lock)", flush=True)
    if ppo_promoted:
        challenger_path.replace(MODELS / "ppo_champion.zip")
    reward_curve = None
    telemetry = None
    curve_path = MODELS / "ppo_challenger_curve.json"
    if curve_path.exists():
        try:
            sidecar = json.loads(curve_path.read_text())
            reward_curve = sidecar.get("curve")
            # SB3 training telemetry (2026-10-26, logging-only
            # capture in ppo_train): entropy / KL / clip / explained
            # variance / value loss, lifted verbatim into the run
            # record for the Training Lab telemetry panel. Runs
            # graded before the capture shipped carry None -- the
            # panel says "not recorded", never a reconstruction.
            telemetry = sidecar.get("telemetry")
        except Exception:
            reward_curve = None
            telemetry = None
    log_run("ppo", {"challenger": chall, "champion": champ,
                    "reason": ppo_reason, "deep": deep,
                    "timesteps": timesteps,
                    "pending_approval": ppo_pending,
                    "reward_curve": reward_curve,
                    "telemetry": telemetry}, ppo_promoted)
    add_lab_note("PPO promotion gate — PENDING KASS APPROVAL"
                 if ppo_pending else "PPO promotion gate", ppo_reason)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
