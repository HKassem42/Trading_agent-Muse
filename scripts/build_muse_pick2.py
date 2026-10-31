"""Muse's Pick #2 challenger LAB study (2026-10-27, Kass's order).

Kass: "You go ahead and try to beat your own champion first muse,
take another stab at it and call it Muse model #2 or whatever u
called the first one but #2." The champion is Muse's own Pick #1
stacker (run 69), crowned Lab Stacker Champion #3 by Kass's
approval on 2026-10-26. Pre-registered design (frozen in
refs/muse-pick-2-challenger-2026-10-27.md BEFORE grading):

1. Muse's Pick #2 Stacker -- Pick #1's recipe (scaler -> SMOTE ->
   RF + XGBoost + LightGBM + ExtraTrees -> LR, imported from
   models.muse_pick) on Champion #3's scored-102 frame PLUS four
   new causal features from the archive-mining candidates:
   rolling Hurst exponent (100 bars), rolling return skew /
   kurtosis / lag-1 autocorrelation (100 bars), each percentile-
   scored through the project's trailing-4,320-bar machinery
   (src/features/pick2_features.py). 98 columns. Graded vs
   Champion #3 (run 69) re-scored on identical holdout timestamps
   (two independent paths, must agree exactly): strict accuracy
   win, precision, up-calls, walk-forward. A strict win with
   <= 7 up-calls or precision <= 0.05 is DEGENERATE and counts
   as a LOSS (the run 63 rule, pre-registered).
2. Muse's Pick #2 PPO -- Pick #1's reward shaping (-0.5 x step
   drawdown increase, training only) gets its first FAIR test:
   Pick #1's confound was 24-candle training episodes vs the
   duel's 264-candle window. Pick #2 trains fresh (60,000 steps,
   seed 42, ent_coef 0.05, scored-eyes protocol, telemetry on)
   on DUEL-LENGTH episodes (264 candles,
   DuelLengthShapedTradingEnv) consuming the Pick #2 stacker's
   cross-fitted votes; state 102 wide (98 scored + vote + 3
   portfolio). The duel: challenger pair vs champion pair
   (run 44 PPO on prod21 + run 69 votes), 3 windows x 10
   episodes, next-candle-open fills, gate formula + two-part
   floor via gate_decision().

LAB ONLY under the promotion lock: nothing installs, nothing
touches either seat or either book; a win records
pending_approval for Kass. Champion/seat/shadow files are
hash-guarded before/after (the paper books are read-hash info
only -- the live loops write them on candle closes).

Usage: .venv/bin/python scripts/build_muse_pick2.py
Phases checkpoint to data/muse_pick2/; reruns skip done phases.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.market_data import fetch_klines  # noqa: E402
from features import pick2_features, scored_eyes  # noqa: E402
from models import muse_pick, muse_pick2  # noqa: E402
from models.muse_pick import (  # noqa: E402
    crossfit_p_up_pick, train_pick_and_save)
from models.muse_pick2 import DuelLengthShapedTradingEnv  # noqa: E402
from models.ppo_train import train as train_ppo  # noqa: E402
import models.ppo_train as ppo_train_mod  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import (  # noqa: E402
    _agg, add_lab_note, evaluate_windows, gate_decision, log_run)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "muse_pick2"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
ENV_SUP_PATH = LAB / "env_sup.pkl"
PROD_SUP_PATH = LAB / "prod_sup.pkl"
SCORED_RAW_PATH = LAB / "scored_raw.pkl"
STACKER_PATH = LAB / "stacker_pick2.pkl"
VOTES_PATH = LAB / "votes.npz"
ARM = LAB / "arm60"

HOLDOUT = 792
WINDOW = 264
STEPS = 60_000
COLS94 = None  # set in main from the champion's recorded columns
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_stacker_meta": MODELS / "live_seat" / "stacker.json",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
    "shadow_stacker": MODELS / "shadow_seat" / "stacker.pkl",
    "shadow_stacker_meta": MODELS / "shadow_seat" / "stacker.json",
    "shadow_ppo": MODELS / "shadow_seat" / "ppo.zip",
}
SAMPLE_COLS = ["score_master", "Relative Strength Index (RSI)",
               "Rolling Hurst Exponent (100)"]
DESIGN_STACKER = (
    "Muse's Pick #2 stacker: Pick #1 recipe (RF+XGB+LGBM+ET -> "
    "LR), scored-102 + Hurst/skew/kurtosis/autocorr (98 feats)")
DESIGN_PPO = (
    "Muse's Pick #2 PPO: SB3 PPO, 102-wide scored state + Pick "
    "#2 Stacker votes, reward shaping -0.5 x step DD increase, "
    "264-candle training episodes")


# ---------------------------------------------------------------- state
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    _save_json(PROGRESS_PATH, {"done": sorted(progress() | {phase})})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard() -> dict:
    return {k: _sha(p) for k, p in GUARD_FILES.items()} | {
        "paper_db_info_only": _sha(ROOT / "data" / "paper.db"),
        "paper_shadow_db_info_only": _sha(ROOT / "data" / "paper_shadow.db")}


def _assert_guard(before: dict, after: dict) -> None:
    for k in GUARD_FILES:
        assert after[k] == before[k], (
            f"GUARD TRIPPED on {k}: {before[k]} -> {after[k]}")


def cols98() -> list[str]:
    return list(scored_eyes.champion_columns()) + list(
        pick2_features.NEW_COLUMNS)


def get_df() -> pd.DataFrame:
    if DF_PATH.exists():
        return pd.read_pickle(DF_PATH)
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(DF_PATH)
    return df


def get_frames():
    """(df, prod_sup, split, holdout_start, hold_start_pos)."""
    df = get_df()
    from features.technical_indicators import make_supervised
    prod_sup = make_supervised(df)
    split = len(prod_sup) - HOLDOUT
    holdout_start = prod_sup.index[split]
    hold_start_pos = int(df.index.get_indexer([holdout_start])[0])
    assert df.index[hold_start_pos] == holdout_start
    assert prod_sup.index[:split].max() < holdout_start
    return df, prod_sup, split, holdout_start, hold_start_pos


def _train_env_sup(env_sup, holdout_start):
    cols = cols98()
    train = env_sup[env_sup.index < holdout_start].dropna(subset=cols)
    assert len(train) > 5_000, f"pick2 train frame too small: {len(train)}"
    assert train.index.max() < holdout_start
    return train


# ---------------------------------------------------------------- phases
def phase_data() -> None:
    df, prod_sup, split, holdout_start, hold_start_pos = get_frames()
    save_result("meta", {
        "rows_raw": len(df), "rows_sup": len(prod_sup),
        "split": split, "holdout_start": str(holdout_start),
        "hold_start_pos": hold_start_pos,
        "data_first": str(df.index[0]), "data_last": str(df.index[-1]),
        "guard_before": _guard(),
        "state_size": len(cols98()) + 1 + 3,
        "n_features": len(cols98()),
        "champion_run": scored_eyes.champion_meta().get("run_id"),
        "design": {"stacker": DESIGN_STACKER, "ppo": DESIGN_PPO,
                   "episode_candles": muse_pick2.EPISODE_CANDLES,
                   "shaping_coef": muse_pick.SHAPING_COEF,
                   "new_features": pick2_features.NEW_COLUMNS,
                   "windows": {"hurst": pick2_features.HURST_WINDOW,
                               "moments": pick2_features.MOMENT_WINDOW,
                               "autocorr":
                                   pick2_features.AUTOCORR_WINDOW}},
    })
    print(f"data: {len(df)} raw rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(prod_sup)}, holdout from {holdout_start}", flush=True)


def phase_frames() -> None:
    """Pick #2 frame (98 cols) + causality spot-checks, Pick #1's
    pattern extended to the new columns."""
    df, prod_sup, split, holdout_start, hold_start_pos = get_frames()
    cols = cols98()
    cols94 = scored_eyes.champion_columns()
    print("building pick2 scored frame (94 champion + 4 new) ...",
          flush=True)
    t0 = time.time()
    scored_raw, built, evidence = pick2_features.build_pick2_frame(df)
    assert built == cols, "pick2 frame columns differ from the design"
    print(f"  pick2 frame built in {time.time() - t0:.0f}s; "
          f"evidence={evidence}", flush=True)
    env_sup = scored_eyes.assemble_env_sup(prod_sup, scored_raw, cols)

    # The scored span's percentile windows must be clean: no raw
    # NaN inside any window the scorer reads (a mid-span NaN would
    # silently corrupt later scores -- stop the arm if so).
    raw = pick2_features.raw_new_features(df)
    p0 = evidence["pick2_p0"]
    from features import scored_features as _sf
    win_start = p0 - _sf.WINDOW + 1
    for c in pick2_features.RAW_COLUMNS:
        seg = raw[c].to_numpy(float)[win_start:]
        assert np.isfinite(seg).all(), (
            f"raw series {c} has NaN inside the scored span -- "
            f"STOP: percentile windows would be corrupted")

    hold_env = env_sup.iloc[split:]
    hold_prod = prod_sup.iloc[split:]
    assert list(hold_env.index) == list(hold_prod.index), (
        "env holdout timestamps differ from production holdout")
    assert np.isfinite(hold_env[cols].to_numpy(float)).all(), (
        "non-finite pick2 feature inside the holdout")
    assert (hold_env["close"].to_numpy(float)
            == hold_prod["close"].to_numpy(float)).all()
    assert (hold_env["target"].to_numpy()
            == hold_prod["target"].to_numpy()).all()
    train = _train_env_sup(env_sup, holdout_start)
    vals = train[cols].to_numpy(float)
    assert np.isfinite(vals).all()
    assert vals.min() >= -1.0 and vals.max() <= 1.0

    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    cuts = [9000, hold_start_pos + 300]
    trunc_evidence = {}
    for K in cuts:
        frame_t, _, _ = pick2_features.build_pick2_frame(df.iloc[:K + 1])
        per = {}
        for c in SAMPLE_COLS:
            a = scored_raw[c].iloc[:K + 1].to_numpy(float)
            b = frame_t[c].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            per[c] = {"max_abs_diff": d, "shared_rows": int(both.sum())}
            assert d <= 1e-9, f"causality FAIL (pick2, K={K}): {c} {d}"
        pos_of = df.index.get_indexer(prod_sup.index)
        cand = [t for t, p in zip(prod_sup.index, pos_of)
                if p <= K and np.isfinite(
                    scored_raw.loc[t, SAMPLE_COLS[1]])]
        rows = cand[-200:]
        assert len(rows) >= 50, f"too few shared rows at K={K}"
        v_full = champ_model.predict_proba(
            scored_raw.reindex(rows)[cols94])[:, 1]
        v_trunc = champ_model.predict_proba(
            frame_t.reindex(rows)[cols94])[:, 1]
        vd = float(np.abs(v_full - v_trunc).max())
        per["__votes__"] = {"max_abs_diff": vd, "rows": len(rows)}
        assert vd <= 1e-12, f"vote alignment FAIL (K={K}): {vd}"
        trunc_evidence[str(K)] = per
        print(f"causality truncation K={K}: PASS (scores exact on "
              f"{len(SAMPLE_COLS)} sampled columns incl. Hurst; "
              f"run-69 votes identical on {len(rows)} shared rows)",
              flush=True)

    env_sup.to_pickle(ENV_SUP_PATH)
    prod_sup.to_pickle(PROD_SUP_PATH)
    scored_raw.to_pickle(SCORED_RAW_PATH)
    save_result("frames", {
        "train_rows": len(train),
        "dropped_warmup_train_rows": int(split - len(train)),
        "holdout_rows": len(hold_env),
        "frame_evidence": evidence,
        "causality_truncation": trunc_evidence,
        "train_first": str(train.index[0]),
        "train_last": str(train.index[-1]),
    })
    print(f"frames: train {len(train)} rows (warmup-dropped "
          f"{split - len(train)}), holdout {len(hold_env)} finite + "
          f"label/close-identical to production", flush=True)


def phase_stacker() -> None:
    """Train + grade Muse's Pick #2 Stacker against Champion #3
    (run 69) re-scored on the identical holdout timestamps -- the
    champion grade computed TWO WAYS that must agree exactly."""
    from sklearn.metrics import accuracy_score, precision_score

    cols = cols98()
    cols94 = scored_eyes.champion_columns()
    df, prod_sup, split, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    scored_raw = pd.read_pickle(SCORED_RAW_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    Xtr = scored_raw.reindex(train.index)[cols]
    ytr = train["target"].astype(int)
    assert np.isfinite(Xtr.to_numpy(float)).all()
    hold_idx = prod_sup.index[split:]
    Xhold = scored_raw.reindex(hold_idx)[cols]
    yhold = prod_sup.iloc[split:]["target"].astype(int)
    assert np.isfinite(Xhold.to_numpy(float)).all()

    t0 = time.time()
    score = train_pick_and_save(Xtr, ytr, STACKER_PATH)
    print(f"pick2 stacker trained in {time.time() - t0:.0f}s "
          f"(wf acc={score['accuracy']:.4f})", flush=True)
    model = joblib.load(STACKER_PATH)
    proba = model.predict_proba(Xhold)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yhold, pred))
    prec = float(precision_score(yhold, pred, zero_division=0))
    up_calls = int(pred.sum())

    # Champion #3, way A: the installed artifact on this study's
    # own scored frame (its recorded 94 columns).
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    proba_c = champ_model.predict_proba(
        scored_raw.reindex(hold_idx)[cols94])[:, 1]
    pred_c = (proba_c >= 0.5).astype(int)
    acc_c = float(accuracy_score(yhold, pred_c))
    prec_c = float(precision_score(yhold, pred_c, zero_division=0))
    up_c = int(pred_c.sum())
    # Champion #3, way B: nightly's champion-frame path on the
    # same raw data (the exact path the midnight retrain grades
    # with). The two INPUT frames must be bit-identical and the
    # two GRADES (labels, accuracy, precision, up-calls) exactly
    # equal. Probabilities themselves may wobble by 1 ulp between
    # calls -- the tree ensembles sum votes in parallel
    # (n_jobs=-1), the same 1-ulp flake the crowning report
    # recorded in test_shadow_desk -- so the probability check is
    # a 1e-12 bound, recorded, not a bit-equality demand.
    meta = scored_eyes.champion_meta()
    X_b, tag = nightly.champion_holdout_frame(
        df, prod_sup.iloc[split:], meta)
    assert tag == "scored102"
    X_a = scored_raw.reindex(hold_idx)[cols94]
    assert list(X_b.index) == list(X_a.index)
    assert float(np.abs(X_b[cols94].to_numpy(float)
                        - X_a.to_numpy(float)).max()) == 0.0, (
        "champion two-way INPUT frames differ -- no number trusted")
    proba_b = champ_model.predict_proba(X_b[cols94])[:, 1]
    two_way_proba_diff = float(np.abs(proba_b - proba_c).max())
    assert two_way_proba_diff <= 1e-12, (
        f"champion two-way grade DISAGREES ({two_way_proba_diff}) "
        f"-- no number is trusted")
    pred_b = (proba_b >= 0.5).astype(int)
    assert (pred_b == pred_c).all(), (
        "champion two-way LABELS differ -- no number is trusted")
    acc_b = float(accuracy_score(yhold, pred_b))
    assert acc_b == acc_c

    scaler = model.named_steps["scaler"]
    stack = model.named_steps["stack"]
    Xs_hold = scaler.transform(Xhold)
    base_acc, imps = {}, {}
    for name, est in stack.named_estimators_.items():
        base_acc[name] = float(accuracy_score(
            yhold, est.predict(Xs_hold)))
        imp = np.asarray(est.feature_importances_, dtype=float)
        imps[name] = imp / imp.sum() if imp.sum() > 0 else imp
    agg_imp = np.mean([imps[k] for k in ("rf", "xgb", "lgbm", "et")],
                      axis=0)
    order = np.argsort(-agg_imp)
    top15 = [{"feature": cols[i], "importance": float(agg_imp[i])}
             for i in order[:15]]
    new_rank = {c: int(np.where(order == cols.index(c))[0][0]) + 1
                for c in pick2_features.NEW_COLUMNS}
    meta_coef = {name: float(c) for name, c in zip(
        ("rf", "xgb", "lgbm", "et"), stack.final_estimator_.coef_[0])}
    win = acc > acc_c
    degenerate = bool(win and (up_calls <= 7 or prec <= 0.05))
    effective_win = bool(win and not degenerate)
    save_result("stacker", {
        "acc": acc, "precision": prec, "up_calls": up_calls,
        "p_up_max": float(proba.max()), "p_up_mean": float(proba.mean()),
        "wf_acc": score["accuracy"], "wf_precision": score["precision"],
        "folds": score["folds"], "train_rows": int(len(train)),
        "champion_acc": acc_c, "champion_precision": prec_c,
        "champion_up_calls": up_c,
        "champion_p_up_max": float(proba_c.max()),
        "champion_two_way_agree": True,
        "champion_two_way_max_proba_diff": two_way_proba_diff,
        "win": win, "degenerate": degenerate,
        "effective_win": effective_win,
        "base_learner_holdout_acc": base_acc, "meta_coef": meta_coef,
        "top15": top15, "new_feature_ranks": new_rank,
    })
    print(f"stacker: pick2 acc={acc:.4f} prec={prec:.4f} "
          f"up={up_calls}/792 wf={score['accuracy']:.4f} vs "
          f"Champion #3 (run 69, re-scored two ways, agree) "
          f"acc={acc_c:.4f} prec={prec_c:.4f} up={up_c}/792 -> "
          f"{'WIN' if win else 'no win'}"
          f"{' (DEGENERATE -> LOSS per pre-registration)' if degenerate else ''}",
          flush=True)


def phase_votes() -> None:
    cols = cols98()
    cols94 = scored_eyes.champion_columns()
    _, prod_sup, split, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    scored_raw = pd.read_pickle(SCORED_RAW_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    X_raw = scored_raw.reindex(train.index)[cols]
    assert np.isfinite(X_raw.to_numpy(float)).all()
    y = train["target"].astype(int)
    print(f"cross-fitting PICK #2 stackers on {len(train)} train "
          f"rows (production anti-leak pattern, pick recipe) ...",
          flush=True)
    t0 = time.time()
    p_up_train = crossfit_p_up_pick(X_raw, y)
    print(f"  crossfit done in {time.time() - t0:.0f}s", flush=True)
    assert (p_up_train[:1] == 0.5).all()

    pick_model = joblib.load(STACKER_PATH)
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    hold_idx = prod_sup.index[split:]
    X_hold = scored_raw.reindex(hold_idx)[cols]
    X_hold94 = scored_raw.reindex(hold_idx)[cols94]
    assert np.isfinite(X_hold.to_numpy(float)).all()
    p_up_hold_pick = pick_model.predict_proba(X_hold)[:, 1]
    p_up_hold_champ = champ_model.predict_proba(X_hold94)[:, 1]
    np.savez(VOTES_PATH, p_up_train=p_up_train,
             p_up_hold_pick=p_up_hold_pick,
             p_up_hold_champ=p_up_hold_champ,
             train_index=train.index.astype(str),
             hold_index=hold_idx.astype(str))
    save_result("votes", {
        "train_vote_mean": float(np.mean(p_up_train)),
        "train_vote_std": float(np.std(p_up_train)),
        "train_vote_neutral_share": float(np.mean(p_up_train == 0.5)),
        "pick_hold_vote_mean": float(np.mean(p_up_hold_pick)),
        "pick_hold_vote_std": float(np.std(p_up_hold_pick)),
        "champ_hold_vote_mean": float(np.mean(p_up_hold_champ)),
        "champ_hold_vote_std": float(np.std(p_up_hold_champ)),
    })
    print(f"votes: train crossfit (pick2) mean={np.mean(p_up_train):.3f} "
          f"std={np.std(p_up_train):.3f}; holdout pick2 "
          f"mean={np.mean(p_up_hold_pick):.3f}; holdout champ69 "
          f"mean={np.mean(p_up_hold_champ):.3f}", flush=True)


def _load_votes():
    z = np.load(VOTES_PATH)
    return z["p_up_train"], z["p_up_hold_pick"], z["p_up_hold_champ"]


def _curve_and_telemetry(arm: Path):
    p = arm / "ppo_challenger_curve.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return d.get("curve"), d.get("telemetry")
        except Exception:
            return None, None
    return None, None


def phase_train() -> None:
    """60k fresh SB3 PPO on the DUEL-LENGTH shaped env (264-candle
    episodes). The production TradingEnv is swapped for the study
    subclass inside ppo_train's module namespace for THIS call
    only and restored immediately after."""
    cols = cols98()
    _, _, _, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    p_up_train, _, _ = _load_votes()
    assert len(p_up_train) == len(train)
    ARM.mkdir(parents=True, exist_ok=True)
    original = ppo_train_mod.TradingEnv
    ppo_train_mod.TradingEnv = DuelLengthShapedTradingEnv
    try:
        out = train_ppo(
            timesteps=STEPS, warm_start=False,
            champion_path=ARM / "ppo_fresh.zip",  # nonexistent: fresh
            stacker_path=STACKER_PATH,
            db_path=LAB / "train_tmp.db",
            ent_coef=0.05,
            sup=train, p_up=p_up_train,
            feature_columns=cols)
    finally:
        ppo_train_mod.TradingEnv = original
    assert ppo_train_mod.TradingEnv is original
    curve, telemetry = _curve_and_telemetry(ARM)
    save_result("arm", {"challenger_path": str(out), "timesteps": STEPS,
                        "episode_candles": muse_pick2.EPISODE_CANDLES,
                        "reward_curve": curve, "telemetry": telemetry})
    print(f"arm: fresh {STEPS} steps on 102-wide SHAPED state with "
          f"264-candle episodes -> {out}; "
          f"telemetry={'captured' if telemetry else 'MISSING'}",
          flush=True)


def phase_duel() -> None:
    cols = cols98()
    _, prod_sup, split, _, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    _, p_up_hold_pick, p_up_hold_champ = _load_votes()
    hold_p = prod_sup.iloc[split:]
    hold_s = env_sup.iloc[split:]
    assert list(hold_s.index) == list(hold_p.index)
    windows_p = [hold_p.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    windows_s = [hold_s.iloc[k * WINDOW:(k + 1) * WINDOW]
                 for k in range(3)]
    p_ups_champ = [p_up_hold_champ[k * WINDOW:(k + 1) * WINDOW]
                   for k in range(3)]
    p_ups_pick = [p_up_hold_pick[k * WINDOW:(k + 1) * WINDOW]
                  for k in range(3)]

    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows_p, p_ups_champ)
    assert champ_res is not None, "champion failed to grade"
    path = Path(results()["arm"]["challenger_path"])
    res = evaluate_windows(path, windows_s, p_ups_pick,
                           feature_columns=cols)
    assert res is not None, "pick2 challenger failed to grade"
    for k, w in enumerate(windows_p):
        closes = w["close"].to_numpy(dtype=float)
        bench = float(100_000.0 * (closes[-1] / closes[0] - 1.0))
        for r in (champ_res, res):
            assert abs(r["windows"][k]["bench_pnl"] - bench) < 1e-6
    for r in (champ_res, res):
        assert abs(r["agg_score"]
                   - float(np.mean([w["score"]
                                    for w in r["windows"]]))) < 1e-9
        assert abs(r["total_pnl"]
                   - float(sum(w["total_pnl"]
                               for w in r["windows"]))) < 1e-9
    champ_agg = _agg(champ_res)
    chall_agg = _agg(res)
    wins = sum(1 for cw, hw in zip(champ_res["windows"], res["windows"])
               if hw["score"] > cw["score"])
    formula_pass = bool(res["agg_score"] > champ_res["agg_score"]
                        and wins >= 2)
    promotable, floor_fail = gate_decision(chall_agg, formula_pass)
    save_result("duel", {
        "champion": champ_agg,
        "challenger": chall_agg, "windows_won": wins,
        "formula_pass": formula_pass, "floor_fail": floor_fail,
        "promotable": promotable})
    print(f"duel: pick2 agg={chall_agg['score']:.2f} "
          f"(pnl={chall_agg['total_pnl']:.2f}) vs champion "
          f"agg={champ_agg['score']:.2f} "
          f"(pnl={champ_agg['total_pnl']:.2f}); won {wins}/3; "
          f"formula={formula_pass} floor_fail={floor_fail} "
          f"promotable={promotable}", flush=True)


def _already_registered() -> int:
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM runs WHERE metrics LIKE "
            "'%\"family\": \"muse_pick2\"%'").fetchone()[0]
    finally:
        con.close()
    return int(n)


def phase_register() -> None:
    res = results()
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    if _already_registered() >= 2:
        print("register: muse_pick2 runs already present — skipping "
              "duplicate registration", flush=True)
        save_result("registered", {"guard_after": guard_after,
                                   "note": "existing registration "
                                           "verified; guard intact"})
        return
    cols = cols98()
    st = res["stacker"]
    duel = res["duel"]
    champ = duel["champion"]
    chall = duel["challenger"]

    # ---- stacker run ----
    win = bool(st["effective_win"])
    strict_win = bool(st["win"])
    degen = bool(st["degenerate"])
    if win:
        s_reason = (
            f"Muse's Pick #2 stacker holdout acc {st['acc']:.4f} "
            f"beat Champion #3 (run 69) {st['champion_acc']:.4f} on "
            f"the shared holdout timestamps (strict win, precision "
            f"{st['precision']:.4f}, {st['up_calls']}/792 up-calls) "
            f"— WOULD PROMOTE: held as PENDING KASS APPROVAL under "
            f"the promotion lock; champion kept (study verdict only)")
    elif degen:
        s_reason = (
            f"Muse's Pick #2 stacker holdout acc {st['acc']:.4f} "
            f"strictly beat Champion #3 (run 69) "
            f"{st['champion_acc']:.4f} BUT the anatomy is DEGENERATE "
            f"(up-calls {st['up_calls']}/792, precision "
            f"{st['precision']:.4f}) — counted as a LOSS under the "
            f"pre-registered run-63 rule; champion kept (study "
            f"verdict only)")
    else:
        s_reason = (
            f"Muse's Pick #2 stacker holdout acc {st['acc']:.4f} "
            f"did not beat Champion #3 (run 69) "
            f"{st['champion_acc']:.4f} on the shared holdout "
            f"timestamps; champion kept (study verdict only)")
    s_metrics = {
        "challenger_acc": st["acc"], "champion_acc": st["champion_acc"],
        "challenger_wf_acc": st["wf_acc"],
        "challenger_wf_precision": st["wf_precision"],
        "precision": st["precision"],
        "champion_precision": st["champion_precision"],
        "folds": st["folds"], "pending_approval": win,
        "reason": s_reason, "deep": False,
        "family": muse_pick2.FAMILY,
        "muse_pick2": {
            "label": "Stacker",
            "verdict": "PASS" if win else "FAIL",
            "design": DESIGN_STACKER,
            "features": cols, "n_features": len(cols),
            "new_features": pick2_features.NEW_COLUMNS,
            "margin": st["acc"] - st["champion_acc"],
            "train_rows": st["train_rows"],
            "up_calls": st["up_calls"],
            "champion_up_calls": st["champion_up_calls"],
            "strict_win": strict_win,
            "degenerate": degen,
            "base_learner_holdout_acc": st["base_learner_holdout_acc"],
            "meta_coef": st["meta_coef"],
            "top_features": [t["feature"] for t in st["top15"]],
            "new_feature_ranks": st["new_feature_ranks"],
        }}
    log_run("stacker", s_metrics, False)
    add_lab_note(
        "Muse's Pick #2 Stacker (family=muse_pick2)",
        f"{DESIGN_STACKER}. Holdout acc {st['acc']:.4f} vs "
        f"Champion #3 (run 69, re-scored two ways on identical "
        f"timestamps) {st['champion_acc']:.4f}; precision "
        f"{st['precision']:.4f} vs {st['champion_precision']:.4f}; "
        f"up-calls {st['up_calls']}/792 vs "
        f"{st['champion_up_calls']}/792; walk-forward "
        f"{st['wf_acc']:.4f}. "
        f"{'WIN — PENDING KASS APPROVAL under the promotion lock.' if win else 'DEGENERATE strict win — counted as a LOSS under the pre-registered run-63 rule.' if degen else 'No win.'} "
        f"Nothing installed.")
    print(f"register: Muse's Pick #2 Stacker logged (promoted=0, "
          f"pending={win})", flush=True)

    # ---- PPO run ----
    promotable = bool(duel["promotable"])
    wins = duel["windows_won"]
    if promotable:
        p_reason = (
            f"Muse's Pick #2 PPO (shaped, 264-candle episodes) won "
            f"{wins}/3 windows, aggregate {chall['score']:.0f} vs "
            f"champion {champ['score']:.0f} (gate pass incl. "
            f"two-part floor) — WOULD PROMOTE: held as PENDING KASS "
            f"APPROVAL under the promotion lock; champion kept "
            f"(study verdict only)")
    elif duel["floor_fail"] is not None:
        p_reason = (
            f"Muse's Pick #2 PPO (shaped, 264-candle episodes) "
            f"passed the relative gate (aggregate "
            f"{chall['score']:.0f} vs champion {champ['score']:.0f}, "
            f"won {wins}/3) but failed the floor — "
            f"{duel['floor_fail']}; champion kept")
    else:
        p_reason = (
            f"Muse's Pick #2 PPO (shaped, 264-candle episodes) won "
            f"{wins}/3 windows, aggregate {chall['score']:.0f} vs "
            f"champion {champ['score']:.0f}; champion kept (study "
            f"verdict)")
    arm = res["arm"]
    p_metrics = {
        "challenger": chall, "champion": champ,
        "reason": p_reason, "deep": False,
        "timesteps": STEPS,
        "pending_approval": promotable,
        "reward_curve": arm.get("reward_curve"),
        "telemetry": arm.get("telemetry"),
        "family": muse_pick2.FAMILY,
        "muse_pick2": {
            "label": "PPO",
            "verdict": "PASS" if promotable else "FAIL",
            "design": DESIGN_PPO,
            "features": cols, "n_features": len(cols),
            "state_size": len(cols) + 1 + 3,
            "transform": "score/50 - 1 ([0,100] -> [-1,1])",
            "vote_source": "Muse's Pick #2 Stacker: cross-fitted "
                           "out-of-fold (pick recipe) in training, "
                           "the fitted pick2 stacker on the holdout",
            "warm_start": False,
            "formula_pass": duel["formula_pass"],
            "floor_fail": duel["floor_fail"],
            "windows_won": wins,
            "training_episode_candles": muse_pick2.EPISODE_CANDLES,
            "reward_shaping": {
                "coef": muse_pick.SHAPING_COEF,
                "term": "-0.5 x (step increase in current-episode "
                        "drawdown, percentage points, floored at 0)",
                "base": "standard dense MtM log-return % clipped "
                        "[-1,1] + terminal +1/-1 + standard activity "
                        "penalty (all unchanged)",
                "scope": "training only; the duel grades both sides "
                         "in the standard production env",
                "fair_test": "training episodes are 264 candles "
                             "(duel-window length), answering Pick "
                             "#1's horizon confound",
            }}}
    log_run("ppo", p_metrics, False)
    add_lab_note(
        "Muse's Pick #2 PPO (family=muse_pick2)",
        f"{DESIGN_PPO}. Fresh 60k steps, seed 42, consuming the "
        f"Pick #2 stacker's votes. Duel vs champion pair (run 44 "
        f"PPO + run 69 votes): aggregate {chall['score']:.0f} (P&L "
        f"${chall['total_pnl']:+,.0f}) vs champion "
        f"{champ['score']:.0f} (P&L ${champ['total_pnl']:+,.0f}), "
        f"won {wins}/3. Buy-and-hold over the duel windows: "
        f"${champ['bench_pnl']:+,.0f}. "
        f"{'WOULD PROMOTE — PENDING KASS APPROVAL under the promotion lock.' if promotable else 'No promotion.'} "
        f"Nothing installed.")
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    save_result("registered", {
        "stacker_win": win, "stacker_strict_win": strict_win,
        "stacker_degenerate": degen,
        "ppo_promotable": promotable,
        "guard_after": guard_after})
    print("register: Muse's Pick #2 PPO logged (promoted=0); "
          "champion + seat + shadow guard intact", flush=True)


PHASES = [("data", phase_data), ("frames", phase_frames),
          ("stacker", phase_stacker), ("votes", phase_votes),
          ("train", phase_train), ("duel", phase_duel),
          ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== Muse's Pick #2 challenger study (phases done: "
          f"{sorted(done)}) ===", flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== Muse's Pick #2 challenger study complete ===", flush=True)


if __name__ == "__main__":
    main()
