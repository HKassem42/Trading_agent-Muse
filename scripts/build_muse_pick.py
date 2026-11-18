"""Muse's Pick challenger LAB study (2026-10-26, Kass's order).

Kass: "go through [the RL/ML doc], pick something and create your
own next challenger -- a stacker and the brain for the Challenger.
Make sure to label it that it's just your pic." Source doc: his
Drive spreadsheet "Machine Learning and Reinforcement Learning
Kinds, Definitions, and Use Cases" (id
1r5x4Ze25CFWYBQGPCvqSOp4f-UwY_99jyefhbD54Ld0, tab "Table 1";
Kass confirmed the identification 2026-10-26). The pick, from its
rows: Supervised Learning + Reinforcement Learning paradigms, the
Reward Shaping technique, and the Stable Baselines3 RL library.

THE PAIR (pre-registered -- one shot each, no tuning sweeps):
1. Muse's Pick Stacker -- the production stacker recipe with ONE
   added base learner (ExtraTrees -> RF + XGBoost + LightGBM +
   ExtraTrees -> LogisticRegression) on Champion #2's scored-102
   frame (src/features/scored_features.py), graded against
   Champion #2 (run 64) re-scored on the identical holdout
   timestamps: holdout accuracy (strict win), precision, up-calls,
   walk-forward. A win with ~0 up-calls or ~0 precision is flagged
   DEGENERATE, exactly as run 63 was.
2. Muse's Pick PPO -- SB3 PPO, fresh brain, 60,000 steps, seed 42
   (the scored-eyes study's 60k arm recipe verbatim: 98-wide state
   = 94 scored features (s/50-1) + the stacker vote + 3 portfolio
   readings; votes cross-fitted in training, the fitted stacker on
   the holdout), consuming MY pick stacker's votes, with the
   pick's signature REWARD SHAPING: step reward = the standard
   dense mark-to-market reward - 0.5 x (the step's increase in
   current-episode drawdown, % points, floored at 0); the standard
   terminal +1/-1 rides along. Shaping applies in TRAINING only
   (ShapedTradingEnv subclass, swapped into ppo_train for this
   call); the duel grades both sides in the standard production
   env. The duel: challenger pair (pick stacker + pick PPO) vs
   champion pair (run 44 PPO on prod21 + run 64 votes, graded
   exactly as the scored-eyes study graded its champion side),
   3 windows x 10 episodes, next-candle-open fills, gate formula +
   two-part floor.

LAB ONLY under the promotion lock: nothing installs, nothing
touches either seat or either book; a win records pending_approval
for Kass. Champion/seat/shadow files are hash-guarded before/after
(the paper books are read-hash info only -- the live loops write
them on candle closes throughout the study window).

Usage: .venv/bin/python scripts/build_muse_pick.py
Phases checkpoint to data/muse_pick/; reruns skip done phases.
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
from features import scored_eyes, scored_features  # noqa: E402
from models import muse_pick  # noqa: E402
from models.muse_pick import (  # noqa: E402
    ShapedTradingEnv, crossfit_p_up_pick, train_pick_and_save)
from models.ppo_train import train as train_ppo  # noqa: E402
import models.ppo_train as ppo_train_mod  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import (  # noqa: E402
    _agg, add_lab_note, evaluate_windows, gate_decision, log_run)

MODELS = ROOT / "models"
LAB = ROOT / "data" / "muse_pick"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
ENV_SUP_PATH = LAB / "env_sup.pkl"
PROD_SUP_PATH = LAB / "prod_sup.pkl"
SCORED_RAW_PATH = LAB / "scored_raw.pkl"
STACKER_PATH = LAB / "stacker_pick.pkl"
VOTES_PATH = LAB / "votes.npz"
ARM = LAB / "arm60"

HOLDOUT = 792
WINDOW = 264
STEPS = 60_000
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
               "Distance to SMA 50 (%)"]
DESIGN_STACKER = (
    "Muse's Pick stacker: production stacker recipe + ExtraTrees "
    "(RF+XGB+LGBM+ET -> LR), scored-102 eyes (Champion #2's frame)")
DESIGN_PPO = (
    "Muse's Pick PPO: SB3 PPO, 98-wide scored state + Muse's Pick "
    "Stacker votes, reward shaping -0.5 x step drawdown increase "
    "(training only)")


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
    cols = scored_eyes.champion_columns()
    train = env_sup[env_sup.index < holdout_start].dropna(subset=cols)
    assert len(train) > 5_000, f"scored train frame too small: {len(train)}"
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
        "state_size": scored_eyes.STATE_SIZE,
        "n_scored": len(scored_eyes.champion_columns()),
    })
    print(f"data: {len(df)} raw rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(prod_sup)}, holdout from {holdout_start}", flush=True)


def phase_frames() -> None:
    """Identical construction + causality spot-check to the
    scored-eyes study's frames phase."""
    df, prod_sup, split, holdout_start, hold_start_pos = get_frames()
    cols = scored_eyes.champion_columns()
    print("building scored frame (Champion #2's eyes) ...", flush=True)
    t0 = time.time()
    prod2, scored_raw, env_sup, built = scored_eyes.build_frames(df)
    assert built == cols
    assert list(prod2.index) == list(prod_sup.index)
    print(f"  scored frame built in {time.time() - t0:.0f}s", flush=True)

    hold_env = env_sup.iloc[split:]
    hold_prod = prod_sup.iloc[split:]
    assert list(hold_env.index) == list(hold_prod.index), (
        "env holdout timestamps differ from production holdout")
    assert np.isfinite(hold_env[cols].to_numpy(float)).all(), (
        "non-finite scored feature inside the holdout")
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
        frame_t, _ = scored_features.build_scored_frame(df.iloc[:K + 1])
        per = {}
        for c in SAMPLE_COLS:
            a = scored_raw[c].iloc[:K + 1].to_numpy(float)
            b = frame_t[c].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            per[c] = {"max_abs_diff": d, "shared_rows": int(both.sum())}
            assert d <= 1e-9, f"causality FAIL (scored, K={K}): {c} {d}"
        pos_of = df.index.get_indexer(prod_sup.index)
        cand = [t for t, p in zip(prod_sup.index, pos_of)
                if p <= K and np.isfinite(
                    scored_raw.loc[t, SAMPLE_COLS[1]])]
        rows = cand[-200:]
        assert len(rows) >= 50, f"too few shared rows at K={K}"
        v_full = champ_model.predict_proba(
            scored_raw.reindex(rows)[cols])[:, 1]
        v_trunc = champ_model.predict_proba(
            frame_t.reindex(rows)[cols])[:, 1]
        vd = float(np.abs(v_full - v_trunc).max())
        per["__votes__"] = {"max_abs_diff": vd, "rows": len(rows)}
        assert vd <= 1e-12, f"vote alignment FAIL (K={K}): {vd}"
        trunc_evidence[str(K)] = per
        print(f"causality truncation K={K}: PASS (scores exact on "
              f"{len(SAMPLE_COLS)} sampled columns; run-64 votes "
              f"identical on {len(rows)} shared rows)", flush=True)

    env_sup.to_pickle(ENV_SUP_PATH)
    prod_sup.to_pickle(PROD_SUP_PATH)
    scored_raw.to_pickle(SCORED_RAW_PATH)
    save_result("frames", {
        "train_rows": len(train),
        "dropped_warmup_train_rows": int(split - len(train)),
        "holdout_rows": len(hold_env),
        "causality_truncation": trunc_evidence,
        "train_first": str(train.index[0]),
        "train_last": str(train.index[-1]),
    })
    print(f"frames: train {len(train)} rows (warmup-dropped "
          f"{split - len(train)}), holdout {len(hold_env)} finite + "
          f"label/close-identical to production", flush=True)


def phase_stacker() -> None:
    """Train + grade Muse's Pick Stacker against Champion #2
    re-scored on the identical holdout timestamps."""
    from sklearn.metrics import accuracy_score, precision_score

    cols = scored_eyes.champion_columns()
    _, prod_sup, split, holdout_start, _ = get_frames()
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
    print(f"pick stacker trained in {time.time() - t0:.0f}s "
          f"(wf acc={score['accuracy']:.4f})", flush=True)
    model = joblib.load(STACKER_PATH)
    proba = model.predict_proba(Xhold)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yhold, pred))
    prec = float(precision_score(yhold, pred, zero_division=0))
    up_calls = int(pred.sum())

    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    proba_c = champ_model.predict_proba(Xhold)[:, 1]
    pred_c = (proba_c >= 0.5).astype(int)
    acc_c = float(accuracy_score(yhold, pred_c))
    prec_c = float(precision_score(yhold, pred_c, zero_division=0))
    up_c = int(pred_c.sum())

    # Decision anatomy: each base learner's own holdout grade on the
    # scaled frame, the meta-learner's weights, and what the stack
    # leans on (mean base-learner importance).
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
    meta_coef = {name: float(c) for name, c in zip(
        ("rf", "xgb", "lgbm", "et"), stack.final_estimator_.coef_[0])}
    win = acc > acc_c
    degenerate = bool(win and (up_calls <= 7 or prec <= 0.05))
    save_result("stacker", {
        "acc": acc, "precision": prec, "up_calls": up_calls,
        "p_up_max": float(proba.max()), "p_up_mean": float(proba.mean()),
        "wf_acc": score["accuracy"], "wf_precision": score["precision"],
        "folds": score["folds"], "train_rows": int(len(train)),
        "champion_acc": acc_c, "champion_precision": prec_c,
        "champion_up_calls": up_c,
        "champion_p_up_max": float(proba_c.max()),
        "win": win, "degenerate": degenerate,
        "base_learner_holdout_acc": base_acc, "meta_coef": meta_coef,
        "top15": top15,
    })
    print(f"stacker: pick acc={acc:.4f} prec={prec:.4f} "
          f"up={up_calls}/792 wf={score['accuracy']:.4f} vs "
          f"Champion #2 (re-scored) acc={acc_c:.4f} "
          f"prec={prec_c:.4f} up={up_c}/792 -> "
          f"{'WIN' if win else 'no win'}"
          f"{' (DEGENERATE)' if degenerate else ''}", flush=True)


def phase_votes() -> None:
    cols = scored_eyes.champion_columns()
    _, prod_sup, split, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    scored_raw = pd.read_pickle(SCORED_RAW_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    X_raw = scored_raw.reindex(train.index)[cols]
    assert np.isfinite(X_raw.to_numpy(float)).all()
    y = train["target"].astype(int)
    print(f"cross-fitting PICK stackers on {len(train)} train rows "
          f"(production anti-leak pattern, pick recipe) ...", flush=True)
    t0 = time.time()
    p_up_train = crossfit_p_up_pick(X_raw, y)
    print(f"  crossfit done in {time.time() - t0:.0f}s", flush=True)
    assert (p_up_train[:1] == 0.5).all()

    pick_model = joblib.load(STACKER_PATH)
    champ_model = joblib.load(MODELS / "stacker_champion.pkl")
    hold_idx = prod_sup.index[split:]
    X_hold = scored_raw.reindex(hold_idx)[cols]
    assert np.isfinite(X_hold.to_numpy(float)).all()
    p_up_hold_pick = pick_model.predict_proba(X_hold)[:, 1]
    p_up_hold_champ = champ_model.predict_proba(X_hold)[:, 1]
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
    print(f"votes: train crossfit (pick) mean={np.mean(p_up_train):.3f} "
          f"std={np.std(p_up_train):.3f}; holdout pick "
          f"mean={np.mean(p_up_hold_pick):.3f}; holdout champ64 "
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
    """60k fresh SB3 PPO on the shaped env. The production
    TradingEnv is swapped for the study subclass inside ppo_train's
    module namespace for THIS call only and restored immediately
    after -- the production module on disk is never edited, and the
    duel phase below grades through nightly's own env reference."""
    cols = scored_eyes.champion_columns()
    _, _, _, holdout_start, _ = get_frames()
    env_sup = pd.read_pickle(ENV_SUP_PATH)
    train = _train_env_sup(env_sup, holdout_start)
    p_up_train, _, _ = _load_votes()
    assert len(p_up_train) == len(train)
    ARM.mkdir(parents=True, exist_ok=True)
    original = ppo_train_mod.TradingEnv
    ppo_train_mod.TradingEnv = ShapedTradingEnv
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
                        "reward_curve": curve, "telemetry": telemetry})
    print(f"arm: fresh {STEPS} steps on 98-wide SHAPED state -> {out}; "
          f"telemetry={'captured' if telemetry else 'MISSING'}",
          flush=True)


def phase_duel() -> None:
    cols = scored_eyes.champion_columns()
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
    assert res is not None, "pick challenger failed to grade"
    # Cross-checks (two ways), as in the scored-eyes duel.
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
    print(f"duel: pick agg={chall_agg['score']:.2f} "
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
            "'%muse_pick%'").fetchone()[0]
    finally:
        con.close()
    return int(n)


def phase_register() -> None:
    res = results()
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    if _already_registered() >= 2:
        print("register: muse_pick runs already present — skipping "
              "duplicate registration", flush=True)
        save_result("registered", {"guard_after": guard_after,
                                   "note": "existing registration "
                                           "verified; guard intact"})
        return
    cols = scored_eyes.champion_columns()
    st = res["stacker"]
    duel = res["duel"]
    champ = duel["champion"]
    chall = duel["challenger"]

    # ---- stacker run ----
    win = bool(st["win"])
    degen = bool(st["degenerate"])
    if win and degen:
        # Run 63's exact treatment: the strict-rule win is recorded
        # AND pends for Kass, with the degenerate anatomy flagged on
        # the record itself -- Kass declined 63 on exactly this
        # evidence; the call is his, not the study's.
        s_reason = (
            f"Muse's Pick stacker holdout acc {st['acc']:.4f} beat "
            f"Champion #2 {st['champion_acc']:.4f} on the shared "
            f"holdout timestamps (strict win) BUT the anatomy is "
            f"DEGENERATE (up-calls {st['up_calls']}/792, precision "
            f"{st['precision']:.4f}, P(up) max {st['p_up_max']:.3f}) "
            f"— flagged exactly as run 63 was — WOULD PROMOTE: held "
            f"as PENDING KASS APPROVAL under the promotion lock; "
            f"champion kept (study verdict only)")
    elif win:
        s_reason = (
            f"Muse's Pick stacker holdout acc {st['acc']:.4f} beat "
            f"Champion #2 {st['champion_acc']:.4f} on the shared "
            f"holdout timestamps (strict win, precision "
            f"{st['precision']:.4f}, {st['up_calls']}/792 up-calls) "
            f"— WOULD PROMOTE: held as PENDING KASS APPROVAL under "
            f"the promotion lock; champion kept (study verdict only)")
    else:
        s_reason = (
            f"Muse's Pick stacker holdout acc {st['acc']:.4f} did "
            f"not beat Champion #2 {st['champion_acc']:.4f} on the "
            f"shared holdout timestamps; champion kept (study "
            f"verdict only)")
    s_metrics = {
        "challenger_acc": st["acc"], "champion_acc": st["champion_acc"],
        "challenger_wf_acc": st["wf_acc"],
        "challenger_wf_precision": st["wf_precision"],
        "precision": st["precision"],
        "champion_precision": st["champion_precision"],
        "folds": st["folds"], "pending_approval": win,
        "reason": s_reason, "deep": False,
        "family": muse_pick.FAMILY,
        "muse_pick": {
            "label": "Stacker",
            "verdict": "PASS" if win else "FAIL",
            "design": DESIGN_STACKER,
            "features": cols, "n_features": len(cols),
            "margin": st["acc"] - st["champion_acc"],
            "train_rows": st["train_rows"],
            "up_calls": st["up_calls"],
            "champion_up_calls": st["champion_up_calls"],
            "degenerate": degen,
            "base_learner_holdout_acc": st["base_learner_holdout_acc"],
            "meta_coef": st["meta_coef"],
            "top_features": [t["feature"] for t in st["top15"]],
        }}
    log_run("stacker", s_metrics, False)
    add_lab_note(
        "Muse's Pick Stacker (family=muse_pick)",
        f"{DESIGN_STACKER}. Holdout acc {st['acc']:.4f} vs "
        f"Champion #2 (run 64, re-scored on identical timestamps) "
        f"{st['champion_acc']:.4f}; precision {st['precision']:.4f} "
        f"vs {st['champion_precision']:.4f}; up-calls "
        f"{st['up_calls']}/792 vs {st['champion_up_calls']}/792; "
        f"walk-forward {st['wf_acc']:.4f}. "
        f"{'STRICT WIN with DEGENERATE anatomy (flagged as run 63 was) — PENDING KASS APPROVAL under the promotion lock.' if degen else 'WIN — PENDING KASS APPROVAL under the promotion lock.' if win else 'No win.'} "
        f"Nothing installed.")
    print(f"register: Muse's Pick Stacker logged (promoted=0, "
          f"pending={win})", flush=True)

    # ---- PPO run ----
    promotable = bool(duel["promotable"])
    wins = duel["windows_won"]
    if promotable:
        p_reason = (
            f"Muse's Pick PPO (reward-shaped) won {wins}/3 windows, "
            f"aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f} (gate pass incl. two-part floor) "
            f"— WOULD PROMOTE: held as PENDING KASS APPROVAL under "
            f"the promotion lock; champion kept (study verdict only)")
    elif duel["floor_fail"] is not None:
        p_reason = (
            f"Muse's Pick PPO (reward-shaped) passed the relative "
            f"gate (aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f}, won {wins}/3) but failed the "
            f"floor — {duel['floor_fail']}; champion kept")
    else:
        p_reason = (
            f"Muse's Pick PPO (reward-shaped) won {wins}/3 windows, "
            f"aggregate {chall['score']:.0f} vs champion "
            f"{champ['score']:.0f}; champion kept (study verdict)")
    arm = res["arm"]
    p_metrics = {
        "challenger": chall, "champion": champ,
        "reason": p_reason, "deep": False,
        "timesteps": STEPS,
        "pending_approval": promotable,
        "reward_curve": arm.get("reward_curve"),
        "telemetry": arm.get("telemetry"),
        "family": muse_pick.FAMILY,
        "muse_pick": {
            "label": "PPO",
            "verdict": "PASS" if promotable else "FAIL",
            "design": DESIGN_PPO,
            "features": cols, "n_features": len(cols),
            "state_size": scored_eyes.STATE_SIZE,
            "transform": "score/50 - 1 ([0,100] -> [-1,1])",
            "vote_source": "Muse's Pick Stacker: cross-fitted "
                           "out-of-fold (pick recipe) in training, "
                           "the fitted pick stacker on the holdout",
            "warm_start": False,
            "formula_pass": duel["formula_pass"],
            "floor_fail": duel["floor_fail"],
            "windows_won": wins,
            "reward_shaping": {
                "coef": muse_pick.SHAPING_COEF,
                "term": "-0.5 x (step increase in current-episode "
                        "drawdown, percentage points, floored at 0)",
                "base": "standard dense MtM log-return % clipped "
                        "[-1,1] + terminal +1/-1 + standard activity "
                        "penalty (all unchanged)",
                "scope": "training only; the duel grades both sides "
                         "in the standard production env",
            }}}
    log_run("ppo", p_metrics, False)
    add_lab_note(
        "Muse's Pick PPO (family=muse_pick)",
        f"{DESIGN_PPO}. Fresh 60k steps, seed 42, consuming the "
        f"pick stacker's votes. Duel vs champion pair (run 44 PPO + "
        f"run 64 votes): aggregate {chall['score']:.0f} (P&L "
        f"${chall['total_pnl']:+,.0f}) vs champion "
        f"{champ['score']:.0f} (P&L ${champ['total_pnl']:+,.0f}), "
        f"won {wins}/3. Buy-and-hold over the duel windows: "
        f"${champ['bench_pnl']:+,.0f}. "
        f"{'WOULD PROMOTE — PENDING KASS APPROVAL under the promotion lock.' if promotable else 'No promotion.'} "
        f"Nothing installed.")
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    save_result("registered", {
        "stacker_win": win, "stacker_degenerate": degen,
        "ppo_promotable": promotable,
        "guard_after": guard_after})
    print("register: Muse's Pick PPO logged (promoted=0); champion "
          "+ seat + shadow guard intact", flush=True)


PHASES = [("data", phase_data), ("frames", phase_frames),
          ("stacker", phase_stacker), ("votes", phase_votes),
          ("train", phase_train), ("duel", phase_duel),
          ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== Muse's Pick challenger study (phases done: "
          f"{sorted(done)}) ===", flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== Muse's Pick challenger study complete ===", flush=True)


if __name__ == "__main__":
    main()
