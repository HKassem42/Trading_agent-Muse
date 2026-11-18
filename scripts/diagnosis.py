"""Lab diagnosis battery (2026-10-21, Kass green-lit both tracks).

Trading is PARKED: this script is lab-only. It never runs the live loop,
never touches the paper ledger, never promotes anything, and writes all
artifacts under data/diagnosis/ (models/ champions are read, never
written -- each PPO arm trains with champion_path pointed inside the lab
dir so train()'s challenger/curve outputs land there too).

Experiments:
1. Feature ablation on the clean stacker framework: per-family-only and
   leave-one-family-out variants, graded exactly like production (fit on
   the train frame -> holdout accuracy/precision on the last 792 rows;
   walk_forward_score on the train frame). Plus obv_slope vs netvol_rel
   correlation (Pearson + Spearman + sign agreement) and permutation
   importance of the full model on the holdout as a second angle.
2. Stacker-vote A/B: two PPO trainings, fresh, seed fixed by
   models.ppo_train (42), identical recipe/data, differing ONLY in the
   stacker p_up fed to training and grading: cross-fitted true votes vs
   a constant 0.5. (Fresh, not warm-started: the run-17 champion was
   itself trained with true votes, so warm-starting both arms from it
   would contaminate the constant arm; migrate_v51's production recipe
   was likewise a fresh train.)
3. Learning curve: one fresh 300k-step PPO with true votes, same recipe,
   graded identically.

Resumable: phases are marked in data/diagnosis/progress.json; ablation
variants are checkpointed individually in data/diagnosis/results.json;
the supervised frame is cached to data/diagnosis/sup.pkl so a restart
never shifts the holdout windows mid-battery. Safe to relaunch.

Usage: .venv/bin/python scripts/diagnosis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS, make_supervised  # noqa: E402
from models import stacker  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "diagnosis"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
SUP_PATH = LAB / "sup.pkl"
VOTES_PATH = LAB / "votes.npz"

# Feature families: a partition of FEATURE_COLUMNS (21). Kass's five
# personal indicators span two families: the volume-flow trio
# (obv_slope, adl_slope, netvol_rel) plus ha_dir / ewo. h4_atr_rel sits
# in multi_tf (it is a 4h feature); the 1h atr_rel / volatility_21 are
# the volatility family. volume_change is plain unsigned volume -> base.
FAMILIES = {
    "base_ta": ["body_pct", "upper_wick", "lower_wick", "range_pct",
                "rsi_c", "macd_hist_rel", "dist_vwap", "dist_sma50",
                "price_change", "volume_change"],
    "multi_tf": ["h4_rsi_c", "h4_dist_sma", "d1_dist_sma", "h4_atr_rel"],
    "volatility": ["atr_rel", "volatility_21"],
    "volume_flow": ["obv_slope", "adl_slope", "netvol_rel"],
    "kass_indicators": ["ha_dir", "ewo"],
}

PPO_TIMESTEPS_AB = 100_000
PPO_TIMESTEPS_LONG = 300_000
ENT_COEF = 0.05  # production challenger setting (nightly / migrate_v51)


# ---------------------------------------------------------------- state
def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    done = progress() | {phase}
    _save_json(PROGRESS_PATH, {"done": sorted(done)})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


# ---------------------------------------------------------------- data
def get_sup() -> pd.DataFrame:
    if SUP_PATH.exists():
        return pd.read_pickle(SUP_PATH)
    from data.market_data import fetch_klines
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    sup = make_supervised(df)
    LAB.mkdir(parents=True, exist_ok=True)
    sup.to_pickle(SUP_PATH)
    return sup


def split_frames(sup: pd.DataFrame):
    """Production split: last 792 rows = holdout, 3 windows of 264."""
    split = len(sup) - 792
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min(), "train/holdout overlap"
    windows = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    return train_sup, hold_sup, windows


def variant_columns() -> dict[str, list[str]]:
    """full + per-family-only + leave-one-family-out column sets."""
    out = {"full": list(FEATURE_COLUMNS)}
    for fam, cols in FAMILIES.items():
        out[f"only_{fam}"] = list(cols)
        out[f"loo_{fam}"] = [c for c in FEATURE_COLUMNS if c not in cols]
    return out


def aggregate_from_windows(windows: list[dict]) -> float:
    """Second-angle aggregate: plain mean of per-window duel scores,
    computed from the stored window records (evaluate_windows does the
    same internally; the report cross-checks the two)."""
    return float(np.mean([w["score"] for w in windows]))


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    sup = get_sup()
    train_sup, hold_sup, _ = split_frames(sup)
    feats = sup[FEATURE_COLUMNS]
    checks = {
        "feature_nan_free": bool(not feats.isna().any().any()),
        "features_within_contract": bool(
            (feats.min().min() >= -1.0) and (feats.max().max() <= 1.0)),
        "train_holdout_disjoint": bool(
            train_sup.index.max() < hold_sup.index.min()),
        "family_partition_exact": bool(
            sorted(sum(FAMILIES.values(), [])) == sorted(FEATURE_COLUMNS)),
    }
    save_result("meta", {
        "sup_rows": len(sup), "train_rows": len(train_sup),
        "holdout_rows": len(hold_sup),
        "sup_start": str(sup.index.min()), "sup_end": str(sup.index.max()),
        "holdout_start": str(hold_sup.index.min()),
        "checks": checks,
    })
    print(f"setup: {len(sup)} rows, train {len(train_sup)}, "
          f"holdout {len(hold_sup)}, checks={checks}", flush=True)
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_corr() -> None:
    from scipy.stats import pearsonr, spearmanr
    sup = get_sup()
    a, b = sup["obv_slope"], sup["netvol_rel"]
    pear = pearsonr(a, b)
    spear = spearmanr(a, b)
    out = {
        "n": int(len(sup)),
        "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "sign_agreement_frac": float((np.sign(a) == np.sign(b)).mean()),
        "mean_abs_diff": float((a - b).abs().mean()),
        "exact_equal_frac": float((a == b).mean()),
    }
    save_result("obv_netvol_corr", out)
    print(f"corr: pearson={out['pearson_r']:.4f} spearman="
          f"{out['spearman_rho']:.4f} sign_agree="
          f"{out['sign_agreement_frac']:.3f}", flush=True)


def phase_ablation() -> None:
    from sklearn.metrics import accuracy_score, precision_score
    sup = get_sup()
    train_sup, hold_sup, _ = split_frames(sup)
    Xtr_all, ytr = train_sup[FEATURE_COLUMNS], train_sup["target"]
    Xhold_all, yhold = hold_sup[FEATURE_COLUMNS], hold_sup["target"]
    done_variants = results().get("ablation", {})
    for name, cols in variant_columns().items():
        if name in done_variants:
            print(f"ablation {name}: cached -- skipped", flush=True)
            continue
        t0 = time.time()
        Xtr, Xhold = Xtr_all[cols], Xhold_all[cols]
        pipe = stacker.build_pipeline(random_state=42)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xhold)
        wf = stacker.walk_forward_score(Xtr, ytr, random_state=42)
        rec = {
            "features": cols, "n_features": len(cols),
            "holdout_acc": float(accuracy_score(yhold, pred)),
            "holdout_prec": float(precision_score(yhold, pred,
                                                  zero_division=0)),
            "holdout_n": int(len(yhold)),
            "wf_acc": wf["accuracy"], "wf_prec": wf["precision"],
            "train_n": int(len(ytr)),
            "seconds": round(time.time() - t0, 1),
        }
        abl = results().get("ablation", {})
        abl[name] = rec
        save_result("ablation", abl)
        print(f"ablation {name}: holdout={rec['holdout_acc']:.4f} "
              f"wf={rec['wf_acc']:.4f} ({rec['seconds']:.0f}s)", flush=True)


def phase_permimp() -> None:
    from sklearn.inspection import permutation_importance
    sup = get_sup()
    train_sup, hold_sup, _ = split_frames(sup)
    pipe = stacker.build_pipeline(random_state=42)
    pipe.fit(train_sup[FEATURE_COLUMNS], train_sup["target"])
    r = permutation_importance(
        pipe, hold_sup[FEATURE_COLUMNS], hold_sup["target"],
        n_repeats=30, random_state=42, scoring="accuracy", n_jobs=1)
    out = {c: {"mean": float(r.importances_mean[i]),
               "std": float(r.importances_std[i])}
           for i, c in enumerate(FEATURE_COLUMNS)}
    save_result("permutation_importance", out)
    top = sorted(out.items(), key=lambda kv: -kv[1]["mean"])[:5]
    print("permimp top5: " + ", ".join(
        f"{k}={v['mean']:+.4f}" for k, v in top), flush=True)


def phase_votes() -> None:
    import joblib
    from training.nightly import crossfit_p_up
    sup = get_sup()
    train_sup, hold_sup, windows = split_frames(sup)
    Xtr, ytr = train_sup[FEATURE_COLUMNS], train_sup["target"]
    p_train = crossfit_p_up(Xtr, ytr)
    champ = joblib.load(MODELS / "stacker_champion.pkl")
    p_hold = champ.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    np.savez(VOTES_PATH, p_train=p_train, p_hold=p_hold)
    save_result("votes", {
        "train_mean": float(p_train.mean()), "train_std": float(p_train.std()),
        "hold_mean": float(p_hold.mean()), "hold_std": float(p_hold.std()),
        "train_neutral_frac": float((p_train == 0.5).mean()),
    })
    print(f"votes: train p_up mean={p_train.mean():.3f} std="
          f"{p_train.std():.3f}; holdout mean={p_hold.mean():.3f}", flush=True)


def _grade(model_path: Path, windows, p_up_windows) -> dict | None:
    from training.nightly import evaluate_windows
    ev = evaluate_windows(model_path, windows, p_up_windows)
    if ev is None:
        return None
    ws = [{"score": w["score"], "total_pnl": w["total_pnl"],
           "bench_pnl": w["bench_pnl"], "mean_reward": w["mean_reward"],
           "max_drawdown": w["max_drawdown"],
           "action_mix": w["action_mix"]} for w in ev["windows"]]
    return {
        "windows": ws,
        "agg_score_stored": ev["agg_score"],
        "agg_score_recomputed": aggregate_from_windows(ws),
        "total_pnl": ev["total_pnl"],
        "bench_pnl_total": float(sum(w["bench_pnl"] for w in ws)),
        "mean_reward": ev["mean_reward"],
    }


def _train_and_grade(arm: str, timesteps: int, vote_mode: str) -> None:
    from models.ppo_train import train as train_ppo
    sup = get_sup()
    train_sup, hold_sup, windows = split_frames(sup)
    votes = np.load(VOTES_PATH)
    if vote_mode == "true":
        p_train = votes["p_train"]
        p_windows = [votes["p_hold"][k * 264:(k + 1) * 264] for k in range(3)]
    elif vote_mode == "const":
        p_train = np.full(len(train_sup), 0.5)
        p_windows = [np.full(264, 0.5) for _ in range(3)]
        assert (p_train == 0.5).all(), "const arm contaminated"
    else:
        raise ValueError(vote_mode)
    arm_dir = LAB / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out_zip = train_ppo(
        timesteps=timesteps, warm_start=False,
        champion_path=arm_dir / "ppo_champion.zip",  # output dir anchor only
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=arm_dir / "train_tmp.db",
        ent_coef=ENT_COEF, sup=train_sup, p_up=p_train)
    train_secs = time.time() - t0
    from stable_baselines3 import PPO
    obs_dim = int(PPO.load(out_zip).observation_space.shape[0])
    assert obs_dim == 25, f"obs dim {obs_dim} != 25"
    graded = _grade(out_zip, windows, p_windows)
    save_result(f"ppo_{arm}", {
        "timesteps": timesteps, "vote_mode": vote_mode,
        "warm_start": False, "ent_coef": ENT_COEF, "obs_dim": obs_dim,
        "train_seconds": round(train_secs, 1), "graded": graded,
        "model_path": str(out_zip),
    })
    if graded:
        print(f"ppo {arm}: agg={graded['agg_score_stored']:,.2f} "
              f"pnl={graded['total_pnl']:+,.2f} "
              f"bench={graded['bench_pnl_total']:+,.2f} "
              f"({train_secs:.0f}s train)", flush=True)
    else:
        print(f"ppo {arm}: GRADING FAILED (evaluate_windows -> None)",
              flush=True)


CHUNK_STEPS = 25_000  # ~8 min/chunk at the observed ~52 steps/s: sized so
# a runtime kill (observed every ~30-90 min, three kills during this
# battery) costs at most one small chunk of training.


def chunk_plan(done_chunks: list[int], total: int = PPO_TIMESTEPS_LONG,
               chunk: int = CHUNK_STEPS) -> list[tuple[int, int]]:
    """Remaining (chunk_index, timesteps) pairs to reach `total` steps.

    The 300k arm is trained in checkpointed 100k chunks: this runtime
    SIGKILLs long processes (the arm died silently at ~45 min twice --
    no traceback, log stops mid-phase), so one uninterrupted 300k
    learn() is not reliably runnable here. Chunking mirrors how the
    production nightly itself accumulates training (warm-start chunks),
    and total steps seen is the learning-curve variable of interest.
    """
    plan, acc, idx = [], 0, 0
    while acc < total:
        idx += 1
        steps = min(chunk, total - acc)
        if idx not in done_chunks:
            plan.append((idx, steps))
        acc += steps
    return plan


def phase_ppo_true300() -> None:
    """300k-step learning-curve arm, chunked + checkpointed (see
    chunk_plan). Env/broker/hyperparameters mirror models.ppo_train
    exactly (MlpPolicy, seed 42, lr 3e-4, n_steps 512, batch 64,
    ent_coef 0.05, $100k scratch broker 25% / 5%)."""
    import os

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from environment.trading_env import TradingEnv
    from trading.broker import PaperBroker

    sup = get_sup()
    train_sup, hold_sup, windows = split_frames(sup)
    votes = np.load(VOTES_PATH)
    p_train = votes["p_train"]
    p_windows = [votes["p_hold"][k * 264:(k + 1) * 264] for k in range(3)]
    arm_dir = LAB / "true300"
    arm_dir.mkdir(parents=True, exist_ok=True)
    done_chunks = results().get("ppo_true300_chunks", [])
    t_start = time.time()
    for idx, steps in chunk_plan(list(done_chunks)):
        work_db = arm_dir / f"train_tmp_{os.getpid()}.db"
        broker = PaperBroker(work_db, starting_balance=100_000,
                             max_position_pct=0.25, stop_loss_pct=0.05)
        env = Monitor(TradingEnv(train_sup, p_train, broker))
        prev = arm_dir / f"chunk{idx - 1}.zip"
        if idx > 1 and prev.exists():
            model = PPO.load(prev, env=env)
            model.ent_coef = ENT_COEF
        else:
            model = PPO("MlpPolicy", env, verbose=0, seed=42,
                        learning_rate=3e-4, n_steps=512, batch_size=64,
                        ent_coef=ENT_COEF)
        t0 = time.time()
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        out = arm_dir / f"chunk{idx}.zip"
        model.save(out)
        try:
            work_db.unlink(missing_ok=True)
        except OSError:
            pass
        done_chunks = sorted(set(done_chunks) | {idx})
        save_result("ppo_true300_chunks", done_chunks)
        print(f"ppo true300 chunk {idx} ({steps} steps) done in "
              f"{time.time() - t0:.0f}s -> {out}", flush=True)
    n_chunks = -(-PPO_TIMESTEPS_LONG // CHUNK_STEPS)
    final_zip = arm_dir / f"chunk{n_chunks}.zip"
    obs_dim = int(PPO.load(final_zip).observation_space.shape[0])
    assert obs_dim == 25, f"obs dim {obs_dim} != 25"
    graded = _grade(final_zip, windows, p_windows)
    save_result("ppo_true300", {
        "timesteps": PPO_TIMESTEPS_LONG, "vote_mode": "true",
        "warm_start": False, "ent_coef": ENT_COEF, "obs_dim": obs_dim,
        "chunked": True, "chunks": done_chunks,
        "train_seconds_this_run": round(time.time() - t_start, 1),
        "graded": graded, "model_path": str(final_zip),
    })
    if graded:
        print(f"ppo true300: agg={graded['agg_score_stored']:,.2f} "
              f"pnl={graded['total_pnl']:+,.2f} "
              f"bench={graded['bench_pnl_total']:+,.2f}", flush=True)
    else:
        print("ppo true300: GRADING FAILED (evaluate_windows -> None)",
              flush=True)


PHASES = [
    ("setup", phase_setup),
    ("corr", phase_corr),
    ("ablation", phase_ablation),
    ("permimp", phase_permimp),
    ("votes", phase_votes),
    ("ppo_true100", lambda: _train_and_grade("true100", PPO_TIMESTEPS_AB, "true")),
    ("ppo_const100", lambda: _train_and_grade("const100", PPO_TIMESTEPS_AB, "const")),
    ("ppo_true300", phase_ppo_true300),
]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== diagnosis battery (phases done: {sorted(done) or 'none'}) ===",
          flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done -- skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"--- phase {name} done in {time.time() - t0:.0f}s ---",
              flush=True)
    print("=== diagnosis battery complete ===", flush=True)


if __name__ == "__main__":
    main()
