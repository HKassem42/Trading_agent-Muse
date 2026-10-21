"""v4 migration (2026-10-20, Kass approved the full premium upgrade).

Moves the brains onto features v4 (16 features: +4h/daily context) and the
Coinbase data feed, so the live loop never faces an unrunnable champion:

1. Archive the current champions (rollback shelf).
2. Retrain the stacker on the 16-feature recipe; the old 12-feature
   champion cannot score the new frame, so the challenger takes the
   crown as the only runnable stacker (same pattern as the 1h migration).
3. Stacker ablation experiment: train two 100k-step brains on identical
   data -- one with the real stacker vote, one with a constant neutral
   0.5 vote -- and grade both on the 3 gate windows. This answers "is
   the coin-flip stacker earning its input slot?" with data. The result
   is logged as a lab note; production keeps the real vote either way
   until Kass decides.
4. Train the production PPO fresh (200k steps) on the train window with
   cross-fitted votes, grade it on the gate windows, promote it (the old
   champion is dimension-incompatible), and log everything.

Usage: .venv/bin/python scripts/migrate_v4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.market_data import fetch_klines  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS, feature_matrix, make_supervised  # noqa: E402
from models import stacker  # noqa: E402
from models.ppo_train import train as train_ppo  # noqa: E402
from training.nightly import (  # noqa: E402
    add_lab_note,
    archive_champions,
    crossfit_p_up,
    evaluate_windows,
    log_run,
)

MODELS = ROOT / "models"
DATA = ROOT / "data"


def main() -> None:
    import os
    resume_mode = os.environ.get("TAM_MIGRATE_FROM", "")
    resume = resume_mode in ("ablation", "ablation_eval")
    print(f"=== v4 migration (resume_from={resume_mode or 'start'}) ===",
          flush=True)
    if not resume:
        archive_champions("pre_v4")

    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    sup = make_supervised(df)
    print(f"supervised rows: {len(sup)} (features: {len(FEATURE_COLUMNS)})",
          flush=True)
    split = len(sup) - 792
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min()
    windows_sup = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    Xtr, ytr = feature_matrix(train_sup)
    Xhold, yhold = feature_matrix(hold_sup)

    if resume:
        # Stacker phase already completed in the crashed first run (the
        # 16-feature champion is in place and its run is logged); do not
        # duplicate the record.
        print("stacker: skipped (already migrated)", flush=True)
    else:
        # -- stacker on the new recipe --
        score = stacker.train_and_save(Xtr, ytr, MODELS / "stacker_challenger.pkl")
        new_model = joblib.load(MODELS / "stacker_challenger.pkl")
        from sklearn.metrics import accuracy_score, precision_score
        pred = new_model.predict(Xhold)
        acc = float(accuracy_score(yhold, pred))
        prec = float(precision_score(yhold, pred, zero_division=0))
        (MODELS / "stacker_challenger.pkl").replace(MODELS / "stacker_champion.pkl")
        (MODELS / "stacker_champion.json").write_text(json.dumps({
            "accuracy": acc, "precision": prec,
            "wf_accuracy": score["accuracy"], "wf_precision": score["precision"],
            "folds": score["folds"],
            "model_path": str(MODELS / "stacker_champion.pkl"),
        }))
        reason = (f"v4 stacker (16 features) holdout acc {acc:.4f}; the previous "
                  f"12-feature champion cannot score the new recipe, so the "
                  f"challenger takes the crown as the only runnable stacker")
        log_run("stacker", {"challenger_acc": acc, "champion_acc": -1.0,
                            "challenger_wf_acc": score["accuracy"],
                            "precision": prec, "reason": reason}, True)
        add_lab_note("Features v4 migration", reason)
        print(f"stacker: acc={acc:.4f} prec={prec:.4f} wf={score['accuracy']:.4f}",
              flush=True)

    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold_all = stacker_model.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    p_up_windows = [p_up_hold_all[k * 264:(k + 1) * 264] for k in range(3)]
    p_up_train = crossfit_p_up(Xtr, ytr)

    # Phase markers make the long migration restart-proof: each phase
    # re-runs from scratch only if it has not completed and marked itself.
    prog_path = MODELS / "migrate_v4_progress.json"
    if prog_path.exists():
        try:
            done = set(json.loads(prog_path.read_text()).get("done", []))
        except Exception:
            done = set()
    else:
        done = set()

    def mark(phase: str) -> None:
        done.add(phase)
        prog_path.write_text(json.dumps({"done": sorted(done)}))

    # -- ablation: real stacker vote vs constant neutral vote --
    if "with" not in done:
        print("ablation: training brain WITH the stacker vote (100k)...",
              flush=True)
        p_with = train_ppo(timesteps=100_000, warm_start=False,
                           champion_path=MODELS / "ppo_champion.zip",
                           stacker_path=MODELS / "stacker_champion.pkl",
                           db_path=DATA / "train_tmp.db", ent_coef=0.05,
                           sup=train_sup, p_up=p_up_train)
        p_with.replace(MODELS / "ablation_with_stacker.zip")
        mark("with")
    else:
        print("ablation: WITH brain already done — skipped", flush=True)
    if "without" not in done:
        print("ablation: training brain WITHOUT the stacker vote (100k)...",
              flush=True)
        p_without = train_ppo(timesteps=100_000, warm_start=False,
                              champion_path=MODELS / "ppo_champion.zip",
                              stacker_path=MODELS / "stacker_champion.pkl",
                              db_path=DATA / "train_tmp.db", ent_coef=0.05,
                              sup=train_sup,
                              p_up=np.full(len(train_sup), 0.5))
        p_without.replace(MODELS / "ablation_without_stacker.zip")
        mark("without")
    else:
        print("ablation: WITHOUT brain already done — skipped", flush=True)
    if "graded" not in done:
        ev_with = evaluate_windows(MODELS / "ablation_with_stacker.zip",
                                   windows_sup, p_up_windows)
        # The no-vote brain must be graded with the same neutral votes it
        # trained on, or the comparison measures the wrong thing.
        neutral_windows = [np.full(len(w), 0.5) for w in windows_sup]
        ev_without = evaluate_windows(MODELS / "ablation_without_stacker.zip",
                                      windows_sup, neutral_windows)
        s_with = ev_with["agg_score"] if ev_with else float("nan")
        s_without = ev_without["agg_score"] if ev_without else float("nan")
        verdict = (f"ablation over 3 windows: WITH stacker vote aggregate "
                   f"{s_with:,.0f} vs WITHOUT (constant 0.5) {s_without:,.0f} -- "
                   + ("the stacker signal is earning its slot"
                      if s_with > s_without else
                      "the stacker signal is NOT earning its slot at 100k steps"))
        add_lab_note("Stacker ablation (100k steps)", verdict)
        print(verdict, flush=True)
        mark("graded")
    else:
        print("ablation: already graded — skipped", flush=True)

    # -- production PPO, fresh on the new recipe --
    if "production" not in done:
        print("training production PPO (200k fresh)...", flush=True)
        challenger = train_ppo(timesteps=200_000, warm_start=False,
                               champion_path=MODELS / "ppo_champion.zip",
                               stacker_path=MODELS / "stacker_champion.pkl",
                               db_path=DATA / "train_tmp.db", ent_coef=0.05,
                               sup=train_sup, p_up=p_up_train)
        ev = evaluate_windows(challenger, windows_sup, p_up_windows)
        agg = ev["agg_score"] if ev else -1e9
        bench = float(sum(w["bench_pnl"] for w in ev["windows"])) if ev else 0.0
        reason = (f"v4 brain (16 features, Coinbase data, ATR/trailing exits, "
                  f"75% exposure cap) graded {agg:,.0f} across the 3 gate "
                  f"windows; buy-and-hold over the same windows ${bench:+,.0f}; "
                  f"the old champion cannot run the new recipe, so this brain "
                  f"takes the crown")
        chall_rec = {"score": agg,
                     "total_pnl": ev["total_pnl"] if ev else 0.0,
                     "mean_reward": ev["mean_reward"] if ev else 0.0,
                     "max_drawdown": float(max(w["max_drawdown"] for w in ev["windows"])) if ev else 0.0,
                     "bench_pnl": bench,
                     "action_mix": ev["windows"][-1]["action_mix"] if ev else {},
                     "windows": [{"score": round(w["score"], 2),
                                  "pnl": round(w["total_pnl"], 2),
                                  "bench": round(w["bench_pnl"], 2)}
                                 for w in ev["windows"]] if ev else [],
                     "runnable": True}
        reward_curve = None
        curve_path = MODELS / "ppo_challenger_curve.json"
        if curve_path.exists():
            try:
                reward_curve = json.loads(curve_path.read_text()).get("curve")
            except Exception:
                reward_curve = None
        challenger.replace(MODELS / "ppo_champion.zip")
        log_run("ppo", {"challenger": chall_rec,
                        "champion": {"score": -1e9, "runnable": False},
                        "reason": reason, "deep": False,
                        "timesteps": 200_000,
                        "reward_curve": reward_curve}, True)
        add_lab_note("v4 PPO promoted", reason)
        print(f"ppo: aggregate={agg:,.2f} bench={bench:+,.2f} PROMOTED",
              flush=True)
        mark("production")
    else:
        print("production PPO already done — skipped", flush=True)
    print("=== migration done ===", flush=True)


if __name__ == "__main__":
    main()
