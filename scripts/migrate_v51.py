"""v5 migration (2026-10-20, Kass: "include shorting now", tied to his
indicator features).

Moves the brains onto features v5 (21 features: + Heikin-Ashi direction,
OBV slope, ADL slope, net volume share, Elliott Wave Oscillator -- Kass's
own indicator set, computable only since the Coinbase feed made volume
real) and a 4-action env with SHORT support (25-dim observations), so the
live loop never faces an unrunnable champion:

1. Archive the current champions (rollback shelf, tag pre_v5).
2. Retrain the stacker on the 21-feature recipe; the v4 champion cannot
   score the new frame, so the challenger takes the crown as the only
   runnable stacker (same pattern as the 1h and v4 migrations).
3. Train the production PPO fresh (200k steps) on the train window with
   cross-fitted votes, grade it on the 3 gate windows, promote it (the
   old champion is dimension-incompatible: 24-dim obs -> 25), log the
   run (action mix includes the short share) and a lab note.

Phase markers (models/migrate_v51_progress.json) make the long run
restart-proof -- the v4 migration was killed four times by runtime
restarts before adopting them. Each phase re-runs only if it has not
completed and marked itself. Safe to relaunch any number of times.

Usage: .venv/bin/python scripts/migrate_v5.py
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
PHASES = ("archive", "stacker", "production")


def main() -> None:
    prog_path = MODELS / "migrate_v51_progress.json"
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

    print(f"=== v5.1 migration (phases done: {sorted(done) or 'none'}) ===",
          flush=True)

    if "archive" not in done:
        archive_champions("pre_v5")
        mark("archive")
        print("archive: champions shelved (pre_v5)", flush=True)
    else:
        print("archive: already done -- skipped", flush=True)

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

    if "stacker" not in done:
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
        reason = (f"v5 stacker (21 features incl. Kass's HA/OBV/ADL/NetVol/EWO) "
                  f"holdout acc {acc:.4f}; the previous 16-feature champion "
                  f"cannot score the new recipe, so the challenger takes the "
                  f"crown as the only runnable stacker")
        log_run("stacker", {"challenger_acc": acc, "champion_acc": -1.0,
                            "challenger_wf_acc": score["accuracy"],
                            "precision": prec, "reason": reason}, True)
        add_lab_note("Features v5.1 migration (multi-timeframe alignment fix)", reason)
        print(f"stacker: acc={acc:.4f} prec={prec:.4f} wf={score['accuracy']:.4f}",
              flush=True)
        mark("stacker")
    else:
        print("stacker: already migrated -- skipped", flush=True)

    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    p_up_hold_all = stacker_model.predict_proba(hold_sup[FEATURE_COLUMNS])[:, 1]
    p_up_windows = [p_up_hold_all[k * 264:(k + 1) * 264] for k in range(3)]
    p_up_train = crossfit_p_up(Xtr, ytr)

    # -- production PPO, fresh on the new recipe (SHORT enabled) --
    if "production" not in done:
        print("training production PPO (200k fresh, 4 actions)...", flush=True)
        challenger = train_ppo(timesteps=200_000, warm_start=False,
                               champion_path=MODELS / "ppo_champion.zip",
                               stacker_path=MODELS / "stacker_champion.pkl",
                               db_path=DATA / "train_tmp.db", ent_coef=0.05,
                               sup=train_sup, p_up=p_up_train)
        ev = evaluate_windows(challenger, windows_sup, p_up_windows)
        agg = ev["agg_score"] if ev else -1e9
        bench = float(sum(w["bench_pnl"] for w in ev["windows"])) if ev else 0.0
        mix = ev["windows"][-1]["action_mix"] if ev else {}
        reason = (f"v5.1 brain (21 features, SHORT enabled, FIXED multi-timeframe alignment (no look-ahead), Coinbase data, "
                  f"ATR/trailing exits both directions, 75% exposure cap) "
                  f"graded {agg:,.0f} across the 3 gate windows; buy-and-hold "
                  f"over the same windows ${bench:+,.0f}; short share of "
                  f"actions {mix.get('short', 0.0):.1%}; the old champion "
                  f"cannot run the new 25-dim recipe, so this brain takes "
                  f"the crown")
        chall_rec = {"score": agg,
                     "total_pnl": ev["total_pnl"] if ev else 0.0,
                     "mean_reward": ev["mean_reward"] if ev else 0.0,
                     "max_drawdown": float(max(w["max_drawdown"] for w in ev["windows"])) if ev else 0.0,
                     "bench_pnl": bench,
                     "action_mix": mix,
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
        add_lab_note("v5 PPO promoted", reason)
        print(f"ppo: aggregate={agg:,.2f} bench={bench:+,.2f} "
              f"short_share={mix.get('short', 0.0):.1%} PROMOTED", flush=True)
        mark("production")
    else:
        print("production PPO already done -- skipped", flush=True)
    print("=== migration done ===", flush=True)


if __name__ == "__main__":
    main()
