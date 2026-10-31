"""Eyes duel (Kass-approved lab study, 2026-10-29): does the stacker
upgrade actually change P&L, or only exam scores?

Same driver both sides -- PPO run 44 (models/ppo_champion.zip, hash
pinned) -- graded by the standard nightly duel (3 chronological
11-day windows x 10 episodes, next-candle-open fills, score =
P&L - 0.5 x max drawdown) on tonight's shared holdout (last 792
hourly candles, evaluation only). The ONLY difference between the
two sides is the eyes feeding P(up) votes into the brain:

  Side A: stacker run 64 (former Champion #2, scored102 frame,
          shelved at models/archive/20261027T023218_20261026_pre_champion3/)
  Side B: stacker run 73 (current Champion #4, pick2_98 frame,
          models/stacker_champion.pkl)

Each stacker is read on its OWN feature frame via nightly's
champion_holdout_frame -- the established champion-grading path --
and its holdout accuracy/precision tonight is recorded alongside
the duel so exam-vs-money can be compared directly.

Lab only. Nothing is promoted, installed, or written to any seat;
results land in data/eyes_duel/results.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.market_data import fetch_klines  # noqa: E402
from features.technical_indicators import make_supervised, feature_matrix  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import _agg, champion_holdout_frame, evaluate_windows  # noqa: E402

MODELS = ROOT / "models"
OUT = ROOT / "data" / "eyes_duel"
PPO_ZIP = MODELS / "ppo_champion.zip"
PPO_SHA256 = "e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb"

SIDES = {
    "run64": {
        "label": "Run 44 PPO + Run 64 eyes (scored102, former Champion #2)",
        "pkl": MODELS / "archive" / "20261027T023218_20261026_pre_champion3" / "stacker_champion.pkl",
        "meta": MODELS / "archive" / "20261027T023218_20261026_pre_champion3" / "stacker_champion.json",
        "pkl_sha256": "c0373d1536b20da8",  # prefix checked below
    },
    "run73": {
        "label": "Run 44 PPO + Run 73 eyes (pick2_98, Champion #4)",
        "pkl": MODELS / "stacker_champion.pkl",
        "meta": MODELS / "stacker_champion.json",
        "pkl_sha256": "89b633945843062a",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha256(PPO_ZIP) == PPO_SHA256, "ppo_champion.zip is not run 44"
    for side in SIDES.values():
        assert sha256(side["pkl"]).startswith(side["pkl_sha256"]), \
            f"stacker artifact mismatch: {side['pkl']}"

    # Fresh data + shared holdout, exactly as the nightly builds it.
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h", period="730d")
            break
        except Exception as e:
            print(f"fetch attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))
    sup = make_supervised(df)
    split = len(sup) - 792
    hold_sup = sup.iloc[split:]
    assert len(hold_sup) == 792
    windows_sup = [hold_sup.iloc[k * 264:(k + 1) * 264] for k in range(3)]
    _Xhold, yhold = feature_matrix(hold_sup)
    print(f"holdout {hold_sup.index[0]} .. {hold_sup.index[-1]} "
          f"({len(hold_sup)} candles)", flush=True)

    from sklearn.metrics import accuracy_score, precision_score

    results = {
        "study": "eyes_duel_64_vs_73",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "holdout_start": str(hold_sup.index[0]),
        "holdout_end": str(hold_sup.index[-1]),
        "holdout_rows": len(hold_sup),
        "ppo": "run 44 (models/ppo_champion.zip)",
        "sides": {},
    }
    for key, side in SIDES.items():
        meta = json.loads(side["meta"].read_text())
        model = joblib.load(side["pkl"])
        X_hold, tag = champion_holdout_frame(df, hold_sup, meta)
        assert list(X_hold.index) == list(hold_sup.index), \
            f"{key}: frame timestamps misaligned with holdout"
        votes = model.predict_proba(X_hold)[:, 1]
        assert len(votes) == 792 and np.isfinite(votes).all()
        assert votes.min() >= 0.0 and votes.max() <= 1.0
        pred = model.predict(X_hold)
        acc = float(accuracy_score(yhold, pred))
        prec = float(precision_score(yhold, pred, zero_division=0))
        up_calls = int((pred == 1).sum())
        print(f"{key}: tag={tag} acc={acc:.4f} prec={prec:.4f} "
              f"up_calls={up_calls}/792 vote_mean={votes.mean():.4f}",
              flush=True)
        p_up_windows = [votes[k * 264:(k + 1) * 264] for k in range(3)]
        res = evaluate_windows(PPO_ZIP, windows_sup, p_up_windows)
        assert res is not None, f"{key}: duel evaluation failed"
        flat = _agg(res)
        flat.update({"label": side["label"], "feature_set": tag,
                     "holdout_acc": acc, "holdout_precision": prec,
                     "up_calls": up_calls,
                     "vote_mean": float(votes.mean())})
        results["sides"][key] = flat
        print(f"{key}: duel agg={flat['score']:.2f} "
              f"pnl={flat['total_pnl']:.2f} maxDD={flat['max_drawdown']:.2f} "
              f"bench={flat['bench_pnl']:.2f}", flush=True)

    a, b = results["sides"]["run64"], results["sides"]["run73"]
    results["delta_73_minus_64"] = {
        "agg_score": b["score"] - a["score"],
        "total_pnl": b["total_pnl"] - a["total_pnl"],
        "holdout_acc": b["holdout_acc"] - a["holdout_acc"],
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["delta_73_minus_64"], indent=2), flush=True)
    print("EYES DUEL DONE", flush=True)


if __name__ == "__main__":
    main()
