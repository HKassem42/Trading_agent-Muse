"""One-off retrain for the 1h / features-v3 migration (2026-10-20).

Stacker: walk-forward training on 1h v3 features, same protocol as the
nightly job (train_and_save = TimeSeriesSplit CV score + full fit). The
challenger is promoted to champion regardless of the accuracy comparison:
the old champion was fit on the v2 recipe (it expects an `rsi_14n` column)
and cannot score v3 frames at all -- same documented situation as the
2026-10-20 PPO recovery promotion. The reason is recorded in the registry.

Usage: .venv/bin/python scripts/retrain_stacker_1h.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.yfinance_adapter import fetch_klines
from features.technical_indicators import feature_matrix, make_supervised
from models import stacker
from src.training.nightly import log_run

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HOLDOUT_ROWS = 240  # last 10 days of hourly candles, reserved for the PPO duel


def main() -> None:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    sup = make_supervised(df)
    train_sup = sup.iloc[:-HOLDOUT_ROWS]
    print(f"rows total={len(sup)} train={len(train_sup)} "
          f"holdout={len(sup) - len(train_sup)}", flush=True)
    X, y = feature_matrix(train_sup)

    champ_path = MODELS / "stacker_champion.json"
    champ_acc = json.loads(champ_path.read_text())["accuracy"] if champ_path.exists() else -1

    score = stacker.train_and_save(X, y, MODELS / "stacker_challenger.pkl")
    print(f"stacker 1h/v3 walk-forward: acc={score['accuracy']:.4f} "
          f"prec={score['precision']:.4f} folds={score['folds']} "
          f"(old 15m champion acc={champ_acc:.4f})", flush=True)

    reason = ("features v3 recipe change (rsi_14n -> rsi_c, strict [-1,1] "
              "scaling) + 1h timeframe: old champion cannot score v3 frames; "
              "promoted as the only runnable stacker for the 1h era")
    (MODELS / "stacker_challenger.pkl").replace(MODELS / "stacker_champion.pkl")
    champ_path.write_text(json.dumps(score))
    log_run("stacker", {**score, "challenger_acc": score["accuracy"],
                        "champion_acc": champ_acc, "timeframe": "1h",
                        "features": "v3", "promotion_reason": reason}, True)
    print("promoted stacker challenger -> champion; run logged", flush=True)


if __name__ == "__main__":
    main()
