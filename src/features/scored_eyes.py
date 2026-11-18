"""Scored-Eyes PPO -- LAB ONLY (Kass approved, 2026-10-25).

The study question: the lab stacker champion (Champion #2, run 64)
reads the matrix-v3 scored frame -- 94 percentile-scored indicators
-- while the PPO champion decides from the 21 raw production
features. The stacker's whole view reaches the PPO compressed into
its single vote. Does a PPO that sees the SAME scored eyes as the
stacker beat the 21-feature champion in the standard duel?

Construction rules for the challenger's state (98 numbers):
- the 94 scored features of Champion #2's frame
  (src/features/scored_features.py), restricted to run 64's
  RECORDED columns (models/stacker_champion.json -- never a rebuilt
  guess at what the champion reads);
- each score s in [0, 100] mapped into the env's [-1, 1] contract
  by the affine centering s/50 - 1 -- the same affine family the
  env itself applies to the stacker vote (2p - 1 on [0, 1]);
- then the vote (Champion #2 in grading; cross-fitted scored
  stackers in training, mirroring the production anti-leak pattern)
  and the env's 3 portfolio readings, built by TradingEnv exactly
  as for the production brain. 94 + 1 + 3 = 98.

The frame assembly here only JOINS the scored frame onto the
production supervised frame (timestamps + close / atr_pct_raw /
target carried untouched) and applies the transform; the scored
values themselves come from scored_features, whose causality was
verified under the v5.1 standard in the full-indicator stacker
study and the seat-separation work -- and the study builder
re-runs a truncation spot-check on its own built frame.

Production never imports this module; the live loop and the
nightly retrain stay on FEATURE_COLUMNS whatever the study
concludes. A study win installs nothing (the promotion lock,
2026-10-25).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import scored_features
from features.technical_indicators import make_supervised

ROOT = Path(__file__).resolve().parents[2]
FAMILY = "scored_eyes_ppo"
N_SCORED = 94
STATE_SIZE = N_SCORED + 1 + 3  # scores + vote + portfolio readings


def score_to_state(values):
    """Map percentile scores [0, 100] into the env's [-1, 1]
    contract: s/50 - 1 (so 0 -> -1, 50 -> 0, 100 -> +1). The same
    affine centering family TradingEnv applies to the stacker vote
    (2p - 1 for p in [0, 1]). Works on arrays, Series and frames;
    NaN (pre-warmup rows) passes through as NaN."""
    return values / 50.0 - 1.0


def champion_meta() -> dict:
    """The lab stacker champion's metadata sidecar."""
    p = ROOT / "models" / "stacker_champion.json"
    return json.loads(p.read_text())


# The scored frames a lab stacker champion may read, by its
# recorded feature_set tag: scored102 (94, Champions #2/#3) and
# pick2_98 (98, Champion #4 -- the scored frame plus Pick #2's
# four causal features).
_CHAMPION_FRAME_COLS = {"scored102": 94, "pick2_98": 98}


def champion_columns() -> list[str]:
    """The reigning lab champion's recorded feature columns, in
    the champion's own order -- the exact columns the crowned
    stacker reads (94 for the scored102 champions; 98 since run
    73 was crowned Champion #4 on 2026-10-27)."""
    meta = champion_meta()
    want = _CHAMPION_FRAME_COLS.get(meta.get("feature_set"))
    assert want is not None, (
        f"lab champion is not a scored-frame stacker: "
        f"{meta.get('feature_set')!r}")
    cols = list(meta.get("feature_columns") or [])
    assert len(cols) == want, f"expected {want} columns, got {len(cols)}"
    return cols


def assemble_env_sup(prod_sup: pd.DataFrame, scored_raw: pd.DataFrame,
                     columns: list[str]) -> pd.DataFrame:
    """The challenger's env frame: the production supervised frame
    (timestamps, OHLC, atr_pct_raw, target -- untouched) plus the 94
    scored columns, transformed into [-1, 1], aligned by timestamp.
    Rows before the scored region opens carry NaN in the scored
    columns; callers drop them for training and assert the holdout
    is fully finite."""
    out = prod_sup.copy()
    raw = scored_raw.reindex(prod_sup.index)[columns]
    out[columns] = score_to_state(raw)
    return out


def build_frames(df: pd.DataFrame):
    """(prod_sup, scored_raw, env_sup, columns) over a raw OHLCV
    frame: the production supervised set, Champion #2's scored
    frame restricted to its recorded columns (in its order), and
    the assembled env frame."""
    prod_sup = make_supervised(df)
    scored, built_cols = scored_features.build_scored_frame(df)
    columns = champion_columns()
    assert set(built_cols) == set(columns), (
        "scored frame builder columns differ from the champion's "
        "recorded set")
    scored_raw = scored[columns]
    env_sup = assemble_env_sup(prod_sup, scored_raw, columns)
    return prod_sup, scored_raw, env_sup, columns
