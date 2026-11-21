"""Scored-Eyes PPO (LAB study, 2026-10-25, Kass approved): a PPO
challenger whose market features are the SAME scored-indicator eyes
as lab stacker Champion #2 (run 64, the matrix-v3 scored102 set) --
94 percentile scores mapped into the env's [-1, 1] contract -- plus
Champion #2's vote and the 3 portfolio readings: a 98-number state,
vs the production champion's 25 (21 raw + vote + 3).

Pins:
- the score->state transform (the affine [0,100] -> [-1,1] map, the
  same family as the env's own vote centering 2p-1);
- the challenger's column set is EXACTLY run 64's recorded 94 (from
  models/stacker_champion.json), never a rebuilt guess;
- the env-sup assembly (transform + timestamp alignment, production
  columns -- close / atr_pct_raw -- carried untouched);
- the env state layout at width 98 (features | centered vote |
  portfolio), so the study brain sees what the record says it saw;
- the compact feed's family registration (code 13, display name,
  models-index headline/verdict);
- scored-frame causality in miniature: a truncation rebuild must
  reproduce the full frame's scores -- and a fitted stacker's votes
  -- on every shared row (the vote at row i reads only candles <= i).

Run: .venv/bin/python -m pytest tests/test_scored_eyes.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from environment.trading_env import TradingEnv  # noqa: E402
from features import scored_eyes, scored_features  # noqa: E402
from models import stacker  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402
from src.dashboards import compact  # noqa: E402


def _frame(n=7000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


# ------------------------------------------------------------ transform

def test_score_to_state_is_the_affine_centering():
    # The same affine family the env applies to the vote (2p-1 on
    # [0,1]): scores live on [0,100], so the map is s/50 - 1.
    out = scored_eyes.score_to_state(np.array([0.0, 25.0, 50.0,
                                               75.0, 100.0]))
    assert np.allclose(out, [-1.0, -0.5, 0.0, 0.5, 1.0])
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 100, 10_000)
    t = scored_eyes.score_to_state(s)
    assert t.min() >= -1.0 and t.max() <= 1.0


def test_score_to_state_preserves_series_index():
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    s = pd.Series([0.0, 50.0, 100.0], index=idx)
    t = scored_eyes.score_to_state(s)
    assert list(t.index) == list(idx)
    assert np.allclose(t.to_numpy(), [-1.0, 0.0, 1.0])


# ------------------------------------------------------------ columns

def test_champion_columns_are_the_recorded_columns():
    meta = json.loads((ROOT / "models" / "stacker_champion.json")
                      .read_text())
    # RE-PINNED 2026-11-21 (crown-agnostic): this used to pin run 96
    # (prod21, Champion #5, crowned 2026-11-02) -- the crown has
    # since moved through the HA line to run 126 (ha10, Champion
    # #13). The pin's substance is unchanged: the champion's columns
    # are its RECORDED columns, resolved from its own recorded
    # feature set -- so the test tracks the crown instead of rotting
    # at every crowning. The scored102 set this study was built
    # against is still run 64's recorded 94, read from the registry
    # (it no longer depends on who holds the crown) -- and because
    # the reigning champion is NOT a scored-frame stacker, the
    # scored-eyes bridge must refuse it loudly rather than feed it
    # the wrong frame.
    from features.technical_indicators import FEATURE_COLUMNS
    from models.muse_pick10 import HA_COLUMNS
    by_set = {"ha10": list(HA_COLUMNS),
              "prod21": list(FEATURE_COLUMNS)}
    assert meta["feature_set"] in by_set, meta["feature_set"]
    assert meta["feature_columns"] == by_set[meta["feature_set"]]
    import sqlite3
    con = sqlite3.connect(ROOT / "data" / "training.db")
    m64 = json.loads(con.execute(
        "SELECT metrics FROM runs WHERE id=64").fetchone()[0])
    con.close()
    recorded94 = m64["full_indicator_stacker"]["features"]
    assert len(recorded94) == 94
    assert set(scored_features.scored_columns()) == set(recorded94)
    with pytest.raises(AssertionError, match="not a scored-frame"):
        scored_eyes.champion_columns()


# ------------------------------------------------------------ env sup

def test_assemble_env_sup_transforms_and_aligns():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    prod = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0,
         "atr_pct_raw": 0.01, "target": [0, 1, 0, 1]}, index=idx)
    cols = ["c_a", "c_b"]
    raw = pd.DataFrame({"c_a": [0.0, 50.0, 100.0, 25.0],
                        "c_b": [100.0, 100.0, 0.0, 50.0]}, index=idx)
    env_sup = scored_eyes.assemble_env_sup(prod, raw, cols)
    assert np.allclose(env_sup["c_a"].to_numpy(),
                       [-1.0, 0.0, 1.0, -0.5])
    assert np.allclose(env_sup["c_b"].to_numpy(),
                       [1.0, 1.0, -1.0, 0.0])
    # Production columns ride along untouched (broker rules read
    # close / atr_pct_raw; the target is the stacker's label).
    assert (env_sup["close"] == 2.0).all()
    assert (env_sup["atr_pct_raw"] == 0.01).all()
    assert list(env_sup["target"]) == [0, 1, 0, 1]
    assert list(env_sup.index) == list(idx)


# ------------------------------------------------------------ env state

def test_env_state_is_98_wide_with_the_recorded_layout(tmp_path):
    cols = [f"c{k}" for k in range(94)]
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    sup = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                        "close": 100.0}, index=idx)
    for j, c in enumerate(cols):
        sup[c] = -1.0 + 2.0 * j / 93.0  # distinct value per column
    p_up = np.full(n, 0.75)
    broker = PaperBroker(tmp_path / "t.db", starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    env = TradingEnv(sup, p_up, broker, feature_columns=cols)
    assert env.observation_space.shape == (scored_eyes.STATE_SIZE,)
    assert scored_eyes.STATE_SIZE == 98
    obs, _ = env.reset(seed=0, options={"start": 0})
    assert obs.shape == (98,)
    # market block: the 94 feature values, in column order
    assert np.allclose(obs[:94],
                       [-1.0 + 2.0 * j / 93.0 for j in range(94)],
                       atol=1e-6)
    # then the centered vote: 2*0.75 - 1 = 0.5
    assert obs[94] == pytest.approx(0.5)
    # then the portfolio readings of a fresh, flat account:
    # all cash, no position, no unrealized P&L
    assert np.allclose(obs[95:], [1.0, 0.0, 0.0], atol=1e-6)


# ------------------------------------------------------------ feed family

def _fake_run(label="60k Fresh", verdict="FAIL", score=-10.0):
    return {"id": 999, "ts": "2026-10-25T23:00:00+00:00", "kind": "ppo",
            "promoted": False,
            "metrics": {
                "challenger": {"score": score, "total_pnl": -5.0},
                "champion": {"score": 1.0, "total_pnl": 2.0},
                "reason": "study verdict",
                "family": "scored_eyes_ppo",
                "scored_eyes_ppo": {"label": label,
                                    "verdict": verdict,
                                    "design": "scored eyes"}}}


def test_compact_family_registered():
    assert compact._FAMILY_CODES["scored_eyes_ppo"] == 13
    r = _fake_run()
    name = compact._display_name(r, None)
    assert name == "Scored-Eyes PPO — 60k Fresh"
    entry = compact._model_entry(r, name)
    assert entry[3] == 13          # family code rides the index row
    assert entry[5] == -10.0       # headline = challenger agg score
    assert entry[6] == -5          # P&L, whole dollars
    assert entry[7] == "FAIL"      # the family's recorded verdict


# ------------------------------------------------------------ causality

def test_scored_frame_truncation_and_vote_alignment():
    """Miniature of the study's spot-check: scores rebuilt on a
    truncated frame must equal the full frame's on every shared row,
    and a stacker fitted once must produce identical votes from the
    truncated frame's features -- the vote at row i reads only
    candles <= i."""
    df = _frame()
    names = ["Relative Strength Index (RSI)", "Distance to SMA 50 (%)"]
    full, cols = scored_features.build_scored_frame(df, names=names)
    assert set(names) <= set(cols)
    K = len(df) - 500
    trunc, _ = scored_features.build_scored_frame(df.iloc[:K + 1],
                                                  names=names)
    for c in names:
        a = full[c].iloc[:K + 1].to_numpy(float)
        b = trunc[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 500
        assert np.allclose(a[both], b[both], rtol=0, atol=1e-12), c
    # Vote alignment: fit on early scored rows of the FULL frame,
    # then compare votes from full vs truncated features.
    scored = full[names].copy()
    scored["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    scored = scored.dropna()
    X = scored[names]
    y = scored["target"]
    pipe = stacker.build_pipeline(random_state=42)
    pipe.fit(X.iloc[:1200], y.iloc[:1200])
    rows = X.index[1500:2000]
    v_full = pipe.predict_proba(X.loc[rows])[:, 1]
    Xt = trunc[names].reindex(rows)
    assert np.isfinite(Xt.to_numpy(float)).all()
    v_trunc = pipe.predict_proba(Xt)[:, 1]
    assert np.allclose(v_full, v_trunc, rtol=0, atol=1e-12)
