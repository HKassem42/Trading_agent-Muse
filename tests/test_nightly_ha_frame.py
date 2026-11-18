"""Nightly HA-frame wiring tests (Kass's Pick #11 line, 2026-11-05).

Contracts: the champion frame resolver serves the ha10 frame
(the Pick #11 champion's own feature set); walk_forward_score
and crossfit_p_up honor an injected pipeline factory (the HA
recipe) while defaulting to the production stacker recipe.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS
from training import nightly


def _df(n=300):
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.ones(n)}, index=idx)


def test_champion_frame_ha10_branch():
    df = _df()
    hold = df.iloc[-50:].copy()
    hold["target"] = 1
    meta = {"feature_set": "ha10", "feature_columns": HA_COLUMNS}
    X, tag = nightly.champion_holdout_frame(df, hold, meta)
    assert tag == "ha10"
    assert list(X.columns) == HA_COLUMNS
    assert X.index.equals(hold.index)
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_champion_frame_unknown_tag_still_raises():
    df = _df(60)
    hold = df.iloc[-10:].copy()
    with pytest.raises(ValueError):
        nightly.champion_holdout_frame(df, hold,
                                       {"feature_set": "nope"})


def _dummy_factory(random_state=42):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    return Pipeline([("clf", DummyClassifier(strategy="prior"))])


def test_walk_forward_score_honors_factory():
    from models import stacker
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series((rng.random(120) > 0.5).astype(int))
    out = stacker.walk_forward_score(X, y, n_splits=3,
                                     pipeline_factory=_dummy_factory)
    assert out["folds"] == 3
    assert 0.0 <= out["accuracy"] <= 1.0


def test_crossfit_p_up_honors_factory():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = pd.Series((rng.random(120) > 0.5).astype(int))
    p = nightly.crossfit_p_up(X, y, folds=3,
                              pipeline_factory=_dummy_factory)
    assert len(p) == 120
    assert (p >= 0.0).all() and (p <= 1.0).all()
    # leading segment keeps the neutral 0.5 convention
    assert (p[:30] == 0.5).all()
