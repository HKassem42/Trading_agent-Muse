"""Champion frame dispatch for the pick2_98 tag (2026-10-27).

Run 73 (Muse's Pick #2 Stacker) was crowned Lab Stacker Champion
#4 under Kass's delegation of 2026-10-27 ("do whatever you think
is best"). Its frame is Champion #3's scored-102 frame PLUS Pick
#2's four causal features (98 columns, src/features/
pick2_features.py) -- a frame nightly's champion_holdout_frame
dispatch did not know. These tests pin:

- the pick2_98 branch: the frame comes from pick2_features,
  rows are reindexed to the holdout timestamps, columns are the
  champion meta's recorded columns (in the meta's order);
- the scored102 / prod21 branches are untouched;
- an unknown tag still raises (a silent fallback would misgrade
  the champion and corrupt every verdict built on the grade);
- pick2_features.builder_for_tag resolves both scored tags and
  raises on anything else (the shadow seat's builder slot).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import pick2_features  # noqa: E402
from training import nightly  # noqa: E402


def _hold_sup(n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2026-10-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"close": 100.0}, index=idx)


def test_pick2_98_dispatch_uses_pick2_frame_and_meta_columns(
        monkeypatch):
    hold = _hold_sup()
    full_idx = pd.date_range("2026-09-30", periods=30, freq="1h",
                             tz="UTC")
    cols = ["b_col", "a_col", "extra_col"]
    frame = pd.DataFrame(
        np.arange(30 * 3, dtype=float).reshape(30, 3),
        index=full_idx, columns=cols)
    seen = {}

    def fake_build(df):
        seen["df_len"] = len(df)
        return frame, list(cols), {"fake": True}

    monkeypatch.setattr(pick2_features, "build_pick2_frame",
                        fake_build)
    raw = pd.DataFrame({"close": 1.0}, index=full_idx)
    meta = {"feature_set": "pick2_98",
            "feature_columns": ["a_col", "b_col"]}
    X, tag = nightly.champion_holdout_frame(raw, hold, meta)
    assert tag == "pick2_98"
    assert seen["df_len"] == 30  # the raw frame, not the holdout
    assert list(X.columns) == ["a_col", "b_col"]  # the meta's order
    assert list(X.index) == list(hold.index)       # holdout-aligned
    want = frame.reindex(hold.index)[["a_col", "b_col"]]
    assert np.allclose(X.to_numpy(float), want.to_numpy(float))


def test_pick2_98_dispatch_defaults_to_the_built_columns(monkeypatch):
    hold = _hold_sup(3)
    full_idx = pd.date_range("2026-09-30", periods=10, freq="1h",
                             tz="UTC")
    cols = ["x", "y"]
    frame = pd.DataFrame(7.0, index=full_idx, columns=cols)
    monkeypatch.setattr(
        pick2_features, "build_pick2_frame",
        lambda df: (frame, list(cols), {}))
    X, tag = nightly.champion_holdout_frame(
        pd.DataFrame({"close": 1.0}, index=full_idx), hold,
        {"feature_set": "pick2_98"})
    assert tag == "pick2_98" and list(X.columns) == cols
    assert list(X.index) == list(hold.index)


def test_unknown_champion_tag_still_raises():
    with pytest.raises(ValueError):
        nightly.champion_holdout_frame(
            _hold_sup(), _hold_sup(),
            {"feature_set": "no_such_frame"})


def test_builder_for_tag_resolves_scored_tags():
    from features import scored_features

    assert pick2_features.builder_for_tag("pick2_98") is \
        pick2_features.build_pick2_frame_pair
    assert pick2_features.builder_for_tag("scored102") is \
        scored_features.build_scored_frame
    with pytest.raises(ValueError):
        pick2_features.builder_for_tag("prod21")
    with pytest.raises(ValueError):
        pick2_features.builder_for_tag(None)
