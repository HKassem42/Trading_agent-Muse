"""Deep-run stacker re-grade skip (deep-train resume fix, 2026-11-04).

Why runs 93/94/95 exist: the Nov 1 weekly deep train died three
times, and every relaunch re-ran the WHOLE nightly pipeline from
the top -- including training, grading, and registering the stacker
challenger again. Three registry rows for what was one deep run's
one stacker challenger (their metrics differ slightly only because
each relaunch re-fetched a shifted frame; the run, the champion,
and the recipe were the same).

The skip keys on that identity: a stacker row registered by a DEEP
run carries deep=True, the deep run's id (the pipeline's date-based
run_id), and the sha256 of the champion stacker file it was graded
against. A later attempt of the same deep run -- same run id, same
champion stacker bytes -- finds that row and skips re-grading /
re-registering. Nightly (non-deep) runs never skip and never
match, and a different night or a different champion never skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import nightly  # noqa: E402

RUN_ID = "20261101"
CHAMP_SHA = "a" * 64


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "DATA", tmp_path)
    return tmp_path


def _register_stacker(metrics):
    nightly.log_run("stacker", metrics, False)


def _deep_metrics(**over):
    m = {"challenger_acc": 0.5152, "champion_acc": 0.5328,
         "challenger_wf_acc": 0.4984, "precision": 0.5117,
         "folds": 5, "pending_approval": False,
         "reason": "stacker challenger holdout acc 0.5152 did not "
                   "beat champion 0.5328",
         "deep": True, "deep_run_id": RUN_ID,
         "champion_stacker_sha256": CHAMP_SHA}
    m.update(over)
    return m


def test_same_deep_run_stacker_is_found(registry):
    _register_stacker(_deep_metrics())
    found = nightly.find_registered_deep_stacker(RUN_ID, CHAMP_SHA)
    assert found is not None
    assert found["challenger_acc"] == 0.5152
    assert found["deep_run_id"] == RUN_ID


def test_no_skip_for_other_run_or_other_champion(registry):
    _register_stacker(_deep_metrics())
    assert nightly.find_registered_deep_stacker("20261108",
                                                CHAMP_SHA) is None
    assert nightly.find_registered_deep_stacker(RUN_ID,
                                                "b" * 64) is None
    assert nightly.find_registered_deep_stacker(RUN_ID, None) is None


def test_nightly_rows_never_match(registry):
    """A plain nightly stacker row from the same date (the midnight
    retrain runs hours before the deep train) carries no deep tag
    and must not suppress the deep run's own grading."""
    nightly_metrics = _deep_metrics()
    for key in ("deep", "deep_run_id", "champion_stacker_sha256"):
        nightly_metrics.pop(key)
    _register_stacker(nightly_metrics)
    assert nightly.find_registered_deep_stacker(RUN_ID,
                                                CHAMP_SHA) is None


def test_skip_decision_requires_deep_mode(registry):
    _register_stacker(_deep_metrics())
    assert nightly.deep_stacker_skip(False, RUN_ID, CHAMP_SHA) is None
    assert nightly.deep_stacker_skip(True, RUN_ID,
                                     CHAMP_SHA) is not None
    assert nightly.deep_stacker_skip(True, "20261108",
                                     CHAMP_SHA) is None
