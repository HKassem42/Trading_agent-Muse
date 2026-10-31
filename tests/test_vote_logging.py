"""Grading-time vote logging (Kass-approved 2026-10-31): when the
nightly grader scores a stacker challenger it already computes the
holdout up-probability votes -- and used to throw them away, which
is why run 73's dashboard votes panel had to borrow a study array
and why any future champion would have had no votes block at all.
Now the run record itself carries a compact summary
(metrics["votes"]: a 20-bin histogram over [0, 1] plus n/min/max/
median and the two coin-flip band shares) in the exact shape the
compact training feed serves.

This is INSTRUMENTATION ONLY. The adversarial core of this file:
the histogram is computed off to the side and merged into the
metrics dict by stacker_run_metrics; if the histogram code fails
for any reason, the run record must come out BIT-IDENTICAL to the
record with no votes at all -- same accuracy, same verdict fields,
just no "votes" key. A display extra can never change a grade.

Run: .venv/bin/python -m pytest tests/test_vote_logging.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import stacker  # noqa: E402
from training import nightly  # noqa: E402

ARGS = dict(challenger_acc=0.5126, champion_acc=0.5290,
            challenger_wf_acc=0.5117, precision=0.5, folds=5,
            pending_approval=False,
            reason="stacker challenger holdout acc 0.5126 did not "
                   "beat champion 0.5290")


def _independent_stats(votes):
    """n/min/max/median and the two band shares recomputed with
    explicit counting loops (not the helper's expressions) -- the
    two computations must agree exactly."""
    v = [float(x) for x in votes]
    return {
        "n": len(v),
        "min": round(min(v), 4),
        "max": round(max(v), 4),
        "med": round(float(np.median(v)), 4),
        "mid": round(100.0 * sum(
            1 for x in v if 0.45 <= x <= 0.55) / len(v), 1),
        "tight": round(100.0 * sum(
            1 for x in v if 0.48 <= x <= 0.52) / len(v), 1),
    }


def _check_summary(stored, votes):
    """The stored summary must equal a from-the-array recompute:
    stats by explicit counting, bins by a fresh np.histogram (the
    same convention the feed test pins for the study array)."""
    stats = _independent_stats(votes)
    for key, val in stats.items():
        assert stored[key] == val, key
    bins, _ = np.histogram(np.asarray(votes, dtype=float),
                           bins=20, range=(0.0, 1.0))
    assert stored["bins"] == [int(x) for x in bins]
    assert sum(stored["bins"]) == stored["n"] == len(votes)


def test_vote_histogram_matches_independent_recompute():
    # Values sit clear of bin edges (except the exact 0.0 / 0.5 /
    # 1.0 anchors) so a hand count is unambiguous.
    votes = np.array([0.0, 0.03, 0.4499, 0.451, 0.4999, 0.5, 0.519,
                      0.549, 0.551, 0.97, 1.0])
    got = nightly.vote_histogram(votes)
    _check_summary(got, votes)
    hand = [0] * 20
    for x in votes:
        hand[min(int(float(x) * 20), 19)] += 1
    assert got["bins"] == hand


def test_grading_a_synthetic_stacker_stores_the_histogram():
    rng = np.random.default_rng(7)
    cols = [f"f{k}" for k in range(6)]
    X = pd.DataFrame(rng.normal(size=(160, 6)), columns=cols)
    y = pd.Series((X["f0"] + rng.normal(scale=0.5, size=160)) > 0)
    X_hold = pd.DataFrame(rng.normal(size=(48, 6)), columns=cols)
    pipe = stacker.build_pipeline(random_state=42)
    pipe.fit(X, y.astype(int))
    votes = pipe.predict_proba(X_hold)[:, 1]

    metrics = nightly.stacker_run_metrics(**ARGS, votes=votes)
    # Every pre-existing field rides through untouched.
    for key, val in ARGS.items():
        assert metrics[key] == val
    _check_summary(metrics["votes"], votes)
    # The stored form survives the registry's JSON round-trip and
    # stays compact (the whole point of summarizing, not dumping).
    stored = json.loads(json.dumps(metrics))
    assert stored["votes"] == metrics["votes"]
    assert len(json.dumps(metrics["votes"])) <= 300


def test_histogram_failure_never_changes_the_run_record(monkeypatch):
    votes = np.linspace(0.4, 0.6, 33)
    plain = nightly.stacker_run_metrics(**ARGS, votes=None)
    assert "votes" not in plain

    def boom(_votes):
        raise RuntimeError("histogram exploded")

    monkeypatch.setattr(nightly, "vote_histogram", boom)
    failed = nightly.stacker_run_metrics(**ARGS, votes=votes)
    # Bit-identical record, minus the histogram that never landed.
    assert json.dumps(failed, sort_keys=True) == \
        json.dumps(plain, sort_keys=True)

    monkeypatch.undo()
    ok = nightly.stacker_run_metrics(**ARGS, votes=votes)
    _check_summary(ok.pop("votes"), votes)
    assert ok == plain


def test_no_votes_and_empty_votes_store_nothing():
    assert "votes" not in nightly.stacker_run_metrics(**ARGS)
    assert "votes" not in nightly.stacker_run_metrics(
        **ARGS, votes=np.array([]))
    with pytest.raises(Exception):
        nightly.vote_histogram(np.array([]))
