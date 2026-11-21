"""TRAINING FEED SHIPS FULL TEXT, NOT ONLY PREVIEWS (audit 2026-11-03).

The 2026-11-03 rendered inspection of the Training Lab confirmed:
the lab-notebook notes ride in the feed as ~90-char previews ending
in "…" and the FULL text is discarded by the generator -- no field
carries it, so no expander in the client could ever recover it.
Two Kass-directive notes (the 2026-11-02 trailing-stop change,
1,218 chars, and the run-96 crowning, 795 chars) were unreadable
past ~90 chars anywhere on the dashboard. Run "why" verdicts have
the same shape (100-char cap, e.g. run 96's reason ends "0.5303…").

Contract: whenever a preview is trimmed, the feed ALSO ships the
complete text -- notes as notes[].full, run verdicts as
runs[].why_full -- byte-identical to the stored source. Untrimmed
previews need no extra field (the preview IS the full text).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_notes_ship_full_text_when_preview_trimmed():
    from src.dashboards import compact

    feed = compact.build_compact_training()
    stored = {n["title"]: n["detail"]
              for n in compact._lab_notes_raw()}
    assert feed["notes"], "expected lab notes in the feed"
    trimmed = 0
    for note in feed["notes"]:
        full_source = stored[note["h"]]
        if note["d"].endswith("…"):
            trimmed += 1
            assert "full" in note, (
                f"note {note['h']!r} preview is trimmed but the feed "
                "ships no full text")
            assert note["full"] == full_source
            assert not note["full"].endswith("…")
            # The preview is an honest prefix of the full text.
            assert note["full"].startswith(note["d"][:-1])
        else:
            assert note["d"] == full_source
    # The current notebook's two long Kass-directive notes must be
    # among the trimmed ones -- this is the regression that matters.
    assert trimmed >= 2


def test_run_why_ships_full_text_when_trimmed():
    from src.dashboards import compact

    feed = compact.build_compact_training()
    runs = {r["i"]: r for r in feed["runs"]}
    # RE-PINNED 2026-11-21: run 96 rolled out of the detail window
    # as the registry grew (the window is the newest runs and rolls
    # every night), so no single run id is pinned any more -- the
    # contract is checked for EVERY trimmed verdict in the window,
    # which is what the test is actually about.
    trimmed = [r for r in feed["runs"] if r["why"].endswith("…")]
    assert trimmed, "expected at least one trimmed run verdict"
    for r in trimmed:
        assert "why_full" in r
        assert not r["why_full"].endswith("…")
        assert r["why_full"].startswith(r["why"][:-1])
