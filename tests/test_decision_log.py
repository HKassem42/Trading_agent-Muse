"""Per-candle decision logging (Kass 2026-10-31): the loops record
what the bot was thinking on EVERY processed closed candle -- the
vote, the stance, the gate's say (Book #3), whether anything was
taken, and the exact feature values the eyes voted on.

Instrumentation only, same discipline as grading-time vote
logging: the helper is failure-isolated (a logging failure can
never raise into a trading path), idempotent per candle (loop
retries must not double-log), and the master feed carries the last
24 thoughts per book with features only on the newest one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading import decision_log as dl


def _rec(ts="2026-10-31T21:00:00+00:00", **kw):
    rec = {"ts": ts, "close": 77523.93, "p_up": 0.4954,
           "stance": "hold", "action": "HOLD", "taken": False,
           "reason": "stance_hold", "gate_proba": None,
           "gate_threshold": None, "gate_verdict": None,
           "pos_before": 0.0, "pos_after": 0.0,
           "equity": 100000.0, "stop_level": None,
           "exposure_pct": 0.0,
           "features": {"rsi_c": 0.12, "ewo": -0.34}}
    rec.update(kw)
    return rec


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()
            if x.strip()]


# ------------------------------------------------------------ helper

def test_append_round_trip(tmp_path):
    rec = _rec()
    assert dl.append_decision("book1", rec, decisions_dir=tmp_path)
    rows = _lines(tmp_path / "book1.jsonl")
    assert rows == [rec]


def test_append_dedupes_same_candle(tmp_path):
    assert dl.append_decision("book1", _rec(), decisions_dir=tmp_path)
    # A retry of the same candle must not double-log, even if the
    # record content differs slightly.
    assert not dl.append_decision(
        "book1", _rec(reason="no_fill"), decisions_dir=tmp_path)
    assert len(_lines(tmp_path / "book1.jsonl")) == 1
    # A different candle appends normally.
    assert dl.append_decision(
        "book1", _rec(ts="2026-10-31T22:00:00+00:00"),
        decisions_dir=tmp_path)
    assert len(_lines(tmp_path / "book1.jsonl")) == 2


def test_append_failure_isolated(tmp_path):
    # decisions_dir is an existing FILE: mkdir/open must fail and
    # the helper must swallow it (never raise into a trading path).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    assert not dl.append_decision("book1", _rec(),
                                  decisions_dir=blocker)
    # An unserializable record value likewise: no raise, no
    # partial line left behind.
    bad = _rec(features={"x": object()})
    assert not dl.append_decision("book2", bad,
                                  decisions_dir=tmp_path)
    p = tmp_path / "book2.jsonl"
    assert not p.exists() or _lines(p) == []


def test_describe_outcome_variants():
    fill = lambda side, reason: SimpleNamespace(side=side,
                                                reason=reason)
    # Quiet candle: stance hold, nothing happened.
    assert dl.describe_outcome(0, None, None) == \
        ("HOLD", False, "stance_hold")
    # PPO buy that filled.
    assert dl.describe_outcome(1, None, fill("BUY", "ppo_live")) == \
        ("BUY", True, "ppo_live")
    # Buy dispatch that covered a short is a COVER.
    assert dl.describe_outcome(1, None, fill("BUY", "cover")) == \
        ("COVER", True, "cover")
    # Short open.
    assert dl.describe_outcome(3, None, fill("SELL", "ppo_live")) == \
        ("SHORT", True, "ppo_live")
    # Stop fired on a hold candle: the exit is the outcome.
    assert dl.describe_outcome(
        0, fill("SELL", "trailing_stop"), None) == \
        ("SELL", True, "trailing_stop")
    # Attempt the broker declined (cap / flat): not taken.
    assert dl.describe_outcome(1, None, None) == \
        ("BUY", False, "no_fill")
    # Gate veto: wanted to buy, gate said no.
    assert dl.describe_outcome(1, None, None,
                               gate_verdict="VETOED") == \
        ("BUY", False, "gate_veto")


# --------------------------------------------------------- feed side

def test_master_decision_block_carries_log_and_latest(tmp_path,
                                                      monkeypatch):
    from dashboards import compact

    d = tmp_path / "decisions"
    d.mkdir()
    rows = []
    for i in range(3):
        rec = _rec(ts=f"2026-10-31T{19 + i:02d}:00:00+00:00",
                   p_up=0.4 + i / 10.0,
                   features={"f0": float(i)})
        rows.append(rec)
    (d / "book1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(compact, "DECISIONS_DIR", d)
    log, latest = compact._decision_log_block("book1")
    assert [r["ts"] for r in log] == [r["ts"] for r in rows]
    assert all("features" not in r for r in log)
    assert latest["ts"] == rows[-1]["ts"]
    assert latest["features"] == {"f0": 2.0}
    assert log[-1]["p_up"] == rows[-1]["p_up"]


def test_master_decision_block_caps_at_24(tmp_path, monkeypatch):
    from dashboards import compact

    d = tmp_path / "decisions"
    d.mkdir()
    rows = [_rec(ts=f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00")
            for i in range(30)]
    (d / "book2.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(compact, "DECISIONS_DIR", d)
    log, latest = compact._decision_log_block("book2")
    assert len(log) == 24
    assert log[0]["ts"] == rows[6]["ts"]
    assert latest["ts"] == rows[-1]["ts"]


def test_master_decision_block_no_file_is_empty(tmp_path,
                                                 monkeypatch):
    from dashboards import compact

    monkeypatch.setattr(compact, "DECISIONS_DIR",
                        tmp_path / "decisions")
    assert compact._decision_log_block("book3") == ([], None)
