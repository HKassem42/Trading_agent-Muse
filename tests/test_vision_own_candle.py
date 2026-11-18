"""VISION FEEDS READ THE BOOK'S OWN CANDLE (audit 2026-11-03).

Every p_up on a book's desk must be that book's OWN recorded vote:
the value its loop wrote to its state file (last_p_up) for the
candle it last processed (last_ts). The master feed learned this
on 2026-11-03 (Book #2's headline carried Book #3's vote). The
vision builders had the same flaw one level down: they recomputed
the seat stacker's vote on the NEWEST closed candle at generation
time. Books #2/#3 carry byte-identical run-73 eyes, so when a
newer candle had closed but Book #2's loop had not processed it
yet, Book #2's vision feed showed p_up 0.4797 @21:00 -- Book #3's
recorded vote -- while Book #2's own state said 0.4712 @20:00 and
its snapshot (correctly) showed 0.4712. One desk, two votes.

Pin: each vision builder reads its book's state file; when the
state candle exists in the frame, the WHOLE panel (scores, f21
rows, ts) describes that candle and p_up is the recorded vote
verbatim. Fallback (state missing / candle not in frame) keeps
the previous newest-closed behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_shadow_desk import _ohlcv  # noqa: E402

SENTINEL_P = 0.4242  # a vote the fake model below never returns


def _fake_scored_env(monkeypatch, compact, seat_fn_name, meta_path):
    meta = json.loads(Path(meta_path).read_text())
    columns = list(meta["feature_columns"])
    raw = _ohlcv(300, start="2026-10-20")

    def fake_builder(df):
        frame = pd.DataFrame(63.2, index=df.index, columns=columns)
        return frame, list(columns)

    class _Vote:
        def predict_proba(self, X):
            return np.array([[0.4877, 0.5123]])

    monkeypatch.setattr(compact, seat_fn_name, lambda: (_Vote(), meta))
    return raw, fake_builder


def _write_state(tmp_path, ts, p_up):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(
        {"last_ts": ts, "last_p_up": p_up, "last_stance": "hold"}))
    return state


def test_shadow_vision_reads_books_own_candle(monkeypatch, tmp_path):
    from src.dashboards import compact

    raw, fake_builder = _fake_scored_env(
        monkeypatch, compact, "_shadow_seat_stacker",
        ROOT / "models" / "shadow_seat" / "stacker.json")
    own_ts = raw.index[-3]  # one candle older than newest closed
    state = _write_state(tmp_path, pd.Timestamp(own_ts).isoformat(),
                         SENTINEL_P)
    monkeypatch.setattr(compact, "SHADOW_STATE", state)

    v0 = compact.build_shadow_vision(
        0, fetch=lambda *a, **kw: raw, scored_builder=fake_builder)
    assert v0["ts"] == pd.Timestamp(own_ts).isoformat()
    assert v0["p_up"] == SENTINEL_P


def test_gated_vision_reads_books_own_candle(monkeypatch, tmp_path):
    from src.dashboards import compact

    raw, fake_builder = _fake_scored_env(
        monkeypatch, compact, "_gated_seat_stacker",
        ROOT / "models" / "gated_seat" / "stacker.json")
    own_ts = raw.index[-3]
    state = _write_state(tmp_path, pd.Timestamp(own_ts).isoformat(),
                         SENTINEL_P)
    monkeypatch.setattr(compact, "GATED_STATE", state)

    v0 = compact.build_gated_vision(
        0, fetch=lambda *a, **kw: raw, scored_builder=fake_builder)
    assert v0["ts"] == pd.Timestamp(own_ts).isoformat()
    assert v0["p_up"] == SENTINEL_P


def test_live_vision_reads_books_own_candle(monkeypatch, tmp_path):
    import data.market_data as md
    from src.dashboards import compact

    # 1500 rows: add_features' daily features need ~21 days of
    # hourly data before dropna leaves any complete row.
    raw = _ohlcv(1500, start="2024-01-01")
    monkeypatch.setattr(md, "fetch_klines", lambda *a, **kw: raw)
    # A candle safely inside the warmed feature frame, NOT the
    # newest closed one.
    own_ts = raw.index[-6]
    state = _write_state(tmp_path, pd.Timestamp(own_ts).isoformat(),
                         SENTINEL_P)

    out = compact.build_vision(state_path=state)
    assert out["ts"] == pd.Timestamp(own_ts).isoformat()
    assert out["p_up"] == SENTINEL_P
