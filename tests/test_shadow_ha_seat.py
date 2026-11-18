"""SHADOW DESK, HA EYES (Kass's order, 2026-11-09: "promote all
the champions that won and add the highest one of our numbers to
book 2 — change out the model in book two to our highest %").

Book #2's eyes swap run 73 (Champion #4, pick2_98 scored frame)
-> run 119 (Champion #11, ha10: the 10 Heikin-Ashi candle values,
holdout 0.5530 — the highest of the four crowned winners 113 /
115 / 117 / 119). The driver stays PPO run 44. The book was
MID-SHORT at the swap and was NOT re-seeded (the 2026-10-27
precedent; the position record rides in the manifest's
eyes_swapped entry).

Pins:
- the shadow engine accepts an ha10 seat: seat_columns returns
  the 10 HA columns, unknown tags still fail loudly;
- the engine's frame builder for an ha10 seat assembles exactly
  models.muse_pick10.build_ha_matrix's HA matrix (two independent
  assemblies agree), and the vote on it is the model's own
  predict_proba on that frame;
- the shadow vision feed for an ha10 seat emits the HA payload
  (ha_rows + the book's OWN recorded vote via the state pin),
  never the scored groups;
- the desk identity block labels ha10 eyes as HA candle values,
  not scored indicators.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.muse_pick10 import HA_COLUMNS, build_ha_matrix  # noqa: E402

SHADOW_SEAT = ROOT / "models" / "shadow_seat"


def _ohlcv(n: int, seed: int = 7, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close,
        "volume": np.abs(rng.normal(100, 20, n)),
    }, index=idx)


def _ha_meta() -> dict:
    return {"run_id": 119, "champion_no": 11, "feature_set": "ha10",
            "feature_columns": list(HA_COLUMNS),
            "accuracy": 0.553030303030303}


def _tmp_seat(tmp_path, meta) -> Path:
    seat = tmp_path / "seat"
    seat.mkdir()
    (seat / "stacker.json").write_text(json.dumps(meta))
    (seat / "stacker.pkl").write_bytes(b"stub")
    (seat / "ppo.zip").write_bytes(b"stub")
    return seat


# ------------------------------------------------------- engine seat

def test_shadow_seat_columns_accept_ha10(tmp_path):
    from trading import live_shadow

    seat = _tmp_seat(tmp_path, _ha_meta())
    assert live_shadow.seat_columns(seat) == list(HA_COLUMNS)


def test_shadow_seat_columns_still_reject_unknown_frames(tmp_path):
    from trading import live_shadow

    meta = _ha_meta()
    meta["feature_set"] = "bogus_frame"
    seat = _tmp_seat(tmp_path, meta)
    with pytest.raises(ValueError):
        live_shadow.seat_columns(seat)


class _RowVote:
    """A model whose vote is a deterministic function of the row it
    is given: two assemblies of the same candle must yield the
    same vote, a wrong-frame assembly must not."""

    def predict_proba(self, X):
        row = X.iloc[0]
        p = 0.5 + 0.25 * float(row["ha_dir"]) + 0.1 * float(row["ha_slope"])
        return np.array([[1.0 - p, p]])


def test_shadow_ha_frame_builder_matches_direct_ha_matrix(tmp_path):
    from trading import live_shadow

    seat = _tmp_seat(tmp_path, _ha_meta())
    raw = _ohlcv(800)
    builder = live_shadow.seat_frame_builder(seat)
    frame, cols = builder(raw)
    assert list(cols) == list(HA_COLUMNS)
    direct = build_ha_matrix(raw)[HA_COLUMNS]
    pd.testing.assert_frame_equal(frame[HA_COLUMNS], direct)

    finite = frame[HA_COLUMNS].dropna()
    assert len(finite) > 100
    ts = finite.index[-1]
    model = _RowVote()
    got = live_shadow.champion2_vote(model, frame, ts, list(HA_COLUMNS))
    expect = float(model.predict_proba(direct.loc[[ts], list(HA_COLUMNS)])[0, 1])
    assert got == expect
    assert 0.0 <= got <= 1.0


# ------------------------------------------------------------ feeds

def test_shadow_vision_ha10_payload_is_pinned_to_own_candle(
        monkeypatch, tmp_path):
    from src.dashboards import compact

    meta = _ha_meta()
    raw = _ohlcv(400, start="2026-10-20")

    class _Vote:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    monkeypatch.setattr(compact, "_shadow_seat_stacker",
                        lambda: (_Vote(), meta))
    # The vision builder branches on the seat's own meta file:
    # point it at a fabricated ha10 seat (the swapped seat).
    seat_dir = tmp_path / "shadow_seat"
    seat_dir.mkdir()
    (seat_dir / "stacker.json").write_text(json.dumps(meta))
    monkeypatch.setattr(compact, "SHADOW_SEAT_DIR", seat_dir)
    own_ts = raw.index[-3]  # a closed candle, not the newest
    state = tmp_path / "live_state_shadow.json"
    state.write_text(json.dumps({
        "last_ts": pd.Timestamp(own_ts).isoformat(),
        "last_p_up": 0.4321, "last_stance": "hold"}))
    monkeypatch.setattr(compact, "SHADOW_STATE", state)

    v0 = compact.build_shadow_vision(0, fetch=lambda *a, **kw: raw)
    assert v0["ts"] == pd.Timestamp(own_ts).isoformat()
    assert v0["p_up"] == 0.4321  # the book's OWN recorded vote
    assert "groups" not in v0 and "master" not in v0
    assert [r[0] for r in v0["ha_rows"]] == list(HA_COLUMNS)

    v1 = compact.build_shadow_vision(1, fetch=lambda *a, **kw: raw)
    assert [r[0] for r in v1["ha_rows"]] == list(HA_COLUMNS)
    from features.technical_indicators import FEATURE_COLUMNS
    assert [r[0] for r in v1["f21"]] == FEATURE_COLUMNS


def test_shadow_desk_block_labels_ha10_eyes_as_ha_values():
    from src.dashboards import compact

    man = {"desk": "TradingAgent Live — Shadow (Champion #11 Eyes)",
           "driver": {"brain": "PPO", "run_id": 44},
           "eyes": {"brain": "Stacker", "run_id": 119,
                    "champion": "Champion #11",
                    "feature_set": "ha10", "n_features": 10},
           "started": "2026-10-26T01:00:00", "starting_balance": 100000}
    d = compact._shadow_desk_block(man)
    assert d["book_name"] == "Book #2 — Champion #11 Shadow"
    assert d["eyes"]["run"] == 119
    assert d["eyes"]["label"] == ("Eyes: Stacker Champion #11 · "
                                  "Run 119 · 10 HA candle values")
