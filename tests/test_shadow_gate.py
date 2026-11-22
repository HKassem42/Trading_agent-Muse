"""BOOK #2 GATE (Kass's direct order, 2026-11-21, dashboard
Upgrades chat: "since models are different on book 3 and book 2
I want to try adding the gate on book 2 that book 3 has... the
gate to be at 51 percent on book 2 and keep book 3 the way it
is").

Book #2 (PPO run 44 + Stacker run 119 ha10 eyes) gains Book #3's
gate: run 82's frozen meta-label model (LightGBM, hash-pinned,
never retrained), judging ENTRY attempts only at a seat
threshold of 0.51 (Book #3 keeps its own 0.55; its seat, model
and manifest are untouched by this change).

The gate's 104-vector is the study's own (training.meta_label
feature_row): the 98 pick2_98 scored values at the candle (the
frame the gate was trained on -- Book #2's loop builds that
frame for the gate alone; its eyes keep reading ha10), then
[p_up, |p_up-0.5|, atr_rel, volatility_21, atr_pct_raw, side]
where p_up is Book #2's own eyes vote (run 119) -- the vote
behind the proposal being judged, exactly as Book #3's gate
reads its own eyes' vote (run 73).

Pins:
- the seat: gate_model.pkl in the shadow seat is a byte-copy of
  the frozen run-82 artifact (sha cf7630c0...), hash-pinned in
  data/shadow_seat.json next to a 0.51 threshold; a tampered or
  missing gate file is a loud failure;
- parity: the shadow gate proba for a vector equals the gated
  desk's gate proba for the same vector on the real model (one
  judge, one math -- training.meta_label.MetaGate);
- semantics (Book #3's, mirrored): entries and presses are
  judged, exits/covers/through-flat closes and stops are never
  gated; a veto substitutes HOLD before dispatch and is recorded
  in paper_shadow.db's gate_decisions with the threshold it was
  judged at (0.51); taken attempts are recorded too;
- a shadow manifest WITHOUT a gate block runs exactly as the
  book ran before 2026-11-21 (no gate table, no gate fields);
- Book #3's seat files and manifest are bit-identical after
  everything here (the change is Book #2's alone).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
MODELS = ROOT / "models"
SHADOW_SEAT = MODELS / "shadow_seat"
GATED_SEAT = MODELS / "gated_seat"
SHADOW_MANIFEST = DATA / "shadow_seat.json"
GATED_MANIFEST = DATA / "gated_seat.json"

GATE_SHA = "cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a"
RUN44_PPO_SHA = "e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb"
RUN119_STK_SHA = "63476e31f74445959de4718e5d69a71c0dae496b5c68a3d95c2cdacde498e861"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


# ------------------------------------------------------------- the seat

def test_shadow_seat_carries_the_frozen_run82_gate():
    gate = SHADOW_SEAT / "gate_model.pkl"
    assert gate.exists()
    assert _sha(gate) == GATE_SHA
    # byte-identical to Book #3's seat copy and to the study source
    assert _sha(GATED_SEAT / "gate_model.pkl") == GATE_SHA
    assert _sha(DATA / "meta_label" / "meta_model.pkl") == GATE_SHA
    cols_file = SHADOW_SEAT / "gate_columns.json"
    assert cols_file.exists()
    cols = json.loads(cols_file.read_text())["feature_columns"]
    gated_meta = json.loads((GATED_SEAT / "stacker.json").read_text())
    assert cols == list(gated_meta["feature_columns"])
    assert len(cols) == 98


def test_shadow_manifest_pins_the_gate_at_051():
    man = json.loads(SHADOW_MANIFEST.read_text())
    # driver / eyes pins unchanged by the gate addition
    assert man["driver"]["run_id"] == 44
    assert man["driver"]["sha256"] == RUN44_PPO_SHA
    assert man["eyes"]["run_id"] == 119
    assert man["eyes"]["sha256"] == RUN119_STK_SHA
    gate = man["gate"]
    assert gate["run_id"] == 82
    assert gate["threshold"] == 0.51
    assert gate["sha256"] == GATE_SHA
    assert gate["file"] == "models/shadow_seat/gate_model.pkl"
    assert "pick2_98" in gate["feature_vector"]


def test_book3_seat_and_manifest_untouched_by_the_gate_addition():
    man3 = json.loads(GATED_MANIFEST.read_text())
    assert man3["gate"]["threshold"] == 0.55
    assert man3["gate"]["sha256"] == GATE_SHA
    assert man3["driver"]["run_id"] == 99
    assert man3["eyes"]["run_id"] == 73


def test_verify_seat_rejects_a_tampered_gate_model(tmp_path):
    from trading import live_shadow

    seat = tmp_path / "seat"
    seat.mkdir()
    for name in ("ppo.zip", "stacker.pkl", "stacker.json",
                 "gate_columns.json"):
        (seat / name).write_bytes((SHADOW_SEAT / name).read_bytes())
    (seat / "gate_model.pkl").write_bytes(b"not the run 82 gate")
    man = json.loads(SHADOW_MANIFEST.read_text())
    with pytest.raises(ValueError):
        live_shadow.verify_seat(man, seat)


def test_verify_seat_accepts_the_real_shadow_seat_with_gate():
    from trading import live_shadow

    man = json.loads(SHADOW_MANIFEST.read_text())
    live_shadow.verify_seat(man, SHADOW_SEAT)  # must not raise


# ------------------------------------------------------------ parity

def test_shadow_gate_proba_matches_gated_desk_on_the_real_model():
    """One judge, one math: for the same 104-vector inputs the
    shadow wrapper and the gated desk's wrapper return the same
    probability from the real run-82 model."""
    import joblib

    from trading import live_gated, live_shadow

    model = joblib.load(SHADOW_SEAT / "gate_model.pkl")
    rng = np.random.default_rng(82)
    frame_row = rng.uniform(0.0, 100.0, 98)
    for side in (1, -1):
        a = live_shadow.gate_proba(model, frame_row, 0.62, 0.4,
                                   0.3, 0.012, side, threshold=0.51)
        b = live_gated.gate_proba(model, frame_row, 0.62, 0.4,
                                  0.3, 0.012, side, threshold=0.51)
        assert a == b
        assert 0.0 <= a <= 1.0


# ------------------------------------------------------------ engine

class _StubStacker:
    def __init__(self, p: float = 0.62):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1.0 - self.p, self.p]])


class _StubPPO:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.observation_space = type("S", (), {"shape": (25,)})()

    def predict(self, obs, deterministic=True):
        a = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        assert obs.shape == (25,)
        return np.array([a]), None


class _SequenceGateModel:
    """P(profitable) scripted per call; an unscripted extra call
    (e.g. an exit being gated) fails the test."""

    def __init__(self, probas):
        self.probas = list(probas)
        self.classes_ = np.array([0, 1])
        self.calls = 0

    def predict_proba(self, X):
        assert X.shape == (1, 104), X.shape
        assert self.calls < len(self.probas), \
            "gate evaluated more times than scripted"
        p = self.probas[self.calls]
        self.calls += 1
        return np.array([[1.0 - p, p]])


def _fake_builder(columns, value=50.0):
    def build(raw):
        frame = pd.DataFrame(value, index=raw.index, columns=columns)
        return frame, list(columns)
    return build


def _tmp_cfg(tmp_path):
    from trading.live_shadow import ShadowConfig

    return ShadowConfig(
        db_file=tmp_path / "paper_shadow.db",
        state_file=tmp_path / "live_state_shadow.json",
        seat_dir=SHADOW_SEAT,
        manifest_file=tmp_path / "shadow_seat.json",
    )


def _seeded(tmp_path, monkeypatch, with_gate=True):
    from trading import live_shadow

    monkeypatch.setattr(
        live_shadow, "SOURCE_STACKER", SHADOW_SEAT / "stacker.pkl")
    monkeypatch.setattr(
        live_shadow, "SOURCE_STACKER_META", SHADOW_SEAT / "stacker.json")
    monkeypatch.setattr(
        live_shadow, "SOURCE_PPO", SHADOW_SEAT / "ppo.zip")
    monkeypatch.setattr(
        live_shadow, "SOURCE_GATE", SHADOW_SEAT / "gate_model.pkl")
    monkeypatch.setattr(
        live_shadow, "SOURCE_GATE_META", GATED_SEAT / "stacker.json")
    meta = json.loads((SHADOW_SEAT / "stacker.json").read_text())
    eyes_columns = list(meta["feature_columns"])
    gate_meta = json.loads((GATED_SEAT / "stacker.json").read_text())
    gate_columns = list(gate_meta["feature_columns"])
    cfg = _tmp_cfg(tmp_path)
    start = (pd.Timestamp.now().floor("h")
             - pd.Timedelta(hours=600)).strftime("%Y-%m-%d %H:%M")
    raw = _ohlcv(601, start=start)
    out = live_shadow.seed(cfg, fetch=lambda *a, **kw: raw)
    assert out["seeded"] is True
    if not with_gate:
        man = json.loads(cfg.manifest_file.read_text())
        man.pop("gate", None)
        cfg.manifest_file.write_text(json.dumps(man, indent=1))
    return cfg, eyes_columns, gate_columns, start


def _run(cfg, raw, actions, gate, gate_columns, eyes_columns):
    from trading import live_shadow

    return live_shadow.run_once(
        cfg, fetch=lambda *a, **kw: raw, stacker=_StubStacker(),
        ppo=_StubPPO(actions),
        scored_builder=_fake_builder(eyes_columns),
        gate_model=gate,
        gate_frame_builder=_fake_builder(gate_columns))


def _decisions(cfg):
    con = sqlite3.connect(cfg.db_file)
    try:
        return con.execute(
            "SELECT ts, side, proba, threshold, verdict "
            "FROM gate_decisions ORDER BY id").fetchall()
    finally:
        con.close()


def _trades(cfg):
    con = sqlite3.connect(cfg.db_file)
    try:
        return con.execute(
            "SELECT side, reason FROM trades ORDER BY id").fetchall()
    finally:
        con.close()


def _jsonl_records(cfg):
    path = cfg.state_file.parent / "decisions" / "book2.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


def test_vetoed_entry_is_skipped_and_logged(tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.30])  # below 0.51: veto
    res = _run(cfg, raw2, [1, 0], gate, gate_cols, eyes_cols)
    assert res["processed"] == 2
    assert res["vetoes"] == 1
    assert _trades(cfg) == []
    rows = _decisions(cfg)
    assert len(rows) == 1
    ts, side, proba, threshold, verdict = rows[0]
    assert ts == raw2.index[-3].isoformat()
    assert side == 1 and proba == pytest.approx(0.30)
    assert threshold == 0.51
    assert verdict == "VETOED"
    state = json.loads(cfg.state_file.read_text())
    assert state["btc"] == 0.0
    assert state["last_gate_decision"]["verdict"] == "VETOED"
    assert state["last_gate_decision"]["threshold"] == 0.51
    recs = [r for r in _jsonl_records(cfg)
            if r["gate_verdict"] is not None]
    assert len(recs) == 1
    assert recs[0]["gate_verdict"] == "VETOED"
    assert recs[0]["gate_threshold"] == 0.51
    assert recs[0]["taken"] is False


def test_allowed_entry_executes_and_is_logged_taken(
        tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.90])
    res = _run(cfg, raw2, [1, 0], gate, gate_cols, eyes_cols)
    assert res["vetoes"] == 0
    assert _trades(cfg) == [("BUY", "ppo_live")]
    rows = _decisions(cfg)
    assert len(rows) == 1 and rows[0][4] == "TAKEN"
    assert rows[0][2] == pytest.approx(0.90)
    assert rows[0][3] == 0.51


def test_threshold_boundary_is_taken_at_exactly_051(
        tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.51])  # >= threshold: taken
    _run(cfg, raw2, [1, 0], gate, gate_cols, eyes_cols)
    assert _decisions(cfg)[0][4] == "TAKEN"
    assert _trades(cfg) == [("BUY", "ppo_live")]


def test_press_is_an_entry_attempt_and_is_gated(tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw2 = _ohlcv(603, start=start)
    # Candle 1: BUY from flat (taken). Candle 2: BUY again while
    # long -- a press is an entry attempt (the study's semantics):
    # the gate judges it too, and this one is vetoed.
    gate = _SequenceGateModel([0.90, 0.20])
    res = _run(cfg, raw2, [1, 1], gate, gate_cols, eyes_cols)
    assert gate.calls == 2
    assert res["vetoes"] == 1
    assert _trades(cfg) == [("BUY", "ppo_live")]  # only the first
    assert [r[4] for r in _decisions(cfg)] == ["TAKEN", "VETOED"]


def test_exits_are_never_gated(tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.90])  # exactly one evaluation
    _run(cfg, raw2, [1, 2], gate, gate_cols, eyes_cols)
    assert gate.calls == 1
    assert _trades(cfg) == [("BUY", "ppo_live"), ("SELL", "ppo_live")]
    assert len(_decisions(cfg)) == 1


def test_cover_is_never_gated(tmp_path, monkeypatch):
    cfg, eyes_cols, gate_cols, start = _seeded(tmp_path, monkeypatch)
    raw_a = _ohlcv(602, start=start)
    gate = _SequenceGateModel([0.90])
    _run(cfg, raw_a, [3], gate, gate_cols, eyes_cols)
    short_btc = json.loads(cfg.state_file.read_text())["btc"]
    assert short_btc < 0
    raw_b = _ohlcv(603, start=start)
    _run(cfg, raw_b, [1], gate, gate_cols, eyes_cols)
    assert gate.calls == 1  # the cover never reached the gate
    assert [t[0] for t in _trades(cfg)] == ["SELL", "BUY"]
    assert len(_decisions(cfg)) == 1


# ------------------------------------------------------------ feeds

def _patch_shadow_feed_paths(monkeypatch, tmp_path, rows):
    from src.dashboards import compact

    db = tmp_path / "paper_shadow.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE gate_decisions("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
        "side INTEGER NOT NULL, action INTEGER NOT NULL, "
        "proba REAL NOT NULL, threshold REAL NOT NULL, "
        "verdict TEXT NOT NULL)")
    for r in rows:
        con.execute(
            "INSERT INTO gate_decisions"
            "(ts, side, action, proba, threshold, verdict) "
            "VALUES (?,?,?,?,?,?)", r)
    con.commit()
    con.close()
    monkeypatch.setattr(compact, "SHADOW_DB", db)
    monkeypatch.setattr(compact, "SHADOW_MANIFEST", SHADOW_MANIFEST)
    return compact


def test_shadow_desk_block_and_feeds_carry_the_gate(
        monkeypatch, tmp_path):
    rows = [("2026-11-21T13:00:00", 1, 1, 0.62, 0.51, "TAKEN"),
            ("2026-11-21T14:00:00", -1, 3, 0.21, 0.51, "VETOED")]
    compact = _patch_shadow_feed_paths(monkeypatch, tmp_path, rows)
    d = compact._shadow_desk_block()
    assert d["driver"]["run"] == 44 and d["eyes"]["run"] == 119
    assert d["gate"]["run"] == 82 and d["gate"]["threshold"] == 0.51
    assert "0.51" in d["gate"]["label"]
    g = compact._shadow_gate_summary()
    assert g["attempts"] == 2 and g["taken"] == 1 and g["vetoed"] == 1
    assert g["last"]["verdict"] == "VETOED"
    assert g["threshold"] == 0.51
    v = compact.build_shadow_veto()
    assert v["total"] == 2 and v["taken"] == 1 and v["vetoed"] == 1
    assert v["gate"]["run"] == 82 and v["gate"]["threshold"] == 0.51
    # newest first
    assert v["rows"][0][3] == "VETOED" and v["rows"][0][1] == -1
    assert v["rows"][0][2] == pytest.approx(0.21)
    assert v["rows"][1][3] == "TAKEN" and v["rows"][1][1] == 1


def test_shadow_gate_feeds_are_empty_before_the_first_decision(
        monkeypatch, tmp_path):
    compact = _patch_shadow_feed_paths(monkeypatch, tmp_path, [])
    g = compact._shadow_gate_summary()
    assert g["attempts"] == 0 and g["last"] is None
    v = compact.build_shadow_veto()
    assert v["total"] == 0 and v["rows"] == []


def test_shadow_race_labels_name_the_gate_when_seated():
    """Kass's 2026-11-21 order: the gate must be named "anywhere
    else we need to know" Book #2 is gated at 51% — the race
    labels on Book #2's own desk and on the #3/#4/#5 race charts
    named Book #3's gate ("Meta-Label Run 82 gate @ 0.55") but
    not Book #2's. Every race label must name it the same way,
    derived from the seat manifest, never hard-coded."""
    from src.dashboards import compact

    desk = compact._shadow_desk_block()
    assert desk["gate"]["run"] == 82
    assert desk["gate"]["threshold"] == 0.51
    label = compact._shadow_race_label(desk)
    assert "Champion #11 Run 119 eyes" in label
    assert "Meta-Label Run 82 gate @ 0.51" in label
    for build in (compact.build_shadow_race, compact.build_gated_race,
                  compact.build_ha_race, compact.build_champ_race):
        race = build()
        assert "Meta-Label Run 82 gate @ 0.51" in race["b2"]["label"], build
        assert "Champion #11 Run 119 eyes" in race["b2"]["label"]


def test_shadow_race_label_omits_the_gate_when_unseated():
    """Pre-2026-11-21 shape: a shadow seat with no gate block
    keeps the plain label — no phantom gate text."""
    from src.dashboards import compact

    desk = {"book_name": "Book #2 — Champion #11 Shadow",
            "driver": {"run": 44},
            "eyes": {"champion": "Champion #11", "run": 119}}
    assert compact._shadow_race_label(desk) == (
        "Book #2 — Champion #11 Shadow (PPO Run 44 driver · "
        "Champion #11 Run 119 eyes)")


def test_manifest_without_gate_block_runs_ungated(
        tmp_path, monkeypatch):
    """The pre-2026-11-21 behavior, pinned: no gate block in the
    manifest -> no gate table, no gate fields, entries execute."""
    cfg, eyes_cols, gate_cols, start = _seeded(
        tmp_path, monkeypatch, with_gate=False)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([])  # any evaluation fails the test
    res = _run(cfg, raw2, [1, 0], gate, gate_cols, eyes_cols)
    assert res["processed"] == 2
    assert _trades(cfg) == [("BUY", "ppo_live")]
    con = sqlite3.connect(cfg.db_file)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    assert "gate_decisions" not in tables
    recs = _jsonl_records(cfg)
    assert recs and all(r["gate_verdict"] is None for r in recs)
