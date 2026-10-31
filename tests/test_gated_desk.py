"""BOOK #3 — GATED SHADOW (forward test, Kass approved 2026-10-30;
pre-registered in refs/forward-test-gated-book3-2026-10-30.md).

Book #3 is Book #2's exact twin (Driver PPO run 44, Eyes Stacker
run 73 pick2_98, production broker, $100k flat) with ONE
difference: run 82's frozen meta-label gate vetoes ENTRY attempts
whose P(profitable) < 0.55. Pins:

- the gate seat: byte-copies of run 44 / run 73 / the run-82 gate
  model (sha256 cf7630...), hash-pinned in data/gated_seat.json;
  a tampered or missing seat file is a LOUD failure;
- the 104-vector: the live gate probability for a candle equals
  the study's own MetaGate probability on the study's own saved
  holdout arrays (frame_hold.npy / votes_hold.npy / context.npz)
  -- the live path routes through meta_label.feature_row, never
  a re-derived variant;
- veto behavior: a sub-0.55 entry is skipped AND logged in the
  gate_decisions table (the test's primary evidence); an allowed
  entry executes and is logged TAKEN; exits and position
  management (sell, cover, through-flat close) are NEVER gated
  and never logged as decisions;
- the PERIMETER: a gated seed + step leaves Book #1 and Book #2
  files bit-identical, and the gated write paths are disjoint
  from both books' paths by construction;
- idempotence + the start floor, mirroring the shadow desk;
- the feeds: snapshot identity + gate summary, the veto feed's
  rows/counts, and the 4-way race invariants.
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
GATED_SEAT = MODELS / "gated_seat"
GATED_MANIFEST = DATA / "gated_seat.json"

RUN44_PPO_SHA = "e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb"
RUN73_STK_SHA = "89b633945843062a85d71bf8a2c97c238446ce5bd4fdfb783280daa2e75364a7"
GATE82_SHA = "cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ohlcv(n: int, seed: int = 7, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq="1h")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998, "close": close,
        "volume": np.full(n, 1000.0)}, index=idx)


# ------------------------------------------------------------- the seat

def test_gated_seat_files_are_byte_copies_of_the_pinned_artifacts():
    assert _sha(GATED_SEAT / "ppo.zip") == RUN44_PPO_SHA
    assert _sha(GATED_SEAT / "stacker.pkl") == RUN73_STK_SHA
    assert _sha(GATED_SEAT / "gate_model.pkl") == GATE82_SHA
    # and the gate source itself is still the study's frozen file
    assert _sha(DATA / "meta_label" / "meta_model.pkl") == GATE82_SHA


def test_gated_manifest_pins_the_forward_test_configuration():
    man = json.loads(GATED_MANIFEST.read_text())
    assert man["book"] == "Book #3"
    assert man["driver"]["run_id"] == 44
    assert man["driver"]["sha256"] == RUN44_PPO_SHA
    assert man["eyes"]["run_id"] == 73
    assert man["eyes"]["sha256"] == RUN73_STK_SHA
    assert man["eyes"]["feature_set"] == "pick2_98"
    assert man["eyes"]["n_features"] == 98
    gate = man["gate"]
    assert gate["run_id"] == 82
    assert gate["sha256"] == GATE82_SHA
    assert gate["threshold"] == 0.55
    assert man["starting_balance"] == 100000
    assert man["book_db"] == "data/paper_gated.db"
    assert man["state_file"] == "data/live_state_gated.json"
    assert man["started"]


def test_verify_seat_rejects_a_tampered_gate_model(tmp_path):
    from trading import live_gated

    seat = tmp_path / "seat"
    seat.mkdir()
    import shutil
    for f in ("ppo.zip", "stacker.pkl", "stacker.json", "gate_model.pkl"):
        shutil.copyfile(GATED_SEAT / f, seat / f)
    man = live_gated.build_manifest("2026-10-30T00:00:00+00:00", seat)
    live_gated.verify_seat(man, seat)  # untampered: passes
    blob = bytearray((seat / "gate_model.pkl").read_bytes())
    blob[100] ^= 0xFF
    (seat / "gate_model.pkl").write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        live_gated.verify_seat(man, seat)


def test_verify_seat_missing_gate_model_is_loud(tmp_path):
    from trading import live_gated

    seat = tmp_path / "seat"
    seat.mkdir()
    import shutil
    for f in ("ppo.zip", "stacker.pkl", "stacker.json", "gate_model.pkl"):
        shutil.copyfile(GATED_SEAT / f, seat / f)
    man = live_gated.build_manifest("2026-10-30T00:00:00+00:00", seat)
    (seat / "gate_model.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        live_gated.verify_seat(man, seat)


def test_gated_write_paths_are_disjoint_from_books_1_and_2():
    from trading import live_gated

    cfg = live_gated.DEFAULT_CONFIG
    gated = {str(cfg.db_file), str(cfg.state_file),
             str(cfg.manifest_file), str(cfg.seat_dir)}
    others = {
        str(DATA / "paper.db"), str(DATA / "live_state.json"),
        str(DATA / "paper_shadow.db"),
        str(DATA / "live_state_shadow.json"),
        str(DATA / "live_seat.json"), str(DATA / "shadow_seat.json"),
        str(MODELS / "live_seat"), str(MODELS / "shadow_seat"),
    }
    assert gated.isdisjoint(others)


# ------------------------------------------- gate vector parity (real)

class _RecordingModel:
    """Stands in for the fitted meta-model: records every vector
    it is scored on and returns a fixed P(profitable)."""

    def __init__(self, p: float):
        self.p = p
        self.classes_ = np.array([0, 1])
        self.seen: list = []

    def predict_proba(self, X):
        self.seen.append(np.asarray(X, dtype=float).copy())
        return np.array([[1.0 - self.p, self.p]])


def test_live_gate_vector_is_the_study_feature_row():
    from training.meta_label import feature_row
    from trading import live_gated

    rng = np.random.default_rng(3)
    row98 = rng.normal(size=98)
    model = _RecordingModel(0.42)
    got = live_gated.gate_proba(model, row98, 0.61, 0.012, 0.03,
                                0.004, -1)
    assert got == pytest.approx(0.42)
    want = feature_row(row98.reshape(1, -1), np.array([0.61]),
                       np.array([0.012]), np.array([0.03]),
                       np.array([0.004]), 0, -1)
    assert model.seen[0].shape == (1, 104)
    np.testing.assert_array_equal(model.seen[0][0], want)
    assert model.seen[0][0][98] == 0.61
    assert model.seen[0][0][99] == abs(0.61 - 0.5)
    assert model.seen[0][0][103] == -1.0


def test_live_gate_proba_matches_the_study_on_its_own_holdout():
    """The strongest parity pin available: the study saved its
    holdout frame, votes and context arrays. The live 1-row path
    must reproduce MetaGate's probability on those exact arrays
    to the last bit, for both sides."""
    import joblib

    from training.meta_label import MetaGate
    from trading import live_gated

    lab = DATA / "meta_label"
    frame = np.load(lab / "frame_hold.npy")
    votes = np.load(lab / "votes_hold.npy")
    ctx = np.load(lab / "context.npz")
    model = joblib.load(lab / "meta_model.pkl")
    study = MetaGate(model, frame, votes, ctx["atr_rel_ho"],
                     ctx["vol21_ho"], ctx["atr_raw_ho"])
    for idx in (0, 7, 100, 400, 791):
        for side in (1, -1):
            live = live_gated.gate_proba(
                model, frame[idx], float(votes[idx]),
                float(ctx["atr_rel_ho"][idx]),
                float(ctx["vol21_ho"][idx]),
                float(ctx["atr_raw_ho"][idx]), side)
            assert live == study.proba(idx, side)


# --------------------------------------- engine: stubs + fake frames

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
    """P(profitable) scripted per predict_proba call; running off
    the end of the script is a test failure (an unexpected extra
    gate evaluation -- e.g. an exit being gated)."""

    def __init__(self, probas):
        self.probas = list(probas)
        self.classes_ = np.array([0, 1])
        self.calls = 0

    def predict_proba(self, X):
        assert self.calls < len(self.probas), \
            "gate evaluated more times than scripted"
        p = self.probas[self.calls]
        self.calls += 1
        return np.array([[1.0 - p, p]])


def _fake_scored_builder(columns):
    def build(raw):
        frame = pd.DataFrame(50.0, index=raw.index, columns=columns)
        return frame, list(columns)
    return build


def _tmp_cfg(tmp_path):
    from trading.live_gated import GatedConfig

    return GatedConfig(
        db_file=tmp_path / "paper_gated.db",
        state_file=tmp_path / "live_state_gated.json",
        seat_dir=GATED_SEAT,
        manifest_file=tmp_path / "gated_seat.json",
    )


def _seeded(tmp_path):
    """(cfg, columns, start) with the desk seeded on a 601-row
    frame whose last row is the forming candle."""
    from trading import live_gated

    meta = json.loads((GATED_SEAT / "stacker.json").read_text())
    columns = list(meta["feature_columns"])
    cfg = _tmp_cfg(tmp_path)
    start = (pd.Timestamp.now().floor("h")
             - pd.Timedelta(hours=600)).strftime("%Y-%m-%d %H:%M")
    raw = _ohlcv(601, start=start)
    out = live_gated.seed(cfg, fetch=lambda *a, **kw: raw)
    assert out["seeded"] is True
    return cfg, columns, start, raw


def _decisions(cfg):
    con = sqlite3.connect(cfg.db_file)
    try:
        return con.execute(
            "SELECT ts, side, proba, verdict FROM gate_decisions "
            "ORDER BY id").fetchall()
    finally:
        con.close()


def _trades(cfg):
    con = sqlite3.connect(cfg.db_file)
    try:
        return con.execute(
            "SELECT side, reason FROM trades ORDER BY id").fetchall()
    finally:
        con.close()


def test_vetoed_entry_is_skipped_and_logged(tmp_path):
    from trading import live_gated

    cfg, columns, start, _raw = _seeded(tmp_path)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.30])  # below 0.55: veto
    res = live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 0]), scored_builder=_fake_scored_builder(columns),
        gate_model=gate)
    assert res["processed"] == 2
    assert res["vetoes"] == 1
    assert _trades(cfg) == []                      # nothing executed
    rows = _decisions(cfg)
    assert len(rows) == 1
    ts, side, proba, verdict = rows[0]
    assert ts == raw2.index[-3].isoformat()        # first new candle
    assert side == 1 and proba == pytest.approx(0.30)
    assert verdict == "VETOED"
    state = json.loads(cfg.state_file.read_text())
    assert state["btc"] == 0.0
    assert state["last_gate_decision"]["verdict"] == "VETOED"


def test_allowed_entry_executes_and_is_logged_taken(tmp_path):
    from trading import live_gated

    cfg, columns, start, _raw = _seeded(tmp_path)
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.90])  # above 0.55: taken
    res = live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 0]), scored_builder=_fake_scored_builder(columns),
        gate_model=gate)
    assert res["vetoes"] == 0
    assert _trades(cfg) == [("BUY", "ppo_live")]
    rows = _decisions(cfg)
    assert len(rows) == 1 and rows[0][3] == "TAKEN"
    assert rows[0][2] == pytest.approx(0.90)


def test_exits_and_management_are_never_gated(tmp_path):
    from trading import live_gated

    cfg, columns, start, _raw = _seeded(tmp_path)
    # Candle 1: buy (gate consulted once, allowed). Candle 2: SELL
    # exits the long -- the gate must NOT be consulted again (the
    # script has exactly one probability) and the exit must fill.
    raw2 = _ohlcv(603, start=start)
    gate = _SequenceGateModel([0.90])
    live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 2]), scored_builder=_fake_scored_builder(columns),
        gate_model=gate)
    assert gate.calls == 1
    assert _trades(cfg) == [("BUY", "ppo_live"), ("SELL", "ppo_live")]
    assert len(_decisions(cfg)) == 1  # only the entry was a decision
    state = json.loads(cfg.state_file.read_text())
    assert state["btc"] == 0.0


def test_cover_and_through_flat_close_are_never_gated(tmp_path):
    from trading import live_gated

    cfg, columns, start, _raw = _seeded(tmp_path)
    # Short opens (gated, allowed), then a BUY while short is a
    # COVER -- management, not an entry: executes ungated. (The
    # cover is tranche-capped like every broker fill, so it may
    # cover only part of the short; the pin is that it EXECUTES
    # without a gate evaluation and shrinks the short.)
    raw_a = _ohlcv(602, start=start)   # one new closed candle
    gate = _SequenceGateModel([0.90])
    live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw_a, stacker=_StubStacker(),
        ppo=_StubPPO([3]), scored_builder=_fake_scored_builder(columns),
        gate_model=gate)
    short_btc = json.loads(cfg.state_file.read_text())["btc"]
    assert short_btc < 0
    raw_b = _ohlcv(603, start=start)   # the cover candle
    live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw_b, stacker=_StubStacker(),
        ppo=_StubPPO([1]), scored_builder=_fake_scored_builder(columns),
        gate_model=gate)
    assert gate.calls == 1  # the cover never reached the gate
    # short fill records as SELL, the cover as BUY (broker semantics)
    assert [t[0] for t in _trades(cfg)] == ["SELL", "BUY"]
    assert len(_decisions(cfg)) == 1
    state = json.loads(cfg.state_file.read_text())
    assert short_btc < state["btc"] <= 0.0


def _book12_fingerprint() -> dict:
    """Everything Book #3 must never touch: both other books'
    ledgers, states, seats, manifests, the lab champions and the
    registry -- as content hashes."""
    out = {}
    for rel in ("paper.db", "live_state.json", "paper_shadow.db",
                "live_state_shadow.json", "live_seat.json",
                "shadow_seat.json", "training.db"):
        out[rel] = _sha(DATA / rel)
    for rel in ("live_seat/ppo.zip", "live_seat/stacker.pkl",
                "live_seat/stacker.json", "shadow_seat/ppo.zip",
                "shadow_seat/stacker.pkl", "shadow_seat/stacker.json",
                "ppo_champion.zip", "stacker_champion.pkl",
                "stacker_champion.json"):
        out[rel] = _sha(MODELS / rel)
    return out


def test_gated_engine_perimeter_idempotence_and_floor(tmp_path):
    from trading import live_gated

    cfg, columns, start, raw = _seeded(tmp_path)
    anchor = raw.index[-2]
    state = json.loads(cfg.state_file.read_text())
    assert state["last_ts"] == anchor.isoformat()
    assert state["cash"] == 100000.0 and state["btc"] == 0.0
    con = sqlite3.connect(cfg.db_file)
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert con.execute("SELECT ts, equity FROM equity").fetchall() == [
        (anchor.isoformat(), 100000.0)]
    con.close()

    before = _book12_fingerprint()
    raw2 = _ohlcv(603, start=start)
    assert raw2.index[-4] == anchor
    res = live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 0]),
        scored_builder=_fake_scored_builder(columns),
        gate_model=_SequenceGateModel([0.90]))
    assert res["processed"] == 2
    assert _trades(cfg) == [("BUY", "ppo_live")]

    # Idempotence: the same frame again processes and writes nothing.
    res2 = live_gated.run_once(
        cfg, fetch=lambda *a, **kw: raw2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 0]),
        scored_builder=_fake_scored_builder(columns),
        gate_model=_SequenceGateModel([0.90]))
    assert res2["processed"] == 0
    con = sqlite3.connect(cfg.db_file)
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM equity").fetchone()[0] == 3
    assert con.execute(
        "SELECT COUNT(*) FROM gate_decisions").fetchone()[0] == 1
    con.close()

    # PERIMETER: Books #1/#2 + lab + registry bit-identical.
    assert _book12_fingerprint() == before


def test_gated_engine_refuses_to_run_unseeded(tmp_path):
    from trading import live_gated

    cfg = _tmp_cfg(tmp_path)
    raw = _ohlcv(240, start="2026-10-20")
    with pytest.raises(RuntimeError):
        live_gated.run_once(
            cfg, fetch=lambda *a, **kw: raw, stacker=_StubStacker(),
            ppo=_StubPPO([0]),
            scored_builder=_fake_scored_builder(["x"] * 98),
            gate_model=_SequenceGateModel([]))


def test_seed_refuses_to_seed_over_an_inconsistent_ledger(tmp_path):
    from trading.broker import PaperBroker
    from trading import live_gated

    cfg = _tmp_cfg(tmp_path)
    bk = PaperBroker(cfg.db_file, starting_balance=100_000)
    bk.snapshot("2026-10-30T00:00:00+00:00", 100.0)  # equity, no state
    raw = _ohlcv(601, start="2026-10-01")
    with pytest.raises(RuntimeError):
        live_gated.seed(cfg, fetch=lambda *a, **kw: raw)


# ------------------------------------------------------------ the feeds

def _fabricate_gated_world(tmp_path):
    """Book #3 fabricated: flat $100k at start, one TAKEN decision
    and one VETO, then +1%; Book #2 +2% over the window; Book #1
    mid-life $90k -> +$900; BTC +10%."""
    from trading.broker import PaperBroker
    from trading import live_gated

    b1 = tmp_path / "paper.db"
    bk = PaperBroker(b1, starting_balance=100_000)
    bk.cash, bk.btc = 90_000.0, 0.0
    bk.snapshot("2026-10-30T14:00:00+00:00", 100.0)
    bk.cash = 90_900.0
    bk.snapshot("2026-10-30T15:00:00+00:00", 110.0)

    b2 = tmp_path / "paper_shadow.db"
    bk2 = PaperBroker(b2, starting_balance=100_000)
    bk2.snapshot("2026-10-30T14:00:00+00:00", 100.0)
    bk2.cash = 102_000.0
    bk2.snapshot("2026-10-30T15:00:00+00:00", 110.0)

    b3 = tmp_path / "paper_gated.db"
    bk3 = PaperBroker(b3, starting_balance=100_000)
    bk3.snapshot("2026-10-30T14:00:00+00:00", 100.0)
    live_gated.record_gate_decision(
        b3, "2026-10-30T15:00:00+00:00", 1, 1, 0.83, "TAKEN")
    live_gated.record_gate_decision(
        b3, "2026-10-30T16:00:00+00:00", -1, 3, 0.21, "VETOED")
    bk3.cash = 101_000.0
    bk3.snapshot("2026-10-30T15:00:00+00:00", 110.0)

    state3 = tmp_path / "live_state_gated.json"
    state3.write_text(json.dumps({
        "last_ts": "2026-10-30T15:00:00+00:00", "last_stance": "hold",
        "last_p_up": 0.5123, "cash": 101000.0, "btc": 0.0,
        "entry_price": None, "last_entry_price": None,
        "entry_atr_pct": None, "high_water": None, "low_water": None,
        "stop_anchor": None, "started": "2026-10-30T14:00:00+00:00"}))
    manifest = tmp_path / "gated_seat.json"
    manifest.write_text(json.dumps({
        "book": "Book #3",
        "desk": "TradingAgent Live — Gated (Book #3 · Run 82 Gate)",
        "started": "2026-10-30T14:00:00+00:00",
        "starting_balance": 100000,
        "driver": {"brain": "PPO", "run_id": 44},
        "eyes": {"brain": "Stacker", "run_id": 73,
                 "champion": "Champion #4", "feature_set": "pick2_98",
                 "n_features": 98},
        "gate": {"brain": "Meta-Label", "run_id": 82,
                 "threshold": 0.55}}))
    return b1, b2, b3, state3, manifest


def _patch_gated_paths(monkeypatch, tmp_path):
    from src.dashboards import compact

    b1, b2, b3, state3, manifest = _fabricate_gated_world(tmp_path)
    monkeypatch.setattr(compact, "GATED_DB", b3)
    monkeypatch.setattr(compact, "GATED_STATE", state3)
    monkeypatch.setattr(compact, "GATED_MANIFEST", manifest)
    monkeypatch.setattr(compact, "GATED_RACE_B2_DB", b2)
    monkeypatch.setattr(compact, "GATED_RACE_B1_DB", b1)
    monkeypatch.setattr(compact, "SHADOW_DB", b2)
    return compact


def test_gated_snapshot_carries_identity_and_gate_summary(
        monkeypatch, tmp_path):
    compact = _patch_gated_paths(monkeypatch, tmp_path)
    s = compact.build_gated_compact_trading()
    assert s["equity"] == 101000.0 and s["position"] == "FLAT"
    d = s["desk"]
    assert d["book"] == "Book #3"
    assert d["driver"]["run"] == 44
    assert d["eyes"]["run"] == 73
    assert d["gate"]["run"] == 82
    assert d["gate"]["threshold"] == 0.55
    g = s["gate"]
    assert g["attempts"] == 2 and g["taken"] == 1 and g["vetoed"] == 1
    assert g["last"]["verdict"] == "VETOED"


def test_gated_veto_feed_rows_and_counts(monkeypatch, tmp_path):
    compact = _patch_gated_paths(monkeypatch, tmp_path)
    v = compact.build_gated_veto()
    assert v["total"] == 2 and v["taken"] == 1 and v["vetoed"] == 1
    assert v["gate"]["run"] == 82 and v["gate"]["threshold"] == 0.55
    rows = v["rows"]
    assert len(rows) == 2
    # newest first: the VETOED row was recorded last
    assert rows[0][3] == "VETOED" and rows[0][1] == -1
    assert rows[0][2] == pytest.approx(0.21)
    assert rows[1][3] == "TAKEN" and rows[1][1] == 1


def test_gated_race_alignment_invariants(monkeypatch, tmp_path):
    compact = _patch_gated_paths(monkeypatch, tmp_path)
    r = compact.build_gated_race()
    assert r["start"] == "2026-10-30T14:00:00+00:00"
    assert r["b3"]["eq"] == 101000.0 and r["b3"]["pnl"] == 1000.0
    assert r["b2"]["eq"] == 102000.0 and r["b2"]["pnl"] == 2000.0
    # Book #1 rebased to ITS equity at the race start ($90k)
    assert r["b1"]["pnl"] == 900.0
    assert r["bh"]["pnl"] == 10000.0  # BTC +10% on $100k
    assert r["delta"] == -1000.0      # b3 - b2, the primary comparison
    for key in ("b3", "b2", "b1", "bh"):
        curve = r[key]["curve"]
        assert curve and curve[0][0] >= r["start"]
