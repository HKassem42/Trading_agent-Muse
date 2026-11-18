"""SHADOW DESK (Kass directive, 2026-10-25: the seated champion
gets "his own trading dashboard so we don't have to wait to plug
that in and test" + "make sure to label everything properly").

A parallel, fully independent LIVE paper book for the pairing
Driver: PPO run 44 - Eyes: Stacker Champion #11 (run 119, the
HA nightly stacker, ha10) -- so it builds a live track record
without touching Book #1 (the driver's seat: PPO 44 + stacker
run 40 prod21). The desk was seeded 2026-10-25 with Champion
#2's eyes (run 64 -- the pairing graded +$461.81 as the champion
side of the scored-eyes study); on 2026-10-26 Kass crowned run
69 Champion #3 and seated it here, superseding the run-64 pin
(the book was flat, $100,000, 0 fills, at that swap). On
2026-10-27 Muse crowned run 73 Champion #4 under Kass's
delegation and swapped the eyes 69 -> 73 MID-POSITION (Book #2
was short -0.311631 BTC @ $80,222.97, opened under run 69's
vote). On 2026-11-09 Kass ordered the four pending HA winners
crowned and the highest seated here: run 119 (Champion #11)
replaced run 73 MID-POSITION (short -0.342743 BTC @ $74,364.07),
and the eyes' frame changed with it (pick2_98 -> ha10); the
swaps and the position records ride in the manifest's
eyes_swapped entry.

Pins:
- the shadow seat: byte-copies of run 44's PPO and run 119's
  stacker, hash-pinned in data/shadow_seat.json, PINNED to run
  119 (the seat does not follow future lab crownings); a tampered
  or missing seat file is a loud failure, never a silent fallback;
- the shadow vote: the engine's vote for a candle equals the
  champion's own predict_proba on its own frame assembled the
  study's way (two independent assemblies must agree);
- the PERIMETER: a shadow run writes ONLY the shadow DB/state --
  production paper.db rows, live_state.json, the live seat, the lab
  champions and training.db are bit-identical after it;
- idempotence + the start floor: each closed candle is processed
  exactly once, and candles at or before the seed anchor are never
  backfilled into trades;
- the feeds: snapshot/ledger/market shapes mirror the Live Desk
  contracts; the race feed aligns both books + buy-and-hold to the
  shadow start with exact invariants; the vision feed carries the
  seat's own frame rows for the book's own last candle (the HA
  rows since Champion #11; the scored-groups contract of the
  retired seats is pinned in git history).
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

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
SHADOW_SEAT = MODELS / "shadow_seat"
SHADOW_MANIFEST = DATA / "shadow_seat.json"

RUN44_PPO_SHA = "e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb"
RUN119_STK_SHA = "63476e31f74445959de4718e5d69a71c0dae496b5c68a3d95c2cdacde498e861"
RUN119_META_SHA = "3a5394382babf89d05e120d23482fdf352fb7d2dec0cc62fd78f4ddee4e0bbdb"
# The champion meta carries the crowning timestamp, so its sha
# changes at every crowning; the pins that matter are the seat
# copies' hashes above plus the meta's content (run 119 /
# Champion #11 / ha10 / 10 columns, asserted in the manifest
# test).


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

def test_shadow_seat_files_are_byte_copies_of_the_pinned_runs():
    assert (SHADOW_SEAT / "ppo.zip").exists()
    assert (SHADOW_SEAT / "stacker.pkl").exists()
    assert (SHADOW_SEAT / "stacker.json").exists()
    # The seat pins run 44's PPO and run 119's stacker BY HASH.
    # The lab champion files are a separate record: the lab PPO
    # crown moved to run 112 on 2026-11-06 and the lab stacker
    # crown will move again at future crownings; the seat must
    # NOT follow either (the manifest's own "pinned / not
    # follow" rule). Today the seat stacker happens to be
    # byte-equal to the lab champion file because Kass seated
    # the sitting champion itself — a fact, not a coupling.
    assert _sha(SHADOW_SEAT / "ppo.zip") == RUN44_PPO_SHA
    assert _sha(SHADOW_SEAT / "stacker.pkl") == RUN119_STK_SHA
    assert _sha(SHADOW_SEAT / "stacker.json") == RUN119_META_SHA
    meta = json.loads((SHADOW_SEAT / "stacker.json").read_text())
    assert meta["run_id"] == 119 and meta["champion_no"] == 11
    assert meta["feature_set"] == "ha10"
    assert len(meta["feature_columns"]) == 10


def test_shadow_manifest_pins_run119_and_records_the_configuration():
    man = json.loads(SHADOW_MANIFEST.read_text())
    assert man["driver"]["run_id"] == 44
    assert man["driver"]["sha256"] == RUN44_PPO_SHA
    assert man["eyes"]["run_id"] == 119
    assert man["eyes"]["champion"] == "Champion #11"
    assert man["eyes"]["name"] == "HA Nightly Stacker"
    assert man["eyes"]["feature_set"] == "ha10"
    assert man["eyes"]["n_features"] == 10
    assert man["eyes"]["sha256"] == RUN119_STK_SHA
    assert man["eyes"]["meta_sha256"] == RUN119_META_SHA
    # Pinned: the seat must NOT follow future lab crownings.
    assert "119" in man["pinned"] and "not follow" in man["pinned"].lower()
    # The swap history is on the record: 64 -> 69 (Kass, flat
    # book), 69 -> 73 (Muse's ruling under Kass's 2026-10-27
    # delegation, MID-POSITION), then 73 -> 119 (Kass's order
    # 2026-11-09, MID-POSITION: the open short was opened under
    # run 73's votes and the record says so).
    hist = {(h["from_run"], h["to_run"])
            for h in man["eyes_swap_history"]}
    assert hist == {(64, 69), (69, 73), (73, 119)}
    assert man["eyes_swapped"]["from_run"] == 73
    assert man["eyes_swapped"]["to_run"] == 119
    assert "SHORT" in man["eyes_swapped"]["note"]
    assert "74,364.07" in man["eyes_swapped"]["note"]
    # The graded exam is attributed to the pairing it graded
    # (44 + run 69, from the Pick #2 duel record), not to the
    # current one, and it CARRIES its values on the manifest.
    assert man["wfe_exam"]["driver_run"] == 44
    assert man["wfe_exam"]["eyes_run"] == 69
    assert man["wfe_exam"]["pnl"] == 162.12
    assert man["wfe_exam"]["pct"] == 0.1621
    assert man["wfe_exam"]["days"] == 33
    assert "162.12" in man["wfe_exam"]["source"]
    assert man["starting_balance"] == 100000
    assert man["book_db"].endswith("paper_shadow.db")
    assert man["state_file"].endswith("live_state_shadow.json")
    assert man["started"]  # the desk start (seed anchor candle)


def test_verify_seat_rejects_a_tampered_copy(tmp_path):
    from trading import live_shadow

    seat = tmp_path / "seat"
    seat.mkdir()
    (seat / "ppo.zip").write_bytes(b"not the run 44 brain")
    (seat / "stacker.pkl").write_bytes(
        (SHADOW_SEAT / "stacker.pkl").read_bytes())
    (seat / "stacker.json").write_bytes(
        (SHADOW_SEAT / "stacker.json").read_bytes())
    man = json.loads(SHADOW_MANIFEST.read_text())
    with pytest.raises(ValueError):
        live_shadow.verify_seat(man, seat)


def test_verify_seat_accepts_the_real_shadow_seat():
    from trading import live_shadow

    man = json.loads(SHADOW_MANIFEST.read_text())
    live_shadow.verify_seat(man, SHADOW_SEAT)  # must not raise


# ------------------------------------------------- vote parity (2 ways)

def test_shadow_vote_matches_champion_computed_the_study_way():
    """The engine's vote for a candle == the champion's predict_proba on
    its own frame restricted to its recorded columns -- computed
    here independently of the engine's own assembly path. Since
    Champion #11 (run 119, 2026-11-09) the study's way is the
    HA candle matrix: models.muse_pick10.build_ha_matrix (the
    frame Pick #11 was graded on and nightly grades the HA
    champions with)."""
    import joblib

    from models.muse_pick10 import HA_COLUMNS, build_ha_matrix
    from trading import live_shadow

    df = _ohlcv(7000)
    frame = build_ha_matrix(df)[HA_COLUMNS]
    meta = json.loads((SHADOW_SEAT / "stacker.json").read_text())
    want = list(meta["feature_columns"])
    assert list(want) == list(HA_COLUMNS) and len(want) == 10
    finite = frame[want].dropna()
    assert len(finite) > 100
    ts = finite.index[-1]
    # Way 1 (the study's): direct predict on the restricted frame.
    # The reference artifact is run 119's pinned model — the seat
    # copy IS run 119's artifact (sha-pinned RUN119_STK_SHA
    # above); the lab champion file is a separate record that
    # future crownings will move.
    champ = joblib.load(SHADOW_SEAT / "stacker.pkl")
    expect = float(champ.predict_proba(frame.loc[[ts], want])[0, 1])
    # Way 2: the engine's vote — its own seat frame builder's
    # assembly, on the SHADOW seat copy.
    eng_frame, eng_cols = live_shadow.seat_frame_builder(
        SHADOW_SEAT)(df)
    assert list(eng_cols) == want
    seat_model = joblib.load(SHADOW_SEAT / "stacker.pkl")
    got = live_shadow.champion2_vote(seat_model, eng_frame, ts, want)
    assert got == expect
    assert 0.0 <= got <= 1.0


# --------------------------------------- engine: perimeter + semantics

class _StubStacker:
    def __init__(self, p: float = 0.62):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1.0 - self.p, self.p]])


class _StubPPO:
    """Scripted actions by call order; obs dim mirrors run 44 (25)."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.observation_space = type("S", (), {"shape": (25,)})()

    def predict(self, obs, deterministic=True):
        a = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        assert obs.shape == (25,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        return np.array([a]), None


def _fake_scored_builder(columns):
    """Fast stand-in for build_scored_frame: the champion's columns
    filled with in-range percentiles on every row (the engine tests
    exercise plumbing, not indicator math -- the real builder is
    parity-tested above and causality-pinned in
    test_seat_separation.py)."""
    def build(raw):
        frame = pd.DataFrame(50.0, index=raw.index, columns=columns)
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


def _production_fingerprint() -> dict:
    """Everything the shadow must never touch, as content (not just
    mtimes): production ledger rows + state, and the artifact hashes."""
    con = sqlite3.connect(DATA / "paper.db")
    trades = con.execute("SELECT COUNT(*), MAX(id) FROM trades").fetchone()
    equity = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
    con.close()
    return {
        "trades": trades, "equity_rows": equity,
        "live_state": (DATA / "live_state.json").read_bytes(),
        "paper_db_sha": _sha(DATA / "paper.db"),
        "training_db_sha": _sha(DATA / "training.db"),
        "seat_ppo": _sha(MODELS / "live_seat" / "ppo.zip"),
        "seat_stk": _sha(MODELS / "live_seat" / "stacker.pkl"),
        "lab_stk": _sha(MODELS / "stacker_champion.pkl"),
        "lab_ppo": _sha(MODELS / "ppo_champion.zip"),
    }


def test_shadow_engine_perimeter_idempotence_and_floor(
        tmp_path, monkeypatch):
    from trading import live_shadow

    # ensure_seat verifies the seat against the engine's canonical
    # sources, which pointed at the lab champion files while the
    # seat's eyes WERE the lab champion. Since Kass's 2026-11-02
    # LAB-ONLY crowning of run 96, the lab files are a different
    # brain and the seat deliberately diverges — and since the
    # 2026-11-06 crowning of run 112 the lab PPO file diverged
    # too; pin the sources to the seat's own pinned artifacts for
    # this fixture (the hash pins themselves live in the seat
    # tests above).
    monkeypatch.setattr(
        live_shadow, "SOURCE_STACKER", SHADOW_SEAT / "stacker.pkl")
    monkeypatch.setattr(
        live_shadow, "SOURCE_STACKER_META", SHADOW_SEAT / "stacker.json")
    monkeypatch.setattr(
        live_shadow, "SOURCE_PPO", SHADOW_SEAT / "ppo.zip")

    meta = json.loads((SHADOW_SEAT / "stacker.json").read_text())
    columns = list(meta["feature_columns"])
    cfg = _tmp_cfg(tmp_path)
    # 25 days ending at the current hour: past every prod21 warmup
    # (the daily features need ~20 days) so the new candles carry
    # complete feature rows, and fresh enough that the engine's
    # staleness alarm (2.5h, production behavior) stays quiet.
    start = (pd.Timestamp.now().floor("h")
             - pd.Timedelta(hours=600)).strftime("%Y-%m-%d %H:%M")
    raw = _ohlcv(601, start=start)  # last row = the forming candle
    fetch = lambda *a, **kw: raw  # noqa: E731 - same frame for both periods

    # Seed: anchor = newest CLOSED candle (last row is still forming).
    anchor = raw.index[-2]
    out = live_shadow.seed(cfg, fetch=fetch)
    assert out["started"] == anchor.isoformat()
    state = json.loads(cfg.state_file.read_text())
    assert state["last_ts"] == anchor.isoformat()
    assert state["cash"] == 100000.0 and state["btc"] == 0.0
    con = sqlite3.connect(cfg.db_file)
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    eq = con.execute("SELECT ts, equity FROM equity").fetchall()
    assert eq == [(anchor.isoformat(), 100000.0)]
    con.close()

    before = _production_fingerprint()

    # Two new closed candles arrive (frame grows by 2; the last row is
    # the new forming candle). Actions: buy on the first, hold after.
    raw2 = _ohlcv(603, start=start)
    assert raw2.index[-4] == anchor  # same history, extended tail
    fetch2 = lambda *a, **kw: raw2  # noqa: E731
    ppo = _StubPPO([1, 0])
    res = live_shadow.run_once(
        cfg, fetch=fetch2, stacker=_StubStacker(),
        ppo=ppo, scored_builder=_fake_scored_builder(columns))
    assert res["processed"] == 2
    con = sqlite3.connect(cfg.db_file)
    fills = con.execute(
        "SELECT side, reason FROM trades ORDER BY id").fetchall()
    assert fills == [("BUY", "ppo_live")]
    eq_rows = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
    assert eq_rows == 3  # anchor + 2 processed candles
    con.close()
    state = json.loads(cfg.state_file.read_text())
    assert state["last_ts"] == raw2.index[-2].isoformat()
    assert state["btc"] > 0  # the scripted buy opened a long

    # Idempotence: the same frame again processes nothing, writes
    # nothing.
    res2 = live_shadow.run_once(
        cfg, fetch=fetch2, stacker=_StubStacker(),
        ppo=_StubPPO([1, 0]),
        scored_builder=_fake_scored_builder(columns))
    assert res2["processed"] == 0
    con = sqlite3.connect(cfg.db_file)
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM equity").fetchone()[0] == 3
    con.close()

    # PERIMETER: production is bit-identical after all of it.
    assert _production_fingerprint() == before


def test_shadow_engine_refuses_to_run_unseeded(tmp_path):
    from trading import live_shadow

    cfg = _tmp_cfg(tmp_path)
    raw = _ohlcv(240, start="2026-10-20")
    with pytest.raises(RuntimeError):
        live_shadow.run_once(cfg, fetch=lambda *a, **kw: raw,
                             stacker=_StubStacker(), ppo=_StubPPO([0]),
                             scored_builder=_fake_scored_builder(
                                 FEATURE_COLUMNS))


# ------------------------------------------------------------ the feeds

def _fabricate_books(tmp_path):
    """A tiny two-book world: book #2 flat $100k at the start, then
    +2%; book #1 at $90k at the start (mid-life), then +$900; BTC
    +10% across the window. Returns (b1_db, b2_db, state2, manifest)."""
    from trading.broker import PaperBroker

    b1 = tmp_path / "paper.db"
    bk = PaperBroker(b1, starting_balance=100_000)
    bk.cash, bk.btc = 90_000.0, 0.0
    bk.snapshot("2026-10-26T00:00:00", 100.0)
    bk.cash = 90_900.0
    bk.snapshot("2026-10-26T01:00:00", 110.0)

    b2 = tmp_path / "paper_shadow.db"
    bk2 = PaperBroker(b2, starting_balance=100_000)
    bk2.snapshot("2026-10-26T00:00:00", 100.0)
    bk2.cash = 102_000.0
    bk2.snapshot("2026-10-26T01:00:00", 110.0)

    state2 = tmp_path / "live_state_shadow.json"
    state2.write_text(json.dumps({
        "last_ts": "2026-10-26T01:00:00", "last_stance": "hold",
        "last_p_up": 0.5123, "cash": 102000.0, "btc": 0.0,
        "entry_price": None, "last_entry_price": None,
        "entry_atr_pct": None, "high_water": None, "low_water": None,
        "stop_anchor": None, "started": "2026-10-26T00:00:00"}))
    manifest = tmp_path / "shadow_seat.json"
    manifest.write_text(json.dumps({
        "started": "2026-10-26T00:00:00", "starting_balance": 100000,
        "driver": {"brain": "PPO", "run_id": 44},
        "eyes": {"brain": "Stacker", "run_id": 64,
                 "champion": "Champion #2", "feature_set": "scored102",
                 "n_features": 94}}))
    return b1, b2, state2, manifest


def _patch_shadow_paths(monkeypatch, tmp_path):
    from src.dashboards import compact

    b1, b2, state2, manifest = _fabricate_books(tmp_path)
    monkeypatch.setattr(compact, "SHADOW_DB", b2)
    monkeypatch.setattr(compact, "SHADOW_STATE", state2)
    monkeypatch.setattr(compact, "SHADOW_MANIFEST", manifest)
    monkeypatch.setattr(compact, "SHADOW_RACE_B1_DB", b1)
    return compact


def test_shadow_snapshot_carries_the_desk_identity(monkeypatch, tmp_path):
    compact = _patch_shadow_paths(monkeypatch, tmp_path)
    s = compact.build_shadow_compact_trading()
    assert s["equity"] == 102000.0 and s["position"] == "FLAT"
    d = s["desk"]
    assert d["book"] == "Book #2"
    assert d["driver"]["run"] == 44
    assert d["eyes"]["run"] == 64
    assert d["eyes"]["n_features"] == 94
    assert d["started"] == "2026-10-26T00:00:00"


def test_shadow_race_alignment_invariants(monkeypatch, tmp_path):
    compact = _patch_shadow_paths(monkeypatch, tmp_path)
    r = compact.build_shadow_race()
    assert r["start"] == "2026-10-26T00:00:00"
    # All three series share the race timeline and start at the start.
    n = len(r["b2"]["curve"])
    assert n == len(r["b1"]["curve"]) == len(r["bh"]["curve"]) == 2
    assert r["b2"]["curve"][0] == ["2026-10-26T00:00:00", 100000.0]
    assert r["bh"]["curve"][0] == ["2026-10-26T00:00:00", 100000.0]
    # Buy-and-hold: BTC 100 -> 110 is +10% on $100k.
    assert r["bh"]["eq"] == 110000.0 and r["bh"]["pnl"] == 10000.0
    # Book #2 raced from its own $100k; book #1 from ITS value at the
    # race start ($90k), not from $100k -- a fair race.
    assert r["b2"]["pnl"] == 2000.0
    assert r["b1"]["pnl"] == 900.0
    assert r["delta"] == 1100.0  # b2 pnl - b1 pnl over the same window
    assert r["b1"]["label"].startswith("Book #1")
    assert r["b2"]["label"].startswith("Book #2")


def test_shadow_vision_ha_rows_match_the_seat_columns(
        monkeypatch, tmp_path):
    """Since Champion #11 (ha10) took the seat, the shadow vision
    feed is the HA payload: both parts carry the seat's 10 HA
    rows in the champion's own column order, part 0 headlines
    the vote computed on those columns, and part 1 rides the
    driver's production 21 along for continuity. (The scored
    groups partition this test pinned for Champions #2–#4 is
    retired with those seats.)"""
    from models.muse_pick10 import HA_COLUMNS
    from src.dashboards import compact

    meta = json.loads((SHADOW_SEAT / "stacker.json").read_text())
    columns = list(meta["feature_columns"])
    assert columns == list(HA_COLUMNS)
    raw = _ohlcv(300, start="2026-10-20")

    seen = {}

    class _Vote:
        def predict_proba(self, X):
            seen["cols"] = list(X.columns)
            return np.array([[0.4877, 0.5123]])

    monkeypatch.setattr(compact, "_shadow_seat_stacker",
                        lambda: (_Vote(), meta))
    v0 = compact.build_shadow_vision(0, fetch=lambda *a, **kw: raw)
    v1 = compact.build_shadow_vision(1, fetch=lambda *a, **kw: raw)
    # The book's own state pin names a real-market candle that is
    # not in this synthetic frame, so the vote is the model's own
    # computation on the newest complete synthetic candle.
    assert v0["p_up"] == 0.5123
    assert seen["cols"] == columns  # the champion's own column order
    assert "groups" not in v0 and "master" not in v0
    for part in (v0, v1):
        assert [r[0] for r in part["ha_rows"]] == columns
        for _name, _label, val in part["ha_rows"]:
            assert isinstance(val, float)
    # The raw production 21 ride along in part 1 for continuity.
    assert [r[0] for r in v1["f21"]] == FEATURE_COLUMNS


def test_shadow_labels_follow_the_seat_manifest(monkeypatch, tmp_path):
    """Every identity label on the shadow desk derives from the
    seat manifest: after Kass's 2026-10-26 seat swap (Champion #3
    eyes, run 69), the desk block and the race label must name the
    NEW pairing -- no Champion #2 / run 64 text may survive in the
    feed's identity fields."""
    compact = _patch_shadow_paths(monkeypatch, tmp_path)
    man = json.loads(Path(compact.SHADOW_MANIFEST).read_text())
    man["eyes"] = {"brain": "Stacker", "run_id": 69,
                   "champion": "Champion #3",
                   "feature_set": "scored102", "n_features": 94}
    Path(compact.SHADOW_MANIFEST).write_text(json.dumps(man))
    d = compact._shadow_desk_block()
    assert d["book_name"] == "Book #2 — Champion #3 Shadow"
    assert d["desk"] == "TradingAgent Live — Champion #3 (Shadow)"
    assert d["eyes"]["run"] == 69
    assert d["eyes"]["label"] == ("Eyes: Stacker Champion #3 · "
                                  "Run 69 · 94 scored indicators")
    race = compact.build_shadow_race()
    assert "Champion #3 Run 69 eyes" in race["b2"]["label"]
    assert "Champion #2" not in race["b2"]["label"]
    assert race["desk"]["book_name"] == "Book #2 — Champion #3 Shadow"
