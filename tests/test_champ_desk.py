"""Book #5 (Champions desk) perimeter + engine tests.

Pins: the champ seat files are byte copies of the LAB champion
artifacts (the follows-the-crown sources); the manifest records
Muse's Book #5 configuration (seat policy, the sitting pair,
era-1 lineage); a tampered seat is refused; the eyes frame
dispatches on the seated champion's own feature_set and fails
loudly on anything else; the desk's write paths are disjoint
from Books #1-#4; the engine refuses to run unseeded; a seeded
desk processes each new candle exactly once; a re-pin with no
crown change is a verified no-op.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.technical_indicators import FEATURE_COLUMNS
from models.muse_pick10 import HA_COLUMNS
from trading import live_champ
from trading.live_champ import ChampConfig

ROOT = Path(__file__).resolve().parents[1]


def _tmp_cfg(tmp_path) -> ChampConfig:
    return ChampConfig(db_file=tmp_path / "paper_champ.db",
                       state_file=tmp_path / "live_state_champ.json",
                       seat_dir=tmp_path / "champ_seat",
                       manifest_file=tmp_path / "champ_seat.json")


def test_champ_seat_files_are_byte_copies_of_the_lab_champions(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_champ.ensure_seat(cfg.seat_dir)
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert sha(cfg.seat_dir / "stacker.pkl") == sha(
        ROOT / "models/stacker_champion.pkl")
    assert sha(cfg.seat_dir / "stacker.json") == sha(
        ROOT / "models/stacker_champion.json")
    assert sha(cfg.seat_dir / "ppo.zip") == sha(
        ROOT / "models/ppo_champion.zip")


def test_champ_manifest_pins_the_book5_configuration(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_champ.ensure_seat(cfg.seat_dir)
    man = live_champ.build_manifest("2026-11-12T00:00:00+00:00",
                                    cfg.seat_dir)
    champs = live_champ.lab_champions()
    assert man["book"] == "Book #5"
    assert man["seat_policy"] == "follows_the_crown"
    assert man["driver"]["run_id"] == champs["driver"]["run_id"]
    assert man["eyes"]["run_id"] == champs["eyes"]["run_id"]
    assert man["eyes"]["feature_set"] == champs["eyes"]["feature_set"]
    assert man["book_db"] == "data/paper_champ.db"
    assert man["lineage"][-1]["era"] == 1
    assert man["lineage"][-1]["driver_run"] == champs["driver"]["run_id"]
    assert man["lineage"][-1]["eyes_run"] == champs["eyes"]["run_id"]


def test_verify_seat_rejects_a_tampered_copy(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_champ.ensure_seat(cfg.seat_dir)
    man = live_champ.build_manifest("2026-11-12T00:00:00+00:00",
                                    cfg.seat_dir)
    live_champ.verify_seat(man, cfg.seat_dir)
    (cfg.seat_dir / "stacker.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        live_champ.verify_seat(man, cfg.seat_dir)


def test_champ_write_paths_are_disjoint_from_books_1_to_4():
    cfg = live_champ.DEFAULT_CONFIG
    others = {"paper.db", "paper_shadow.db", "paper_gated.db",
              "paper_ha.db", "live_state.json", "live_state_shadow.json",
              "live_state_gated.json", "live_state_ha.json",
              "shadow_seat.json", "gated_seat.json", "ha_seat.json",
              "live_seat", "shadow_seat", "gated_seat", "ha_seat"}
    mine = {cfg.db_file.name, cfg.state_file.name,
            cfg.manifest_file.name, cfg.seat_dir.name}
    assert not (mine & others)


def test_eyes_frame_dispatches_on_the_seated_feature_set():
    raw = pd.read_pickle(ROOT / "data/muse_pick9/raw_ohlcv.pkl")
    frame, cols = live_champ.eyes_frame(raw, "ha10")
    assert cols == HA_COLUMNS
    assert list(frame.columns) == HA_COLUMNS
    frame21, cols21 = live_champ.eyes_frame(raw, "prod21")
    assert cols21 == list(FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        live_champ.eyes_frame(raw, "scored102")


def test_eyes_vote_matches_the_direct_computation():
    raw = pd.read_pickle(ROOT / "data/muse_pick9/raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(ROOT / "data/muse_pick9/prod_sup.pkl")
    ts = prod_sup.index[-1]
    frame, cols = live_champ.eyes_frame(raw, "ha10")
    model = joblib.load(ROOT / "models/stacker_champion.pkl")
    via_engine = live_champ.eyes_vote(model, frame, ts, cols)
    direct = float(model.predict_proba(
        frame.loc[[ts], HA_COLUMNS])[0, 1])
    assert via_engine == direct
    assert 0.0 <= via_engine <= 1.0


_FRAME_START = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(
    hours=2401)


def _frame(n=2400, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    idx = pd.date_range(start=_FRAME_START, periods=n, freq="h",
                        tz="UTC")
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 10.0)}, index=idx)


class _StubStacker:
    def predict_proba(self, X):
        return np.tile(np.array([[0.4, 0.6]]), (len(X), 1))


class _StubPPO:
    class _Space:
        shape = (25,)

    observation_space = _Space()

    def predict(self, obs, deterministic=True):
        return np.array(0), None


def test_champ_engine_refuses_to_run_unseeded(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        live_champ.run_once(cfg, fetch=lambda *a, **k: _frame())


def test_champ_engine_processes_each_candle_once(tmp_path,
                                                  monkeypatch):
    # Close-mode semantics (the rollback path): since Kass's
    # 2026-11-13 order the book's default is open-decision mode,
    # whose once-only property is pinned in
    # tests/test_live_champ_open_mode.py.
    monkeypatch.setattr(live_champ, "DECISION_MODE", "close")
    cfg = _tmp_cfg(tmp_path)
    frames = {"cur": _frame(2400)}

    def fetch(symbol, interval="1h", period="5d"):
        return frames["cur"].copy()

    out = live_champ.seed(cfg, fetch=fetch)
    assert out["seeded"] is True
    frames["cur"] = _frame(2402)
    res = live_champ.run_once(cfg, fetch=fetch, stacker=_StubStacker(),
                              ppo=_StubPPO())
    assert res["processed"] == 2
    res2 = live_champ.run_once(cfg, fetch=fetch, stacker=_StubStacker(),
                               ppo=_StubPPO())
    assert res2["processed"] == 0


def test_repin_without_a_crown_change_is_a_verified_noop(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    frames = {"cur": _frame(2400)}

    def fetch(symbol, interval="1h", period="5d"):
        return frames["cur"].copy()

    live_champ.seed(cfg, fetch=fetch)
    out = live_champ.repin(cfg)
    assert out["repinned"] is False
    man = live_champ.load_manifest(cfg)
    assert len(man["lineage"]) == 1


def test_champ_indicators_feed_shape_and_ranges(monkeypatch):
    """The Book #5 indicators feed (2026-11-12 redo): ten series
    over the HA stream, positional on start/step, RSI in plain
    0-100 units, every signed production series inside its
    [-1, 1] squeeze, and "now" equal to each series' last value.
    Fetch is stubbed -- no network, deterministic frame."""
    import data.market_data as md
    from dashboards import compact

    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2026-09-01", periods=n, freq="1h", tz="UTC").tz_localize(None)
    close = 70000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    frame = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close,
        "volume": rng.uniform(50, 500, n),
    }, index=idx)
    monkeypatch.setattr(md, "fetch_klines", lambda *a, **k: frame)

    out = compact.build_champ_indicators()
    series = out["series"]
    assert set(series) == {"rsi", "macd", "ewo", "obv", "adl",
                           "netvol", "atr", "sma50", "boll", "relvol"}
    lens = {len(v) for v in series.values()}
    assert len(lens) == 1 and 48 <= lens.pop() <= 96
    assert all(0.0 <= v <= 100.0 for v in series["rsi"])
    for key in ("macd", "ewo", "obv", "adl", "netvol", "atr", "sma50"):
        assert all(-1.0 <= v <= 1.0 for v in series[key]), key
    assert all(v >= 0 for v in series["relvol"])
    for key, vals in series.items():
        assert out["now"][key] == vals[-1]
    last = pd.Timestamp(out["ts_last"])
    n_pts = len(series["rsi"])
    assert int(last.timestamp()) == out["start"] + (n_pts - 1) * out["step"]
