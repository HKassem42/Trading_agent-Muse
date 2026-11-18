"""Book #4 (HA desk) perimeter + engine tests.

Pins: the HA seat files are byte copies of the pinned artifacts
(Pick #11's study artifact + run 44's PPO); the manifest records
the configuration Kass ordered; a tampered seat is refused; the
engine's vote equals the study computation; the desk's write
paths are disjoint from Books #1-#3; the engine refuses to run
unseeded; a seeded desk processes each new candle exactly once.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.muse_pick10 import HA_COLUMNS
from trading import live_ha
from trading.live_ha import HAConfig

ROOT = Path(__file__).resolve().parents[1]


def _tmp_cfg(tmp_path) -> HAConfig:
    return HAConfig(db_file=tmp_path / "paper_ha.db",
                    state_file=tmp_path / "live_state_ha.json",
                    seat_dir=tmp_path / "ha_seat",
                    manifest_file=tmp_path / "ha_seat.json")


def test_ha_seat_files_are_byte_copies_of_the_pinned_artifacts(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_ha.ensure_seat(cfg.seat_dir)
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert sha(cfg.seat_dir / "stacker.pkl") == sha(
        ROOT / "data/muse_pick11/pick11_stacker.pkl")
    assert sha(cfg.seat_dir / "stacker.json") == sha(
        ROOT / "data/muse_pick11/seat_meta.json")
    assert sha(cfg.seat_dir / "ppo.zip") == sha(
        ROOT / "models/ppo_champion.zip")


def test_ha_manifest_pins_the_book4_configuration(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_ha.ensure_seat(cfg.seat_dir)
    man = live_ha.build_manifest("2026-11-05T00:00:00+00:00",
                                 cfg.seat_dir)
    assert man["book"] == "Book #4"
    assert man["driver"]["run_id"] == 44
    assert man["eyes"]["run_id"] == 110
    assert man["eyes"]["feature_set"] == "ha10"
    assert man["eyes"]["n_features"] == 10
    assert man["eyes"]["champion"] == "Champion #6"
    assert man["book_db"] == "data/paper_ha.db"


def test_verify_seat_rejects_a_tampered_copy(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    live_ha.ensure_seat(cfg.seat_dir)
    man = live_ha.build_manifest("2026-11-05T00:00:00+00:00",
                                 cfg.seat_dir)
    live_ha.verify_seat(man, cfg.seat_dir)
    (cfg.seat_dir / "stacker.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        live_ha.verify_seat(man, cfg.seat_dir)


def test_ha_write_paths_are_disjoint_from_books_1_to_3():
    cfg = live_ha.DEFAULT_CONFIG
    others = {"paper.db", "paper_shadow.db", "paper_gated.db",
              "live_state.json", "live_state_shadow.json",
              "live_state_gated.json", "shadow_seat.json",
              "gated_seat.json", "live_seat", "shadow_seat",
              "gated_seat"}
    mine = {cfg.db_file.name, cfg.state_file.name,
            cfg.manifest_file.name, cfg.seat_dir.name}
    assert not (mine & others)


def test_ha_vote_matches_the_study_computation():
    raw = pd.read_pickle(ROOT / "data/muse_pick9/raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(ROOT / "data/muse_pick9/prod_sup.pkl")
    ts = prod_sup.index[-1]
    frame, cols = live_ha._ha_frame(raw)
    model = joblib.load(ROOT / "data/muse_pick11/pick11_stacker.pkl")
    via_engine = live_ha.ha_vote(model, frame, ts, cols)
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


def test_ha_engine_refuses_to_run_unseeded(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        live_ha.run_once(cfg, fetch=lambda *a, **k: _frame())


def test_ha_engine_processes_each_candle_once(tmp_path):
    cfg = _tmp_cfg(tmp_path)
    frames = {"cur": _frame(2400)}

    def fetch(symbol, interval="1h", period="5d"):
        return frames["cur"].copy()

    out = live_ha.seed(cfg, fetch=fetch)
    assert out["seeded"] is True
    # two more closed candles appear (+ a new forming candle)
    frames["cur"] = _frame(2402)
    res = live_ha.run_once(cfg, fetch=fetch, stacker=_StubStacker(),
                           ppo=_StubPPO())
    assert res["processed"] == 2
    res2 = live_ha.run_once(cfg, fetch=fetch, stacker=_StubStacker(),
                            ppo=_StubPPO())
    assert res2["processed"] == 0
