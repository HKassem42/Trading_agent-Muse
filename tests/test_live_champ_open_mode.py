"""Book #5 open-decision mode (Kass's order, 2026-11-13).

The Champions book decided on hourly OPENS from MODE_FROM_TS:
at each boundary the pair reads the PREVIOUS candle's features
and fills at the new candle's open; protective stops still run
on each candle at its close, before the next open decision.
These tests pin the semantics adversarially -- causality (the
decision candle's own data must not leak into its decision),
fill price (the open, exactly), ordering (stop before the next
decide), the transition (pre-mode candles get no decision), and
idempotence (a second wake in the same hour decides nothing).

The experiment ended on Kass's order 2026-11-17: the production
default is DECISION_MODE='close' again (the rollback was taken),
so the open-mode tests below pin DECISION_MODE='open'
EXPLICITLY via monkeypatch -- they no longer ride the module
default. The close-mode rollback test pins 'close' the same
way.
"""

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.technical_indicators import FEATURE_COLUMNS, add_features
from src.trading import decision_log as dl
from src.trading import live_champ
from src.trading.live_champ import ChampConfig, run_once


def _frame(n=1200, seed=7):
    """Synthetic hourly candles ending at the current hour (the
    last row is the forming candle), so the engine's stale-data
    guard sees a live-shaped feed."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    idx = pd.date_range(end=end, periods=n, freq="1h")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = rng.uniform(10, 20, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": vol}, index=idx)


class _StubStacker:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])


class _StubPPO:
    class _Space:
        shape = (25,)

    observation_space = _Space()

    def __init__(self, actions):
        self._actions = list(actions)
        self.obs_seen = []

    def predict(self, obs, deterministic=True):
        self.obs_seen.append(np.array(obs, dtype=np.float64))
        action = self._actions.pop(0) if self._actions else 0
        return np.array(action), None


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cfg(tmp_path, last_ts):
    seat = tmp_path / "seat"
    seat.mkdir()
    (seat / "stacker.pkl").write_bytes(b"stub-stacker")
    (seat / "ppo.zip").write_bytes(b"stub-ppo")
    meta = {"run_id": 126, "champion_no": 13,
            "feature_set": "prod21",
            "feature_columns": list(FEATURE_COLUMNS)}
    (seat / "stacker.json").write_text(json.dumps(meta))
    manifest = {
        "driver": {"run_id": 112, "sha256": _sha(seat / "ppo.zip")},
        "eyes": {"run_id": 126, "champion": "Champion #13",
                 "sha256": _sha(seat / "stacker.pkl"),
                 "meta_sha256": _sha(seat / "stacker.json")},
    }
    man_file = tmp_path / "champ_seat.json"
    man_file.write_text(json.dumps(manifest))
    state = {"last_ts": pd.Timestamp(last_ts).isoformat(),
             "cash": 100000.0, "btc": 0.0, "entry_price": None,
             "last_entry_price": None, "entry_atr_pct": None,
             "high_water": None, "low_water": None,
             "stop_anchor": None, "last_stance": None,
             "last_p_up": None}
    state_file = tmp_path / "live_state_champ.json"
    state_file.write_text(json.dumps(state))
    return ChampConfig(db_file=tmp_path / "paper_champ.db",
                       state_file=state_file, seat_dir=seat,
                       manifest_file=man_file)


def _trades(cfg):
    con = sqlite3.connect(cfg.db_file)
    try:
        return con.execute(
            "SELECT ts, side, price, qty, reason FROM trades "
            "ORDER BY id").fetchall()
    finally:
        con.close()


def _decisions(cfg):
    path = cfg.state_file.parent / "decisions" / "book5.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)
            for line in path.read_text().splitlines() if line.strip()]


def _setup(tmp_path, monkeypatch, actions, mutate=None):
    """Frame landmarks: C0 C1 C2 F -- F forming, C2 newest
    closed. MODE_FROM = C2, state last_ts = C0."""
    frame = _frame()
    if mutate:
        frame = mutate(frame)
    c0, c1, c2, f = (frame.index[-4], frame.index[-3],
                     frame.index[-2], frame.index[-1])
    monkeypatch.setattr(live_champ, "DECISION_MODE", "open")
    monkeypatch.setattr(live_champ, "MODE_FROM_TS", c2.isoformat())
    cfg = _cfg(tmp_path, c0)
    ppo = _StubPPO(actions)
    out = run_once(cfg, fetch=lambda *a, **k: frame,
                   stacker=_StubStacker(), ppo=ppo)
    return cfg, frame, (c0, c1, c2, f), ppo, out


def test_decide_fills_at_open_with_prev_candle_features(
        tmp_path, monkeypatch):
    """DECIDE(C2) must fill at open[C2] and its observation must
    be built from C1's features -- C2's own (spiked) row must
    not leak in. DECIDE(F) then fires on the forming candle."""
    def spike(frame):
        # +1%: large enough to make C2's feature row unmistakably
        # its own, small enough not to arm the 2% trailing stop
        # when C2's stop check runs after the open fill.
        c2 = frame.index[-2]
        frame.loc[c2, "close"] = frame.loc[c2, "open"] * 1.01
        frame.loc[c2, "high"] = frame.loc[c2, "close"] * 1.001
        return frame

    cfg, frame, (c0, c1, c2, f), ppo, out = _setup(
        tmp_path, monkeypatch, actions=[1, 0], mutate=spike)
    assert out["processed"] == 4  # stop C1, decide C2, stop C2, decide F
    trades = _trades(cfg)
    assert len(trades) == 1
    ts, side, price, _qty, _reason = trades[0]
    assert ts == c2.isoformat() and side == "BUY"
    assert price == pytest.approx(float(frame.loc[c2, "open"]))

    feat = add_features(frame)
    want = np.array([float(feat.loc[c1, c]) for c in FEATURE_COLUMNS]
                    + [2.0 * 0.6 - 1.0], dtype=np.float64)
    got = ppo.obs_seen[0][:22]
    assert np.allclose(got, want, atol=1e-5)
    leaked = np.array([float(feat.loc[c2, c]) for c in FEATURE_COLUMNS],
                      dtype=np.float64)
    assert not np.allclose(ppo.obs_seen[0][:21], leaked, atol=1e-5)

    state = json.loads(cfg.state_file.read_text())
    assert state["last_open_ts"] == f.isoformat()
    assert state["last_stop_ts"] == c2.isoformat()
    # last_ts = the candle the recorded vote was computed on
    assert state["last_ts"] == c2.isoformat()
    assert state["last_p_up"] == 0.6

    # A second wake in the same hour decides nothing new.
    out2 = run_once(cfg, fetch=lambda *a, **k: frame,
                    stacker=_StubStacker(), ppo=_StubPPO([1, 1]))
    assert out2["processed"] == 0
    assert len(_trades(cfg)) == 1


def test_transition_candle_gets_no_decision(tmp_path, monkeypatch):
    """Candles before MODE_FROM keep their close-mode record:
    with MODE_FROM at the forming candle, the newest closed
    candle is stop-checked/marked only -- no decision, no fill
    stamped on it -- and the first decision lands on the
    forming candle's open."""
    frame = _frame()
    c1, c2, f = frame.index[-3], frame.index[-2], frame.index[-1]
    monkeypatch.setattr(live_champ, "DECISION_MODE", "open")
    monkeypatch.setattr(live_champ, "MODE_FROM_TS", f.isoformat())
    cfg = _cfg(tmp_path, c1)
    out = run_once(cfg, fetch=lambda *a, **k: frame,
                   stacker=_StubStacker(), ppo=_StubPPO([1]))
    assert out["processed"] == 2  # stop C2, decide F
    trades = _trades(cfg)
    assert len(trades) == 1
    assert trades[0][0] == f.isoformat()
    assert trades[0][2] == pytest.approx(float(frame.loc[f, "open"]))
    recs = _decisions(cfg)
    assert [r["ts"] for r in recs] == [f.isoformat()]


def test_stop_runs_before_next_decide_and_logs_exit(
        tmp_path, monkeypatch):
    """A position opened at C2's open is exposed to C2's own
    wick: the stop exits inside C2, the exit is logged as a
    second record for C2, and DECIDE(F) afterwards sees a flat
    book."""
    def wick(frame):
        c2 = frame.index[-2]
        frame.loc[c2, "low"] = frame.loc[c2, "open"] * 0.80
        frame.loc[c2, "close"] = frame.loc[c2, "open"] * 0.99
        return frame

    cfg, frame, (c0, c1, c2, f), ppo, out = _setup(
        tmp_path, monkeypatch, actions=[1, 0], mutate=wick)
    trades = _trades(cfg)
    assert len(trades) == 2
    assert trades[0][1] == "BUY"
    assert trades[0][2] == pytest.approx(float(frame.loc[c2, "open"]))
    assert trades[1][0] == c2.isoformat() and trades[1][1] == "SELL"
    assert trades[1][4] in ("stop_loss", "trailing_stop")
    assert float(frame.loc[c2, "low"]) < trades[1][2] \
        < float(frame.loc[c2, "open"])

    state = json.loads(cfg.state_file.read_text())
    assert state["btc"] == 0.0
    recs = _decisions(cfg)
    c2_recs = [r for r in recs if r["ts"] == c2.isoformat()]
    assert len(c2_recs) == 2  # the open decision AND the stop exit
    assert c2_recs[1]["reason"] in ("stop_loss", "trailing_stop")
    f_rec = [r for r in recs if r["ts"] == f.isoformat()]
    assert len(f_rec) == 1 and f_rec[0]["pos_before"] == 0.0


def test_decision_log_duplicate_ts_opt_in(tmp_path):
    """The shared dedupe stays the default; only the explicit
    opt-in appends a second record for one candle."""
    rec = dl.make_record(ts="2026-11-13T20:00:00", close=100.0,
                         p_up=0.5, stance="hold", action="HOLD",
                         taken=False, reason="stance_hold")
    assert dl.append_decision("book5", rec, decisions_dir=tmp_path)
    assert not dl.append_decision("book5", rec,
                                  decisions_dir=tmp_path)
    assert dl.append_decision("book5", rec, decisions_dir=tmp_path,
                              allow_duplicate_ts=True)


def test_close_mode_rollback_still_works(tmp_path, monkeypatch):
    """DECISION_MODE='close' keeps the original loop: newly
    closed candles are processed at their closes."""
    monkeypatch.setattr(live_champ, "DECISION_MODE", "close")
    frame = _frame()
    c0, c2 = frame.index[-4], frame.index[-2]
    cfg = _cfg(tmp_path, c0)
    out = run_once(cfg, fetch=lambda *a, **k: frame,
                   stacker=_StubStacker(), ppo=_StubPPO([0, 0]))
    assert out["processed"] == 2  # C1 and C2, the closed pair
    state = json.loads(cfg.state_file.read_text())
    assert state["last_ts"] == c2.isoformat()
