"""GATED live loop — Book #3, the run-82 gate forward test
(Kass approved 2026-10-30; pre-registered in
refs/forward-test-gated-book3-2026-10-30.md BEFORE the first run).

Book #3 began as the EXACT twin of Book #2 (src/trading/
live_shadow.py): Driver PPO run 44, Eyes Stacker Champion #4
(run 73, Muse's Pick #2, pick2_98 frame), production
PaperBroker, closed-1h decisions, $100,000 FLAT — with every
ENTRY attempt first evaluated by run 82's frozen meta-label
gate (refs/meta-label-study-2026-10-30.md).

SEGMENT 2 (Kass's direct order, 2026-11-03): the driver became
PPO run 99 (seated by Kass's pick, recorded duel verdict NOT
PROMOTED) and the gate's veto threshold went to 0.50.

SEGMENT 3 (Kass's direct order, 2026-11-04): the gate's veto
threshold is back to 0.55, read from the seat manifest
(data/gated_seat.json) — the threshold is a seat parameter
Kass sets; driver (run 99) and eyes (run 73) unchanged; the
gate MODEL stays run 82's frozen artifact, hash-pinned, never
retrained or refit. Segment 1 (2026-10-30 → 2026-11-03T23:00Z)
ran run 44 @ 0.55 and closed with 7 attempts, 0 taken,
7 vetoed. Segment 2 (2026-11-04T00:00Z → 2026-11-04T21:00Z)
ran run 99 @ 0.50 and closed with 14 attempts, 9 taken,
5 vetoed. The segment record is the amendment in
refs/forward-test-gated-book3-2026-10-30.md.

If the gate's P(profitable) for an attempt is below the seat
threshold, the entry is VETOED — logged in this book's own
gate_decisions table and never executed. Exits and all
position management (sells, covers, through-flat closes,
protective stops) are never gated.

The gate is the study's own machinery, not a re-implementation:
the 104-vector is built by training.meta_label.feature_row (the
98 pick2_98 values in the champion's recorded column order +
[p_up, |p_up - 0.5|, atr_rel, volatility_21, atr_pct_raw, side])
and scored by training.meta_label.MetaGate with the seat
manifest's threshold (0.55 in Segment 1; 0.50 in Segment 2;
0.55 again since Kass's 2026-11-04 order). The gate model file is a byte-copy of the
study's artifact, hash-pinned in the seat manifest; it is never
retrained or refit during the test. Gate parity with the study
is pinned in tests/test_gated_desk.py against the study's own
saved holdout arrays.

The gate decision point is the live loop's dispatch point:
after the candle's protective stop check, before the order is
dispatched — the exact point at which Book #2 executes its
entries, so every Book #2 entry fill has a corresponding Book
#3 decision row while the books' positions agree (the forward
test's primary evidence design).

PERIMETER (pinned by tests/test_gated_desk.py): this module
WRITES only its own files:
    models/gated_seat/          byte-copies of the pinned artifacts
    data/gated_seat.json        the gated seat manifest (hashes)
    data/paper_gated.db         Book #3's own ledger + decisions
    data/live_state_gated.json  Book #3's own loop state
It reads Book #1/#2 ledgers only through the dashboard race
feed (read-only), never opens another book's file for writing,
and a missing or tampered seat file is a loud failure — never
a fallback to the lab champions or another book's seat.

Usage:
    python -m src.trading.live_gated --seed   # one-time launch
    python -m src.trading.live_gated          # one loop step
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.market_data import fetch_klines
from features.technical_indicators import FEATURE_COLUMNS, add_features
from models.stacker import load as load_stacker
from trading import live_shadow as _shadow
from trading.broker import PaperBroker
from trading.decision_log import (append_decision, broker_stop_level,
                                  describe_outcome, make_record,
                                  stance_name)
from training.meta_label import GATE_THRESHOLD, MetaGate, is_entry_attempt

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

SEAT_DIR = MODELS / "gated_seat"
SEAT_STACKER = SEAT_DIR / "stacker.pkl"
SEAT_STACKER_META = SEAT_DIR / "stacker.json"
SEAT_PPO = SEAT_DIR / "ppo.zip"
SEAT_GATE = SEAT_DIR / "gate_model.pkl"

# Canonical sources the seat was copied from (the pinned runs).
SOURCE_STACKER = MODELS / "stacker_champion.pkl"        # run 73
SOURCE_STACKER_META = MODELS / "stacker_champion.json"   # run 73
SOURCE_PPO = MODELS / "ppo_run99.zip"                    # run 99 (preserved copy;
# ppo_challenger.zip is overwritten by every nightly retrain)
SOURCE_GATE = DATA / "meta_label" / "meta_model.pkl"     # run 82

STARTING_BALANCE = 100_000.0
DRIVER_RUN = 99
EYES_RUN = 73
GATE_RUN = 82
# The seat's current gate threshold (Segment 3). At runtime the
# manifest (data/gated_seat.json) is authoritative — this is
# only build_manifest's seed-time value. Kass sets it; the gate
# MODEL below stays frozen by hash regardless.
SEAT_GATE_THRESHOLD = 0.55
# The frozen gate identity (run-82 study record): the seat copy
# and its canonical source must both hash to this, always.
GATE_MODEL_SHA256 = \
    "cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a"


@dataclass(frozen=True)
class GatedConfig:
    db_file: Path = DATA / "paper_gated.db"
    state_file: Path = DATA / "live_state_gated.json"
    seat_dir: Path = SEAT_DIR
    manifest_file: Path = DATA / "gated_seat.json"


DEFAULT_CONFIG = GatedConfig()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------- the seat

def ensure_seat(seat_dir: Path = SEAT_DIR) -> None:
    """Copy the pinned artifacts into the gated seat if absent.

    A seat file that exists but differs from its canonical source
    is a loud failure (never silently overwritten -- a changed
    seat is a decision, not an accident). The gate source must
    additionally hash to the study's frozen identity: if the lab
    file at data/meta_label/meta_model.pkl is ever replaced, this
    desk refuses to seat the replacement -- run 82's gate is the
    frozen artifact named in the pre-registration, by hash."""
    seat_dir.mkdir(parents=True, exist_ok=True)
    pairs = ((seat_dir / "stacker.pkl", SOURCE_STACKER),
             (seat_dir / "stacker.json", SOURCE_STACKER_META),
             (seat_dir / "ppo.zip", SOURCE_PPO),
             (seat_dir / "gate_model.pkl", SOURCE_GATE))
    for dest, src in pairs:
        if not src.exists():
            raise FileNotFoundError(f"gated seat source missing: {src}")
        if dest.name == "gate_model.pkl" and \
                _sha256(src) != GATE_MODEL_SHA256:
            raise ValueError(
                f"gate model source {src} hashes to {_sha256(src)}, "
                f"not the frozen run-82 identity {GATE_MODEL_SHA256}; "
                f"refusing to seat a different gate")
        if dest.exists():
            if _sha256(dest) != _sha256(src):
                raise ValueError(
                    f"gated seat file {dest} differs from its pinned "
                    f"source {src}; refusing to overwrite -- a seat "
                    f"change ends the forward test (see the "
                    f"pre-registration)")
        else:
            shutil.copyfile(src, dest)


def build_manifest(started: str, seat_dir: Path = SEAT_DIR) -> dict:
    """The gated seat ledger (data/gated_seat.json): the twin
    pairing + the frozen gate, hash-pinned, with the test record."""
    meta = json.loads((seat_dir / "stacker.json").read_text())
    cols = list(meta.get("feature_columns") or [])
    return {
        "desk": "TradingAgent Live — Gated (Book #3 · Run 82 Gate)",
        "book": "Book #3",
        "driver": {
            "brain": "PPO", "run_id": DRIVER_RUN,
            "file": "models/gated_seat/ppo.zip",
            "sha256": _sha256(seat_dir / "ppo.zip"),
            "source": "models/ppo_run99.zip",
        },
        "eyes": {
            "brain": "Stacker", "run_id": EYES_RUN,
            "name": "Muse's Pick #2 Stacker",
            "champion": "Champion #4",
            "feature_set": meta.get("feature_set", "scored102"),
            "n_features": len(cols),
            "file": "models/gated_seat/stacker.pkl",
            "sha256": _sha256(seat_dir / "stacker.pkl"),
            "meta_file": "models/gated_seat/stacker.json",
            "meta_sha256": _sha256(seat_dir / "stacker.json"),
            "source": "models/stacker_champion.pkl",
        },
        "gate": {
            "brain": "Meta-Label (LightGBM)", "run_id": GATE_RUN,
            "family": "meta_label",
            "file": "models/gated_seat/gate_model.pkl",
            "sha256": _sha256(seat_dir / "gate_model.pkl"),
            "source": "data/meta_label/meta_model.pkl",
            "threshold": SEAT_GATE_THRESHOLD,
            "feature_vector": "104 = 98 pick2_98 scored values "
                              "(champion's recorded column order) + "
                              "[p_up, |p_up-0.5|, atr_rel, "
                              "volatility_21, atr_pct_raw, side], "
                              "built by src/training/meta_label.py "
                              "feature_row",
            "study": "refs/meta-label-study-2026-10-30.md",
            "forward_test": "refs/forward-test-gated-book3-"
                            "2026-10-30.md",
        },
        "book_db": "data/paper_gated.db",
        "state_file": "data/live_state_gated.json",
        "starting_balance": int(STARTING_BALANCE),
        "started": started,
        "seeded_by": "Book #3 gated forward test build (Kass "
                     "approved the forward test 2026-10-30); "
                     "driver re-seated to run 99 and threshold "
                     "set to 0.50 by Kass's direct order "
                     "2026-11-03 (Segment 2)",
        "pinned": "Driver pinned to PPO run 99 (Kass's pick "
                  "2026-11-03; recorded duel verdict NOT "
                  "PROMOTED), eyes pinned to Stacker run 73 "
                  "(Champion #4), gate MODEL pinned to run 82's "
                  "frozen meta-model by hash (NEVER retrained or "
                  "refit), gate threshold a seat parameter "
                  "(0.55 in Segment 1; 0.50 since 2026-11-03). "
                  "This seat does NOT follow future lab "
                  "crownings. Segment record: the amendment in "
                  "refs/forward-test-gated-book3-2026-10-30.md.",
        "note": "Book #3: the gated shadow book. Segment 1: "
                "Book #2's exact twin (run 44 driver) plus run "
                "82's entry veto at 0.55 (7 attempts, 0 taken). "
                "Segment 2: run 99 driver, veto at 0.50. Primary "
                "evidence: the gate_decisions table in "
                "data/paper_gated.db; each row records the "
                "threshold it was judged at.",
    }


def load_manifest(cfg: GatedConfig = DEFAULT_CONFIG) -> dict:
    if not cfg.manifest_file.exists():
        raise FileNotFoundError(
            f"gated manifest missing: {cfg.manifest_file} "
            f"(seed the desk first: python -m src.trading.live_gated "
            f"--seed)")
    return json.loads(cfg.manifest_file.read_text())


def verify_seat(manifest: dict, seat_dir: Path) -> None:
    """Every gated seat file must hash to the manifest's pin --
    a swapped or corrupted artifact (gate included) fails loudly
    before any candle is processed (never a silent fallback)."""
    checks = [
        (seat_dir / "ppo.zip", manifest["driver"]["sha256"], "PPO"),
        (seat_dir / "stacker.pkl", manifest["eyes"]["sha256"], "stacker"),
        (seat_dir / "stacker.json", manifest["eyes"]["meta_sha256"],
         "stacker meta"),
        (seat_dir / "gate_model.pkl", manifest["gate"]["sha256"],
         "gate model"),
    ]
    for path, want, what in checks:
        if not path.exists():
            raise FileNotFoundError(f"gated seat {what} missing: {path}")
        got = _sha256(path)
        if got != want:
            raise ValueError(
                f"gated seat {what} hash mismatch: {path} is {got}, "
                f"manifest pins {want}; refusing to trade on an "
                f"unpinned artifact")


def seat_columns(seat_dir: Path = SEAT_DIR) -> list[str]:
    return _shadow.seat_columns(seat_dir)


def seat_frame_builder(seat_dir: Path = SEAT_DIR):
    return _shadow.seat_frame_builder(seat_dir)


# -------------------------------------------------------- the gate math

def gate_proba(model, frame_row, p_up: float, atr_rel: float,
               vol21: float, atr_raw: float, side: int,
               threshold: float = GATE_THRESHOLD) -> float:
    """Run 82's P(profitable) for ONE decision candle, computed by
    the study's own MetaGate/feature_row on 1-row arrays -- the
    identical function the duel used, so the live vector is the
    study's vector by construction (parity pinned in tests
    against the study's saved holdout arrays)."""
    gate = MetaGate(model,
                    np.asarray([frame_row], dtype=float),
                    np.array([float(p_up)]),
                    np.array([float(atr_rel)]),
                    np.array([float(vol21)]),
                    np.array([float(atr_raw)]),
                    threshold=float(threshold))
    return gate.proba(0, int(side))


# ------------------------------------------------------ decision record

def ensure_decisions_table(db_file: Path) -> None:
    with sqlite3.connect(db_file) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS gate_decisions("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, side INTEGER NOT NULL, "
            "action INTEGER NOT NULL, proba REAL NOT NULL, "
            "threshold REAL NOT NULL, verdict TEXT NOT NULL)")


def record_gate_decision(db_file: Path, ts: str, side: int,
                         action: int, proba: float, verdict: str,
                         threshold: float = GATE_THRESHOLD) -> None:
    """One durable row per entry attempt — the forward test's
    primary evidence. Written by the gated engine itself at
    decision time, before the (possibly vetoed) dispatch."""
    ensure_decisions_table(db_file)
    with sqlite3.connect(db_file) as con:
        con.execute(
            "INSERT INTO gate_decisions"
            "(ts, side, action, proba, threshold, verdict) "
            "VALUES (?,?,?,?,?,?)",
            (ts, int(side), int(action), float(proba),
             float(threshold), verdict))


# ------------------------------------------------------------ the state

def load_state(cfg: GatedConfig) -> dict:
    if cfg.state_file.exists():
        try:
            return json.loads(cfg.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            print("live_state_gated.json unreadable; rebuilding from "
                  "the gated ledger", flush=True)
            return _shadow.rebuild_state_from_ledger(cfg)
    return {"last_ts": None}


def save_state(cfg: GatedConfig, state: dict) -> None:
    _shadow.save_state(cfg, state)


# ---------------------------------------------------------------- seed

def seed(cfg: GatedConfig = DEFAULT_CONFIG, fetch=None) -> dict:
    """One-time launch: seat the pinned artifacts (gate included),
    open Book #3 FLAT at $100,000, and anchor the loop at the
    newest CLOSED candle — candles at or before the anchor are
    never backfilled into trades; the desk's first decision is
    the first candle that closes after launch. Idempotent: an
    already-seeded desk is verified, not re-seeded."""
    ensure_seat(cfg.seat_dir)
    if cfg.state_file.exists():
        man = load_manifest(cfg)
        verify_seat(man, cfg.seat_dir)
        state = load_state(cfg)
        return {"seeded": False, "started": man.get("started"),
                "last_ts": state.get("last_ts")}
    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="5d")
    closed = raw.iloc[:-1]
    if closed.empty:
        raise RuntimeError("market fetch returned no closed candles")
    anchor = closed.index[-1]
    anchor_ts = pd.Timestamp(anchor).isoformat()
    price = float(closed.iloc[-1]["close"])
    if cfg.db_file.exists():
        con = sqlite3.connect(cfg.db_file)
        try:
            n = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
        except sqlite3.Error:
            n = 0
        con.close()
        if n:
            raise RuntimeError(
                "gated ledger already has equity rows but no state "
                "file; refusing to seed over an inconsistent book")
    broker = PaperBroker(cfg.db_file, starting_balance=STARTING_BALANCE,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    broker.snapshot(anchor_ts, price)  # the flat $100k anchor row
    ensure_decisions_table(cfg.db_file)
    state = {"last_ts": anchor_ts, "started": anchor_ts,
             "desk": "gated_book3",
             "cash": STARTING_BALANCE, "btc": 0.0,
             "entry_price": None, "last_entry_price": None,
             "entry_atr_pct": None, "high_water": None,
             "low_water": None, "stop_anchor": None,
             "last_stance": None, "last_p_up": None,
             "last_gate_decision": None}
    save_state(cfg, state)
    if not cfg.manifest_file.exists():
        cfg.manifest_file.write_text(
            json.dumps(build_manifest(anchor_ts, cfg.seat_dir),
                       indent=1))
    man = load_manifest(cfg)
    verify_seat(man, cfg.seat_dir)
    return {"seeded": True, "started": anchor_ts, "equity": broker.cash,
            "anchor_price": price,
            "gate_sha256": man["gate"]["sha256"]}


# ------------------------------------------------------------ main loop

def run_once(cfg: GatedConfig = DEFAULT_CONFIG, fetch=None,
             stacker=None, ppo=None, scored_builder=None,
             gate_model=None) -> dict:
    if not cfg.state_file.exists() or not cfg.manifest_file.exists():
        raise RuntimeError(
            "gated desk is not seeded; run "
            "python -m src.trading.live_gated --seed first "
            "(the desk never backfills: its first decision is the "
            "first candle closed after launch)")
    man = load_manifest(cfg)
    verify_seat(man, cfg.seat_dir)
    threshold = float((man.get("gate") or {}).get(
        "threshold", GATE_THRESHOLD))
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"gated manifest threshold {threshold} is not a "
            f"probability; refusing to run")
    # The threshold is a SEAT parameter: it lives in the manifest
    # and changes only on Kass's order (Segment 1 ran at the
    # study's 0.55; Kass set 0.50 on 2026-11-03 and restored
    # 0.55 on 2026-11-04). What stays
    # frozen is the gate MODEL — verify_seat hash-pins it above.
    fetch = fetch or fetch_klines
    state = load_state(cfg)
    last_ts = state.get("last_ts")

    # Cheap pre-check, exactly as the shadow loop's.
    probe = fetch("BTC-USD", interval="1h", period="5d")
    probe_closed = probe.iloc[:-1]
    if probe_closed.empty:
        raise RuntimeError("market fetch returned no closed candles")
    newest = probe_closed.index[-1]
    if last_ts and pd.Timestamp(last_ts) >= newest:
        return {"processed": 0, "vetoes": 0, "cash": state.get("cash"),
                "btc": state.get("btc")}

    if stacker is None:
        stacker = load_stacker(cfg.seat_dir / "stacker.pkl")
    columns = seat_columns(cfg.seat_dir)
    if ppo is None:
        from stable_baselines3 import PPO
        ppo = PPO.load(cfg.seat_dir / "ppo.zip")
    if gate_model is None:
        import joblib
        gate_model = joblib.load(cfg.seat_dir / "gate_model.pkl")
    expected_dim = len(FEATURE_COLUMNS) + 1 + 3
    model_dim = int(ppo.observation_space.shape[0])
    if model_dim != expected_dim:
        return {"processed": 0, "vetoes": 0,
                "skipped": f"gated PPO expects {model_dim}-dim obs, "
                           f"recipe builds {expected_dim}-dim"}

    raw = fetch("BTC-USD", interval="1h", period="730d")
    full = raw.iloc[:-1]
    if full.empty:
        raise RuntimeError("market fetch returned no closed candles")
    newest_full = full.index[-1]
    age_h = (pd.Timestamp.now(tz=newest_full.tz)
             - newest_full).total_seconds() / 3600
    if age_h > 2.5:
        raise RuntimeError(
            f"stale market data: newest closed candle is {age_h:.1f}h "
            f"old ({newest_full}); refusing to trade on it")
    # Features on the FULL frame including the forming candle, for
    # the same reason live.py / live_shadow.py do; only closed
    # rows are ever processed, and both frames are causal.
    feat = add_features(raw)
    if scored_builder is None:
        scored_builder = seat_frame_builder(cfg.seat_dir)
    scored_frame, _built_cols = scored_builder(raw)
    broker = _shadow._restore_broker(cfg, state)
    df = full
    if last_ts:
        df = df[df.index > pd.Timestamp(last_ts)]

    processed = 0
    vetoes = 0
    last_price = 0.0
    for ts, row in df.iterrows():
        frow = feat.loc[ts]
        if frow[FEATURE_COLUMNS].isna().any():
            print(f"candle {ts} has NaN prod21 features; stopping "
                  f"gated run here", flush=True)
            break
        srow = scored_frame.loc[ts, columns]
        if srow.isna().any():
            print(f"candle {ts} has NaN scored features; stopping "
                  f"gated run here", flush=True)
            break
        p_up = _shadow.champion2_vote(stacker, scored_frame, ts, columns)
        price = float(row["close"])
        last_price = price
        # The observation is Book #2's, byte for byte.
        market = [float(frow[c]) for c in FEATURE_COLUMNS] + [2.0 * p_up - 1.0]
        eq = broker.equity(price)
        obs = np.array(market + [
            broker.cash / max(eq, 1e-9),
            broker.btc * price / max(eq, 1e-9),
            broker.unrealized_pct(price),
        ], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        action, _ = ppo.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        ts_s = ts.isoformat()
        atr_raw = frow.get("atr_pct_raw")
        atr_pct = (float(atr_raw) if atr_raw is not None
                   and np.isfinite(float(atr_raw)) else None)
        if broker.btc != 0 and broker.entry_atr_pct is None:
            broker.entry_atr_pct = atr_pct
        ctx_dict = {
            "f": [round(float(frow[c]), 4) for c in FEATURE_COLUMNS],
            "p": round(p_up, 4), "eyes": _shadow._eyes_tag(man),
            "g": None, "grun": GATE_RUN}
        # The protective stop check runs FIRST, exactly as in
        # Book #2 — exits are never gated, and a stop fill on this
        # candle settles the position before any entry is judged.
        pos_before = broker.btc
        stop_fill = broker.check_stop(price, ts=ts_s,
                                      low=float(row["low"]),
                                      high=float(row["high"]),
                                      context=json.dumps(ctx_dict))
        # ---- THE GATE (the only difference from Book #2) ----
        side = is_entry_attempt(action, broker.btc)
        gate_p = None
        gate_verdict = None
        vetoed = False
        if side != 0:
            if atr_pct is None:
                print(f"candle {ts} has no finite atr_pct_raw; the "
                      f"gate vector cannot be built exactly — "
                      f"stopping gated run here", flush=True)
                break
            gate_p = gate_proba(
                gate_model, srow.to_numpy(dtype=float), p_up,
                float(frow["atr_rel"]), float(frow["volatility_21"]),
                atr_pct, side, threshold=threshold)
            verdict = "TAKEN" if gate_p >= threshold else "VETOED"
            gate_verdict = verdict
            record_gate_decision(cfg.db_file, ts_s, side, action,
                                 gate_p, verdict, threshold=threshold)
            state["last_gate_decision"] = {
                "ts": ts_s, "side": side, "action": action,
                "proba": round(gate_p, 4), "threshold": threshold,
                "verdict": verdict}
            vetoed = verdict == "VETOED"
            if vetoed:
                vetoes += 1
            ctx_dict["g"] = round(gate_p, 4)
        ctx = json.dumps(ctx_dict)
        executed = 0 if vetoed else action
        fill = None
        if executed == 1:
            fill = broker.buy(price, reason="ppo_live", ts=ts_s,
                              atr_pct=atr_pct, context=ctx)
        elif executed == 2:
            fill = broker.sell(price, reason="ppo_live", ts=ts_s,
                               context=ctx)
        elif executed == 3:
            fill = broker.short_open(price, reason="ppo_live", ts=ts_s,
                                     atr_pct=atr_pct, context=ctx)
        broker.snapshot(ts_s, price)
        # Decision log (instrumentation only, 2026-10-31): what the
        # bot was thinking on this candle, gate included -- the
        # proba/verdict are recorded for TAKEN attempts too, not
        # just vetoes. Values already in hand; failure-isolated;
        # never alters trading.
        _d_action, _d_taken, _d_reason = describe_outcome(
            action, stop_fill, fill, gate_verdict=gate_verdict)
        _d_eq = broker.equity(price)
        append_decision("book3", make_record(
            ts=ts_s, close=price, p_up=round(p_up, 4),
            stance=stance_name(action), action=_d_action,
            taken=_d_taken, reason=_d_reason,
            gate_proba=(round(gate_p, 4) if gate_p is not None
                        else None),
            gate_threshold=threshold, gate_verdict=gate_verdict,
            pos_before=pos_before, pos_after=broker.btc,
            equity=round(_d_eq, 2),
            stop_level=broker_stop_level(broker),
            exposure_pct=(round(100.0 * abs(broker.btc) * price
                                / _d_eq, 2) if _d_eq else None),
            features={c: float(srow[c]) for c in columns}),
            decisions_dir=cfg.state_file.parent / "decisions")
        state["last_ts"] = ts_s
        state["last_stance"] = ["hold", "buy", "sell", "short"][executed]
        state["last_p_up"] = round(p_up, 4)
        state["cash"] = broker.cash
        state["btc"] = broker.btc
        state["entry_price"] = broker.entry_price
        state["last_entry_price"] = broker.entry_price
        state["entry_atr_pct"] = broker.entry_atr_pct
        state["high_water"] = broker.high_water
        state["low_water"] = broker.low_water
        state["stop_anchor"] = broker.stop_anchor
        save_state(cfg, state)
        processed += 1

    return {"processed": processed, "vetoes": vetoes,
            "equity": broker.equity(last_price) if processed else None,
            "cash": broker.cash, "btc": broker.btc}


if __name__ == "__main__":
    if "--seed" in sys.argv:
        print(json.dumps(seed()), flush=True)
    else:
        print(json.dumps(run_once()), flush=True)
