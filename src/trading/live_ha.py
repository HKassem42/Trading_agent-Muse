"""HA live loop (Book #4) -- Kass's order, 2026-11-05.

A fourth, fully independent paper-trading book for the pairing
    Driver: PPO run 44  -  Eyes: Stacker Champion #6 (run 110,
    Muse's Pick #11: the flat Pick recipe on the 10-value
    Heikin-Ashi candle matrix)
Kass crowned Pick #11 Lab Stacker Champion #6 on 2026-11-05 and
ordered a live book for it the same day ("I also want a #4
trading live book for it... make sure the chart in the book
reads like a HA candle stream"). The PPO reads the production
21 features exactly as the other books' driver does, but its
stacker vote comes from the seated Pick #11 stacker reading the
Heikin-Ashi candle matrix (feature_set tag "ha10") -- the first
eyes in the program whose native frame is HA candles. Kass's
reason for the whole HA line: the smoothed red/green runs make
trend visible, and Pick #11 is the first stacker whose
walk-forward agreed with its exam (0.5234 vs 0.5039 majority,
all five folds above).

PERIMETER (pinned by tests/test_ha_desk.py): this module reads
production state (the other books' ledgers, for the race feed's
source data) but WRITES only its own files:
    models/ha_seat/            byte-copies of the pinned artifacts
    data/ha_seat.json          the HA seat manifest (hashes)
    data/paper_ha.db           Book #4's own ledger
    data/live_state_ha.json    Book #4's own loop state
It never opens another book's file for writing, and a missing
or tampered HA seat file is a loud failure -- never a fallback
to the lab champions or another seat (the seat-separation
doctrine of src/trading/live.py, applied to the HA seat).

The seat is PINNED to run 110: a future lab crowning does not
move this desk. The stacker source is the Pick #11 study
artifact itself (data/muse_pick11/pick11_stacker.pkl), so the
seat is immune to later crownings by construction. It changes
only by Kass's explicit decision, executed manually.

Trading mechanics are NOT forked: the broker is the production
PaperBroker (25% tranche, 75% exposure cap, 0.15% fees, ATR
stops/trailing, stop-anchor ratchet) and the observation is
built exactly as src/trading/live.py builds it -- only the
vote's source differs. Decisions are deterministic argmax on
newly CLOSED 1h candles, processed once each (tracked in the HA
state file); candles at or before the seed anchor are never
backfilled.

Usage:
    python -m src.trading.live_ha --seed   # one-time launch
    python -m src.trading.live_ha          # one loop step
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.market_data import fetch_klines
from features.technical_indicators import FEATURE_COLUMNS, add_features
from models.stacker import load as load_stacker
from trading.broker import PaperBroker
from trading.decision_log import (append_decision, broker_stop_level,
                                  describe_outcome, make_record,
                                  stance_name)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

SEAT_DIR = MODELS / "ha_seat"
SEAT_STACKER = SEAT_DIR / "stacker.pkl"
SEAT_STACKER_META = SEAT_DIR / "stacker.json"
SEAT_PPO = SEAT_DIR / "ppo.zip"

# Canonical sources the seat was copied from (the pinned runs).
SOURCE_STACKER = DATA / "muse_pick11" / "pick11_stacker.pkl"  # run 110
SOURCE_STACKER_META = DATA / "muse_pick11" / "seat_meta.json"  # run 110
SOURCE_PPO = MODELS / "ppo_champion.zip"                 # run 44

STARTING_BALANCE = 100_000.0
DRIVER_RUN = 44
EYES_RUN = 110


@dataclass(frozen=True)
class HAConfig:
    db_file: Path = DATA / "paper_ha.db"
    state_file: Path = DATA / "live_state_ha.json"
    seat_dir: Path = SEAT_DIR
    manifest_file: Path = DATA / "ha_seat.json"


DEFAULT_CONFIG = HAConfig()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------- the seat

def ensure_seat(seat_dir: Path = SEAT_DIR) -> None:
    """Copy the pinned artifacts into the HA seat if absent.

    A seat file that exists but differs from its canonical source
    is a loud failure (never silently overwritten -- a changed seat
    is a decision, not an accident)."""
    seat_dir.mkdir(parents=True, exist_ok=True)
    for dest, src in ((seat_dir / "stacker.pkl", SOURCE_STACKER),
                      (seat_dir / "stacker.json", SOURCE_STACKER_META),
                      (seat_dir / "ppo.zip", SOURCE_PPO)):
        if not src.exists():
            raise FileNotFoundError(f"HA seat source missing: {src}")
        if dest.exists():
            if _sha256(dest) != _sha256(src):
                raise ValueError(
                    f"HA seat file {dest} differs from its pinned "
                    f"source {src}; refusing to overwrite -- a seat "
                    f"change is Kass's explicit decision")
        else:
            shutil.copyfile(src, dest)


def build_manifest(started: str, seat_dir: Path = SEAT_DIR) -> dict:
    """The HA seat ledger (data/ha_seat.json): which runs'
    artifacts this desk trades, hash-pinned, with the pin note."""
    meta = json.loads((seat_dir / "stacker.json").read_text())
    cols = list(meta.get("feature_columns") or [])
    return {
        "desk": "TradingAgent Live — HA Desk (Book #4 · Pick #11 Eyes)",
        "book": "Book #4",
        "driver": {
            "brain": "PPO", "run_id": DRIVER_RUN,
            "file": "models/ha_seat/ppo.zip",
            "sha256": _sha256(seat_dir / "ppo.zip"),
            "source": "models/ppo_champion.zip",
        },
        "eyes": {
            "brain": "Stacker", "run_id": EYES_RUN,
            "name": "Muse's Pick #11 Stacker — Flat Recipe on "
                    "HA Candles",
            "champion": "Champion #6",
            "feature_set": meta.get("feature_set", "ha10"),
            "n_features": len(cols),
            "file": "models/ha_seat/stacker.pkl",
            "sha256": _sha256(seat_dir / "stacker.pkl"),
            "meta_file": "models/ha_seat/stacker.json",
            "meta_sha256": _sha256(seat_dir / "stacker.json"),
            "source": "data/muse_pick11/pick11_stacker.pkl",
        },
        "book_db": "data/paper_ha.db",
        "state_file": "data/live_state_ha.json",
        "starting_balance": int(STARTING_BALANCE),
        "started": started,
        "seeded_by": "Book #4 HA desk build (Kass's order "
                     "2026-11-05, the same day he crowned Pick "
                     "#11 Lab Stacker Champion #6)",
        "grading_record": {
            "eyes_run": 110,
            "exam_acc": 0.5341, "exam_precision": 0.5260,
            "wf_acc": 0.5234, "wf_majority": 0.5039,
            "source": "Muse's Pick #11 pre-registered grading "
                      "(refs/muse-pick-11-preregistration-"
                      "2026-11-05.md): the eyes passed their "
                      "stacker bar on the Picks #9/#10 identical "
                      "exam with walk-forward agreement. The "
                      "PAIRING (PPO run 44 + Pick #11 eyes) has "
                      "no duel exam; this book IS its forward "
                      "test.",
        },
        "pinned": "Stacker pinned to run 110 (Champion #6, "
                  "Muse's Pick #11) and PPO pinned to run 44: "
                  "this seat does NOT follow future lab "
                  "crownings and is NOT the driver's seat. The "
                  "stacker source is the Pick #11 study artifact "
                  "itself, so later crownings cannot move it. "
                  "It changes only by Kass's explicit decision, "
                  "executed manually.",
        "note": "Book #4: the HA book. Same driver as Books "
                "#1/#2 (PPO run 44), but the eyes read "
                "Heikin-Ashi candles (10 values per candle: "
                "color, body, range, wicks, signed streak, "
                "slope, displacement, raw return, body share) "
                "instead of the production or scored frames -- "
                "and this desk's price chart renders the HA "
                "candle stream, per Kass's order. Trading itself "
                "(fills, stops, equity) is on real prices, "
                "exactly like the other books.",
    }


def load_manifest(cfg: HAConfig = DEFAULT_CONFIG) -> dict:
    if not cfg.manifest_file.exists():
        raise FileNotFoundError(
            f"HA manifest missing: {cfg.manifest_file} "
            f"(seed the desk first: python -m src.trading.live_ha "
            f"--seed)")
    return json.loads(cfg.manifest_file.read_text())


def verify_seat(manifest: dict, seat_dir: Path) -> None:
    """Every HA seat file must hash to the manifest's pin --
    a swapped or corrupted artifact fails loudly before any candle
    is processed (never a silent fallback)."""
    checks = [
        (seat_dir / "ppo.zip", manifest["driver"]["sha256"], "PPO"),
        (seat_dir / "stacker.pkl", manifest["eyes"]["sha256"], "stacker"),
        (seat_dir / "stacker.json", manifest["eyes"]["meta_sha256"],
         "stacker meta"),
    ]
    for path, want, what in checks:
        if not path.exists():
            raise FileNotFoundError(f"HA seat {what} missing: {path}")
        got = _sha256(path)
        if got != want:
            raise ValueError(
                f"HA seat {what} hash mismatch: {path} is {got}, "
                f"manifest pins {want}; refusing to trade on an "
                f"unpinned artifact")


# The one frame an HA seat stacker may read: the 10-value
# Heikin-Ashi candle matrix (feature_set "ha10", Champion #6).
# Anything else is a loud failure -- never a silent fallback to
# another frame.
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix


def seat_columns(seat_dir: Path = SEAT_DIR) -> list[str]:
    """The seated champion's recorded feature columns, from the
    seat's own metadata sidecar -- the exact 10 HA columns the
    crowned Pick #11 stacker reads, in its own order."""
    meta = json.loads((seat_dir / "stacker.json").read_text())
    cols = list(meta.get("feature_columns") or [])
    if meta.get("feature_set") != "ha10" or cols != HA_COLUMNS:
        raise ValueError(
            f"HA seat stacker is not the pinned ha10 artifact "
            f"(feature_set={meta.get('feature_set')!r}, "
            f"columns={len(cols)})")
    return cols


def seat_frame_builder(seat_dir: Path = SEAT_DIR):
    """The HA frame builder: the loop must build the frame the
    SEATED stacker reads -- the Heikin-Ashi candle matrix."""
    meta = json.loads((seat_dir / "stacker.json").read_text())
    if meta.get("feature_set") != "ha10":
        raise ValueError(
            f"HA seat frame requested for a non-ha10 stacker "
            f"(feature_set={meta.get('feature_set')!r})")
    return _ha_frame


def _ha_frame(raw: pd.DataFrame):
    """(frame, columns): the HA matrix over the raw candles."""
    frame = build_ha_matrix(raw)
    return frame[HA_COLUMNS], list(HA_COLUMNS)


def ha_vote(model, ha_frame: pd.DataFrame, ts,
            columns: list[str]) -> float:
    """The seated champion's P(up) for one candle: its own
    predict_proba on the HA frame restricted to its recorded
    columns -- the same computation Pick #11 was graded with."""
    return float(model.predict_proba(ha_frame.loc[[ts], columns])[0, 1])


# ------------------------------------------------------------ the state

def load_state(cfg: HAConfig) -> dict:
    if cfg.state_file.exists():
        try:
            return json.loads(cfg.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            print("live_state_ha.json unreadable; rebuilding from "
                  "the HA ledger", flush=True)
            return rebuild_state_from_ledger(cfg)
    return {"last_ts": None}


def rebuild_state_from_ledger(cfg: HAConfig) -> dict:
    """Mirror of live.py's rebuild, bound to the HA ledger."""
    import sqlite3

    state: dict = {"last_ts": None}
    if not cfg.db_file.exists():
        return state
    con = sqlite3.connect(cfg.db_file)
    try:
        row = con.execute(
            "SELECT cash_after, btc_after FROM trades ORDER BY id DESC "
            "LIMIT 1").fetchone()
        eq = con.execute(
            "SELECT ts FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
        if row is None:
            if eq is not None:
                state["last_ts"] = eq[0]
            return state
        cash, btc = float(row[0]), float(row[1])
        state.update({"cash": cash, "btc": btc,
                      "last_ts": eq[0] if eq else None})
        entry = None
        if btc != 0:
            fills = con.execute(
                "SELECT side, price, qty FROM trades ORDER BY id ASC"
            ).fetchall()
            pos, held, cost = 0.0, 0.0, 0.0
            for side, price, qty in fills:
                d = qty if side == "BUY" else -qty
                if pos == 0.0 or (d > 0) == (pos > 0):
                    cost += price * qty
                    held += qty
                else:
                    if held > 0:
                        cost *= max(0.0, (held - qty) / held)
                    held -= qty
                pos += d
                if abs(pos) <= 1e-10:
                    pos, held, cost = 0.0, 0.0, 0.0
            entry = cost / held if held > 1e-10 else None
        state["entry_price"] = entry
        state["last_entry_price"] = entry
        return state
    finally:
        con.close()


def save_state(cfg: HAConfig, state: dict) -> None:
    import os

    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, cfg.state_file)


def _restore_broker(cfg: HAConfig, state: dict) -> PaperBroker:
    broker = PaperBroker(cfg.db_file, starting_balance=STARTING_BALANCE,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    if state.get("cash") is not None:
        broker.cash = float(state["cash"])
        broker.btc = float(state.get("btc", 0.0))
        broker.entry_price = state.get("entry_price")
        broker.entry_atr_pct = state.get("entry_atr_pct")
        broker.high_water = state.get("high_water")
        broker.low_water = state.get("low_water")
        broker.stop_anchor = state.get("stop_anchor")
    return broker


# ---------------------------------------------------------------- seed

def seed(cfg: HAConfig = DEFAULT_CONFIG, fetch=None) -> dict:
    """One-time launch: seat the pinned artifacts, open Book #4
    FLAT at $100,000, and anchor the loop at the newest CLOSED
    candle -- candles at or before the anchor are never backfilled
    into trades; the desk's first decision is the first candle
    that closes after launch. Idempotent: an already-seeded desk
    is verified, not re-seeded."""
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
        import sqlite3

        con = sqlite3.connect(cfg.db_file)
        try:
            n = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
        except sqlite3.Error:
            n = 0
        con.close()
        if n:
            raise RuntimeError(
                "HA ledger already has equity rows but no state "
                "file; refusing to seed over an inconsistent book")
    broker = PaperBroker(cfg.db_file, starting_balance=STARTING_BALANCE,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    broker.snapshot(anchor_ts, price)  # the flat $100k anchor row
    state = {"last_ts": anchor_ts, "started": anchor_ts,
             "desk": "ha_book4",
             "cash": STARTING_BALANCE, "btc": 0.0,
             "entry_price": None, "last_entry_price": None,
             "entry_atr_pct": None, "high_water": None,
             "low_water": None, "stop_anchor": None,
             "last_stance": None, "last_p_up": None}
    save_state(cfg, state)
    if not cfg.manifest_file.exists():
        cfg.manifest_file.write_text(
            json.dumps(build_manifest(anchor_ts, cfg.seat_dir),
                       indent=1))
    man = load_manifest(cfg)
    verify_seat(man, cfg.seat_dir)
    return {"seeded": True, "started": anchor_ts, "equity": broker.cash,
            "anchor_price": price}


# ------------------------------------------------------------ main loop

def _default_ha_builder(raw: pd.DataFrame):
    """The seat's own HA frame (see seat_frame_builder)."""
    return seat_frame_builder()(raw)


def _eyes_tag(man: dict) -> str:
    """The fill-context tag naming the seated eyes, derived from
    the manifest (e.g. "champion4_run73") -- a hard-coded tag
    went stale at the 2026-10-26 swap (fills kept recording the
    previous eyes), so the label now follows the seat record."""
    import re

    eyes = man.get("eyes") or {}
    m = re.search(r"(\d+)", str(eyes.get("champion") or ""))
    run = eyes.get("run_id")
    return f"champion{m.group(1)}_run{run}" if m else f"run{run}"


def run_once(cfg: HAConfig = DEFAULT_CONFIG, fetch=None,
             stacker=None, ppo=None, ha_builder=None) -> dict:
    if not cfg.state_file.exists() or not cfg.manifest_file.exists():
        raise RuntimeError(
            "HA desk is not seeded; run "
            "python -m src.trading.live_ha --seed first "
            "(the desk never backfills: its first decision is the "
            "first candle closed after launch)")
    man = load_manifest(cfg)
    verify_seat(man, cfg.seat_dir)
    fetch = fetch or fetch_klines
    state = load_state(cfg)
    last_ts = state.get("last_ts")

    # Cheap pre-check (the 730d HA warmup is only worth building
    # when a new closed candle actually exists -- most 15-minute
    # wakes process nothing, exactly like the production loop).
    probe = fetch("BTC-USD", interval="1h", period="5d")
    probe_closed = probe.iloc[:-1]
    if probe_closed.empty:
        raise RuntimeError("market fetch returned no closed candles")
    newest = probe_closed.index[-1]
    if last_ts and pd.Timestamp(last_ts) >= newest:
        return {"processed": 0, "cash": state.get("cash"),
                "btc": state.get("btc")}

    if stacker is None:
        stacker = load_stacker(cfg.seat_dir / "stacker.pkl")
    columns = seat_columns(cfg.seat_dir)
    if ppo is None:
        from stable_baselines3 import PPO
        ppo = PPO.load(cfg.seat_dir / "ppo.zip")
    expected_dim = len(FEATURE_COLUMNS) + 1 + 3
    model_dim = int(ppo.observation_space.shape[0])
    if model_dim != expected_dim:
        return {"processed": 0,
                "skipped": f"HA-desk PPO expects {model_dim}-dim obs, "
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
    # the same reason live.py does (the 4h bucket-closing candle
    # needs the next bucket's first candle to exist); only closed
    # rows are ever processed, and both frames are causal.
    feat = add_features(raw)
    if ha_builder is None:
        ha_builder = seat_frame_builder(cfg.seat_dir)
    ha_frame, _built_cols = ha_builder(raw)
    broker = _restore_broker(cfg, state)
    df = full
    if last_ts:
        df = df[df.index > pd.Timestamp(last_ts)]

    processed = 0
    last_price = 0.0
    for ts, row in df.iterrows():
        frow = feat.loc[ts]
        if frow[FEATURE_COLUMNS].isna().any():
            print(f"candle {ts} has NaN prod21 features; stopping "
                  f"HA run here", flush=True)
            break
        srow = ha_frame.loc[ts, columns]
        if srow.isna().any():
            print(f"candle {ts} has NaN HA features; stopping "
                  f"HA run here", flush=True)
            break
        p_up = ha_vote(stacker, ha_frame, ts, columns)
        price = float(row["close"])
        last_price = price
        # The observation is live.py's, byte for byte: prod21
        # features + the centered vote + the 3 portfolio readings.
        # Only the vote's SOURCE differs (the seated champion's
        # HA eyes instead of the seat stacker's prod21 eyes).
        market = [float(frow[c]) for c in FEATURE_COLUMNS] + [2.0 * p_up - 1.0]
        eq = broker.equity(price)
        obs = np.array(market + [
            broker.cash / max(eq, 1e-9),
            broker.btc * price / max(eq, 1e-9),
            broker.unrealized_pct(price),
        ], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        action, _ = ppo.predict(obs, deterministic=True)
        ts_s = ts.isoformat()
        ctx = json.dumps({
            "f": [round(float(frow[c]), 4) for c in FEATURE_COLUMNS],
            "p": round(p_up, 4), "eyes": _eyes_tag(man)})
        atr_raw = frow.get("atr_pct_raw")
        atr_pct = (float(atr_raw) if atr_raw is not None
                   and np.isfinite(float(atr_raw)) else None)
        if broker.btc != 0 and broker.entry_atr_pct is None:
            broker.entry_atr_pct = atr_pct
        pos_before = broker.btc
        stop_fill = broker.check_stop(price, ts=ts_s, low=float(row["low"]),
                                      high=float(row["high"]), context=ctx)
        fill = None
        if int(action) == 1:
            fill = broker.buy(price, reason="ppo_live", ts=ts_s,
                              atr_pct=atr_pct, context=ctx)
        elif int(action) == 2:
            fill = broker.sell(price, reason="ppo_live", ts=ts_s,
                               context=ctx)
        elif int(action) == 3:
            fill = broker.short_open(price, reason="ppo_live", ts=ts_s,
                                     atr_pct=atr_pct, context=ctx)
        broker.snapshot(ts_s, price)
        # Decision log (instrumentation only, 2026-10-31): what the
        # bot was thinking on this candle, from values already in
        # hand -- the HA row this candle's vote was computed
        # from included. Failure-isolated; never alters trading.
        _d_action, _d_taken, _d_reason = describe_outcome(
            int(action), stop_fill, fill)
        _d_eq = broker.equity(price)
        append_decision("book4", make_record(
            ts=ts_s, close=price, p_up=round(p_up, 4),
            stance=stance_name(int(action)), action=_d_action,
            taken=_d_taken, reason=_d_reason,
            pos_before=pos_before, pos_after=broker.btc,
            equity=round(_d_eq, 2),
            stop_level=broker_stop_level(broker),
            exposure_pct=(round(100.0 * abs(broker.btc) * price
                                / _d_eq, 2) if _d_eq else None),
            features={c: float(srow[c]) for c in columns}),
            decisions_dir=cfg.state_file.parent / "decisions")
        state["last_ts"] = ts_s
        state["last_stance"] = ["hold", "buy", "sell", "short"][int(action)]
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

    return {"processed": processed,
            "equity": broker.equity(last_price) if processed else None,
            "cash": broker.cash, "btc": broker.btc}


if __name__ == "__main__":
    if "--seed" in sys.argv:
        print(json.dumps(seed()), flush=True)
    else:
        print(json.dumps(run_once()), flush=True)
