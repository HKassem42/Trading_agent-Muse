"""Champions live loop (Book #5) -- Muse's book, Kass's order,
2026-11-12.

Kass gave Muse full freedom over this book ("I want you to pick
everything about book #5... You have FULL FREE ACCESS to however
you want it to be and look and trade and rules"). Muse's picks:

    Seat policy: FOLLOWS THE CROWN. This book always trades the
    sitting LAB champions as a pair --
        Driver: the lab PPO champion   (run 112 at launch)
        Eyes:   the lab stacker champion (run 126, Champion #13,
                ha10 frame, at launch)
    The seat holds PINNED byte copies in models/champ_seat/ and
    is re-pinned by Muse as part of each crowning (the crowning
    Kass delegated to Muse on 2026-11-12), carry-over style: an
    open position rides through a re-pin, exactly like Book #2's
    eyes swaps. The manifest (data/champ_seat.json) records every
    seat era. Between crowns the seat never drifts: the loop
    verifies the pinned copies against the manifest on every run
    and fails loudly on any mismatch -- never a fallback.

    Rules: the production broker chassis, exactly as the pair is
    graded in the lab (25% tranche, 75% exposure cap, 0.15% fees,
    ATR stops/trailing with the 2% trail arm). No invented
    overlays: this book's job is to give the lab's best pair a
    clean forward track record, so it trades the pair AS GRADED.

    Decision timing: this book decides on hourly CLOSES, like
    every other book. An open-decision experiment (Kass's order,
    2026-11-13) ran from MODE_FROM_TS until Kass ordered it
    ended on 2026-11-17: during that window the pair read the
    last CLOSED candle's features at each hourly boundary and
    filled at the new candle's OPEN (the fill convention the
    pair was graded with in its duels), with protective
    stops/trailing still evaluating on each candle at its close,
    before the next open decision. Candles in that window keep
    the open-mode record they already have; candles after the
    revert are close-mode again. _run_once_open is kept intact
    as the experiment's code and a rollback-forward path.

PERIMETER: this module WRITES only its own files:
    models/champ_seat/          pinned copies of the lab champions
    data/champ_seat.json        the seat manifest (hashes, lineage)
    data/paper_champ.db         Book #5's own ledger
    data/live_state_champ.json  Book #5's own loop state
    data/decisions/book5.jsonl  its decision log (via decision_log)
It never opens another book's file for writing. It READS the lab
champion files (models/stacker_champion.pkl/.json,
models/ppo_champion.zip) only inside ensure_seat()/repin() --
the crowning-time seat refresh -- never during a trading step.

The eyes' frame follows the SEATED stacker champion's own
feature_set tag: "ha10" reads the Heikin-Ashi candle matrix;
"prod21" reads the production 21 features. Any other frame is a
loud failure (extend deliberately when a champion earns it).
The PPO driver always reads the production 21 features plus the
eyes' vote plus the 3 portfolio readings -- the observation every
book's driver is graded and traded with.

Usage:
    python -m src.trading.live_champ --seed   # one-time launch
    python -m src.trading.live_champ --repin  # crowning-time seat refresh
    python -m src.trading.live_champ          # one loop step
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
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

SEAT_DIR = MODELS / "champ_seat"
SEAT_STACKER = SEAT_DIR / "stacker.pkl"
SEAT_STACKER_META = SEAT_DIR / "stacker.json"
SEAT_PPO = SEAT_DIR / "ppo.zip"

# Canonical sources: the LAB champion files themselves. They move
# only when a crowning installs a new champion; this seat copies
# them at seed time and at each crowning-time re-pin.
SOURCE_STACKER = MODELS / "stacker_champion.pkl"
SOURCE_STACKER_META = MODELS / "stacker_champion.json"
SOURCE_PPO = MODELS / "ppo_champion.zip"
REGISTRY = DATA / "training.db"

STARTING_BALANCE = 100_000.0
DESK_NAME = "TradingAgent Live — Champions (Book #5)"

# Decision timing for this book only (see the module docstring).
# "open" (Kass's order, 2026-11-13): decisions fired at each
# candle's open from MODE_FROM_TS onward -- the open-decision
# experiment, which ran until Kass ordered it ended on
# 2026-11-17. "close" (the original loop, restored 2026-11-17):
# decisions fire on closed candles, like every other book.
# _run_once_open stays intact below as history / rollback-forward.
DECISION_MODE = "close"
MODE_FROM_TS = "2026-11-13T20:00:00"


@dataclass(frozen=True)
class ChampConfig:
    db_file: Path = DATA / "paper_champ.db"
    state_file: Path = DATA / "live_state_champ.json"
    seat_dir: Path = SEAT_DIR
    manifest_file: Path = DATA / "champ_seat.json"


DEFAULT_CONFIG = ChampConfig()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------- the lab champions

def lab_champions() -> dict:
    """Who holds the lab crowns right now, from the governance
    records: the stacker champion's own meta file (written by the
    crowning) and the registry's newest promoted PPO run."""
    smeta = json.loads(SOURCE_STACKER_META.read_text())
    eyes = {
        "run_id": smeta["run_id"],
        "champion_no": smeta.get("champion_no"),
        "champion": (f"Champion #{smeta['champion_no']}"
                     if smeta.get("champion_no") else None),
        "feature_set": smeta.get("feature_set"),
        "feature_columns": list(smeta.get("feature_columns") or []),
        "accuracy": smeta.get("accuracy"),
        "precision": smeta.get("precision"),
        "wf_accuracy": smeta.get("wf_accuracy"),
        "display": smeta.get("display"),
    }
    con = sqlite3.connect(REGISTRY)
    try:
        row = con.execute(
            "SELECT id, metrics FROM runs WHERE kind='ppo' "
            "AND promoted=1 ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        con.close()
    if row is None:
        raise RuntimeError("registry has no promoted PPO champion")
    ppo_run, ppo_metrics = int(row[0]), json.loads(row[1])
    ch = ppo_metrics.get("challenger") or {}
    driver = {
        "run_id": ppo_run,
        "duel_score": ch.get("score"),
        "duel_pnl": ch.get("total_pnl"),
        "duel_windows": [w.get("score") for w in ch.get("windows", [])],
        "duel_bench_pnl": ch.get("bench_pnl"),
        "reason": ppo_metrics.get("reason"),
    }
    return {"driver": driver, "eyes": eyes}


# ------------------------------------------------------------- the seat

def ensure_seat(seat_dir: Path = SEAT_DIR) -> None:
    """Copy the current lab champion artifacts into the seat if
    absent (seed time). An existing seat is NOT compared against
    the sources -- between crowns the sources legitimately move
    ahead of the pin; the manifest is the seat's truth and
    verify_seat() guards it. A seat that exists without a manifest
    is a loud failure."""
    seat_dir.mkdir(parents=True, exist_ok=True)
    pairs = ((seat_dir / "stacker.pkl", SOURCE_STACKER),
             (seat_dir / "stacker.json", SOURCE_STACKER_META),
             (seat_dir / "ppo.zip", SOURCE_PPO))
    if all(dest.exists() for dest, _src in pairs):
        return
    for dest, src in pairs:
        if not src.exists():
            raise FileNotFoundError(f"champ seat source missing: {src}")
        if not dest.exists():
            shutil.copyfile(src, dest)


def build_manifest(started: str, seat_dir: Path = SEAT_DIR,
                   lineage: list | None = None) -> dict:
    """The champ seat ledger (data/champ_seat.json): the pinned
    pair, hash-pinned, with the follows-the-crown doctrine and
    the full seat-era lineage."""
    champs = lab_champions()
    drv, eyes = champs["driver"], champs["eyes"]
    meta = json.loads((seat_dir / "stacker.json").read_text())
    if meta.get("run_id") != eyes["run_id"]:
        raise ValueError(
            "seat stacker meta does not match the lab champion "
            f"record (seat run {meta.get('run_id')} vs lab "
            f"{eyes['run_id']}); re-pin the seat first")
    era = {
        "era": (lineage[-1]["era"] + 1) if lineage else 1,
        "pinned_at": _now(),
        "driver_run": drv["run_id"],
        "eyes_run": eyes["run_id"],
        "eyes_champion": eyes["champion"],
    }
    lineage = (lineage or []) + [era]
    return {
        "desk": DESK_NAME,
        "book": "Book #5",
        "seat_policy": "follows_the_crown",
        "driver": {
            "brain": "PPO", "run_id": drv["run_id"],
            "file": "models/champ_seat/ppo.zip",
            "sha256": _sha256(seat_dir / "ppo.zip"),
            "source": "models/ppo_champion.zip",
        },
        "eyes": {
            "brain": "Stacker", "run_id": eyes["run_id"],
            "name": eyes.get("display"),
            "champion": eyes["champion"],
            "feature_set": eyes["feature_set"],
            "n_features": len(eyes["feature_columns"]),
            "file": "models/champ_seat/stacker.pkl",
            "sha256": _sha256(seat_dir / "stacker.pkl"),
            "meta_file": "models/champ_seat/stacker.json",
            "meta_sha256": _sha256(seat_dir / "stacker.json"),
            "source": "models/stacker_champion.pkl",
        },
        "book_db": "data/paper_champ.db",
        "state_file": "data/live_state_champ.json",
        "starting_balance": int(STARTING_BALANCE),
        "started": started,
        "seeded_by": "Book #5 Champions build (Kass's full-freedom "
                     "order 2026-11-12: Muse picks everything "
                     "about this book)",
        "grading_record": {
            "driver_run": drv["run_id"],
            "driver_duel_score": drv["duel_score"],
            "driver_duel_pnl": drv["duel_pnl"],
            "driver_duel_windows": drv["duel_windows"],
            "driver_duel_bench_pnl": drv["duel_bench_pnl"],
            "driver_reason": drv["reason"],
            "eyes_run": eyes["run_id"],
            "eyes_exam_acc": eyes["accuracy"],
            "eyes_precision": eyes["precision"],
            "eyes_wf_acc": eyes["wf_accuracy"],
            "source": "Lab crowning records: the PPO champion's "
                      "own crowning duel (registry) and the "
                      "stacker champion's crowning exam (its "
                      "champion meta). The PAIR has no joint "
                      "exam; this book IS its forward test.",
        },
        "lineage": lineage,
        "pinned": "FOLLOWS THE CROWN: the seat holds pinned byte "
                  "copies of the sitting lab champions and is "
                  "re-pinned by Muse at each crowning (Kass "
                  "delegated crowning to Muse 2026-11-12), "
                  "carry-over style -- an open position rides "
                  "through a re-pin. Between crowns the pinned "
                  "copies are the truth and are hash-verified on "
                  "every run; a mismatch is a loud failure, "
                  "never a silent fallback.",
        "note": "Book #5: Muse's book. The sitting lab "
                "champions trade as a pair -- the PPO champion "
                "drives, the stacker champion's vote is its eyes "
                "-- under the production broker chassis exactly "
                "as graded. When a crown changes hands, the new "
                "champion takes this seat at the next re-pin and "
                "the lineage above grows an era.",
    }


def load_manifest(cfg: ChampConfig = DEFAULT_CONFIG) -> dict:
    if not cfg.manifest_file.exists():
        raise FileNotFoundError(
            f"champ manifest missing: {cfg.manifest_file} "
            f"(seed the desk first: python -m src.trading."
            f"live_champ --seed)")
    return json.loads(cfg.manifest_file.read_text())


def verify_seat(manifest: dict, seat_dir: Path) -> None:
    """Every champ seat file must hash to the manifest's pin --
    a swapped or corrupted artifact fails loudly before any
    candle is processed (never a silent fallback)."""
    checks = [
        (seat_dir / "ppo.zip", manifest["driver"]["sha256"], "PPO"),
        (seat_dir / "stacker.pkl", manifest["eyes"]["sha256"], "stacker"),
        (seat_dir / "stacker.json", manifest["eyes"]["meta_sha256"],
         "stacker meta"),
    ]
    for path, want, what in checks:
        if not path.exists():
            raise FileNotFoundError(f"champ seat {what} missing: {path}")
        got = _sha256(path)
        if got != want:
            raise ValueError(
                f"champ seat {what} hash mismatch: {path} is {got}, "
                f"manifest pins {want}; refusing to trade on an "
                f"unpinned artifact")


def repin(cfg: ChampConfig = DEFAULT_CONFIG) -> dict:
    """Crowning-time seat refresh: copy the CURRENT lab champion
    artifacts over the seat and rewrite the manifest with a new
    lineage era. Book state (position, cash, stop) is untouched --
    the new pair inherits the book exactly as it stands."""
    man = load_manifest(cfg)
    champs = lab_champions()
    if (champs["driver"]["run_id"] == man["driver"]["run_id"]
            and champs["eyes"]["run_id"] == man["eyes"]["run_id"]):
        verify_seat(man, cfg.seat_dir)
        return {"repinned": False,
                "driver_run": man["driver"]["run_id"],
                "eyes_run": man["eyes"]["run_id"]}
    for dest, src in ((cfg.seat_dir / "stacker.pkl", SOURCE_STACKER),
                      (cfg.seat_dir / "stacker.json",
                       SOURCE_STACKER_META),
                      (cfg.seat_dir / "ppo.zip", SOURCE_PPO)):
        if not src.exists():
            raise FileNotFoundError(f"champ seat source missing: {src}")
        shutil.copyfile(src, dest)
    new_man = build_manifest(man["started"], cfg.seat_dir,
                             lineage=man.get("lineage") or [])
    cfg.manifest_file.write_text(json.dumps(new_man, indent=1))
    verify_seat(new_man, cfg.seat_dir)
    return {"repinned": True,
            "driver_run": new_man["driver"]["run_id"],
            "eyes_run": new_man["eyes"]["run_id"],
            "era": new_man["lineage"][-1]["era"]}


# ------------------------------------------------------------ the frames

def seat_columns(seat_dir: Path = SEAT_DIR) -> list[str]:
    """The seated champion's recorded feature columns, from the
    seat's own metadata sidecar, in its own order."""
    meta = json.loads((seat_dir / "stacker.json").read_text())
    cols = list(meta.get("feature_columns") or [])
    if not cols:
        raise ValueError("champ seat stacker meta has no "
                         "feature_columns")
    return cols


def seat_feature_set(seat_dir: Path = SEAT_DIR) -> str:
    meta = json.loads((seat_dir / "stacker.json").read_text())
    return str(meta.get("feature_set") or "")


def eyes_frame(raw: pd.DataFrame, feature_set: str,
               feat: pd.DataFrame | None = None):
    """(frame, columns) the seated eyes read. ha10: the
    Heikin-Ashi candle matrix. prod21: the production feature
    frame (pass the already-computed one to reuse it). Anything
    else is a loud failure -- never a silent fallback."""
    if feature_set == "ha10":
        from models.muse_pick10 import HA_COLUMNS, build_ha_matrix

        frame = build_ha_matrix(raw)
        return frame[HA_COLUMNS], list(HA_COLUMNS)
    if feature_set == "prod21":
        frame = feat if feat is not None else add_features(raw)
        return frame[FEATURE_COLUMNS], list(FEATURE_COLUMNS)
    raise ValueError(
        f"champ seat eyes frame {feature_set!r} is not supported "
        f"by Book #5's engine (ha10 / prod21); extend deliberately "
        f"when a champion earns a new frame")


def eyes_vote(model, frame: pd.DataFrame, ts,
              columns: list[str]) -> float:
    """The seated champion's P(up) for one candle: its own
    predict_proba on its own frame restricted to its recorded
    columns -- the same computation it was graded with."""
    return float(model.predict_proba(frame.loc[[ts], columns])[0, 1])


# ------------------------------------------------------------ the state

def load_state(cfg: ChampConfig) -> dict:
    if cfg.state_file.exists():
        try:
            return json.loads(cfg.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            print("live_state_champ.json unreadable; rebuilding "
                  "from the champ ledger", flush=True)
            return rebuild_state_from_ledger(cfg)
    return {"last_ts": None}


def rebuild_state_from_ledger(cfg: ChampConfig) -> dict:
    """Mirror of live.py's rebuild, bound to the champ ledger."""
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


def save_state(cfg: ChampConfig, state: dict) -> None:
    import os

    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, cfg.state_file)


def _restore_broker(cfg: ChampConfig, state: dict) -> PaperBroker:
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

def seed(cfg: ChampConfig = DEFAULT_CONFIG, fetch=None) -> dict:
    """One-time launch: seat the current lab champions, open
    Book #5 FLAT at $100,000, and anchor the loop at the newest
    CLOSED candle -- candles at or before the anchor are never
    backfilled into trades. Idempotent: an already-seeded desk
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
        con = sqlite3.connect(cfg.db_file)
        try:
            n = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
        except sqlite3.Error:
            n = 0
        con.close()
        if n:
            raise RuntimeError(
                "champ ledger already has equity rows but no state "
                "file; refusing to seed over an inconsistent book")
    broker = PaperBroker(cfg.db_file, starting_balance=STARTING_BALANCE,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    broker.snapshot(anchor_ts, price)  # the flat $100k anchor row
    state = {"last_ts": anchor_ts, "started": anchor_ts,
             "desk": "champ_book5",
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

def _eyes_tag(man: dict) -> str:
    """The fill-context tag naming the seated eyes, derived from
    the manifest (e.g. "champion13_run126") -- the label follows
    the seat record, never a hard-coded string."""
    import re

    eyes = man.get("eyes") or {}
    m = re.search(r"(\d+)", str(eyes.get("champion") or ""))
    run = eyes.get("run_id")
    return f"champion{m.group(1)}_run{run}" if m else f"run{run}"


# --------------------------------------- open-decision mode (Book #5)

def _persist_open_state(cfg: ChampConfig, state: dict,
                        broker: PaperBroker) -> None:
    """Save the state with the broker's fields refreshed from the
    broker itself (the same fields the close-mode loop saves)."""
    state["cash"] = broker.cash
    state["btc"] = broker.btc
    state["entry_price"] = broker.entry_price
    state["last_entry_price"] = broker.entry_price
    state["entry_atr_pct"] = broker.entry_atr_pct
    state["high_water"] = broker.high_water
    state["low_water"] = broker.low_water
    state["stop_anchor"] = broker.stop_anchor
    save_state(cfg, state)


def _run_once_open(cfg: ChampConfig, fetch, stacker, ppo,
                   man: dict, state: dict) -> dict:
    """One loop step in open-decision mode (Kass's order,
    2026-11-13). Two event kinds, processed in chronological
    order across every candle the book has not yet handled:

    DECIDE at a candle's open (candles >= MODE_FROM_TS): the
    pair reads the PREVIOUS candle's features and vote -- the
    last closed candle at that boundary -- and fills at this
    candle's open. The observation, vote and broker dispatch are
    the close-mode loop's, byte for byte; only the feature
    candle (previous) and the fill price (open) differ.

    STOP at a candle's close (every closed candle): the
    production broker's wick stop/trailing check for that
    candle, exactly as the close-mode loop runs it, plus the
    candle's close-mark equity row. A candle's STOP always runs
    before the next candle's DECIDE (the exit happens inside
    the candle's hour; the decision at its boundary).

    State pointers: last_open_ts (newest candle decided) and
    last_stop_ts (newest candle stop-checked), both defaulting
    to the close-mode last_ts at the switch. last_ts keeps its
    feed meaning -- the candle the recorded vote (last_p_up)
    was computed on -- so the desk's own-candle pin stays exact.
    """
    mode_from = pd.Timestamp(MODE_FROM_TS)
    last_open = state.get("last_open_ts") or state.get("last_ts")
    last_stop = state.get("last_stop_ts") or state.get("last_ts")
    last_open_ts = pd.Timestamp(last_open) if last_open else None
    last_stop_ts = pd.Timestamp(last_stop) if last_stop else None

    # Cheap pre-check, mirroring the close-mode loop: the 730d
    # warmup is only worth building when an event is actually due.
    probe = fetch("BTC-USD", interval="1h", period="5d")
    if probe.empty or len(probe) < 2:
        raise RuntimeError("market fetch returned no closed candles")
    probe_closed = probe.iloc[:-1]
    newest_probe = probe_closed.index[-1]
    forming_probe = probe.index[-1]
    work_due = (
        (last_stop_ts is None or newest_probe > last_stop_ts)
        or (forming_probe >= mode_from
            and (last_open_ts is None or forming_probe > last_open_ts))
        or (newest_probe >= mode_from
            and (last_open_ts is None or newest_probe > last_open_ts)))
    if not work_due:
        return {"processed": 0, "cash": state.get("cash"),
                "btc": state.get("btc")}

    if stacker is None:
        stacker = load_stacker(cfg.seat_dir / "stacker.pkl")
    columns = seat_columns(cfg.seat_dir)
    fset = seat_feature_set(cfg.seat_dir)
    if ppo is None:
        from stable_baselines3 import PPO
        ppo = PPO.load(cfg.seat_dir / "ppo.zip")
    expected_dim = len(FEATURE_COLUMNS) + 1 + 3
    model_dim = int(ppo.observation_space.shape[0])
    if model_dim != expected_dim:
        return {"processed": 0,
                "skipped": f"champ-desk PPO expects {model_dim}-dim "
                           f"obs, recipe builds {expected_dim}-dim"}

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
    # Features on the FULL frame including the forming candle,
    # exactly as the close-mode loop builds them; decisions read
    # only closed rows, stop checks read their own closed candle.
    feat = add_features(raw)
    eframe, ecols = eyes_frame(raw, fset, feat=feat)
    if list(ecols) != list(columns):
        raise ValueError(
            f"eyes frame columns {list(ecols)} do not match the "
            f"seat's recorded columns {list(columns)}")
    broker = _restore_broker(cfg, state)
    closed_set = set(full.index)

    events = []
    for ts in raw.index:
        if ts >= mode_from and (last_open_ts is None
                                or ts > last_open_ts):
            events.append((ts, 1, "decide", ts))
        if ts in closed_set and (last_stop_ts is None
                                 or ts > last_stop_ts):
            events.append((ts + pd.Timedelta(hours=1), 0, "stop", ts))
    events.sort(key=lambda e: (e[0], e[1]))

    index = list(raw.index)
    processed = 0
    last_price = 0.0
    for _when, _ord, kind, ts in events:
        ts_s = ts.isoformat()
        if kind == "decide":
            loc = raw.index.get_loc(ts)
            if loc == 0:
                continue
            prev = index[loc - 1]
            frow = feat.loc[prev]
            if frow[FEATURE_COLUMNS].isna().any():
                print(f"candle {prev} has NaN prod21 features; "
                      f"stopping champ run here", flush=True)
                break
            srow = eframe.loc[prev, columns]
            if srow.isna().any():
                print(f"candle {prev} has NaN eyes features; "
                      f"stopping champ run here", flush=True)
                break
            p_up = eyes_vote(stacker, eframe, prev, columns)
            price = float(raw.loc[ts, "open"])
            last_price = price
            # The observation is live.py's, byte for byte -- over
            # the PREVIOUS candle's readings, marked at this
            # candle's open (the price the fill executes at).
            market = [float(frow[c]) for c in FEATURE_COLUMNS] \
                + [2.0 * p_up - 1.0]
            eq = broker.equity(price)
            obs = np.array(market + [
                broker.cash / max(eq, 1e-9),
                broker.btc * price / max(eq, 1e-9),
                broker.unrealized_pct(price),
            ], dtype=np.float32)
            obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
            action, _ = ppo.predict(obs, deterministic=True)
            ctx = json.dumps({
                "f": [round(float(frow[c]), 4) for c in FEATURE_COLUMNS],
                "p": round(p_up, 4), "eyes": _eyes_tag(man)})
            atr_raw = frow.get("atr_pct_raw")
            atr_pct = (float(atr_raw) if atr_raw is not None
                       and np.isfinite(float(atr_raw)) else None)
            if broker.btc != 0 and broker.entry_atr_pct is None:
                broker.entry_atr_pct = atr_pct
            pos_before = broker.btc
            fill = None
            if int(action) == 1:
                fill = broker.buy(price, reason="ppo_live", ts=ts_s,
                                  atr_pct=atr_pct, context=ctx)
            elif int(action) == 2:
                fill = broker.sell(price, reason="ppo_live", ts=ts_s,
                                   context=ctx)
            elif int(action) == 3:
                fill = broker.short_open(price, reason="ppo_live",
                                         ts=ts_s, atr_pct=atr_pct,
                                         context=ctx)
            broker.snapshot(ts_s, price)
            _d_action, _d_taken, _d_reason = describe_outcome(
                int(action), None, fill)
            _d_eq = broker.equity(price)
            append_decision("book5", make_record(
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
            state["last_open_ts"] = ts_s
            state["last_ts"] = prev.isoformat()
            state["last_stance"] = ["hold", "buy", "sell",
                                    "short"][int(action)]
            state["last_p_up"] = round(p_up, 4)
            _persist_open_state(cfg, state, broker)
            processed += 1
            continue

        # kind == "stop": this closed candle's protective check.
        row = raw.loc[ts]
        price = float(row["close"])
        last_price = price
        # The stop fill's context is this candle's own readings
        # when they compute cleanly; the stop itself must run
        # regardless -- a protective exit never waits on features.
        ctx = json.dumps({"eyes": _eyes_tag(man)})
        p_ctx = state.get("last_p_up")
        feats_ctx = None
        try:
            frow_c = feat.loc[ts]
            srow_c = eframe.loc[ts, columns]
            if not frow_c[FEATURE_COLUMNS].isna().any() \
                    and not srow_c.isna().any():
                p_ctx = eyes_vote(stacker, eframe, ts, columns)
                ctx = json.dumps({
                    "f": [round(float(frow_c[c]), 4)
                          for c in FEATURE_COLUMNS],
                    "p": round(p_ctx, 4), "eyes": _eyes_tag(man)})
                feats_ctx = {c: float(srow_c[c]) for c in columns}
                atr_raw_c = frow_c.get("atr_pct_raw")
                if broker.btc != 0 and broker.entry_atr_pct is None \
                        and atr_raw_c is not None \
                        and np.isfinite(float(atr_raw_c)):
                    broker.entry_atr_pct = float(atr_raw_c)
        except Exception:
            pass
        pos_before = broker.btc
        stop_fill = broker.check_stop(
            price, ts=ts_s, low=float(row["low"]),
            high=float(row["high"]), context=ctx)
        broker.snapshot(ts_s, price)
        if stop_fill is not None:
            _d_action, _d_taken, _d_reason = describe_outcome(
                0, stop_fill, None)
            _d_eq = broker.equity(price)
            append_decision("book5", make_record(
                ts=ts_s, close=price,
                p_up=round(float(p_ctx), 4) if p_ctx is not None
                else 0.5,
                stance=stance_name(0), action=_d_action,
                taken=_d_taken, reason=_d_reason,
                pos_before=pos_before, pos_after=broker.btc,
                equity=round(_d_eq, 2),
                stop_level=broker_stop_level(broker),
                exposure_pct=(round(100.0 * abs(broker.btc) * price
                                    / _d_eq, 2) if _d_eq else None),
                features=feats_ctx),
                decisions_dir=cfg.state_file.parent / "decisions",
                allow_duplicate_ts=True)
        state["last_stop_ts"] = ts_s
        _persist_open_state(cfg, state, broker)
        processed += 1

    return {"processed": processed,
            "equity": broker.equity(last_price) if processed else None,
            "cash": broker.cash, "btc": broker.btc, "mode": "open"}


def run_once(cfg: ChampConfig = DEFAULT_CONFIG, fetch=None,
             stacker=None, ppo=None) -> dict:
    if not cfg.state_file.exists() or not cfg.manifest_file.exists():
        raise RuntimeError(
            "champ desk is not seeded; run "
            "python -m src.trading.live_champ --seed first "
            "(the desk never backfills: its first decision is the "
            "first candle closed after launch)")
    man = load_manifest(cfg)
    verify_seat(man, cfg.seat_dir)
    fetch = fetch or fetch_klines
    state = load_state(cfg)
    last_ts = state.get("last_ts")

    if DECISION_MODE == "open":
        return _run_once_open(cfg, fetch=fetch, stacker=stacker,
                              ppo=ppo, man=man, state=state)

    # Cheap pre-check (the 730d warmup is only worth building
    # when a new closed candle actually exists -- most 15-minute
    # wakes process nothing, exactly like the other loops).
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
    fset = seat_feature_set(cfg.seat_dir)
    if ppo is None:
        from stable_baselines3 import PPO
        ppo = PPO.load(cfg.seat_dir / "ppo.zip")
    expected_dim = len(FEATURE_COLUMNS) + 1 + 3
    model_dim = int(ppo.observation_space.shape[0])
    if model_dim != expected_dim:
        return {"processed": 0,
                "skipped": f"champ-desk PPO expects {model_dim}-dim "
                           f"obs, recipe builds {expected_dim}-dim"}

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
    eframe, ecols = eyes_frame(raw, fset, feat=feat)
    if list(ecols) != list(columns):
        raise ValueError(
            f"eyes frame columns {list(ecols)} do not match the "
            f"seat's recorded columns {list(columns)}")
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
                  f"champ run here", flush=True)
            break
        srow = eframe.loc[ts, columns]
        if srow.isna().any():
            print(f"candle {ts} has NaN eyes features; stopping "
                  f"champ run here", flush=True)
            break
        p_up = eyes_vote(stacker, eframe, ts, columns)
        price = float(row["close"])
        last_price = price
        # The observation is live.py's, byte for byte: prod21
        # features + the centered vote + the 3 portfolio readings.
        # Only the vote's SOURCE differs (the seated stacker
        # champion's eyes instead of the seat stacker's).
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
        # Decision log (instrumentation only): what the pair was
        # thinking on this candle, from values already in hand --
        # the eyes row this candle's vote was computed from
        # included. Failure-isolated; never alters trading.
        _d_action, _d_taken, _d_reason = describe_outcome(
            int(action), stop_fill, fill)
        _d_eq = broker.equity(price)
        append_decision("book5", make_record(
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
    elif "--repin" in sys.argv:
        print(json.dumps(repin()), flush=True)
    else:
        print(json.dumps(run_once()), flush=True)
