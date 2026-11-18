"""Coinbase Level 2 order-book collector (launched 2026-11-18).

Maintains the full BTC-USD book from the Coinbase Advanced Trade
WebSocket `level2` channel (public market data, no auth) and stores
ONLY per-minute aggregates in data/l2_book.db (table l2_minutes);
hourly rollups live in data.l2_features / table l2_hours. Raw book
snapshots and the update stream are never stored -- the stream is
thousands of messages a second and worthless to us after aggregation.

This is a data-collection runway for a future pre-registered study.
It touches no trading code and makes no decisions.

Honesty rules (refs/level2-collector-2026-11-18.md):
- The book is trusted only between a fresh snapshot and the next
  disconnect / sequence gap. Any gap tears the session down and the
  book is rebuilt from a new snapshot -- a stale book is never sampled.
- Each minute row's `coverage` is the true number of seconds that
  minute the book was live and consistent (0-60, never rounded up).
- A minute whose book is not live at sample time is still written,
  with NULL features and its true coverage, so "we were down" is
  visible in the data instead of silently absent.

Run as a systemd service (tam-l2-collector):
    .venv/bin/python -m src.data.l2_collector
Liveness for other jobs: data/l2_status.json (rewritten every 10s).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data import l2_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB_PATH = DATA / "l2_book.db"
STATUS_PATH = DATA / "l2_status.json"
LOG_PATH = ROOT / "logs" / "l2_collector.log"

WS_URL = "wss://advanced-trade-ws.coinbase.com"
PRODUCT = "BTC-USD"
WS_STALE_SECS = 20.0          # no message for this long -> session is dead
MAX_BACKOFF_SECS = 30.0

log = logging.getLogger("l2_collector")

MINUTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS l2_minutes (
    ts INTEGER PRIMARY KEY,
    mid REAL, spread_bps REAL, micro_dev_bps REAL,
    imb_10bps REAL, imb_50bps REAL, imb_100bps REAL, imb_200bps REAL,
    depth_1pct_btc REAL,
    best_bid_qty REAL, best_ask_qty REAL,
    n_bid_levels INTEGER, n_ask_levels INTEGER,
    coverage REAL NOT NULL,
    sampled_at INTEGER NOT NULL
)
"""

_MINUTE_COLUMNS = (
    "ts", "mid", "spread_bps", "micro_dev_bps",
    "imb_10bps", "imb_50bps", "imb_100bps", "imb_200bps",
    "depth_1pct_btc", "best_bid_qty", "best_ask_qty",
    "n_bid_levels", "n_ask_levels", "coverage", "sampled_at",
)

# depth bands as (column suffix in bps, fraction of mid)
BANDS = ((10, 0.001), (50, 0.005), (100, 0.01), (200, 0.02))


# ------------------------------------------------------------ book + math

class OrderBook:
    """Full L2 book: price -> BTC quantity, per side."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_snapshot(self, updates: list[dict]) -> None:
        self.bids = {}
        self.asks = {}
        self._apply(updates)

    def apply_update(self, updates: list[dict]) -> None:
        self._apply(updates)

    def _apply(self, updates: list[dict]) -> None:
        for u in updates:
            side = u.get("side")
            if side == "bid":
                book = self.bids
            elif side == "offer":
                book = self.asks
            else:
                continue
            price = float(u["price_level"])
            qty = float(u["new_quantity"])
            if qty == 0.0:
                book.pop(price, None)
            else:
                book[price] = qty

    def best_bid(self) -> tuple[float, float]:
        price = max(self.bids)
        return price, self.bids[price]

    def best_ask(self) -> tuple[float, float]:
        price = min(self.asks)
        return price, self.asks[price]


def book_features(bids: dict[float, float],
                  asks: dict[float, float]) -> dict | None:
    """Per-sample features from a book state; None if the book is unusable.

    Definitions (hand-verified in tests/test_l2_collector.py):
    - mid = (best_bid + best_ask) / 2
    - spread_bps = (best_ask - best_bid) / mid * 1e4
    - microprice = (best_ask*bid_qty + best_bid*ask_qty) / (bid_qty+ask_qty)
      at top of book; micro_dev_bps = (microprice - mid) / mid * 1e4
    - imb_<band> = (bid_qty - ask_qty) / (bid_qty + ask_qty), quantities
      in BTC summed over levels within +-band of mid
    - depth_1pct_btc = total BTC within +-1% of mid, both sides
    An empty side or a crossed book (best_bid >= best_ask) yields None.
    """
    if not bids or not asks:
        return None
    best_bid = max(bids)
    best_ask = min(asks)
    if best_bid >= best_ask:
        return None
    bid_qty = bids[best_bid]
    ask_qty = asks[best_ask]
    mid = (best_bid + best_ask) / 2.0
    micro = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
    out: dict = {
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 1e4,
        "micro_dev_bps": (micro - mid) / mid * 1e4,
        "best_bid_qty": bid_qty,
        "best_ask_qty": ask_qty,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
    }
    for label, frac in BANDS:
        lo = mid * (1.0 - frac)
        hi = mid * (1.0 + frac)
        bq = sum(q for p, q in bids.items() if p >= lo)
        aq = sum(q for p, q in asks.items() if p <= hi)
        out[f"imb_{label}bps"] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
        if label == 100:
            out["depth_1pct_btc"] = bq + aq
    return out


# --------------------------------------------------------------- coverage

class CoverageTracker:
    """Seconds per UTC minute the book was live and consistent.

    Live intervals are banked into per-minute buckets at their true
    boundaries, so a reconnect mid-minute yields a truthfully partial
    coverage for that minute -- never 60, never rounded up.
    """

    def __init__(self) -> None:
        self._live_since: float | None = None
        self._bank: dict[int, float] = {}

    @property
    def is_live(self) -> bool:
        return self._live_since is not None

    def set_live(self, live: bool, ts: float) -> None:
        if live and self._live_since is None:
            self._live_since = ts
        elif not live and self._live_since is not None:
            self._add(self._live_since, ts)
            self._live_since = None

    def _add(self, t0: float, t1: float) -> None:
        while t0 < t1:
            minute = int(t0 // 60) * 60
            seg_end = min(t1, minute + 60.0)
            self._bank[minute] = min(60.0, self._bank.get(minute, 0.0)
                                     + (seg_end - t0))
            t0 = seg_end

    def coverage_for(self, minute_start: int, ts: float) -> float:
        """Finalize one minute: bank the live tail, return its seconds."""
        if self._live_since is not None:
            end = minute_start + 60.0
            if self._live_since < end:
                self._add(self._live_since, end)
                self._live_since = end
        return round(self._bank.pop(minute_start, 0.0), 2)


# ----------------------------------------------------------------- state

class SequenceGapError(Exception):
    """The WS sequence stopped being contiguous: book can no longer be
    trusted, so the session must be torn down and re-snapshotted."""


class CollectorState:
    def __init__(self) -> None:
        self.book = OrderBook()
        self.tracker = CoverageTracker()
        self.started_ts = time.time()
        self.connected = False
        self.has_snapshot = False
        self.last_seq: int | None = None
        self.last_ws_message_ts: float | None = None
        self.last_minute_written: int | None = None
        self.next_minute_to_finalize: int | None = None
        self.reconnect_count = 0
        self.last_error: str | None = None


def _handle_message(state: CollectorState, raw: str, recv_ts: float) -> None:
    msg = json.loads(raw)
    seq = msg.get("sequence_num")
    if isinstance(seq, int):
        if state.last_seq is not None and seq != state.last_seq + 1:
            raise SequenceGapError(
                f"sequence gap: expected {state.last_seq + 1}, got {seq}")
        state.last_seq = seq
    if msg.get("channel") != "l2_data":
        return
    for event in msg.get("events", []):
        if event.get("product_id") not in (None, PRODUCT):
            continue
        updates = event.get("updates") or []
        if event.get("type") == "snapshot":
            state.book.apply_snapshot(updates)
            state.has_snapshot = True
            state.tracker.set_live(True, recv_ts)
            log.info("snapshot applied: %d bid / %d ask levels",
                     len(state.book.bids), len(state.book.asks))
        elif event.get("type") == "update":
            if state.has_snapshot:
                state.book.apply_update(updates)
            # updates before the first snapshot are dropped: the book
            # is not live until a snapshot defines it


# -------------------------------------------------------------- ws session

async def _run_session(state: CollectorState) -> None:
    state.last_seq = None
    async with websockets.connect(
            WS_URL, max_size=2 ** 26, ping_interval=20,
            ping_timeout=20) as ws:
        state.connected = True
        log.info("connected to %s", WS_URL)
        await ws.send(json.dumps({
            "type": "subscribe", "product_ids": [PRODUCT],
            "channel": "level2"}))
        await ws.send(json.dumps({
            "type": "subscribe", "product_ids": [PRODUCT],
            "channel": "heartbeats"}))
        log.info("subscribed: level2 + heartbeats (%s)", PRODUCT)
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=WS_STALE_SECS)
            recv_ts = time.time()
            state.last_ws_message_ts = recv_ts
            _handle_message(state, raw, recv_ts)


async def _supervise(state: CollectorState) -> None:
    backoff = 1.0
    while True:
        session_start = time.time()
        try:
            await _run_session(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # disconnect, gap, timeout, protocol...
            state.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("session ended: %s", state.last_error)
        finally:
            state.connected = False
            state.has_snapshot = False
            state.tracker.set_live(False, time.time())
        state.reconnect_count += 1
        if time.time() - session_start > 60:
            backoff = 1.0
        log.info("reconnecting in %.0fs (reconnect #%d)",
                 backoff, state.reconnect_count)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_SECS)


# --------------------------------------------------------------- sampling

def _finalize_minute(state: CollectorState, conn: sqlite3.Connection,
                     minute_start: int, with_features: bool) -> None:
    coverage = state.tracker.coverage_for(minute_start, time.time())
    feats = None
    if with_features and state.tracker.is_live:
        feats = book_features(state.book.bids, state.book.asks)
    row = {"ts": minute_start, "coverage": coverage,
           "sampled_at": int(time.time())}
    for col in _MINUTE_COLUMNS:
        if col not in row:
            row[col] = feats.get(col) if feats else None
    placeholders = ", ".join("?" for _ in _MINUTE_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO l2_minutes ({', '.join(_MINUTE_COLUMNS)}) "
        f"VALUES ({placeholders})",
        [row[c] for c in _MINUTE_COLUMNS])
    conn.commit()
    state.last_minute_written = minute_start
    if minute_start % 3600 == 3540:
        hour = minute_start - 3540
        rollup = l2_features.rollup_hour(conn, hour)
        if rollup is not None:
            log.info("hour %d rolled up: minutes_present=%d coverage=%.1f "
                     "usable=%d", hour, rollup["minutes_present"],
                     rollup["coverage_mean"] or 0.0, rollup["usable"])


async def _sampler(state: CollectorState, conn: sqlite3.Connection) -> None:
    while True:
        now = time.time()
        boundary = (int(now // 60) + 1) * 60
        await asyncio.sleep(max(0.05, boundary - now + 0.25))
        upto = boundary - 60
        if state.next_minute_to_finalize is None:
            state.next_minute_to_finalize = upto
        minute = state.next_minute_to_finalize
        while minute <= upto:
            # only the just-ended minute can be sampled from the CURRENT
            # book; any older missed minute gets NULL features with its
            # true coverage rather than a retroactively invented book
            _finalize_minute(state, conn, minute,
                             with_features=(minute == upto))
            minute += 60
        state.next_minute_to_finalize = upto + 60


# ----------------------------------------------------------------- status

def _write_status(state: CollectorState) -> None:
    payload = {
        "connected": state.connected,
        "has_snapshot": state.has_snapshot,
        "book_live": state.tracker.is_live,
        "last_ws_message_ts": state.last_ws_message_ts,
        "last_minute_written": state.last_minute_written,
        "bid_levels": len(state.book.bids),
        "ask_levels": len(state.book.asks),
        "reconnect_count": state.reconnect_count,
        "last_error": state.last_error,
        "started_ts": state.started_ts,
        "updated_ts": time.time(),
    }
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(STATUS_PATH)


async def _status_writer(state: CollectorState) -> None:
    while True:
        try:
            _write_status(state)
        except Exception as exc:
            log.warning("status write failed: %s", exc)
        await asyncio.sleep(10)


# -------------------------------------------------------------------- run

async def run() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    setup_conn = sqlite3.connect(str(DB_PATH))
    setup_conn.execute(MINUTE_SCHEMA)
    l2_features.init_hours_table(setup_conn)
    setup_conn.close()
    rolled = l2_features.backfill_rollups(DB_PATH)
    if rolled:
        log.info("startup backfill rolled up hours: %s", rolled)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    state = CollectorState()
    log.info("collector starting (db=%s)", DB_PATH)
    await asyncio.gather(
        _supervise(state),
        _sampler(state, conn),
        _status_writer(state),
    )


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("stopped by KeyboardInterrupt")


if __name__ == "__main__":
    main()
