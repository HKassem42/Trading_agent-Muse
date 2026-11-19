"""Coinbase Level 2 order-book collector (launched 2026-11-18).

Maintains the full BTC-USD book from the Coinbase Advanced Trade
WebSocket `level2` channel (public market data, no auth) and stores
ONLY per-minute aggregates in data/l2_book.db (table l2_minutes);
hourly rollups live in data.l2_features / table l2_hours. Raw book
snapshots and the update stream are never stored -- the stream is
thousands of messages a second and worthless to us after aggregation.

The same connection also carries the trades tape (`market_trades`
channel, added 2026-11-18): every executed trade, aggregated per
minute into l2_trades_minutes / l2_trades_hours. SIDE CONVENTION:
Coinbase documents the trade `side` as the MAKER (resting) order's
side, so all buy/sell accounting inverts it -- a print of side=BUY is
SELL-initiated volume (a resting buy was hit), side=SELL is
BUY-initiated (a resting sell was lifted).

This is a data-collection runway for a future pre-registered study.
It touches no trading code and makes no decisions.

Honesty rules (refs/level2-collector-2026-11-18.md):
- The book is trusted only between a fresh snapshot and the next
  disconnect / sequence gap. Any gap tears the session down and the
  book is rebuilt from a new snapshot -- a stale book is never sampled.
- Each minute row's `coverage` is the true number of seconds that
  minute the stream was live (0-60, never rounded up). The tape has
  its own coverage: the channel has no snapshot to rebuild from, so
  trades during an outage are simply lost and the coverage says so.
- A minute whose stream is not live at sample time is still written,
  with NULL features and its true coverage, so "we were down" is
  visible in the data instead of silently absent. A live minute with
  zero trades is real data and is written with ZEROS, not NULLs.

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
from collections import deque
from datetime import datetime, timezone
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


# ------------------------------------------------------------ trades tape

TRADES_MINUTE_SCHEMA = """
CREATE TABLE IF NOT EXISTS l2_trades_minutes (
    ts INTEGER PRIMARY KEY,
    buy_vol REAL, sell_vol REAL, total_vol REAL,
    n_trades INTEGER, n_buy INTEGER, n_sell INTEGER,
    vwap REAL, max_qty REAL, large_vol REAL,
    coverage REAL NOT NULL,
    sampled_at INTEGER NOT NULL
)
"""

_TRADES_MINUTE_COLUMNS = (
    "ts", "buy_vol", "sell_vol", "total_vol", "n_trades", "n_buy",
    "n_sell", "vwap", "max_qty", "large_vol", "coverage", "sampled_at",
)

LARGE_TRADE_BTC = 0.5        # a "large" print: qty >= 0.5 BTC
_SEEN_IDS_CAP = 300_000      # trade-id dedupe window (ids are unique)


def taker_side(printed_side) -> str | None:
    """Invert Coinbase's maker-side print to the aggressor (taker) side.

    market_trades documents `side` as the MAKER (resting) order's
    side: side=BUY means a resting buy was hit by a seller (taker
    SELL); side=SELL means a resting sell was lifted (taker BUY).
    Anything else is not a venue print we recognize -> None.
    """
    if printed_side == "BUY":
        return "SELL"
    if printed_side == "SELL":
        return "BUY"
    return None


class TapeAccumulator:
    """Running aggregates for one minute of executed trades.

    All volumes are TAKER-side (see taker_side): buy_vol is volume
    where the aggressor bought. `vwap` is over the minute's trades.
    """

    __slots__ = ("buy_vol", "sell_vol", "n_trades", "n_buy", "n_sell",
                 "_sum_pv", "max_qty", "large_vol")

    def __init__(self) -> None:
        self.buy_vol = 0.0
        self.sell_vol = 0.0
        self.n_trades = 0
        self.n_buy = 0
        self.n_sell = 0
        self._sum_pv = 0.0
        self.max_qty = 0.0
        self.large_vol = 0.0

    def add(self, printed_side, price: float, qty: float) -> bool:
        side = taker_side(printed_side)
        if side is None or qty <= 0:
            return False
        if side == "BUY":
            self.buy_vol += qty
            self.n_buy += 1
        else:
            self.sell_vol += qty
            self.n_sell += 1
        self.n_trades += 1
        self._sum_pv += price * qty
        if qty > self.max_qty:
            self.max_qty = qty
        if qty >= LARGE_TRADE_BTC:
            self.large_vol += qty
        return True

    @property
    def total_vol(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def vwap(self) -> float | None:
        total = self.total_vol
        return (self._sum_pv / total) if total > 0 else None


def tape_minute_row(minute_start: int, acc: TapeAccumulator | None,
                    coverage: float, sampled_at: int) -> dict:
    """Build one l2_trades_minutes row (pure function).

    - trades seen -> their aggregates, whatever the coverage
    - no trades but the stream was live (coverage > 0) -> ZEROS:
      a quiet minute is real data (vwap stays NULL: 0/0 is undefined)
    - no trades and the stream was down (coverage == 0) -> NULLs:
      "we were down" must not masquerade as a quiet minute
    """
    row: dict = {"ts": int(minute_start), "coverage": coverage,
                 "sampled_at": int(sampled_at)}
    if acc is not None and acc.n_trades > 0:
        row.update({
            "buy_vol": acc.buy_vol, "sell_vol": acc.sell_vol,
            "total_vol": acc.total_vol, "n_trades": acc.n_trades,
            "n_buy": acc.n_buy, "n_sell": acc.n_sell, "vwap": acc.vwap,
            "max_qty": acc.max_qty, "large_vol": acc.large_vol,
        })
    elif coverage > 0:
        row.update({
            "buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0,
            "n_trades": 0, "n_buy": 0, "n_sell": 0, "vwap": None,
            "max_qty": 0.0, "large_vol": 0.0,
        })
    else:
        for col in _TRADES_MINUTE_COLUMNS:
            if col not in row:
                row[col] = None
    return row


def _parse_trade_ts(text) -> float | None:
    """Coinbase trade `time` (ISO 8601, microsecond precision) -> epoch."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


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
        # trades tape (market_trades on the same socket)
        self.tape_acc: dict[int, TapeAccumulator] = {}
        self.tape_tracker = CoverageTracker()
        self.tape_subscribed = False
        self.tape_open_from: int | None = None
        self.next_tape_minute_to_finalize: int | None = None
        self.last_tape_minute_written: int | None = None
        self.last_tape_minute_trades = 0
        self.last_trade_ts: float | None = None
        self.trades_total = 0
        self.trades_late_dropped = 0
        self._seen_trade_ids: set = set()
        self._seen_trade_order: deque = deque()


def _handle_trade(state: CollectorState, trade: dict) -> None:
    """Bucket one market_trades print by its TRADE time (not receipt).

    Coinbase batches tape updates over ~250ms, so receipt time can
    spill across a minute boundary the trade does not belong to.
    Prints are deduped by trade_id (the subscribe-time snapshot can
    overlap the first updates). A print whose minute is already
    written is counted as late-dropped -- never merged into a
    finalized row, never backfilled.
    """
    if trade.get("product_id") not in (None, PRODUCT):
        return
    trade_ts = _parse_trade_ts(trade.get("time"))
    if trade_ts is None:
        return
    try:
        price = float(trade["price"])
        qty = float(trade["size"])
    except (KeyError, TypeError, ValueError):
        return
    tid = trade.get("trade_id")
    if tid is not None:
        if tid in state._seen_trade_ids:
            return
        state._seen_trade_ids.add(tid)
        state._seen_trade_order.append(tid)
        while len(state._seen_trade_order) > _SEEN_IDS_CAP:
            state._seen_trade_ids.discard(state._seen_trade_order.popleft())
    minute = int(trade_ts // 60) * 60
    if state.tape_open_from is not None and minute < state.tape_open_from:
        state.trades_late_dropped += 1
        return
    acc = state.tape_acc.get(minute)
    if acc is None:
        acc = state.tape_acc[minute] = TapeAccumulator()
    if acc.add(trade.get("side"), price, qty):
        state.trades_total += 1
        if state.last_trade_ts is None or trade_ts > state.last_trade_ts:
            state.last_trade_ts = trade_ts


def _handle_message(state: CollectorState, raw: str, recv_ts: float) -> None:
    msg = json.loads(raw)
    seq = msg.get("sequence_num")
    if isinstance(seq, int):
        if state.last_seq is not None and seq != state.last_seq + 1:
            raise SequenceGapError(
                f"sequence gap: expected {state.last_seq + 1}, got {seq}")
        state.last_seq = seq
    channel = msg.get("channel")
    if channel == "market_trades":
        # the tape is live from its first message of the session
        # (snapshot or update); there is no book to rebuild, so
        # liveness is simply "prints are arriving on this socket"
        if not state.tape_tracker.is_live:
            state.tape_tracker.set_live(True, recv_ts)
        for event in msg.get("events", []):
            for trade in event.get("trades") or []:
                _handle_trade(state, trade)
        return
    if channel != "l2_data":
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
    state.tape_subscribed = False
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
        await ws.send(json.dumps({
            "type": "subscribe", "product_ids": [PRODUCT],
            "channel": "market_trades"}))
        state.tape_subscribed = True
        log.info("subscribed: level2 + heartbeats + market_trades (%s)",
                 PRODUCT)
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
            state.tape_subscribed = False
            state.tracker.set_live(False, time.time())
            state.tape_tracker.set_live(False, time.time())
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


def _finalize_tape_minute(state: CollectorState, conn: sqlite3.Connection,
                          minute_start: int) -> None:
    coverage = state.tape_tracker.coverage_for(minute_start, time.time())
    acc = state.tape_acc.pop(minute_start, None)
    row = tape_minute_row(minute_start, acc, coverage, int(time.time()))
    placeholders = ", ".join("?" for _ in _TRADES_MINUTE_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO l2_trades_minutes "
        f"({', '.join(_TRADES_MINUTE_COLUMNS)}) VALUES ({placeholders})",
        [row[c] for c in _TRADES_MINUTE_COLUMNS])
    conn.commit()
    state.last_tape_minute_written = minute_start
    state.last_tape_minute_trades = acc.n_trades if acc else 0
    if minute_start % 3600 == 3540:
        hour = minute_start - 3540
        rollup = l2_features.rollup_trades_hour(conn, hour)
        if rollup is not None:
            log.info("tape hour %d rolled up: minutes_present=%d "
                     "coverage=%.1f usable=%d", hour,
                     rollup["minutes_present"],
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
        # Tape minutes finalize ONE MINUTE behind the book: Coinbase
        # batches tape updates over ~250ms, so a print executed at
        # :59.9 can arrive just after the boundary; the lag gives
        # every print its full minute to land in its own bucket.
        if state.next_tape_minute_to_finalize is None:
            state.next_tape_minute_to_finalize = upto - 60
            state.tape_open_from = upto - 60
        tape_upto = upto - 60
        tminute = state.next_tape_minute_to_finalize
        while tminute <= tape_upto:
            _finalize_tape_minute(state, conn, tminute)
            tminute += 60
        state.next_tape_minute_to_finalize = tape_upto + 60
        state.tape_open_from = tape_upto + 60


# ----------------------------------------------------------------- status

def _write_status(state: CollectorState) -> None:
    current_minute = int(time.time() // 60) * 60
    current_acc = state.tape_acc.get(current_minute)
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
        # trades tape
        "tape_subscribed": state.tape_subscribed,
        "tape_live": state.tape_tracker.is_live,
        "last_trade_ts": state.last_trade_ts,
        "trades_current_minute": current_acc.n_trades if current_acc else 0,
        "trades_last_minute": state.last_tape_minute_trades,
        "last_tape_minute_written": state.last_tape_minute_written,
        "trades_total": state.trades_total,
        "trades_late_dropped": state.trades_late_dropped,
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
    setup_conn.execute(TRADES_MINUTE_SCHEMA)
    l2_features.init_hours_table(setup_conn)
    l2_features.init_trades_hours_table(setup_conn)
    setup_conn.close()
    rolled = l2_features.backfill_rollups(DB_PATH)
    if rolled:
        log.info("startup backfill rolled up hours: %s", rolled)
    rolled = l2_features.backfill_trades_rollups(DB_PATH)
    if rolled:
        log.info("startup backfill rolled up tape hours: %s", rolled)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    state = CollectorState()
    # tape minutes before this process existed are closed from the
    # start: subscribe-time snapshot prints older than the first
    # finalizable minute count as late-dropped, never as data
    state.tape_open_from = int(time.time() // 60) * 60 - 60
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
