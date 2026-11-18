"""Market-data adapter: Coinbase Exchange (primary), yfinance (fallback).

Drop-in replacement for `data.yfinance_adapter.fetch_klines` with the exact
same contract: fetch_klines(symbol, interval, period) -> pandas DataFrame
with lowercase float columns [open, high, low, close, volume], indexed by
hour-start timestamps (index name "Datetime"), sorted ascending, deduped.
Like the yfinance adapter, timestamps are UTC wall-clock with tz stripped
(tz_localize(None)) -- matching that adapter's actual output shape so every
downstream consumer (feature builder, live loop, snapshot string compares)
sees identical types from either source.

Why Coinbase primary: yfinance candles arrive revised after the fact and
its volume column is unreliable (zero-filled stretches, implausible
magnitudes). The Coinbase Exchange public API needs no auth/key, returns
exchange-native candles that are never revised, and reports volume in BTC
(base units). If Coinbase is unreachable, errors, or returns a frame
covering <90% of the expected hours, we delegate to the yfinance adapter
unchanged. Every call prints which source served the data:
SOURCE=coinbase or SOURCE=yfinance-fallback.

Coinbase Exchange API notes:
- GET /products/{symbol}/candles?granularity=3600&start=...&end=...
- Response rows are [time_unix, low, high, open, close, volume]
  (note: low BEFORE high, open AFTER high -- not OHLC order).
- Max 300 candles per request, so long ranges are chunked into <=300h
  windows and concatenated.
- BTC trades 24/7, so a healthy response has no gaps.

Shared on-disk cache (2026-11-12): every book loop and feed builder
calls fetch_klines every 15 minutes, and a 730d fetch is 59 sequential
chunk requests -- the redundancy was the dominant cause of loop
timeouts. fetch_klines now keeps ONE cache file per (symbol, interval)
under data/cache/, written atomically (temp file + os.replace) so the
four loop processes share it safely. A call is served from cache when
the cached FINAL candles (every bucket that was already closed at
fetch time) cover the requested window at >=90% density; the
still-forming bucket rides along exactly as a direct fetch would
return it. When the cache is stale by <=24h, only the missing tail is
fetched -- starting at the provisional bucket, so a candle that was
forming when cached is always overwritten with its final values and
is never served as closed. A cold cache, a >24h gap, insufficient
history for the requested period, or ANY cache-layer error falls back
to the exact original full-fetch path (including its yfinance
fallback, whose result is then what gets cached). Cache problems can
never break a fetch. Disable with TAM_KLINES_CACHE=0.
"""

from __future__ import annotations

import os
import pickle
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from data import yfinance_adapter

COLUMNS = ["open", "high", "low", "close", "volume"]

_COINBASE_BASE = "https://api.exchange.coinbase.com"
_GRANULARITY = 3600          # seconds; only 1h is supported on this path
_MAX_CANDLES = 300           # Coinbase per-request candle limit
_MIN_COVERAGE = 0.90         # below this fraction of expected hours -> fallback
_TIMEOUT_S = 20
_HEADERS = {"User-Agent": "TradingAgent-Muse/1.0 market-data adapter",
            "Accept": "application/json"}

_DATETIME_CLS = datetime     # real class, bound at import: cache-record
                             # validation must not depend on the module's
                             # (test-patchable) clock seam `datetime`
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
_CACHE_KEEP_DAYS = 732       # 730d window + buffer; older rows trimmed on write
_TAIL_MAX_GAP_H = 24         # staleness beyond this -> full fetch, not a tail


def _fallback(symbol: str, interval: str, period: str, reason: str) -> pd.DataFrame:
    print(f"market_data: SOURCE=yfinance-fallback ({reason})", flush=True)
    return yfinance_adapter.fetch_klines(symbol, interval=interval, period=period)


def _period_timedelta(period: str) -> timedelta:
    """'5d' / '60d' / '730d' -> timedelta. Raises ValueError otherwise."""
    m = re.fullmatch(r"\s*(\d+)\s*d\s*", str(period))
    if not m:
        raise ValueError(f"unsupported period {period!r} (only '<n>d')")
    return timedelta(days=int(m.group(1)))


def _coinbase_chunk(symbol: str, start: datetime, end: datetime) -> list:
    """One <=300-candle Coinbase request, with retries on 429/5xx."""
    url = f"{_COINBASE_BASE}/products/{symbol}/candles"
    params = {
        "granularity": _GRANULARITY,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS,
                                timeout=_TIMEOUT_S)
            if resp.status_code == 200:
                rows = resp.json()
                if not isinstance(rows, list):
                    raise RuntimeError(f"unexpected payload type {type(rows)}")
                return rows
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"coinbase chunk {params['start']} failed: {last_err}")


def _rows_to_frame(raw_rows: list) -> pd.DataFrame:
    """Normalize Coinbase wire rows to the contract frame."""
    if not raw_rows:
        return pd.DataFrame(columns=COLUMNS)
    # Coinbase row layout: [time_unix, low, high, open, close, volume]
    raw = pd.DataFrame(raw_rows,
                       columns=["ts", "low", "high", "open", "close", "volume"])
    raw = raw.drop_duplicates(subset="ts", keep="last")
    df = pd.DataFrame({
        "ts": pd.to_datetime(raw["ts"], unit="s", utc=True).dt.tz_localize(None),
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    }).set_index("ts")
    df.index.name = "Datetime"  # match the yfinance adapter's index name
    df = df[COLUMNS].astype(float).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_coinbase(symbol: str, span: timedelta) -> tuple[pd.DataFrame, int]:
    """Fetch the full range from Coinbase. Returns (frame, expected_hours)."""
    end = _now()
    start = end - span
    expected_hours = int(span.total_seconds() // _GRANULARITY)
    window = timedelta(seconds=_GRANULARITY * _MAX_CANDLES)

    raw_rows: list = []
    t = start
    while t < end:
        t_end = min(t + window, end)
        raw_rows.extend(_coinbase_chunk(symbol, t, t_end))
        t = t_end
        if t < end:
            time.sleep(0.12)  # stay well under public rate limits

    return _rows_to_frame(raw_rows), expected_hours


def _fetch_coinbase_range(symbol: str, start: datetime,
                          end: datetime) -> pd.DataFrame:
    """Coinbase frame for an explicit [start, end] range (cache tails).

    Same chunking and normalization as _fetch_coinbase, but the caller
    picks hour-aligned bounds so the tail starts exactly at the first
    bucket that needs (re)fetching.
    """
    window = timedelta(seconds=_GRANULARITY * _MAX_CANDLES)
    raw_rows: list = []
    t = start
    while t < end:
        t_end = min(t + window, end)
        raw_rows.extend(_coinbase_chunk(symbol, t, t_end))
        t = t_end
        if t < end:
            time.sleep(0.12)  # stay well under public rate limits
    return _rows_to_frame(raw_rows)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_enabled() -> bool:
    return os.environ.get("TAM_KLINES_CACHE", "1") != "0"


def _cache_file(symbol: str, interval: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in f"{symbol}_{interval}")
    return _CACHE_DIR / f"klines_{safe}.pkl"


def _naive_floor_h(dt: datetime) -> pd.Timestamp:
    """Any datetime -> tz-naive (UTC wall clock) Timestamp floored to the hour."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return pd.Timestamp(dt).floor("h")


def _read_cache(symbol: str, interval: str) -> dict | None:
    """Load the cache record, or None on any problem (missing/corrupt)."""
    try:
        path = _cache_file(symbol, interval)
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            rec = pickle.load(fh)
        if not isinstance(rec, dict):
            return None
        frame, fetched_at = rec.get("frame"), rec.get("fetched_at")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        if not isinstance(fetched_at, _DATETIME_CLS):
            return None
        return rec
    except Exception:
        return None


def _write_cache(symbol: str, interval: str, frame: pd.DataFrame,
                 source: str, fetched_at: datetime) -> None:
    """Atomically replace the cache file. Never raises: a failed cache
    write must not break the fetch that produced the frame."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cutoff = _naive_floor_h(fetched_at) - pd.Timedelta(days=_CACHE_KEEP_DAYS)
        frame = frame[frame.index >= cutoff]
        rec = {"frame": frame, "source": source, "fetched_at": fetched_at}
        tmp = _CACHE_DIR / f".tmp-{os.getpid()}-{time.time_ns()}.pkl"
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh)
        os.replace(tmp, _cache_file(symbol, interval))
    except Exception:
        pass


def _covers(final: pd.DataFrame, win_start: pd.Timestamp,
            last_closed: pd.Timestamp) -> bool:
    """True when final (closed) candles span [win_start, last_closed]
    at >= _MIN_COVERAGE density -- the cache form of the <90% rule.
    The left edge tolerates one bucket: a direct Coinbase fetch starts
    at the first hour boundary at/after (now - span), so a stored frame
    may legitimately begin up to one bucket after the floored
    win_start; that bucket simply counts as absent in the density."""
    if final.empty:
        return False
    if final.index[0] > win_start + pd.Timedelta(hours=1) \
            or final.index[-1] < last_closed:
        return False
    total = int((last_closed - win_start) / pd.Timedelta(hours=1)) + 1
    present = int(((final.index >= win_start)
                   & (final.index <= last_closed)).sum())
    return present / max(total, 1) >= _MIN_COVERAGE


def _print_cache_line(source: str, symbol: str, period: str,
                      out: pd.DataFrame, final: pd.DataFrame,
                      win_start: pd.Timestamp,
                      last_closed: pd.Timestamp, how: str) -> None:
    total = int((last_closed - win_start) / pd.Timedelta(hours=1)) + 1
    present = int(((final.index >= win_start)
                   & (final.index <= last_closed)).sum())
    print(f"market_data: SOURCE={source} symbol={symbol} period={period} "
          f"rows={len(out)} coverage={present / max(total, 1):.1%} ({how})",
          flush=True)


def _serve_cached(symbol: str, interval: str, period: str,
                  span: timedelta) -> pd.DataFrame | None:
    """Serve from the shared cache, fetching only a stale tail if needed.

    Returns None whenever the cache cannot answer exactly -- the caller
    then runs the original full-fetch path. May raise on unexpected
    cache corruption; the caller treats any exception as a miss.
    """
    rec = _read_cache(symbol, interval)
    if rec is None:
        return None
    cdf, source = rec["frame"], rec.get("source")
    now = _now()
    now_ts = pd.Timestamp(now.astimezone(timezone.utc).replace(tzinfo=None))
    forming = now_ts.floor("h")
    last_closed = forming - pd.Timedelta(hours=1)
    win_start = (now_ts - pd.Timedelta(span)).floor("h")
    # Buckets that started before the writing fetch's own hour were
    # already closed when cached: their values are final. The bucket
    # AT that hour was still forming -- provisional, never trusted as
    # closed, and always refetched by the tail below.
    prov = _naive_floor_h(rec["fetched_at"])
    final = cdf[cdf.index < prov]
    if final.empty or final.index[0] > win_start + pd.Timedelta(hours=1):
        return None  # insufficient history; a tail cannot fix that

    if prov == forming and cdf.index[-1] == forming \
            and _covers(final, win_start, last_closed):
        out = cdf[cdf.index >= win_start]
        _print_cache_line(source, symbol, period, out, final,
                          win_start, last_closed, "cache")
        return out

    # Stale: fetch only the missing tail (<= _TAIL_MAX_GAP_H of closed
    # candles), starting at the provisional bucket so it is replaced
    # with final values.
    newest_final = final.index[-1]
    if last_closed - newest_final > pd.Timedelta(hours=_TAIL_MAX_GAP_H):
        return None
    tail_start = newest_final + pd.Timedelta(hours=1)
    if prov <= last_closed and prov < tail_start:
        tail_start = prov
    try:
        tail = _fetch_coinbase_range(
            symbol,
            tail_start.to_pydatetime().replace(tzinfo=timezone.utc),
            now)
    except Exception:
        return None  # the full path retries with its own fallback logic
    if tail.empty or tail.index[-1] != forming:
        return None
    if not _covers(tail[tail.index < forming], tail_start, last_closed):
        return None
    merged = pd.concat([final, tail])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _write_cache(symbol, interval, merged, "coinbase", now)
    merged_final = merged[merged.index < forming]
    if not _covers(merged_final, win_start, last_closed):
        return None  # history still short for this period -> full fetch
    out = merged[merged.index >= win_start]
    _print_cache_line("coinbase", symbol, period, out, merged_final,
                      win_start, last_closed, "cache+tail")
    return out


def _full_fetch(symbol: str, interval: str, period: str,
                span: timedelta) -> tuple[pd.DataFrame, str]:
    """The original fetch_klines network path, unchanged. Returns
    (frame, source) where source is 'coinbase' or 'yfinance-fallback'."""
    try:
        df, expected = _fetch_coinbase(symbol, span)
    except Exception as e:
        return _fallback(symbol, interval, period,
                         f"coinbase error: {e}"), "yfinance-fallback"
    if df.empty:
        return _fallback(symbol, interval, period,
                         "coinbase returned no rows"), "yfinance-fallback"
    coverage = len(df) / max(expected, 1)
    if coverage < _MIN_COVERAGE:
        return _fallback(symbol, interval, period,
                         f"coinbase coverage {coverage:.1%} of {expected}h "
                         f"< 90%"), "yfinance-fallback"
    print(f"market_data: SOURCE=coinbase symbol={symbol} period={period} "
          f"rows={len(df)} coverage={coverage:.1%}", flush=True)
    return df, "coinbase"


def fetch_klines(
    symbol: str = "BTC-USD",
    interval: str = "1h",
    period: str = "730d",
) -> pd.DataFrame:
    """Download klines, Coinbase first, yfinance on any problem.

    Same return contract as yfinance_adapter.fetch_klines. Only interval
    '1h' is served by Coinbase; anything else delegates to yfinance.
    The 1h path is served by the shared on-disk cache when possible
    (see the module docstring); the return contract is identical.
    """
    if interval != "1h":
        return _fallback(symbol, interval, period,
                         f"interval {interval!r} not supported on coinbase path")
    try:
        span = _period_timedelta(period)
    except ValueError as e:
        return _fallback(symbol, interval, period, str(e))

    if _cache_enabled():
        try:
            served = _serve_cached(symbol, interval, period, span)
        except Exception:
            served = None  # cache trouble must never break a fetch
        if served is not None:
            return served
    df, source = _full_fetch(symbol, interval, period, span)
    if _cache_enabled():
        _write_cache(symbol, interval, df, source, _now())
    return df
