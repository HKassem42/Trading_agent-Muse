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
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

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


def _fetch_coinbase(symbol: str, span: timedelta) -> tuple[pd.DataFrame, int]:
    """Fetch the full range from Coinbase. Returns (frame, expected_hours)."""
    end = datetime.now(timezone.utc)
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

    if not raw_rows:
        return pd.DataFrame(columns=COLUMNS), expected_hours

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
    return df, expected_hours


def fetch_klines(
    symbol: str = "BTC-USD",
    interval: str = "1h",
    period: str = "730d",
) -> pd.DataFrame:
    """Download klines, Coinbase first, yfinance on any problem.

    Same return contract as yfinance_adapter.fetch_klines. Only interval
    '1h' is served by Coinbase; anything else delegates to yfinance.
    """
    if interval != "1h":
        return _fallback(symbol, interval, period,
                         f"interval {interval!r} not supported on coinbase path")
    try:
        span = _period_timedelta(period)
    except ValueError as e:
        return _fallback(symbol, interval, period, str(e))

    try:
        df, expected = _fetch_coinbase(symbol, span)
    except Exception as e:
        return _fallback(symbol, interval, period, f"coinbase error: {e}")
    if df.empty:
        return _fallback(symbol, interval, period, "coinbase returned no rows")
    coverage = len(df) / max(expected, 1)
    if coverage < _MIN_COVERAGE:
        return _fallback(symbol, interval, period,
                         f"coinbase coverage {coverage:.1%} of {expected}h < 90%")
    print(f"market_data: SOURCE=coinbase symbol={symbol} period={period} "
          f"rows={len(df)} coverage={coverage:.1%}", flush=True)
    return df
