"""Market-data adapter: yfinance.

BTC-USD trades 24/7, so yfinance gives us a key-free feed that matches the
bot's universe. The adapter normalizes everything to a plain OHLCV frame so
other sources (Binance, Alpaca) can plug in later behind the same interface.

yfinance history limits (as of 2026): 5m/15m -> 60d, 1h -> 2y, 1d -> max.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

COLUMNS = ["open", "high", "low", "close", "volume"]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Newer yfinance returns MultiIndex columns like ('Close', 'BTC-USD')."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    # adj_close may exist when auto_adjust=False; drop it, keep raw OHLCV
    df = df.drop(columns=[c for c in ("adj_close",) if c in df.columns])
    return df


def fetch_klines(
    symbol: str = "BTC-USD",
    interval: str = "5m",
    period: str = "60d",
) -> pd.DataFrame:
    """Download klines and return a tz-naive OHLCV frame indexed by time.

    Raises RuntimeError if the feed returns nothing (network/API issue).
    """
    raw = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} {interval}/{period}")
    df = _flatten(raw)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"yfinance frame missing columns: {missing}")
    df = df[COLUMNS].astype(float).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def latest_price(symbol: str = "BTC-USD") -> float:
    """Last close, for the live trading dashboard / paper broker."""
    df = fetch_klines(symbol, interval="1m", period="1d")
    return float(df["close"].iloc[-1])
