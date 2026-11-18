"""Shared on-disk candle cache for data.market_data.fetch_klines.

Adversarial pins for the cache layer (2026-11-12): the four book loops
and the compact feed builders all call fetch_klines every 15 minutes;
without a shared cache each call re-downloads the full window (a 730d
fetch is 59 sequential Coinbase chunk requests), which is what made
loop runs time out. The cache must:

(a) serve a repeat call with ZERO network fetches while it covers the
    last closed hour (the module keeps no in-memory frame state, so
    back-to-back calls exercise the same disk path two OS processes
    would -- that is the two-process simulation),
(b) return frames candle-for-candle identical to the direct fetch,
(c) on staleness fetch ONLY the missing tail, replace the provisional
    (forming) candle with its final values, and never serve a forming
    candle as if it were closed,
(d) fall back to a FULL fetch on a cold cache or a >24h gap,
(e) keep the Coinbase -> yfinance fallback semantics: coverage < 90%
    still delegates, and the fallback frame is what gets cached.

Time and both network seams (_coinbase_chunk, yfinance fetch) are
faked; bucket values are a deterministic function of the bucket hour,
with the still-forming bucket carrying visibly provisional values
(close + 0.25, volume 3.0) so a provisional candle can never be
mistaken for a final one.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data.market_data as md

EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)
T0 = datetime(2026, 11, 12, 15, 23, 41, tzinfo=timezone.utc)


def _floor_h(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def _ts(dt):
    """Aware UTC datetime -> the tz-naive Timestamp the contract uses."""
    return pd.Timestamp(dt.replace(tzinfo=None))


def _n(bucket):
    return int((bucket - EPOCH).total_seconds() // 3600)


class _FakeDatetime(datetime):
    current = T0

    @classmethod
    def now(cls, tz=None):
        v = cls.current
        return v if tz is None else v.astimezone(tz)


def _bucket_row(bucket, provisional=False):
    """Coinbase wire layout: [ts_unix, low, high, open, close, volume]."""
    n = _n(bucket)
    close = 100.5 + n + (0.25 if provisional else 0.0)
    volume = 3.0 if provisional else 10.0
    return [int(bucket.timestamp()), 99.0 + n, 101.0 + n, 100.0 + n,
            close, volume]


class Rig:
    def __init__(self, monkeypatch, tmp_path, sparse=False):
        self.chunk_calls = []      # list of (start, end) aware datetimes
        self.yf_calls = 0
        self.yf_frames = []
        self.sparse = sparse
        monkeypatch.setattr(md, "datetime", _FakeDatetime)
        _FakeDatetime.current = T0
        # raising=False: pre-cache code has no _CACHE_DIR; the patch is
        # a no-op there and the behavioral asserts below fail instead.
        monkeypatch.setattr(md, "_CACHE_DIR", tmp_path / "cache",
                            raising=False)
        monkeypatch.setattr(md, "_coinbase_chunk", self._chunk)
        monkeypatch.setattr(md.yfinance_adapter, "fetch_klines", self._yf)

    @property
    def now(self):
        return _FakeDatetime.current

    def advance(self, hours):
        _FakeDatetime.current = _FakeDatetime.current + hours * H

    def reset_counts(self):
        self.chunk_calls = []
        self.yf_calls = 0

    def _chunk(self, symbol, start, end):
        self.chunk_calls.append((start, end))
        rows = []
        h = _floor_h(start)
        forming = _floor_h(self.now)
        while h < end:
            if not (self.sparse and _n(h) % 2 == 1):
                rows.append(_bucket_row(h, provisional=(h == forming)))
            h += H
        return rows

    def _yf(self, symbol, interval="1h", period="5d"):
        self.yf_calls += 1
        days = int(str(period).rstrip("d"))
        # The real adapter returns tz-naive indexes (tz_localize(None));
        # mirror that or the fake deviates from the contract it fakes.
        start = _floor_h(self.now - timedelta(days=days)).replace(tzinfo=None)
        idx = pd.date_range(start, _floor_h(self.now).replace(tzinfo=None),
                            freq="h")
        n = [_n(t.to_pydatetime().replace(tzinfo=timezone.utc))
             for t in idx]
        df = pd.DataFrame({
            "open": [2000.0 + i for i in n],
            "high": [2001.0 + i for i in n],
            "low": [1999.0 + i for i in n],
            "close": [2000.5 + i for i in n],
            "volume": [20.0] * len(n),
        }, index=idx)
        df.index.name = "Datetime"
        self.yf_frames.append(df)
        return df


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    return Rig(monkeypatch, tmp_path)


def _fetch(period="5d"):
    return md.fetch_klines("BTC-USD", interval="1h", period=period)


def test_a_repeat_call_hits_no_network(rig):
    first = _fetch()
    assert rig.chunk_calls, "cold call must fetch"
    assert first.index.name == "Datetime"
    assert list(first.columns) == md.COLUMNS
    rig.reset_counts()
    second = _fetch()
    assert rig.chunk_calls == [], "warm call re-hit coinbase chunks"
    assert rig.yf_calls == 0, "warm call re-hit the yfinance fallback"
    pd.testing.assert_frame_equal(first, second)


def test_b_cached_output_equals_direct(rig, monkeypatch, tmp_path):
    direct = _fetch()  # cold cache == the direct network path
    # A completely separate, cold cache directory must reproduce the
    # exact same frame through the network path again (deterministic
    # fakes), and the warm serve must equal it candle-for-candle.
    monkeypatch.setattr(md, "_CACHE_DIR", tmp_path / "cache2",
                        raising=False)
    rig.reset_counts()
    cold_again = _fetch()
    assert rig.chunk_calls, "second cold directory should fetch"
    pd.testing.assert_frame_equal(direct, cold_again)
    rig.reset_counts()
    warm = _fetch()
    assert rig.chunk_calls == []
    pd.testing.assert_frame_equal(direct, warm)


def test_c_stale_cache_fetches_only_tail_and_finalizes(rig):
    first = _fetch()
    forming0 = _floor_h(T0)
    # The forming candle went out with provisional values.
    assert first.loc[_ts(forming0), "close"] == \
        100.5 + _n(forming0) + 0.25
    rig.advance(3)
    rig.reset_counts()
    second = _fetch()
    assert rig.chunk_calls, "stale cache must fetch the tail"
    # ONLY the tail: no chunk may reach back before the provisional
    # bucket -- everything older was already final in the cache.
    for start, _end in rig.chunk_calls:
        assert start >= forming0 - H, \
            f"tail fetch reached back to {start}, expected >= {forming0}"
    # The previously provisional bucket now carries FINAL values.
    row = second.loc[_ts(forming0)]
    assert row["close"] == 100.5 + _n(forming0)
    assert row["volume"] == 10.0
    # Merged frame: unique, gapless over the closed range, last row is
    # the new forming bucket (provisional), nothing beyond it.
    assert second.index.is_unique
    forming1 = _floor_h(rig.now)
    assert second.index[-1] == _ts(forming1)
    assert second.loc[_ts(forming1), "close"] == \
        100.5 + _n(forming1) + 0.25
    closed = second.iloc[:-1]
    full = pd.date_range(closed.index[0], closed.index[-1], freq="h")
    assert len(full) == len(closed), "gap in the merged closed range"


def test_d_cold_and_big_gap_do_full_fetch(rig):
    span_start = T0 - timedelta(days=5)
    _fetch()
    assert rig.chunk_calls
    assert min(s for s, _ in rig.chunk_calls) <= span_start + H, \
        "cold cache must request the full window"
    # A >24h hole cannot be trusted to a tail merge: full fetch again.
    rig.advance(30)
    rig.reset_counts()
    _fetch()
    assert rig.chunk_calls
    new_start = rig.now - timedelta(days=5)
    assert min(s for s, _ in rig.chunk_calls) <= new_start + H, \
        ">24h gap must trigger a full fetch, not a tail fetch"


def test_e_fallback_result_is_what_gets_cached(monkeypatch, tmp_path,
                                               capsys):
    rig = Rig(monkeypatch, tmp_path, sparse=True)
    first = _fetch()
    assert rig.yf_calls == 1, "sparse coinbase (<90%) must fall back"
    pd.testing.assert_frame_equal(first, rig.yf_frames[0])
    assert "SOURCE=yfinance-fallback" in capsys.readouterr().out
    rig.reset_counts()
    second = _fetch()
    assert rig.chunk_calls == [], "cached fallback must not re-probe"
    assert rig.yf_calls == 0, "cached fallback must not re-fetch"
    pd.testing.assert_frame_equal(first, second)
    assert "SOURCE=" in capsys.readouterr().out


def test_source_line_printed_on_cache_serve(rig, capsys):
    _fetch()
    capsys.readouterr()
    rig.reset_counts()
    _fetch()
    out = capsys.readouterr().out
    assert "market_data: SOURCE=" in out
