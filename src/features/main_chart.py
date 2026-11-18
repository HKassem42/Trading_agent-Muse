"""Main-chart indicators for the v2 Time-Series ML Training Data
Matrix (Kass green-lit 2026-10-22: "find and include all the main
chart indicators we can also include into our data table that are
not in there right now").

MAIN-CHART = overlays drawn ON the price chart (moving averages,
bands/channels, Ichimoku, SAR, SuperTrend) plus structural levels
(pivots, prior-day extremes, fractal swing support/resistance,
round-number grid) — as opposed to the sub-chart oscillators
(RSI/MACD/Stochastic...) the v1 matrix already carries.

Form conventions (uniform, documented once):
- Every "Distance (%)" column is the SIGNED percent distance of the
  close from the level: (close - level) / close * 100. Positive =
  price sits above the level, negative = below. A model can read
  side and distance from the one number.
- "Side (+1/-1)" columns are discrete regime flags (+1 = indicator
  in its below-price / uptrend mode, -1 = above-price / downtrend).
- All builders are CAUSAL: the value at row t uses only candles
  <= t. Daily-stamped values (pivots, prior-day extremes) are
  computed from the PRIOR completed UTC day and stamped on the
  following day (same discipline as the v1 ADR column).
- Ichimoku spans use the standard plotting convention handled
  causally: the cloud level VISIBLE at row t was fixed 26 bars
  earlier, so Span A/B at t are the (Tenkan+Kijun)/2 and
  (HH52+LL52)/2 values from t-26. Nothing from the future is used;
  the forward plot is exactly why the spans are known in advance.

Input frames: lowercase open/high/low/close/volume columns on a
sorted UTC DatetimeIndex. Values are RAW (natural units / percent),
mirroring the v1 matrix conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    """EMA with an n-bar warmup (NaN until n bars seen) — the
    az/ewm(adjust=False) recursion with min_periods discipline."""
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _dist(close: pd.Series, level: pd.Series) -> pd.Series:
    """Signed percent distance of close from a level."""
    return (close - level) / close.replace(0.0, np.nan) * 100.0


def _hh(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()


def _ll(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).min()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    import pandas_ta as ta
    return ta.atr(df["high"], df["low"], df["close"], length=n)


# ------------------------------------------------------------------
# moving-average distances (levels for 50/100/200 are NOT in v1;
# v1 carries only the SMA(20)/EMA(20) levels)
# ------------------------------------------------------------------

def _dist_sma(n: int):
    def b(df):
        return _dist(df["close"], _sma(df["close"], n))
    return b


def _dist_ema(n: int):
    def b(df):
        return _dist(df["close"], _ema(df["close"], n))
    return b


# ------------------------------------------------------------------
# bands / channels (v1 carries only midpoints or %B/BandWidth)
# ------------------------------------------------------------------

def _bollinger(df, which: str):
    mid = _sma(df["close"], 20)
    sd = df["close"].rolling(20).std()
    level = mid + 2.0 * sd if which == "upper" else mid - 2.0 * sd
    return _dist(df["close"], level)


def _donchian(df, n: int, which: str):
    level = _hh(df["high"], n) if which == "upper" else _ll(df["low"], n)
    return _dist(df["close"], level)


def _keltner(df, which: str):
    import pandas_ta as ta
    r = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2.0)
    for c in r.columns:
        if c.startswith("KCUe_" if which == "upper" else "KCLe_"):
            return _dist(df["close"], r[c])
    raise KeyError("keltner column not found")


def _atr_band_halfwidth(df):
    """(close + ATR14) - close == ATR14: the upper and lower ATR
    band distances have the same magnitude, ATR/close — one column
    carries both."""
    return _atr(df, 14) / df["close"].replace(0.0, np.nan) * 100.0


# ------------------------------------------------------------------
# Ichimoku (hand-rolled; spans shifted +26 per the plotting
# convention, which is what makes them causal)
# ------------------------------------------------------------------

def _tenkan(df):
    return (_hh(df["high"], 9) + _ll(df["low"], 9)) / 2.0


def _kijun(df):
    return (_hh(df["high"], 26) + _ll(df["low"], 26)) / 2.0


def _ichimoku_tenkan_dist(df):
    return _dist(df["close"], _tenkan(df))


def _ichimoku_span_a_dist(df):
    span_a_known = ((_tenkan(df) + _kijun(df)) / 2.0).shift(26)
    return _dist(df["close"], span_a_known)


def _ichimoku_span_b_dist(df):
    span_b_known = ((_hh(df["high"], 52) + _ll(df["low"], 52))
                    / 2.0).shift(26)
    return _dist(df["close"], span_b_known)


def _ichimoku_chikou_dist(df):
    """Chikou span is the close plotted 26 bars back; the distance
    it encodes at row t is close[t] vs the price 26 bars ago that
    the Chikou point is visually compared against."""
    return _dist(df["close"], df["close"].shift(26))


# ------------------------------------------------------------------
# Parabolic SAR / SuperTrend (levels already in v1 — sides and
# distances are the missing pieces)
# ------------------------------------------------------------------

def _psar_parts(df):
    import pandas_ta as ta
    r = ta.psar(df["high"], df["low"], df["close"])
    pl = next(c for c in r.columns if c.startswith("PSARl_"))
    ps = next(c for c in r.columns if c.startswith("PSARs_"))
    return r[pl], r[ps]


def _psar_dist(df):
    pl, ps = _psar_parts(df)
    return _dist(df["close"], pl.fillna(ps))


def _psar_side(df):
    pl, ps = _psar_parts(df)
    side = pd.Series(np.where(pl.notna(), 1.0,
                              np.where(ps.notna(), -1.0, np.nan)),
                     index=df.index)
    return side


def _supertrend_parts(df):
    import pandas_ta as ta
    r = ta.supertrend(df["high"], df["low"], df["close"],
                      length=10, multiplier=3.0)
    lvl = next(c for c in r.columns if c.startswith("SUPERT_")
               and not c.startswith("SUPERTd_")
               and not c.startswith("SUPERTl_")
               and not c.startswith("SUPERTs_"))
    d = next(c for c in r.columns if c.startswith("SUPERTd_"))
    return r[lvl], r[d]


def _supertrend_dist(df):
    lvl, _d = _supertrend_parts(df)
    return _dist(df["close"], lvl)


def _supertrend_side(df):
    _lvl, d = _supertrend_parts(df)
    return d.astype(float)


# ------------------------------------------------------------------
# daily-stamped structure: classic pivots + prior-day extremes
# (computed from the PRIOR completed UTC day, stamped on the next)
# ------------------------------------------------------------------

def _daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last",
                                  "volume": "sum"}).dropna()


def _stamp_daily(df: pd.DataFrame, daily_values: pd.Series) -> pd.Series:
    """daily_values is indexed by the day it APPLIES to (its value
    already derived from earlier days); stamp it on every row of
    that day via merge_asof backward on day starts."""
    left = pd.DataFrame({"ts": df.index})
    right = pd.DataFrame({"day": daily_values.index,
                          "v": daily_values.to_numpy(dtype=float)})
    merged = pd.merge_asof(left, right, left_on="ts", right_on="day",
                           direction="backward")
    return pd.Series(merged["v"].to_numpy(), index=df.index)


def _prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day levels from the prior completed day, indexed by the
    day they apply to: pivot P / R1 / S1, prior high, prior low."""
    d = _daily_frame(df)
    ph, pl, pc = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)
    p = (ph + pl + pc) / 3.0
    out = pd.DataFrame(index=d.index)
    out["pivot"] = p
    out["r1"] = 2.0 * p - pl
    out["s1"] = 2.0 * p - ph
    out["pdh"] = ph
    out["pdl"] = pl
    return out


def _prior_day_dist(key: str):
    def b(df):
        lv = _prior_day_levels(df)
        return _dist(df["close"], _stamp_daily(df, lv[key]))
    return b


# ------------------------------------------------------------------
# fractal swing support / resistance (Williams 2-bar fractals,
# confirmed 2 bars after the swing bar — a level only becomes
# usable at its confirmation time)
# ------------------------------------------------------------------

def fractal_levels(df: pd.DataFrame, lookback: int = 200
                   ) -> tuple[pd.Series, pd.Series]:
    """Return (resistance_level, support_level) per row: the nearest
    confirmed fractal swing high >= close (resistance) / swing low
    <= close (support) among swings from the last `lookback` bars.
    If no confirmed swing sits on the required side, the most recent
    confirmed swing of that type is used (distance then reads
    negative for resistance / positive for support — price has run
    through every recent level)."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)  # noqa: E741
    c = df["close"].to_numpy(float)
    n = len(df)
    # events: (swing_idx, level, confirm_idx = swing_idx + 2)
    highs: list[tuple[int, float, int]] = []
    lows: list[tuple[int, float, int]] = []
    for i in range(2, n - 2):
        w = h[i - 2:i + 3]
        if h[i] == w.max() and int((w == h[i]).sum()) == 1:
            highs.append((i, h[i], i + 2))
        w = l[i - 2:i + 3]
        if l[i] == w.min() and int((w == l[i]).sum()) == 1:
            lows.append((i, l[i], i + 2))
    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    # sliding candidate windows: confirmed events whose swing bar is
    # within the last `lookback` bars (events are time-ordered, so
    # front-pruning keeps each row's scan small)
    win_h: list[tuple[int, float]] = []
    win_l: list[tuple[int, float]] = []
    hi = li = 0
    for t in range(n):
        while hi < len(highs) and highs[hi][2] <= t:
            win_h.append((highs[hi][0], highs[hi][1]))
            hi += 1
        while li < len(lows) and lows[li][2] <= t:
            win_l.append((lows[li][0], lows[li][1]))
            li += 1
        while win_h and win_h[0][0] < t - lookback:
            win_h.pop(0)
        while win_l and win_l[0][0] < t - lookback:
            win_l.pop(0)
        if win_h:
            above = [lv for _sw, lv in win_h if lv >= c[t]]
            res[t] = min(above) if above else win_h[-1][1]
        if win_l:
            below = [lv for _sw, lv in win_l if lv <= c[t]]
            sup[t] = max(below) if below else win_l[-1][1]
    return (pd.Series(res, index=df.index),
            pd.Series(sup, index=df.index))


def _fractal_dist(which: str):
    def b(df):
        res, sup = fractal_levels(df)
        return _dist(df["close"], res if which == "resistance" else sup)
    return b


# ------------------------------------------------------------------
# round-number grid
# ------------------------------------------------------------------

def _round_number_dist(df, grid: float = 5000.0):
    c = df["close"]
    level = (c / grid).round() * grid
    return _dist(c, level)


# ------------------------------------------------------------------
# registry: (column name, builder, one-line definition)
# ------------------------------------------------------------------

INDICATORS = [
    ("Distance to SMA 50 (%)", _dist_sma(50),
     "(Close - SMA(50)) / Close x 100 — signed % distance from the "
     "50-bar simple moving average"),
    ("Distance to SMA 100 (%)", _dist_sma(100),
     "(Close - SMA(100)) / Close x 100 — signed % distance from the "
     "100-bar simple moving average"),
    ("Distance to SMA 200 (%)", _dist_sma(200),
     "(Close - SMA(200)) / Close x 100 — signed % distance from the "
     "200-bar simple moving average"),
    ("Distance to EMA 21 (%)", _dist_ema(21),
     "(Close - EMA(21)) / Close x 100 — signed % distance from the "
     "21-bar exponential moving average (n-bar warmup)"),
    ("Distance to EMA 55 (%)", _dist_ema(55),
     "(Close - EMA(55)) / Close x 100 — signed % distance from the "
     "55-bar exponential moving average (n-bar warmup)"),
    ("Distance to EMA 200 (%)", _dist_ema(200),
     "(Close - EMA(200)) / Close x 100 — signed % distance from the "
     "200-bar exponential moving average (n-bar warmup)"),
    ("Distance to Bollinger Upper Band (20, 2.0) (%)",
     lambda df: _bollinger(df, "upper"),
     "(Close - (SMA20 + 2 SD20)) / Close x 100 — negative inside "
     "the channel, 0 at the upper band"),
    ("Distance to Bollinger Lower Band (20, 2.0) (%)",
     lambda df: _bollinger(df, "lower"),
     "(Close - (SMA20 - 2 SD20)) / Close x 100 — positive inside "
     "the channel, 0 at the lower band"),
    ("Distance to Donchian Upper (20) (%)",
     lambda df: _donchian(df, 20, "upper"),
     "(Close - highest high(20)) / Close x 100 — 0 at the 20-bar "
     "high (v1 carries only the Donchian midpoint)"),
    ("Distance to Donchian Lower (20) (%)",
     lambda df: _donchian(df, 20, "lower"),
     "(Close - lowest low(20)) / Close x 100 — 0 at the 20-bar low"),
    ("Distance to Donchian Upper (55) (%)",
     lambda df: _donchian(df, 55, "upper"),
     "(Close - highest high(55)) / Close x 100 — the Turtle "
     "System-2 channel top"),
    ("Distance to Donchian Lower (55) (%)",
     lambda df: _donchian(df, 55, "lower"),
     "(Close - lowest low(55)) / Close x 100 — the Turtle "
     "System-2 channel bottom"),
    ("Distance to Keltner Upper (20, 2.0) (%)",
     lambda df: _keltner(df, "upper"),
     "(Close - Keltner upper(20, 2.0 ATR)) / Close x 100 (v1 "
     "carries only the Keltner middle line)"),
    ("Distance to Keltner Lower (20, 2.0) (%)",
     lambda df: _keltner(df, "lower"),
     "(Close - Keltner lower(20, 2.0 ATR)) / Close x 100"),
    ("ATR Band Half-Width (14) (%)", _atr_band_halfwidth,
     "ATR(14) / Close x 100 — the distance from close to the "
     "+/-1xATR band edges (both sides equal); ATR itself is not a "
     "v1 matrix column"),
    ("Ichimoku Tenkan-sen Distance (%)", _ichimoku_tenkan_dist,
     "(Close - Tenkan-sen(9)) / Close x 100; Tenkan = (HH9 + "
     "LL9) / 2. v1 carries only the Kijun baseline level"),
    ("Ichimoku Senkou Span A Distance (%)", _ichimoku_span_a_dist,
     "(Close - Span A as plotted at t) / Close x 100; Span A "
     "plotted at t = (Tenkan + Kijun)/2 computed at t-26 — fixed "
     "26 bars in advance, strictly causal"),
    ("Ichimoku Senkou Span B Distance (%)", _ichimoku_span_b_dist,
     "(Close - Span B as plotted at t) / Close x 100; Span B "
     "plotted at t = (HH52 + LL52)/2 computed at t-26"),
    ("Ichimoku Chikou Distance (%)", _ichimoku_chikou_dist,
     "(Close[t] - Close[t-26]) / Close[t] x 100 — the displacement "
     "the Chikou span encodes (Chikou plots close 26 bars back)"),
    ("Parabolic SAR Distance (%)", _psar_dist,
     "(Close - Parabolic SAR) / Close x 100, SAR = pandas_ta psar "
     "(0.02 / 0.2), the same series whose level is a v1 column"),
    ("Parabolic SAR Side (+1/-1)", _psar_side,
     "+1 while the SAR long series is active (SAR below price), "
     "-1 while the short series is active (SAR above price)"),
    ("SuperTrend Distance (%)", _supertrend_dist,
     "(Close - SuperTrend(10, 3.0)) / Close x 100, the same series "
     "whose level is a v1 column"),
    ("SuperTrend Side (+1/-1)", _supertrend_side,
     "SuperTrend(10, 3.0) direction from pandas_ta SUPERTd: +1 "
     "uptrend (line below price), -1 downtrend (line above price)"),
    ("Distance to Prior-Day Pivot Point (%)", _prior_day_dist("pivot"),
     "(Close - P) / Close x 100; P = (prior UTC day H + L + C) / 3, "
     "stamped on the following day (prior completed day only)"),
    ("Distance to Prior-Day Pivot R1 (%)", _prior_day_dist("r1"),
     "(Close - R1) / Close x 100; R1 = 2P - prior-day low"),
    ("Distance to Prior-Day Pivot S1 (%)", _prior_day_dist("s1"),
     "(Close - S1) / Close x 100; S1 = 2P - prior-day high"),
    ("Distance to Prior-Day High (%)", _prior_day_dist("pdh"),
     "(Close - prior completed UTC day's high) / Close x 100"),
    ("Distance to Prior-Day Low (%)", _prior_day_dist("pdl"),
     "(Close - prior completed UTC day's low) / Close x 100"),
    ("Distance to Fractal Resistance (%)", _fractal_dist("resistance"),
     "(Close - nearest confirmed fractal swing high >= Close, "
     "swings from the last 200 bars) / Close x 100; Williams "
     "2-bar fractals, usable only from their confirmation bar "
     "(swing bar + 2); falls back to the most recent swing high"),
    ("Distance to Fractal Support (%)", _fractal_dist("support"),
     "(Close - nearest confirmed fractal swing low <= Close, "
     "swings from the last 200 bars) / Close x 100; same "
     "confirmation discipline as fractal resistance"),
    ("Distance to $5,000 Round Number (%)", _round_number_dist,
     "(Close - nearest multiple of $5,000) / Close x 100 — signed "
     "distance to the psychological round-number grid"),
]


def build_all(df: pd.DataFrame) -> pd.DataFrame:
    """All main-chart feature series for an OHLCV frame."""
    out = {}
    for name, builder, _defn in INDICATORS:
        s = builder(df)
        s = pd.Series(np.asarray(s, dtype=float), index=df.index)
        out[name] = s.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=df.index)
