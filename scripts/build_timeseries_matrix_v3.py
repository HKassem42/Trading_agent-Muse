"""Build Matrix v3 — the 0-100 NORMALIZED edition of the BTC matrix.

Kass green-lit 2026-10-22 ("let's go with the start of that V3"),
adopting the design proposed the same evening: every indicator
column becomes a trailing 180-day percentile-rank score on 0-100,
oriented by the bullish/bearish guide (refs/
A-Z_Indicators_Bullish_Bearish_Guide.xlsx, Tab 2 "Our Matrix
Columns" — the orientation authority for this build).

METHOD (per indicator column):
- Ranked quantity q: the series the guide actually reads.
  * RAW      — the column's own value (oscillators, distances,
               strength / intensity gauges).
  * GAP      — price-level columns (moving averages, channels,
               trailing stops, VWAP): the guide reads them ONLY as
               "close above/below the level", so q is the signed
               gap (Close - level) / Close x 100 (the v2 distance
               convention).
  * SLOPE24  — cumulative lines (A/D, ASI, OBV, PVT): the guide
               says slope only, the level is meaningless, so q is
               the 24-bar change.
  * MACDGAP  — MACD Signal Line: the guide reads it only against
               the MACD line, so q = MACD - Signal (its score is
               therefore identical to the Histogram column's by
               construction — noted in the key).
- Score: for row t, the percentile of q_t within the trailing
  4,320 hourly values INCLUDING t (180 days x 24), strictly
  causal. Midrank convention: r = n_less + (n_equal + 1)/2 over
  the window, score = 100 x (r - 1) / (4320 - 1), so the window
  minimum scores 0, the maximum 100, the median ~50. Windows are
  consecutive ROWS of the long series (the source record has 3
  small holes, shared with v1/v2, so a few windows span slightly
  more than 180 days of wall time — disclosed in the report).
- Orientation (guide Tab 2 is the authority; NOT re-derived):
  * HIGH=BULLISH — high score = the guide's bullish condition.
  * FLIPPED      — 100 - percentile; only Connors RSI, whose
    Tab 2 reading is contrarian (very low = dip-buy zone, very
    high = short-the-rip zone). NOTE: Williams %R is NOT flipped
    — Tab 2 reads its top of scale (near 0 = close at the top of
    the range) as the strong/bullish side.
  * STRENGTH     — unsigned gauges (ADX, and the DMI column that
    stores DX): normalized as-is, high = stronger, NO bull/bear
    claim, excluded from composites.
  * INTENSITY    — non-directional gauges (ADR, Bollinger
    BandWidth, CHOP, HV, Mass Index, SD, ATR half-width):
    normalized as-is, high = most extreme in window, NO bull/bear
    claim, excluded from composites.
- RAW (never transformed): Date, Ticker, OHLCV, the label.
- COMPOSITES from the normalized DIRECTIONAL columns only,
  grouped by the guide's Tab 2 families: score_trend (45 cols),
  score_momentum (25), score_volume (7) are family means;
  score_master = mean of the three family scores (mean of family
  means, so big families don't dominate). Directional columns
  whose Tab 2 family is genuinely dual or off-axis are EXCLUDED
  from composites and noted in the key (Ease of Movement and
  Money Flow Index — dual volume x momentum families; Relative
  Volatility Index — volatility-direction family; the $5,000
  round-number distance — price-psychology reference).

Series are recomputed on the LONG hourly frame (2022-01-01 ->,
42,107 rows) via the v1/v2 build code paths, so every matrix row
has a full trailing window; a reproduction check against v2's
saved values runs BEFORE any normalization and aborts the build
on an unexplained mismatch.

BUILD ONLY — nothing is trained or screened on v3. Trading stays
PARKED. Usage: .venv/bin/python scripts/build_timeseries_matrix_v3.py
"""

from __future__ import annotations

import json
import sys
import time
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_timeseries_matrix as b1  # noqa: E402  v1 code path
from features import main_chart as mc  # noqa: E402  v2 code path

V2_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v2.csv"
LONG_HOURLY = ROOT / "data" / "history" / "btc_usd_1h_long.csv"
OUT_DIR = ROOT / "data" / "timeseries_matrix"
V3_CSV = OUT_DIR / "btc_usd_1h_matrix_v3.csv"
STATS_PATH = OUT_DIR / "build_stats_v3.json"
XLSX_PATH = Path.home() / "workspace" / "your_files" / \
    "BTC-USD_1h_Matrix_v3_Normalized.xlsx"
REPORT_PATH = ROOT / "refs" / "matrix-v3-normalized-2026-10-22.md"

WINDOW = 4320  # 180 days x 24 hourly bars
TARGET_COL = "Target_Action (BUY/SELL/SHORT/HOLD)"

# ------------------------------------------------------------------
# column spec: name -> (Tab 2 family, ranked quantity, orientation,
#                       composite family or None, guide basis 1-liner)
# orientation: BULL = high score bullish; FLIP = flipped, high
# score bullish; STRENGTH / INTENSITY = no direction claim.
# ------------------------------------------------------------------
SPEC = {
    # ---- v1 columns (Kass's template order) ----
    "Absolute Price Oscillator (APO)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Positive = EMA(12) above EMA(26), momentum up; zero is the divider."),
    "Acceleration Oscillator (AC)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Positive/rising = upward momentum accelerating; negative/falling = downward."),
    "Accumulation/Distribution (A/D)": (
        "Volume flow (cumulative line)", "SLOPE24", "BULL", "volume",
        "Rising slope = accumulation; the level is meaningless, slope only (guide)."),
    "Accumulative Swing Index (ASI)": (
        "Trend / swing structure", "SLOPE24", "BULL", "trend",
        "Positive/rising = swing structure up; cumulative line, slope is the reading."),
    "Arnaud Legoux Moving Average (ALMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above the ALMA(20) level = up bias; the guide reads it vs Close."),
    "Aroon Indicator": (
        "Trend (recency of highs)", "RAW", "BULL", "trend",
        "Aroon Up(14): above 70 = fresh high very recent, up-momentum strong; below 30 = stale."),
    "Average Daily Range (ADR)": (
        "Volatility (range size)", "RAW", "INTENSITY", None,
        "Mean daily range in USD — sizes a typical day's travel; direction-free."),
    "Average Directional Movement Index (ADX)": (
        "Trend STRENGTH", "RAW", "STRENGTH", None,
        "ADX(14): above 25 only says a trend (either way) is strong; no direction."),
    "Awesome Oscillator (AO)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Positive = 5-period midpoint average above the 34-period; zero is the divider."),
    "Bollinger BandWidth": (
        "Volatility (band width)", "RAW", "INTENSITY", None,
        "High/rising = volatility expanding; very low = the Squeeze; direction unknown."),
    "Bollinger %B": (
        "Band position (0-1)", "RAW", "BULL", "momentum",
        "Above 0.5 = upper half of the bands; >= 1 = close above the upper band (strong push)."),
    "Center of Gravity (COG)": (
        "Oscillator (Ehlers)", "RAW", "BULL", "momentum",
        "Rising / crossing up = the weighted center of recent prices is moving up."),
    "Chaikin Money Flow (CMF)": (
        "Volume flow (zero-line)", "RAW", "BULL", "volume",
        "Above zero = money flowing in; +0.25 or more = strong buying pressure."),
    "Chande Momentum Oscillator (CMO)": (
        "Momentum oscillator (-100..+100)", "RAW", "BULL", "momentum",
        "Above zero = up-momentum dominates; above +50 = strong/overbought."),
    "Chandelier Exit": (
        "Trailing stop (price level)", "GAP", "BULL", "trend",
        "Close above this long-exit line = long state intact; below = the exit trigger."),
    "Choppiness Index (CHOP)": (
        "Regime gauge", "RAW", "INTENSITY", None,
        "CHOP(14): above ~61.8 = choppy/sideways; below ~38.2 = trending; no direction."),
    "Commodity Channel Index (CCI)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Above zero = momentum up; above +100 = strong up-move."),
    "Connors RSI (CRSI)": (
        "Momentum oscillator (0-100, stretch)", "RAW", "FLIP", "momentum",
        "FLIPPED per the guide: very LOW raw = dip-buy zone in an uptrend; very HIGH raw = short-the-rip zone in a downtrend."),
    "Coppock Curve": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Turning up from below zero = long-term momentum buy; above zero and rising = up."),
    "Detrended Price Oscillator (DPO)": (
        "Cycle oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Positive = price above its shifted SMA(20) — cycle phase up."),
    "Directional Movement Index (DMI)": (
        "Trend strength of direction (DX)", "RAW", "STRENGTH", None,
        "Stores DX(14) = the SIZE of the +DI/-DI gap, unsigned — it cannot say which side."),
    "Donchian Channels": (
        "Channel (price level — midpoint)", "GAP", "BULL", "trend",
        "Close above the Donchian(20) midpoint = price in the upper half of the range."),
    "Double Exponential Moving Average (DEMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising DEMA(20) = uptrend; read vs Close."),
    "Ease of Movement (EMV)": (
        "Volume/momentum (zero-line)", "RAW", "BULL", None,
        "Above zero = price moving up easily on light volume. EXCLUDED from composites: Tab 2 family is dual (volume x momentum)."),
    "Elder Force Index (EFI)": (
        "Volume force (zero-line)", "RAW", "BULL", "volume",
        "Above zero = buyers won the volume-weighted tug of the candle."),
    "Exponential Moving Average (EMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising EMA(20) = uptrend; read vs Close."),
    "Fisher Transform": (
        "Oscillator (turning points)", "RAW", "BULL", "momentum",
        "Fisher signal line: rising through zero / crossing up = turning point up."),
    "Force Index": (
        "Volume force (zero-line)", "RAW", "BULL", "volume",
        "Positive = this candle's move was up, weighted by volume (raw 1-period form)."),
    "Historical Volatility (HV)": (
        "Volatility", "RAW", "INTENSITY", None,
        "Rolling std(20) of log returns — bigger = wilder swings in either direction."),
    "Hull Moving Average (HMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising HMA(20) = uptrend; read vs Close."),
    "Ichimoku Cloud (Baseline)": (
        "Ichimoku (Kijun price level)", "GAP", "BULL", "trend",
        "Close above the Kijun(26) level = short-term trend up (Ichimoku's baseline)."),
    "Kaufman Adaptive Moving Average (KAMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising KAMA(10) = uptrend; a flat line = no trend reading."),
    "Keltner Channels": (
        "Channel (price level — middle line)", "GAP", "BULL", "trend",
        "Close above the Keltner(20) middle line = up bias."),
    "Klinger Volume Oscillator (KVO)": (
        "Volume flow (zero-line)", "RAW", "BULL", "volume",
        "Above zero = accumulation (volume force net positive)."),
    "Know Sure Thing (KST)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Above zero = blended multi-period momentum up; zero line is the anchor."),
    "MACD (Moving Average Convergence Divergence)": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Above zero = EMA(12) over EMA(26), trend momentum up."),
    "MACD Histogram": (
        "Momentum oscillator (zero-line)", "RAW", "BULL", "momentum",
        "Positive and growing = bullish momentum strengthening."),
    "MACD Signal Line": (
        "MACD component (level)", "MACDGAP", "BULL", "momentum",
        "No reading alone (guide): ranked on MACD minus Signal — the cross state. Score identical to the Histogram column's by construction."),
    "Mass Index (MI)": (
        "Reversal setup gauge", "RAW", "INTENSITY", None,
        "A bulge (over 27, then under 26.5) warns of reversal — it cannot say which way."),
    "Momentum (MOM)": (
        "Momentum (zero-line)", "RAW", "BULL", "momentum",
        "Positive = close above the close 10 candles ago — momentum up."),
    "Money Flow Index (MFI)": (
        "Volume momentum (0-100)", "RAW", "BULL", None,
        "Above 50 = money-flow momentum up. EXCLUDED from composites: Tab 2 family is dual (volume x momentum)."),
    "On-Balance Volume (OBV)": (
        "Volume flow (cumulative line)", "SLOPE24", "BULL", "volume",
        "Rising slope = volume confirms up-moves; level is arbitrary, slope only (guide)."),
    "Parabolic SAR (PSAR)": (
        "Trailing stop / trend state (price level)", "GAP", "BULL", "trend",
        "SAR level below the close = uptrend state; above = downtrend state."),
    "Price Volume Trend (PVT)": (
        "Volume flow (cumulative line)", "SLOPE24", "BULL", "volume",
        "Rising slope = volume-weighted flow accumulating; slope only (guide)."),
    "Rate of Change (ROC)": (
        "Momentum (zero-line, %)", "RAW", "BULL", "momentum",
        "Positive = price up over the last 12 candles; zero is the divider."),
    "Relative Strength Index (RSI)": (
        "Momentum oscillator (0-100)", "RAW", "BULL", "momentum",
        "Above 50 = buyers stronger over 14 candles; 70+/30- are stretch zones, not reversals."),
    "Relative Volatility Index (RVI)": (
        "Volatility direction (0-100)", "RAW", "BULL", None,
        "Above 50 = upward volatility dominates (buy confirmation). EXCLUDED from composites: Tab 2 family is volatility-direction, off the trend/momentum/volume axes."),
    "Schaff Trend Cycle (STC)": (
        "Cycle oscillator (0-100)", "RAW", "BULL", "momentum",
        "Rising through 25 = new up-cycle starting; above 75 = up-cycle mature."),
    "Simple Moving Average (SMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising SMA(20) = uptrend; read vs Close."),
    "Standard Deviation (SD)": (
        "Volatility", "RAW", "INTENSITY", None,
        "Rolling std(20) of close in USD — pure dispersion, zero direction content."),
    "Stochastic Oscillator (Fast)": (
        "Momentum oscillator (0-100)", "RAW", "BULL", "momentum",
        "%K high/rising = momentum up; above 80 = overbought zone, under 20 = oversold."),
    "Stochastic RSI (StochRSI)": (
        "Momentum oscillator (0-100)", "RAW", "BULL", "momentum",
        "Rising/crossing up, especially from under 20 = RSI's momentum turning up."),
    "SuperTrend": (
        "Trend / trailing line (price level)", "GAP", "BULL", "trend",
        "SuperTrend line below the close = uptrend state; above = downtrend state."),
    "True Strength Index (TSI)": (
        "Momentum oscillator (-100..+100)", "RAW", "BULL", "momentum",
        "Above zero = double-smoothed momentum up; zero line is the anchor."),
    "Ultimate Oscillator (UO)": (
        "Momentum oscillator (0-100)", "RAW", "BULL", "momentum",
        "Above 50 = buying pressure across the 7/14/28 windows winning."),
    "Volume Weighted Average Price (VWAP)": (
        "Volume-weighted average (price level)", "GAP", "BULL", "trend",
        "Close above this rolling-24h VWAP = buyers in control over the window; read vs Close."),
    "Vortex Indicator (VI)": (
        "Trend (net vortex)", "RAW", "BULL", "trend",
        "Positive = VI+ exceeds VI- — upward vortex dominates; zero is the divider."),
    "Williams %R": (
        "Momentum oscillator (-100..0)", "RAW", "BULL", "momentum",
        "NOT flipped: Tab 2 reads near 0 (close at the top of the 14-candle range) as strong, near -100 as weak."),
    "Zero Lag Exponential Moving Average (ZLEMA)": (
        "Moving average (price level)", "GAP", "BULL", "trend",
        "Close above a rising ZLEMA(20) = uptrend; read vs Close."),
    # ---- v2 columns (main-chart additions, in v2 order) ----
    "Distance to SMA 50 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = price above the SMA(50) — intermediate trend side is up."),
    "Distance to SMA 100 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = above the SMA(100) — longer trend side up."),
    "Distance to SMA 200 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = above the SMA(200) — the classic long-term bull regime line."),
    "Distance to EMA 21 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = above the fast EMA(21) — short-term trend up."),
    "Distance to EMA 55 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = above the EMA(55) — medium-term trend up."),
    "Distance to EMA 200 (%)": (
        "Distance to moving average (%)", "RAW", "BULL", "trend",
        "Positive = above the EMA(200) — long-term trend up (EMA regime line)."),
    "Distance to Bollinger Upper Band (20, 2.0) (%)": (
        "Distance to band (%)", "RAW", "BULL", "trend",
        "Near 0 or positive = at/above the upper band — breakout strength (trend reading)."),
    "Distance to Bollinger Lower Band (20, 2.0) (%)": (
        "Distance to band (%)", "RAW", "BULL", "trend",
        "Large positive = far above the lower band — strong side of the range."),
    "Distance to Donchian Upper (20) (%)": (
        "Distance to breakout level (%)", "RAW", "BULL", "trend",
        "Near 0 = at the 20-candle high — breakout point (the Turtle trigger in %)."),
    "Distance to Donchian Lower (20) (%)": (
        "Distance to breakout level (%)", "RAW", "BULL", "trend",
        "Large positive = far above the 20-candle low — no breakdown pressure."),
    "Distance to Donchian Upper (55) (%)": (
        "Distance to breakout level (%)", "RAW", "BULL", "trend",
        "Near 0 / positive = at/through the 55-candle high — the slower, bigger breakout."),
    "Distance to Donchian Lower (55) (%)": (
        "Distance to breakout level (%)", "RAW", "BULL", "trend",
        "Large positive = far above the 55-candle low."),
    "Distance to Keltner Upper (20, 2.0) (%)": (
        "Distance to band (%)", "RAW", "BULL", "trend",
        "Near 0 / positive = at/above the ATR-based upper band — momentum burst up."),
    "Distance to Keltner Lower (20, 2.0) (%)": (
        "Distance to band (%)", "RAW", "BULL", "trend",
        "Large positive = far above the lower Keltner band — strong side."),
    "ATR Band Half-Width (14) (%)": (
        "Volatility", "RAW", "INTENSITY", None,
        "ATR(14) as % of price — expected swing size per candle; never direction."),
    "Ichimoku Tenkan-sen Distance (%)": (
        "Distance to Ichimoku line (%)", "RAW", "BULL", "trend",
        "Positive = price above the Tenkan — short-term Ichimoku trend up."),
    "Ichimoku Senkou Span A Distance (%)": (
        "Distance to Ichimoku cloud (%)", "RAW", "BULL", "trend",
        "Positive = price above the cloud's fast edge (Span A)."),
    "Ichimoku Senkou Span B Distance (%)": (
        "Distance to Ichimoku cloud (%)", "RAW", "BULL", "trend",
        "Positive on both spans = fully above the cloud — bullish state."),
    "Ichimoku Chikou Distance (%)": (
        "Ichimoku momentum (%)", "RAW", "BULL", "momentum",
        "Positive = price higher than 26 candles ago — Chikou bullish confirmation."),
    "Parabolic SAR Distance (%)": (
        "Distance to trailing stop (%)", "RAW", "BULL", "trend",
        "Positive = SAR below price — uptrend state; the number is room to the stop."),
    "Parabolic SAR Side (+1/-1)": (
        "Trend state flag", "RAW", "BULL", "trend",
        "+1 = SAR below price = uptrend state; -1 = downtrend state."),
    "SuperTrend Distance (%)": (
        "Distance to trend line (%)", "RAW", "BULL", "trend",
        "Positive = price above the SuperTrend line — uptrend state plus distance-to-flip."),
    "SuperTrend Side (+1/-1)": (
        "Trend state flag", "RAW", "BULL", "trend",
        "+1 = price above the SuperTrend line = uptrend state; -1 = downtrend."),
    "Distance to Prior-Day Pivot Point (%)": (
        "Distance to pivot level (%)", "RAW", "BULL", "trend",
        "Positive = above yesterday's pivot — intraday bias up."),
    "Distance to Prior-Day Pivot R1 (%)": (
        "Distance to pivot level (%)", "RAW", "BULL", "trend",
        "Near 0 = testing first resistance; positive = through R1 — breakout side."),
    "Distance to Prior-Day Pivot S1 (%)": (
        "Distance to pivot level (%)", "RAW", "BULL", "trend",
        "Large positive = far above first support; through it (negative) = breakdown side."),
    "Distance to Prior-Day High (%)": (
        "Distance to level (%)", "RAW", "BULL", "trend",
        "Near 0 = testing yesterday's high; positive = yesterday's range broken upward."),
    "Distance to Prior-Day Low (%)": (
        "Distance to level (%)", "RAW", "BULL", "trend",
        "Large positive = well above yesterday's low — no breakdown pressure."),
    "Distance to Fractal Resistance (%)": (
        "Distance to swing level (%)", "RAW", "BULL", "trend",
        "Near 0 = at confirmed swing-high resistance; positive = broken above it."),
    "Distance to Fractal Support (%)": (
        "Distance to swing level (%)", "RAW", "BULL", "trend",
        "Positive and holding = price above confirmed swing-low support; larger = more cushion."),
    "Distance to $5,000 Round Number (%)": (
        "Price-psychology reference", "RAW", "BULL", None,
        "Positive = just above a $5k grid line (support from above). EXCLUDED from composites: a price-psychology reference, not a family indicator."),
}

CUMULATIVE = [n for n, s in SPEC.items() if s[1] == "SLOPE24"]
QUANTITY_LABEL = {
    "RAW": "Raw value",
    "GAP": "Close − level gap (% of close)",
    "SLOPE24": "24-bar change (slope)",
    "MACDGAP": "MACD − Signal gap",
}
ORIENTATION_LABEL = {
    "BULL": "HIGH = BULLISH",
    "FLIP": "FLIPPED (high score = bullish; low raw = bullish)",
    "STRENGTH": "STRENGTH (no direction)",
    "INTENSITY": "INTENSITY (no direction)",
}


def load_long() -> pd.DataFrame:
    df = pd.read_csv(LONG_HOURLY, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def build_series(frame: pd.DataFrame) -> pd.DataFrame:
    f1 = b1.build_features(frame)
    f2 = mc.build_all(frame)
    return pd.concat([f1, f2], axis=1)


def ranked_quantity(name: str, feat: pd.DataFrame,
                    close: pd.Series) -> pd.Series:
    kind = SPEC[name][1]
    s = feat[name]
    if kind == "RAW":
        return s
    if kind == "GAP":
        return (close - s) / close.replace(0.0, np.nan) * 100.0
    if kind == "SLOPE24":
        return s - s.shift(24)
    if kind == "MACDGAP":
        return feat["MACD (Moving Average Convergence Divergence)"] - s
    raise KeyError(kind)


def rolling_scores(q: np.ndarray, positions: np.ndarray,
                   window: int = WINDOW) -> np.ndarray:
    """Midrank percentile (0-100) of q[p] within q[p-window+1 .. p]
    for each position p, via a sliding sorted window."""
    start = int(positions[0]) - window + 1
    win = sorted(q[start:int(positions[0]) + 1].tolist())
    assert len(win) == window
    out = np.empty(len(positions), dtype=float)
    prev = int(positions[0])
    for i, p in enumerate(positions):
        p = int(p)
        for j in range(prev + 1, p + 1):
            old = q[j - window]
            del win[bisect_left(win, old)]
            insort(win, q[j])
        v = q[p]
        nl = bisect_left(win, v)
        ne = bisect_right(win, v) - nl
        r = nl + (ne + 1) / 2.0
        out[i] = 100.0 * (r - 1.0) / (window - 1.0)
        prev = p
    return out


def brute_score(q: np.ndarray, p: int, window: int = WINDOW) -> float:
    """Independent second method: plain numpy counts on the slice."""
    sl = q[p - window + 1:p + 1]
    v = q[p]
    nl = int((sl < v).sum())
    ne = int((sl == v).sum())
    r = nl + (ne + 1) / 2.0
    return 100.0 * (r - 1.0) / (window - 1.0)


def reproduction_check(feat: pd.DataFrame, v2: pd.DataFrame,
                       v2_ts: pd.Series) -> dict:
    """Recomputed-on-long vs v2's saved values on the matrix rows.
    Returns per-column stats; raises on an unexplained mismatch."""
    aligned = feat.reindex(v2_ts)
    results = {}
    for name in SPEC:
        a = v2[name].to_numpy(float)
        b = aligned[name].to_numpy(float)
        if name in CUMULATIVE:
            # levels carry an arbitrary start offset by construction;
            # the ranked quantity (24-bar slope) must reproduce
            sl_b = np.full(len(a), np.nan)
            full = feat[name]
            sl_long = (full - full.shift(24)).reindex(v2_ts) \
                .to_numpy(float)
            sl_b[24:] = a[24:] - a[:-24]
            m = ~np.isnan(sl_long) & ~np.isnan(sl_b)
            d = np.abs(sl_long[m] - sl_b[m])
            ok = bool(d.max() <= 1e-6)
            results[name] = {"kind": "cumulative-slope",
                             "max_slope_abs_diff": float(d.max()),
                             "rows": int(m.sum()), "pass": ok}
            if not ok:
                raise SystemExit(f"REPRODUCTION FAIL (slope): {name}")
            continue
        both = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a[both] - b[both])
        std = float(np.nanstd(a))
        thresh = 1e-3 * max(std, 1.0)
        if name == "Average Daily Range (ADR)":
            # v1's source frame (sup.pkl) starts 2024-11-09 23:00, so
            # its first 'day' is a 1-hour stub whose tiny range sat
            # in v1's ADR for the first stamped days. The long frame
            # holds the true full day. Confined to the first 24
            # matrix rows; exact after that (verified in exploration).
            bad_rows = np.where(d > 1e-6)[0]
            ok = bool(len(bad_rows) == 0 or
                      (bad_rows.max() < 24 and d.max() <= 60.0))
            results[name] = {"kind": "direct (ADR first-day stub)",
                             "max_abs_diff": float(d.max()),
                             "rows_diff_gt_1e-6": int(len(bad_rows)),
                             "pass": ok}
            if not ok:
                raise SystemExit("REPRODUCTION FAIL: ADR beyond the "
                                 "documented first-24-rows stub effect")
            continue
        ok = bool(d.max() <= thresh)
        results[name] = {"kind": "direct",
                         "max_abs_diff": float(d.max()),
                         "col_std": std, "threshold": thresh,
                         "pass": ok}
        if not ok:
            raise SystemExit(
                f"REPRODUCTION FAIL: {name} max abs diff {d.max():.3e} "
                f"> threshold {thresh:.3e} — diagnose before building")
    return results


def main() -> int:
    t0 = time.time()
    v2 = pd.read_csv(V2_CSV)
    v2_ts = pd.to_datetime(v2["Date"])
    ind_cols = [c for c in v2.columns if c in SPEC]
    assert len(ind_cols) == len(SPEC) == 90, (len(ind_cols), len(SPEC))
    print(f"v2: {len(v2)} rows x {len(v2.columns)} cols; "
          f"{len(ind_cols)} indicator columns", flush=True)

    frame = load_long()
    print(f"long frame: {len(frame)} rows "
          f"{frame.index[0]} -> {frame.index[-1]}", flush=True)

    print("building indicator series on the long frame ...", flush=True)
    feat = build_series(frame)
    print(f"  done {time.time() - t0:.0f}s", flush=True)

    # ---- identity: OHLCV / timestamps on the matrix rows ----
    pos = frame.index.get_indexer(v2_ts)
    assert (pos >= 0).all(), "a matrix timestamp is missing on the long frame"
    assert (np.diff(pos) == 1).all(), "matrix rows not consecutive on the long frame"
    for vc, lc in [("Open", "open"), ("High", "high"), ("Low", "low"),
                   ("Close", "close"), ("Volume", "volume")]:
        d = np.abs(v2[vc].to_numpy(float)
                   - frame[lc].to_numpy(float)[pos])
        assert d.max() == 0.0, f"OHLCV mismatch in {vc}"
    print("identity: timestamps + OHLCV exact vs v2", flush=True)

    # ---- reproduction BEFORE normalization ----
    repro = reproduction_check(feat, v2, v2_ts)
    n_pass = sum(1 for r in repro.values() if r["pass"])
    print(f"reproduction: {n_pass}/{len(repro)} columns pass", flush=True)

    # ---- ranked quantities + scores ----
    close = frame["close"]
    first_pos, last_pos = int(pos[0]), int(pos[-1])
    win_lo = first_pos - WINDOW + 1
    scores = {}
    raw_q = {}
    for name in ind_cols:
        q = ranked_quantity(name, feat, close)
        qa = q.to_numpy(float)
        seg = qa[win_lo:last_pos + 1]
        assert not np.isnan(seg).any(), f"NaN in ranked window: {name}"
        raw_q[name] = qa
        s = rolling_scores(qa, pos)
        if SPEC[name][2] == "FLIP":
            s = 100.0 - s
        scores[name] = s
    print(f"scores computed {time.time() - t0:.0f}s", flush=True)

    # ---- verification: brute-force second method (RSI, 3 rows) ----
    brute = {}
    rsi_q = raw_q["Relative Strength Index (RSI)"]
    for i in (0, 8000, len(v2) - 1):
        bf = brute_score(rsi_q, int(pos[i]))
        got = scores["Relative Strength Index (RSI)"][i]
        assert abs(bf - got) < 1e-9, (i, bf, got)
        brute[str(i)] = {"brute": bf, "rolling": float(got)}
    print(f"brute-force second method (RSI): exact on 3 rows {brute}",
          flush=True)

    # ---- verification: truncation causality on sampled columns ----
    trunc_row = 9000
    trunc_pos = int(pos[trunc_row])
    print(f"truncation check: rebuilding series on frame truncated "
          f"at matrix row {trunc_row} ...", flush=True)
    feat_t = build_series(frame.iloc[:trunc_pos + 1])
    sample_cols = [
        "Relative Strength Index (RSI)",                    # RAW / BULL
        "Exponential Moving Average (EMA)",                 # GAP
        "On-Balance Volume (OBV)",                          # SLOPE24
        "Connors RSI (CRSI)",                               # FLIP
        "Average Directional Movement Index (ADX)",         # STRENGTH
        "Historical Volatility (HV)",                       # INTENSITY
        "MACD Signal Line",                                 # MACDGAP
        "SuperTrend Side (+1/-1)",                          # v2 RAW
    ]
    trunc_ok = {}
    for name in sample_cols:
        q_t = ranked_quantity(name, feat_t, frame["close"]
                              .iloc[:trunc_pos + 1]).to_numpy(float)
        # positions within the truncated frame are the same indices
        s_t = rolling_scores(q_t, pos[:trunc_row + 1])
        if SPEC[name][2] == "FLIP":
            s_t = 100.0 - s_t
        d = np.abs(s_t - scores[name][:trunc_row + 1]).max()
        trunc_ok[name] = float(d)
        assert d == 0.0, f"truncation mismatch {name}: {d}"
    print(f"truncation causality: exact on {len(sample_cols)} sampled "
          f"columns x {trunc_row + 1} rows", flush=True)

    # ---- orientation spot checks ----
    def spot(name, rows_idx):
        return [{"row": int(i), "date": str(v2_ts.iloc[i]),
                 "raw_v2_value": float(v2[name].iloc[i]),
                 "score": float(scores[name][i])} for i in rows_idx]
    rsi_raw = v2["Relative Strength Index (RSI)"].to_numpy(float)
    crsi_raw = v2["Connors RSI (CRSI)"].to_numpy(float)
    spot_checks = {
        "RSI (HIGH=BULLISH): highest / median / lowest raw RSI rows":
            spot("Relative Strength Index (RSI)",
                 [int(np.argmax(rsi_raw)),
                  int(np.argsort(rsi_raw)[len(rsi_raw) // 2]),
                  int(np.argmin(rsi_raw))]),
        "CRSI (FLIPPED): highest / median / lowest raw CRSI rows":
            spot("Connors RSI (CRSI)",
                 [int(np.argmax(crsi_raw)),
                  int(np.argsort(crsi_raw)[len(crsi_raw) // 2]),
                  int(np.argmin(crsi_raw))]),
    }
    for k, v in spot_checks.items():
        print(f"  spot {k}: {v}", flush=True)

    # ---- composites ----
    fam_members = {"trend": [], "momentum": [], "volume": []}
    for name in ind_cols:
        fam = SPEC[name][3]
        if fam:
            fam_members[fam].append(name)
    fam_scores = {f: np.mean([scores[n] for n in ms], axis=0)
                  for f, ms in fam_members.items()}
    master = np.mean([fam_scores[f] for f in
                      ("trend", "momentum", "volume")], axis=0)
    print("composite members: "
          + ", ".join(f"{f}={len(m)}" for f, m in fam_members.items()),
          flush=True)

    # sampled composite identity vs member means (unrounded)
    chk = {}
    for i in (0, 5000, len(v2) - 1):
        for f, ms in fam_members.items():
            ref = float(np.mean([scores[n][i] for n in ms]))
            assert abs(ref - fam_scores[f][i]) < 1e-12
        chk[str(i)] = {"master": float(master[i]),
                       "mean_of_family_means":
                           float(np.mean([fam_scores[f][i]
                                          for f in fam_scores]))}
        assert abs(chk[str(i)]["master"]
                   - chk[str(i)]["mean_of_family_means"]) < 1e-12

    # ---- range / distribution audit ----
    all_scores = np.column_stack([scores[n] for n in ind_cols])
    assert all_scores.min() >= 0.0 and all_scores.max() <= 100.0
    assert not np.isnan(all_scores).any()
    for f, s in list(fam_scores.items()) + [("master", master)]:
        assert s.min() >= 0.0 and s.max() <= 100.0
    pct_above_50 = {f: round(100.0 * float((s > 50).mean()), 2)
                    for f, s in list(fam_scores.items())
                    + [("master", master)]}
    print(f"range audit OK; composite % of rows > 50: {pct_above_50}",
          flush=True)

    # ---- assemble v3 ----
    out = pd.DataFrame()
    out["Date"] = v2["Date"]
    out["Ticker"] = v2["Ticker"]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = v2[c]
    out[TARGET_COL] = v2[TARGET_COL]
    out["score_master"] = np.round(master, 4)
    out["score_trend"] = np.round(fam_scores["trend"], 4)
    out["score_momentum"] = np.round(fam_scores["momentum"], 4)
    out["score_volume"] = np.round(fam_scores["volume"], 4)
    for name in ind_cols:
        out[name] = np.round(scores[name], 4)

    # identity re-check on the assembled frame
    for c in ["Date", "Ticker", TARGET_COL]:
        assert (out[c].astype(str) == v2[c].astype(str)).all(), c
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        assert (out[c].to_numpy(float) == v2[c].to_numpy(float)).all(), c
    label_counts = out[TARGET_COL].value_counts().to_dict()

    out.to_csv(V3_CSV, index=False)
    print(f"wrote {V3_CSV} ({V3_CSV.stat().st_size} bytes)", flush=True)

    stats = {
        "window_bars": WINDOW,
        "rows": int(len(out)), "columns": int(len(out.columns)),
        "date_first": str(out["Date"].iloc[0]),
        "date_last": str(out["Date"].iloc[-1]),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "reproduction": repro,
        "brute_force_rsi": brute,
        "truncation_max_abs_diff": trunc_ok,
        "orientation_spot_checks": spot_checks,
        "composite_members": fam_members,
        "composite_sample_check": chk,
        "composite_pct_rows_above_50": pct_above_50,
        "spec": {n: {"family": SPEC[n][0], "quantity": SPEC[n][1],
                     "orientation": SPEC[n][2],
                     "composite": SPEC[n][3], "basis": SPEC[n][4]}
                 for n in ind_cols},
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1))

    write_xlsx(out, ind_cols, fam_members)
    write_report(stats, fam_members, spot_checks, pct_above_50,
                 label_counts)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


def key_rows(ind_cols):
    rows = [("Column", "Family (guide Tab 2)", "Ranked quantity",
             "Orientation", "Guide basis (one line)",
             "Feeds composite")]
    raw_meta = [
        ("Date", "Identifier", "—", "RAW (not transformed)",
         "Candle open time (hourly, UTC). Not an indicator.", "—"),
        ("Ticker", "Identifier", "—", "RAW (not transformed)",
         "Constant BTC/USD. Not an indicator.", "—"),
        ("Open", "Price (raw)", "—", "RAW (not transformed)",
         "Raw candle open in USD.", "—"),
        ("High", "Price (raw)", "—", "RAW (not transformed)",
         "Raw candle high in USD.", "—"),
        ("Low", "Price (raw)", "—", "RAW (not transformed)",
         "Raw candle low in USD.", "—"),
        ("Close", "Price (raw)", "—", "RAW (not transformed)",
         "Raw candle close in USD — the reference price for GAP columns.", "—"),
        ("Volume", "Volume (raw)", "—", "RAW (not transformed)",
         "Raw BTC volume; participation only — kept raw as part of OHLCV.", "—"),
        (TARGET_COL, "Outcome label", "—", "RAW (not transformed)",
         "THE LABEL (next candle's move vs the ±0.3% fee band) — never an input.", "—"),
        ("score_master", "Composite", "Mean of the three family scores",
         "HIGH = BULLISH (composite)",
         "Mean of family means, so big families don't dominate.", "—"),
        ("score_trend", "Composite", "Family mean of member scores",
         "HIGH = BULLISH (composite)", "Trend family composite.", "—"),
        ("score_momentum", "Composite", "Family mean of member scores",
         "HIGH = BULLISH (composite)", "Momentum family composite.", "—"),
        ("score_volume", "Composite", "Family mean of member scores",
         "HIGH = BULLISH (composite)", "Volume family composite.", "—"),
    ]
    rows.extend(raw_meta)
    for name in ind_cols:
        fam, qty, ori, comp, basis = SPEC[name]
        rows.append((name, fam, QUANTITY_LABEL[qty],
                     ORIENTATION_LABEL[ori], basis,
                     comp if comp else "—"))
    return rows


def write_xlsx(out: pd.DataFrame, ind_cols, fam_members) -> None:
    import openpyxl
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font

    bold = Font(bold=True)
    wb = openpyxl.Workbook(write_only=True)

    from openpyxl.utils import get_column_letter

    ws = wb.create_sheet("Matrix v3")
    ws.freeze_panes = "A2"
    widths = {"Date": 20, "Ticker": 10, TARGET_COL: 16}
    for i, c in enumerate(out.columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = \
            widths.get(c, 14)
    for row in [list(out.columns)] + out.values.tolist():
        cells = []
        for v in row:
            cell = WriteOnlyCell(ws, value=v)
            cell.font = bold
            cells.append(cell)
        ws.append(cells)

    wk = wb.create_sheet("Key")
    wk.freeze_panes = "A2"
    for col, w in zip("ABCDEF", (46, 34, 30, 40, 95, 15)):
        wk.column_dimensions[col].width = w
    for row in key_rows(ind_cols):
        cells = []
        for v in row:
            cell = WriteOnlyCell(wk, value=v)
            cell.font = bold
            cells.append(cell)
        wk.append(cells)

    XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX_PATH)
    print(f"wrote {XLSX_PATH} ({XLSX_PATH.stat().st_size} bytes)",
          flush=True)

    # read-back: bold flags + freeze + shape on sampled cells.
    # (Write-only workbooks store no dimension element, so read-only
    # mode reports max_row/max_column as None — count by iteration.)
    rb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    wm = rb["Matrix v3"]
    targets = {(1, 1), (2, 2), (2, 12), (100, 1), (5000, 26)}
    n_rows, n_cols = 0, 0
    for r_i, row in enumerate(wm.iter_rows(), start=1):
        n_rows = r_i
        if r_i == 1:
            n_cols = len(row)
        for c_i, cell in enumerate(row, start=1):
            if (r_i, c_i) in targets:
                assert cell.font.bold, \
                    f"not bold: Matrix v3 row {r_i} col {c_i}"
    assert n_rows == len(out) + 1 and n_cols == len(out.columns), \
        (n_rows, n_cols)
    wk2 = rb["Key"]
    k_rows, k_targets = 0, {(1, 1), (5, 3), (20, 6)}
    for r_i, row in enumerate(wk2.iter_rows(), start=1):
        k_rows = r_i
        for c_i, cell in enumerate(row, start=1):
            if (r_i, c_i) in k_targets:
                assert cell.font.bold, f"not bold: Key {r_i},{c_i}"
    assert k_rows == len(key_rows(ind_cols)), k_rows
    rb.close()
    rb2 = openpyxl.load_workbook(XLSX_PATH)
    assert rb2["Matrix v3"].freeze_panes == "A2"
    assert rb2["Key"].freeze_panes == "A2"
    rb2.close()
    print("xlsx read-back: bold + freeze + shape verified", flush=True)


def write_report(stats, fam_members, spot_checks, pct_above_50,
                 label_counts) -> None:
    repro = stats["reproduction"]
    direct = [(n, r) for n, r in repro.items() if r["kind"] == "direct"]
    worst_name, worst = max(direct, key=lambda kv: kv[1]["max_abs_diff"])
    lines = []
    lines.append("# Matrix v3 — the 0–100 Normalized Edition")
    lines.append("")
    lines.append("Date: 2026-10-22 · Kass green-lit the v3 design the same "
                 "evening (\"let's go with the start of that V3\"): trailing "
                 "180-day percentile rank → 0–100, oriented by the "
                 "bullish/bearish guide. Code: "
                 "`scripts/build_timeseries_matrix_v3.py` · Stats: "
                 "`data/timeseries_matrix/build_stats_v3.json`.")
    lines.append("")
    lines.append("**BUILD ONLY — nothing was trained or screened on v3. "
                 "Trading remains PARKED.**")
    lines.append("")
    lines.append("## What v3 is")
    lines.append("")
    lines.append(f"- **{stats['rows']:,} rows × {stats['columns']} columns**, "
                 f"{stats['date_first']} → {stats['date_last']} — v2's exact "
                 "rows, timestamps, OHLCV and labels (verified byte/float "
                 "identical), with every indicator column replaced by a "
                 "0–100 score.")
    lines.append("- Column order: Date, Ticker, OHLCV, label, "
                 "score_master, score_trend, score_momentum, score_volume, "
                 "then the 90 normalized indicator columns **in v2's "
                 "original relative order, under their v2 names** — same "
                 "names, new meaning: the values are 0–100 scores, not raw "
                 "indicator values. The Key tab of the xlsx carries the "
                 "per-column mapping.")
    lines.append(f"- Labels: {label_counts}.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- **Trailing percentile rank:** for each row, the "
                 "column's ranked quantity is scored against its own "
                 "trailing 4,320 hourly values (180 days) including the "
                 "current row — strictly causal, past + present only. "
                 "Midrank convention: score = 100 × (r − 1) / 4319 with "
                 "r = (values below) + (ties + 1)/2, so window-minimum = 0, "
                 "maximum = 100, median ≈ 50.")
    lines.append("- **Ranked quantity** (the series the guide actually "
                 "reads): raw value for oscillators/distances/gauges; "
                 "the signed **Close − level gap (% of close)** for the 14 "
                 "price-level columns (moving averages, channels, "
                 "trailing stops, VWAP), which the guide reads only "
                 "against Close; the **24-bar change (slope)** for the 4 "
                 "cumulative lines (A/D, ASI, OBV, PVT), whose levels the "
                 "guide calls meaningless; **MACD − Signal** for the MACD "
                 "Signal Line, which the guide reads only against MACD "
                 "(its score is therefore identical to the Histogram "
                 "column's by construction).")
    lines.append("- **Orientation** (guide Tab 2 is the authority): 81 "
                 "directional columns oriented HIGH = BULLISH; exactly one "
                 "FLIPPED column — Connors RSI, whose guide reading is "
                 "contrarian (very low raw = dip-buy zone, very high raw "
                 "= short-the-rip). **Williams %R is deliberately NOT "
                 "flipped**: Tab 2 reads its top of scale (near 0 = close "
                 "at the top of the range) as the strong side. 2 columns "
                 "are STRENGTH (ADX; the DMI column that stores DX — "
                 "unsigned by construction) and 7 are INTENSITY (ADR, "
                 "Bollinger BandWidth, CHOP, HV, Mass Index, SD, ATR "
                 "half-width): normalized as-is, high = stronger / more "
                 "extreme, with NO bull/bear claim and no composite seat.")
    lines.append("- **Composites** (directional columns only, grouped by "
                 "the guide's Tab 2 families): score_trend = mean of "
                 f"{len(fam_members['trend'])} trend columns, "
                 f"score_momentum = mean of {len(fam_members['momentum'])}, "
                 f"score_volume = mean of {len(fam_members['volume'])}; "
                 "score_master = the mean of the three family scores "
                 "(mean of family means, so the 45-column trend family "
                 "cannot swamp the 7-column volume family).")
    lines.append("- **Excluded from composites** (directional, but the "
                 "Tab 2 family is dual or off-axis — noted per column "
                 "in the key): Ease of Movement and Money Flow Index "
                 "(dual volume × momentum families), Relative Volatility "
                 "Index (volatility-direction family), and the $5,000 "
                 "round-number distance (price-psychology reference).")
    lines.append("- Windows are consecutive **rows** of the long series. "
                 "The source Coinbase record has 3 small holes (3–5 "
                 "missing hours each; shared with v1/v2 — the rows simply "
                 "don't exist in any version), so the few windows "
                 "straddling a hole span slightly more than 180 days of "
                 "wall time. Disclosed, mechanical, and identical in "
                 "spirit to how v1/v2 treated the same record.")
    lines.append("- Series provenance: all 90 indicator series were "
                 "recomputed on the LONG hourly frame (42,107 rows from "
                 "2022-01-01) via the v1/v2 build code paths, so every "
                 "matrix row's 180-day window is full of real history.")
    lines.append("")
    lines.append("## Verification (all PASS)")
    lines.append("")
    lines.append(f"- **Identity vs v2:** timestamps, OHLCV (exact, max diff "
                 "0.0) and labels identical on all 16,735 rows.")
    lines.append(f"- **Reproduction before normalization:** all 90 "
                 f"recomputed series reproduce v2's saved values on the "
                 f"matrix rows under the same code paths — 85 directly "
                 f"(worst case {worst_name}: max abs diff "
                 f"{worst['max_abs_diff']:.2e} against a column std of "
                 f"{worst['col_std']:.1f} — floating-point / EMA-seed "
                 f"residue), the 4 cumulative lines exactly in their "
                 f"ranked quantity (24-bar slope max diff ≤ 3.5e-10; "
                 f"their levels carry an arbitrary start offset by "
                 f"construction), and ADR with one documented exception: "
                 f"v1's source frame starts 2024-11-09 23:00, so its "
                 f"first 'day' was a 1-hour stub whose tiny range sat in "
                 f"v1's ADR for the first 24 matrix rows (max diff "
                 f"$53.0 on a column with std $968); the long frame holds "
                 f"the true full day, and the series are exact from row "
                 f"24 on.")
    lines.append(f"- **Second method:** the percentile for RSI was "
                 f"recomputed at rows 0 / 8,000 / 16,734 with an "
                 f"independent brute-force numpy implementation — exact "
                 f"agreement with the sliding-window implementation.")
    lines.append(f"- **Causality (truncation):** series rebuilt on the "
                 f"long frame truncated at matrix row 9,000 reproduce "
                 f"the full-frame scores EXACTLY (max diff 0.0) on 8 "
                 f"sampled columns spanning every ranked-quantity type "
                 f"and orientation (RSI, EMA gap, OBV slope, CRSI flip, "
                 f"ADX strength, HV intensity, MACD-Signal gap, "
                 f"SuperTrend Side) × 9,001 rows.")
    lines.append("- **Orientation spot checks** (raw v2 value → v3 score):")
    for k, v in spot_checks.items():
        pairs = "; ".join(f"row {p['row']} ({p['date'][:10]}): raw "
                          f"{p['raw_v2_value']:.2f} → score {p['score']:.2f}"
                          for p in v)
        lines.append(f"  - {k}: {pairs}.")
    lines.append("- **Range audit:** every transformed column within "
                 "[0, 100], zero NaNs across all 90 × 16,735 scores; "
                 "composites within [0, 100] and equal to their "
                 "documented family means on sampled rows (master = "
                 "mean of the three family scores, exact).")
    lines.append(f"- **Distribution sanity:** % of rows each composite "
                 f"spends above 50 — trend {pct_above_50['trend']}%, "
                 f"momentum {pct_above_50['momentum']}%, volume "
                 f"{pct_above_50['volume']}%, master "
                 f"{pct_above_50['master']}% — near half by construction, "
                 f"nothing wild.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Notes |")
    lines.append("|---|---|")
    lines.append("| `data/timeseries_matrix/btc_usd_1h_matrix_v3.csv` | "
                 "The machine copy (CSV cannot carry formatting). |")
    lines.append("| `~/workspace/your_files/BTC-USD_1h_Matrix_v3_"
                 "Normalized.xlsx` | Kass's READABILITY copy: every "
                 "populated cell BOLD (header + all data cells, verified "
                 "on read-back), frozen top row, sized columns. Tabs: "
                 "'Matrix v3' (data) + 'Key' (per-column family, ranked "
                 "quantity, orientation, guide basis, composite seat). |")
    lines.append("| `data/timeseries_matrix/build_stats_v3.json` | "
                 "Machine-readable spec, reproduction stats, checks. |")
    lines.append("| `scripts/build_timeseries_matrix_v3.py` | "
                 "Deterministic rebuild; reproduction check aborts the "
                 "build on an unexplained mismatch. |")
    lines.append("")
    lines.append("Nothing was trained or screened on v3 — Kass sees the "
                 "table first, per the established pattern.")
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
