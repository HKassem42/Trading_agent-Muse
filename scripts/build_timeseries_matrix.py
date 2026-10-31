"""Build the REAL BTC/USD version of Kass's Time-Series ML Training
Data Matrix (Kass green-lit 2026-10-22: "create the real deal one").

Kass's template (refs/Time-Series_ML_Training_Data_Matrix_from_drive.xlsx)
defines the shape: Date | Ticker | Target_Action | OHLCV | 62 indicator
columns, one row per observation. This script fills that exact shape
with the project's real Coinbase 1h BTC/USD frame -- the same cached
supervised frame the diagnosis battery and the A-Z screening use
(data/diagnosis/sup.pkl, 17,049 rows, 2024-11-09 23:00 ->
2026-10-21 17:00 UTC). Real data only; no synthetic values anywhere.

Values are RAW indicator values in natural units (RSI 0-100, moving
averages at price level, OBV cumulative), NOT the [-1, 1] screening
scalings. Definitions/parameters reuse src/features/az_indicators.py
builders wherever a series exists there (that module is read-only
here); the distance-form az builders (e.g. (close - MA)/close) are
recomputed in their raw level form from the same pandas_ta calls.

Excluded template columns (never fabricated):
- Advance/Decline Line (A/D Line): market breadth -- needs every
  stock on an exchange; a single asset has no advance/decline counts.
- CBOE Volatility Index (VIX): an external index, not derivable
  from BTC candles.
- Beta: needs a benchmark asset series to regress against.
Output therefore has 67 columns (70 - 3).

Target label (computed, never hand-assigned): from the NEXT candle's
return r = close[t+1]/close[t] - 1: BUY if r >= +0.003, SHORT if
r <= -0.003, else HOLD (the 0.3% band = round-trip fees 2 x 0.15%,
so HOLD = the next move would not have covered trading costs).
SELL is deliberately never generated: a sell is an exit decision
that depends on holding a position, which a stateless row does not
have; the enum stays in the header for live-policy use. The final
row (no next candle) is dropped.

Causality: every feature at row t uses only candles <= t (trailing
windows / recursive filters / completed-day stamps, mirroring the
az library). A truncation self-check runs on every build: the
feature frame is recomputed on the first TRUNC_N rows and shared
rows must match the full-frame values exactly on a 9-column sample
spanning price-MA, oscillator, volume and volatility families.

Lab only: no training, no registration, no promotion, trading stays
PARKED. Usage: .venv/bin/python scripts/build_timeseries_matrix.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas_ta as ta  # noqa: E402

from features import az_indicators as az  # noqa: E402  (read-only reuse)

SUP_PATH = ROOT / "data" / "diagnosis" / "sup.pkl"
OUT_DIR = ROOT / "data" / "timeseries_matrix"
CSV_PATH = OUT_DIR / "btc_usd_1h_matrix.csv"
STATS_PATH = OUT_DIR / "build_stats.json"

TRUNC_N = 9000
FEE_BAND = 0.003  # 2 x 0.15% round-trip

TARGET_COL = "Target_Action (BUY/SELL/SHORT/HOLD)"
EXCLUDED = {
    "Advance/Decline Line (A/D Line)":
        "market breadth: needs advance/decline counts across a whole "
        "exchange; not derivable from a single asset's candles",
    "Beta":
        "needs a benchmark asset series to regress against; none is "
        "part of this dataset",
    "CBOE Volatility Index (VIX)":
        "external CBOE index; not derivable from BTC candles",
}


# ------------------------------------------------------------------
# raw builders for columns the az library only carries in distance /
# momentum form, or does not carry at all. All causal.
# ------------------------------------------------------------------

def _col(res: pd.DataFrame, prefix: str) -> pd.Series:
    for c in res.columns:
        if str(c).startswith(prefix):
            return res[c]
    raise KeyError(f"no column starting with {prefix!r}")


def _apo_raw(df):
    c = df["close"]
    return az._ema(c, 12) - az._ema(c, 26)


def _ad_line(df):
    return ta.ad(df["high"], df["low"], df["close"], df["volume"])


def _asi(df):
    """Wilder Accumulative Swing Index, cumulative sum of the Swing
    Index. BTC has no exchange limit move (Wilder's T), so T is set
    to the bar's own K = max(|H - C_prev|, |L - C_prev|), making the
    K/T factor 1 -- the standard limit-free form. Causal loop."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)  # noqa: E741
    c = df["close"].to_numpy(float)
    si = np.zeros(len(df))
    for t in range(1, len(df)):
        cp, op = c[t - 1], o[t - 1]
        a = abs(h[t] - cp)
        b = abs(l[t] - cp)
        cc = abs(h[t] - l[t])
        d = abs(cp - op)
        if a >= b and a >= cc:
            r = a - 0.5 * b + 0.25 * d
        elif b >= a and b >= cc:
            r = b - 0.5 * a + 0.25 * d
        else:
            r = cc + 0.25 * d
        if r == 0:
            si[t] = 0.0
        else:
            si[t] = 50.0 * ((c[t] - cp) + 0.5 * (c[t] - o[t])
                            + 0.25 * (cp - op)) / r
    return pd.Series(np.cumsum(si), index=df.index)


def _adr_raw(df):
    """Average Daily Range in price units: mean (high - low) of the
    prior 14 COMPLETED days, stamped on the following day -- the raw
    form of az _adr_rel (same completed-day discipline)."""
    d = az._daily_frame(df)
    day_rng = (d["high"] - d["low"]).to_numpy()
    out = np.full(len(day_rng), np.nan)
    for i in range(14, len(day_rng)):
        out[i] = day_rng[i - 14:i].mean()
    return az._stamp_next_day(df, out, d.index)


def _dmi_dx(df):
    """Directional Movement Index = DX(14): 100 * |+DI - -DI| /
    (+DI + -DI) from the same pandas_ta ADX(14) result the az
    library screens."""
    r = ta.adx(df["high"], df["low"], df["close"], length=14)
    dmp, dmn = _col(r, "DMP_"), _col(r, "DMN_")
    return 100.0 * (dmp - dmn).abs() / (dmp + dmn).replace(0.0, np.nan)


def _donchian_mid(df):
    r = az._donchian(df)
    return (_col(r, "DCU_") + _col(r, "DCL_")) / 2.0


def _keltner_mid(df):
    return _col(az._keltner(df), "KCBe_")


def _ichimoku_kijun(df):
    h, l = df["high"], df["low"]  # noqa: E741
    return (h.rolling(26).max() + l.rolling(26).min()) / 2.0


def _chandelier_long(df):
    """Chandelier Exit, long line: highest high(22) - 3 x ATR(22)."""
    atr = ta.atr(df["high"], df["low"], df["close"], length=22)
    return df["high"].rolling(22).max() - 3.0 * atr


def _psar_level(df):
    r = ta.psar(df["high"], df["low"], df["close"])
    return _col(r, "PSARl_").fillna(_col(r, "PSARs_"))


def _supertrend_line(df):
    return _col(az._supertrend(df), "SUPERT_")


def _macd_parts(kind):
    def b(df):
        r = ta.macd(df["close"], fast=12, slow=26, signal=9)
        return _col(r, {"line": "MACD_", "hist": "MACDh_",
                        "signal": "MACDs_"}[kind])
    return b


def _klinger_kvo(df):
    """Klinger Volume Oscillator (EMA34 - EMA55 of volume force) --
    the az _klinger construction minus its signal-line subtraction,
    with the same explicit 68-bar warmup."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]  # noqa: E741
    sv = (h + l + c).to_numpy(dtype=float)
    dm = (h - l).to_numpy(dtype=float)
    vol = v.to_numpy(dtype=float)
    trend = np.where(np.diff(sv, prepend=sv[0]) >= 0, 1.0, -1.0)
    cm = np.zeros(len(df))
    vf = np.zeros(len(df))
    for t in range(1, len(df)):
        cm[t] = cm[t - 1] + dm[t] if trend[t] == trend[t - 1] \
            else dm[t - 1] + dm[t]
        if cm[t] > 0:
            vf[t] = vol[t] * abs(2.0 * dm[t] / cm[t] - 1.0) * trend[t]
    vf_s = pd.Series(vf, index=df.index)
    kvo = az._ema(vf_s, 34) - az._ema(vf_s, 55)
    kvo.iloc[:68] = np.nan
    return kvo


def _force_index_raw(df):
    """Elder Force Index, raw 1-period form: volume x close change.
    (The template's separate EFI column carries the EMA13-smoothed
    form via the az library's ta.efi(13) builder.)"""
    return df["volume"] * df["close"].diff(1)


def _vwap_24h(df):
    """Project VWAP convention (production dist_vwap, 2026-10-20):
    rolling 24h sum(close x volume) / sum(volume). Fetch-invariant,
    unlike a cumulative anchor."""
    pv = df["close"] * df["volume"]
    return pv.rolling(24).sum() / df["volume"].rolling(24).sum() \
        .replace(0.0, np.nan)


def _ma_level(kind, length):
    def b(df):
        return getattr(ta, kind)(df["close"], length=length)
    return b


# ------------------------------------------------------------------
# the 59 included indicator columns, in Kass's template order
# (name, builder, definition for the report)
# ------------------------------------------------------------------

INDICATORS = [
    ("Absolute Price Oscillator (APO)", _apo_raw,
     "EMA(12) - EMA(26) of close, price units (raw form of az "
     "az_apo_rel, which divides by close)"),
    ("Acceleration Oscillator (AC)", az._ac,
     "Awesome Oscillator(5,34) - SMA(5) of AO (az _ac)"),
    ("Accumulation/Distribution (A/D)", _ad_line,
     "Chaikin Accumulation/Distribution line, cumulative, "
     "pandas_ta ad"),
    ("Accumulative Swing Index (ASI)", _asi,
     "Wilder ASI: cumulative Swing Index; limit move T set to the "
     "bar's own K (BTC has no exchange limit move), hand"),
    ("Arnaud Legoux Moving Average (ALMA)", _ma_level("alma", 20),
     "ALMA(20) of close, price level, pandas_ta (az MA family "
     "length 20)"),
    ("Aroon Indicator", az._aroon_up,
     "Aroon Up(14), 0-100, pandas_ta (the az library also carries "
     "Aroon Down separately; Kass's demo values match the Up line)"),
    ("Average Daily Range (ADR)", _adr_raw,
     "Mean daily (high - low) over the prior 14 completed days, "
     "price units, stamped on the following day (raw form of az "
     "_adr_rel)"),
    ("Average Directional Movement Index (ADX)", az._adx,
     "ADX(14), 0-100, pandas_ta"),
    ("Awesome Oscillator (AO)", az._ao,
     "AO(5,34) of median price, pandas_ta"),
    ("Bollinger BandWidth", az._bb_bw,
     "Bollinger BandWidth(20, 2.0), pandas_ta BBB (percent form)"),
    ("Bollinger %B", az._bb_pctb,
     "Bollinger %B(20, 2.0), pandas_ta"),
    ("Center of Gravity (COG)", az._cg10,
     "Ehlers Center of Gravity(10), pandas_ta cg"),
    ("Chaikin Money Flow (CMF)", az._cmf20,
     "Chaikin Money Flow(20), pandas_ta"),
    ("Chande Momentum Oscillator (CMO)", az._cmo14,
     "CMO(14), -100..100, pandas_ta"),
    ("Chandelier Exit", _chandelier_long,
     "Chandelier Exit long line: highest high(22) - 3 x ATR(22), "
     "hand (standard 22 / 3.0 parameters)"),
    ("Choppiness Index (CHOP)",
     lambda df: az.REGISTRY["az_chop_14"]["build"](df),
     "Choppiness Index(14), 0-100, pandas_ta chop (az registry "
     "builder)"),
    ("Commodity Channel Index (CCI)", az._cci20,
     "CCI(20), pandas_ta"),
    ("Connors RSI (CRSI)", az._crsi,
     "Connors RSI(3, 100) as implemented by pandas_ta crsi "
     "(az _crsi), 0-100"),
    ("Coppock Curve", az._coppock,
     "Coppock Curve(11,14,10), pandas_ta"),
    ("Detrended Price Oscillator (DPO)", az._dpo20,
     "DPO(20), causal form: close[t-11] - SMA20[t-11], price units "
     "(az _dpo20; never the centered variant)"),
    ("Directional Movement Index (DMI)", _dmi_dx,
     "DX(14) = 100 x |+DI - -DI| / (+DI + -DI), from pandas_ta "
     "ADX(14) components"),
    ("Donchian Channels", _donchian_mid,
     "Donchian(20) midpoint = (upper + lower) / 2, price level, "
     "pandas_ta (single representative line per build spec)"),
    ("Double Exponential Moving Average (DEMA)",
     _ma_level("dema", 20),
     "DEMA(20) of close, price level, pandas_ta"),
    ("Ease of Movement (EMV)", az._eom14,
     "Ease of Movement(14), pandas_ta eom"),
    ("Elder Force Index (EFI)", az._force_index,
     "Elder Force Index, EMA(13)-smoothed, pandas_ta efi "
     "(az _force_index)"),
    ("Exponential Moving Average (EMA)", _ma_level("ema", 20),
     "EMA(20) of close, price level, pandas_ta"),
    ("Fisher Transform", az._fisher9,
     "Fisher Transform(9) signal line (FISHERT), pandas_ta"),
    ("Force Index", _force_index_raw,
     "Force Index, raw 1-period Elder form: volume x (close - "
     "close[1]) (the unsmoothed counterpart of EFI)"),
    ("Historical Volatility (HV)", az._hist_vol20,
     "Rolling std(20) of log returns x 100 (per-bar percent, az "
     "_hist_vol20 convention; not annualized)"),
    ("Hull Moving Average (HMA)", _ma_level("hma", 20),
     "HMA(20) of close, price level, pandas_ta"),
    ("Ichimoku Cloud (Baseline)", _ichimoku_kijun,
     "Kijun-sen(26) = (highest high(26) + lowest low(26)) / 2, "
     "price level (the line inside az _ichimoku, un-divided)"),
    ("Kaufman Adaptive Moving Average (KAMA)",
     _ma_level("kama", 10),
     "KAMA(10) of close, price level, pandas_ta (az MA family "
     "length 10)"),
    ("Keltner Channels", _keltner_mid,
     "Keltner(20, 2.0) middle line (EMA20 basis, KCBe), price "
     "level, pandas_ta kc (single representative line)"),
    ("Klinger Volume Oscillator (KVO)", _klinger_kvo,
     "Klinger Oscillator = EMA(34) - EMA(55) of Klinger's volume "
     "force (az _klinger construction, before its signal-line "
     "subtraction); 68-bar explicit warmup"),
    ("Know Sure Thing (KST)", az._kst,
     "KST line, pandas_ta default ROC/smoothing stack (az _kst)"),
    ("MACD (Moving Average Convergence Divergence)",
     _macd_parts("line"),
     "MACD line (12,26,9), pandas_ta"),
    ("MACD Histogram", _macd_parts("hist"),
     "MACD histogram (12,26,9), pandas_ta"),
    ("MACD Signal Line", _macd_parts("signal"),
     "MACD signal line (12,26,9), pandas_ta"),
    ("Mass Index (MI)", az._mass_index,
     "Mass Index(9,9,25): EMA9/EMA9-of-EMA9 of range, summed "
     "over 25 (az _mass_index)"),
    ("Momentum (MOM)", lambda df: df["close"] - df["close"].shift(10),
     "Momentum(10) = close - close[10], price units, classic form"),
    ("Money Flow Index (MFI)", az._mfi14,
     "MFI(14), 0-100, pandas_ta"),
    ("On-Balance Volume (OBV)",
     lambda df: ta.obv(df["close"], df["volume"]),
     "On-Balance Volume, cumulative, pandas_ta obv"),
    ("Parabolic SAR (PSAR)", _psar_level,
     "Parabolic SAR price level (long/short lines merged), "
     "pandas_ta psar default AF step 0.02 / max 0.2"),
    ("Price Volume Trend (PVT)",
     lambda df: ta.pvt(df["close"], df["volume"]),
     "Price Volume Trend, cumulative, pandas_ta pvt"),
    ("Rate of Change (ROC)", az._roc(12),
     "ROC(12) percent, pandas_ta (az _roc(12))"),
    ("Relative Strength Index (RSI)", az._rsi14,
     "RSI(14), 0-100, pandas_ta"),
    ("Relative Volatility Index (RVI)", az._rel_vol_index,
     "Beyers Relative Volatility Index(14), rolling-std form, "
     "0-100 (az _rel_vol_index)"),
    ("Schaff Trend Cycle (STC)", az._stc,
     "Schaff Trend Cycle, pandas_ta stc defaults, 0-100"),
    ("Simple Moving Average (SMA)", _ma_level("sma", 20),
     "SMA(20) of close, price level, pandas_ta"),
    ("Standard Deviation (SD)",
     lambda df: df["close"].rolling(20).std(),
     "Rolling sample standard deviation of close(20), price "
     "units"),
    ("Stochastic Oscillator (Fast)", az._stoch_k,
     "Stochastic %K(14,3,3), 0-100, pandas_ta (az _stoch_k "
     "parameters)"),
    ("Stochastic RSI (StochRSI)", az._stochrsi_k,
     "StochRSI %K(14,14,3,3), pandas_ta raw output"),
    ("SuperTrend", _supertrend_line,
     "SuperTrend(10, 3.0) line, price level, pandas_ta (az "
     "_supertrend)"),
    ("True Strength Index (TSI)", az._tsi,
     "TSI(25,13) double-smoothed momentum x 100, -100..100 "
     "(az _tsi)"),
    ("Ultimate Oscillator (UO)", az._uo,
     "Ultimate Oscillator(7,14,28), 0-100, pandas_ta"),
    ("Volume Weighted Average Price (VWAP)", _vwap_24h,
     "Rolling 24h VWAP = sum(close x volume, 24) / sum(volume, "
     "24) -- the production dist_vwap convention (fetch-invariant)"),
    ("Vortex Indicator (VI)", az._vortex_diff,
     "Vortex(14) difference VI+ - VI- (az _vortex_diff; single "
     "representative of the two-line indicator)"),
    ("Williams %R", az._willr14,
     "Williams %R(14), -100..0, pandas_ta"),
    ("Zero Lag Exponential Moving Average (ZLEMA)",
     _ma_level("zlma", 20),
     "ZLEMA(20) of close, price level, pandas_ta"),
]

SAMPLE_CHECK_COLUMNS = [
    "Exponential Moving Average (EMA)",       # price-level MA
    "Relative Strength Index (RSI)",          # oscillator
    "Fisher Transform",                       # oscillator
    "On-Balance Volume (OBV)",                # volume, cumulative
    "Accumulation/Distribution (A/D)",        # volume, cumulative
    "Chaikin Money Flow (CMF)",               # volume, bounded
    "Historical Volatility (HV)",             # volatility
    "Keltner Channels",                       # price level / channel
    "Volume Weighted Average Price (VWAP)",   # price/volume level
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All 59 raw indicator series for an OHLCV frame."""
    out = {}
    for name, builder, _defn in INDICATORS:
        t0 = time.time()
        s = builder(df)
        s = pd.Series(np.asarray(s, dtype=float), index=df.index)
        out[name] = s.replace([np.inf, -np.inf], np.nan)
        print(f"  {name}: {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(out, index=df.index)


def compute_labels(df: pd.DataFrame) -> pd.Series:
    r = df["close"].shift(-1) / df["close"] - 1.0
    lab = pd.Series(np.where(r >= FEE_BAND, "BUY",
                             np.where(r <= -FEE_BAND, "SHORT", "HOLD")),
                    index=df.index, dtype=object)
    lab.iloc[-1] = np.nan  # no next candle
    return lab


def truncation_self_check(df: pd.DataFrame, full_feat: pd.DataFrame
                          ) -> dict:
    print(f"self-check: rebuilding features on first {TRUNC_N} rows "
          "...", flush=True)
    trunc_feat = build_features(df.iloc[:TRUNC_N])
    results = {}
    ok_all = True
    for colname in SAMPLE_CHECK_COLUMNS:
        a = full_feat[colname].iloc[:TRUNC_N].to_numpy()
        b = trunc_feat[colname].to_numpy()
        ok = bool(np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True))
        results[colname] = ok
        ok_all = ok_all and ok
        print(f"  check {colname}: {'PASS' if ok else 'FAIL'}",
              flush=True)
    return {"pass": ok_all, "n_rows_compared": TRUNC_N,
            "columns": results}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sup = pd.read_pickle(SUP_PATH)
    df = sup[["open", "high", "low", "close", "volume"]].copy()
    raw_rows = len(df)
    print(f"source: {SUP_PATH.name} OHLCV rows={raw_rows} "
          f"{df.index[0]} -> {df.index[-1]}", flush=True)

    print("building features (full frame) ...", flush=True)
    feat = build_features(df)

    check = truncation_self_check(df, feat)
    print(f"SELF-CHECK: {'PASS' if check['pass'] else 'FAIL'}",
          flush=True)
    if not check["pass"]:
        print("aborting: causality self-check failed; no files "
              "written", flush=True)
        return 1

    labels = compute_labels(df)
    first_valid = {name: int(feat[name].first_valid_index()
                             and df.index.get_loc(feat[name]
                                                  .first_valid_index()))
                   if feat[name].first_valid_index() is not None
                   else None for name in feat.columns}
    longest = sorted(((v, k) for k, v in first_valid.items()
                      if v is not None), reverse=True)[:5]

    matrix = pd.DataFrame(index=df.index)
    matrix["Date"] = df.index.strftime("%Y-%m-%dT%H:%M:%S")
    matrix["Ticker"] = "BTC/USD"
    matrix[TARGET_COL] = labels
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        matrix[c] = df[c.lower()].to_numpy()
    for name, _b, _d in INDICATORS:
        matrix[name] = feat[name].to_numpy()

    before = len(matrix)
    matrix = matrix.iloc[:-1]  # final row: no next candle, no label
    dropped_last = before - len(matrix)
    before_warm = len(matrix)
    matrix = matrix.dropna(subset=[n for n, _b, _d in INDICATORS])
    warmup_dropped = before_warm - len(matrix)

    counts = matrix[TARGET_COL].value_counts()
    dist = {k: {"count": int(counts.get(k, 0)),
                "pct": round(100.0 * counts.get(k, 0) / len(matrix), 2)}
            for k in ["BUY", "HOLD", "SHORT"]}

    matrix.to_csv(CSV_PATH, index=False)
    stats = {
        "source": "data/diagnosis/sup.pkl (Coinbase BTC-USD 1h, "
                  "same frame as diagnosis + A-Z screening)",
        "raw_rows": raw_rows,
        "dropped_final_row_no_next_candle": dropped_last,
        "warmup_rows_dropped": warmup_dropped,
        "final_rows": int(len(matrix)),
        "final_columns": int(len(matrix.columns)),
        "date_first": str(matrix["Date"].iloc[0]),
        "date_last": str(matrix["Date"].iloc[-1]),
        "class_distribution": dist,
        "self_check": check,
        "excluded_columns": EXCLUDED,
        "longest_warmups": [{"column": k, "first_valid_row": v}
                            for v, k in longest],
        "column_definitions": {name: defn for name, _b, defn
                               in INDICATORS},
        "csv_path": str(CSV_PATH),
        "csv_bytes": CSV_PATH.stat().st_size,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in
                      ["raw_rows", "warmup_rows_dropped", "final_rows",
                       "final_columns", "date_first", "date_last",
                       "class_distribution", "csv_bytes"]}, indent=1),
          flush=True)
    print(f"wrote {CSV_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
