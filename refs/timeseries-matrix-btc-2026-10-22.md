# Time-Series ML Training Data Matrix — Real BTC/USD Build

Date: 2026-10-22 · Build script: `scripts/build_timeseries_matrix.py` ·
Dataset: `data/timeseries_matrix/btc_usd_1h_matrix.csv` / `.xlsx`

Kass supplied the format as a draft template ("Time-Series ML Training
Data Matrix.xlsx", Drive, 2026-10-22): Date | Ticker | Target_Action |
OHLCV | 62 indicator columns, one row per observation, 5 illustrative
stock rows. This build fills that exact shape with the project's real
BTC/USD data — same columns, same order, real values throughout.

## Data source and row accounting

- Source: `data/diagnosis/sup.pkl` — the cached Coinbase BTC/USD 1h
  supervised frame used by the diagnosis battery and the A-Z
  screening (OHLCV columns only are used here). No synthetic values
  anywhere; every number below is computed from those candles.
- Raw rows: **17,049** (2024-11-09 23:00 UTC → 2026-10-21 17:00 UTC)
- Final row dropped: **1** (2026-10-21 17:00 — no next candle, so no
  label can be computed for it)
- Warmup rows dropped: **313** (rows where any included indicator is
  still NaN). Longest warmups: ADR 313 rows (it needs 14 completed
  days before its first stamp), Connors RSI 100, Klinger KVO 68
  (explicit warmup, mirroring the az library), KST 44, AC 37.
- **Final dataset: 16,735 rows × 67 columns**,
  **2024-11-23T00:00:00 UTC → 2026-10-21T16:00:00 UTC**, Ticker =
  `BTC/USD` on every row, timestamps hourly and monotonic.
- Inherited source gaps (flagged, not hidden): the Coinbase frame
  itself is missing 5 consecutive candles twice — 2025-10-25
  16:00–20:00 UTC and 2026-05-08 02:00–06:00 UTC — so the matrix has
  exactly two 6-hour timestamp jumps. Consequence: the label of the
  row immediately before each gap is computed against the next
  *available* candle (6h later), and rolling windows spanning a gap
  treat the available candles as consecutive. 2 rows' labels are
  affected; everything else is a clean 1h series.

## Values

RAW indicator values in natural units (RSI on 0–100, moving averages
at price level, OBV/PVT cumulative), **not** the [−1, 1] screening
scalings. Definitions and parameters reuse `src/features/
az_indicators.py` builders wherever a series exists there; where the
az library carries only a distance/momentum form (e.g.
(close − MA)/close), the raw level was recomputed from the same
pandas_ta call with the same parameters. Single-line conventions for
multi-line indicators: Donchian Channels → midpoint
(upper + lower)/2; Keltner Channels → middle line (EMA20 basis);
Chandelier Exit → long-exit line; Ichimoku Cloud (Baseline) →
Kijun-sen(26); Aroon Indicator → Aroon Up(14) (the az library also
screens Aroon Down separately; Kass's template demo values match the
Up line); Vortex Indicator → VI+ − VI−; Directional Movement Index
→ DX(14).

## Excluded template columns (3 of 62 — accounted for, never fabricated)

| Column | Why excluded |
|---|---|
| Advance/Decline Line (A/D Line) | Market breadth: needs advance/decline counts across a whole exchange. A single asset has no breadth. |
| CBOE Volatility Index (VIX) | An external CBOE index series; not derivable from BTC candles. |
| Beta | Needs a benchmark asset series to regress against; none is part of this dataset. |

Output is therefore 67 columns (70 − 3).

## Columns

| # | Column | Definition / parameters / source |
|---|---|---|
| 1 | Date | Candle open time, ISO 8601 hourly, UTC (source frame index) |
| 2 | Ticker | Constant `BTC/USD` |
| 3 | Target_Action (BUY/SELL/SHORT/HOLD) | Computed label — see Label section below |
| 4 | Open / High / Low / Close / Volume | Coinbase BTC/USD 1h candles, as cached in the source frame |
| 5 | Absolute Price Oscillator (APO) | EMA(12) - EMA(26) of close, price units (raw form of az az_apo_rel, which divides by close) |
| 6 | Acceleration Oscillator (AC) | Awesome Oscillator(5,34) - SMA(5) of AO (az _ac) |
| 7 | Accumulation/Distribution (A/D) | Chaikin Accumulation/Distribution line, cumulative, pandas_ta ad |
| 8 | Accumulative Swing Index (ASI) | Wilder ASI: cumulative Swing Index; limit move T set to the bar's own K (BTC has no exchange limit move), hand |
| 9 | Arnaud Legoux Moving Average (ALMA) | ALMA(20) of close, price level, pandas_ta (az MA family length 20) |
| 10 | Aroon Indicator | Aroon Up(14), 0-100, pandas_ta (the az library also carries Aroon Down separately; Kass's demo values match the Up line) |
| 11 | Average Daily Range (ADR) | Mean daily (high - low) over the prior 14 completed days, price units, stamped on the following day (raw form of az _adr_rel) |
| 12 | Average Directional Movement Index (ADX) | ADX(14), 0-100, pandas_ta |
| 13 | Awesome Oscillator (AO) | AO(5,34) of median price, pandas_ta |
| 14 | Bollinger BandWidth | Bollinger BandWidth(20, 2.0), pandas_ta BBB (percent form) |
| 15 | Bollinger %B | Bollinger %B(20, 2.0), pandas_ta |
| 16 | Center of Gravity (COG) | Ehlers Center of Gravity(10), pandas_ta cg |
| 17 | Chaikin Money Flow (CMF) | Chaikin Money Flow(20), pandas_ta |
| 18 | Chande Momentum Oscillator (CMO) | CMO(14), -100..100, pandas_ta |
| 19 | Chandelier Exit | Chandelier Exit long line: highest high(22) - 3 x ATR(22), hand (standard 22 / 3.0 parameters) |
| 20 | Choppiness Index (CHOP) | Choppiness Index(14), 0-100, pandas_ta chop (az registry builder) |
| 21 | Commodity Channel Index (CCI) | CCI(20), pandas_ta |
| 22 | Connors RSI (CRSI) | Connors RSI(3, 100) as implemented by pandas_ta crsi (az _crsi), 0-100 |
| 23 | Coppock Curve | Coppock Curve(11,14,10), pandas_ta |
| 24 | Detrended Price Oscillator (DPO) | DPO(20), causal form: close[t-11] - SMA20[t-11], price units (az _dpo20; never the centered variant) |
| 25 | Directional Movement Index (DMI) | DX(14) = 100 x |+DI - -DI| / (+DI + -DI), from pandas_ta ADX(14) components |
| 26 | Donchian Channels | Donchian(20) midpoint = (upper + lower) / 2, price level, pandas_ta (single representative line per build spec) |
| 27 | Double Exponential Moving Average (DEMA) | DEMA(20) of close, price level, pandas_ta |
| 28 | Ease of Movement (EMV) | Ease of Movement(14), pandas_ta eom |
| 29 | Elder Force Index (EFI) | Elder Force Index, EMA(13)-smoothed, pandas_ta efi (az _force_index) |
| 30 | Exponential Moving Average (EMA) | EMA(20) of close, price level, pandas_ta |
| 31 | Fisher Transform | Fisher Transform(9) signal line (FISHERT), pandas_ta |
| 32 | Force Index | Force Index, raw 1-period Elder form: volume x (close - close[1]) (the unsmoothed counterpart of EFI) |
| 33 | Historical Volatility (HV) | Rolling std(20) of log returns x 100 (per-bar percent, az _hist_vol20 convention; not annualized) |
| 34 | Hull Moving Average (HMA) | HMA(20) of close, price level, pandas_ta |
| 35 | Ichimoku Cloud (Baseline) | Kijun-sen(26) = (highest high(26) + lowest low(26)) / 2, price level (the line inside az _ichimoku, un-divided) |
| 36 | Kaufman Adaptive Moving Average (KAMA) | KAMA(10) of close, price level, pandas_ta (az MA family length 10) |
| 37 | Keltner Channels | Keltner(20, 2.0) middle line (EMA20 basis, KCBe), price level, pandas_ta kc (single representative line) |
| 38 | Klinger Volume Oscillator (KVO) | Klinger Oscillator = EMA(34) - EMA(55) of Klinger's volume force (az _klinger construction, before its signal-line subtraction); 68-bar explicit warmup |
| 39 | Know Sure Thing (KST) | KST line, pandas_ta default ROC/smoothing stack (az _kst) |
| 40 | MACD (Moving Average Convergence Divergence) | MACD line (12,26,9), pandas_ta |
| 41 | MACD Histogram | MACD histogram (12,26,9), pandas_ta |
| 42 | MACD Signal Line | MACD signal line (12,26,9), pandas_ta |
| 43 | Mass Index (MI) | Mass Index(9,9,25): EMA9/EMA9-of-EMA9 of range, summed over 25 (az _mass_index) |
| 44 | Momentum (MOM) | Momentum(10) = close - close[10], price units, classic form |
| 45 | Money Flow Index (MFI) | MFI(14), 0-100, pandas_ta |
| 46 | On-Balance Volume (OBV) | On-Balance Volume, cumulative, pandas_ta obv |
| 47 | Parabolic SAR (PSAR) | Parabolic SAR price level (long/short lines merged), pandas_ta psar default AF step 0.02 / max 0.2 |
| 48 | Price Volume Trend (PVT) | Price Volume Trend, cumulative, pandas_ta pvt |
| 49 | Rate of Change (ROC) | ROC(12) percent, pandas_ta (az _roc(12)) |
| 50 | Relative Strength Index (RSI) | RSI(14), 0-100, pandas_ta |
| 51 | Relative Volatility Index (RVI) | Beyers Relative Volatility Index(14), rolling-std form, 0-100 (az _rel_vol_index) |
| 52 | Schaff Trend Cycle (STC) | Schaff Trend Cycle, pandas_ta stc defaults, 0-100 |
| 53 | Simple Moving Average (SMA) | SMA(20) of close, price level, pandas_ta |
| 54 | Standard Deviation (SD) | Rolling sample standard deviation of close(20), price units |
| 55 | Stochastic Oscillator (Fast) | Stochastic %K(14,3,3), 0-100, pandas_ta (az _stoch_k parameters) |
| 56 | Stochastic RSI (StochRSI) | StochRSI %K(14,14,3,3), pandas_ta raw output |
| 57 | SuperTrend | SuperTrend(10, 3.0) line, price level, pandas_ta (az _supertrend) |
| 58 | True Strength Index (TSI) | TSI(25,13) double-smoothed momentum x 100, -100..100 (az _tsi) |
| 59 | Ultimate Oscillator (UO) | Ultimate Oscillator(7,14,28), 0-100, pandas_ta |
| 60 | Volume Weighted Average Price (VWAP) | Rolling 24h VWAP = sum(close x volume, 24) / sum(volume, 24) -- the production dist_vwap convention (fetch-invariant) |
| 61 | Vortex Indicator (VI) | Vortex(14) difference VI+ - VI- (az _vortex_diff; single representative of the two-line indicator) |
| 62 | Williams %R | Williams %R(14), -100..0, pandas_ta |
| 63 | Zero Lag Exponential Moving Average (ZLEMA) | ZLEMA(20) of close, price level, pandas_ta |
## Target label — computed, never hand-assigned

For each row t: r = close[t+1] / close[t] − 1.

- **BUY** if r ≥ +0.003
- **SHORT** if r ≤ −0.003
- **HOLD** otherwise

The ±0.3% band is the round-trip fee cost (2 × 0.15%): HOLD means
the next candle's move would not have covered the cost of trading
it. Labels were re-verified against the finished CSV row by row
(next-close rule reproduces every label in the final matrix).

**SELL is never generated by this labeler**, by design: a sell is an
exit decision that only makes sense relative to a position the
trader already holds, and a stateless dataset row has no position.
The header keeps the SELL enum because Kass's template defines
Target_Action over the live policy's four actions; the label column
teaches market direction (buy / short / not worth trading), and
turning a long position into a sale remains a policy-layer decision
(the bot's PPO + stop rules), not a data-layer one.

Class distribution (16,735 rows):

| Label | Count | % |
|---|---|---|
| BUY | 2,915 | 17.42 |
| HOLD | 10,976 | 65.59 |
| SHORT | 2,844 | 16.99 |

## Causality self-check

Built into `scripts/build_timeseries_matrix.py` and run on every
build: features are recomputed on the truncated frame (first 9,000
rows) and shared rows must match the full-frame build exactly
(tolerance 1e-9, NaN-aware) on 9 sample columns spanning families —
EMA (price-level MA), RSI + Fisher Transform (oscillators), OBV +
A/D + CMF (volume), Historical Volatility (volatility), Keltner
Channels (channel level), VWAP (price/volume level).

**Result: PASS (all 9 columns, 9,000 rows compared).**

Independent spot verification after the build (separate code path):
SMA(20) at 2025-06-15T12:00 in the CSV equals the hand-computed mean
of the 20 source closes to float precision; RSI spans 11.09–91.88,
Aroon 0–100, CMO −77.8…+83.8 — all inside their natural ranges; no
NaN remains in any indicator column.

## Files

| File | Size | Notes |
|---|---|---|
| `data/timeseries_matrix/btc_usd_1h_matrix.csv` | 19,169,120 bytes (~19.2 MB) | Full dataset, header + 16,735 rows |
| `data/timeseries_matrix/btc_usd_1h_matrix.xlsx` | 13,071,063 bytes (~13.1 MB) | Same data as Excel. The project venv's pandas has neither openpyxl nor xlsxwriter, so the workbook was produced from the final CSV with system python3 (pandas + openpyxl) and read back verified: 16,736 rows incl. header × 67 columns, cell values match the CSV. |
| `data/timeseries_matrix/build_stats.json` | 7,603 bytes | Machine-readable build stats (counts, definitions, self-check detail) |
| `scripts/build_timeseries_matrix.py` | — | Deterministic rebuild; prints per-column progress + self-check; reruns are cheap (~20s) |

Project test suite after the build: **41 passed** (nothing in
production code was touched; `az_indicators.py` was imported
read-only).

## How this plugs in

- This is an **input artifact, not a model**: nothing was trained,
  registered, or promoted on it, and trading remains PARKED.
- Features are **raw**. Any pipeline consuming this matrix must fit
  its scaler on the train split only and apply it to later rows —
  the same train-only-statistics rule the screening program enforces
  (fitting on all rows would leak the future into the scaling).
- Splits must stay chronological (walk-forward / time-series split);
  the label is next-candle by construction, so a shuffled split
  would leak outright.
- The 59 feature columns overlap heavily with the A-Z screening set
  by construction (same definitions) — the screening verdicts
  (refs/az-screening-2026-10-22.md) say which of these carry any
  solo/marginal signal on this same data; this matrix is the
  convenient all-in-one table, not a claim that all 59 help.
- Kass's stock-template columns that don't exist for a single
  crypto asset (A/D Line, VIX, Beta) are the 3 documented exclusions
  above; if a future version wants them, they must come from real
  external series joined on timestamp — never synthesized.
