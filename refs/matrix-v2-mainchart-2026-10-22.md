# Time-Series Matrix v2 — Main-Chart Indicators Extension

Date: 2026-10-22 · Part B of the research build Kass green-lit
2026-10-22 ("find and include all the main chart indicators we can
also include into our data table that are not in there right now").
Code: `src/features/main_chart.py` (tests:
`tests/test_main_chart.py`) · Build:
`scripts/build_timeseries_matrix_v2.py` · Output:
`data/timeseries_matrix/btc_usd_1h_matrix_v2.csv` +
`data/timeseries_matrix/build_stats_v2.json`.

**No model was trained on v2. Trading remains PARKED.**

## What v2 is

- **16,735 rows × 98 columns** (28,536,949 bytes): v1's exact rows,
  labels, and 67 columns **in order, values unchanged**, with **31
  main-chart columns appended**.
- v1 columns were copied from the v1 CSV, never recomputed. The
  written file was read back and checked column by column: string
  columns byte-identical; float columns identical to within
  3.8e-06 — pure CSV text round-trip at the 12th significant digit
  (worst case: Ease of Movement, whose values are ~1e6 in scale).
  Labels verified identical on all 16,735 rows.
- New columns were computed on the **long hourly Coinbase frame**
  (42,107 rows, 2022-01-01 → 2026-10-21, pulled in Part A) so every
  warmup — EMA/SMA 200, Ichimoku spans, fractal lookback,
  prior-day stamps — is covered by real history long before the
  first matrix row (2024-11-23). Result: **zero NaN in all 31 new
  columns across all 16,735 rows**, verified after alignment (0
  missing timestamps).

## Candidate inventory — what "main-chart" means here

Sources: the project's A-Z encyclopedia
(`refs/az-indicator-encyclopedia.md`, esp. Trend + Advanced
Charting & Breadth + Geometry & Levels), the standard charting
canon, and a column-by-column diff against v1's 67. A candidate
was added only if it is (a) **not already present in substance**,
(b) **causally computable from OHLCV alone**, (c) **mechanically
definable without discretion**. v1 already carried many overlays
as single representative lines (Donchian/Keltner *midpoints*,
Kijun level, PSAR level, SuperTrend level, Chandelier long line)
and sub-chart forms (Bollinger %B / BandWidth) — the genuinely
missing pieces were the bands themselves, the sides, the full
Ichimoku cloud, the longer MA grid, and structural levels.

## Conventions (uniform across the 31 columns)

- Every **"Distance (%)"** column is the signed percent distance
  of the close from its level: **(Close − Level) / Close × 100**.
  Positive = price above the level. One number carries side and
  distance.
- **"Side (+1/−1)"** columns are discrete regime flags.
- Daily-stamped columns (pivots, prior-day high/low) use the
  **prior completed UTC day only**, stamped on the following day —
  the v1 ADR discipline.
- **Ichimoku spans, causal convention:** a standard chart plots
  Span A/B 26 bars ahead. The cloud level *visible at row t* was
  therefore fixed at t−26 — Span A at t = (Tenkan+Kijun)/2
  computed at t−26; Span B at t = (HH52+LL52)/2 computed at t−26.
  Strictly causal; nothing from the future enters.
- Fractal swings (Williams 2-bar) become usable **only at their
  confirmation bar** (swing bar + 2), never at the swing bar.

## Added columns (31)

Warmup = bars before the first value on the computation frame;
all are covered by the 2022 lead-in, so matrix rows are complete.

| # | Column | Definition (one line) | Warmup |
|---|---|---|---|
| 1 | Distance to SMA 50 (%) | (Close − SMA(50))/Close × 100 | 49 |
| 2 | Distance to SMA 100 (%) | (Close − SMA(100))/Close × 100 | 99 |
| 3 | Distance to SMA 200 (%) | (Close − SMA(200))/Close × 100 | 199 |
| 4 | Distance to EMA 21 (%) | (Close − EMA(21))/Close × 100, n-bar EMA warmup | 20 |
| 5 | Distance to EMA 55 (%) | (Close − EMA(55))/Close × 100 | 54 |
| 6 | Distance to EMA 200 (%) | (Close − EMA(200))/Close × 100 | 199 |
| 7 | Distance to Bollinger Upper Band (20, 2.0) (%) | (Close − (SMA20 + 2·SD20))/Close × 100 — negative inside the channel | 19 |
| 8 | Distance to Bollinger Lower Band (20, 2.0) (%) | (Close − (SMA20 − 2·SD20))/Close × 100 — positive inside the channel | 19 |
| 9 | Distance to Donchian Upper (20) (%) | (Close − HH(20))/Close × 100 — 0 exactly at a 20-bar high | 19 |
| 10 | Distance to Donchian Lower (20) (%) | (Close − LL(20))/Close × 100 — 0 exactly at a 20-bar low | 19 |
| 11 | Distance to Donchian Upper (55) (%) | Turtle System-2 channel top | 54 |
| 12 | Distance to Donchian Lower (55) (%) | Turtle System-2 channel bottom | 54 |
| 13 | Distance to Keltner Upper (20, 2.0) (%) | (Close − Keltner upper)/Close × 100 (v1 had the midline only) | ~33 |
| 14 | Distance to Keltner Lower (20, 2.0) (%) | (Close − Keltner lower)/Close × 100 | ~33 |
| 15 | ATR Band Half-Width (14) (%) | ATR(14)/Close × 100 — distance to the ±1×ATR band edges (equal both sides). ATR itself was **not** a v1 column | 13 |
| 16 | Ichimoku Tenkan-sen Distance (%) | (Close − (HH9+LL9)/2)/Close × 100 | 8 |
| 17 | Ichimoku Senkou Span A Distance (%) | vs Span A as plotted at t (=(Tenkan+Kijun)/2 at t−26) | 51 |
| 18 | Ichimoku Senkou Span B Distance (%) | vs Span B as plotted at t (=(HH52+LL52)/2 at t−26) | 77 |
| 19 | Ichimoku Chikou Distance (%) | (Close[t] − Close[t−26])/Close × 100 — the displacement the Chikou span encodes | 26 |
| 20 | Parabolic SAR Distance (%) | (Close − SAR)/Close × 100, same pandas_ta series whose level is a v1 column | 1 |
| 21 | Parabolic SAR Side (+1/−1) | +1 while the SAR long series is active (below price), −1 while short series active | 1 |
| 22 | SuperTrend Distance (%) | (Close − SuperTrend(10, 3.0))/Close × 100, level already in v1 | 9 |
| 23 | SuperTrend Side (+1/−1) | pandas_ta SUPERTd: +1 uptrend, −1 downtrend | 9 |
| 24 | Distance to Prior-Day Pivot Point (%) | P = (prior UTC day H+L+C)/3, stamped next day | 1 day |
| 25 | Distance to Prior-Day Pivot R1 (%) | R1 = 2P − prior-day low | 1 day |
| 26 | Distance to Prior-Day Pivot S1 (%) | S1 = 2P − prior-day high | 1 day |
| 27 | Distance to Prior-Day High (%) | vs prior completed UTC day's high | 1 day |
| 28 | Distance to Prior-Day Low (%) | vs prior completed UTC day's low | 1 day |
| 29 | Distance to Fractal Resistance (%) | vs nearest **confirmed** fractal swing high ≥ Close (swings from last 200 bars; most-recent fallback) | ~5 |
| 30 | Distance to Fractal Support (%) | vs nearest confirmed fractal swing low ≤ Close (same discipline) | ~5 |
| 31 | Distance to $5,000 Round Number (%) | (Close − nearest multiple of $5,000)/Close × 100 | 0 |

## Considered but rejected (with reasons)

| Candidate | Reason |
|---|---|
| Rolling swing HH/LL (20 / 50) as separate columns | **Duplicate-in-substance**: a rolling highest-high/lowest-low channel *is* the Donchian channel — added as Donchian (20) and (55) distances. The distinct structural form (confirmed swings) is the fractal pair. |
| High-Low Bands, Price Channel | Same HH/LL construction as Donchian under other names — duplicate-in-substance. |
| Darvas Box | Its levels are rolling swing high/low boxes — covered by Donchian distances + fractal S/R. |
| STARC Bands | MA ± k·ATR family; differs from the added Keltner bands only in center/smoothing choice — same construction family. |
| Envelopes / Moving-Average Envelope | MA ± fixed % — an exact transform of the MA distance columns. |
| Ichimoku Kijun-sen distance | Kijun **level** is a v1 column; its distance is an exact transform of two v1 columns (level + Close). Not added, same rule as SMA(20)/EMA(20) distances (levels in v1). |
| WMA, SMMA, TEMA, VWMA, McGinley Dynamic, JMA, VIDYA | Moving-average family variants. The canonical grid (SMA/EMA distances + v1's HMA/DEMA/KAMA/ALMA/ZLEMA levels) already represents the family; each additional smoother is the same signal re-weighted. (VWMA's volume-weighted *level* niche is already carried by v1's rolling VWAP.) |
| Guppy GMMA / Rainbow ribbons | Bundles of MAs — the MA-distance family is already covered. |
| Camarilla / Woodie / Demark pivot points | Coefficient variants of the classic pivot family; the classic (Standard) pivots added here are the canonical representative. |
| Chandelier Exit distance | Level already in v1; its family (ATR trailing-stop overlays) is represented by the added SuperTrend distance + side. |
| Anchored VWAP / VWAP bands | Anchor choice is discretionary (which swing/event to anchor to) — fails the mechanical-definition test. Rolling VWAP is in v1. |
| ZigZag | **Non-causal by design**: a swing is only known retrospectively and the line repaints. The fractal columns capture the same swings under a causal confirmation rule. |
| Fibonacci retracements / extensions / fans, Gann angles / fans / square, Andrews' Pitchfork, Elliott Wave counts | Drawing tools requiring discretionary anchor/swing selection — no unique mechanical definition from OHLCV. |
| Linear Regression Channel, Projection Bands, Standard Error Bands | Regression-band overlays with discretionary band-width conventions; the mid-line is a smoothed-price variant (MA-distance family in substance). |
| Heikin-Ashi / Renko / Kagi / Point & Figure | Price *transformations*, not indicators — they replace the candle set rather than describe it. HA is analyzed in the streak study (Part A) and already lives in the production feature set as ha_dir. |
| Volume Profile (POC / value area) | Needs volume-at-price (intrabar distribution); not exactly computable from OHLCV alone. |
| Market breadth (A/D Line, McClellan, TRIN), VIX | External data — the same three documented v1 exclusions (breadth needs a whole exchange; VIX is an external index). |

## Verification

- **Tests first** (`tests/test_main_chart.py`, 13 tests):
  hand-computed fixtures for pivots (2-day crafted frame,
  P/R1/S1/prior-day H/L to the cent), Donchian and Bollinger
  (independent recomputation on fixture arrays), SMA/EMA distance
  arithmetic + EMA warmup discipline, Ichimoku warmup pattern
  (Tenkan first valid at bar 8, Span A at 51, Span B at 77) +
  flat-price collapse + Chikou on an arithmetic series, a crafted
  **SAR direction-flip** series (+1 through a 30-bar uptrend, one
  flip, −1 through the downtrend; sign agreement with the distance
  column), SuperTrend sides, fractal confirmation timing (level
  unusable before swing bar + 2), round-number distances above and
  below the grid — and the **causality truncation self-check on
  real data** (2,600-row sup frame vs its first 1,800 rows, 10
  sample columns, exact match). All pass.
- **Build-level self-check (PASS):** features rebuilt on the first
  30,000 rows of the long frame match the full-frame build exactly
  (atol 1e-9, NaN-aware) on 8 sampled new columns spanning every
  family (SMA 200, Donchian 55, Span A, SAR, SuperTrend side,
  pivots, fractal resistance, round numbers) — 30,000 shared
  timestamps.
- **Range sanity on the finished v2** (independent read-back):
  Donchian Upper distance ∈ [−15.23, **0.00**] and Lower ∈
  [**0.00**, +15.16] on all 16,735 rows — the exact sign pattern
  the definitions require (price can touch but never cross its own
  trailing channel). Round-number distance ∈ [−3.99, +4.00]% —
  inside the theoretical bound 2500/min(Close) ≈ 4.03%. ATR
  half-width 0.11–2.73% (plausible hourly BTC). Sides exactly ±1.
  All 31 new columns fully finite.
- **Full project suite after both parts: 84 passed** (59 at the
  start of this build + 13 new main-chart tests + 12 gate-floor
  tests landed by parallel work). Nothing in production code,
  paper.db, champion files, or the live loop was touched.

## Files

| File | Notes |
|---|---|
| `data/timeseries_matrix/btc_usd_1h_matrix_v2.csv` | 16,735 rows × 98 cols, 28.5 MB — v1's 67 columns + the 31 above |
| `data/timeseries_matrix/build_stats_v2.json` | Machine-readable stats: definitions, self-check, NaN counts, round-trip diff |
| `src/features/main_chart.py` | The 31 builders + registry (name, builder, definition) |
| `tests/test_main_chart.py` | 13 tests incl. fixtures + truncation causality |
| `scripts/build_timeseries_matrix_v2.py` | Deterministic rebuild (~15 s), self-check on every run |
