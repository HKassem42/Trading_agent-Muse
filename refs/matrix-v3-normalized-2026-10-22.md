# Matrix v3 — the 0–100 Normalized Edition

Date: 2026-10-22 · Kass green-lit the v3 design the same evening ("let's go with the start of that V3"): trailing 180-day percentile rank → 0–100, oriented by the bullish/bearish guide. Code: `scripts/build_timeseries_matrix_v3.py` · Stats: `data/timeseries_matrix/build_stats_v3.json`.

**BUILD ONLY — nothing was trained or screened on v3. Trading remains PARKED.**

## What v3 is

- **16,735 rows × 102 columns**, 2024-11-23T00:00:00 → 2026-10-21T16:00:00 — v2's exact rows, timestamps, OHLCV and labels (verified byte/float identical), with every indicator column replaced by a 0–100 score.
- Column order: Date, Ticker, OHLCV, label, score_master, score_trend, score_momentum, score_volume, then the 90 normalized indicator columns **in v2's original relative order, under their v2 names** — same names, new meaning: the values are 0–100 scores, not raw indicator values. The Key tab of the xlsx carries the per-column mapping.
- Labels: {'HOLD': 10976, 'BUY': 2915, 'SHORT': 2844}.

## Method

- **Trailing percentile rank:** for each row, the column's ranked quantity is scored against its own trailing 4,320 hourly values (180 days) including the current row — strictly causal, past + present only. Midrank convention: score = 100 × (r − 1) / 4319 with r = (values below) + (ties + 1)/2, so window-minimum = 0, maximum = 100, median ≈ 50.
- **Ranked quantity** (the series the guide actually reads): raw value for oscillators/distances/gauges; the signed **Close − level gap (% of close)** for the 14 price-level columns (moving averages, channels, trailing stops, VWAP), which the guide reads only against Close; the **24-bar change (slope)** for the 4 cumulative lines (A/D, ASI, OBV, PVT), whose levels the guide calls meaningless; **MACD − Signal** for the MACD Signal Line, which the guide reads only against MACD (its score is therefore identical to the Histogram column's by construction).
- **Orientation** (guide Tab 2 is the authority): 81 directional columns oriented HIGH = BULLISH; exactly one FLIPPED column — Connors RSI, whose guide reading is contrarian (very low raw = dip-buy zone, very high raw = short-the-rip). **Williams %R is deliberately NOT flipped**: Tab 2 reads its top of scale (near 0 = close at the top of the range) as the strong side. 2 columns are STRENGTH (ADX; the DMI column that stores DX — unsigned by construction) and 7 are INTENSITY (ADR, Bollinger BandWidth, CHOP, HV, Mass Index, SD, ATR half-width): normalized as-is, high = stronger / more extreme, with NO bull/bear claim and no composite seat.
- **Composites** (directional columns only, grouped by the guide's Tab 2 families): score_trend = mean of 45 trend columns, score_momentum = mean of 25, score_volume = mean of 7; score_master = the mean of the three family scores (mean of family means, so the 45-column trend family cannot swamp the 7-column volume family).
- **Excluded from composites** (directional, but the Tab 2 family is dual or off-axis — noted per column in the key): Ease of Movement and Money Flow Index (dual volume × momentum families), Relative Volatility Index (volatility-direction family), and the $5,000 round-number distance (price-psychology reference).
- Windows are consecutive **rows** of the long series. The source Coinbase record has 3 small holes (3–5 missing hours each; shared with v1/v2 — the rows simply don't exist in any version), so the few windows straddling a hole span slightly more than 180 days of wall time. Disclosed, mechanical, and identical in spirit to how v1/v2 treated the same record.
- Series provenance: all 90 indicator series were recomputed on the LONG hourly frame (42,107 rows from 2022-01-01) via the v1/v2 build code paths, so every matrix row's 180-day window is full of real history.

## Verification (all PASS)

- **Identity vs v2:** timestamps, OHLCV (exact, max diff 0.0) and labels identical on all 16,735 rows.
- **Reproduction before normalization:** all 90 recomputed series reproduce v2's saved values on the matrix rows under the same code paths — 85 directly (worst case Klinger Volume Oscillator (KVO): max abs diff 4.89e-04 against a column std of 18.2 — floating-point / EMA-seed residue), the 4 cumulative lines exactly in their ranked quantity (24-bar slope max diff ≤ 3.5e-10; their levels carry an arbitrary start offset by construction), and ADR with one documented exception: v1's source frame starts 2024-11-09 23:00, so its first 'day' was a 1-hour stub whose tiny range sat in v1's ADR for the first 24 matrix rows (max diff $53.0 on a column with std $968); the long frame holds the true full day, and the series are exact from row 24 on.
- **Second method:** the percentile for RSI was recomputed at rows 0 / 8,000 / 16,734 with an independent brute-force numpy implementation — exact agreement with the sliding-window implementation.
- **Causality (truncation):** series rebuilt on the long frame truncated at matrix row 9,000 reproduce the full-frame scores EXACTLY (max diff 0.0) on 8 sampled columns spanning every ranked-quantity type and orientation (RSI, EMA gap, OBV slope, CRSI flip, ADX strength, HV intensity, MACD-Signal gap, SuperTrend Side) × 9,001 rows.
- **Orientation spot checks** (raw v2 value → v3 score):
  - RSI (HIGH=BULLISH): highest / median / lowest raw RSI rows: row 15221 (2026-08-19): raw 91.88 → score 100.00; row 10416 (2026-01-31): raw 50.23 → score 53.00; row 10430 (2026-01-31): raw 11.09 → score 0.00.
  - CRSI (FLIPPED): highest / median / lowest raw CRSI rows: row 9622 (2025-12-29): raw 98.90 → score 0.00; row 12122 (2026-04-12): raw 51.05 → score 49.34; row 7724 (2025-10-10): raw 2.53 → score 100.00.
- **Range audit:** every transformed column within [0, 100], zero NaNs across all 90 × 16,735 scores; composites within [0, 100] and equal to their documented family means on sampled rows (master = mean of the three family scores, exact).
- **Distribution sanity:** % of rows each composite spends above 50 — trend 49.01%, momentum 48.39%, volume 49.85%, master 49.15% — near half by construction, nothing wild.

## Files

| File | Notes |
|---|---|
| `data/timeseries_matrix/btc_usd_1h_matrix_v3.csv` | The machine copy (CSV cannot carry formatting). |
| `~/workspace/your_files/BTC-USD_1h_Matrix_v3_Normalized.xlsx` | Kass's READABILITY copy: every populated cell BOLD (header + all data cells, verified on read-back), frozen top row, sized columns. Tabs: 'Matrix v3' (data) + 'Key' (per-column family, ranked quantity, orientation, guide basis, composite seat). |
| `data/timeseries_matrix/build_stats_v3.json` | Machine-readable spec, reproduction stats, checks. |
| `scripts/build_timeseries_matrix_v3.py` | Deterministic rebuild; reproduction check aborts the build on an unexplained mismatch. |

Nothing was trained or screened on v3 — Kass sees the table first, per the established pattern.
