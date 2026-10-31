# Candle-Streak Tables on Long History — 12h / 1D / 1Month, Raw + Heikin-Ashi

Date: 2026-10-22 · Part A of the research build Kass green-lit
2026-10-22 ("create the tables and test those for the time frames
12hr, 1D, 1Month… experiment with pulling in more history with
Coinbase API and see if that effects anything").
Scripts: `scripts/pull_coinbase_history.py`,
`scripts/verify_history.py`, `scripts/streak_tables.py`.
Tables: `data/streaks/streaks_<tf>_<type>.csv` (6 files, tidy
metric rows) + `data/streaks/streak_summary.json` +
`data/streaks/unconditional_baselines.json`.

**No model was trained. Trading remains PARKED.** This is the
descriptive evidence layer Kass asked for before any decision.

## Headline verdict

Kass's question was whether the numbers get *better* at slower
timeframes with more history. They get **bigger** (moves are
larger), but not **cleaner** — and the one cell that looked like a
discovery on 2 years of daily data **reversed sign** on 11 years:

> **Re-test of the daily HA green-exhaustion cell.** On the v1
> matrix window (2024-11-23 → 2026-10-21), after 5 / 6 straight
> Heikin-Ashi green days the next 3 days averaged **−1.21% /
> −0.98%** (n = 31 / 26) — apparent exhaustion. This pipeline
> reproduces those figures to the third decimal on the same
> window. On the full 2015–2026 history (n = 174 / 144), the same
> cells read **+0.82% / +1.02%** over the next 3 days and
> **+1.24% / +1.89%** over the next 6 (P(>0) 0.52 / 0.58; the
> k = 6 median is also positive, +1.03%). **The exhaustion effect
> was a 2-year-window artifact. It is dead.**

What survives, at hypothesis grade only: at the daily timeframe,
long streaks of *either* kind are followed by returns at or
modestly above the (strongly positive) base rate — mild momentum
after green streaks, a mild bounce after very long HA red streaks.
At 12h there is exactly one above-baseline cell worth a second
look. At 1Month the samples are anecdotes (cells rest on n = 1–17)
and BTC's secular drift (+6.2%/month unconditional) swamps any
streak effect. Details and cautions below.

## The history pull (Coinbase public API, no auth)

| File | Span | Rows | Gaps | Duplicates |
|---|---|---|---|---|
| `data/history/btc_usd_1d_long.csv` | 2015-07-20 → 2026-10-21 | **4,112 days** | **0** | 0 |
| `data/history/btc_usd_1h_long.csv` | 2022-01-01 → 2026-10-21 23:00 | **42,107 hours** | 3 gaps, 13 candles | 0 |

- Coinbase's first BTC/USD candle is 2015-07-20; the daily series
  is **complete** from that day to 2026-10-21 (4,112/4,112).
- The hourly pull is checkpointed/resumable (per-chunk files under
  `data/history/chunks_hourly/`, progress in
  `data/history/pull_progress.json`).
- The 13 missing hourly candles sit in 3 gaps: 3 candles on
  2023-03-04, and 5 each on 2025-10-25 and 2026-05-08 — the two
  2025/2026 gaps are *the same Coinbase-side absences* already
  documented in the v1 matrix frame, reproduced independently by
  this fresh pull.
- **Two-angle verification** (`scripts/verify_history.py`): the
  pulled daily candles were cross-checked against daily resamples
  of the v1 matrix's hourly candles over the 695 fully covered
  overlap days — **Open / High / Low / Close match to the cent on
  695 of 695 days (100%, max abs diff 0.0)**. The pull and the
  project's existing frame agree exactly.

## Definitions (all causal — no future information anywhere)

- **Color (raw):** green if close ≥ open, else red.
- **Heikin-Ashi:** standard recursion — HA close = (O+H+L+C)/4;
  HA open[0] = (O[0]+C[0])/2; HA open[t] = (HA open[t−1] +
  HA close[t−1])/2. Color = HA close ≥ HA open.
- **Streak of exact length k at candle t:** the run of same-color
  candles ending at t has length exactly k (it may continue; only
  information up to t is used). Forward returns are REAL-price,
  close[t+h]/close[t] − 1.
- **Completed runs** (for run-length stats) end where the next
  candle's color differs; the final run is included,
  right-censored. **Correctness asserts inside the script:** run
  lengths must tile the series (sum == candle count) and green-run
  / red-run counts must differ by ≤ 1 — the fix for the previous
  run-aggregation bug (run ends were mis-marked by comparing
  run-lengths instead of colors). Both asserts pass on all six
  tables.
- Horizons: 1 / 3 / 6 candles (12h, 1D); 1 / 2 / 3 candles
  (1Month).
- Fee band: the project's round-trip cost is **0.30%** (2 ×
  0.15%). Compared per timeframe in each section.
- Multiple comparisons: this study prints 3 timeframes × 2 candle
  types × 2 colors × 7 streak lengths × 3 horizons = **252
  forward-return cells**. Under pure noise, ~13 cells would clear
  p < 0.05 by chance. Any single cell below is a lead, not a
  finding, until it survives a dedicated walk-forward test.

## The tide: unconditional forward returns (no streak condition)

BTC rose from ~$280 (Jul 2015) to ~$81k (Oct 2026) over the daily
window, and ~$46k → ~$81k over the 12h window (2022→2026). Every
conditional number below floats on this drift:

| Timeframe | h=1 | h=3 (h=2 for 1mo) | h=6 (h=3 for 1mo) |
|---|---|---|---|
| 12h | +0.03% (P .51) | +0.10% (P .50) | +0.20% (P .52) |
| 1D | +0.20% (P .52) | +0.59% (P .55) | +1.20% (P .55) |
| 1Month | +6.25% (P .57) | +13.79% (P .63) | +22.18% (P .62) |

A streak cell only says something if it clearly beats its row of
this table.

## 12h (n = 3,510 candles, 2022-01-01 → 2026-10-21)

Run stats — raw: green avg 1.94 / red avg 1.89 candles (median
2 / 1; ~50% of runs are a single candle; 7+ ≈ 1%). HA: green avg
**4.25** / red avg **4.04** (median 3; 7+ ≈ 18%) — the HA doubling
seen at 1h and 6h repeats here (mechanical: HA open averages its
own past, so HA color flips less often *by construction*).

Continuation P(next candle same color | streak k): raw hovers at
0.42–0.51 — a coin flip with a whisper of anti-persistence. HA
runs 0.71–0.80 — but that predicts the next **HA candle's color**
(a formula output), not price: HA forward P(>0) stays at base
rates.

Forward returns after streaks (mean %; full grid incl. medians in
the CSVs). Standouts and non-standouts:

- HA green k = 6: h3 **+0.64%** (median +0.55, P .62, **n = 102**)
  vs unconditional +0.10% — the only 12h cell clearly above its
  baseline with a three-digit n. k = 7–8 point the same way with
  smaller n. Hypothesis-grade: "HA trend maturity at 12h does not
  mean exhaustion; if anything the opposite."
- HA red k = 5–6: h3 −0.15 / −0.05, h6 −0.36 / −0.10 — **no**
  dip-bounce at 12h; slightly negative means, medians ≈ 0.
- Raw 12h: incoherent — means and medians routinely disagree in
  sign, k ≥ 7 cells rest on n ≤ 10. Nothing here survives its own
  error bars.
- Fee band: nearly every h1 cell has |mean| < 0.30% — at a
  one-candle horizon the fee eats the whole effect, everywhere.

## 1D (n = 4,112 candles, 2015-07-20 → 2026-10-21)

Run stats — raw: green avg 1.99 / red avg 1.81 (median 2 / 1).
HA: green avg **4.73** / red avg **3.92** (median 3; max 30 / 23).

Continuation: raw 0.44–0.59 (coin flip; red k = 4 dips to 0.36,
n = 105). HA 0.70–0.83 (formula persistence, as above).

Forward returns after green streaks — mean % (median) · P(>0) · n:

| Type | k | h1 | h3 | h6 |
|---|---|---|---|---|
| raw | 3 | +0.22 (−0.01) · .50 · 252 | +0.80 (+0.60) · .58 · 252 | +1.67 (+0.66) · .53 · 252 |
| raw | 4 | +0.82 (+0.39) · .55 · 126 | **+2.21 (+0.82) · .63 · 126** | +3.05 (+1.56) · .63 · 126 |
| raw | 5 | +0.64 (−0.11) · .49 · 70 | +2.18 (+0.53) · .59 · 70 | +3.14 (+0.43) · .51 · 70 |
| raw | 6 | +1.48 (+0.81) · .59 · 34 | +2.93 (+1.95) · .68 · 34 | +4.50 (+2.06) · .62 · 34 |
| HA | 3 | +0.46 (+0.06) · .51 · 282 | +0.65 (+0.37) · .53 · 282 | +0.89 (+0.17) · .53 · 282 |
| HA | 4 | +0.29 (+0.24) · .54 · 226 | +0.64 (+0.23) · .53 · 226 | +1.24 (+0.29) · .52 · 226 |
| HA | 5 | +0.13 (−0.09) · .48 · 174 | **+0.82 (+0.14) · .52 · 174** | +1.24 (+0.33) · .52 · 174 |
| HA | 6 | +0.64 (+0.07) · .51 · 144 | **+1.02 (+1.03) · .58 · 144** | +1.89 (+0.94) · .58 · 144 |

Forward returns after red streaks — mean % (median) · P(>0) · n:

| Type | k | h1 | h3 | h6 |
|---|---|---|---|---|
| raw | 3 | +0.16 (+0.27) · .54 · 226 | +0.60 (+0.58) · .59 · 226 | +0.76 (+0.58) · .54 · 226 |
| raw | 4 | +1.05 (+0.59) · .63 · 105 | +0.93 (+0.94) · .62 · 105 | +0.77 (+0.32) · .54 · 105 |
| raw | 5 | −0.22 (+0.18) · .53 · 38 | −1.38 (−0.76) · .42 · 38 | −0.77 (+0.49) · .55 · 38 |
| raw | 6 | −0.82 (+0.65) · .56 · 18 | −1.44 (+0.61) · .56 · 18 | −2.55 (+1.44) · .61 · 18 |
| HA | 5 | +0.27 (+0.35) · .59 · 155 | +0.31 (+0.67) · .59 · 155 | +1.10 (+1.05) · .58 · 154 |
| HA | 6 | +0.10 (+0.36) · .55 · 109 | +0.55 (+0.71) · .61 · 109 | +1.32 (+1.99) · .60 · 108 |
| HA | 7 | +0.10 (−0.16) · .46 · 78 | +1.12 (+1.82) · .68 · 78 | +2.50 (+1.48) · .62 · 77 |
| HA | 8 | +0.53 (+0.69) · .60 · 58 | **+2.18 (+1.79) · .72 · 58** | +2.93 (+1.87) · .64 · 58 |

Reading against the baseline (+0.59 / +1.20 at h3 / h6):

- **Green side = mild momentum, not exhaustion.** Both raw and HA
  green streaks precede above-baseline returns that grow with k
  (raw k = 4–6 strongly so, but n falls to 34). This is the exact
  opposite of the 2-year HA cell — see the headline re-test.
- **Red side (HA) = mild bounce at long streaks.** HA red k = 7–8
  sit clearly above baseline across h3/h6 with consistent
  medians and P .60–.72 (n = 58–78): the strongest above-baseline
  pattern in the whole study — and still only ~1–1.7 points over
  a rising base rate, inside a 252-cell scan. Hypothesis-grade.
- Raw red k = 5–6 means are negative while medians are positive:
  a minority of crash-continuation episodes drag the mean. No
  clean tradable story.
- Fee band: daily moves dwarf the 0.30% round trip (baseline h1
  alone is +0.20%); at this timeframe **the binding constraint is
  the base rate, not the fee** — a streak rule must beat +0.59%
  (h3) just to beat doing nothing, before it beats anything else.

## 1Month (n = 136 candles, Jul 2015 → Oct 2026) — thin by construction

136 monthly candles is all BTC-on-Coinbase history there is.
First (Jul 2015) and last (Oct 2026) months are partial. Treat
everything here as regime description, not evidence:

- Runs: raw green avg 2.36 (33 runs) / red avg 1.81 (32 runs);
  HA green avg **5.71** (only 14 runs; 43% are 7+ months) / red
  avg 4.31 (13 runs). A "streak" at this scale is a bull or bear
  *era* (2017 run-up, 2021 top, 2022 bear…).
- Forward cells rest on n = 1–17. After HA green k = 2, next
  month averaged +11.9% (n = 12) vs +6.25% unconditional — the
  same mild-momentum direction as daily, on a sample where two
  observations would flip it. HA red k = 6–8 cells read −17 to
  −28% on **n = 2–3** (the 2018 and 2022 bear continuations) —
  quoted only to show why they must not be quoted.
- Continuation for monthly HA green runs 0.57–1.00 on n = 7–14:
  monthly trends are the market's regimes; this is a tautology
  with error bars, not a signal.

## Bottom line for the drawing board

1. The daily HA green-exhaustion idea — the single most promising
   cell from the 2-year study — does not survive longer history;
   it inverts. More history did exactly its job: it killed a
   mirage before it cost anything.
2. Slower timeframes do not rescue streaks as a standalone
   strategy: 12h is noise around a small drift, 1Month is
   anecdote. The least-weak daily patterns (mild momentum after
   green streaks; mild bounce after very long HA red streaks) are
   base-rate-flavored and would need the full walk-forward +
   duel gauntlet — against buy-and-hold on the same windows —
   before being called anything but leads.
3. HA streak lengths roughly double raw ones at every timeframe
   (1h ≈ 4.0, 12h ≈ 4.1–4.3, 1D ≈ 3.9–4.7, 1Mo ≈ 4.3–5.7 vs raw
   ≈ 1.8–2.4): a clearer *picture* of trend persistence, whose
   forward-return information at 12h/1D is no stronger than raw —
   and at 1h was weaker (previous study). HA color predicts the
   next HA candle; it does not obviously predict price.
