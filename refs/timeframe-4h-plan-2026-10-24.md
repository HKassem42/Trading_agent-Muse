# Pre-registered test plan — Drawing-Board Option A: the 4-hour timeframe

Date: 2026-10-24 · Status: **PLAN — written before any training or
grading.** Kass's order (2026-10-24 ~01:15 ET): "start with A then B
then C" — A is the 4h timeframe, the cleanest untried axis, with data
already in hand. This file is the contract: the rules below are fixed
before any result exists, and any forced deviation is reported as a
deviation in the final report, never quietly absorbed.

## The question

Every prior test — the TA stacker, the PPO, HA-Seq, the 132-indicator
A-Z screen, the GBM focus/lag tests, the v3 composite screen — graded
the same information on **1h** bars and returned null (six independent
nulls). This test changes exactly one thing: the bar size. Same
indicators, same learner family, same duel discipline, on **4h** bars.
Does the same information behave differently at 4h?

## Data (preparation census — computed before this plan was finalized,
no model fitted, nothing graded)

- **Bars:** `data/history/btc_usd_1h_long.csv` (42,107 hourly bars,
  2022-01-01 → 2026-10-21 23:00 UTC) resampled to 4h bins
  (`closed="left", label="left"` — a bar's timestamp is its OPEN time,
  the hourly convention). OHLCV: open = first, high = max, low = min,
  close = last, volume = sum. **Only complete bars are kept** (a bin
  must contain exactly 4 hourly bars). 10,530 bins → 6 incomplete bins
  dropped, all at the source frame's 3 known gaps (2023-03-04 ×2 bins,
  2025-10-25 ×2, 2026-05-08 ×2) → **10,524 complete bars**.
- **Slice:** bars with open time in the v1/v2 matrix span
  [2024-11-23 00:00, 2026-10-21 16:00] (the v1 matrix's first/last row
  timestamps) → **exactly 4,183 bars**. Features are computed on the
  full long 4h frame first (warmups), then sliced — the v2 pattern.
- **Features: the 90 indicator columns of matrix v2** — the 59 v1
  columns via `scripts/build_timeseries_matrix.py`'s builders
  (az_indicators + local raw builders) and the 31 main-chart columns
  via `src/features/main_chart.py::build_all`, both run verbatim on
  the 4h frame. All parameters are bar counts, as throughout technical
  analysis (an SMA-20 at 4h is 20 bars); the builders are
  timeframe-agnostic. Prior-day pivot/H-L features remain
  prior-**calendar-day** (the builder groups by UTC date) as required.
  One flagged instance of the bar-count rule: the VWAP column's
  builder rolls **24 bars** (96 hours at 4h; 24 hours at 1h) — the
  identical code path, on the record here rather than silent.
  **Exclusions: NONE.** The pre-plan census found zero NaN in all 90
  columns across the slice, so no column needs dropping.
- **Labels:** the v1 convention verbatim — next surviving bar's close
  return r: BUY if r ≥ +0.003, SHORT if r ≤ −0.003, else HOLD (the
  0.30% band = 2 × 0.15% round-trip cost; a cost, not a tuned knob).
  Class split, full slice (4,183): **BUY 1,302 (31.1%) / HOLD 1,656
  (39.6%) / SHORT 1,225 (29.3%)** — far more balanced than 1h
  (17.4/65.6/17.0), because 4h moves clear the band more often.
  Pre-duel rows (3,985): BUY 1,249 / HOLD 1,559 / SHORT 1,177.
  Labels were recomputed by a second independent path in the census
  and agreed on every sliced row.

## Arms (2) — fixed

Both LightGBM multiclass with `models.gbm_focus.LGBM_PARAMS`
**verbatim** (300 trees, lr 0.05, 31 leaves, min-child 40,
subsample 0.9, colsample 0.9, λ=1.0, seed 42) and **balanced
class weights** (inverse class frequency — the focus test's arm-B/D
treatment, the style that traded most honestly). Duel policy: the
pre-registered conviction rule of `models.gbm_policy` (LONG at
P(BUY) ≥ 0.40 ≥ P(SHORT); SHORT at P(SHORT) ≥ 0.40 > P(BUY); else the
flat band), unchanged from runs 21–27.

1. **GBM-All4h** — all 90 features at 4h.
2. **GBM-Focus4h** — Kass's focus-11 features at 4h (the direct
   timeframe twin of the 1h focus test): Volume, OBV, Force Index,
   EFI, HV, Mass Index, Bollinger BandWidth, Connors RSI, ASI,
   Ultimate Oscillator, Choppiness Index.

## Walk-forward (fixed)

TimeSeriesSplit, 5 folds, over the **3,985 pre-duel rows** only.
Report pooled accuracy **next to** the majority baseline of the
graded rows, and balanced accuracy next to 1/3 chance, plus the
per-fold spread. Accuracy alone is never reported as success.

## Duel (fixed)

Signal-policy broker harness — the mirror of
`training.nightly.evaluate_ppo` used by the GBM/composite studies
(`StudyBroker`/`StudyEnv`, default `ExitConfig` = the verified
production exit recipe: through-flat transitions, 2×ATR stop with
the 2.5% floor, trailing as designed, no take-profit, stop-anchor
ratchet), scratch ledger DBs in /tmp, $100,000 episodes.

- **Windows are CALENDAR windows:** 66 bars = 11 days each. Standard
  set = the final 198 sliced bars (3 × 66; first window opens
  **2026-09-18 20:00 UTC** — the 1h standard span starts 2026-09-19
  11:00; the ~15h difference is bar alignment plus the two in-span
  gap holes shifting positional slicing). Robustness set = the 198
  bars immediately before (first window opens 2026-08-16 20:00) —
  the same calendar spans as the exit/size/composite studies.
- **Episode geometry scales with the bar:** 10 episodes per window,
  starting every 6 bars (one day), 6 steps each — the exact analog
  of 1h's 10 × 24 inside 264 (10 graded days + 1 buffer day).
- **Fills:** next-bar open. Tranches 25% of equity, 75% exposure cap,
  0.15% fee per side, $10 min ticket — the production broker rules.
- **Signals:** for the standard windows, each arm's final model is
  fit on all 3,985 pre-duel rows; for the robustness windows, each
  arm is **refit on rows strictly before the robustness region**
  (out-of-sample there — the lag/composite discipline).
- **Fee transparency:** net AND gross (pre-fee) P&L, total fees,
  round trips and exact non-hold activity share for every duel.
- **There is NO PPO-champion party at 4h.** The champion is a 1h
  brain; it cannot be graded on 4h bars and is not a duel opponent
  here. The yardsticks are **buy-and-hold on the same windows** and
  the floor conditions below. (B&H per window = $100k × last/first
  close − 1 over the window's bars, the harness formula.)

## Pre-registered verdict rules — all three or NULL

1. **Walk-forward:** pooled balanced accuracy > 1/3 AND at least 4
   of 5 folds' balanced accuracy > 1/3 (the fold-spread clause, the
   composite screen's disclosed operationalization) AND pooled
   accuracy ≥ the majority baseline of the graded rows.
2. **Duel floor (absolute — no relative formula exists at 4h):**
   total duel P&L > $0 AND total duel P&L > buy-and-hold on the same
   windows AND non-hold activity share ≥ 10%.
3. **Robustness:** robustness-set total P&L ≥ $0.

Any failure = the arm is NULL, registered promoted=0, and the design
is **not re-tuned for a second shot**. A full pass is a report to
Kass, not a promotion — promotion would be his separate decision.

## Integrity checks (run with the build, results in the report)

- Resample correctness on a hand-checkable slice: one 4h bar
  recomputed from its 4 hourly bars by an independent path.
- Truncation causality: features recomputed on a truncated long
  frame match the full-frame values on sampled columns/rows.
- Label recomputation by a second path on sampled rows.
- Zero-conviction duel sanity: a flat signal produces zero fills
  and P&L $0 through the 4h harness.
- Fee identity: gross = net + total fees, exactly, per duel.

## Registration

One training.db run per arm, `family="tf4h"`, promoted=0 always,
with duel + walk-forward metrics and the rule-by-rule verdict in the
record; family code 8 in the dashboards' models index ("4H
Timeframe — All Features" / "4H Timeframe — Focus 11"); one lab note.

## What this plan is not

- Not a production change: the live bot (un-parked 2026-10-24 on
  Kass's override, run-44 champion) is untouched by this lab study;
  nothing here promotes, whatever the outcome.
- Not a search over timeframes: only 4h is tested, under this one
  design. A null closes Option A; it does not license trying 2h/8h
  next without Kass's direction (Options B and C are his stated
  sequence).
