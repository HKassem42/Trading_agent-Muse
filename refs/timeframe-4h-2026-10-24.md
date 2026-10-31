# Drawing-Board Option A — The 4-Hour Timeframe Test (2026-10-24)

**Verdict: NULL — the seventh independent null. Both arms FAIL all
three pre-registered rules.** Registered as training.db runs **50**
(GBM-All4h) and **51** (GBM-Focus4h), family `tf4h`, promoted=0.
Executed exactly per the pre-registered plan
(`refs/timeframe-4h-plan-2026-10-24.md`, written before any training
or grading). Lab only — the live bot, champions, paper.db and
live_state were never touched (champion mtimes verified unchanged
at the end); no dashboard pushes (the 15-minute job syncs).

**The question Kass asked:** does the same indicator information
behave differently on 4h bars? **Answer: it behaves worse.** The
faint ranking tilt the weighted 1h arms showed (balanced accuracy
0.41–0.42) shrinks to 0.34–0.35 at 4h — a couple of points over the
1/3 chance line — and the trading loses money *before* fees on the
standard windows, which the 1h arms did not.

## Design (as pre-registered)

- **Bars:** the long hourly frame (42,107 rows, 2022→) resampled to
  4h bins; a bin survives only with exactly 4 hourly bars (the
  source frame's 3 gap spans drop 6 bins) → **10,524 complete 4h
  bars**, open-time stamped. Features computed on the full long
  frame, sliced to the v1/v2 matrix span → **4,183 bars**
  (2024-11-23 00:00 → 2026-10-21 16:00).
- **Features:** the same 90 matrix-v2 indicator columns (59 v1
  builders + 31 main-chart), same production code paths.
  **Exclusions: none** — zero NaNs in all 90 columns in-slice
  (census in the plan; pinned by `tests/test_tf4h.py`). One naming
  note: the guide's "VWAP" builder rolls 24 *bars* (24h at 1h, 96h
  at 4h) and prior-day features remain prior-calendar-day — same
  code, disclosed in the plan.
- **Labels:** next-bar close return, v1 convention unchanged:
  ≥ +0.30% BUY / ≤ −0.30% SHORT / else HOLD.
- **Arms:** LightGBM, `LGBM_PARAMS` verbatim from the GBM focus
  study, inverse-frequency class weights, 0.40 conviction rule —
  **GBM-All4h** (all 90 features) and **GBM-Focus4h** (Kass's
  focus-11 at 4h, the direct timeframe twin of 1h run 22).
- **Walk-forward:** TimeSeriesSplit(5) over the 3,985 pre-duel rows.
- **Duel:** the standard calendar span as 66-bar windows (standard
  starts 2026-09-18 20:00; robustness 2026-08-16 20:00), episodes
  scaled to the bar (10 × 6 steps, starts every 6 bars), signal-
  policy broker harness, next-bar-open fills, 25% / 75% / 0.15%,
  the verified exit recipe. **No PPO champion party at 4h** (a 1h
  brain cannot be graded on 4h bars) — yardsticks are buy-and-hold
  and the plan's absolute floor.

## 4h class split (4,183 bars)

| Class | Count | Share |
|---|---|---|
| BUY | 1,302 | 31.1% |
| HOLD | 1,656 | 39.6% |
| SHORT | 1,225 | 29.3% |

Pre-duel rows (3,985): BUY 1,249 / HOLD 1,559 / SHORT 1,177. Note
the structural change vs 1h (HOLD 65.6% there): a ±0.3% band binds
far less when bars are 4× longer, so the majority baseline falls to
**0.4057** — an easier bar than 1h's 0.6690. Neither arm clears it.

## Walk-forward (pooled OOF: 3,320 graded rows)

| Arm | Accuracy | Majority base | Balanced acc | Pred mix B/H/S |
|---|---|---|---|---|
| GBM-All4h | 0.3370 | 0.4057 | 0.3423 | 29.5 / 28.7 / 41.9 % |
| GBM-Focus4h | 0.3554 | 0.4057 | 0.3523 | 34.4 / 33.9 / 31.7 % |

Fold balanced accuracies — All4h: 0.3568 / 0.3219 / 0.3515 / 0.3479
/ 0.3383 (4/5 above 1/3). Focus4h: 0.3601 / 0.3095 / 0.3920 / 0.3628
/ 0.3490 (4/5 above 1/3). The balanced-accuracy legs of rule 1
pass; the raw-accuracy leg fails for both arms by 5–7 points.

## Duel — standard windows (3 × 66 bars)

| Arm | Agg score | Net P&L | Gross P&L | Fees | B&H | Closed trips W/L | Non-hold |
|---|---|---|---|---|---|---|---|
| GBM-All4h | −1,714.04 | **−$3,625.26** | −$2,390.82 | $1,234.44 | +$4,416.50 | 10 (3/7) | 15.6% |
| GBM-Focus4h | −1,739.30 | **−$3,596.43** | −$2,058.59 | $1,537.84 | +$4,416.50 | 13 (6/7) | 22.2% |

Per window (net P&L): All4h −$2,017.70 / −$825.14 / −$782.42;
Focus4h −$3,067.19 / −$297.55 / −$231.69. Side split: All4h long
−$540.54 / short −$3,084.72; Focus4h long −$196.50 / short
−$3,399.93. Activity clears the 10% floor for both arms — this
null is not an inactivity artifact.

## Duel — robustness windows (the preceding 198 bars)

| Arm | Agg score | Net P&L | Gross P&L | Fees | B&H | Closed trips W/L | Non-hold |
|---|---|---|---|---|---|---|---|
| GBM-All4h | −3,620.65 | **−$8,356.02** | −$5,653.66 | $2,702.36 | +$28,678.14 | 9 (1/8) | 40.0% |
| GBM-Focus4h | −1,967.65 | **−$4,192.91** | −$2,396.36 | $1,796.55 | +$28,678.14 | 18 (6/12) | 25.6% |

The robustness span (2026-08-16 → 09-18) was a strong up-leg; the
short-leaning signals fought it (All4h's robustness loss is 100%
short-side). Buy-and-hold's +$28,678 is the window's own benchmark,
not a strategy result.

## Verdict, rule by rule (all three required)

| Rule | GBM-All4h | GBM-Focus4h |
|---|---|---|
| 1. WF: balanced > 1/3 pooled + ≥4/5 folds, AND accuracy ≥ majority | **FAIL** — bal 0.3423 ✓, folds 4/5 ✓, acc 0.3370 < 0.4057 ✗ | **FAIL** — bal 0.3523 ✓, folds 4/5 ✓, acc 0.3554 < 0.4057 ✗ |
| 2. Duel: P&L > $0 AND > B&H AND non-hold ≥ 10% | **FAIL** — −$3,625 vs +$4,417 (activity ✓) | **FAIL** — −$3,596 vs +$4,417 (activity ✓) |
| 3. Robustness P&L ≥ $0 | **FAIL** — −$8,356 | **FAIL** — −$4,193 |
| **Verdict** | **FAIL** | **FAIL** |

## 1h vs 4h — the focus features, head to head

Same 11 features, same learner, same weights, same conviction rule,
same calendar duel span. (1h figures: run 22 / refs/gbm-focus-
2026-10-22.md; its 60-feature arm D alongside.)

| | 1h Focus B (run 22) | 1h All-60 D | **4h Focus4h (run 51)** | **4h All4h (run 50)** |
|---|---|---|---|---|
| WF accuracy | 0.5508 | 0.5668 | 0.3554 | 0.3370 |
| Majority baseline | 0.6690 | 0.6690 | 0.4057 | 0.4057 |
| Accuracy gap | −11.8 pts | −10.2 pts | −5.0 pts | −6.9 pts |
| Balanced accuracy | 0.4145 | 0.4163 | **0.3523** | **0.3423** |
| Duel net P&L | −$8,432.41 | −$3,178.41 | −$3,596.43 | −$3,625.26 |
| Duel gross P&L | ≈ breakeven* | ≈ breakeven* | **−$2,058.59** | **−$2,390.82** |
| Closed/env trips | 66 | 45 | 13 | 10 |
| Net P&L per trip | −$128 | −$71 | **−$277** | **−$363** |
| B&H, same span | +$717.54 | +$717.54 | +$4,416.50 | +$4,416.50 |

\* The 1h study didn't break gross out in its duel table; the later
size-study fee analysis showed the 1h GBM signals ≈ breakeven
*before* fees on the standard windows (losses were mostly fees).
The 4h arms have no such excuse: they lose money gross. B&H differs
between timeframes because each harness benchmarks its own windows
(the 4h set ends at the 2026-10-21 16:00 bar; the 1h set at the
final 1h candle) — an endpoint effect, not a grading difference.

**Reading.** Three things change at 4h, all for the worse:
(1) the ranking tilt weakens — balanced accuracy falls from ~0.41
to ~0.35, within ~2 points of chance, even though the accuracy bar
(the majority baseline) drops from 0.669 to 0.406;
(2) the losses stop being a fee story — at 1h the signals were
roughly breakeven before costs and died on churn; at 4h they are
negative before costs on the standard windows, so cheaper trading
would not save them;
(3) the smaller absolute loss (−$3.6k vs −$8.4k) is only fewer
trades (13 vs 66) — per closed trip, 4h is more than twice as bad
(−$277 vs −$128). The timeframe axis does not rescue this
information set. Combined with six prior nulls on 1h, the classic-
indicator information itself — not the bar size, not the fees, not
the exits, not the sizing — is what keeps failing.

## Registration & integrity

- Runs 50 (All4h) / 51 (Focus4h) in training.db, family `tf4h`,
  promoted=0; compact.py models index wired (family code 8, names
  "4H Timeframe — All Features" / "4H Timeframe — Focus 11",
  verdicts read from the runs' own records — verified in the built
  feed: both rows render FAIL with the duel aggregates above).
  Training feed 6,924 bytes at 51 runs — under the 7,100-byte size
  sentinel, no re-baseline needed. One lab note added.
- Tests: `tests/test_tf4h.py` (9 tests — resample hand-check,
  label second-path + inclusive boundary in exact binary
  arithmetic, 4h truncation causality on 6 sampled columns, span/
  feature census, flat-signal and always-long duel geometry with
  the fee identity, verdict mechanics, arm features). **Full suite:
  156 passed** (147 baseline + 9).
- Causality/integrity: resample verified against its 4 hourly bars
  by hand; labels recomputed by a second path (probe census agreed
  on all 4,183 rows; the unit test pins the boundary semantics);
  truncation causality exact on sampled v1 and main-chart columns
  at 4h. Champion mtimes unchanged end-to-end; all grading on
  scratch DBs.

## Deviations from the plan

**None** in design, data, arms, rules, or reporting. Build-machinery
notes (no effect on any number): the working frame's target column
is excluded from the feature census by name (a first WF pass caught
it trying to train on the label column — fixed in
`tf4h.all_feature_columns`, census still exactly 90); the focus
arm's matrix-name "Volume" maps to the frame's raw `volume` column
(the same column the 1h focus arm used); the registration phase
crashed once on a lab-note argument *after* both runs were logged —
a DB-seeded resume guard was added and the runs exist exactly once
(verified: 51 total runs, ids 50/51 are the tf4h pair).

## What this closes, and what's next

Option A is answered: 4h is a null, worse per-trade than 1h. Per
Kass's order (A → B → C), the next drawing-board item is **B —
daily regime data** (Fear & Greed / FRED): a genuinely new
information type rather than another transformation of the same
OHLCV. Then **C — the two-step hierarchy** (trade-or-not →
direction).
