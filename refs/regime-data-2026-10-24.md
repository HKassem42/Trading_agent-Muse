# Drawing-Board Option B — Daily Regime Data Test (2026-10-24)

**Verdict: NULL — the eighth independent null. All three arms FAIL
their pre-registered rules.** Registered as training.db runs **52**
(GBM-Focus+Regime), **53** (GBM-Regime-Only), **54** (F&G Filter),
family `regime_data`, promoted=0. Executed exactly per the
pre-registered plan (`refs/regime-data-plan-2026-10-24.md`, written
before any training or grading). Lab only — the live bot, champions,
paper.db and live_state were never touched by this study (champion
mtimes verified unchanged end-to-end); no dashboard pushes (the
15-minute job syncs; the new family renders in the models index,
verified below).

**The question Kass asked:** every prior test rearranged the same
1h candle information and failed. Does a genuinely **new
information type** — daily regime/context series (Fear & Greed,
VIX, broad USD, yield-curve slope, fed funds) — change the answer
when used as context on the 1h frame? **Answer: no.** The regime
features are not inert — they move walk-forward accuracy and gross
duel P&L a little — but the movement does not survive fees on the
standard windows and reverses outright on the robustness windows.
The F&G veto filter barely fired (3 of 157 active rows) and where
it fired it did not help.

## Data (fetched 2026-10-24; raw CSVs in `data/regime/`)

| Series | Source | Fetched span | Valued rows | Stamps in matrix span | Lag |
|---|---|---|---|---|---|
| Fear & Greed | Alternative.me API | 2018-02-01 → 2026-10-24 | 3,184 | 698 / 698 days | D+1 |
| VIXCLS | FRED fredgraph | 1990-01-02 → 2026-10-22 | 9,301 | 492 (last 2026-10-21) | D+2 |
| DTWEXBGS (broad USD) | FRED fredgraph | 2006-01-02 → 2026-10-16 | 5,212 | 474 | D+2 |
| T10Y2Y | FRED fredgraph | 1976-06-01 → 2026-10-23 | 12,596 | 476 (last 2026-10-21) | D+2 |
| DFF | FRED fredgraph | 1954-07-01 → 2026-10-22 | 26,412 | 698 (last 2026-10-21) | D+2 |

All five pre-registered series were kept; nothing was substituted.
DTWEXBGS's freshest stamp is 2026-10-16 (the source publishes it
late), so the final matrix days read that stamp — disclosed in the
plan before grading. One census clarification: fredgraph prints
US-market-holiday dates with **blank** values (VIX 6 such rows
in-span, USD 21, T10Y2Y 22 — Christmas, New Year, Thanksgiving,
MLK Day, etc.). Those dates are non-stamps under the convention
below, treated identically to weekends; the plan's coverage table
counted CSV rows (498/495/498) where this table counts valued
stamps (492/474/476). No feature value differs from the plan's
definition because of it.

**FRED revision limitation (stated in the plan, repeated here):**
fredgraph serves current, *revised* values; first-published
vintages are not retrievable from it. The D+2 lag mitigates timing
(the value could not have been known before D+2 even unrevised);
it does not remove revision look-ahead. Fear & Greed is published
once and never revised.

## Availability convention + test results

`usable(d) = freshest stamp s with s + lag ≤ d`; an hourly row
takes the usable value of its UTC date; forward-fill only. Features
per series: usable **level** + **5-calendar-day change** (10 regime
columns), zero NaNs across all 16,735 matrix rows (build census).

Pinned by `tests/test_regime_data.py` (9 tests, all passing —
written before the build): stamp exactness on ordinal fixtures
including the midnight boundary (23:00 reads the old stamp, 00:00
the new, never earlier); stamp exactness at D+2 on a
weekend/holiday-gapped fixture, brute-force checked hour by hour;
**truncation invariance** (features rebuilt from a daily frame cut
at T0 are identical for every hour with `date ≤ T0 + lag`, both
lags); **extra-lag staleness identity** (lag+1 features at hour t
equal lag features at t − 24h, exactly); the 5-day change
definition; filter boundary mechanics (25.0 allows longs, 75.0
allows shorts); verdict mechanics. Build-level spot check also
passed: the final matrix day's F&G level is the **Oct 20** stamp
(D+1), not Oct 21's.

## Walk-forward (TimeSeriesSplit 5, 13,300 graded rows)

Majority baseline of the graded rows: **0.6690** (as in every 1h
study). Focus B's recorded WF (run 22) alongside — arm 1's direct
comparison.

| Arm | Accuracy | vs baseline | Balanced acc | Fold BA spread |
|---|---|---|---|---|
| Focus B (run 22, recorded) | 0.5508 | below | 0.4145 | — |
| **1. Focus+Regime** | **0.5687** | below | 0.4021 | 0.377 / 0.360 / 0.420 / 0.414 / 0.411 (5/5 > 1/3) |
| **2. Regime-Only** | 0.5355 | below | 0.3579 | 0.354 / **0.333** / 0.338 / 0.375 / 0.364 (4/5 > 1/3) |

Reading: adding the regime features **raised raw accuracy ~1.8
points** over Focus B (0.5687 vs 0.5508) and **lowered balanced
accuracy ~1.2 points** (0.4021 vs 0.4145) — a wash, and both arms
remain ~10 points under the always-HOLD accuracy baseline.
Regime-Only on its own shows the familiar faint tilt (pooled
balanced 0.3579, most folds a hair over chance) with no accuracy
competitiveness. The new information whispers, like the old.

## Duel — standard windows (3 × 264h, from 2026-09-19 11:00 UTC)

Yardsticks graded fresh through the same mirror (cross-checked
against `nightly.evaluate_windows` on the champion: **diff
0.0000**): the current champion (run 44's brain) agg **−662.43**,
net −$994.49 (gross −$318.83, fees $675.65), buy-and-hold
**+$717.54** — identical to the composite screen's fresh grade.
The unfiltered Focus B base (run 22's own cached probabilities,
integrity reference, not an arm) graded agg **−3,329.80**, net
−$8,138.37 (gross +$879.33, fees $9,017.70) — **exactly** the size
study's current-code reproduction of run 22 (its pre-anchor-fix
record was −3,476.80), confirming the signal lineage to the cent.

| Arm | Agg score | Net P&L | Gross P&L | Fees | B&H | Closed W/L | Non-hold |
|---|---|---|---|---|---|---|---|
| **1. Focus+Regime** | −3,982.52 | **−$9,293.66** | +$2,766.41 | $12,060.08 | +$717.54 | 57 (18/39) | 55.4% |
| **3. F&G Filter** | −3,372.13 | **−$8,265.37** | +$527.94 | $8,793.31 | +$717.54 | 66 (16/50) | 35.1% |
| (ref) Focus B base | −3,329.80 | −$8,138.37 | +$879.33 | $9,017.70 | +$717.54 | 68 (17/51) | 36.0% |
| (ref) Champion (run 44) | −662.43 | −$994.49 | −$318.83 | $675.65 | +$717.54 | 5 (1/4) | 45.1% |

- **Arm 1** improved the *gross* result over its base (+$2,766 vs
  +$879 — the regime features did add gross signal on these
  windows) but traded much more (non-hold 55.4% vs 36.0%), paid
  $3,042 more in fees, and finished **$1,155 worse net** than the
  base it was meant to lift. Its losses are short-side, as usual
  (−$8,597 short / −$696 long).
- **Arm 3 (filter):** only **3 vetoes** in the whole standard
  duel (of 157 active conviction rows) — the duel span simply
  contains almost no extreme-band conviction hours. All 3 landed
  in window 0, which got slightly *worse* (net −$3,603 vs base
  −$3,476): the vetoed trades had contributed +$351 gross, the
  fee saving was $224, net effect **−$127**. Windows 1–2 are
  identical to the base to the cent. The veto principle was barely
  tested by this market span, and where tested, it did not pay.

## Duel — robustness windows (the preceding 792h, from 2026-08-17)

Refits on rows strictly before the region, per plan. B&H on this
span: **+$27,401.66** (a strong up-leg; the short-leaning signals
fought it — the same pattern as every prior study).

| Arm | Agg score | Net P&L | Gross P&L | Fees | Closed W/L | Non-hold |
|---|---|---|---|---|---|---|
| **1. Focus+Regime** | −5,228.46 | **−$11,934.96** | −$4,804.65 | $7,130.31 | 60 (15/45) | 29.9% |
| **3. F&G Filter** | −8,429.01 | **−$20,858.95** | −$10,870.61 | $9,988.34 | 68 (16/52) | 38.3% |
| (ref) Champion (run 44) | −4,805.59 | −$10,750.15 | −$8,986.07 | $1,764.08 | 11 (3/8) | 47.5% |

Arm 1's standard-window gross gain **reverses** here: −$4,805
gross. Arm 3 fired **zero vetoes** on this span, so its robustness
line *is* its refit base signal's line — the filter changed
nothing at all where the losses were largest.

## Verdict, rule by rule (as pre-registered)

| Rule | 1. Focus+Regime | 2. Regime-Only | 3. F&G Filter |
|---|---|---|---|
| 1. WF: balanced > 1/3 pooled + ≥4/5 folds, AND accuracy ≥ majority | **FAIL** — bal legs pass (0.4021, 5/5 folds); acc 0.5687 < 0.6690 ✗ | **FAIL** — bal 0.3579 ✓, folds 4/5 ✓; acc 0.5355 < 0.6690 ✗ | — (no WF by design) |
| 2. Duel: P&L > $0 AND > B&H AND non-hold ≥ 10% | **FAIL** — −$9,294 (activity ✓ 55.4%) | — (diagnostic) | **FAIL** — −$8,265 (activity ✓ 35.1%) |
| 3. Robustness P&L ≥ $0 | **FAIL** — −$11,935 | — | **FAIL** — −$20,859 |
| **Verdict** | **FAIL (NULL)** | **FAIL (diagnostic)** | **FAIL (NULL)** |

## Registration & integrity

- Runs 52 / 53 / 54 in training.db, family `regime_data`,
  promoted=0; compact.py models index wired (family code 9, names
  "Regime Data — Focus+Regime" / "Regime Data — Regime Only" /
  "Regime Data — F&G Filter", verdicts read from the runs' own
  records — verified in the built feed: all three rows render FAIL
  with the duel aggregates above). One lab note added.
- Training feed: 7,268B at 54 runs — tripped the 7,100B diet
  sentinel, re-baselined to **7,400B** per that test's own
  documented pattern (measured value + ~130B headroom, dated
  comment in `tests/test_compact_feeds.py`), still far under the
  16,384B hard budget.
- Tests: `tests/test_regime_data.py` (9 tests — availability
  convention ×4, change definition, filter boundaries, verdict
  mechanics, real-data census). **Full suite: 165 passed** (156
  baseline + 9).
- Causality/integrity: the four availability tests above; the
  mirror-vs-nightly champion cross-check (diff 0.0000); the
  unfiltered-base lineage grade landing exactly on the size
  study's independent current-code reproduction (−3,329.80);
  fee identity gross = net + fees holds exactly in every duel
  table. Champion mtimes unchanged end-to-end; all grading on
  scratch DBs in /tmp.

## Deviations from the plan

**None in design, data selection, features, lags, bands, arms,
windows, or rules.** Build/lineage notes, disclosed rather than
silent:

- The plan's coverage census counted in-span CSV *rows*; the
  operative stamp counts exclude fredgraph's blank-valued US
  holiday rows (see Data). A description correction, not a design
  change — the usable-series construction is exactly the plan's.
- Arm 3's standard-window base signal is run 22's **cached**
  probabilities (`data/gbm_focus/probs_B.npz`, bit-identical to
  its recorded grading) — the plan's "recorded Focus-B-style
  signal" taken literally, with stronger provenance than a refit;
  its robustness base is a fresh refit on pre-region rows, as the
  plan specifies.
- The feed-size sentinel re-baseline (above) follows the
  sentinel's own documented maintenance pattern, as the plan
  anticipated.

## What this closes, and what's next

Option B is answered: genuinely new daily information does not
lift the harness. The regime features added a little gross on one
window set and took it back with fees at home and losses on the
robustness set; the extreme-band veto is a fair principle that
this market span barely engaged. Combined with seven prior nulls
— across features, algorithms, composites, exits, sizing, and
timeframe — the evidence now covers the information axis too.
Per Kass's order (A → B → C), the remaining drawing-board item is
**C — the two-step hierarchy** (trade-or-not → direction).

Machinery: `src/models/regime_data.py` · Driver:
`scripts/build_regime_data.py` · Raw records:
`data/regime_data/results.json` · Raw data: `data/regime/`.
