# Pre-registered test plan — Drawing-Board Option B: daily regime data

Date: 2026-10-24 · Status: **PLAN — written before any training or
grading.** Kass's order (2026-10-24 ~01:15 ET): "start with A then B
then C" — A (4h timeframe) completed NULL (runs 50–51); B is next.
This file is the contract: the rules below are fixed before any
result exists, and any forced deviation is reported as a deviation
in the final report, never quietly absorbed.

## The question

Every prior test — TA stacker, PPO, HA-Seq, the 132-indicator A-Z
screen, GBM focus/lag, the v3 composite screen, the 4h twins —
rearranged the **same 1h candle information** and returned null
(seven independent nulls). This test changes the information itself:
**daily regime/context series** — a genuinely new information type —
used as context on the 1h frame, as features and as a filter. Not as
standalone entry signals.

## Data (preparation census — fetched 2026-10-24, before this plan
was finalized; no model fitted, nothing graded)

Raw CSVs saved under `data/regime/`. Matrix span to cover:
2024-11-23 → 2026-10-21 (the v1 1h matrix, 16,735 hourly rows).

| Series | Source | Span fetched | Stamps in matrix span | Decision |
|---|---|---|---|---|
| Fear & Greed Index | Alternative.me API (free, no key) | 2018-02-01 → 2026-10-24 (3,184 rows) | 698 / 698 calendar days | **KEEP** |
| VIXCLS (VIX close) | FRED fredgraph (no key) | 1990-01-02 → 2026-10-22 | 498 (business days), last stamp 2026-10-21 | **KEEP** |
| DTWEXBGS (broad USD index) | FRED fredgraph | 2006-01-02 → 2026-10-16 | 495 (business days) | **KEEP** — the source publishes this series with a lag; its freshest stamp is 2026-10-16, so the last matrix days read that stamp (usable from 10-18). Every matrix day still receives a usable value; the end-of-span staleness (≤5 days on this one series) is disclosed here, not discovered later. |
| T10Y2Y (10y–2y slope) | FRED fredgraph | 1976-06-01 → 2026-10-23 | 498 (business days), last stamp 2026-10-21 | **KEEP** |
| DFF (fed funds effective) | FRED fredgraph | 1954-07-01 → 2026-10-22 | 698 (daily), last stamp 2026-10-21 | **KEEP** |

No series is dropped; no substitute may be added after results are
seen. No missing-value (".") markers occur inside the matrix span
in any FRED series. Fear & Greed's only gaps are 2018-04-13→17 and
2024-10-25→27 (pre-span); in-span it is complete (698/698 days).

**Disclosed limitation (FRED revisions):** fredgraph serves
*current, revised* values. First-published (vintage) values are not
retrievable from this endpoint, so the features below are computed
from revised history. The D+2 availability lag mitigates the
*timing* leak (a value could not have been known before D+2 even
unrevised); it does **not** remove revision look-ahead for series
whose past values were later restated. This is a known, bounded
contamination, stated here and repeated in the report — it is one
more reason a marginal "pass" would need skepticism, and it cannot
manufacture a null into a pass in the flattering direction for the
*filter* arm's source-native F&G series (Alternative.me values are
published once and not revised).

## Availability convention — the central causality discipline

A daily value stamped day **D** becomes usable only from:

- **Fear & Greed: D+1 00:00 UTC** (the index for day D is computed
  from day D's market action and published after the day closes);
- **FRED series: D+2 00:00 UTC** (publication lag plus a safety day
  for revisions/scheduling).

Formally: `usable(date d) = value of the freshest stamp s with
s + lag ≤ d`. An hourly row at time t takes the usable value of its
UTC calendar date. Forward-fill only, ever — across calendar days
of the *usable* series (weekends/holidays read the freshest usable
stamp); never backward-fill, never interpolate.

Pinned by tests (`tests/test_regime_data.py`):

1. **Stamp exactness** — on a synthetic series whose value is its
   stamp's day ordinal, the feature at every hour equals the ordinal
   of exactly `date(t) − lag` (freshest-stamp form for gapped
   series), including the midnight boundary (23:00 reads the old
   value, 00:00 the new).
2. **Truncation invariance** — features rebuilt from a daily frame
   truncated at day T0 are identical to the full-frame features for
   every hour with `date(t) ≤ T0 + lag`.
3. **Extra-lag staleness identity** — features computed with
   `lag + 1` at hour t equal features with `lag` at hour t − 24h:
   an extra day of lag makes values staler and changes nothing
   else.
4. **Change definition** — `chg5(d) = usable(d) − usable(d − 5
   calendar days)`, verified on fixtures.

## Features (fixed; ≤ 2 per series, 10 total)

For each kept series: the **usable level** and its **5-calendar-day
change** (regime direction), mapped onto every hourly row:

`fng_level, fng_chg5, vix_level, vix_chg5, usd_level, usd_chg5,
t10y2y_level, t10y2y_chg5, dff_level, dff_chg5`

(`usd` = DTWEXBGS.) Computed on the full daily history, then mapped
to the matrix span — zero NaNs permitted inside the span (census
checked in the build; a NaN aborts).

## Arms (fixed)

All LightGBM multiclass with `models.gbm_focus.LGBM_PARAMS`
**verbatim** and **balanced class weights** (inverse class
frequency — the focus test's arm-B treatment), fit through
gbm_focus's own fold/final helpers. Duel policy: the pre-registered
conviction rule of `models.gbm_policy` (LONG at P(BUY) ≥ 0.40 ≥
P(SHORT); SHORT at P(SHORT) ≥ 0.40 > P(BUY); else the flat band),
unchanged from runs 21–27. Focus-11 = gbm_focus.FOCUS_FEATURES on
the v1 matrix, exactly as runs 21–24.

1. **GBM-Focus+Regime** — focus-11 **plus** the 10 regime features
   (21 features). The question: does new information lift the known
   harness? Walk-forward + duel + robustness. Its walk-forward is
   compared directly to Focus B's recorded WF (run 22: accuracy
   0.5508, balanced 0.4145).
2. **GBM-Regime-Only** (diagnostic) — the 10 regime features alone:
   does the new information carry anything by itself?
   **Walk-forward only** (never duel-graded, so a lucky diagnostic
   cannot be cherry-picked into a trade claim — the composite
   screen's family-diagnostic discipline).
3. **F&G Filter** — the Focus-B-style signal (focus-11, arm-B fit,
   no regime features — the same fits that produce arm 1's
   comparison baseline), with the conviction **vetoed by regime**
   before it reaches the duel:
   - a LONG conviction is zeroed whenever the usable F&G level at
     that hour is **< 25**;
   - a SHORT conviction is zeroed whenever the usable F&G level is
     **> 75**;
   - all other convictions pass through with magnitude unchanged.
   Principle, fixed here: the veto lines are Alternative.me's own
   published extreme bands (0–24 Extreme Fear, 76–100 Extreme
   Greed) — *do not buy into extreme fear; do not short into
   extreme greed.* Semantics in this harness, stated plainly: a
   zeroed conviction reads as "signal gone" to the position-aware
   wrapper, so a vetoed direction cannot be entered **and** an
   open position in that direction is flattened by the standard
   exit mapping. Duel + robustness only (no walk-forward of its
   own; its signal's WF reference is Focus B's recorded WF above).

## Walk-forward (fixed; arms 1–2)

TimeSeriesSplit, 5 folds, over the **15,961 pre-duel rows** only
(duel region starts 2026-09-19 11:00 UTC; 13,300 OOF graded rows).
Report pooled accuracy **next to** the majority baseline of the
graded rows (0.6690 in every prior 1h study) and balanced accuracy
next to 1/3 chance, plus the per-fold spread.

## Duel (fixed; arms 1 and 3)

Signal-policy broker harness — the mirror of
`training.nightly.evaluate_ppo` used by the GBM/composite/4h
studies (`models.composite_screen.evaluate_windows_composite`:
StudyBroker/StudyEnv, default `ExitConfig` = the verified
production exit recipe — through-flat transitions, 2×ATR stop with
the 2.5% floor, trailing as designed, no take-profit, stop-anchor
ratchet), scratch ledger DBs in /tmp, $100,000 episodes,
next-candle-open fills, 25% tranches, 75% cap, 0.15% fees.

- **Windows:** the standard 3×264h set (from 2026-09-19 11:00 UTC)
  and the **robustness set** (the preceding 792h, from 2026-08-17
  11:00 UTC) — the same calendar spans as every 1h study.
- **Fits:** standard windows — final models fit on all 15,961
  pre-duel rows; robustness — **refit on rows strictly before the
  robustness region** (out-of-sample there).
- **Fee transparency:** net AND gross (pre-fee) P&L, total fees,
  round trips and exact non-hold activity share for every duel.
- **Yardsticks:** buy-and-hold on the same windows, and the
  **current champion graded fresh** through the same harness
  (run 44's PPO brain, `models/ppo_champion.zip`, installed
  2026-10-24 by Kass's manual override). The champion is a
  reporting yardstick; the verdict below is absolute (the tasking
  fixed it so). Before any arm number is trusted, the mirror is
  cross-checked against `nightly.evaluate_windows` on the champion
  and must agree (agg/P&L diff < 0.01) — the composite screen's
  proof step.

## Pre-registered verdict rules

- **Arm 1 (Focus+Regime)** — all three or NULL:
  1. Walk-forward: pooled balanced accuracy > 1/3 AND at least 4
     of 5 folds' balanced accuracy > 1/3 AND pooled accuracy ≥
     the majority baseline of the graded rows.
  2. Duel: total P&L > $0 AND total P&L > buy-and-hold AND
     non-hold activity share ≥ 10%.
  3. Robustness: robustness-set total P&L ≥ $0.
- **Arm 2 (Regime-Only)** — rule 1 only, as a diagnostic. Even a
  rule-1 pass is recorded as "WF-only — not a tradable claim"
  (verdict string `WF ONLY` unless rule 1 fails, in which case
  `FAIL`); it can never be a full PASS without a duel it will
  never receive under this plan.
- **Arm 3 (F&G Filter)** — rules 2 and 3 only (duel + robustness),
  both or NULL.

Any failure = the arm is NULL, registered promoted=0, and the
design is **not re-tuned for a second shot**. A full pass (arm 1
or 3) is a report to Kass, not a promotion.

## Integrity checks (run with the build, results in the report)

- The four availability tests above (unit tests, synthetic
  fixtures) — all must pass before grading.
- Feature census on the real frame: 16,735 rows × 10 regime
  columns, zero NaNs in-span; F&G level within [0, 100].
- Filter identity: with F&G forced to 50 everywhere, the filtered
  conviction equals the base conviction exactly (the veto is the
  only change the filter ever makes); veto counts reported per
  region.
- Fee identity per duel: gross = net + total fees, exactly.
- Zero-conviction sanity is inherited from the shared harness
  (pinned by the composite screen's tests against the real
  broker).

## Registration

One training.db run per arm, `family="regime_data"`, promoted=0
always, with duel + walk-forward metrics and the rule-by-rule
verdict in the record; family code 9 in the dashboards' models
index ("Regime Data — Focus+Regime" / "Regime Data — Regime Only"
 / "Regime Data — F&G Filter"); one lab note. The training-feed
size sentinel is respected: if the 3 new index rows trip the
7,100B diet sentinel, it is re-baselined per that test's own
documented pattern (measured value + headroom, dated comment) —
the same conscious act the size study performed — and disclosed
in the report.

## What this plan is not

- Not a production change: the live bot (un-parked 2026-10-24 on
  Kass's override, run-44 champion) is untouched by this lab
  study; nothing here promotes, whatever the outcome.
- Not a search over regime sources or bands: the five series, the
  two-feature-per-series cap, the D+1/D+2 lags, and the 25/75
  veto lines are fixed above. A null closes Option B; it does not
  license trying other bands, lags, or series next without Kass's
  direction (Option C — the two-step hierarchy — is his stated
  next item).
