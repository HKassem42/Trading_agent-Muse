# Muse's Pick #11 — Flat Recipe on HA Candles — Pre-registration

**Frozen:** 2026-11-05, BEFORE any Pick #11 fit. Ordered by Kass
2026-11-05 14:02 ET ("Run Pick #11: the flat recipe on HA
candles as the candidate, with its own walk-forward"),
following the Pick #10 report.

## Why this study exists

In Pick #10 (run 109, FAIL), the control arm — the standard
Pick recipe (scaler → SMOTE → RF + XGB + LGBM + ExtraTrees → LR
meta, seed 42) fitted on the 10-value Heikin-Ashi candle matrix,
read flat one row at a time — printed exam accuracy **0.5341**
on Picks #9/#10's exam, above both champions re-scored on that
exam (run 96: 0.5215, run 73: 0.5189) and near the clean-era
best (0.5354). It was a CONTROL: never the candidate, no
walk-forward of its own. Picks #7 and #8 both taught the same
lesson — a control's exam print, untested by walk-forward,
cannot carry a conclusion.

Pick #11 promotes that exact configuration to CANDIDATE on the
**identical exam** (Pick #9's saved frame; last 792 supervised
rows from 2026-10-03 19:00 UTC, majority 0.5088; champions'
re-scores reused — literally the same rows, two-way verified in
Pick #9) and gives it the missing grading: a purged
walk-forward on the candidate itself.

## Frozen design

- **Candidate:** `build_pick_pipeline(random_state=42)` on the
  10 HA columns exactly as built for Pick #10
  (`data/muse_pick10/ha_matrix.pkl`; builder causality-tested
  there). Target: next-1h direction. Threshold 0.50.
- **Determinism expectation:** the exam fit is the same
  computation as Pick #10's flat control, so its exam accuracy
  is expected to reproduce 0.5341 exactly; any deviation is
  reported, not hidden. The walk-forward is new evidence
  either way.
- **Walk-forward:** `purged_wf_splits`, 5 folds, 24-row
  embargo, over the training region — the Picks #9/#10
  convention.
- **Anatomy:** Pick #10's control anatomy is on record in its
  exam.json; the candidate's own anatomy is recomputed and
  judged by the same run-63 rule.

## The bar (frozen)

Registered **PENDING KASS (WOULD PROMOTE)** only if ALL hold:

1. exam acc > run 96 re-scored (0.5215), AND
2. exam acc > run 73 re-scored (0.5189), AND
3. exam acc > exam majority (0.5088), AND
4. clean anatomy (up-calls ≥ 8 AND precision > 0.05), AND
5. purged walk-forward mean acc > walk-forward majority.

Otherwise **FAIL**. If PENDING and exam acc also exceeds
**0.5354**, the record notes BEAT THE BAR — the clean-era best.

A pass would make Pick #11 a candidate for Kass's ruling under
the promotion lock — nothing seats itself. Note for the record:
the HA matrix's 10 columns are a *different feature set* from
the production 21, so a crowned Pick #11 would be a lab stacker
champion on its own frame (the registry already carries
per-frame champions: prod21 / scored102 / pick2_98); any live
use would be a separate Kass decision.

## Guards

Champion / live-seat / shadow-seat / gated-seat / run-73
artifact hashes captured before the fits and re-checked before
registration; any change aborts registration. Lab only.
