# Muse's Pick #8 — "Plain 21, Slow Clock" — Pre-registration

**Frozen:** 2026-11-05, BEFORE any Pick #8 fit. Ordered by Kass
2026-11-05 13:09 ET ("Run Pick #8: test the plain 21 features at
24h as the candidate"), following the Pick #7 report.

## Why this study exists

In Pick #7 (run 106, FAIL), the control arm C1 — the standard
Pick recipe (scaler → SMOTE → RF + XGB + LGBM + ExtraTrees → LR
meta, seed 42) fitted on the plain 21 production features alone,
target = next-24h direction — printed exam accuracy **0.5612**
on the 768-row 24h exam, the highest clean number in the
program. It was a CONTROL: never the candidate, no walk-forward
of its own, and the 24h exam's labels overlap 23/24 (effective
independent rows ≈ 32), so the number cannot carry a conclusion.

Pick #8 promotes that exact configuration to CANDIDATE and gives
it the full grading Pick #7's candidate got: the same exam, plus
a purged, embargoed walk-forward on the candidate itself.

## Frozen design

- **Candidate:** `build_pick_pipeline(random_state=42)` on the
  canonical 21 production feature columns ONLY (from
  `features.technical_indicators.FEATURE_COLUMNS`, read off
  `data/muse_next_try/prod_sup.pkl` — the exact source Pick #7's
  C1 control used).
- **Target:** next-24h direction, `horizon_labels(close, 24)`.
- **Exam:** IDENTICAL to Pick #7's — the standard 792-row
  holdout window restricted to rows with a defined 24h label
  (768 rows from 2026-10-02 06:00 UTC), majority baseline
  0.5091. The build machine-checks its split against Pick #7's
  saved meta (train 12,021 after the 24-row purge, exam 768,
  majority equal) and refuses to proceed if they differ.
- **Split discipline:** `exam_split` with a 24-row boundary
  purge; walk-forward = `purged_wf_splits`, 3 folds, 24-row
  embargoes — the Pick #7 machinery, unchanged, already pinned
  by `tests/test_muse_pick7.py`.
- **Threshold** fixed 0.50.
- **Determinism expectation:** the candidate fit is the same
  computation as Pick #7's C1 control, so its exam accuracy is
  expected to reproduce 0.5611979166666666 exactly. Any
  deviation is reported in the results and lab note, not hidden;
  the walk-forward is new evidence either way.

## The bar (frozen)

Registered **PENDING KASS (WOULD PROMOTE)** only if ALL hold:

1. exam acc > **0.5404** — Pick #7's registered candidate on
   this identical exam (the same-exam bar), AND
2. exam acc > exam majority **0.5091**, AND
3. clean anatomy (run-63 rule: up-calls ≥ 8 AND precision
   > 0.05), AND
4. purged walk-forward mean acc > walk-forward majority.

Otherwise registered **FAIL**. If PENDING and exam acc also
exceeds 0.5354 (run 73/78's clean-era best), the record notes
BEAT THE BAR — flagged as a cross-exam reference only, since
exams differ.

## What a pass would and would not mean (frozen)

- A pass makes Pick #8 a *candidate for Kass's ruling*, nothing
  more. The promotion lock applies unchanged.
- Pick #8 is a **24h-horizon** model. The sitting champions and
  the live PPO contract are 1h. A pass does NOT make it a
  drop-in replacement for anything; seating it anywhere would
  need its own design decision by Kass.
- The exam's effective sample is ~32 independent rows. Even a
  pass is weak evidence, and the record must say so.

## Guards

Champion / live-seat / shadow-seat / gated-seat artifact hashes
are captured before the fits and re-checked before
registration; any change aborts registration. Lab only.
