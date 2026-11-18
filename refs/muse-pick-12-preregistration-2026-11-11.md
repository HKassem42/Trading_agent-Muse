# Muse's Pick #12 — HA Suite (Flat Recipe on HA Candles + HA Indicators) — Pre-registration

**Frozen:** 2026-11-11, BEFORE any Pick #12 fit. Ordered by Kass
2026-11-11 11:29 ET ("Go ahead and run it and if they fail
let's just leave our current champ to keep retraining"),
following the run 123 crowning and Muse's Pick #12 proposal.

## Why this study exists

Pick #11 (run 110, PASS, Champion #6) proved the flat Pick
recipe on the 10 Heikin-Ashi candle values. Its nightly line
has since produced Champions #7–#12, but the wins arrive at
one-candle margins — the same recipe re-rolled on new data.
What has never been tested is the proven recipe on NEW
INFORMATION: Kass's own 2026-10-30 design called for a full
indicator suite computed ON the Heikin-Ashi candles (the
2026-10-30 HA studies tested trading RULES on that idea and
failed; the flat stacking recipe has never run on it).

Pick #12 = the exact Pick recipe (`build_pick_pipeline`,
seed 42) on **19 columns**: the 10 HA candle values
(`build_ha_matrix`, unchanged) PLUS 9 suite columns computed
on the HA candles with production's own transforms
(`src/models/muse_pick12.py`): ha_rsi_c, ha_macd_hist_rel,
ha_ewo, ha_obv_slope, ha_adl_slope, ha_netvol_rel,
ha_atr_rel, ha_dist_sma50, ha_volatility_21. Shared transforms
reuse production's FROZEN scale constants (FEATURE_SCALES,
fit to raw candles in Oct 2026) — no scaling is fit to this
study's data. Volume-flow columns use real volume; OBV signs
by HA-close-to-HA-close moves, ADL by the HA candle's money
flow multiplier, net volume by the HA bar's own direction —
the production conventions carried onto HA bars.

## The exam (identical, again)

Pick #9's saved frame: `data/muse_pick9/raw_ohlcv.pkl`
(17,510 raw rows) and `prod_sup.pkl` (17,050 supervised rows);
the exam is the last 792 supervised rows from 2026-10-03
19:00 UTC, split at row 16,258, exam majority **0.5088**.
Target: next-1h direction. Threshold 0.50.

**Frame-identity check (setup stage, before any candidate
fit):** the ha10 block rebuilt from the saved raw candles
must equal `data/muse_pick10/ha_matrix.pkl` exactly, and
Pick #11's own artifact (run 110) re-scored on the exam must
reproduce **0.5341** exactly. Any deviation aborts the study.

**Champion re-scores (setup stage, before any candidate
fit):** the sitting champion run 123 and former champion
run 119 (both ha10 models) are re-scored on the identical
exam rows and recorded in the study meta. Tonight's-crowned
numbers (0.5568 / 0.5530) come from fresher exams and are NOT
the bar — the bar is what those same models score HERE.

## The bar (frozen)

Registered **PENDING KASS (WOULD PROMOTE)** only if ALL hold:

1. exam acc > run 123 re-scored on this exam, AND
2. exam acc > 0.5341 (Pick #11's print on this exam — the
   plain-ha10 frame the suite claims to improve on), AND
3. exam acc > exam majority (0.5088), AND
4. clean anatomy (up-calls ≥ 8 AND precision > 0.05), AND
5. purged walk-forward (5 folds, 24-row embargo, training
   region) mean acc > walk-forward majority.

Otherwise **FAIL**.

## Kass's standing instruction for this study

On FAIL: leave the current champion (run 123) to keep
retraining — the nightly HA line, all seats, and all books
stay exactly as they are. On PASS: the run records as
PENDING KASS under the promotion lock; nothing seats itself
and any nightly adoption is a separate Kass decision.

## Guards and caveats

Champion / seat / archive artifact hashes captured before
the fits and re-checked before registration; any change
aborts registration. Lab only. Selection-pressure caveat,
stated in advance: this exam family has graded Picks #9,
#10, #11 and many champion re-scores — a pass is evidence
for Kass's ruling, not proof of edge.
