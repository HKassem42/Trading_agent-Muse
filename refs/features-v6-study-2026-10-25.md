# Curated Features v6 — Lab Study (2026-10-25)

Kass green-lit a curated wider feature set as a registered lab family
(`features_v6`, family code 11), graded through the production gate
under the promotion lock. Full machine record:
`data/features_v6/results.json`; builder: `scripts/build_features_v6.py`;
feature module (lab only): `src/features/features_v6.py`.

## Construction

- Start: production 21 (`FEATURE_COLUMNS`). **Dropped `obv_slope`** —
  the known duplicate of `netvol_rel` (r = 0.999, Track A diagnosis
  2026-10-21). `netvol_rel` kept: it is the bounded share by
  construction; `obv_slope` is the rescaled near-copy.
- Additions: AZ-screen candidates whose screen max |r| vs the 21 was
  < 0.90, in family-diverse priority order, greedily re-checked on the
  study's own train frame: max |r| vs the retained 20 AND every
  already-accepted addition < 0.90. Cap 16 additions.
- **Accepted (16) — train max |r| (vs) / screen max_corr_21:**
  az_er_10 .274 (volatility_21) / .276 · az_adx_14 .497 (volatility_21)
  / .500 · az_supertrend_dir .658 (ewo) / .657 · az_vhf_28 .514
  (az_adx_14) / .445 · az_aroon_down .585 (rsi_c) / .583 · az_cci_20
  .384 (volatility_21) / .383 · az_dpo_20 .578 (dist_sma50) / .577 ·
  az_stc .501 (rsi_c) / .502 · az_fisher_9 .753 (rsi_c) / .753 ·
  az_chop_14 .580 (az_er_10) / .486 · az_mass_index .413 (volatility_21)
  / .408 · az_chaikin_vol .430 (range_pct) / .429 · az_ulcer_14 .679
  (ewo) / .669 · az_vroc_14 .622 (az_chaikin_vol) / .483 ·
  az_klinger_diff .693 (macd_hist_rel) / .691 · az_mfi_14 .778 (rsi_c)
  / .780.
- Not reached (cap): az_ret_kurt_20, az_acf_1_50, az_pv_corr_20,
  az_ibs + all 8 fallbacks. No candidate was rejected on correlation.
- **Total: 36 features.** All within [-1, 1]; every addition built by
  the AZ registry's own causal builders with train-fitted scalers.

## Causality validation (v5.1 standard, tests/test_features_v6.py)

10 tests, all passing: retained-set composition; every candidate tied
to the screen record with max_corr_21 < 0.90 and present in the AZ
registry; frame contract (finite, [-1,1]); added columns equal
`az_indicators.compute_az` two-ways (atol 1e-12); **truncation test
across all rows** (K ∈ {1000, 1200, 1400}); **future-perturbation
test** (scrambling all candles after row 1200 leaves values ≤ 1200
unchanged); env/PPO threading incl. production-default invariance.

## Training & grading (same windows, same rules as production)

Data: 17,510 hourly bars (Coinbase, 730d); holdout = last 792 sup rows
from 2026-09-22 20:00 UTC; 3 windows × 264; next-candle-open fills;
score = P&L − 0.5 × maxDD; 10 episodes/window.

- **Stacker** (run 61): v6 holdout accuracy **0.4861** (walk-forward
  0.4979, holdout precision 0.4633) vs champion stacker **0.4811** on
  the identical holdout timestamps → strict win by the production
  rule. Under the promotion lock: recorded **PENDING KASS APPROVAL**,
  promoted=0, champion stacker untouched.
- **PPO** (run 62): fresh 60,000 steps on the 36 features (no
  warm-start exists across feature versions). Duel: challenger window
  scores **−1,179.7 / −1,428.7 / −435.2**, aggregate **−1,014.52**,
  P&L **−$151.14**, maxDD $2,235.35, mix hold .59 / short .36 / buy
  .02 / sell .03 — vs champion (run 44 lineage) windows
  +211.0 / −114.5 / −255.5, aggregate **−52.98**, P&L **+$160.05**,
  maxDD $303.52. Buy-and-hold over the windows: −$6,735.24.
  **Won 0/3 windows → formula FAIL; the floor was never engaged.**
  Cross-checks inside the builder: per-window buy-and-hold identical
  between the two gradings and recomputed from raw closes to < 1e-6;
  aggregates reconcile from window records exactly.

## Verdict

Tenth independent null for the trading side: a curated, non-redundant
36-feature set did not produce a tradable PPO — it lost every window
to the incumbent champion. The one positive fragment is the stacker:
+0.005 holdout accuracy over the champion on identical timestamps —
inside noise territory for a single holdout, and it sits **pending
Kass's decision** under the lock; nothing was installed. Champion
files verified hash-identical before/after the study; the paper book
was never opened by the study (the live loop's own hourly writes
during the window are its normal operation).
