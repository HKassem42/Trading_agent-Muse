# Full-Indicator Stacker — Lab Study (2026-10-25)

Kass's direction: "incorporate all those sub indicators and main
indicators into the stacker and let it make those decisions based on
those indicators." The production stacker recipe (StandardScaler →
SMOTE → RF + XGBoost + LightGBM → LogisticRegression, production
target: 1 if the next candle closes higher) trained on the full
computable indicator universe, graded against the current 21-feature
champion stacker on the same exam. Lab family
`full_indicator_stacker` (code 12), runs 63–64. Full machine record:
`data/full_indicator_stacker/results.json`; builder:
`scripts/build_full_indicator_stacker.py`. PPO was NOT retrained;
nothing installed (promotion lock).

## Construction

- Fresh BTC/USD 1h frame (Coinbase, 730d): 17,510 bars,
  2024-10-25 23:00 UTC → 2026-10-25 22:00 UTC; supervised 17,054
  rows; holdout = last 792 rows from 2026-09-22 23:00 UTC — the
  nightly split, train/holdout disjointness asserted. (One day
  fresher than the v6 study's holdout; the champion was re-scored on
  these exact timestamps, so all comparisons are same-exam.)
- Feature construction reuses the matrix builders' own code paths
  (`build_timeseries_matrix_v3.build_series` = v1 AZ builders +
  `main_chart.build_all`; `ranked_quantity` / `rolling_scores` for
  the scores) — no re-implementation.
- **Arm A — Raw 98 (run 63):** the matrix-v2 set: 59 raw AZ series +
  31 main-chart overlays = **90 features**, natural units. Train
  rows 16,262 (no warmup drops beyond production's own trim).
- **Arm B — Scored 102 (run 64):** the matrix-v3 set: the same 90
  series as trailing-4,320-bar percentile scores (0–100, oriented
  per the bullish/bearish guide; composites = family means over
  trend 45 / momentum 25 / volume 7 members, master = mean of the
  three) + 4 composites = **94 features**. Scores exist from bar
  4,632 (2025-05-06); train rows 12,086 (4,176 warmup rows dropped
  — the honest cost of the 180-day percentile window).

## Causality (re-verified on the fresh build, v5.1 standard)

- Raw arm: full rebuilds on frames truncated at bar 9,000 and at
  bar 17,018 (inside the holdout) — 6 sampled columns (RSI, Connors
  RSI, OBV, Distance to SMA 200 %, SuperTrend Side, Distance to
  Fractal Resistance %) match the full-frame values on **every**
  shared row (atol 1e-9). PASS.
- Scored arm: percentile scores checked two ways — brute-force
  numpy counts vs the sliding-window implementation, exact on
  sampled rows; truncation through the full scoring path exact
  (0.0 max diff) on 3 columns × 12,387 rows. All scores and
  composites within [0, 100]. PASS. No column failed; none dropped.

## Grades (identical holdout timestamps)

| model | holdout acc | precision | walk-forward acc | pred "up" |
|---|---|---|---|---|
| Champion stacker (21 feats, re-scored) | 0.4823 | 0.4719 | (stored 0.4798-era) | 392/792 |
| Arm A — Raw 98 (90 feats) | **0.5101** | 0.0000 | 0.4934 | **0/792** |
| Arm B — Scored 102 (94 feats) | **0.4975** | 0.4854 | **0.5121** | 342/792 |

Holdout up-rate 0.4899 → the always-"not up" majority baseline is
exactly 0.5101.

- **Arm A is a degenerate win.** By the production rule (strict
  holdout-accuracy win) it beat the champion, +0.0278 — and it is
  recorded that way. But the anatomy: the fitted meta-learner's
  coefficients on all three base learners are negative (RF −0.07,
  XGB −0.47, LGBM −0.12), the stack's P(up) never reaches 0.5
  (max 0.483), and it called "up" **zero** times in 792 candles —
  its 0.5101 is the majority-class rate to the fourth decimal.
  Ironically its base learners were the strongest raw graders in
  the study (RF 0.5316 on 253 up-calls, XGB 0.5177, LGBM 0.4975 —
  each above the champion), but the stack built on them collapsed
  to a constant answer, and its walk-forward (0.4934) is the
  weakest of the three full pipelines.
- **Arm B is a genuine, small win.** A real classifier (342
  up-calls, precision 0.4854 — the best precision of the three),
  +0.0152 over the champion on the holdout, and the best
  walk-forward in the study (0.5121 vs Arm A 0.4934; the v6
  stacker's WF was 0.4979). Base learners 0.5025 / 0.5177 / 0.5253,
  all above the champion; meta leans on RF (+1.29). Both margins
  remain single-holdout sized — the WF ordering (B > champion-era
  marks > A) is the more durable signal in this study.

## What the stacker leans on (top features, mean importance)

Importance is strikingly flat in both arms — the top feature
carries ~0.019 against a uniform-share 0.011 — no indicator
dominates; the vote is a thin spread over short-horizon oscillators.

- Arm A: Williams %R .0194, Stochastic RSI .0188, Ultimate
  Oscillator .0188, Connors RSI .0185, Acceleration Oscillator
  .0159, ASI .0158, Chaikin Money Flow .0157, Historical Volatility
  .0156, Force Index .0154, Distance to Fractal Resistance .0153,
  Distance to $5,000 Round Number .0152, Mass Index .0151,
  Bollinger %B .0149, Distance to Fractal Support .0149, Choppiness
  .0148.
- Arm B: Stochastic RSI (score) .0188, Connors RSI (score) .0163,
  Distance to $5,000 Round Number (score) .0161, Force Index
  (score) .0154, Williams %R (score) .0150, Mass Index .0149,
  Choppiness .0149, RVI .0148, ADR .0138, ASI .0138, Acceleration
  Oscillator .0134, CMF .0132, DPO .0131, ADX .0131, Klinger VO
  .0130.

## Redundancy (train frame, |r| pairs)

| set | pairs >0.90 | pairs >0.95 | clusters @0.90 | largest | cols with a >0.90 twin |
|---|---|---|---|---|---|
| Production 21 | 5 / 210 | 3 | 3 | 4 | 8 / 21 |
| Arm A (90 raw) | 145 / 4,005 | 103 | 10 | 15 | 45 / 90 |
| Arm B (94 scored) | 128 / 4,371 | 44 | 13 | 33 | 57 / 94 |

Half of Arm A's columns have a near-twin; Arm B's percentile
transform breaks the tightest duplicates (>0.95 pairs fall from
103 to 44) but concentrates correlation into one 33-member
trend cluster.

## Verdict

Both arms are registered as strict-rule wins and sit **PENDING
KASS APPROVAL** under the promotion lock; the dashboards' gate
surfaces run 64 (the newest pending), and both detail rows carry
their records. Nothing was installed; champion files verified
hash-identical before/after; the paper book was never opened by
the study. Read plainly: the full indicator universe did not
produce a decisive stacker — Arm A's win is majority-class
collapse wearing a crown, Arm B's is a real but small edge
(+0.015 holdout, best-in-study walk-forward). Whether either is
worth productionizing is Kass's call under the lock; note the live
PPO consumes the stacker's P(up), so swapping the stacker changes
a live input — the same contamination consideration that governed
run 61's decline applies.
