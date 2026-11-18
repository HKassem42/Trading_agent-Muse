# Muse's Pick #4 / #5 / #6 — Pre-registration (2026-11-04)

Kass's order (2026-11-04 15:04 ET, "Select best new model" side
chat): "Name the 3 models you did, Muse's Pick #4 #5 #6. Then go
ahead and do what you need to and build a model that will beat my
champions. Change whatever you need in your set up and get it
right... Try your hardest and train the models however you want."

Frozen BEFORE any candidate is fitted or graded. Lab only under
the promotion lock: a win is recorded PENDING KASS APPROVAL;
nothing is crowned, seated, or installed by this study.

## The bar

Shared exam: the Next Try study's frame — BTC/USD 1h, supervised
rows 2024-11-04 → 2026-11-04, holdout = last 792 candles from
2026-10-02 06:00 UTC, train = the 12,045 rows finite on the full
frame before it. Champions re-scored on these identical
timestamps during the Next Try study, two independent ways, exact
agreement:

- Run 96 (Lab Stacker Champion #5): holdout acc **0.5265**
- Run 73 (Muse's Pick #2): holdout acc **0.5227**
- Majority baseline on this exam: 0.5101
- All-time clean-era best holdout accuracy: **0.5354** (runs
  73/78, their original exams)

Verdicts, per pick:
- **BEAT THE BAR**: holdout acc > 0.5354 with clean anatomy
  (up-calls ≥ 8 and precision > 0.05 — the run-63 rule).
- **PENDING KASS**: holdout acc > 0.5265 (beats both champions
  re-scored on this exam) with clean anatomy →
  pending_approval = true, champion kept. (Same treatment the
  nightly's run 100 received at 0.5341 on its own exam.)
- Otherwise FAIL. All three verdicts reported exactly as they
  land, with walk-forward and window accuracies on the record.

## Common protocol

- Eyes for all three picks: Pick #2's 98 scored columns (the
  run-73 frame: 94 scored + Hurst/skew/kurtosis/autocorr),
  extracted from data/muse_next_try/frame120.pkl — the frame
  whose causality was truncation-tested in the Next Try study.
- Threshold fixed at 0.50. No threshold tuning (Try #1's lesson).
- No frame/knob selection on the holdout. The only train-side
  choices are the frozen constants below.
- Walk-forward reported as 3-fold TimeSeriesSplit on train
  (labeled wf3; the verdict rides on the holdout, WF is the
  honesty check).
- Holdout grades computed twice: once in the build, once by an
  independent verifier re-reading the saved probabilities.
- Champion/seat/shadow/gated files hash-guarded before/after.

## Pick #4 — "Three Clocks" (multi-horizon ensemble)

Lever: the target's time structure. Every prior stacker learned
one clock — the next candle. Three Pick-recipe pipelines are
fitted on the same 98 eyes with three causal label horizons:
next-candle direction (y1, the standard target), 4-hour
direction (close[t+4] > close[t]), and 24-hour direction
(close[t+24] > close[t]). The vote is the equal-weight average
of the three heads' P(up) — frozen weights, no fitting of the
blend. Mechanism: the slower heads regularize the vote toward
structure that persists beyond one candle's noise. The 24h head
trains only on rows where t+24 exists. Graded on y1, the
champions' own exam — same target, same rows, same bar.
(Distinct from the closed hierarchy study, which gated a 1h
model by a separately-predicted regime, and from gbm_lag, which
added snapshot lag columns to a plain GBM.)

## Pick #5 — "Fresh Eyes" (recency-weighted bagged vote)

Lever: which rows matter, and variance. BTC regimes drift;
every champion weights a 500-day-old candle the same as
yesterday's and trusts a single seed. Design: no SMOTE (the
target is ~49/51 — SMOTE is ceremonial at this balance);
StandardScaler; the Pick recipe's four tree learners
(RF/XGB/LGBM/ET, champion hyperparameters) fitted at 5 seeds
(42–46) = 20 learners; every learner fitted with sample weights
w = 0.5^(age_days/120) — exponential recency decay, half-life
frozen at 120 days; the vote is the plain average of all 20
learners' probabilities. No stacking meta-learner: the levers
under test are recency and seed-bagging, nothing else.

## Pick #6 — "Wide Net, Clean Labels" (diversity stack)

Levers: learner diversity + label quality, bundled as one
training-side design. (a) Label cleaning: drop training rows
whose |next-candle return| falls in the bottom quartile of the
TRAIN rows' |next-candle return| (threshold computed on train
only; inside WF folds, on that fold's train part only) —
near-zero moves are label noise. (b) A wider stack:
StackingClassifier (cv=3, LR meta) over seven base learners —
the champion four trees plus kNN (k=40), an MLP (48,24;
max_iter 250), and logistic regression — on StandardScaler, no
SMOTE (kNN/MLP need the scaled space undistorted). Predicts on
all holdout rows regardless of their move size.

## Registration

Families muse_pick4 / muse_pick5 / muse_pick6 (registry runs
104/105/106 in grading order), display names "Muse's Pick #4",
"Muse's Pick #5", "Muse's Pick #6". compact.py gains family
codes 24/25/26 by the established pattern so the Training Lab
feed cannot KeyError on them.
