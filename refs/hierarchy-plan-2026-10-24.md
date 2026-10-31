# Pre-registered test plan — Drawing-Board Option C: the two-step hierarchy

Date: 2026-10-24 · Status: **PLAN — written before any training or
grading.** Kass's order (2026-10-24 ~01:15 ET): "start with A then
B then C" — A (4h timeframe) completed NULL (runs 50–51), B (daily
regime data) completed NULL (runs 52–54); C is the last item in his
sequence. This file is the contract: the rules below are fixed
before any result exists, and any forced deviation is reported as
a deviation in the final report, never quietly absorbed.

## The question

Every flat 3-class model tried on this data drowns in HOLD: the
prior arms predicted HOLD 69–71% of the time (lag arms), the
weighted focus arm 61%, and the models spend their capacity
separating BUY from SHORT on rows where nothing is tradeable. This
is Kass's own idea: split the decision in two.

- **Stage 1 (the gate)** — binary: will the next candle move
  enough to clear the fee band (TRADE: the matrix label is BUY or
  SHORT, i.e. |next-bar return| > 0.30%) or not (NO-TRADE: label
  HOLD)?
- **Stage 2 (direction)** — binary UP/DOWN, trained **only** on
  true TRADE rows (UP = label BUY, DOWN = label SHORT).

Combined policy: NO-TRADE → HOLD; TRADE → BUY if stage 2 says UP,
SHORT if stage 2 says DOWN. The end-to-end prediction is the only
graded object for the verdict; the stages are graded as diagnostics
so the report can say *where* the hierarchy loses — at the gate or
at direction.

## Data (fixed)

The v1 BTC matrix (`data/timeseries_matrix/btc_usd_1h_matrix.csv`,
16,735 hourly rows, 2024-11-23 → 2026-10-21; labels from next-candle
return, ±0.3% fee band: BUY 2,915 / HOLD 10,976 / SHORT 2,844),
loaded exactly as runs 21–27 loaded it (`models.gbm_focus`
loader). Walk-forward and final fits use the **15,961 pre-duel
rows** only; the duel region starts **2026-09-19 11:00 UTC**; the
robustness region is the preceding 792h (from 2026-08-17 11:00
UTC). Same graded rows, same split as the focus / lag / composite /
regime studies, so results are comparable.

## Arms (fixed; LightGBM throughout)

Hyperparameters are `models.gbm_focus.LGBM_PARAMS`' shared values
verbatim (300 trees, lr 0.05, 31 leaves, min-child 40, row/col
subsample 0.9, λ = 1.0, seed 42) with the mechanically required
change for a binary stage: `objective="binary"` and no `num_class`
(the multiclass settings cannot express a binary problem; nothing
else is altered). Class weighting = inverse class frequency
(`gbm_focus.balanced_weights`), computed **within each stage's own
training rows** (stage 1 over all training rows; stage 2 over the
training rows that are true TRADE).

1. **Hierarchy-Focus** — both stages on Kass's focus-11 features
   (`gbm_focus.FOCUS_FEATURES`, exactly as runs 21–24).
2. **Hierarchy-All** — both stages on the full feature set of
   the focus test's all-features arm
   (`gbm_focus.matrix_feature_columns`: all 60 matrix feature
   columns).

## Decision rule (fixed, no tuning)

Gate fires when **P(TRADE) ≥ 0.50**. Direction by stage-2 argmax
(higher of P(UP), P(DOWN); an exact tie goes UP — the LONG-first
convention of every prior rule in this project). No threshold is
searched, swept, or adjusted after results are seen.

For the duel, the end-to-end prediction rides the env's vote
channel as signed conviction **+1 (BUY) / −1 (SHORT) / 0 (HOLD)**
encoded `p_up = (c + 1) / 2` — the composite screen's rule-arm
encoding, driven by the same saved rule-policy zip
(`models.gbm_policy`), whose position-aware wrapper flattens when
the signal is gone or flipped. The encoding is lossless for this
policy: the wrapper fires at |c| ≥ 0.40, so {+1, 0, −1} reproduces
the end-to-end predictions exactly.

## Walk-forward (fixed)

TimeSeriesSplit, 5 folds, over the 15,961 pre-duel rows (13,300
graded rows), identical fold indices for both arms. Per fold, each
stage is fit on that fold's training rows only (stage 2 on the
training rows that are true TRADE). Reported per arm:

- **End-to-end (verdict-bearing):** pooled accuracy next to the
  majority baseline recomputed on the graded rows (0.6690 in every
  prior 1h study), pooled balanced accuracy next to 1/3 chance,
  and the per-fold balanced-accuracy spread.
- **Stage diagnostics (never verdict-bearing):**
  - gate balanced accuracy (binary, pooled + per fold) and the
    true TRADE base rate of the graded rows;
  - stage-2 direction accuracy on **true-TRADE** graded rows
    (stage 2 evaluated on every such row, gated or not);
  - stage-2 direction accuracy on **gate-fired true-TRADE**
    graded rows (the rows the combined policy actually trades
    that were truly tradeable);
  - gate fire rate and end-to-end activity share (predicted
    non-HOLD share of graded rows).

## Duel (fixed; both arms)

Signal-policy broker harness — the mirror of
`training.nightly.evaluate_ppo` used by the GBM / composite / 4h /
regime studies (`models.composite_screen.evaluate_windows_composite`:
StudyBroker/StudyEnv, default `ExitConfig` = the verified
production exit recipe — through-flat transitions, 2×ATR stop with
the 2.5% floor, trailing as designed, no take-profit, stop-anchor
ratchet), scratch ledger DBs in /tmp, $100,000 episodes,
next-candle-open fills, 25% tranches, 75% cap, 0.15% fees.

- **Windows:** the standard 3×264h set (from 2026-09-19 11:00 UTC)
  and the **robustness set** (the preceding 792h, from 2026-08-17
  11:00 UTC).
- **Fits:** standard windows — final models fit on all 15,961
  pre-duel rows (stage 2 on those rows' TRADE subset); robustness
  — **refit on rows strictly before the robustness region**
  (out-of-sample there).
- **Fee transparency:** net AND gross (pre-fee) P&L, total fees,
  round trips and exact non-hold activity share for every duel.
- **Yardsticks:** buy-and-hold on the same windows, and the
  **current champion graded fresh** through the same harness
  (run 44's PPO brain, `models/ppo_champion.zip`, installed
  2026-10-24 by Kass's manual override). Before any arm number is
  trusted, the mirror is cross-checked against
  `nightly.evaluate_windows` on the champion and must agree
  (agg/P&L diff < 0.01) — the composite screen's proof step.

## Pre-registered verdict rules (per arm; all three or NULL)

1. **Walk-forward:** pooled end-to-end balanced accuracy > 1/3
   AND at least 4 of 5 folds' balanced accuracy > 1/3 AND pooled
   end-to-end accuracy ≥ the majority baseline of the graded
   rows. (The operationalization the composite and regime plans
   fixed for "beats chance across the fold spread".)
2. **Duel:** total P&L > $0 AND total P&L > buy-and-hold on the
   same windows AND non-hold activity share ≥ 10%.
3. **Robustness:** robustness-set total P&L ≥ $0.

Any failure = the arm is NULL, registered promoted=0, and the
design is **not re-tuned for a second shot** — no different gate
threshold, no different stage split, no third arm. A full pass is
a report to Kass, not a promotion.

## Hierarchy-vs-flat comparison (reporting, fixed)

The report places each hierarchy arm next to the flat model on
identical features, from the flat models' own records:

- **Focus B** (run 22, focus-11 flat, class-weighted): WF
  accuracy 0.5508 / balanced 0.4145 (majority baseline 0.6690);
  duel P&L **−$8,138** as re-graded under current production
  code by the size study (its recorded-at-the-time duel was
  −$8,432, graded before the stop-anchor fix; the re-grade is
  the like-for-like number against today's harness).
- **Focus D** (run 24, all-60 flat, class-weighted): WF
  accuracy 0.5668 / balanced 0.4163; duel P&L −$3,178 as
  recorded (no current-code re-grade of D exists; labelled as
  recorded wherever quoted).

## Integrity checks (run with the build, results in the report)

- Stage-fit audits: every fit records exactly the timestamps it
  consumed; stage-2 fits consume only TRADE rows inside the
  caller's training mask (poison-tested: corrupting validation
  rows changes no fit input).
- Decision-rule unit tests: P(TRADE) = 0.50 fires, just below
  holds; direction argmax mapping incl. the tie→UP convention;
  conviction encoding equals the end-to-end labels.
- Fee identity per duel: gross = net + total fees, exactly
  (inherited from the shared harness, pinned by the composite
  screen's tests against the real broker).
- Champion/paper/live-state mtimes verified unchanged at the
  end (paper.db / live_state.json move only via the live loop;
  this lab study never opens them for writing).

## Registration

One training.db run per arm, `family="hierarchy"`, promoted=0
always, with walk-forward + duel metrics and the rule-by-rule
verdict in the record; family code **10** in the dashboards'
models index ("Two-Step Hierarchy — Focus 11" / "Two-Step
Hierarchy — All Features"); one lab note. The training-feed size
sentinel is respected: if the 2 new index rows trip the 7,400B
diet sentinel, it is re-baselined per that test's own documented
pattern (measured value + headroom, dated comment) and disclosed
in the report.

## What this plan is not

- Not a production change: the live bot (un-parked 2026-10-24 on
  Kass's override, run-44 champion) is untouched by this lab
  study; nothing here promotes, whatever the outcome.
- Not a threshold search in disguise: the 0.50 gate, the
  argmax direction rule, and the two feature sets are the whole
  design. A null closes Drawing-Board Option C — and with it
  Kass's A→B→C sequence — without licensing a variant (a
  different band, a third stage, a cascade) unless Kass directs
  one from the drawing board.
