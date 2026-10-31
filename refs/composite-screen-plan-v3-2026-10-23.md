# Pre-registered test plan — Matrix v3 composite scores

Date: 2026-10-23 · Status: **PLAN ONLY — not run. Awaiting Kass's go.**
Written in advance, before any composite is trained or screened, so the rules below cannot be tuned to the results. If Kass green-lights the run, this file is the contract; any deviation gets reported as a deviation, not quietly absorbed.

## The question

Matrix v3 (16,735 rows × 102 cols, refs/matrix-v3-normalized-2026-10-22.md) converted every indicator to a 0–100 trailing-180-day percentile score and built four composites: `score_master`, `score_trend` (45 cols), `score_momentum` (25), `score_volume` (7). Normalization adds no information — the individual indicators are expected to stay null, as the assistant told Kass on 2026-10-22 (~1-in-5 prior). The one genuinely new object is the **composite**: an average of many weak readings. This plan tests only whether the composites carry anything.

## Fixed design (no changes after results are seen)

**Features:** the 4 composite columns only. Family membership stays exactly as built in v3 — no adding, dropping, or re-weighting columns after grading.

**Arms (3):**
1. **Rule–Master (no learning):** score_master ≥ 60 → BUY bias, ≤ 40 → SHORT bias, otherwise HOLD. Thresholds are symmetric around 50 and fixed here, in advance.
2. **Rule–Family (diagnostic only):** the same 60/40 rule applied solo to score_trend, score_momentum, and score_volume. Walk-forward graded only; these do **not** go to the duel, so a lucky family can't be cherry-picked into a trade claim.
3. **GBM–4:** LightGBM on the 4 composites, class-weighted (inverse class frequency — the arm style that traded most honestly in the focus test), same hyperparameters as refs/gbm-focus-2026-10-22.md (chosen once there, never tuned against duel windows).

**Walk-forward:** TimeSeriesSplit, 5 folds, over the 15,961 pre-duel matrix rows only (duel region starts 2026-09-19 11:00 UTC) — identical harness to the GBM focus and lag tests, so results are comparable to runs 21–27. Report accuracy **next to** the majority baseline (0.6690 on the graded rows) and balanced accuracy next to 0.33 chance. Accuracy alone is never reported as success (HOLD is ~66% of rows).

**Duel:** final models graded by the standard `evaluate_windows` machinery on the standard 3×264h windows — same champion, next-candle-open fills, 25% tranches, 75% cap, 0.15% fees, and the exit recipe verified in the exit study (through-flat transitions, 2×ATR stop with the 2.5% floor, trailing as designed, no take-profit). Arms 1 and 3 only. Robustness: the same arms are also graded on the older window set used by the exit/size studies — a standard-windows pass that loses money on the older set is a fail, not a pass.

**Fee transparency:** report pre-fee P&L alongside net P&L for every duel arm. The size study (refs/size-study-5pct-2026-10-22.md) showed Lag F and Focus B were breakeven-to-positive before fees and lost only to churn — a composite that is only "less negative after fees" has no edge worth having.

## Pre-registered verdict rules

A composite claim survives **only if all three** hold:
1. Walk-forward balanced accuracy beats chance (0.33) by a margin that survives the fold spread (not one lucky fold), and accuracy is not below the 0.6690 majority baseline.
2. Duel: passes the relative formula **and** Kass's two-part gate floor — ≥10% non-hold decisions, duel P&L > $0, duel P&L > buy-and-hold on the same windows.
3. Older-window robustness set: P&L not negative.

Any failure = the composites are null, the result is registered like runs 21–27 (promoted=0), and **the thresholds, families, and arms are not re-tuned for a second shot**. One clean test, then the book on v3 composites is closed unless Kass directs otherwise from the drawing board.

## What this plan is not

- Not a production change: trading stays parked at $99,232.84 / 0 BTC regardless of outcome until a challenger passes the full gate under the normal promotion path.
- Not a re-run of the GBM focus/lag tests on normalized data — tree models split on rank order, which percentile scoring preserves, so those results would not change and are not re-claimed here.
- Not permission to run: Kass said "let's go with the start of that V3" for the build; the build is done and verified. Training/screening on v3 was explicitly left as his next call (2026-10-22 report).
