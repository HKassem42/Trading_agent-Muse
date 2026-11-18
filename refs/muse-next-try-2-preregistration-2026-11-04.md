# Muse's Next Try #2 — Pre-registration (frozen BEFORE any grading)

Date: 2026-11-04. Same Kass order as Next Try #1 ("pick my next
model... WHATEVER you want... beat your previous try... GO!").
Label: **Muse's Next Try #2** — family tag `muse_next_try2`.

## What Try #1 taught (train-data evidence only)

Try #1 (run 101) is a clean FAIL and stands as recorded: holdout
0.4899 vs run 96's 0.5265 and run 73's 0.5227 re-scored on the same
exam; control 98 eyes 0.5101; majority baseline 0.5101. Two design
defects are visible from TRAIN data alone, without touching any
alternative frame's holdout grade (none was computed):

1. **The proxy lied.** Frame screening used a fast ExtraTrees+
   LightGBM proxy, which ranked the prod21+info frame (e) first at
   0.5158 train-CV — but the actual Pick recipe's walk-forward on
   that same frame and rows is 0.4895. A selector that does not
   rank frames the way the real recipe does is not a selector.
2. **The threshold knob manufactured calls.** The train-OOF rule
   (use θ* if it beats 0.50 by ≥ 0.005) fired at θ=0.42 on a
   +0.0118 OOF gain and produced 792/792 up-calls on the holdout —
   the "win" was noise amplified into a degenerate stance.

## Try #2 design (the fixes, and nothing else)

- **Frames:** the three all-scored candidates from Try #1 —
  (a) control 98, (b) info 112, (d) full 120. All values live on
  the same 0..100 percentile scale, the Pick recipe's native
  habitat (Pick #2's win was on exactly such a frame). The mixed-
  scale prod21+info frame (e) is excluded: its recipe-WF failure
  is already measured.
- **Selection:** the REAL Pick recipe's own walk-forward —
  3-fold TimeSeriesSplit on the identical 12,045 train rows —
  picks the frame. No proxy, no other knob.
- **Threshold:** fixed at 0.50. The knob is removed.
- **Recipe:** `build_pick_pipeline` unchanged (scaler → SMOTE →
  RF+XGB+LGBM+ET → LR).
- **Grading (once):** the selected pipeline, fit on all train
  rows, graded on the same shared 792-candle holdout. Champions
  were re-scored for Try #1 on this exam (run 96: 0.5265, run 73:
  0.5227, two-way agreed) — those re-scores stand and are reused;
  the challenger grade is computed once.
- **Exam-reuse disclosure:** this is the second challenger graded
  on this holdout window (Try #1 was the first; the Nov-4 nightly
  also graded its own challenger on its overlapping window, as
  every study in this program does). Frame selection uses train
  data only, but a win here carries less evidence than a
  first-use win — it will be reported that way, and the promotion
  lock still applies: a win is PENDING KASS APPROVAL, nothing
  installs, seats/books untouched, guards hash-checked.
- **Verdict:** WIN = holdout accuracy strictly beats BOTH
  re-scored champions (0.5265 / 0.5227) with non-degenerate
  anatomy (up-calls > 7, precision > 0.05). Precision, up-calls,
  5-fold WF of the winner, per-window accuracies, and the
  majority baseline (0.5101) are reported regardless.
