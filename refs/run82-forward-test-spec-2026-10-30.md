# Run 82 Forward Test — Pre-Registered Spec (DRAFT, awaiting Kass)

Prepared 2026-10-30 by Muse (proactive research) after run 82's
lab duel PASS. This spec is frozen BEFORE any forward data exists.
It authorizes nothing by itself: building the gated shadow book
starts only on Kass's explicit approval, the same standard as
every seat change under the promotion lock.

## Why a forward test, and why this shape

Run 82 (meta-label gate, family `meta_label`) passed its frozen
duel criteria — aggregate −36.79 vs the champion pair's −210.17,
2/3 windows, P&L +$37.31 vs −$305.28 — but the whole result
turned on 7 entry attempts and 2 vetoes, and the gate's own
out-of-fold discrimination was 0.518 balanced accuracy
(refs/meta-label-study-2026-10-30.md, Results). A pass that thin
has not earned a crown; it has earned forward evidence on trades
the gate has never seen.

The clean control already exists: **Book #2 (shadow) runs the
exact ungated pair — PPO run 44 + stacker run 73 — that the gate
filters.** A gated Book #3 running the same pair in parallel is
therefore a true A/B: same driver, same eyes, same candles; the
only difference is the gate's vetoes.

## The test (frozen configuration)

- **Book #3 (gated shadow)**: fresh $100,000 paper book, FLAT
  start, its own DB/state/seat copies, physically unable to
  write to Book #1 or Book #2 (same guard-hash pattern proven
  for Book #2). Driver PPO run 44 + eyes stacker run 73,
  identical to Book #2.
- **Gate**: the fitted model persisted at
  `data/meta_label/meta_model.pkl` from the run-82 study,
  threshold **0.55** exactly as graded (0.50/0.60 produced the
  identical duel; no re-tuning, no retraining during the test).
  It vetoes ENTRIES only; exits, stops, trailing, sizing (25%
  tranche / 75% cap) and fees are production rules, untouched.
- **Start condition**: both Book #2 and Book #3 flat at the
  start candle, so neither book's track record blends a
  position opened under different rules. (Both books were flat
  as of the Oct 29 5 PM covers; re-verify flat at build time —
  if Book #2 is mid-position, Book #3 still starts flat and the
  comparison window opens at Book #2's next flat point.)
- **Logging (non-negotiable)**: every candidate entry attempt
  on Book #3 is logged with candle time, side, the gate's
  P(profitable), and the verdict (allowed / vetoed). Vetoed
  candidates are ALSO tracked standalone to their production
  exit (the study's labeling method), so a veto can be scored
  right or wrong after the fact — without this, the test can
  show the gated book's P&L but can never show whether the
  vetoes themselves were correct.
- **Dashboards**: a third desk or a clearly labeled Book #3
  panel set, identity labeled on every panel ("Gated Shadow —
  Run 82 Forward Test"), plus a head-to-head strip: Book #3 vs
  Book #2 vs buy-and-hold over the test window. Kass test
  applies (phone width, Lotto language).

## Duration and stop conditions (frozen)

Entry attempts are sparse — the duel replay produced 7 attempts
in 33 days — so the test is candidate-counted, not just timed:

- **Checkpoint**: the first of (a) 10 candidate entry attempts
  on Book #3, or (b) 45 days. Report only; no ruling, no
  changes.
- **Ruling point**: the first of (a) 20 candidate entry
  attempts, or (b) 90 days. At the ruling point the test ends
  and the verdict below is computed and brought to Kass.
- **Early stop (safety)**: if Book #3's equity falls to
  $95,000 or below (Book #2's existing alert line), the test
  pauses and Kass is told the same day. Nothing auto-resumes.

## Verdict criteria at the ruling point (frozen)

All comparisons are over the identical test window, Book #3
(gated) vs Book #2 (ungated control):

1. **Money**: Book #3 net P&L (fees included) vs Book #2's.
2. **Veto quality**: the standalone outcomes of vetoed
   candidates — aggregate net P&L of the vetoed set. A gate
   whose vetoes were, in aggregate, would-be losers is doing
   its job even in a window where both books are red; a gate
   whose vetoes were would-be winners is subtracting value
   even if Book #3 happens to be ahead.
3. **Cost**: fees paid by each book over the window.

- **Forward PASS** (earns the right to ask Kass about a crown):
  Book #3 net P&L > Book #2 net P&L AND the vetoed set's
  aggregate standalone P&L < $0.
- **Forward FAIL** (run 82 recorded as the fifteenth null,
  pending flag cleared): either condition fails.
- **Inconclusive band**: if fewer than 8 candidates occurred
  by day 90, the test is reported as inconclusive — not a
  pass, not a null — and any extension is Kass's call.

Under NO outcome does this test install, crown, or seat
anything by itself. A forward PASS returns to Kass for a
promotion ruling; a FAIL or INCONCLUSIVE result changes
nothing in any book.

## What this does not touch

Book #1's live seat (run 44 PPO + 21-feature stacker), Book
#2's seat and track record, the promotion lock, the midnight
retrain, and the weekly deep train are all unchanged. Run 80
stays on Kass's back burner; run 82's registry state stays
WOULD PROMOTE — PENDING KASS APPROVAL unless Kass rules
otherwise — the forward test is an evidence-gathering ruling,
not a promotion.
