# Meta-Label Gate — Run 83: the strengthened gate (2026-10-30)

**Amendment to refs/meta-label-study-2026-10-30.md (run 82).**
Kass's order (2026-10-30 ~13:30 ET), after run 82 passed thin:
"Fix whatever you want with 82 and run a 83 and see if you can
improve it. See where it's weak and see if u can make it
better." Full design latitude delegated; the house rules are
not delegated away: this document freezes every change BEFORE
the run-83 duel is graded, grading is byte-identical to run
82's frozen protocol, lab-only, promotion lock, no seat, book,
or production file is touched.

## What run 82 diagnosed (the fix list)

1. **Weak gate eyes.** Meta-model OOF balanced accuracy
   0.518 / AUC 0.521 — barely above chance. The hierarchy-era
   tradeability fragment measured 0.613 / 0.620; run 82 did
   not replicate it.
2. **Thin data.** 489 candidates from ONE replay of the
   champion pair over the training region; the duel turned
   on 7 entry attempts and 2 vetoes — variance dominates.
3. **Standalone labels.** Each candidate was labeled by a
   standalone simulation of that single trade; real entries
   interact with book state (existing position, adds, the
   exposure cap), so the labels misprice context.
4. **Single uncalibrated LightGBM.** Probabilities came out
   polarized (vetoed 0.49 / 0.03, allowed 0.68–0.93) and the
   0.50 / 0.55 / 0.60 thresholds produced identical duels —
   the probability scale was not earning its keep.

## What changes (frozen) — and what does not

Unchanged from run 82: the champion pair (PPO run 44, sha256
e610ddf3…4cb, on run 73 stacker votes, pick2_98); the data
(run 82's persisted fetch — the holdout is byte-identical:
2026-09-27 16:00 UTC → 2026-10-30 15:00 UTC, the last 792
supervised candles; training region the 12,079 finite-pick2
rows before it); the duel (3 chronological 264-candle windows
× 10 episodes, next-candle-open fills, production
`evaluate_windows` for the champion arm, the mirror evaluator
with its pass-all equivalence pin and ledger re-sum two-way
check for the gated arms); the verdict (formula pass =
aggregate > champion AND ≥ 2/3 windows, then
`gate_decision()`'s two-part floor); the graded threshold
**0.55** (0.50 / 0.60 reported as sensitivity, never graded);
registration as ONE run, kind `ppo`, family `meta_label`,
label **"Pick A-83"**, promoted = 0, pending_approval per the
verdict; guard hashes on both seats, lab champions and both
live-state files identical before/after; the paper books are
never opened.

### Change 1 — two candidate streams, multi-offset (fixes #2)

- **Stream 1 (the pair's own proposals, ×3 offsets).** The
  run-82 extraction replay (duel-identical episodes: 264-
  candle blocks, 10 episodes × 24 steps, next-open fills,
  broker reset per episode) is run at episode-start offsets
  **0, 8 and 16** within each block. Same decisions, three
  book-state phasings — each offset is a genuinely different
  trajectory, so its candidates and labels are real, not
  copies.
- **Stream 2 (vote-crossing proposals).** A deterministic
  rule policy over the same blocks (offset 0 tiling): reading
  the cross-fitted votes at the decision candle — flat and
  votes cross UP through 0.55 (prev < 0.55 ≤ now) → buy;
  flat and votes cross DOWN through 0.45 (prev > 0.45 ≥ now)
  → short; long and cross down → sell; short and cross up →
  buy (cover). No adds. Exits are the production stops plus
  those rule exits — every candidate is a real simulated
  decision under production broker rules (25% tranche, 75%
  cap, ATR stop/trailing, 0.15%/side), never a synthetic
  trade.
- Candidates are pooled for training (the duel still gates
  ONLY stream-1 proposals — the pair's entries — exactly as
  run 82). Exact duplicates (same decision candle, side and
  stream) are dropped and counted; candidates from different
  offsets are distinct by construction (different book
  states, potentially different labels).

### Change 2 — in-context labels by FIFO tranche attribution (fixes #3)

Each candidate is labeled INSIDE its own replay trajectory —
no standalone simulation. A FIFO tranche ledger rides every
replay episode:

- Every fill that increases the position in its direction
  opens a tranche (signed qty, entry price, entry fee). A
  candidate IS its entry fill: the fill produced by its
  recorded entry attempt. An attempt whose fill is refused
  (dust / exposure cap) opens no tranche — the candidate is
  dropped and counted.
- Every fill that reduces the position closes tranches
  oldest-first. A closed slice of a tranche contributes
  `qty × (exit_price − entry_price) × side_sign − entry-fee
  share − exit-fee share` (fee shares pro-rata by qty).
- A tranche still open when its episode ends is marked at
  the episode's final mark close (the close the production
  env marks that episode at — positional start + 24) with
  NO exit fee charged — the env's own accounting. Its
  outcome is final at that mark.
- A tranche's contribution is the sum over its closed
  slices plus its marked remainder. **Label = 1 iff the
  candidate's tranche contribution is strictly > 0.**
- The tranche ledger consumes the broker's own recorded
  fills (side / price / qty / fee, in ledger order) — the
  same arithmetic `resum_episode_pnl` audits: BUY pays
  qty×price + fee, SELL receives qty×price − fee.
- An outcome's close position is the candle at which its
  closing fill executed (marked tranches: episode start +
  24); features and history filters below use exactly that.

### Change 3 — 114 features: run 82's 104 + 10 causal context (fixes #1)

Indices 0–103 are run 82's frozen vector, byte-identical
(98 pick2 values at the decision candle, then p_up,
|p_up − 0.5|, atr_rel, volatility_21, atr_pct_raw, side).
Appended, all computed from data at or before the decision
candle only:

- **[104] p_up slope 6** = votes[t] − votes[t−6];
  **[105] p_up slope 24** = votes[t] − votes[t−24].
  Within-region votes (cross-fitted on training, the
  champion's on the holdout); if t−k precedes the region's
  start the slope is 0.0.
- **[106] stream win-rate, last 20** and **[107] stream
  win-rate, expanding**: over the candidate's OWN stream,
  the outcomes (Change 2) whose close position ≤ the
  decision position, in close order — last-20 window and
  all-history. Both Laplace-smoothed: (wins + 2) / (n + 4);
  with no history both are 0.5. In the duel the stream-1
  history is seeded with ALL stream-1 training outcomes
  (they precede the holdout) and grows as duel tranches
  close; processing order there is chronological, so the
  same ≤ rule is what the live gate sees.
- **[108] implied stop distance, ATR units** =
  stop_pct(atr) / atr, where stop_pct is the broker's own
  `_stop_pct_for` derivation (min(max(2.0 × atr_pct_raw,
  0.025), 0.08)) at the decision candle; 0.0 when
  atr_pct_raw is not finite-positive.
- **[109] volatility-regime percentile**: the decision
  candle's volatility_21 ranked within its trailing 500
  candles (fraction of the window's earlier values strictly
  below it; 0.5 when the window has < 2 values). Computed
  once over the full supervised series — the value at t
  reads candles ≤ t only.
- **[110] drawdown from the 100-candle high** =
  close[t] / max(close[t−99..t]) − 1 (≤ 0).
- **[111] hour sin, [112] hour cos**: sin/cos of
  2π × hour(UTC) / 24 at the decision candle.
- **[113] stream id**: 1.0 = pair proposals (stream 1),
  0.0 = vote-crossing proposals (stream 2). The duel gate
  always evaluates stream 1.

### Change 4 — model zoo, OOF-only selection, calibration (fixes #4, #1)

Compared on TRAINING-REGION out-of-fold metrics ONLY
(TimeSeriesSplit(4) over the pooled candidate set in time
order — decision position, then stream): balanced accuracy
at the 0.5 cut and AUC.

- (a) **lgbm82** — run 82's LightGBM, unchanged (control).
- (b) **stacker** — the program's stacker recipe
  (`stacker.build_pipeline(42)`: scaler → SMOTE →
  RF+XGB+LGBM → LR).
- (c) **lgbm82 + isotonic** calibration, (d) **lgbm82 +
  Platt** (logistic on the probability), (e) **stacker +
  isotonic**, (f) **stacker + Platt**. Calibrated variants
  are scored by NESTED out-of-fold: within each outer fold,
  the calibrator is fit on inner TimeSeriesSplit(3)
  out-of-fold probabilities of that fold's training part —
  no test fold touches its own calibrator.
- **Selection rule (frozen)**: highest OOF balanced
  accuracy; ties (|Δ| < 1e-12) broken by higher AUC. The
  duel is graded with ONLY the selected model. Selection
  never sees the holdout.
- The final model is the selected variant fit on ALL
  candidates; for a calibrated variant the calibrator is
  fit on the base model's plain TimeSeriesSplit(4)
  out-of-fold probabilities over the full candidate set
  (the standard OOF-calibration construction).

## Improvement bar (frozen)

Run 83 **improves on run 82** iff its graded (0.55)
aggregate AND total P&L both beat run 82's frozen numbers
(agg −36.79, P&L +$37.31) under the identical criteria. The
verdict is stated plainly either way; promotability
(formula + floor vs the champion pair) is recorded
separately, exactly as run 82. A promotable 83 that does
not beat 82 is recorded as promotable-but-not-an-improvement;
a non-promotable 83 is a failed improvement attempt — NOT
numbered among the program nulls (run 82's pass stands as
the gate's record; 83 is its challenger).

## Holdout-reuse caveat (recorded per the mandate)

Runs 82 and 83 are both graded on this same 33-day holdout.
Freezing the criteria and selecting the model on
training-region OOF only keeps the comparison honest, but
cross-study selection pressure on this holdout exists: two
shots at the same 792 candles are weaker evidence than one,
and a pass here — for either run — is not forward evidence.
The clean test remains a forward trial, which is Kass's
call, not this study's.

## What would make 83 a failure

The selected model's OOF eyes stay at chance (the richer
features and second stream don't move balanced accuracy),
or the gate's vetoes don't separate winners from losers on
the holdout, or the floor fails, or the improvement bar
misses. Any of those is recorded plainly, with the full
OOF table and the three-way duel table, win or lose.

## Results

_(appended after the run — the design above was frozen first)_

**Registered as training run 83 (kind ppo, family meta_label,
promoted=0, pending_approval=1): PASS by the frozen criteria —
WOULD PROMOTE, held PENDING KASS APPROVAL under the promotion
lock. The frozen improvement bar vs run 82 was also met. Read
the evidence-weight paragraph before the numbers: the graded
duel turned on ONE filled trade.**

### Candidates and labels

- Stream 1 (pair replay, offsets 0/8/16): 1,450 tranche-opening
  candidates, 13 entry attempts refused (no tranche).
- Stream 2 (vote-crossing rule): only 25 candidates. The
  crossfit OOF votes are heavily smoothed toward 0.5, so band
  crossings (0.55 up / 0.45 down) from a flat book are rare.
  Recorded as observed; the stream contributed little.
- Exact duplicates across the offset phasings: 835 dropped by
  the frozen dedupe rule. The pair attempts entries at largely
  the same candles regardless of episode phasing, so the
  offsets multiplied RAW attempts ~3× but unique decision
  points only modestly. **Pooled training set: 640 candidates
  (615 stream 1 + 25 stream 2) vs run 82's 489 — a real but
  much smaller gain than the raw count suggests.**
- In-context base rate 0.441 (stream 1: 0.444; stream 2: 0.360)
  vs run 82's standalone 0.440. Mean tranche contribution
  −$55.57 vs run 82's mean standalone P&L −$149.02 (in-context
  tranches carry pro-rata exit fees and add-tranches are
  smaller; the two means are not the same quantity).

### Model zoo — training-region OOF only (n_oof 512 of 640)

| variant | balanced acc | AUC | accuracy |
|---|---|---|---|
| lgbm82 (run 82 control) | 0.5015 | 0.5088 | — |
| **stacker (SELECTED)** | **0.5275** | **0.5158** | 0.5449 |
| lgbm82+isotonic | 0.4742 | 0.4881 | — |
| lgbm82+platt | 0.5015 | 0.4680 | — |
| stacker+isotonic | 0.5124 | 0.5022 | — |
| stacker+platt | 0.5000 | 0.4689 | — |

Selection per the frozen rule: the plain stacker (highest
balanced accuracy). Run 82's LightGBM recipe on run 83's data
scores 0.5015 — WORSE than its 0.518 on run 82's standalone
dataset, so the eyes' gain came from the model class, not the
data recipe. Calibration hurt every variant (Platt AUCs below
0.5: calibrators fitted on small inner-OOF sets degraded the
pooled ranking). The selected eyes (0.5275) remain barely
above chance — better than run 82's 0.518, nowhere near the
hierarchy-era fragment (0.613/0.620), which still does not
replicate.

### The duel (shared holdout, graded threshold 0.55)

| arm | W1 score/P&L | W2 score/P&L | W3 score/P&L | agg | total P&L |
|---|---|---|---|---|---|
| champion (44+73) | 85.4 / +207.96 | −208.4 / −138.96 | −507.5 / −374.28 | −210.17 | −$305.28 |
| run 82 gate @0.55 | 397.1 / +411.59 | 0.0 / 0.00 | −507.5 / −374.28 | −36.79 | +$37.31 |
| **run 83 gate @0.55** | **114.4 / +236.92** | **0.0 / 0.00** | **0.0 / 0.00** | **+38.12** | **+$236.92** |

Buy-and-hold −$8,459.84. Formula pass (agg beats champion,
3/3 windows) AND the two-part floor passed (executed non-hold
35% on the recorded last-window mix, P&L > $0 > B&H) →
promotable. Improvement bar met on both frozen numbers
(+38.12 > −36.79 and +$236.92 > +$37.31). Threshold
sensitivity: @0.50 agg −155.10 / P&L −$140.06 (vetoed 2/6,
fees $187.35); @0.60 agg 0.00 / P&L $0.00 (vetoed 9/9 — the
book never traded; a sensitivity arm only, it would fail the
floor's activity test). Unlike run 82, the three thresholds
now separate — the stacker's probabilities are polarized
below 0.55 for most candidates.

Veto/fees detail @0.55: the gated trajectory attempted 8
entries (vetoes change the trajectory — a vetoed entry leaves
the book flat to attempt again; the ungated champion attempted
6), vetoed 7 (87.5%), filled 1, total fills 1, fees $37.50 vs
the champion's $262.35 (6 filled / 7 fills). Two-way check:
ledger re-sum +$236.92 vs evaluator +$236.92, diff $0.0000.
The mirror83 evaluator with no gate reproduced the production
champion to 1e-6 on all three windows, and the champion arm
reproduced run 82's recorded champion numbers exactly — the
holdout is byte-identical, as designed. A post-duel probe
re-ran the 0.55 arm and reproduced agg +38.12 / P&L +$236.92
to the cent (determinism check).

### Evidence weight — read this before the verdict

The improvement is real under the frozen criteria and thin
in exactly the way run 82's was, only more so: **the entire
graded result is one filled short in window 1 that made
+$236.92, plus seven vetoes that kept the book out of
everything else (windows 2 and 3 are exactly flat).** The
gate at 0.55 is close to a "never trade" switch with a single
exception; whether that exception was skill or luck cannot
be told from one trade. The OOF eyes improved modestly
(0.5275 vs 0.518) and remain near chance. Two gates have now
passed on this same holdout (see the reuse caveat above) —
the selection pressure warning applies with full force. A
forward test remains the only clean evidence; Kass already
approved one for run 82's gate (Book #3 — Gated Shadow). What
to do with run 83 — forward-test it instead, alongside, or
not at all — is Kass's call, as is any promotion. Nothing was
installed; no seat, book, or champion changed (guard hashes
identical before/after, verified in the register phase).

### Verification

- New adversarial suite tests/test_meta_label83_gate.py:
  17/17 (FIFO attribution hand-computed, replay-level label
  causality under future garble, causality of every new
  context series, stream-history market-time filter, 114-vector
  layout, rule truth table, extraction isolation, zoo
  selection rule, mirror83 == production).
- Run 82's suite untouched: 17/17.
- Full suite (post-registration, authoritative run): 358
  passed, 5 failed, none of them a regression from run 83:
  test_muse_pick3's run-75 detail-window pin and
  test_training_panels' gate-history pin (both documented
  pre-existing Oct 29–30 registry aging); the shadow-vote
  1-ulp float flake documented with run 82 (passes
  standalone, verified); and two test_incorporation_feeds
  registry-fact pins whose failing asserts read only data
  fixed before run 83 existed — stacker audit counts (run 83
  is a ppo run; it changes no stacker field) and the
  telemetry newest-4 window (aged out by nightly runs 77–81;
  runs 82/83 carry no telemetry). Same registry-aging class,
  verified by reading the failing asserts. test_compact_feeds
  passes in full. (An earlier draft of this section carried
  a wrong suite figure — 342/4 — conflated from run 82's
  baseline; corrected 2026-10-30 after the actual suite
  output was re-read and each failure re-diagnosed.)
- Dashboards: compact training + gate feeds pushed verbatim
  (training snapshot + gate on both dashboards) and read back —
  newest run 83 "Meta-Label Gate — Pick A-83" PASS on the
  Training Lab, gate pending = run 83 on both dashboards.
- Machine record: data/meta_label83/results.json; final model
  data/meta_label83/meta_model83.pkl; duel ledgers under
  data/meta_label83/eval_ledgers/.
