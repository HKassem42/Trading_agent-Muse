# Muse's Pick #9 — LSTM Sequence Stacker — Pre-registration

**Frozen:** 2026-11-05, BEFORE any Pick #9 fit. Ordered by Kass
2026-11-05 13:18 ET: "do one more model, return back to the 1 hr
matrix but I want you to pick a complete new technique from our
rl combo document and train something we have not tried before
at all... pick whatever you want that u think we can beat our
best score."

## Source and selection

Source document, per Kass's instruction: **"Trading Bot ML/RL
Training Combinations"** (the RL/ML combinations document; its
catalog is inventoried in refs/archive-mining-2026-10-27.md §2a).
Its core-model list: Gradient Boosted Trees (tried — the stacker
recipe), Random Forest (tried — in the recipe), **LSTM —
"time-windowed matrices, e.g. 24-hour lookbacks" (never built)**,
DRL: PPO (tried — champion), SAC (never built), DQN (never
built), SVM (never built), meta-labeling (tried, runs 82–84),
SMOTE (tried, in the recipe).

**Pick: the LSTM over time-windowed matrices** — the document's
own prescription, and the one untried entry whose information
basis differs from everything that has failed: every stacker so
far reads ONE candle's feature row; this candidate reads the
ordered last-24-candles window of the 1h feature matrix (the
gap Kass himself flagged: "why do the eyes read one candle
instead of the last 24").

**Disclosure (exactness of "never tried"):** run 20 (HA-Seq v1,
2026-10-22) used a CNN+LSTM over raw Heikin-Ashi candle *shapes*
at 90-candle windows — a different input representation and
architecture, in the pre-standard era, graded a vacuous whisper
(acc 0.5189, sub-1σ). The document's technique — an LSTM over
the *feature matrix* windowed at 24h — has never been built or
graded in this program. SAC/DQN remain genuinely untried too;
they were passed over here because they are graded by duel P&L,
not by the accuracy score Kass asked to beat (0.5354), and every
RL duel since run 44 has lost on the same coin-flip input.

## Frozen design

- **Frame:** fresh production-path data — `fetch_klines`
  730d 1h → `make_supervised` (production 21 features, causal
  v5.1-clean). Exam = the standard holdout: the last **792**
  supervised rows. Features truncated-rebuild causality is
  re-verified at two cuts before fitting (the candidate consumes
  these features).
- **Candidate:** sequences of the last **24** rows of the
  21-feature matrix (sequence for supervised row *t* = rows
  *t−23…t*, label = row *t*'s next-1h direction). Features
  standardized with TRAIN-region statistics only. Model:
  LSTM(hidden 64, 1 layer) → dropout 0.2 → linear head;
  BCE loss with train class-balance pos_weight; Adam lr 1e-3,
  batch 256, ≤40 epochs, early stop (patience 5) on the last
  792 rows of the training span. **Seeds 42 / 7 / 123; the
  candidate probability is the mean of the three seeds**
  (declared seed-ensembling, not seed-picking).
- **Controls on the identical exam:** champions re-scored two
  independent ways each with exact agreement required —
  run 96 (prod21 artifact) and run 73 (pick2_98 artifact) —
  plus the flat Pick recipe on prod21 rows (the sequence-vs-flat
  read).
- **Walk-forward:** 5 folds over the training region via
  `purged_wf_splits` with a 24-row embargo; the candidate in
  each fold is the same 3-seed ensemble.
- **Threshold** fixed 0.50.

## The bar (frozen)

Registered **PENDING KASS (WOULD PROMOTE)** only if ALL hold:

1. exam acc > run 96 re-scored on this exam, AND
2. exam acc > run 73 re-scored on this exam, AND
3. exam acc > exam majority, AND
4. clean anatomy (run-63 rule: up-calls ≥ 8 AND precision
   > 0.05), AND
5. walk-forward mean acc > walk-forward majority.

Otherwise **FAIL**. If PENDING and exam acc also exceeds
**0.5354** (the clean-era best, runs 73/78), the record notes
BEAT THE BAR — that is the score Kass asked this pick to beat.

## Guards

Champion / live-seat / shadow-seat / gated-seat / run-73
artifact hashes captured before the fits and re-checked before
registration; any change aborts registration. Lab only under
the promotion lock; nothing installs.
