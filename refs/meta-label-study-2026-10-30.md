# Meta-Label Gate — Pick A (2026-10-30)

Kass's direction (2026-10-29 ~19:32 ET): "think about creating a
different system when we get to model 80 and see if we can do
better." The A/B/C/D design proposal was presented with the
run-80 retrain report (2026-10-30 ~00:23 ET); Kass delegated the
pick the same morning: "You go ahead and pick our next model.
You choose." Muse picked **A — the tradeability gate /
meta-labeling system**, the only shape that builds on the one
fragment fourteen nulls left standing. This document freezes
the design BEFORE any system code is written or any result is
seen. One shot, no tuning sweeps: the graded configuration is
the one written here.

## Hypothesis

A learned filter that predicts "will THIS candidate trade make
money after fees" (meta-labeling) can veto the champion pair's
losing trades and beat the run 44 PPO + run 73 stacker champion
pair in the standard duel. Grounds: in the two-step hierarchy
study (2026-10-24, refs/hierarchy-2026-10-24.md) TRADEABILITY
was predictable (gate balanced accuracy 0.613 / 0.620) while
DIRECTION was a coin flip (0.5014 / 0.4952) — and the eyes duel
(2026-10-29) proved a better signal fed as 1 of 25 soft inputs
changes nothing, so the filter is wired as a HARD VETO on
entries, never as an observation. The risk chassis (production
broker rules: 25% tranche, 75% cap, ATR stop/trailing, 0.15%
fee/side) is kept exactly as-is; the gate only decides whether
an entry the champion brain already wanted is allowed to
happen.

## The system (frozen)

- **Primary signal**: the champion pair itself — PPO run 44
  (`models/ppo_champion.zip`, sha256
  e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb)
  deciding on run 73 stacker votes (Lab Stacker Champion #4,
  `models/stacker_champion.pkl`, meta feature_set `pick2_98`),
  replayed exactly as the nightly duel replays it
  (TradingEnv, next-candle-open fills, PaperBroker $100k /
  25% tranche / 5% fallback stop, 24-candle episodes from
  starts k*24, broker reset per episode).
- **Candidate trade**: every ENTRY attempt the brain makes in
  the replay — a buy action while flat-or-long (opens/adds a
  long) or a short action while flat-or-short (opens/adds a
  short), judged on the broker position at decision time.
  Sells, covers, and through-flat closes are position
  management, never candidates, and are never gated.
- **Meta-label** (training region only): each candidate is
  re-simulated STANDALONE on a fresh PaperBroker ($100k, same
  config): fill at the next candle's open with the decision
  candle's ATR (the duel's fill convention), then
  `check_stop` on every candle from the fill candle onward
  under production exit rules. Exit = the broker's
  stop_loss / trailing_stop; if the position is still open at
  the last training-region candle it is closed at that close.
  Label = 1 iff the round trip's net P&L (final cash −
  $100,000, both sides' fees included by the broker's own
  accounting) is strictly > 0, else 0. A candidate whose
  standalone entry fill is refused (dust/cap) is dropped and
  counted. Labels read post-entry candles for the OUTCOME
  only — that is the meta-labeling design, not leakage; the
  model never sees post-entry features.
- **Meta features (104, frozen)**: the pick2_98 scored
  feature row at the DECISION candle (98 values, the
  champion's own frame via `champion_holdout_frame`) plus six
  context values: the stacker vote the pair traded on
  (p_up), |p_up − 0.5|, the production `atr_rel`,
  `volatility_21`, the raw `atr_pct_raw`, and the side
  (+1 long / −1 short). Every feature reads data up to the
  decision candle only.
- **Votes provenance** (the program's cross-fit standard,
  audit 2026-10-20): training-region votes are CROSS-FITTED
  pick-recipe votes (`crossfit_p_up_pick`, TimeSeriesSplit
  folds=4, seed 42) — out-of-fold, matching the distribution
  the gate faces at duel time. Holdout votes are the fitted
  Champion #4's, exactly as the standard duel consumes them.
  Candidate extraction replays the pair on the cross-fitted
  votes; the meta-model's training rows carry those same
  votes as their p_up feature.
- **Training region**: tonight's supervised frame minus the
  last 792 candles (the shared holdout), restricted to rows
  where all 98 pick2 values are finite (the scored frame's
  warmup prefix drops out; the remainder must be hourly-
  contiguous, asserted). Tiled into 264-candle blocks (tail
  remainder dropped); 10 episodes per block, duel-identical.
- **Meta-model**: `LGBMClassifier(n_estimators=300,
  learning_rate=0.05, num_leaves=31, min_child_samples=20,
  subsample=0.9, colsample_bytree=0.9, random_state=42)`,
  fit on ALL training-region candidates. No class weighting,
  no feature selection, no other algorithm. Diagnostics
  (reported, never graded): candidate count, label base
  rate, and TimeSeriesSplit(4) out-of-fold accuracy /
  balanced accuracy / AUC over the candidate set in time
  order.
- **Gate rule (THE graded configuration)**: in the duel, an
  entry attempt executes iff meta P(profitable) ≥ **0.55**;
  a vetoed entry is replaced by hold for that step (the
  production stop check still runs — exits are never gated).
  Thresholds 0.50 and 0.60 are also run and REPORTED as
  sensitivity; only 0.55 is graded.

## Grading (frozen)

- Data: fresh `fetch_klines("BTC-USD", 1h, 730d)` at run
  time (nightly's retry pattern); holdout = last 792
  candles; 3 chronological windows × 264; identical data
  for both arms.
- Champion arm: `nightly.evaluate_windows` + `_agg` on the
  champion pair, production machinery untouched.
- Gated arms: a mirror evaluator that replays
  `evaluate_ppo` line-for-line (same episode starts, broker
  config, scoring) with a TradingEnv subclass whose only
  change is substituting hold for a vetoed entry BEFORE the
  production step runs — fills, stops, rewards and round-trip
  counting are the production code's own. Per-window ledgers
  are persisted under `data/meta_label/eval_ledgers/`.
  **Equivalence pin**: the mirror with the gate disabled
  (pass-all) must reproduce the production champion numbers
  (per-window score / P&L / max DD and the action mix) to
  within 1e-6; if it does not, the study ABORTS and nothing
  is registered. Champion fees/entries are read from that
  verified pass-all ledger.
- **Two-way P&L check**: the gated (0.55) arm's total P&L is
  computed a second, independent way — a re-sum over the
  persisted SQL ledgers only (per-fill cash/btc audit against
  the broker's recorded cash_after/btc_after, episodes marked
  at their final closes). The two computations must agree to
  under $0.01 or the study aborts.
- **Verdict**: formula pass = gated(0.55) aggregate score >
  champion aggregate AND gated wins ≥ 2 of 3 window scores.
  Then `gate_decision()`'s two-part floor on the gated
  record (non-hold ≥ 10% of the EXECUTED action mix —
  vetoed entries count as hold, the system's real activity;
  total P&L > $0; total P&L > buy-and-hold). Formula AND
  floor pass → **WOULD PROMOTE — PENDING KASS APPROVAL**
  (registry `pending_approval=1`, `promoted=0`). Anything
  less → the fifteenth null, recorded as such. Under NO
  outcome does this study install, crown, or move any model
  file or seat: the promotion lock governs, and this study
  never lifts it.
- **Registration**: one run, kind `ppo`, family `meta_label`
  (dashboard family code 17), display name "Meta-Label
  Gate — Pick A", metrics in the standard challenger/champion
  shape plus a `meta_label` sub-record carrying the full
  diagnostics: candidate count, base rate, OOF metrics,
  per-threshold veto counts / veto rate / fees for both arms,
  entries taken vs vetoed, and the sensitivity table. Lab
  note appended. Machine record in
  `data/meta_label/results.json` (candidates dataset, eval
  ledgers, and the fitted meta-model persisted alongside).
- **Guards**: sha256 of the live seat, shadow seat, lab
  champion files and both live-state files are taken before
  and after and must be identical. The study never opens
  `data/paper.db` / `data/paper_shadow.db`, never imports
  the live loops, and writes only under `data/meta_label/`,
  the registry, the lab notebook, and its own new files:
  `src/training/meta_label.py`,
  `scripts/run_meta_label_study.py`,
  `tests/test_meta_label_gate.py`, plus the three small
  family-registry additions in `src/dashboards/compact.py`
  (family code 17, display name, index verdict — the
  muse_pick3 precedent; without them the training feed
  cannot render a new family at all).

## What would make this a null

The gate's vetoes don't separate winners from losers out of
sample (the hierarchy fragment doesn't survive contact with
real candidate trades), or they do but the surviving trades
still can't beat the ungated pair on aggregate + windows, or
the floor fails (too little executed activity, P&L ≤ 0, or
P&L ≤ buy-and-hold). Any of those is recorded plainly as
null #15 — the same standard the previous fourteen were held
to.

## Results

_(appended after the run — the design above was frozen first)_

Ran 2026-10-30 (`scripts/run_meta_label_study.py`, machine
record `data/meta_label/results.json`). Registered as **run 82**
(kind `ppo`, family `meta_label`, promoted = 0,
pending_approval = 1) with a lab note. Guard hashes (both
seats, lab champions, both live-state files) identical
before/after; the paper books were never opened.

**Data.** Fresh 730d fetch; supervised holdout = the standard
last 792 candles (2026-09-27 16:00 UTC → 2026-10-30 15:00
UTC). Training region: the 12,079 rows before the holdout
whose pick2_98 values are all finite (2025-05-11 23:00 →
2026-09-27 15:00 UTC). The raw feed carries two known 6h
holes (2025-10-25, 2026-05-08); the duel machinery is
positional throughout (nightly included), so they misalign
nothing — recorded here because the runner's first draft
asserted a contiguity the feed never had.

**Candidates and labels.** The duel-style replay of the
champion pair over the training region produced **489 entry
attempts** (224 long, 265 short) against cross-fitted pick
votes. All 489 labeled standalone, 0 dropped. Base rate
**0.440** (215 winners / 274 losers): the pair's average
candidate trade, taken alone under production exits, LOSES
−$149.02 (winners +$647.12, losers −$773.73, mean fees
$74.95). Exits: 265 stop_loss, 224 trailing_stop, 0 horizon.

**Meta-model diagnostics (not graded).** Time-ordered OOF
over 388 covered candidates: accuracy 0.523, **balanced
accuracy 0.518, AUC 0.521** — barely above chance, and far
weaker than the hierarchy study's 0.613/0.620 (a different,
cheaper candidate definition: vote-direction entries, not
this brain's real attempts). OOF gate rate at 0.55: 45.6%
allowed.

**The duel** (identical holdout, next-open fills; champion
graded by production `evaluate_windows`, gated arms by the
mirror evaluator after its pass-all form reproduced the
production champion on all 3 windows within 1e-6):

| | W1 score / P&L | W2 score / P&L | W3 score / P&L | Agg | Total P&L |
|---|---|---|---|---|---|
| Champion pair (44+73) | 85.4 / +207.96 | −208.4 / −138.96 | −507.5 / −374.28 | **−210.17** | **−$305.28** |
| Gated pair @ 0.55 | 397.1 / +411.59 | 0.0 / 0.00 | −507.5 / −374.28 | **−36.79** | **+$37.31** |

Buy-and-hold over the same windows: −$8,459.84. Champion:
6 entries attempted, 6 filled, fees $262.35. Gated: **7
attempted, 2 vetoed (28.6%), 5 filled, fees $187.50**
($74.85 saved). The two vetoes, with the model's
P(profitable): a window-1 short at 0.4934 and a window-2
long at 0.0283; the five allowed attempts scored
0.684–0.927. Window 3 was identical in both arms.

**Sensitivity.** Thresholds 0.50 / 0.55 / 0.60 produce
IDENTICAL duels — no attempt's probability fell between
0.4934 and 0.6842, so the veto set never changes across
the pre-registered band. Only 0.55 is graded, per the
design.

**Floor.** Executed non-hold share 36.7% ≥ 10%; P&L
+$37.31 > $0; P&L > buy-and-hold. `gate_decision()` passes.

**Two-way P&L check.** The gated total P&L recomputed from
the persisted episode ledgers alone (per-fill cash/position
audit against the broker's own recorded balances, episodes
marked at the production close): **+$37.31 vs the
evaluator's +$37.31 — difference $0.0000**. An independent
diagnostic re-run of the gated arm afterwards reproduced
agg −36.79 / P&L +$37.31 / 7 attempts / 2 vetoes exactly.

**Verdict: PASS by the frozen criteria — WOULD PROMOTE,
held as PENDING KASS APPROVAL under the promotion lock.**
Formula (agg −36.79 > −210.17 AND 2/3 windows) and the
two-part floor both pass. Nothing was installed, crowned,
or seated; run 44 + run 73 remain the champions everywhere.

**How much weight this pass can bear — read before acting
on it.** This is the thinnest pass the frozen criteria
could produce, and it is recorded as a pass because the
criteria were frozen before the run, not because the
evidence is strong:

- The entire duel turned on **7 entry attempts and 2
  vetoes**. The champion pair barely traded this holdout;
  on a busier holdout the same gate could look completely
  different.
- The meta-model's out-of-fold discrimination is a coin
  flip with a slight lean (balanced acc 0.518). The two
  vetoes happened to be right — one of them (0.4934) was
  itself a coin-flip call.
- The total P&L swing is +$342.59 on a $100k book (0.34%),
  and window 3 — the worst window — was untouched.
- The hierarchy fragment (0.613/0.620) did NOT replicate
  at anything like its original strength on real candidate
  trades (0.518). What survived is closer to "vetoing the
  model's most confident losers didn't hurt, twice."

The honest summary: **not a fifteenth null by the
pre-registered standard — and not a demonstrated edge
either.** The gate is a recorded, pending system whose
only honest next test would be forward evidence (e.g. a
shadow trial), which is Kass's decision to make, not this
study's.
