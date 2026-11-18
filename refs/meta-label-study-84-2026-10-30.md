# Meta-Label Gate — Run 84: the activity-calibrated gate (2026-10-30)

**Amendment to refs/meta-label-study-83-2026-10-30.md (run 83).**
Kass's order (2026-10-30 ~14:12 ET), after run 83 passed on a
single filled trade: "Can we make the models trade 30 percent
of the time instead of 10 percent then rerun it and see or fix
whatever u can to make it trade more so we get clear results.
Fix whatever u have to."

The 10% Kass cites is the promotion floor's minimum executed
activity (gate_decision()'s non-hold ≥ 10%). Run 83 passed
that floor while vetoing 7 of 8 entry attempts and filling
exactly ONE trade — a pass with almost no evidence in it. This
study changes the gate's OPERATING POINT so it trades at
roughly 30% instead, and re-grades. Everything else — the
eyes, the data, the duel, the verdict machinery — is run 83's,
unchanged. The house rules are not delegated away: this
document freezes the design BEFORE any run-84 holdout run,
grading is run 82/83's frozen protocol, lab-only, promotion
lock, no seat, book, or production file is touched.

## What does NOT change

- **Gate model: run 83's frozen fitted model, byte-identical**
  (`data/meta_label83/meta_model83.pkl`, sha256
  ddbe2b4870f5464f80ea4fa2bb60386a322cfbc43743cc89b6f8cb4326c744ae —
  the stacker-recipe variant selected by run 83's OOF-only
  zoo). No refit, no new model class, no new features. The
  114-vector, FIFO in-context histories, and gate mechanics
  are meta_label83's.
- **Data**: run 82's persisted fetch, byte-identical — the
  holdout is the same 792 candles (2026-09-27 16:00 UTC →
  2026-10-30 15:00 UTC), the training region the same 12,079
  rows, candidates run 83's persisted pooled set (640:
  615 stream 1 + 25 stream 2).
- **Duel**: 3 chronological 264-candle windows × 10 episodes,
  next-candle-open fills, production `evaluate_windows` for
  the champion arm, a mirror evaluator with the pass-all
  equivalence pin and the per-arm ledger re-sum two-way check
  for the gated arms.
- **Verdict (primary arm)**: formula pass = aggregate >
  champion aggregate AND ≥ 2/3 window scores, then
  `gate_decision()`'s two-part floor. Registration as ONE
  run, kind `ppo`, family `meta_label`, label **"Pick A-84"**,
  promoted = 0, pending_approval per the primary verdict.
- **Guards**: sha256 of the live seat, shadow seat, GATED
  seat (Book #3, new since run 83), lab champions and all
  three live-state files identical before/after. The three
  paper books (paper.db / paper_shadow.db / paper_gated.db)
  are information-only: their own loops write them on candle
  closes; this study never opens any of them.

## Change 1 — activity calibration of the threshold (the core change)

Run 83's graded threshold 0.55 sat far up the frozen model's
probability scale: the stacker's probabilities are polarized,
most candidates score below 0.55, and the gate degenerated
into a near-never-trade switch (1 fill in 33 days). Run 84
keeps the model and moves the line:

- **τ84 is chosen from TRAINING-REGION evidence only.** Take
  run 83's pooled training candidates (640) and compute
  TimeSeriesSplit(4) OUT-OF-FOLD probabilities with run 83's
  exact stacker recipe (`meta_label83.oof_proba_base(
  "stacker", X, y, folds=4)` — run 83 persisted only the OOF
  metrics, not per-candidate probabilities, so they are
  regenerated with the identical seeded recipe; the
  regeneration must reproduce run 83's recorded stacker OOF
  — n_oof 512, balanced accuracy 0.5275, AUC 0.5158 — within
  tolerance, or the study aborts before any holdout run).
  In-sample probabilities from the frozen fitted model are
  NEVER used for calibration: the frozen model was fit on
  these very rows.
- **τ84 = the 70th percentile** (numpy linear interpolation)
  of the OOF probabilities of the **stream-1** candidates
  (615; the OOF-covered subset) — the line that would have
  PASSED ≈ 30% of stream-1 entry attempts in training, under
  the gate's own `proba >= threshold` allow rule. The exact
  τ84 and the pass rate it implies on those OOF probabilities
  are published in the calibration block below BEFORE any
  holdout run. Stream-2 candidates do not set the line (the
  graded system gates the pair's proposals; stream 2 is the
  secondary arm's flow, gated at the same τ84).
- **τ84 is never adjusted after seeing holdout behavior.**
  No sensitivity band is graded; the duel runs τ84 only
  (plus the two equivalence pins below, which are machinery
  checks, not candidate configurations).

### Calibration result (frozen before the duel)

Computed 2026-10-30 from training-region data only, before
any run-84 holdout run. The OOF regeneration check passed
exactly: the regenerated stacker OOF over the pooled 640
candidates reproduces run 83's recorded metrics to the fourth
decimal (n_oof 512, balanced accuracy 0.5275, AUC 0.5158) —
the seeded recipe is deterministic and the calibration below
stands on the same probabilities run 83 selected on.

- **tau84 = 0.507079** — the 70th percentile of the **500**
  stream-1 candidates' OOF probabilities (615 stream-1
  candidates, 500 OOF-covered), implying a training pass rate
  of **30.0%** under the gate's `>=` rule. (For scale: run
  83's graded line was 0.55; tau84 sits just below it — the
  frozen model's probabilities are polarized, so the 70th
  percentile is not far under the old line.)

## Change 2 — the arms

### Primary arm (graded): stream 1 at τ84

The champion pair's entries (PPO run 44 on run 73 votes,
exactly the stream the previous gates graded) gated by the
frozen run-83 model at τ84, in the standard 3-window duel.
Reported: per-window and aggregate score / P&L, formula +
floor verdict, realized pass rate (allowed / attempted),
attempts / vetoes / fills, fees, executed action mix, and
**time-in-position share** (fraction of the 720 duel steps
after which the book held a non-zero position — sampled from
the broker after every step).

### Secondary arm (reported, NOT the headline grade): both streams at τ84

Kass's deeper scarcity: the pair alone attempts only 6–8
entries per 33 days, so even a perfectly calibrated gate on
stream 1 alone may still trade rarely. The secondary arm
gates BOTH proposal streams at the same τ84 in the same duel
windows, one book per episode:

- **Precedence (frozen)**: at each step the PPO decides
  first. If the PPO's action is non-hold, it is the step's
  proposal — an entry is gated as stream 1, position
  management executes ungated — and the rule does not fire
  that step (a vetoed stream-1 entry consumes the step as a
  hold, exactly as the primary arm). Only when the PPO holds
  does the vote-crossing rule (run 83 Change 1's truth table,
  on the champion's fitted holdout votes, current book state)
  propose: its entries are gated as stream 2 (stream-id
  feature 0.0, stream-2 win-rate history seeded with the 25
  stream-2 training outcomes), its exits execute ungated.
  Production broker rules throughout; the rule never adds
  beyond its own design (entries only from flat).
- Stream-1 win-rate history is seeded with all stream-1
  training outcomes, exactly as runs 82/83; each stream's
  history grows only with its OWN tranches' outcomes in the
  duel book (FIFO tranche attribution, the run-83 tracker,
  tranches tagged by the stream whose entry opened them).
- Reported with the same full accounting as the primary arm
  (per-window scores, pass/veto/fill counts split by stream,
  fees, time-in-position). Promotability language applies to
  the primary arm only; the secondary arm's result is
  described plainly, including if it beats or loses to the
  primary.

### Machinery pins (run before the graded arms; failure aborts)

1. **Pass-all pin**: the run-84 mirror evaluator with no gate
   (stream-1 mode) reproduces the production champion
   per-window score / P&L / max drawdown and action mix to
   within 1e-6 — run 82/83's equivalence standard.
2. **Run-83 reproduction pin**: the run-84 machinery in
   stream-1 mode with the frozen model at run 83's graded
   threshold 0.55 reproduces run 83's recorded 0.55 arm
   (agg +38.12, P&L +$236.92, 8 attempted / 7 vetoed / 1
   filled, fees $37.50) — score/P&L within $0.01, counts
   exact. This ties the new evaluator + Gate84 history
   plumbing to the trusted run-83 numbers before τ84 is
   graded on it.
3. **Veto-accounting cross-check** (every gated arm): the
   gate object's own per-stream attempt/veto counts must
   equal the env's entries_attempted / entries_vetoed
   totals read from the mirror's gate_stats.
4. **Two-way P&L check** (every gated arm + pins that carry
   ledgers): the arm's total P&L recomputed from its
   persisted episode ledgers alone must agree with the
   evaluator's to under $0.01, or the study aborts.

## Comparisons (frozen)

Primary arm vs run 83 @0.55 (agg +38.12 / P&L +$236.92) vs
run 82 @0.55 (agg −36.79 / P&L +$37.31) vs the ungated
champion pair (agg −210.17 / P&L −$305.28) vs buy-and-hold
(−$8,459.84), all on this same holdout. The question Kass
asked — does trading ~30% of the time produce CLEARER
evidence — is answered on realized activity (fills, distinct
trades, time-in-position) as much as on P&L: a τ84 pass on
one fill would be run 83's failure mode repeated, and is
reported as such.

## Holdout-reuse caveat (recorded, applies with full force)

This is the THIRD grading on this same 33-day holdout (runs
82, 83, 84). τ84 itself is honest — chosen on training-region
OOF probabilities only — but the cross-study selection
pressure is now substantial: three systems have been
compared, amended, and re-compared against the same 792
candles, and each pass on it is weaker evidence than the one
before. A pass here is recorded exactly as the frozen
criteria dictate, and it is NOT forward evidence. Book #3 —
Gated Shadow, the live forward test of run 82's gate — 
remains the clean instrument; nothing in this study touches
it, and what to do with run 84 (forward-test it instead,
alongside, or not at all) is Kass's call, as is any
promotion.

## What would make 84 a failure

τ84's realized holdout activity stays near run 83's (the
training-implied ~30% pass rate does not transfer — the
holdout attempts' probabilities sit lower on the frozen
model's scale), or the extra trades the lower line admits
lose more than the vetoes were saving (P&L worse than run
83's, or negative), or the formula/floor fails. Any of those
is recorded plainly, with the full four-way table. A
non-promotable 84 is a failed calibration attempt — NOT
numbered among the program nulls (runs 82/83's passes stand
as the gate's record; 84 is the operating-point challenger).

## Results

_(appended after the run — the design above was frozen first)_

**Registered as training run 84 (kind ppo, family meta_label,
promoted=0, pending_approval=0): FAIL — the primary arm passed
the formula (agg −85.62 > champion −210.17, 3/3 windows) but
failed the two-part floor on P&L (−$1.10 ≤ $0). NOT promotable;
no pending approval recorded (the gate pending on the
dashboards remains run 83). Guard hashes (live / shadow /
gated seats, lab champions, all three live-state files)
identical before/after; no paper book was opened.**

### The duel (shared holdout, graded threshold tau84 = 0.507079)

| arm | W1 score/P&L | W2 score/P&L | W3 score/P&L | agg | total P&L |
|---|---|---|---|---|---|
| champion (44+73) | 85.4 / +207.96 | −208.4 / −138.96 | −507.5 / −374.28 | −210.17 | −$305.28 |
| run 82 gate @0.55 | 397.1 / +411.59 | 0.0 / 0.00 | −507.5 / −374.28 | −36.79 | +$37.31 |
| run 83 gate @0.55 | 114.4 / +236.92 | 0.0 / 0.00 | 0.0 / 0.00 | +38.12 | +$236.92 |
| **run 84 primary @tau84** | **114.4 / +236.92** | **0.0 / 0.00** | **−371.2 / −238.03** | **−85.62** | **−$1.10** |
| run 84 secondary (both streams) | 114.4 / +236.92 | 0.0 / 0.00 | −371.2 / −238.03 | −85.62 | −$1.10 |

Buy-and-hold −$8,459.84. The whole story is in window 3: the
window-1 winner (+$236.92) passes at both thresholds and is
identical in every gated arm. At 0.55 the gate vetoed
everything in window 3 and the arm stayed flat; at tau84 it
admitted window-3 trades that lost **−$238.03** — almost
exactly the window-1 win. The marginal trades the stricter
line was keeping out were, on this holdout, losers.

**Activity (the point of the study).** Primary arm: 6 entry
attempts (the trajectory diverges from run 83's 8 — admitted
entries change the book state), 3 vetoed, **realized pass
rate 50%** (3/6 — the holdout attempts scored higher on the
frozen model than the training distribution implied by the
30% target), **3 filled** (vs run 83's 1, run 82's 5), fees
$112.50 (champion $262.35, run 83 $37.50), executed non-hold
36.3% (floor's activity leg passed), **time-in-position
4.6%** of the 720 duel steps vs run 83's 1.25% and the
ungated champion's 5.1%. The calibrated gate's book was
positioned about as often as the champion's own — Kass's
"trade more" was delivered; it just didn't pay.

**Secondary arm — stream 2 never fired.** Both-stream totals
are identical to the primary's to the cent, with stream-2
attempts = 0. Verified cause (not an evaluator defect): run
73's fitted holdout votes range **0.4615–0.5421** over all 792
candles — they never leave the rule's [0.45, 0.55] band, so
there are **zero raw vote crossings** on this holdout for the
rule to propose from, at any threshold plumbing. (The unit
tests prove the combined machinery fires stream-2 proposals
when crossings exist; on this data there are none.) The
deeper scarcity is upstream of the gate: the eyes themselves
rarely express a view strong enough to cross the bands.

**Machinery pins.** Pass-all mirror84 == production champion
to 1e-6 on all 3 windows (score / P&L / max DD / action mix).
The same machinery at 0.55 reproduced run 83's recorded arm
exactly (agg +38.12, P&L +$236.92, 8 attempted / 7 vetoed /
1 filled, fees $37.50). Veto-accounting cross-checks (gate
counters == env counters) passed on every gated arm.
Two-way P&L checks: ledger re-sum vs evaluator diff **$0.0000**
on the primary, secondary, and reproduction arms.

### Did trading more produce clearer evidence? Yes — against the gate

This is the clearest of the three gate results, and it cuts
against reading runs 82/83 as an edge:

- Run 83's +$236.92 pass rested on ONE trade plus seven
  vetoes. Run 84 kept that same winning trade, admitted the
  next tier of candidates the 0.55 line had excluded, and
  those trades summed to −$238.03. Net of everything, the
  calibrated system made **−$1.10 in 33 days** — statistically
  and economically zero.
- The gate's discrimination, such as it is (OOF balanced
  accuracy 0.5275), appears concentrated in its most extreme
  vetoes; the broad middle of its ranking — the part a 30%
  operating point must admit — carries no measurable edge on
  this holdout.
- Three gradings on this holdout (82, 83, 84) now read:
  +$37.31 on 5 fills, +$236.92 on 1 fill, −$1.10 on 3 fills.
  The pattern is what near-chance eyes plus selection
  pressure predicts. The frozen criteria record run 84 as a
  failed calibration (formula pass, floor fail) — not a
  program null, and not a promotion candidate. Book #3's
  forward test of run 82's gate remains the only clean
  evidence in flight; nothing here touches it.

### Verification

- New adversarial suite tests/test_meta_label84_gate.py:
  **10/10** (hand-computed 70th-percentile calibration with
  the >= pass convention; stream/NaN/order provenance of
  tau; per-stream gate plumbing; deny-all / allow-all /
  PPO-precedence duel integration on synthetic windows;
  gate-vs-env veto accounting; pair-mode deny-all flatness).
  Watched failing 10/10 on the stub before implementation.
- Full suite (post-registration): **368 passed, 5 failed** —
  none a regression from run 84's code. Four are run 83's
  documented baseline (test_muse_pick3's run-75 detail-window
  pin, test_training_panels' gate-history pin, two
  test_incorporation_feeds registry-fact pins); the
  intermittent shadow-vote 1-ulp flake passed this run. The
  fifth is NEW but is the documented registry-growth
  tripwire, not a defect: test_compact_feeds' size sentinel
  (training feed 12,562B at 84 runs vs the 12,350B target
  re-baselined at the 83-run registry; the feed stays far
  under the 16,384B hard budget asserted in the same test).
  Each registered run adds a models-index row; run 84's row
  crossed the line. The diet / re-baseline belongs to the
  dashboard feed workstream (AGENTS.md documents the pattern)
  and was deliberately NOT done inside this grading study.
- Dashboards: compact training + gate feeds pushed verbatim
  and read back — Training Lab newest run 84 "Meta-Label
  Gate — Pick A-84" FAIL, gate pending = run 83 on BOTH
  dashboards (run 84 recorded no pending approval), parked 0.
  Disclosure: the first training push carried two hand-
  transcription errors (a telemetry row and a feature-list
  tail); both were caught against the machine-generated feed
  and corrected by a full verbatim re-push, verified field-
  by-field on read-back (telemetry row 79, reads.c2 lists,
  models/audit counts).
- Machine record: data/meta_label84/results.json; duel
  ledgers under data/meta_label84/eval_ledgers/; persisted
  OOF probabilities data/meta_label84/oof_proba84.npz.
