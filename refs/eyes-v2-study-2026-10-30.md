# Eyes v2 — the tradeable-move stacker (run 85, 2026-10-30)

Kass's order (2026-10-30 ~14:31 ET): "What can we change to
make the eyes better — do whatever and change whatever to get
us results." Full design latitude delegated; the house rules
are not delegated away: this document freezes the design
BEFORE any Eyes v2 grading, lab-only, promotion lock, no
seat, book, or production file is touched. Amendment folded
in BEFORE the freeze (Kass, 2026-10-30 ~14:33 ET): "Can't we
have it look back at the last 24 candles to make the choice
instead of one?" — his suggestion is zoo arm (d), recorded
here as Kass's design input. It reaches the holdout only if
it wins selection on training-region practice data.

## The diagnosis (what this fixes)

The champion eyes barely express a view. Run 73's fitted
holdout votes span **0.4615–0.5421 across all 792 candles**
(run 84's stream-2 finding); the live seat stacker ranged
46–54% over 40 days (2026-10-30 gauge audit). Downstream,
everything starves: the PPO reads the vote as 1 of 25 soft
inputs (the 2026-10-29 eyes duel: a +2.3pt exam gain moved
money by $0.00), meta-label gates veto a trickle of
proposals, and vote-crossing streams never fire because the
votes never cross their bands. Fourteen nulls say next-candle
1h direction on classic indicators is a coin flip. The two
fragments that ever showed signal were TRADEABILITY
prediction (hierarchy study gate balanced accuracy
0.613/0.620) and Pick #2's moment features. Eyes v2 therefore
changes the QUESTION the eyes answer: not "up or down next
candle?" but "which way does the next tradeable move go?" —
the learnable target, pointed at the decision the system
actually makes.

## The system (frozen)

- **Primary signal under test**: a NEW stacker (the eyes),
  graded against the incumbent Lab Stacker Champion #4
  (run 73, `models/stacker_champion.pkl`, sha256 89b63394…,
  pick2_98). The money pair is PPO run 44
  (`models/ppo_champion.zip`, sha256 e610ddf3…4cb, frozen
  artifact) — the same driver the eyes duel used.
- **Data**: run 82's persisted fetch (`data/meta_label/
  df.pkl`, 17,510 hourly candles, 2024-10-30 16:00 →
  2026-10-30 15:00 UTC stamps) — no fresh fetch, so the
  holdout is byte-identical to runs 82–84. `sup =
  make_supervised(df)`; holdout = the last 792 sup rows
  (2026-09-27 16:00 → 2026-10-30 15:00), asserted equal to
  run 82's persisted `hold_sup.pkl`; training region = the
  sup rows before it.
- **Features (106, frozen)**: the pick2_98 frame (the
  champion's own 98 columns, in its recorded order, via
  `build_pick2_frame(df)` reindexed to sup rows) PLUS eight
  causal context features computed on the raw df and aligned
  to sup timestamps, in this order:
  1. `ctx_rv24` — std of log returns over the trailing 24
     candles (pandas rolling std, ddof=1);
  2. `ctx_rv96` — the same over 96 candles;
  3. `ctx_rv_ratio` — ctx_rv24 / ctx_rv96;
  4. `ctx_ret4` — close/close(−4) − 1;
  5. `ctx_ret24` — close/close(−24) − 1;
  6. `ctx_dd100` — close / max(close, trailing 100) − 1;
  7. `ctx_hour_sin`, 8. `ctx_hour_cos` — sin/cos of
     2π × hour/24 at the candle stamp (UTC).
  Raw units (the pick pipeline standardizes; the LSTM
  standardizes per fit; LightGBM is scale-free). Every
  feature at row t reads candles ≤ t only.
- **Labels — triple-barrier tradeable move (frozen)**. The
  ATR series is the production one in the supervised frame:
  `atr_pct_raw` × close, i.e. `_atr(df, 14)` — Wilder true
  range, EWM α=1/14, min_periods=14. Barrier width
  b(t) = **1.0 × ATR(14) at t**; up barrier U = close(t) +
  b(t), down barrier D = close(t) − b(t); horizon **24
  candles**. Scan j = t+1 … t+24 (candle HIGH/LOW):
  - high(j) ≥ U first → label 1; low(j) ≤ D first → label 0;
  - a single candle spanning BOTH barriers is resolved by
    that candle's own direction: close(j) ≥ open(j) → up
    first (1), else down first (0);
  - neither barrier touched: label by the 24-candle net
    move — 1 if close(t+24) > close(t), 0 if <; an exact
    tie drops the row (counted).
  Rows whose horizon runs past the series end carry no
  label. Labels read future prices for the OUTCOME only —
  the meta-labeling convention, identical in kind to runs
  82–84; no feature ever does.
- **Row sets (frozen)**:
  - *Common training rows* — sup positions t in the
    training region satisfying ALL of: (i) the 106 features
    are finite at t; (ii) the 106 features are finite over
    the whole trailing 24-row window t−23…t (sequence
    feasibility, shared by every arm so all OOF scores cover
    identical rows); (iii) a triple-barrier label is
    defined; (iv) **t + 24 < split** (purge: the label
    horizon lies entirely inside the training region — no
    training label is computed from holdout candles).
  - *Exam rows* — the first **768** holdout candles
    (positions split … split+767): every row whose
    24-candle label horizon fits inside the holdout, scored
    for BOTH eyes on identical timestamps against the
    triple-barrier labels.
  - *Duel votes* — the selected model's P(up-move) at ALL
    792 holdout positions (features exist at all 792; only
    exam scoring needs labels).

### The zoo — selection on training-region OOF only

TimeSeriesSplit(4) over the common training rows, in time
order. **Every arm's OOF predictions are scored against the
triple-barrier labels** (balanced accuracy at the 0.5 cut,
and AUC) — one yardstick for all four arms. Selection rule
(frozen): highest OOF balanced accuracy; ties
(|Δ| < 1e-12) broken by higher AUC. ONLY the selected arm is
graded on the holdout. (Known, disclosed: fold-boundary
training rows' labels read ≤24 candles into the next fold's
feature region — outcomes, never features; the convention of
runs 82–84, identical for every arm.)

- **(a) control** — run 73's exact recipe
  (`muse_pick.build_pick_pipeline(42)`: scaler → SMOTE →
  RF+XGB+LGBM+ExtraTrees → LR) trained on the NEW
  triple-barrier labels, 106 features.
- **(b) LightGBM** — run 82's frozen config
  (`LGBM_PARAMS`: 300 trees, lr 0.05, 31 leaves,
  min_child 20, subsample/colsample 0.9, seed 42) on the
  NEW labels.
- **(c) label ablation** — the control recipe trained on
  the OLD labels (the supervised frame's next-candle
  `target`) with the NEW 106 features; its OOF predictions
  are still scored against the triple-barrier labels, which
  is what isolates the label change from the feature change.
- **(d) sequence eyes — KASS'S ARM**. A compact PyTorch
  LSTM whose input at t is the SEQUENCE of the last 24
  candles of the same 106-feature frame (24 × 106), the
  sequence ending AT t (causal by construction; no padding
  is ever read from the future — sequences are gathered,
  not padded). Per-feature standardization uses the FIT
  rows' statistics only (mean/std over all timesteps of the
  fit rows; std floored at 1e-8). Architecture (frozen):
  LSTM(input 106, hidden 48, 1 layer, batch_first) →
  final-timestep hidden state → Dropout(0.3) →
  Linear(48 → 1). ≈30k parameters (exact count recorded by
  the runner). Loss: BCEWithLogits with pos_weight =
  #neg/#pos of the fit rows. AdamW (lr 1e-3, weight decay
  1e-2), batch 256, ≤ 40 epochs, early stopping on the fit
  rows' chronological last 15% (validation), the 24 rows
  immediately before the validation segment purged from the
  fit subset, patience 6 on validation BCE, best state
  restored. Seeds: torch + numpy 42. CPU.
  - **Fallback (d′)**: if (d)'s training raises or yields
    non-finite probabilities in ANY fold or the final fit,
    substitute the flattened-lag variant and record the
    substitution + reason here and in the machine record:
    series = the 8 context features + the top-24 pick2
    columns by arm (a)'s final fit's base-learner impurity
    importances (mean across RF/XGB/LGBM/ET inside the
    fitted stack, pick2 columns only); inputs = those 32
    series at lags 0…23, flattened lag-major (768 values);
    model = run 82's LightGBM config.
- **Final model**: the selected arm refit on ALL common
  training rows (for (d): same protocol, validation = the
  last 15% of the training rows). Persisted under
  `data/eyes_v2/` with its holdout votes.

## Grading (frozen) — TWO bars, both required

1. **Exam.** On the 768 exam rows, identical timestamps,
   predictions = vote ≥ 0.5:
   - new eyes: accuracy, precision, up-calls vs the
     triple-barrier labels;
   - run 73 re-scored: the fitted champion on its own
     pick2_98 frame (`champion_holdout_frame`), same rows,
     same labels: accuracy, precision, up-calls. (Its
     next-candle exam on the same rows and its recorded
     0.5354 are diagnostics for continuity, not the bar.)
   - vote spread per eye over all 792 holdout votes: min,
     max, % outside [0.45, 0.55], % outside [0.40, 0.60].
   - **Exam win = new accuracy STRICTLY > run 73's
     re-scored accuracy.** Anatomy (up-calls, precision,
     label base rate) is reported with the verdict so a
     degenerate win is visible in the record.
2. **Money test.** The standard 3-window duel (nightly
   machinery, `evaluate_windows`, next-candle-open fills),
   PPO run 44 frozen, on the same 3 × 264 windows:
   challenger arm consumes the new eyes' 792 votes;
   champion arm consumes run 73's fitted votes.
   **Money win = challenger aggregate score > champion
   aggregate score.**
   - **Two-way pin**: the fresh champion arm must reproduce
     runs 82–84's recorded champion numbers on this
     byte-identical holdout (aggregate −210.17, P&L
     −$305.28) to the cent; if it does not, the study
     ABORTS and nothing is registered.

**Verdict accounting (frozen).** Both bars pass → **WOULD
PROMOTE — PENDING KASS APPROVAL** (registry
pending_approval=1, promoted=0). Exam win only → recorded
as an exam win with pending_approval=0 (the eyes-duel
lesson: exam gains alone have never moved this brain's
money). Money win only → also not promotable; recorded
plainly. Neither → a failed eyes attempt. This is a
COMPONENT study: like run 84's failed calibration attempt,
a failure here is NOT numbered among the program nulls
(those count full-system tests); a pass is recorded as the
gate record stands — nothing about runs 82–84 changes.
Under NO outcome does this study install, crown, or move
any model file or seat: the promotion lock governs, and a
shadow-eyes swap is Kass's explicit call, never automatic.

**Registration**: one run, kind `stacker`, family `eyes_v2`
(compact family code 18 — the three small registry
additions in `src/dashboards/compact.py`, the meta_label
precedent), metrics in the nightly stacker shape
(challenger_acc / champion_acc = the two exam accuracies,
precision / champion_precision, challenger_wf_acc = the
selected arm's OOF accuracy, folds = 4, pending_approval per
the verdict, reason) plus an `eyes_v2` sub-record carrying
the full diagnostics: design, label stats (base rates,
tie-drops), the zoo OOF table, the selected arm, exam tables
for both eyes incl. spread, the duel table, and the
fallback flag if (d′) fired. Lab note appended. Machine
record in `data/eyes_v2/results.json`.

**Guards**: sha256 of the live seat, shadow seat, gated
seat (incl. the Book #3 gate model + manifest), lab
champions and all three live-state files, taken before and
after, must be identical. The study never opens
`data/paper.db` / `data/paper_shadow.db` /
`data/paper_gated.db` for writing (they are information-only
guard entries — the books' own loops write them), never
imports the live loops, and writes only under
`data/eyes_v2/`, the registry, the lab notebook, and its
own new files: `src/training/eyes_v2.py`,
`scripts/run_eyes_v2_study.py`, `tests/test_eyes_v2.py`,
plus the family-registry additions in
`src/dashboards/compact.py`.

## Holdout-reuse caveat (recorded per the mandate)

This is the FOURTH grading use of the same 33-day holdout
(runs 82, 83, 84, now 85). The criteria are frozen and the
model class is selected on training-region OOF only, which
keeps the comparison honest, but cross-study selection
pressure on these 792 candles is now substantial: a pass
here is weaker evidence than any prior pass, and it is not
forward evidence. The clean tests remain forward ones
(Book #3's gated forward test is already running for the
gate; an eyes forward test would be a shadow-eyes swap,
which is Kass's explicit call).

## What would make Eyes v2 a failure

The zoo's OOF eyes stay at chance on the tradeable-move
labels (the triple-barrier target is no more learnable than
next-candle direction), or the selected eyes express no
stronger views than run 73 (spread stays inside 46–54%), or
they win the exam but not the money (the eyes-duel pattern
repeats), or they lose both bars. Any of those is recorded
plainly, with the full OOF table, exam table and duel
table, win or lose.

## Results

_(appended after the run — the design above was frozen first)_

**Registered as training run 85 (kind stacker, family
eyes_v2, promoted=0, pending_approval=0): FAIL — neither
frozen bar passed.** The new eyes express genuinely stronger
views than run 73 (that part of the design worked), and the
views are wrong more often than chance; paired with run 44's
brain they changed no money trade at all. Guard hashes (both
seats, gated seat incl. Book #3's gate model, lab champions,
all three live-state files) identical before/after; the
paper books were never opened.

**Frames and labels.** 12,032 common training rows (run 82's
12,079 finite-pick2 rows − 24 purge − 23 sequence warmup),
training base rate 0.490. Triple-barrier labels over the
full series: up-first 8,317 / down-first 8,482 /
neither-then-up 139 / neither-then-down 108 / tie-drops 0 —
the 1.0×ATR barrier is touched within 24 candles on ~98.6%
of rows, so the label is almost purely "which side of the
range breaks first," and it is balanced. Exam base rate
0.5013.

**Model zoo — training-region OOF only (n_oof 9,624, all
arms scored against the triple-barrier labels):**

| arm | balanced acc | AUC | accuracy |
|---|---|---|---|
| **(a) pick recipe, TB labels (SELECTED)** | **0.5120** | 0.5053 | 0.5088 |
| (c) pick recipe, OLD labels, new features | 0.5017 | 0.5158 | 0.5030 |
| (b) LightGBM, TB labels | 0.4936 | 0.4790 | 0.4935 |
| (d) LSTM sequence eyes (Kass's arm) | 0.4937 | 0.4879 | 0.4922 |

Selection per the frozen rule: arm (a). No fallback fired —
the LSTM trained cleanly in every fold (24s total; early
stopping bit quickly) and still scored below chance: the
24-candle sequence view added nothing on practice data, so
per the design it never reached the holdout. Arm (c)'s AUC
(0.5158) was the zoo's best ranking score but its balanced
accuracy at the 0.5 cut — the frozen selection metric — was
0.5017, a coin flip. The hierarchy-era tradeability
fragment (gate BA 0.613/0.620) does NOT replicate on these
triple-barrier labels either: the best OOF separation any
arm found is 0.512, within noise of chance at n=9,624
(±~0.010). Read plainly: on this feature frame, "which way
the next ATR-sized move goes" is no more predictable than
next-candle direction was.

**The exam (768 rows, identical timestamps, triple-barrier
labels):**

| | accuracy | precision | up-calls |
|---|---|---|---|
| Eyes v2 (arm a) | 0.4779 | 0.4881 | 672 |
| Run 73 re-scored | **0.5117** | **0.5117** | 426 |

**Exam win: NO** — the new eyes are BELOW chance (0.4779 on
a 0.5013 base rate) and below the incumbent. Their anatomy
is the mirror of run 73's conservatism: they call up on
87.5% of exam rows. Continuity diagnostics: run 73's
next-candle accuracy on this holdout is 0.5240 (its recorded
0.5354 was graded on its crowning night's fresher holdout;
the eyes-duel night read 0.5253 — same neighborhood).

**Vote spread (all 792 holdout votes) — the one design goal
that landed:**

| | min | max | % outside 45–55% | % outside 40–60% |
|---|---|---|---|---|
| Eyes v2 | 0.4355 | 0.6090 | **21.5%** | 0.13% |
| Run 73 | 0.4615 | 0.5421 | 0.0% | 0.0% |

The tradeable-move eyes DO express stronger views — a range
half again as wide, a fifth of votes outside the 45–55 band
where run 73 has none. Expression was never the binding
constraint, though: the wider votes carry no information
(exam accuracy below chance), so the spread reads as
confidence without content.

**The money test.** The two-way pin held: the fresh 44+73
champion arm reproduced runs 82–84's recorded aggregate
−210.17 / P&L −$305.28 exactly. The challenger arm (44 +
Eyes v2) graded **aggregate −210.17 / P&L −$305.28 —
identical to the champion to the cent**, same windows, same
fills. **Money win: NO.** The eyes-duel lesson repeats in
its purest form: even votes this different (672 vs 426
up-calls on exam rows, a materially wider spread) never
flipped run 44's argmax on a single duel decision that
mattered — its 21 features + portfolio state dominate the
one P(up) input at any spread this feature set can produce.

**Verdict: FAIL (neither bar).** Recorded as a failed eyes
attempt — a component study, not numbered among the program
nulls (run 84's precedent). Nothing was installed, crowned,
or seated; run 73 remains Lab Stacker Champion #4 and Book
#2's shadow eyes; Book #3's forward test is untouched. Per
the pre-registration, a shadow-eyes swap would be Kass's
explicit call — and on this evidence there is nothing to
swap in.

**What this closes and what it doesn't.** Eyes v2 tested
the two most promising remaining levers on the current
feature frame — a tradeable-move target and (via Kass's
arm) raw 24-candle sequences — and both came back at
chance, in OOF and on the holdout, while vote *expression*
proved easy to widen and worthless without information.
The bottleneck is the information content of the pick2_98
classic-indicator frame itself, not the label, the horizon,
the model class, or the vote scale. The levers named when
this study was ordered remain genuinely untested: new
information (funding / open-interest feeds) and
Heikin-Ashi-centered structure (Kass's parked follow-up).
Fourth use of this holdout — the caveat stands; nothing
here is forward evidence either way.

### Verification

- New adversarial suite tests/test_eyes_v2.py: 27/27
  (label paths hand-computed incl. both-barrier candle rule
  and horizon boundary; context + sequence causality under
  future garble; purge/feasibility row set; selection rule;
  lag-layout; LSTM smoke + invariance; compact registry).
  Watched failing 27/27 on the stub before implementation.
- Champion two-way pin: reproduced exactly in the duel
  phase (above).
- Dashboards: compact training + gate feeds pushed verbatim
  to both dashboards and read back — newest run 85
  "Eyes v2 — Tradeable-Move Stacker" FAIL, gate pending
  still run 83 (no new pending), parked 0.
- Machine record: data/eyes_v2/results.json; selected model
  data/eyes_v2/eyes_v2_model.pkl; holdout votes for both
  eyes under data/eyes_v2/.
- Full suite: 393 passed / 7 failed. Five failures are the
  documented pre-85 baseline (training-feed size sentinel,
  audit/telemetry registry pins, run-75 detail-window pin,
  gate-history pin); one is the known intermittent 1-ulp
  shadow-vote flake. The seventh —
  test_recipe_panel_matches_code_constants_two_ways — was
  aged by THIS registration: its tripwire pins the naive
  all-runs version walk's newest study-label pollution,
  which run 85's "Eyes v2" label ("2") superseded over the
  features-v6 runs ("6"). The protected feed value is
  unaffected (feat_v still "5.1", verified in the pushed
  feed); the pin was rebaselined in-study ("6" → "2",
  comment updated) and re-runs green, as do all 27
  tests/test_eyes_v2.py tests.
