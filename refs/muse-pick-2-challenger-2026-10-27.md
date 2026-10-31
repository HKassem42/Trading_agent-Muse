# Muse's Pick #2 Challenger — Lab Study (2026-10-27)

Kass's order (2026-10-27 ~12:03 ET): "You go ahead and try to
beat your own champion first muse, take another stab at it and
call it Muse model #2 or whatever u called the first one but
#2."

The champion to beat is Muse's own Pick #1: its stacker (run 69)
was crowned Lab Stacker Champion #3 by Kass's approval on
2026-10-26 and sits in the shadow desk's eyes; its PPO (run 70)
lost. Pick #2 is a fresh challenger pair against that crown,
built under the same rules as Pick #1: lab only, promotion lock
on, every number logged, nothing installed — a win records
PENDING KASS APPROVAL.

## Pre-registered design (frozen before any grading — one shot each, no tuning sweeps)

### Eyes — "Muse's Pick #2 Stacker"

- Frame: Champion #3's scored-102 frame (94 percentile-scored
  columns, trailing-4,320-bar percentile machinery) PLUS four new
  causal features drawn from the archive-mining candidates
  (refs/archive-mining-2026-10-27.md, candidates 1 and 5):
  - **Rolling Hurst exponent** — rescaled-range (R/S) of log
    price, trailing **100-bar** window; H = log(R/S)/log(100),
    S the window's sample std (ddof=1); flat windows (S = 0)
    are NaN.
  - **Return skewness** — pandas unbiased rolling skew of log
    returns, trailing **100-bar** window.
  - **Return kurtosis** — pandas unbiased rolling excess
    kurtosis of log returns, trailing **100-bar** window.
  - **Return autocorrelation, lag-1** — rolling Pearson
    correlation of log returns with their 1-bar lag, trailing
    **100-bar** window.
  Each raw series is percentile-scored through the project's own
  trailing-4,320-bar `rolling_scores` (the identical function
  that scores every champion column), over the champion frame's
  scored positions (extended only if a new series' warmup
  demands it), no orientation flip. **98 feature columns.**
- Recipe: Pick #1's, unchanged — StandardScaler → SMOTE →
  RF + XGBoost + LightGBM + ExtraTrees → LogisticRegression
  (imported from `models.muse_pick`). The single varied lever
  on the eyes is the four new features.
- Grading: identical holdout timestamps (the last 792 supervised
  rows of a fresh 730d pull); Champion #3 (run 69, the installed
  `models/stacker_champion.pkl`) re-scored on the same
  timestamps through two independent paths that must agree
  exactly before any challenger number is trusted. Strict
  accuracy win; precision and up-call count on the record;
  walk-forward (5-fold TimeSeriesSplit) reported with the
  Pick #1 caveat (run 69's recorded WF 0.5022 was measured on
  Pick #1's train frame; tonight's WF is this study's frame).
- **Degeneracy rule (pre-registered):** a strict win with
  ≤ 7 up-calls or precision ≤ 0.05 is a **LOSS** (verdict FAIL,
  not pending) — the run 63 rule, applied as the intake's
  anti-degeneracy check demands.

### Brain — "Muse's Pick #2 PPO"

- Pick #1's reward shaping gets its first FAIR test. Pick #1's
  post-mortem named the confound: shaping penalized drawdown
  increases inside 24-candle training episodes, while the
  duel's damage compounds across a 264-candle window the reward
  never saw (day-scale caution did not compound into
  episode-scale caution; run 70 traded 7× the champion's rate
  into the falling third window).
- Design: SB3 PPO under the scored-eyes protocol — fresh brain,
  **60,000 steps, seed 42, ent_coef 0.05** — consuming the Pick
  #2 stacker's cross-fitted votes (pick recipe, training) and
  its fitted votes (holdout), state = the 98 scored features
  mapped s/50 − 1 + the vote + 3 portfolio readings (**102
  wide**). The shaped reward is Pick #1's, unchanged:
  step reward = standard dense MtM reward − 0.5 × (the step's
  increase in the current episode's drawdown, % points, floored
  at 0); SB3 telemetry capture on.
- **The one varied lever: training episodes span 264 candles**
  (11 days × 24) instead of 24, via `DuelLengthShapedTradingEnv`
  (src/models/muse_pick2.py), swapped into ppo_train for this
  study's train call only. The terminal +1/−1 therefore lands on
  the 264-candle episode's P&L sign and the activity penalty is
  the standard one; both are inherited unchanged and recorded
  here so the horizon change is fully described, not smuggled.
- Grading: the standard duel, pair vs pair — challenger pair
  (Pick #2 stacker + Pick #2 PPO) vs champion pair (run 44 PPO
  on prod21 windows + **run 69's** holdout votes), recomputed
  fresh on the same windows; 3 chronological windows × 10
  episodes, next-candle-open fills, score = P&L − 0.5 × maxDD
  per window, aggregate = mean of window scores. The gate
  formula (aggregate beats champion AND ≥ 2/3 windows) and
  Kass's two-part floor via `gate_decision()` (non-hold ≥ 10%,
  P&L > $0, P&L > buy-and-hold) are both reported exactly as the
  gate would judge them; under the promotion lock a pass
  records `pending_approval`, never an install.

### Governance

- LAB ONLY. Guard hashes (lab champions, live seat, shadow
  seat) verified bit-identical before/after; both paper books
  never opened by the study (read-hash info only — the live
  loops write them on candle closes).
- Tests first: tests/test_muse_pick2.py pins the new features
  against hand-computed / scipy second methods, the
  truncate-and-append invariance across all rows, the new
  columns' percentile scores against the builder's brute-force
  scorer, and the 264-candle env in lockstep against the
  production env. A causality failure stops the arm.
- Builder: `scripts/build_muse_pick2.py`; machine record:
  `data/muse_pick2/results.json`; registry family
  `muse_pick2` (dashboard family code 15), runs labeled
  "Muse's Pick #2 Stacker" / "Muse's Pick #2 PPO".

## Results

_(appended after the build — the design above was frozen first)_

Build: `scripts/build_muse_pick2.py`, checkpointed phases,
2026-10-27. Machine record: `data/muse_pick2/results.json`.
Feature module: `src/features/pick2_features.py`; env:
`src/models/muse_pick2.py::DuelLengthShapedTradingEnv` (swapped
into ppo_train for the study's train call only, as registered).

### Feature validation (tests BEFORE the build)

Two batteries, both green before any grading: the study battery
(`tests/test_muse_pick2.py`, 11 tests — R/S Hurst vs an explicit
plain-list hand computation at rel 1e-12, moments vs scipy's
unbiased estimators, autocorrelation vs manual Pearson,
truncate-and-append invariance, new-column percentile scores vs a
brute-force second scorer, 264-candle env lockstep incl. shaping,
family render) and an independent second battery
(`tests/test_pick2_features.py`, 7 tests — expanding-prefix
hand references in plain Python, causality across ALL rows,
two-estimator Hurst characterization). End-to-end causality was
re-proven inside the build on the real pull: the scored frame
rebuilt from truncated data at K = 10,000 and K = 17,018 matched
the full build exactly on sampled columns including Hurst, and
run 69's holdout votes reproduced identically on 200 shared rows.

### The data

Fresh pull 2026-10-27: 17,724 hourly candles (1,107 warmup +
16,617 supervised rows). Split: 12,080 train rows (4,176
warmup-dropped — the pull reaches further back than Pick #1's, so
the raw scored features carry longer context), 792 holdout rows
whose labels and closes are identical to the production holdout
by index.

### Eyes — WIN (genuine) → PENDING KASS APPROVAL (run 73)

Champion #3 re-scored TWO ways first (this study's frame path vs
nightly's `champion_holdout_frame`, the midnight retrain's own
path): input frames bit-identical, grades exactly equal,
probabilities within 1 ulp (max diff 1.11e-16 — the tree
ensembles sum votes in parallel, the same 1-ulp phenomenon the
Champion #3 crowning report recorded; logged in results.json).

| | Pick #2 Stacker (run 73) | Champion #3 run 69 (re-scored) |
|---|---|---|
| Holdout accuracy | **0.5354** | 0.5290 |
| Precision | **0.5250** | 0.5204 |
| Up-calls | **440 / 792** | 417 / 792 |
| Walk-forward accuracy | **0.5177** | 0.5022 (its own record) |
| Walk-forward precision | 0.5178 | — |

Strict accuracy win **+0.0063**, precision also higher, 440
up-calls — genuinely two-sided, nowhere near the pre-registered
degeneracy line (≤ 7 up-calls / precision ≤ 0.05). Effective win
= true. (Champion re-score 0.5290 vs its recorded 0.5265: the
holdout window shifted one day on the fresh pull; the comparison
is same-timestamps, same-frame.) Base learners standalone:
ExtraTrees 0.5455, RF 0.5366, LGBM 0.5354, XGB 0.5240; meta
coefficients ET +1.41, RF +0.75, LGBM −0.50, XGB −0.01. New-feature
importance ranks (of 102): **lag-1 autocorrelation #5**,
**kurtosis #7**, Hurst #46, skewness #47 — the moment features
carried weight; Hurst and skew were weak. Registered **run 73**,
promoted = 0, `pending_approval: true`. Artifact
`data/muse_pick2/stacker_pick2.pkl`, sha256
`89b633945843062a85d71bf8a2c97c238446ce5bd4fdfb783280daa2e75364a7`.

**VERDICT (eyes): WIN — pending Kass's decision under the
promotion lock. Nothing crowned, seated, or swapped.**

### Brain — LOSS (run 74); reward shaping fairly tested, still loses

Cross-fitted Pick #2 votes on train: mean 0.499, std 0.010
(neutral-led as designed); holdout vote means: Pick #2 0.502,
champion 0.500. PPO trained fresh, 60,000 steps, seed 42, on the
102-wide shaped state with **264-candle episodes** — the Pick #1
confound removed. Telemetry captured: 118 rollouts, entropy
1.3831 → 0.9520 (no collapse), final clip fraction 0.0668,
explained variance 0.0006, value loss 0.1586; shaped episode
reward improved −13.16 → −8.49 per 2,640-step block.

Duel (Pick #2 pair vs champion pair recomputed fresh, same
windows):

| Window | Pick #2 score | Pick #2 P&L | Champion score | Champion P&L | B&H |
|---|---|---|---|---|---|
| 1 | −1,086.99 | −$614.60 | +190.23 | +$312.79 | +$1,431.90 |
| 2 | **+428.27** | **+$469.73** | −114.46 | −$69.80 | −$4,792.55 |
| 3 | −2,556.41 | −$1,954.81 | −232.63 | −$80.87 | −$935.79 |
| **Total** | **−1,071.71** | **−$2,099.69** | **−52.29** | **+$162.12** | −$4,296.44 |

Windows won **1/3**; **formula pass: NO**; the gate's floor is
therefore not reached (`floor_fail = null` in gate semantics).
For the record: challenger non-hold 45.8% (≥ 10% ✓) but P&L
negative — it would have failed the floor anyway.
**Promotable: NO.** Action mix (final window): hold 0.542 /
buy 0.025 / sell 0.392 / short 0.042 (champion: 0.567 / 0.000 /
0.425 / 0.008). Max drawdown $1,203.19 vs champion $303.52;
0.167 round trips/day vs 0.067. Registered **run 74**,
promoted = 0. Artifact `data/muse_pick2/arm60/ppo_challenger.zip`,
sha256 `b5012a1e7ca0fe9f026361247755e00cb494eeece87fcea9588322a0d958734d`.

**VERDICT (brain): LOSS — the thirteenth trading null, and the
first FAIR test of Pick #1's reward shaping.** At duel-length
training episodes the −0.5 × drawdown-increase term still
produced a brain that lost two of three windows and bled
−$1,954.81 in the falling third window at 4× the champion's
drawdown. Pick #1's failure was not (only) the episode-length
artifact: the shaping hypothesis is tested at both horizons and
closed.

### Guardrails

- Promotion lock held: both runs non-promoted; the stacker win
  recorded pending Kass. Guard hashes (lab champions, live seat,
  shadow seat) verified bit-identical before/after by the
  builder's register phase; book DBs never opened for writing
  (all grading was simulation on the study holdout).
- Suite: **289 passed** (final full run, log-verified). Two
  honest re-pins in `tests/test_incorporation_feeds.py`, both
  recorded-fact pins that runs 73/74 legitimately changed:
  `test_audit_matches_registry_facts` (stacker pending 0 → 1;
  best-of-the-rest stacker accuracy 0.5101 → 0.5354, now above
  the champion while it pends — the auditor's luckiest-draw
  context working as designed) and
  `test_training_feed_carries_audit_and_telemetry` (telemetry
  block rows [72, 70] → [74, 72, 70]; "from" stays 70). Both
  re-pinned with dated comments, per the Champion #3 crowning
  precedent. The known intermittent 1-ulp flake
  (test_shadow_desk vote equality) failed in two of the
  mid-study full runs and passed in the final run and
  standalone — unchanged from its crowning-report status.
- Feeds: training + gate pushed verbatim to both dashboards and
  read back — All Models lists runs 73 and 74; gate pending =
  run 73 ("Muse's Pick #2 Stacker", WOULD PROMOTE — AWAITING
  KASS); gate challenger stays run 72, the newest NIGHTLY PPO
  challenger — lab-study runs surface via All Models + pending,
  never the duel chip (the Pick #1 precedent: run 70 likewise
  never displaced the nightly challenger).

---

## Addendum 2026-10-27 (later the same day): run 73 crowned Champion #4; shadow eyes swapped

Kass delegated the Pick #2 rulings to Muse at ~14:08 ET
("Do whatever you think is best..."). Muse's ruling, on the
evidence above (genuine strict win, precision 0.5250, 440/792
up-calls, AND walk-forward agreement — the point Pick #1's
winner lacked): **run 73 IS crowned Lab Stacker Champion #4**,
and the shadow desk's eyes swap to it.

- Run 69's lab artifacts were archived first to
  `models/archive/20261027T182106_20261027_pre_champion4/`
  (stacker sha 20705159775147d1d…, meta sha d0c1cf7c7472c637…).
- Run 73 installed as `models/stacker_champion.pkl` (sha
  89b633945843062a…, byte-identical to the study artifact) with
  a new meta carrying feature_set "pick2_98" — a new frame tag
  added to nightly's `champion_holdout_frame` dispatch the same
  day, proven end-to-end before tonight's retrain.
- Verification, two ways: the INSTALLED file re-scored cold on
  this study's holdout reproduces **0.5354 accuracy / 0.5250
  precision / 440 up-calls** exactly; graded again through
  nightly's own champion-frame path, identical accuracy and
  up-calls (max probability difference 2.2e-16).
- Registry: run 73 promoted=1; its pending_approval resolved
  APPROVED with the delegation provenance on the record
  (pending_decided_by "Muse (delegated by Kass 2026-10-27)").
  Run 69 is former Champion #3, in no seat. Roster status:
  run 73 = 1 (current lab champion; the roster's status 3 means
  lab champion AND in the DRIVER's seat, which run 73 is not),
  run 69 = 4, run 40 = 2.
- **Shadow swap (69 → 73), mid-position.** Book #2 at swap time
  (2026-10-27T18:21Z): **SHORT −0.311631 BTC @ $80,222.97**
  (opened 2026-10-27T14:00Z under run 69's vote, p_up 0.4961),
  cash $124,962.50, equity $99,909.26 (17:00Z mark). By the
  recorded ruling the book was NOT re-seeded; the swap and the
  position record ride in `data/shadow_seat.json`'s
  `eyes_swapped` entry (with the full swap history), the seat
  README documents the exception, and fills from the swap
  onward carry the `champion4_run73` context tag. Run 73's
  first live vote (17:00Z candle, verified read-only two ways):
  0.49654. The race pairing is now [44, 73]; the retired 44+69
  exam (+$162.12, +0.1621%, 33 days — this study's duel champion
  side) is recorded as pexam, attributed in the manifest's
  `wfe_exam` entry. The LIVE SEAT (Book #1) was not touched.
