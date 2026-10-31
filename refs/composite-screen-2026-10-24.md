# Matrix v3 composite screen — results (2026-10-24)

Executed **exactly** as pre-registered in
`refs/composite-screen-plan-v3-2026-10-23.md` (Kass green-lit
2026-10-24 ~00:31 ET). Features: the four v3 composites only
(`score_master`, `score_trend` 45 cols, `score_momentum` 25,
`score_volume` 7; membership frozen as built). Machinery:
`src/models/composite_screen.py`, driver
`scripts/build_composite_screen.py`, raw records in
`data/composite_screen/results.json`.

**Verdict up front: NULL — the sixth independent null on 1h BTC.**
Both duel-graded arms fail all three pre-registered verdict rules.
Per the plan, the thresholds, families, and arms are not re-tuned
for a second shot; the book on v3 composites is closed unless Kass
directs otherwise from the drawing board.

## Which champion graded the duel

`models/ppo_champion.zip` as it stands now: **run 44's PPO brain,
installed 2026-10-24 by Kass's manual override** (sha256
`e610ddf3…`; the recorded run-44 gate verdict remains NOT PROMOTED —
the override is Kass's, documented in the lab notebook). Graded
fresh through the same harness as the arms:

| Champion (run 44 brain) | Standard windows |
|---|---:|
| Aggregate score | −662.43 |
| P&L net / gross / fees | −$994.49 / −$318.83 / $675.65 |
| Buy-and-hold, same windows | +$717.54 |
| Round trips / non-hold share | 4 / 45.1% |
| Robustness windows: agg / P&L / B&H | −4,805.59 / −$10,750.15 / +$27,401.66 |

Harness proof: the duel mirror
(`composite_screen.evaluate_windows_composite`) reproduces
`training.nightly.evaluate_windows` on this champion **exactly**
(aggregate diff 0.0000, P&L diff 0.0000) before any arm number was
trusted.

## Walk-forward (TimeSeriesSplit 5 folds, 15,961 pre-duel rows; 13,300 graded rows)

Accuracy is shown next to the 0.6690 majority baseline and balanced
accuracy next to 0.33 chance, per the plan.

| Arm | Accuracy | vs 0.6690 | Balanced acc | Fold BA spread (5 folds) |
|---|---:|---:|---:|---|
| Rule–Master | 0.3778 | −0.2912 | 0.3624 | 0.343 – 0.373 |
| Rule–Trend (diag) | 0.3552 | −0.3138 | 0.3589 | 0.345 – 0.366 |
| Rule–Momentum (diag) | 0.3719 | −0.2971 | 0.3559 | 0.337 – 0.368 |
| Rule–Volume (diag) | 0.3717 | −0.2973 | 0.3632 | 0.351 – 0.374 |
| GBM–4 | 0.4571 | −0.2119 | 0.3636 | 0.342 – 0.376 |

Every arm's balanced accuracy sits consistently a little above
chance — all 25 arm-folds land above 1/3, spread is tight — so a
small ranking tilt exists. But every arm's raw accuracy is 21–31
points **below** just-always-saying-HOLD: the rules spend 64–68% of
rows predicting BUY/SHORT against a label set that is 66% HOLD.
The tilt does not survive contact with the class mix.

## Duel — standard windows (B&H +$717.54; champion agg −662.43)

| Arm | Agg | P&L net | P&L gross | Fees | Trips | Non-hold | Won |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule–Master | −7,716.07 | −$19,629.30 | −$6,174.42 | $13,454.88 | 66 | 80.0% | 0/3 |
| GBM–4 | −4,932.52 | −$12,468.83 | +$584.70 | $13,053.53 | 117 | 52.2% | 0/3 |

Per-window scores (arm vs champion −2,361.1 / +312.2 / +61.6):
Rule–Master −4,454.8 / −6,776.8 / −11,916.6; GBM–4 −7,378.1 /
−3,389.3 / −4,030.3. Rule–Master closed 67 segmented trips with
**0 winners**. GBM–4 closed 118 segments, 26W / 92L.

## Duel — robustness windows (older set; B&H +$27,401.66)

| Arm | Agg | P&L net | P&L gross | Fees | Trips |
|---|---:|---:|---:|---:|---:|
| Rule–Master | −3,062.64 | −$6,438.85 | +$5,193.86 | $11,632.71 | 49 |
| GBM–4 | −7,770.34 | −$19,518.28 | −$6,142.60 | $13,375.68 | 116 |

GBM–4 was refit on rows strictly before the robustness region
(out-of-sample there), mirroring the lag build's discipline;
Rule–Master has no fit (its scores are causal trailing percentiles).

## Verdict, rule by rule (pre-registered)

**Rule–Master (run 45) — FAIL on all three rules.**
1. Walk-forward: **FAIL.** Balanced accuracy passes its half
   (pooled 0.3624 > 1/3, 5/5 folds > 1/3) but accuracy 0.3778 is
   below the 0.6690 majority baseline.
2. Duel + floor: **FAIL.** The relative formula itself fails
   (aggregate −7,716.07 does not beat the champion's −662.43;
   0/3 windows won), so the production `gate_decision` never
   reaches the floor. Independently, P&L −$19,629.30 < $0 and
   < B&H would have failed the floor anyway.
3. Robustness: **FAIL.** P&L −$6,438.85 < 0.

**GBM–4 (run 46) — FAIL on all three rules.**
1. Walk-forward: **FAIL.** Balanced accuracy 0.3636 (5/5 folds
   > 1/3), but accuracy 0.4571 < 0.6690.
2. Duel + floor: **FAIL.** Formula fails (aggregate −4,932.52 vs
   champion −662.43; 0/3 windows); P&L also < $0.
3. Robustness: **FAIL.** P&L −$19,518.28 < 0.

**Diagnostics (runs 47–49, walk-forward only by design):**
Rule–Trend / Rule–Momentum / Rule–Volume show the same shape as
the master rule (balanced 0.356–0.363, accuracy 0.355–0.372) —
no family is hiding a signal the master missed, and per the plan
none of them was duel-graded.

## Fee transparency (the plan's required lens)

The size study's disease, again: **GBM–4 on the standard windows
was +$584.70 before fees** — breakeven gross — and −$12,468.83
after $13,053.53 of fees on 117 round trips. Rule–Master's
robustness set was +$5,193.86 gross → −$6,438.85 net. But fees are
not the whole story this time: Rule–Master was negative even
pre-fee on the standard windows (−$6,174.42, zero winning trips),
and GBM–4 was −$6,142.60 pre-fee on the robustness set. Churn
taxes these signals; on half the cells there was nothing to tax.

## Registration

Runs **45** (Rule–Master), **46** (GBM–4), **47** (Rule–Trend),
**48** (Rule–Momentum), **49** (Rule–Volume) — family
`composite_screen` (models-index code 7, display names
"Composite Screen — …"), all **promoted=0**, one summarizing lab
note. Training feed after registration: 6,982B at 49 runs (under
the 7,100B sentinel; no rebaseline needed). Nothing was promoted —
under the plan even a full pass would have been a report to Kass,
not a promotion.

## Integrity

Champion files untouched (ppo_champion.zip / stacker_champion.pkl
mtimes identical start to finish); all duel brokers ran on
scratch DBs in /tmp; dashboards not pushed (the 15-minute live
job syncs). Test suite: **147 passed** (134 at start + 6 new
composite-screen tests + 7 stop-mirror tests landed by the
parallel un-park work). The composite-screen tests pin the 60/40
boundaries (60.0 is BUY, 40.0 is SHORT), v3↔v2 row/label/OHLC
identity, and the net = gross − fees identity against the real
broker, long and short.

## Deviations

- **From the plan: none.** Features, arms, thresholds,
  hyperparameters, windows, verdict rules all as written.
- **Interpretation disclosed:** the plan's rule 1 says balanced
  accuracy must beat chance "by a margin that survives the fold
  spread (not one lucky fold)" without a formula. Applied as:
  pooled BA > 1/3 AND ≥ 4 of 5 folds individually > 1/3. Both
  duel arms in fact went 5/5 on folds and still failed rule 1 on
  the accuracy condition, so the interpretation decided nothing.
- **Context, not deviation:** the plan's "trading stays parked"
  line was overtaken by Kass's manual override hours before this
  run (bot live, run 44 brain champion). The screen stayed lab-only
  regardless; paper.db / live_state.json changed only under the
  live loop, never under this build.
- **Collateral repair (outside the plan, flagged for the parent):**
  the champion override had re-broken the size study's two pinned
  identity tests — `size_study.load_model("ppo")` read the LIVE
  champion file, the same live-input flaw pre-flight Finding 1
  fixed for its stacker votes, one level deeper. The suite now
  loads run 17's brain from `data/size_study/frozen_ppo_champion.zip`
  (sha256 `7d6ef059…`, byte-identical to the archived pre-override
  champion) via the `FROZEN_PPO` constant in
  `src/models/size_study.py`, and both tests pass with their
  **original** pins (−2,037.2361 / −2,062.9741) unmodified. During
  the repair window the suite transiently showed unrelated failures
  while the parallel un-park workstream was mid-edit on the
  dashboard stop mirror; final state is 147/147 green.
