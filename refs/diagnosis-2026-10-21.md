# Lab Diagnosis Battery — 2026-10-21

Trading is PARKED. Everything below is lab work on the FIXED v5.1 feature
pipeline: no live run, no live-state or paper-DB writes, no promotions.
Raw results: `data/diagnosis/results.json` (script: `scripts/diagnosis.py`,
resumable, phase-marked; tests: `tests/test_diagnosis.py`).

Pre-flight: full suite 18/18 before the battery; 25/25 after (18 original
+ 5 diagnosis + 2 added the same day by the dashboard track).

## Methods

- **Data** (fetched once, cached to `data/diagnosis/sup.pkl` so restarts
  could not shift the windows): Coinbase BTC-USD 1h, 730d → 17,510 candles
  → **17,049 supervised rows** (2024-11-09 23:00 UTC → 2026-10-21 17:00 UTC).
  Production split: train = first 16,257 rows; holdout = last **792 rows**
  (from 2026-09-18 18:00 UTC), 3 gate windows of 264.
- **Self-checks (all true)**: features NaN-free; all features inside the
  [−1, +1] contract; train/holdout disjoint; family partition exact.
- **Stacker grading** mirrors production exactly (same
  `stacker.build_pipeline`, seed 42): fit on train → accuracy/precision on
  the 792-row holdout, plus `walk_forward_score` (TimeSeriesSplit, 5 folds)
  on the train frame. Holdout standard error at p≈0.5, n=792 ≈ **±1.8 pp**
  — differences smaller than ~3.5 pp on the holdout alone are not decisive;
  WF fold-mean SE ≈ ±0.4 pp on n≈16,257, but 11 variants were compared, so
  extremes of ±1 pp are expected from multiple comparisons alone.
- **PPO grading** is the standard duel scorer (`evaluate_windows` →
  `evaluate_ppo`): 3 windows × 10 deterministic episodes, next-candle-open
  fills, score = total P&L − 0.5 × max within-episode drawdown, aggregate =
  mean of window scores (recomputed independently from the stored window
  records — matched the stored aggregate exactly for all three arms).
  Buy-and-hold benchmark on this battery's windows: **+$3,944.19** total.
  (Run 17's B&H was +$4,478 on the earlier fetch's windows; the window set
  shifted by hours between fetches. All comparisons *inside* this battery
  use identical windows.)
- **PPO arms** are fresh trainings (seed 42, fixed inside
  `models.ppo_train`), ent_coef 0.05 (the production challenger setting),
  identical data/recipe; the only difference is the p_up array. Fresh, not
  warm-started: the run-17 champion was itself trained with true votes, so
  warm-starting both arms from it would have contaminated the constant arm;
  migrate_v51's production train was likewise fresh. The constant arm was
  also *graded* with a constant 0.5 vote — that is its deployment
  condition, so the comparison is system-vs-system.
- The 300k arm was trained in 12 checkpointed 25k chunks (identical
  env/broker/hyperparameters to `models.ppo_train`; continuation via
  `PPO.load` + `learn(reset_num_timesteps=False)`). Reason: the runtime
  SIGKILLed the battery process three times mid-arm with no traceback
  (logs stop mid-phase) — one uninterrupted 300k learn was not reliably
  runnable on this box. Total optimizer steps = 300,000; production's own
  nightly accumulates training in warm-start chunks the same way.

## Experiment 1 — Feature ablation

Families (a partition of the 21 features; Kass's five personal indicators
span volume_flow + kass_indicators):

| family | features |
|---|---|
| base_ta (10) | body_pct, upper_wick, lower_wick, range_pct, rsi_c, macd_hist_rel, dist_vwap, dist_sma50, price_change, volume_change |
| multi_tf (4) | h4_rsi_c, h4_dist_sma, d1_dist_sma, h4_atr_rel |
| volatility (2) | atr_rel, volatility_21 |
| volume_flow (3) | obv_slope, adl_slope, netvol_rel |
| kass_indicators (2) | ha_dir, ewo |

Results (holdout n=792; WF on train n=16,257). Recomputed full-feature
baseline: holdout **0.4609** / WF **0.5017** (run 16 stored: 0.4596 /
0.4930 — same verdict, slightly shifted windows):

| variant | holdout acc | holdout prec | WF acc | WF prec |
|---|---|---|---|---|
| full (21) | 0.4609 | 0.4669 | 0.5017 | 0.5007 |
| only_base_ta | **0.5341** | **0.5450** | 0.5072 | 0.5098 |
| loo_base_ta | 0.4646 | 0.4711 | **0.4873** | 0.4845 |
| only_multi_tf | 0.4899 | 0.4967 | 0.5033 | 0.5024 |
| loo_multi_tf | 0.5114 | 0.5196 | 0.5075 | 0.5099 |
| only_volatility | 0.4495 | 0.4657 | 0.5016 | 0.5025 |
| loo_volatility | 0.4899 | 0.4988 | 0.5010 | 0.5003 |
| only_volume_flow | 0.4975 | 0.5065 | 0.5046 | 0.5063 |
| loo_volume_flow | 0.4861 | 0.4946 | 0.4975 | 0.4962 |
| only_kass_indicators | 0.5013 | 0.5118 | **0.5118** | **0.5148** |
| loo_kass_indicators | 0.4684 | 0.4774 | 0.5001 | 0.4974 |

### (c) obv_slope vs netvol_rel duplication — CONFIRMED, near-total

n=17,049: **Pearson r = 0.99904, Spearman ρ = 0.99895** (both p ≈ 0.0),
sign agreement **99.71%**, mean |difference| **0.0016** on a [−1, 1] scale.
On gapless hourly Coinbase data the two features are the same signal to
three decimal places. One of them is a free dimension: it adds nothing and
costs model capacity. (Fixing this is a recipe change — discussion-stage,
not done here.)

### (d) Permutation importance of the full model (holdout, 30 repeats)

**All 21 features have negative mean importance.** Permuting any single
feature *raises* or leaves unchanged the full model's holdout accuracy:
least negative ewo / h4_rsi_c (−0.0012 ± 0.007), most negative lower_wick
(−0.0263 ± 0.009), macd_hist_rel (−0.0196), price_change (−0.0178),
rsi_c (−0.0172). The deployed-form model extracts no usable out-of-sample
information from any individual feature.

### Verdicts — Experiment 1

- **No family carries a robust, demonstrated edge.** Every variant's WF
  accuracy sits in 0.487–0.512 and every holdout number except one sits
  within ~2σ of a coin flip.
- **base_ta carries whatever faint signal exists.** It is the only family
  whose evidence points the same way twice: solo it posts the best holdout
  by a wide margin (0.5341, +7.3 pp over full — ≈1.9σ above chance on its
  own, suggestive not conclusive), and removing it produces the worst WF
  of any variant (0.4873) and the joint-worst LOFO holdout.
- **The v4/v5 additions dilute in combination.** The full model is the
  second-worst holdout performer of all 11 variants. Removing multi_tf
  improves *both* metrics (holdout 0.4609→0.5114, WF 0.5017→0.5075);
  removing volatility improves the holdout by +2.9 pp. Solo, multi_tf and
  volume_flow look WF-neutral (~0.503–0.505) — they are not individually
  toxic, they add variance the stacker spends on noise.
- **Kass's indicators (ha_dir + ewo)**: best WF of any variant solo
  (0.5118) but holdout only 0.5013 — and removing them moves the two
  metrics in opposite directions (holdout 0.4609→0.4684, WF
  0.5017→0.5001), both shifts well inside noise. Inconsistent across the
  two angles = no demonstrated standalone edge; at most a mild
  complementarity the current stacker can't exploit.
- **obv_slope ≡ netvol_rel** (r = 0.999): one is redundant, full stop.

## Experiment 2 — Stacker-vote A/B (100k steps each)

Cross-fitted training votes are nearly constant to begin with: mean
0.4990, **std 0.0142** (holdout votes from the champion stacker: mean
0.5001, std 0.0136; 20.0% of training votes are the neutral 0.5 by
crossfit construction). The "true" arm's advantage over a constant is a
±0.03 wiggle in one of 25 observation dims.

| arm | window scores | aggregate | total P&L | B&H | mean reward |
|---|---|---|---|---|---|
| true vote 100k | −853.48 / −3,700.49 / −275.85 | **−1,609.94** | **−$2,104.51** | +$3,944.19 | −0.749 |
| const 0.5 100k | −1,474.44 / −1,521.31 / −2,459.39 | **−1,818.38** | **−$2,699.26** | +$3,944.19 | −0.602 |

Action mixes (hold/buy/sell/short): true — w1 .62/.09/.17/.12,
w2 .75/.12/.05/.08, w3 .83/.00/.06/.11; const — w1 .54/.02/.08/.37,
w2 .64/.00/.13/.23, w3 .65/.00/.17/.18.

**Verdict: the vote neither helps nor hurts measurably.** The true arm is
nominally ahead (+208 aggregate, +$595 P&L) but both arms lose heavily to
buy-and-hold, the between-arm gap is small next to the per-window swings
(±$2k) of a single seed, and the mechanism caps the possible effect: a
std-0.014 input cannot drive a large difference. The stacker vote is not
why the PPO loses, and removing it would neither fix nor worsen the
brain — it would only simplify the observation. (Single-seed comparison;
a multi-seed A/B could shrink the uncertainty on the small gap, but no
plausible seed variation turns either arm profitable on these windows.)

## Experiment 3 — Learning curve (true vote, 300k vs 100k)

| arm | window scores | aggregate | total P&L | mean reward |
|---|---|---|---|---|
| true 100k | −853.48 / −3,700.49 / −275.85 | −1,609.94 | −$2,104.51 | −0.749 |
| true 300k | −1,089.56 / −4,756.98 / −5,143.26 | **−3,663.27** | **−$8,084.33** | −1.040 |

The 300k brain is worse in **all three windows**. Its policy is also
qualitatively different: hold share collapses (~0.62–0.83 → ~0.20–0.32)
and short share jumps to 0.36–0.46 — more training produced a busier,
more aggressively short policy that loses more. **Verdict: more steps do
not help; they hurt. Undertraining is refuted as the explanation for the
clean brain's failure.** (Sunday's 300k deep train will add a
warm-started point; nothing here suggests it will reverse the picture.)

## Bottom line

Three independent suspects were tested and cleared: it is not the
stacker vote (Exp 2), not training length (Exp 3), and not one bad feature
family (Exp 1 — no family's removal rescues the model, and permutation
importance says the full model uses *none* of its features productively
out-of-sample). What faint structure exists lives in the plain base-TA
features (solo holdout 0.5341), and even that is ~2σ on one 33-day
holdout with a WF of only 0.5072 backing it. The clean conclusion stands:
**this feature set + recipe shows no demonstrated edge on 1h BTC**, and
further tuning of the same recipe — more steps, vote surgery, family
pruning — is polishing a null. The remaining live options are the ones
already on the table for a joint decision: a genuinely different feature
family (the HA-sequence idea) or a different timeframe. The park
condition is unaffected by anything in this battery.

## Integrity & process notes

- Production artifacts untouched: `models/ppo_champion.zip` mtime
  2026-10-21 13:40 UTC and `models/stacker_champion.pkl` 12:36 UTC both
  predate the battery (~17:52 UTC start); `data/paper.db` mtime 14:19 UTC
  (parked-state era, before the battery). All battery artifacts live under
  `data/diagnosis/` (arm zips, chunk checkpoints, cached frame, votes,
  results.json). The battery never imported `src.trading.live` and never
  opened the paper ledger (all brokers ran on scratch DBs in the lab dir).
- The runtime killed the battery process three times mid-PPO-arm (no
  tracebacks; logs stop mid-phase). Phases/variants/chunks checkpointed to
  disk made each kill cost at most one 25k chunk; no result was recomputed
  or lost. Wall time was dominated by ~52 env-steps/s on the shared box.
- One design choice to flag: PPO arms were trained **fresh** rather than
  warm-started from the run-17 champion — required for a controlled vote
  A/B (the champion itself was trained on true votes) and consistent with
  migrate_v51's fresh production train. Grading used each arm's matched
  vote feed (see Methods).
