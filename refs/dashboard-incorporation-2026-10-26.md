# Dashboard Incorporation — Method Note (2026-10-26)

Kass approved (2026-10-26) incorporating six frameworks from his
"Architecting the Ultimate Quantitative Trading and MLOps
Dashboard" research doc, plus applicable findings from Muse's RL
research (`~/workspace/research_notes/rl-trading-what-works-
20261019-2259/report.md`). Display + telemetry only: no trading
decision, gate, promotion, sizing, or model artifact changed.
This note records the methods, the assumptions, and — where the
stored data could not honestly support the doc's formula — what
shipped instead and why.

## 1. Friction + risk proximity (Live)

- Fees: summed verbatim from the production ledger (all-time and
  trailing 30 days, wall clock).
- Decision-to-fill drift: per fill, fill price vs the decision
  candle's close. The close is read from the equity ledger's
  `price` at the fill's timestamp — the live loop snapshots the
  same close it fills at, so this is the decision price by
  construction, not a proxy. Fills with no matching equity row
  are excluded and counted (`excl`). bps = (fill − close)/close ×
  10⁴; $ is signed as a cost (paying up on a buy / receiving less
  on a sell is positive). **Today drift is exactly 0.00 bps on all
  29 fills: the paper broker fills AT the decision close.** That
  is the honest readout — the panel caption says so, and states
  the real friction is fees ($1,518.53 all-time). A live account
  would add spread/slippage the paper engine does not simulate.
- Risk proximity: all-time equity peak from the equity ledger;
  cushion above the $95,000 alert floor ($ and %); exposure =
  |BTC| × price / equity vs the 75% cap. Distance-to-stop is NOT
  duplicated (it lives in Position).

## 2. Walk-Forward Efficiency (Shadow)

- Exam side: read from the registry, never typed in — the
  champion side of the newest `scored_eyes_ppo` study run. Runs
  65/66 graded exactly the Shadow pairing (PPO run 44 + Champion
  #2 run 64 eyes) on the shared 33-day holdout: score +39.76,
  P&L +$461.81 on $100k → exam return +0.4618% over 33 days.
  (Verified two ways: refs/scored-eyes-ppo-study-2026-10-25.md
  table and run 65/66 `metrics.champion` in training.db.)
- Live side: Book #2 equity since its 2026-10-26T01:00 UTC start.
- The doc annualizes both sides before dividing; annualizing a
  sub-day sample is meaningless, and equal-window annualization
  cancels in the ratio anyway. The feed therefore carries the
  raw returns and computes `ratio = live_pct / exam_pct` ONLY
  when it can mean something: ≥ 1 live fill AND ≥ 7 live days
  AND a positive exam. Until then ratio is null and the panel
  shows the components + "not yet meaningful — N days / M
  fills". Bands (doc's): >80% exceptional, 50–80% proceed,
  30–50% borderline/halt, <30% reject.

## 3. Overfitting Auditor (Training) — why no DSR

The doc's auditor headline is the Deflated Sharpe Ratio. DSR
needs, per Bailey & López de Prado: the champion strategy's
per-period return series (for its Sharpe, skewness, kurtosis and
T) AND the cross-sectional variance of Sharpe ratios across all
N trials. Our registry stores neither: PPO runs store duel
aggregates (score = P&L − 0.5·maxDD over 3 windows, plus the 3
window sub-scores) and stacker runs store holdout accuracies.
Three window aggregates are not a return series, and duel scores
are not Sharpe ratios; a "DSR" computed from them would be a
fabricated statistic wearing the formula's clothes. **Not
shipped, deliberately.**

Shipped instead (all read verbatim from stored records):
- The trial ledger: total runs (68), per brain — tries /
  promoted / declined by Kass / pending Kass.
- Best-of-N context: each champion's headline score printed next
  to the trial count that produced it, the median challenger
  score, and the best-of-the-rest score. Distributions cover
  post-v5.1-fix runs only (pre-fix grades were earned with
  leaked features and are not comparable draws); the counts keep
  pre-fix runs, since they consumed research degrees of freedom.
- What it proves: the champion is the maximum of N noisy draws,
  so its score overstates expected live performance; the median
  try is the honest center. Today's sharpest examples, both
  surfaced deliberately rather than filtered out: the stacker
  champion (run 64, 0.4975) is NOT the post-fix maximum — run
  63's degenerate majority-class 0.5101 is, and Kass declined it;
  and the PPO best-of-the-rest is 0.0 — run 20's HA-seq
  challenger, which scored P&L $0 − 0.5 × $0 drawdown by never
  trading at all (the vacuous pass the two-part gate floor was
  built to block). Degenerate maxima are the auditor's best
  teaching examples; the median PPO duel score (−1,958.2) is
  the honest center next to them.
- Future path to a real DSR/PBO: persist per-window (or
  per-episode) return series for every graded run, then compute
  Sharpe per trial and deflate. Recorded as a finding, not built.

## 4. Drift monitor (Live + Shadow) — PSI

- Method: `src/features/drift.py`. Bins are the BASELINE's
  deciles (quantile edges, deduped), applied identically to both
  samples; PSI = Σ (a − e)·ln(a/e) over bin shares, zero shares
  floored at 1e-4 (standard). Hand-computed toy cases pin the
  implementation in tests/test_drift_psi.py.
- Windows: live = last 168 closed hourly candles (7 days);
  baseline = the 792 closed candles (33 days) ending at the last
  closed candle at or before the newest registry grading — i.e.
  the current grading holdout window, recomputed through the
  production pipeline (`add_features` for the 21; the scored
  frame builder for Champion #2's set). Causality: closed
  candles only; both windows come from the same fetch + pipeline
  as live inference, so train/serve skew cannot masquerade as
  drift.
- Degenerate baselines (near-constant feature → < 2 usable bin
  edges) report "no baseline" (null), never a number.
- Known property, stated plainly: nightly re-grading makes the
  baseline RECENT (the holdout is always the freshest 33 days)
  and the live week overlaps its tail. PSI here therefore reads
  as "this week vs the exam window" regime shift, not "live vs
  ancient training data". Bands (doc's): < 0.1 stable,
  0.1–0.25 moderate, ≥ 0.25 significant. Drift flags regime
  change; it does not predict losses — the panels say so.
- Shadow scope: worst-10 of the 94 scored features + band counts
  (the full grid would break the shadow snapshot's size budget);
  the panel labels the scope.

## 5. Per-fill attribution (Live) — why not SHAP

The doc asks for SHAP waterfall attribution per trade. Two
blockers, both verified 2026-10-26: (a) `shap` is not installed;
(b) more fundamentally, the decision-maker is the PPO (a neural
policy over 25 inputs) and the explainable model is the stacker
— a StackingClassifier whose meta-learner (logistic regression)
composes three base learners' out-of-fold votes. TreeSHAP on the
base learners does not compose through the meta-learner into an
exact additive split of the vote, so any "SHAP" numbers would
fail the additivity check (attributions + base value = vote)
that makes SHAP meaningful.

Shipped (the sanctioned fallback): per fill, the top-5 features
ranked by |squeezed value at the fill| × the seat stacker's
global importance (mean of the three fitted base learners'
`feature_importances_`, each normalized; extracted from the
actual seat artifact `models/live_seat/stacker.pkl`). Direction
= sign of value × importance (pushed the vote up/down), shown
with plain-unit values. The panel labels these "strongest
signals at this fill", states they explain the STACKER's vote,
and that the PPO made the trade call weighing that vote against
all 25 inputs. Not SHAP, not causal — and labeled as such.

## 6. PPO telemetry (Training) + RL research integration

- Past runs: SB3 telemetry was NEVER persisted (verified: no
  entropy/approx_kl/clip_fraction/explained_variance/value_loss
  key in any of the 68 runs' metrics). The panel shows "Not
  recorded" for them. No reconstruction was attempted — entropy
  histories cannot be re-derived from a saved policy.
- Capture (logging-only): `_TelemetryCallback` in
  `src/models/ppo_train.py` reads `model.logger.name_to_value`
  at rollout end (+ training end for the final dump) and writes
  a summary {ent_start, ent_end, ent_min, kl_end, clip_end,
  ev_end, vl_end, rollouts, ent_curve ≤24 pts} as a sibling key
  in `ppo_challenger_curve.json`; `nightly.py` lifts it verbatim
  into the PPO run record (the Sunday deep train runs the same
  `nightly.main` + `train()` path, so it is covered). Note SB3
  logs `entropy_loss` = −entropy; the callback stores entropy
  itself. One-rollout logger lag is documented in the callback.
- Outcome-neutrality proof: tests/test_ppo_telemetry.py trains
  the same seed twice (telemetry on/off) and asserts the
  learning curves are BIT-IDENTICAL and the final policies'
  action distributions match to 1e-6. The callback reads logger
  state only; it never touches the env, the model, or any RNG.
- RL research findings applied: (i) entropy-collapse caption
  tied to the three 15m-era no-trade convergences (Briola et
  al. 2021: sparse reward → "not trading at all"; our reward is
  now the dense per-step MTM the research ranks #1 — telemetry
  watches the failure mode the research predicts); (ii) the
  panel's single-seed note (ava-28: seed std up to 1.85 Sharpe;
  ≥5 seeds is the discipline — our nightly is single-seed 42).
- NOT built (data does not exist): multi-seed evaluation
  variance display. Nightly grades are single-seed; there is no
  seed spread to show. Recorded here as a finding for Kass: if
  he wants it, the nightly PPO duel would need k-seed grading
  (a training-behavior change — his call, not a dashboard one).

## Feed sizes (2026-10-26, after)

trading 2,136B (unchanged) · ledger 1,522 → 1,709B · vision
1,698 → 1,920B · fillctx 416 → 859B · training 11,078 →
11,225B (budget 16,384B) · shadow snapshot and race: measured at
push time in the build report.
