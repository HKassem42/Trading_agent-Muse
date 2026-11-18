# Muse's Pick #14 — "The Committee" — Pre-registration

Written 2026-11-18 (~00:35 ET), BEFORE any candidate was fitted.
Commissioned by Kass on 2026-11-18 as dealer's choice ("run your
next muse pick on whatever you want... let's try to beat
everything we have ever done"), with one hard-thinking pass on
training a brain and stacker. LAB ONLY — win or lose, no seat,
book, or live file is touched on the strength of this document;
a PASS is ruled on by Muse under Kass's 2026-11-12 standing
crowning delegation, applying the standard below. This study
itself crowns nothing and installs nothing.

## The observation behind the pick

Every lab stacker champion since #6 comes from one recipe — the
flat Pick recipe (seed 42: scaler → SMOTE → RF + XGB + LGBM +
ExtraTrees → LR meta) on the ha10 frame — and the nightly
challengers of that same recipe win or lose their exams by 2–4
candles (run 137 won by ~2 and was declined; run 138 lost by ~4).
At that margin the exam is measuring seed noise, not skill: the
single seed's random forests / boosters land where they land.
The cheapest variance reduction available — with zero new
features, zero new hyperparameters, and zero new look-ahead
surface — is to average the recipe over several seeds. A
committee cannot invent information, but it can stop the exam
from being a coin flip about which seed happened to run.

## The candidate (ONE change only)

The exact Pick recipe (`build_pick_pipeline`) and the exact
ha10 frame (`build_ha_matrix`, unchanged), with the single seed
replaced by a **committee of 7 fixed seeds**:

    seeds = [42, 7, 123, 2024, 555, 9001, 31337]

Each seed runs the full recipe (scaler → SMOTE → RF + XGB +
LGBM + ExtraTrees → LR meta) on the same training rows. The
committee's P(up) is the **simple average of the 7 seeds'
meta-probabilities**. Nothing else changes: same exam, same
features, same recipe hyperparameters, same 0.50 threshold.

Feature set tag: `ha10ens`. Family: `muse_pick14`.

## The exam (identical to Picks #9–#13)

Saved frame `data/muse_pick9/raw_ohlcv.pkl` (17,510 rows) +
`prod_sup.pkl` (17,050 supervised rows); split at 16,258; the
exam is the last 792 supervised rows from 2026-10-03 19:00 UTC;
majority baseline 0.5088. Setup gates, run before any candidate
fit: the rebuilt ha10 frame must equal Pick #10's saved matrix
exactly; Pick #11's artifact must reproduce 0.5341 on the exam;
run 126 must re-score 0.5581 and run 123 must re-score 0.5518
on the identical exam rows — otherwise the study aborts.

Champion re-score caveat (carried from Picks #12/#13): runs 126
and 123 were trained on data covering this exam window, so
their re-scores are flattered; the candidate never saw the
window. The bar is nevertheless "beat the re-scored champion,"
the same frozen standard as Picks #12/#13.

## The bar (frozen)

The stacker PASSES only if BOTH hold:

1. Exam accuracy **> 0.5581** — Champion #13 run 126's re-score
   on this exam (0.5580808080808081 = 442/792; the committee
   needs at least 443/792).
2. Purged walk-forward accuracy **≥ 0.5234** — Pick #11
   run 110's recorded WF on this frame (0.5234264432029796;
   5 folds, 24-row embargo, the Pick #13 stage_wf convention,
   computed over the exam's training region with a full
   7-seed committee refit per fold). The committee must not
   buy exam points by giving back walk-forward.

Precision and up-calls are reported for the record, and each
individual seed's exam accuracy is reported alongside (the
noise band the committee averages over). Anything other than
both conditions is a FAIL.

## The PPO phase (only on a stacker PASS)

If — and only if — the stacker passes the frozen bar, a PPO
challenger is trained the standard nightly way, with the
committee standing in for the single stacker:

- warm-start from run 112 (models/ppo_champion.zip), 60,000
  steps, ent_coef 0.05, training region only (the current
  nightly frame: last 792 supervised candles held out);
- its training votes are **cross-fitted committee votes**
  (nightly `crossfit_p_up` semantics with the committee as the
  pipeline factory: TimeSeriesSplit 4 folds, each fold's votes
  from a committee fit ONLY on earlier rows, leading segment
  neutral 0.5 — no seed's model ever sees its vote rows in
  training);
- graded in the standard 3-window duel (3 × 264 candles,
  next-candle-open fills, score = P&L − 0.5 × maxDD) against
  the sitting champion pair — PPO run 112 reading Stacker
  run 126's votes; the challenger reads the committee's votes
  (a fresh committee fit on the duel frame's training region);
- the full gate applies: formula (aggregate beats champion
  AND ≥ 2/3 windows) AND floor (non-hold ≥ 10%, duel P&L > $0,
  P&L > buy-and-hold).

If the PPO also passes the full gate, BOTH runs are marked
pending_approval in the registry — and that is where this
study stops. No champion file is installed, no crowning is
executed, Book #5 is not re-pinned, no seat is touched: the
crowning ruling belongs to the parent agent delivering these
results. If the PPO fails, it is recorded NOT PROMOTED as
usual, and the stacker run stands on its own verdict.

On a stacker FAIL: no PPO is trained, nothing is installed,
the Pick is a FAIL, and the HA nightly line keeps retraining
exactly as now.

## Guards

SHA-256 of every champion/seat artifact (lab, live seat,
shadow, gated, HA, champ seats) is recorded before the study
and asserted unchanged after registration. Registration writes
exactly one stacker registry row (family `muse_pick14`) and —
only on a stacker PASS — exactly one PPO registry row, plus
one lab note covering the whole study.
