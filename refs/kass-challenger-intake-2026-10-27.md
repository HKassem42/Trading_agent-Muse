# Kass's Challenger — Intake & Pre-Registration Template (prepared 2026-10-27)

Prepared in advance of Kass's own challenger proposal (he said 2026-10-26 evening he is
researching one and will bring it on 2026-10-27). Nothing here is built or green-lit;
it exists so his idea can be turned into a clean lab study the moment he describes it,
under the same rules as Muse's Pick: lab only, promotion lock on, every number logged.

## The bars any challenger must clear (current, verified 2026-10-27)

- **Stacker bar:** Lab Stacker Champion #3 = Muse's Pick Stacker, run 69 —
  holdout accuracy **0.5265**, precision 0.5167, 418/792 up-calls, scored-102 frame.
  Tonight's first defense (2026-10-27 midnight retrain): challenger run 71 scored
  0.5088 — crown held.
- **PPO bar:** live-seat champion is run 44's PPO (manual override, recorded verdict
  NOT PROMOTED under the two-part gate floor). Midnight 2026-10-27: challenger
  run 72 aggregate −1,629.46 vs champion +145.85, won 1/3 windows — champion kept.
- **Gate floor (Kass, 2026-10-22):** promotable only if formula pass AND non-hold
  ≥10% AND duel P&L > $0 AND P&L > buy-and-hold on the same windows.
- **Stacker anti-degeneracy check:** a win must be reported with precision and
  up-call count alongside accuracy (run 63's 0.5101 / 0.0000-precision collapse is
  the cautionary case). A constant-answer win does not count.
- **Grading:** identical holdout candles / identical duel windows as the champion's
  recorded grade; load-bearing numbers computed two independent ways.

## Intake questions (fill in when Kass describes the idea)

1. What changes — features/eyes, brain/algorithm, reward, sizing, or risk rule?
   (One lever per study; Muse's Pick changed the reward + one base learner and its
   PPO half's failure mode — day-scale training episodes vs 11-day duel episodes —
   is the standing warning about horizon mismatch.)
2. What is the hypothesis in one sentence — what should improve, and why?
3. Which champion does it duel — stacker (run 69), PPO (run 44), or both?
4. Pre-registered pass bar (fill BEFORE training): e.g. strict holdout win vs run 69
   with precision ≥ champion's and ≥100 up-calls; or PPO 3-window duel passing the
   two-part gate floor.
5. Kill criteria / what result closes the idea as a null?

## Archive-mining candidates available to fold in (Kass's call; refs/archive-mining-2026-10-27.md)

Ranked in the mining report:
1. Rolling Hurst exponent as a regime feature (not in the 188 screen or scored-102).
2. A SAC challenger brain (all 70+ brains to date are PPO; SAC's continuous actions
   would also unlock partial SCALE_OUT exits).
3. Volatility-targeted sizing (different lever from fixed-tranche size studies).
4. A 2.5% daily drawdown circuit breaker (from Kass's Master Deck).
5. Statistical-moment / time-of-day features (ranked last — the scored-eyes null
   says more information was not the binding constraint).

## Open decision for Kass — canonical source for future picks

The archive contained a Google Doc titled "Trading Bot ML/RL Training Combinations"
(ha.kassem42 share, modified 2026-10-26) — nearly verbatim the title Kass gave for
the Muse's Pick source. It was unreachable during the pick; the pick used the
spreadsheet Kass confirmed instead ("Machine Learning and Reinforcement Learning
Kinds, Definitions, and Use Cases") and stands as executed. The doc is a 6-section
catalog (HPO, meta-labeling, CPCV/Deflated Sharpe/PBO, shadow/WFE, telemetry,
feature families incl. LSTM + SAC) with NO literal reward-shaping row.
**Which document rules future picks is Kass's, pending.**

## Standing constraints

- Lab only; live seat (run 44 PPO + 21-feature stacker, Book #1) and shadow seat
  (run 44 PPO + run 69 eyes, Book #2) untouched by any study.
- A winning challenger is surfaced as PENDING Kass's decision — never installed.
- Declined/closed shapes are not re-offered: v6 stacker, run 67, the look-ahead-era
  trophy pair (PPO run 15 + Stacker run 10), reward shaping as designed in run 70.
