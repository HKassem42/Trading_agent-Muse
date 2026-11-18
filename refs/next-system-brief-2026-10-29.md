# Next System — design brief staged at Kass's direction (2026-10-29)

Kass, 2026-10-29 ~19:32 ET: "think about creating a different system
when we get to model 80 and see if we can do better, let's let our
live traders have more time."

Status: DESIGN SEEDS ONLY — nothing built, nothing green-lit beyond
thinking. Registry is at run 79; run 80 arrives with tonight's
midnight retrain. Live traders (Book #1 seat, Book #2 shadow) stay
exactly as they are and keep accumulating track record meanwhile.

## What fourteen nulls + the eyes duel say the new system must respect

1. **Direction from classic indicators is a coin flip** on 1h BTC —
   stacker exams top out ~0.53, every PPO variant lost its duel.
   A new system that is just a better direction-guesser repeats the
   program's central failure.
2. **The one surviving fragment: TRADEABILITY was predictable**
   (two-step hierarchy, 2026-10-24: gate balanced acc 0.613/0.620
   while direction scored 0.5014/0.4952). Knowing WHEN a move is
   worth trading beat knowing WHICH WAY. Meta-labeling in Kass's
   RLML Combinations doc is the same idea from the literature side.
3. **Soft inputs get ignored** (eyes duel, 2026-10-29): run 44 made
   identical money trades under two very different vote series. If
   a signal matters, wire it as a gate/veto/size decision — never
   as 1 of 25 observation numbers.
4. **Fees bury turnover** (hierarchy-focus: +$3,986 gross, −$7,041
   fees). Low turnover by design; a trade must clear its fee hurdle
   before it exists.
5. **The risk layer is the part that demonstrably works** — ATR
   stops, trailing exits, 75% cap delivered the live books' green
   October. Keep the production broker/risk rules as the chassis.
6. **Kass's original intent is still unbuilt**: midnight learning
   from the bot's OWN live trades (experience replay), not market
   candles. The live ledger now holds real decisions+outcomes
   (Book #1: 32 fills) and grows every week the traders run — the
   "give them more time" instruction directly feeds this dataset.
7. **Closed doors stay closed**: reward shaping (tested twice),
   exposure-sizing-as-fix, v6 features, trophy pair, runs 61/63/67/78.

## Candidate shapes to bring to Kass when run 80 lands

- **A. Meta-label / tradeability gate system**: a primary signal
  (even a simple one) proposes trades; a learned filter — trained to
  predict "does THIS trade make money after fees" — vetoes or sizes
  them. Directly builds on fragment #2 + lesson #3.
- **B. Experience-replay learner**: nightly training on the live
  ledger (the bot's own fills, P&L-attributed) per Kass's original
  design; self-selected-sample hazard managed with the simulated
  episodes as a control group.
- **C. SAC challenger brain** (from Kass's RLML catalog; the program
  has only ever run PPO) — continuous sizing instead of discrete
  actions, wired through the same gate.
- **D. Regime-conditional ensemble**: different small specialists per
  volatility regime with a hard drawdown circuit breaker (2.5%/day,
  from the archive-mining candidates).

## External benchmarks (research 2026-10-29, for the run-80 design)

Full report: Library artifact "Best Automated Trading Systems — The
Numbers" (research_notes/best-automated-trading-systems-benchmark-20261030-0023).

- Best ever: Renaissance Medallion — 39.1% net avg/yr 1988–2018,
  Sharpe >2, win rate just **50.75%** of trades. Even the greatest
  system ever built wins barely half its trades; its edge is sizing,
  diversification across thousands of signals, and cost control.
- Sustained professional bands: Sharpe 1.0–1.5 = excellent
  (Two Sigma Spectrum 1.37, D.E. Shaw Composite ~1.33 filed period);
  1.5–2.0 world-class; **>3 sustained = presume overfit** until
  deflated-Sharpe/PBO and live proof say otherwise.
- Honest ML/RL out-of-sample: BTC direction accuracy that survives
  is **52–55%** (our stackers at 0.525–0.535 already sit in this
  band); best RL backtest Sharpes ~1.0–1.3 with no live replication;
  rigorous walk-forward RL ≈ Sharpe 0.33, not significant.
- Retail bot claims (3Commas/Pionex etc.) are marketing-grade: no
  audited forward records exist; 74–89% of retail CFD accounts lose.
- Realistic target for a solo BTC/USD hourly system: walk-forward
  net Sharpe 0.5–1.0 sustained (1.0–1.5 excellent), beat buy-and-hold
  risk-adjusted with materially smaller drawdowns rather than on raw
  return, and treat backtest Sharpe >2 / accuracy >65% as a bug.

## Ground rules carried over

Pre-registered design before any build; adversarial tests first;
the 3-window gate + two-part floor unchanged; promotion lock
unchanged; lab only until Kass rules. Validation should add CPCV
(combinatorial purged CV) from the Combinations doc so the new
system's first grades are harder to fool than the old ones were.
