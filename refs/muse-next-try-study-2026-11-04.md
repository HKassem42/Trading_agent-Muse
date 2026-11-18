# Muse's Next Try — Study Report (2026-11-04)

Kass's order (2026-11-03 ~23:56 ET): pick the next model from start
to finish, any knobs, any process, new matrix/eyes/rules allowed,
beat the previous try, label it Muse's Next Try. GO.

Pre-registrations (both frozen before their grading):
refs/muse-next-try-preregistration-2026-11-04.md (+ Amendments 1–2)
refs/muse-next-try-2-preregistration-2026-11-04.md

## The idea

Every prior clean-era null read BTC's own spot candles. The one
untested lever in the program's own conclusion was NEW information.
Try #1/#2 add Binance perpetual positioning (funding, open
interest, long/short ratios, taker flow, basis — official Binance
Vision archive, OKX tail/gap stitch disclosed), ETH cross-asset
moves, the Master Score's 6h/24h trajectory, and calendar
cyclicals: 22 new causal columns on top of Pick #2's 98 (120 max).
The new Excel matrix: refs/Muse_Next_Try_Matrix.xlsx (also in the
goal's Library files).

Verification before any grade: 7 adversarial tests (hand-computed
funding/metrics/perp/calendar references, truncation-rebuild
causality over ALL rows) — they caught a pandas unit mismatch and
a wrong fixture expectation before training; the frame build's own
truncation checks caught nothing further. Three build bugs were
found and fixed pre-grading (perp label-alignment all-NaN trap,
Vision's missing 2026-10-28, metrics publication gaps) — all
logged in the pre-registration amendments.

## Try #1 — run 101 — FAIL (stands as recorded)

- Frame screening (train-only, fast proxy): control 98 → 0.5111,
  info 112 → 0.5141, seq/cal 106 → 0.5136, full 120 → 0.5152,
  prod21+info 35 → **0.5158 (selected)**.
- But the Pick recipe's own walk-forward on the selected frame:
  **0.4895** — the proxy ranked frames differently than the recipe
  does. The train-OOF threshold rule then fired at θ=0.42 and the
  holdout stance degenerated to 792/792 up-calls.
- Holdout (792 candles from 2026-10-02 06:00 UTC): **acc 0.4899**
  (at θ=0.50: 0.4482 — anti-correlated), precision 0.4899,
  WF 0.4895. Champions re-scored on the identical exam, two
  independent ways each, agreeing exactly: run 96 → **0.5265**
  (prec 0.5145, 447 up), run 73 → **0.5227** (prec 0.5115, 434 up).
  Control 98 eyes (ablation): 0.5101. Majority baseline: 0.5101.
- New-feature importance in the failed model was real (ETH Return
  #5, Perp Basis #7, Taker L/S #8 of 35) — the information is not
  noise to the trees; the frame/recipe pairing was wrong.

## Try #2 — run 102 — FAIL (recipe-selected frame, θ fixed 0.50)

- Selection by the Pick recipe's OWN 3-fold train walk-forward
  (identical 12,045 rows): control 98 → **0.5039 (selected)**,
  full 120 → 0.5028, info 112 → 0.4937. Read that again: with the
  real recipe as the judge, BOTH new-information frames ranked
  BELOW the control on train data. The proxy's preference for
  them in Try #1 was proxy-specific noise.
- Winner's 5-fold WF: 0.4992. Holdout: **acc 0.5101** (precision
  0.5000, 430/792 up-calls, windows 0.5152 / 0.5114 / 0.5038) —
  exactly the majority baseline (0.5101) and exactly Try #1's
  ablation-control grade, a reproducibility match. Still below
  both re-scored champions (0.5265 / 0.5227).

## Verdict on the new-information hypothesis

Measured three independent ways — fast-proxy CV, the real
recipe's walk-forward, and the holdout — derivatives positioning,
ETH cross-asset moves, score trajectory, and calendar effects do
NOT lift next-hour BTC direction above the existing eyes. The
positioning data is presumably already reflected in price within
the hour. This closes the "new information" lever AT THE 1-HOUR
HORIZON; the same slow signals (funding/OI extremes persist for
days) remain untested at a 24h horizon, which would be a
different target and a different exam — Kass's call, not assumed.

Side note: the same night's regular retrain produced run 100
(stacker 0.5341 on its own exam, PENDING Kass) and run 103 (PPO,
lost 0/3). Those are the nightly's, untouched by this study.

## Bottom line for the leaderboard

Clean-era best holdout remains **0.5354** (run 73, Muse's Pick #2,
tied by run 78) on its original exam; on the fresh Nov-4 exam the
re-scored champions read run 96 → 0.5265, run 73 → 0.5227.
Nothing was promoted, seated, or installed; champion/seat/shadow/
gated files hash-verified unchanged across both tries. The
derivatives-positioning dataset itself (funding/OI/ratios/taker/
basis, 730 days, hourly-aligned) is now built and cached in
data/muse_next_try/external/ for any future study.
