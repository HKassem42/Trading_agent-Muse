# Scored-Eyes PPO — Lab Study (2026-10-25)

**Status: COMPLETE — NULL RESULT (eleventh independent trading null).**
Approved by Kass 2026-10-25 from his question: the PPO sees only 25
numbers (21 raw features + the stacker's single vote + 3 portfolio)
while stacker Champion #2 sees 94 scored indicators — shouldn't the
brain have the same eyes? This study gave the PPO the stacker's full
scored view. Lab-only under the promotion lock; the production
feature path, both champions, and the driver's seat were never
touched (guard hashes verified before/after, below).

## Construction

- **State = 98 numbers**, all in [-1, 1] under the same state contract
  as production: Champion #2's (run 64) **94 scored features**
  transformed `score/50 - 1` + the stacker vote centered `2p - 1`
  + the same 3 portfolio numbers (cash share, BTC share, unrealized
  % clipped) the production state carries.
- **Vote sourcing (identical both sides of the duel):** training rows
  use cross-fitted votes from a pre-holdout stacker (same
  cross-fitting discipline as production, so no in-sample vote leak);
  holdout/duel rows use run 64 itself. The champion side was graded
  on its normal prod21 25-wide state **plus the same run-64 votes**,
  so the only difference between the sides is the market-feature
  block (94 scored vs 21 raw).
- **Two fresh arms** (no warm start exists across state shapes):
  60k steps ("60k Fresh", nightly scale) and 300k steps in 25k
  chunks ("300k Deep", Sunday scale). Same reward (per-step MTM %
  + terminal ±1), same broker, same everything else.
- Data: fresh 730d 1h frame; shared holdout = last 33 days
  (792 candles, from 2026-09-23), fenced identically to production.

## Causality validation (v5.1 standard)

Truncation checks at rows 9,000 and 17,018: recomputing the scored
frame on truncated data reproduces the full-frame values with
**max_abs_diff 0.0** for every scored feature (votes agree to
1.1e-16). Vote distributions sane: train crossfit mean 0.499 /
std 0.013; holdout (run 64) mean 0.499 / std 0.009.

## Training & grading (same windows, same rules as production)

Standard 3-window duel (3 chronological windows × 264 candles,
10 episodes each, next-candle-open fills, score = P&L − 0.5×maxDD),
then Kass's two-part floor (non-hold ≥ 10%, P&L > $0, P&L > B&H).
Buy-and-hold on these windows: **−$7,353.41** (−1,988.51 / −3,798.46
/ −1,566.43) — a falling holdout.

| Side | Agg score | P&L | MaxDD | Windows (score) | Mix (last window) |
|---|---|---|---|---|---|
| Champion (prod21 + run-64 votes) | **+39.76** | **+$461.81** | $350.63 | +284.85 / −114.46 / −51.11 | hold .56 / buy 0 / sell .43 / short .01 |
| Run 65 — 60k Fresh | −2,016.57 | −$4,056.09 | $1,710.59 | −2,230.17 / −944.61 / −2,874.93 | hold .37 / buy .05 / sell .42 / short .15 |
| Run 66 — 300k Deep | −1,660.15 | −$2,850.15 | $1,618.74 | −1,193.62 / −1,351.35 / −2,435.49 | hold .20 / buy .12 / sell .55 / short .13 |

Both arms: 30 episodes graded. Round-trips/day: 0.53 (60k) vs
1.03 (300k) — the deep arm traded about twice as often and still
lost less, but lost decisively. Mean reward: −0.852 (60k),
−0.527 (300k), champion side positive on the same windows.

## Verdict

- **Formula: FAIL for both arms — 0/3 windows won.** The two-part
  floor never engaged (it is only evaluated on a formula pass).
- Registered as runs **65** (60k Fresh, deep=0) and **66** (300k
  Deep, deep=1), family `scored_eyes_ppo`, promoted=0,
  pending_approval=0 on both — correct under the promotion lock:
  nothing passed, so nothing pends for Kass.
- **The richer eyes did not help.** Giving the PPO the stacker's
  full 94-indicator scored view — the exact information whose
  absence Kass questioned — produced a worse trader than the
  21-feature champion on identical windows, at both training
  budgets. This is the eleventh independent null on BTC classic
  indicators, and it closes the "the PPO is information-starved"
  hypothesis specifically: information was not the binding
  constraint. (Fees remain a suspect — both arms bled most on
  high-activity windows — but no arm here was close enough for a
  fee story to rescue.)

## Perimeter verification

Guard hashes (sha256, first 8) — builder's guard_before ==
guard_after == live files at completion, three ways:

| File | Hash |
|---|---|
| models/ppo_champion.zip (lab) | e610ddf3 |
| models/stacker_champion.pkl (lab, run 64) | c0373d15 |
| models/stacker_champion.json | 52a93161 |
| models/live_seat/stacker.pkl (run 40) | 0dd62fc4 |
| models/live_seat/stacker.json | 4cc2147e |
| models/live_seat/ppo.zip | e610ddf3 |

Full suite after completion + repairs: **227 passed** (see below).
Training + gate feeds pushed verbatim; Training dashboard read back
66/66 models with runs 65/66 named "Scored-Eyes PPO — 60k Fresh" /
"— 300k Deep", FAIL verdicts, gate pending = null. Live Desk read
back untouched: SHORT 0.293605 BTC @ $84,495.30, cash $124,003.84,
effective stop $81,285.07 (anchor $86,607.68), equity $100,498.

## Execution incidents (recorded for the paper trail)

- The original coordinator agent was killed by a runtime restart;
  the detached builder survived on checkpoints. The builder then
  **crashed once** in arm300 segment 12 with
  `sqlite3.OperationalError: disk I/O error` in the broker's record
  write on its own scratch DB — diagnosed as a transient I/O fault
  (195G free, scratch file intact, segment-11 checkpoint zip
  verified loadable with the correct Box(98) space). One relaunch
  was killed by a second runtime restart minutes in; the third
  launch (setsid+nohup) ran to completion. Checkpoints held
  throughout; no phase was ever duplicated (pgrep-verified before
  every relaunch).
- **Two latent defects surfaced by this study's registration**,
  both fixed tests-first during completion:
  1. `add_lab_note` truncated `data/lab_notes.json` to the last
     30 notes. The scored-eyes note append crossed the cap and
     evicted the oldest note — the 2026-10-21 "Features v5.1
     migration" note — which was run 16's only "5.1" attestation,
     so the models index renamed run 16 "Stacker v5". The cap is
     removed (the notebook is the paper trail the name derivations
     read; display windowing lives in the feed builders). The
     evicted note was **restored verbatim**: its title from
     `scripts/migrate_v51.py` (the script that authored it) and
     its detail from run 16's own recorded `metrics.reason` in
     training.db, at run 16's timestamp. Run 16 reads "Stacker
     v5.1" again.
  2. The training feed's learning curve (`rcurve`) copied the
     newest PPO run's recorded reward curve uncapped; run 66's
     300k curve (288 points, ~4.1KB) took the feed to 13,706B.
     The feed now stride-downsamples the curve to ≤ 64 points,
     first and last recorded points exact, every shown point a
     recorded point. Post-diet feed: 10,412B at 66 runs; the size
     sentinel was re-baselined to 10,700B per its documented
     pattern (dated entry in tests/test_compact_feeds.py).
