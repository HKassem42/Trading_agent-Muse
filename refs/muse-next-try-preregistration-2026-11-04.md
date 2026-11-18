# Muse's Next Try — Pre-registration (frozen BEFORE any grading)

Date: 2026-11-04 (ordered by Kass 2026-11-03 ~23:56 ET, side chat)
Order in Kass's words: "Look at the library and go thru and pick my next
model from start to finish. WHATEVER you want to try to get the highest
model score out of all we have so far and beat your previous try.
Controlling whatever knobs you want and whatever process you want. If
that means you wanna add stuff or make entire new excel sheet/matrix
save in library then use with whatever eyes levelers rules to trading
anything!! Go for it. Label it as ur next try. GO!"

Label: **Muse's Next Try** — family tag `muse_next_try`.

## The bar (clean era only)

Pre-fix runs 1–15 (holdout up to 0.6364) were graded under the v4/v5
look-ahead bug and are NOT a valid bar (Kass ordered them off all
displays 2026-10-30). The clean-era leaderboard to beat:

| Metric | Best clean-era value | Run |
|---|---|---|
| Holdout accuracy | **0.5354** | 73 (Muse's Pick #2) and 78 (tie) |
| Holdout precision | **0.5298** | 96 (current Lab Champion #5) |
| Walk-forward accuracy | **0.5177** | 73 (Muse's Pick #2) |

Muse's previous try = Muse's Pick #2 (run 73): holdout 0.5354,
precision 0.5250, WF 0.5177. Beating it means beating 0.5354 on a
fresh shared holdout with non-degenerate anatomy.

## Hypothesis (the one genuinely untested lever)

Every prior clean-era null (direction, tradeability, regime, HA form,
sequence eyes, reward shaping, meta-label gates) read the SAME
information: BTC's own spot candles and indicators derived from them.
The program's own conclusion (2026-11-01) named the sole untested
lever: **new information**. Derivatives positioning is not in the spot
candles. This try adds it, plus sequence/calendar context the
one-candle eyes never had:

- **Derivatives positioning (Binance Vision public archive, BTCUSDT
  perpetual)** — funding rate (level + 24h mean), open interest
  (1h % change, 24h % change, level), long/short ratios (all accounts,
  top-trader accounts, top-trader positions), taker L/S volume ratio,
  perpetual basis vs spot, perp taker-buy share, perp/spot volume
  ratio. (Live Binance API is geo-blocked 451 from here; the Vision
  archive is the same exchange's official historical record and
  covers the full 730d span.)
- **Cross-asset (Coinbase via the project's own adapter)** — ETH 1h
  return, ETH/BTC ratio 24h change.
- **Sequence** — Master Score at t−6h and t−24h, its 24h change,
  prior-candle return.
- **Calendar** — hour-of-day and day-of-week cyclical (sin/cos).

## Frame

Pick #2's 98 columns (scored-102 + Hurst/skew/kurtosis/autocorr —
Champion #4's frame, the best eyes on record) PLUS 22 new causal
columns = **120 columns**. Every new raw series is percentile-scored
through the project's own trailing-4,320-bar machinery, exactly like
the other scored columns; calendar columns enter as (sin/cos+1)×50
on the same 0–100 scale.

Causality rule (one rule for every external series): a feature at
row t may use only external observations timestamped **< t+1h** —
i.e. through the close of candle t, the same footing the spot
features already use (they read candle t in full). Funding uses the
last settlement ≤ t. Truncation-rebuild tests pin this before any
training.

## Model selection — train data ONLY (no holdout peeking)

The last 792 hourly candles are the holdout exam, exactly as the
nightly grades. They are touched ONCE, at final grading.

1. **Frame screening (train rows only):** candidate frames —
   (a) 98 control, (b) 98+derivatives+ETH (112), (c) 98+sequence+
   calendar (106), (d) full 120, (e) derivatives+ETH+prod21 (35) —
   are scored with a fast proxy ensemble (ExtraTrees + LightGBM,
   mean of 3-fold TimeSeriesSplit accuracy on train rows). The frame
   with the best screening score is selected; ties go to the simpler
   frame (fewer columns, then earlier in the list).
2. **Recipe (fixed):** Muse's Pick recipe — StandardScaler → SMOTE →
   RF + XGBoost + LightGBM + ExtraTrees → LogisticRegression
   (`models.muse_pick.build_pick_pipeline`, imported unchanged).
3. **Threshold knob (train only):** out-of-fold probabilities from
   the final pipeline's 5-fold walk-forward on train are used to pick
   the decision threshold θ* maximizing train-OOF accuracy over
   0.40–0.60 (step 0.01). θ* is USED only if it beats the 0.50 OOF
   accuracy by ≥ 0.005; otherwise θ = 0.50. This decision is made
   entirely on train data, before the holdout is touched.
4. **Final grading (once):** the selected pipeline, fit on all train
   rows, is graded on the shared 792-candle holdout. Champions are
   re-scored on the IDENTICAL holdout timestamps, two independent
   ways that must agree: run 96 (prod21, Lab Champion #5) and run 73
   (pick2_98, Champion #4 / Muse's previous try).

## Verdict rules (pre-registered)

- **WIN** requires: holdout accuracy strictly greater than BOTH
  re-scored champions AND greater than 0.5354-level performance on
  this fresh exam (reported as the champions' re-scored numbers,
  since the exam window has moved), with non-degenerate anatomy
  (up-calls > 7 and precision > 0.05 — the run-63 rule).
- Precision, up-calls, walk-forward accuracy, per-window holdout
  accuracy (3×264), and the control frame's holdout grade are all
  reported regardless of outcome. The control is graded for ablation
  reporting only — it can NEVER replace the CV-selected frame after
  the fact.
- A win is recorded as PENDING KASS APPROVAL under the promotion
  lock (this order authorized the try; crowning/seating remains
  Kass's call, exactly as with Picks #1/#2). Nothing installs,
  nothing touches the live seat, shadow seat, gated seat, or any
  paper book. Champion/seat/shadow files are hash-guarded
  before/after.
- **Scope:** stacker only. No PPO this try — the PPO line has lost
  14 straight duels on coin-flip direction votes; a brain duel is a
  follow-on only if these eyes first produce a vote worth consuming.
- The new Excel matrix — `Muse_Next_Try_Matrix.xlsx` (design,
  120-column feature matrix spec, leaderboard, results) — is saved
  to the goal's Library files and the project's refs/data.

## What would make this result not count

- Any causality test failure (truncation rebuild changes any scored
  value at a shared row) — stop, fix, rebuild, re-register the fix.
- Champion two-way re-scores disagreeing — no number is trusted.
- External data covering < 95% of scored rows — the affected columns
  are dropped and the frame is re-registered as amended BEFORE
  screening (amendment logged here, still before any grading).

## Amendment 1 (2026-11-04, BEFORE any screening or grading)

Binance Vision publication lag discovered at build time: monthly
perp klines end 2026-10-31, daily metrics end 2026-11-02 23:55,
monthly funding ends with the 2026-11-01 00:00 settlement — while
the spot frame ends 2026-11-04 05:00 UTC. Tail stitch, disclosed:
- Funding settlements after Binance's last are taken from OKX
  BTC-USDT-SWAP funding-rate history (same 8h cadence).
- Perp candles after Binance's last hour (2026-11-02 23:00) are
  taken from OKX BTC-USDT-SWAP 1h candles (close + base volume);
  taker-buy share for those hours carries the last Binance value
  (OKX candles carry no taker split) — at most ~29 hours.
- Metrics keep the pre-registered "last observation < t+1h"
  semantics, so the final ~29 hours read the 2026-11-02 23:55
  snapshot. All tail hours are counted in the study results.
No definition, threshold, or verdict rule changes.

## Amendment 2 (2026-11-04, BEFORE any screening or grading)

The Binance Vision Oct-2026 monthly perp-klines file is missing
2026-10-28 entirely (720 rows for a 31-day month — verified in the
raw archive). Those 24 perp hours are filled from OKX
BTC-USDT-SWAP 1h candles (paged history), taker-buy share carrying
the last Binance value, exactly like the live-edge tail stitch.
Also recorded: two build bugs caught by the pre-registered checks
before any grading — a pandas label-alignment trap that produced
an all-NaN perp frame (fixed with .to_numpy()), and metrics
publication gaps handled by the as-of (forward-fill) reading the
design already implied. No definition or verdict rule changes.
