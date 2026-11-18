# Regime-Swing System — run 86 (2026-10-30)

Kass's delegation (2026-10-30 ~15:25 ET): "Build whatever you
want to get us the best results — if you gotta rebuild the
brain or the eyes or how it trades, do that with everything
that we've learned" + "This will be the last build until
retrain tonight." Full design latitude delegated; the house
rules are not delegated away: this document freezes the
design BEFORE any grading, lab-only, promotion lock, no
seat, book, or production file is touched.

## Why this design (each choice is a recorded lesson)

- **4h decision clock.** The live book's profits came from
  multi-day holds (a 6-day short, +$939 net, closed
  2026-10-27; the Oct 29 covers banked +$568.78 / +$340.84
  after days in position), while every hourly-churn design
  paid fees to noise. Decisions therefore happen on 4h
  closes only.
- **Regime eyes, not direction eyes.** Direction
  classification at short horizons is a coin flip across
  fourteen nulls; the least-bad targets the program ever
  found were the smooth ones (tradeability 0.613/0.620 in
  the hierarchy study; multi-day trend state untested as a
  classifier target). The eyes classify the forward-4-day
  REGIME (UP / DOWN / CHOP) — the smoothest target ever
  tried here.
- **No RL brain.** Three eyes swaps (runs 82–85 era) moved
  run 44's duel by exactly $0.00 — the brain never added
  anything over its inputs. The ML sits where it showed its
  only aptitude (classification); the trading itself is
  deterministic rules.
- **Low turnover by construction.** Fees buried gross edge
  before (Hierarchy-Focus: +$3,986 gross, $7,041 fees).
  Regime changes are infrequent, entries are capped at two
  tranches, and there is no exit-and-reenter churn path.
- **Walk-forward is the PRIMARY validation.** The shared
  33-day holdout has been graded four times today (runs
  82/83/84/85). Primary evidence is walk-forward across the
  training region with per-fold refits; the standard holdout
  duel runs second, for registry consistency, and is
  labeled FIFTH use wherever it appears.

## The system (frozen)

- **Data**: run 82's persisted fetch (`data/meta_label/
  df.pkl`, 17,510 hourly candles, 2024-10-30 16:00 →
  2026-10-30 15:00 UTC) — no fresh fetch, so the holdout is
  byte-identical to runs 82–85. `sup = make_supervised(df)`;
  holdout = the last 792 sup rows (2026-09-27 16:00 →
  2026-10-30 15:00 UTC), asserted equal to run 82's
  persisted `hold_sup.pkl`. The boundary stamp B =
  2026-09-27 16:00 UTC.
- **4h frame**: `df.resample("4h", closed="left",
  label="left")` aggregated open=first / high=max / low=min
  / close=last / volume=sum, stamped at bucket END (index
  + 4h), VALUES-based throughout (`.to_numpy()`; no label
  joins anywhere — the v5.1 trap). A bucket is kept iff it
  contains exactly 4 hourly candles (count integrity); the
  raw series' two known 6h holes (2025-10-25, 2026-05-08)
  drop their partial buckets and are counted in the record.
  Trailing windows are positional over kept buckets (the
  program-wide convention; the holes are the only
  irregularities).
- **ATR**: ATR4h(14) = the production `_atr` (Wilder true
  range, EWM α=1/14, min_periods=14) applied to the 4h
  frame; `atr_pct4h = ATR4h / close`.
- **Regime label (training only)**: at 4h bucket t,
  net24 = close[t+24]/close[t] − 1 (the forward 4 days);
  thr = 0.75 × atr_pct4h[t]. net24 > thr → **UP** (encoded
  2); net24 < −thr → **DOWN** (0); otherwise **CHOP** (1).
  Rows without 24 forward buckets carry no label. Labels
  read future prices for the OUTCOME only (the meta-label
  convention); features never do.
- **Features (115, frozen)**:
  - *Block A — 4h-native trend set (17)*, all trailing,
    computed positionally on the kept-bucket series; the
    value at bucket t reads buckets ≤ t only:
    `rs_ret6` close/close[−6]−1 (1d), `rs_ret24` (4d),
    `rs_ret42` (7d), `rs_dist_sma50` close/SMA50−1,
    `rs_dist_sma200` close/SMA200−1, `rs_sma50_slope`
    SMA50/SMA50[−6]−1, `rs_sma200_slope` SMA200/SMA200[−6]−1,
    `rs_rv24` std of 4h log returns over trailing 24 bars,
    `rs_rv96` over 96, `rs_rv_ratio` rv24/rv96,
    `rs_atr_pct` atr_pct4h, `rs_acf1_42` lag-1 autocorrelation
    of 4h log returns over trailing 42 (the Pick #2
    construction: rolling Pearson of r with r shifted 1),
    `rs_kurt_42` excess kurtosis (pandas) over trailing 42,
    `rs_skew_42` skew over trailing 42, `rs_dd100`
    close/max(close, trailing 100)−1, `rs_dow_sin` /
    `rs_dow_cos` — sin/cos of 2π × day-of-week / 7 of the
    bucket-END stamp (UTC; causal trivially — the calendar
    is known in advance).
  - *Block B — the incumbent eyes' view (98)*: the pick2_98
    row (Lab Champion #4's recorded 98 columns, in order)
    at the 1h candle stamped bucket_end − 1h — the bucket's
    last candle, whose close IS the decision instant —
    taken POSITIONALLY via a stamp→position map over
    df.index, names prefixed `p98_`. Blocks stay separable
    in the record (final-fit importance shares per block
    are recorded) so a later ablation can tell which block
    carried weight.
  - **Decision rows**: kept buckets with all 115 features
    finite. Training decision rows: bucket end ≤ B.
    Holdout decision rows: the 198 buckets with end in
    (B, B + 792h].
- **Eyes zoo** — selection on training-region OOF only:
  - (a) **primary**: the program's stacker recipe
    (`muse_pick.build_pick_pipeline(42)`: scaler → SMOTE →
    RF+XGB+LGBM+ExtraTrees → LR), multiclass on the encoded
    regime labels;
  - (b) **control**: LightGBM, run 82's frozen config
    (`LGBM_PARAMS`), multiclass.
  - OOF protocol: TimeSeriesSplit(4) over the labeled
    training decision rows (the fit-eligible set: label
    defined AND t + 24 ≤ last training bucket — the label
    horizon lies entirely inside the training span),
    expanding windows, in time order. Pooled OOF predictions
    scored by **3-class balanced accuracy** (mean per-class
    recall); ties (|Δ| < 1e-12) broken by **macro
    one-vs-rest AUC** on the pooled OOF, then by arm order
    (a before b). The record carries per-class recall,
    per-class prediction counts, and class base rates — a
    degenerate one-class predictor is visible, never hidden
    (the run-63 lesson). ONLY the selected arm's class is
    used downstream (walk-forward refits, final fit).

## The trader (frozen) — deterministic rules on the 4h clock

State: the broker's actual position (side, tranche count
tracked from the policy's own fills; any full-flat event —
rule exit or broker stop — resets the count to 0).

At each bucket close inside an evaluation span, AFTER that
bucket's last candle's stop check, with regime R = the
eyes' prediction at that bucket:

| position | R = UP | R = DOWN | R = CHOP |
|---|---|---|---|
| flat | buy (enter long) | short (enter short) | hold |
| long, 1 tranche | buy (the one add) | sell (exit) | sell (exit) |
| long, 2 tranches | hold | sell (exit) | sell (exit) |
| short, 1 tranche | buy (cover/exit) | short (the one add) | buy (cover/exit) |
| short, 2 tranches | buy (cover/exit) | hold | buy (cover/exit) |

- Intents execute at the OPEN of the next bucket's first
  1h candle (the duel's next-open convention), carrying
  atr_pct = atr_pct4h at the DECISION bucket (mirroring the
  production env, which fills at next-open with the
  decision candle's ATR). An intent formed at a span's last
  bucket close never fills inside the span (counted as a
  decision, dropped as a fill).
- A flip (long + DOWN, short + UP) exits to flat at that
  next open and nothing more: re-entry is evaluated at
  later closes while flat (no same-open flip-through —
  lower turnover by construction).
- The production PaperBroker chassis is always live: ATR
  stop/trailing with the production constants, 25% tranche
  / 75% exposure caps, 0.15% fee/side, wick-triggered stops
  (`check_stop` on EVERY 1h candle of the span with that
  candle's close/low/high, the fill candle included, after
  its open fill — the production env's ordering). A broker
  stop exit overrides the rules (the position is simply
  gone at the next decision).
- Span accounting (folds and duel windows alike): the book
  starts FLAT with $100,000 at the span's first candle;
  equity is marked at EVERY 1h close of the span; span P&L
  = final marked equity − $100,000 (an open position is
  marked, never force-closed — the production duel's own
  accounting); span max drawdown = worst peak-to-trough on
  that marked curve (the curve starts at $100,000); span
  score = P&L − 0.5 × max drawdown (the program's formula);
  fees = the sum of the ledger's fee column; trades =
  fills in the ledger; time-in-position = share of the
  span's 1h closes with btc ≠ 0.

## Validation (frozen)

1. **Walk-forward (PRIMARY)** — 4 chronological folds over
   the training decision buckets (the decision-bucket span
   split into 4 contiguous near-equal groups). **Amendment
   (2026-10-30, recorded BEFORE any walk-forward grading —
   no fold had been run):** decision rows exist only from
   bucket ~1160 (the pick2 block's finite region), so the
   first of 4 equal folds would have had ZERO earlier
   decision rows to fit on — an expanding-window walk-
   forward needs a burn-in by construction. The span is
   therefore split into 5 contiguous near-equal segments:
   segment 0 is the burn-in (fit-only, never graded), and
   the 4 graded folds are segments 1–4, each fit on ALL
   eligible rows before it (expanding window, same purge).
   Everything else in this section is unchanged. Per fold:
   the selected eyes class refit on fit-eligible rows with
   t + 24 < the fold's first bucket (label horizon ends
   before the fold starts — full purge); the trader runs
   the fold span (the 1h candles of the fold's buckets),
   flat at fold start, marked at fold end. Report per fold
   and total: P&L, max drawdown, score, trades, fees,
   time-in-position, buy-and-hold (100k × (last bucket
   close / previous bucket close − 1) over the same span —
   the previous bucket is the kept bucket immediately
   before the fold's first). Totals: P&L summed, max
   drawdown = worst fold, total score = total P&L − 0.5 ×
   worst-fold drawdown, trades/fees summed, time-in-position
   candle-weighted.
2. **Holdout duel (SECONDARY — FIFTH use of this holdout)**:
   the standard 3 windows (hold_sup rows [k·264, (k+1)·264))
   — the champion arm (PPO run 44 + run 73's fitted votes)
   recomputed fresh by production `evaluate_windows` +
   `_agg`; **pin: aggregate −210.17 / P&L −$305.28 within
   0.011 or the study ABORTS and nothing registers.** The
   challenger: the selected eyes fit on ALL fit-eligible
   training rows; per window, the trader runs the window's
   264 1h candles (66 buckets), flat at window start
   (intra-window swing trades only — the standard duel's
   per-window reset; stated consequence: a swing the regime
   would have carried across a window boundary restarts
   flat). Challenger record in the `_agg` shape: score =
   mean of the 3 window scores, total_pnl = sum, windows
   won counted on window scores, bench_pnl = the sum of
   the windows' production-style benches (100k ×
   (closes[−1]/closes[0] − 1) per window), action_mix =
   the LAST window's decision mix over its 66 decision
   closes ({hold, buy, sell, short} intents; a cover counts
   as buy, a long exit as sell — the env's semantics),
   mean_reward 0.0 (no reward stream exists for a
   deterministic policy; the field is carried for shape
   only and labeled as such in the sub-record).
   **Verdict**: formula = challenger aggregate > champion
   aggregate AND ≥ 2/3 windows; then `gate_decision()`'s
   two-part floor on the challenger record. Formula AND
   floor pass → **WOULD PROMOTE — PENDING KASS APPROVAL**
   (registry pending_approval=1, promoted=0). Anything less
   → **FAIL**, with the failing half (formula / floor)
   named. This is a full-system test: a failure is recorded
   plainly as a failed system attempt, in the run-84/85
   accounting style (component/calibration accounting does
   not apply — the verdict language in the record states
   exactly what failed).

**Two-way checks**: (i) the champion pin above; (ii) the
challenger's headline duel P&L and one full walk-forward
fold's P&L are each recomputed from their persisted ledger
DBs ALONE (the meta-label re-sum arithmetic: BUY pays
qty×price + fee, SELL receives qty×price − fee, audited
against the broker's recorded cash_after/btc_after at every
fill, open position marked at the span's final close) and
must agree with the runner's numbers to < $0.01 or the
study aborts.

**Registration**: one run, kind `ppo`, family
`regime_swing` (compact family code 19 — the registry
additions in `src/dashboards/compact.py`, the meta_label /
eyes_v2 precedent), display label "Regime-Swing", metrics
in the nightly PPO shape (challenger / champion / reason /
deep / timesteps / pending_approval) plus a `regime_swing`
sub-record carrying: the design string, the eyes OOF table
(per-class detail included), the selected arm, the full
walk-forward record, the duel record, turnover / fees /
time-in-position for both validations, and the two-way
check results. Lab note appended. Machine record in
`data/regime_swing/results.json` (fold + window ledgers
persisted under `data/regime_swing/ledgers/`).

**Guards**: sha256 of the live seat, shadow seat, gated
seat (incl. the Book #3 gate model + manifest), lab
champions and all three live-state files, taken before and
after, must be identical (the eyes_v2 guard set, reused).
The study never opens any paper book DB for writing, never
imports the live loops, and writes only under
`data/regime_swing/`, the registry, the lab notebook, and
its own new files: `src/training/regime_swing.py`,
`scripts/run_regime_swing_study.py`,
`tests/test_regime_swing.py`, plus the family-registry
additions in `src/dashboards/compact.py`. The nightly
retrain must be able to run tonight unaffected: no
production module is modified.

## Holdout-reuse caveat (recorded per the mandate)

Fifth grading use of the same 33-day holdout (runs 82, 83,
84, 85, now 86). The walk-forward is the primary validation
precisely because of this; the duel is reported for registry
consistency and cross-study comparability. Cross-study
selection pressure on these 792 candles is by now severe: a
duel pass here is weak evidence, and nothing in this study
is forward evidence either way. The clean tests remain
forward ones (Book #3's gated forward test is running).

## What would make this a failure

The regime eyes stay at chance out-of-fold (3-class balanced
accuracy ≈ 1/3, per-class recalls showing one-class
collapse), or the walk-forward loses money across folds
(the swing rules don't pay on data the eyes never fit), or
the duel fails the formula or the floor (including the
floor's activity leg: a regime system that almost never
changes state may record < 10% non-hold decisions in the
last window — the floor applies as written, and the record
will name it). Any of those is recorded plainly, with the
full OOF table, walk-forward table and duel table, win or
lose.

## Results (2026-10-30, as graded)

**Registered as run 86** (kind ppo, family `regime_swing`
code 19, display "Regime-Swing — 4h Swing"), promoted=0,
pending_approval=0. **Verdict: FAIL** — the relative
formula failed (aggregate below the champion AND only 1/3
windows won), so the gate floor was never reached.

**Frames.** 4,376 kept 4h buckets (4 partials dropped at
the two known holes); 3,019 training decision rows; 2,995
fit-eligible rows; exactly 198 holdout decision buckets.
Training label counts: DOWN 1,188 / CHOP 698 / UP 1,133.

**Eyes OOF** (TimeSeriesSplit(4), n_oof 2,396; chance BA =
1/3, chance AUC = 0.5):

| arm | BA | macro AUC | acc | recalls D/C/U |
|---|---|---|---|---|
| stacker recipe | 0.3283 | 0.4763 | 0.3164 | 0.102 / 0.353 / 0.529 |
| LightGBM (selected) | 0.3289 | 0.4887 | 0.3689 | 0.329 / 0.087 / 0.571 |

**The regime eyes did not beat chance.** Both arms sit at
the 1/3 balanced-accuracy line with macro AUC *below* 0.5 —
the ranking is slightly anti-predictive out-of-fold. The
LightGBM won selection by 0.0006 of BA; each arm nearly
collapses on a different class (stacker on DOWN, LightGBM
on CHOP). The 4-day regime state of BTC is not in this
feature frame either.

**Walk-forward (PRIMARY; burn-in 604 rows, folds = the
next 4 segments):**

| fold | span | fit rows | P&L | B&H | maxDD | fills | fees | RT | TiP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2025-08-20 → 2025-11-29 | 580 | −15,839.41 | −20,207.79 | 21,075.42 | 171 | 7,897.78 | 59 | 87.7% |
| 1 | 2025-11-29 → 2026-03-10 | 1,184 | −15,967.65 | −23,449.03 | 19,227.53 | 203 | 9,297.03 | 71 | 88.0% |
| 2 | 2026-03-10 → 2026-06-19 | 1,788 | −6,638.51 | −9,690.66 | 11,914.68 | 216 | 10,392.53 | 75 | 84.6% |
| 3 | 2026-06-19 → 2026-09-27 | 2,392 | −12,023.38 | +34,683.70 | 15,318.31 | 239 | 10,544.94 | 90 | 76.6% |
| **total** | | | **−50,468.94** | **−18,663.78** | **21,075.42** (worst fold) | **829** | **38,132.29** | **295** | **84.2%** |

Total score = −50,468.94 − 0.5 × 21,075.42 = **−61,006.65**.
Lost money in all four folds on data the system never
trained on; beat buy-and-hold (fell less) in the three
falling folds and badly lagged the one rising fold. The
"low turnover by construction" hope did not survive
contact with the frozen rules: with near-chance eyes the
predicted class flips bucket to bucket, exits fire on
single CHOP closes, and the system churned 829 fills and
$38k of fees at 4h scale — the fee-bleed failure mode,
recreated by prediction churn instead of by design
turnover. No hysteresis/persistence rule was in the frozen
design; adding one post-hoc would be a new study, not a
fix to this one.

**Holdout duel (SECONDARY, fifth use).** Champion pin
reproduced exactly (agg −210.17 / P&L −$305.28).
Challenger: windows +3,279.05 / −3,800.07 / −26.20 → agg
**−1,510.78**, P&L **−$547.22** vs champion −$305.28 (B&H
−$8,459.84); wins 1/3; 91 fills, fees $4,154.68, TiP
66.7% / 82.6% / 73.9%; last-window action mix hold 0.561 /
buy 0.258 / sell 0.091 / short 0.091. Final-fit importance
shares: Block A (native 4h) 34.4%, Block B (pick2_98)
65.6% — recorded for ablation; neither block carried an
edge (OOF above).

**Two-way checks.** Champion pin exact. Every span's
headline P&L (4 WF folds + 3 duel windows) re-summed from
its persisted ledger DB alone: diff $0.0000 on all seven.
Guard hashes identical before/after (Book #3's gated seat
included); the study never opened a paper book.

**Reading.** The last untested axis of the classic-
indicator program — horizon (4 days), target smoothness
(3-state regime), model class, and trader type
(deterministic rules) — returns the same answer as the
fourteen nulls before it: the information is not in the
frame. What remains genuinely untested is unchanged: new
information (funding / open-interest) and Kass's parked
Heikin-Ashi-deep study, both his call.

**Verification.** New adversarial tests 16/16
(`tests/test_regime_swing.py`, watched failing 15/15 on
the stub; the 16th pins the compact family registry).
Full suite after registration: **411 passed / 5 failed** —
all five are the documented pre-existing registry-state
pins (training-feed size sentinel, audit pin, telemetry
pin, run-75 detail-window pin, gate-history strip pin),
none a regression from this build. Feeds: compact
training + gate pushed verbatim to both dashboards and
read back (newest run 86; gate pending still run 83). One
hand-transcription slip in the first snapshot push
(gate-history run 72 P&L sign) was caught by the read-back
and corrected with a full verbatim re-push, verified
clean on the second read-back.
