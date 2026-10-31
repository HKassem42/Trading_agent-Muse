# HA-Deep — the Heikin-Ashi study (eyes run + Kass's Shift Trader, 2026-10-30)

Kass called this study himself (2026-10-30 ~16:53 ET, tapping
"Call the Heikin-Ashi study now"). His rationale, recorded
2026-10-30 ~14:36 ET: he originally picked Heikin-Ashi
because the smoothed red/green runs make trend visually
obvious — sustained directional pressure you can read at a
glance — and his standing judgment is that the program
"hasn't looked into that hard enough to make it work." The
current HA footprint is thin: `ha_dir` is 1 binary-ish
feature of the production 21, raw + HA streak tables were
built descriptively on 2026-10-22, and the early HA-Seq
study was a null on old features and next-candle labels.
None of that is this test. This document freezes the design
BEFORE any grading, lab-only, promotion lock, no seat,
book, or production file is touched.

Two timestamped amendments from Kass were folded in BEFORE
the freeze (nothing had been graded, or even coded, when
they arrived):

- **Amendment 1 (2026-10-30 ~16:54 ET):** "I want all the
  features for that candlestick style and I want a fresh
  best indicators that work with it — main chart indicators
  and sub chart ones." → the feature frame below is the
  full HA-native suite in three blocks (anatomy/streaks,
  main-chart indicators ON the HA candles, sub-chart
  indicators ON the HA candles), chosen on principle and
  frozen — no per-indicator selection by OOF (the A–Z
  screen lesson) — with a per-block ablation reported.
- **Amendment 2 (2026-10-30 ~16:54 ET):** Kass's own
  trading rule as a deterministic arm — trade the HA color
  shift, confirmed on the second candle of the new color,
  exits by the production trailing-stop chassis.
- **Amendment 3 (2026-10-30 ~16:56–16:57 ET):** (A) the
  Shift Trader gets a JUDGED form: at each confirmed shift
  an indicator judgment estimates whether the coming move
  in the shift direction is "a lot or just a little," and
  only "a lot" trades; the pure form is the control; the
  judged form is the registered system. (B) the trader's
  action vocabulary is BUY / SHORT / HOLD only — BUY opens
  a long and closes a short, SHORT opens a short and
  closes a long; there is no sell action.

## Part I — the HA-Deep eyes

### The HA series (frozen construction)

Computed over the FULL raw series (`data/meta_label/
df.pkl`, 17,510 hourly candles — run 82's persisted fetch,
so the holdout is byte-identical to runs 82–86), seeded at
the series start so the recursion's warmup is honest:

- HA close(t) = (O+H+L+C)/4 of the raw candle.
- HA open(0) = (O(0)+C(0))/2; HA open(t) = (HA open(t−1) +
  HA close(t−1))/2 — the PRODUCTION recursion
  (`_heikin_ashi` in `src/features/technical_indicators.py`),
  imported, not re-derived.
- HA high(t) = max(raw high, HA open, HA close);
  HA low(t) = min(raw low, HA open, HA close) — the
  standard HA candle completion (the production code only
  needs open/close for `ha_dir`; the deep frame needs the
  full candle for Stochastic/ADX/ATR-type indicators).
- HA body(t) = HA close − HA open. **Color: body > 0 →
  green (+1); body < 0 → red (−1); body == 0.0 exactly →
  neutral (0).** A neutral candle ends any run and starts
  none (the zero-body rule, pinned in tests).
- A **run** is a maximal sequence of consecutive candles of
  the same nonzero color. streak(t) = signed length of the
  run containing t (+green / −red), 0 at neutral candles.
- HA ATR(14) = the production `_atr` (Wilder true range,
  EWM α=1/14, min_periods=14) applied to the HA candle
  frame; used to normalize HA anatomy features.
  Production raw ATR(14) (`atr_pct_raw` × close in the
  supervised frame) is used ONLY where the frozen spec
  says so: the triple-barrier width (identical to Eyes v2)
  and the Shift Trader's broker ATR context.

### Features — 58, frozen, in three blocks

Every feature is computed from the HA candle series plus
raw volume (Block 1's two HA-vs-raw GAP features compare
the raw close against the HA series — part of Block 1 as
briefed; the raw-price feature frame appears nowhere else
in Part I except the run-73 exam baseline and the
walk-forward comparator). All values at row t read candles
≤ t only, and are proven so under future-garble in tests.
Raw units throughout (the stacker pipeline standardizes;
LightGBM is scale-free; the sequence arm standardizes per
fit) — the Eyes v2 context-feature convention.

**Block 1 — HA anatomy & streaks (22):**

1. `ha_body_atr` — body / HA ATR(14)
2. `ha_body_ratio` — body / (HA high − HA low); 0 if range 0
3. `ha_upwick_atr` — (HA high − max(HA open, HA close)) / HA ATR
4. `ha_lowick_atr` — (min(HA open, HA close) − HA low) / HA ATR
5. `ha_range_atr` — (HA high − HA low) / HA ATR
6. `ha_color` — +1 / −1 / 0 per the color rule
7. `ha_streak` — signed run length at t
8. `ha_streak_body` — sum of body/HA ATR over the current
   run up to and including t (signed; 0 at neutral candles)
9. `ha_streak_rel` — |streak(t)| / mean length of runs that
   ENDED within the trailing 500 candles before t; 1.0 if
   no run ended in that window
10. `ha_gap_ma6` — mean of body/HA ATR over trailing 6
11. `ha_ret6` — HA close / HA close(−6) − 1
12. `ha_ret24` — HA close / HA close(−24) − 1
13. `ha_ret96` — HA close / HA close(−96) − 1
14. `ha_dist_sma50` — HA close / SMA50(HA close) − 1
15. `ha_dist_sma200` — HA close / SMA200(HA close) − 1
16. `ha_sma50_slope` — SMA50 / SMA50(−6) − 1 (HA close)
17. `ha_raw_gap_atr` — (raw close − HA close) / HA ATR
18. `ha_raw_agree` — +1 if sign(raw close − raw open) ==
    sign(HA body) with both nonzero; −1 if opposite with
    both nonzero; else 0
19. `h4_ha_color` — color of the 4h HA series (below)
20. `h4_ha_streak` — signed streak of the 4h HA series
21. `d1_ha_color` — color of the daily HA series
22. `d1_ha_streak` — signed streak of the daily HA series

Multi-timeframe state (19–22): the 4h and 1d bucket series
are built exactly as production builds them
(`df.resample("4h"/"1d", closed="left", label="left")`,
open=first/high=max/low=min/close=last/volume=sum,
dropna); the SAME HA recursion runs over each bucket
series (seeded at its first bucket); color and streak are
computed on that series; each bucket's values are stamped
at bucket END (index + 4h / + 1d) and merged onto the 1h
decision rows by decision time (row stamp + 1h) with
`merge_asof` backward — the production v4 construction,
VALUES-based throughout (the v5.1 trap: no Series +
shifted-index joins anywhere).

**Block 2 — main-chart indicators ON the HA candles (21):**
HA close is the price input unless stated.

23–26. `ha_dist_ema9/21/50/200` — HA close / EMA(L) − 1
27–30. `ha_ema9/21/50/200_slope` — EMA(L) / EMA(L)(−6) − 1
    (EMA = pandas ewm span=L, adjust=False — the
    production EMA convention)
31. `ha_dist_sma9` 32. `ha_dist_sma21` — HA close/SMA − 1
33. `ha_sma9_slope` 34. `ha_sma21_slope` — SMA/SMA(−6) − 1
    (SMA50/200 live in Block 1; no duplication)
35. `ha_bb_pos` — Bollinger(20, 2σ on HA close):
    (HA close − lower)/(upper − lower); 0.5 if width 0
    (σ = rolling std, ddof=1)
36. `ha_bb_width_atr` — (upper − lower) / HA ATR
37. `ha_ichi_tk_gap` — (tenkan − kijun) / HA ATR
38. `ha_ichi_close_kijun` — (HA close − kijun) / HA ATR
39. `ha_ichi_vs_cloud` — (HA close − cloud mid) / HA ATR
40. `ha_ichi_cloud_thick` — (cloud top − cloud bottom) / HA ATR
41. `ha_ichi_in_cloud` — 1 if cloud bottom ≤ HA close ≤
    cloud top else 0
    Ichimoku on HA candles: tenkan = (max HA high 9 +
    min HA low 9)/2; kijun = same over 26; senkou B =
    (max HA high 52 + min HA low 52)/2. The cloud AT t is
    the chart-plotted cloud: span A = (tenkan + kijun)/2
    and span B as computed at **t − 26** (the standard
    26-period displacement) — every cloud input is
    therefore data from ≤ t−26, causal by construction,
    and pinned in the causality tests. Chikou is excluded
    on principle (it restates close-vs-past-close, which
    the ret features already carry).
42. `ha_kc_pos` — Keltner on HA (EMA20 ± 2 × HA ATR):
    (HA close − lower)/(upper − lower); 0.5 if width 0
43. `ha_dist_vwap24` — (HA close − VWAP) / HA ATR, VWAP =
    rolling-24 sum(HA typical × raw volume) /
    sum(raw volume), HA typical = (HA high + HA low +
    HA close)/3 (the production rolling-24h VWAP window)

**Block 3 — sub-chart indicators ON the HA candles (15):**

44. `ha_rsi14` — RSI(14) on HA close (production `_rsi`),
    centered: (RSI − 50)/50
45. `ha_macd_line` 46. `ha_macd_signal` 47. `ha_macd_hist`
    — MACD(12/26/9) on HA close (production EMA
    construction), each divided by HA close
48. `ha_stoch_k` 49. `ha_stoch_d` — Stochastic on HA
    high/low/close: %K = (HA close − min HA low 14) /
    (max HA high 14 − min HA low 14), 0.5 if range 0;
    %D = SMA3(%K); both in [0, 1]
50. `ha_adx` — ADX(14) on HA candles / 100 (Wilder:
    +DM/−DM from HA high/low moves, true range on HA
    candles, all smoothings EWM α=1/14 adjust=False —
    the production `_atr` convention)
51. `ha_di_gap` — (+DI − −DI) / 100 on HA candles
52. `ha_cci` — CCI(20) on HA typical price / 100:
    (TP − SMA20(TP)) / (0.015 × mean |TP − SMA20(TP)| 20);
    0 if the deviation is 0
53. `ha_willr` — Williams %R(14) on HA / 100 ∈ [−1, 0]:
    (max HA high 14 − HA close)/(max − min) × −1;
    −0.5 if range 0
54. `ha_ewo` — (EMA5 − EMA35)/HA close on HA close (his
    own library indicator, HA form — production
    construction)
55. `ha_atr_pct` — HA ATR(14) / HA close
56. `ha_obv_slope` — OBV signed by HA direction: OBV =
    cumsum(raw volume × sign(HA body)); slope =
    OBV.diff(24) / trailing-24 raw volume (the production
    volume-family construction, HA-signed per Amendment 1)
57. `ha_adl_slope` — ADL on HA candles: MFM = ((HA close −
    HA low) − (HA high − HA close))/(HA high − HA low)
    (0 if range 0); ADL = cumsum(MFM × raw volume);
    slope = ADL.diff(24) / trailing-24 raw volume
58. `ha_netvol_rel` — trailing-24 sum(raw volume ×
    sign(HA body)) / trailing-24 raw volume

**Block ablation (frozen, diagnostic only):** after zoo
selection, the stacker recipe (arm (a)'s exact
configuration) is refit under the SAME TimeSeriesSplit(4)
OOF protocol on TB labels over seven feature subsets —
full 58, B1 only, B2 only, B3 only, B1+B2, B1+B3, B2+B3 —
and OOF balanced accuracy is reported per subset, plus
per-block importance shares from arm (a)'s final
full-frame fit (mean normalized impurity importance
across the stack's base learners, summed per block). The
ablation never feeds back into selection (the A–Z lesson:
the suite was chosen on principle; the ablation only
teaches the record which block carries what).

### Labels and row sets (frozen)

- **TB labels**: Eyes v2's `triple_barrier_labels`
  verbatim (1.0 × production ATR(14) at t, 24-candle
  horizon, HIGH/LOW first touch, both-barrier candle by
  its own direction, neither → 24-candle net move, exact
  tie dropped).
- **Next-candle labels**: the supervised frame's `target`.
- **Common training rows**: sup positions t in the
  training region with (i) all 58 HA features finite at t
  AND over the trailing 24-row window t−23…t (sequence
  feasibility, shared by every arm); (ii) the pick2_98
  row finite at t (the walk-forward comparator and run-73
  baseline need it); (iii) TB label defined AND
  next-candle label defined; (iv) t + 24 < split (purge —
  the TB horizon lies inside the training region).
- **Exam rows**: the first 768 holdout positions
  (split … split+767) — TB labels are defined at all of
  them (horizon fits inside the holdout; asserted), and
  next-candle labels exist by construction.
- **Duel votes**: the selected model's votes at ALL 792
  holdout positions (features exist at all 792; asserted
  finite from split−23 to the end).

### The zoo — selection on training-region OOF only

TimeSeriesSplit(4) over the common training rows, in time
order. Each arm's OOF predictions are scored against ITS
OWN labels (balanced accuracy at the 0.5 cut, and AUC) —
arms (a)/(c)/(d) target the TB move, arm (b) targets the
next candle; the selection compares each arm's mastery of
its own question, which is the comparison Kass's two
hypotheses pose. Selection rule (frozen): highest OOF
balanced accuracy; ties (|Δ| < 1e-12) by higher AUC, then
arm order a, b, c, d. ONLY the selected arm is graded on
the holdout as the eyes candidate. (Disclosed: fold-
boundary training rows' TB labels read ≤24 candles into
the next fold's feature region — outcomes, never features;
the runs 82–85 convention, identical for every arm.)

- **(a)** HA frame (58) + the program's stacker recipe
  (`build_pick_pipeline(42)`: scaler → SMOTE →
  RF+XGB+LGBM+ExtraTrees → LR), TB labels.
- **(b)** HA frame (58) + stacker recipe, NEXT-CANDLE
  labels — Kass's original HA intuition in its classic
  form.
- **(c)** HA SEQUENCE — Kass's 24-candle lookback on HA:
  a compact LSTM (Eyes v2's `LSTMEyes` protocol verbatim:
  input dims = 6, hidden 48, dropout 0.3, BCE with
  pos_weight, AdamW lr 1e-3 wd 1e-2, batch 256, ≤40
  epochs, early stop on the fit rows' last 15% with the
  24-row purge, patience 6, seeds 42) whose input at t is
  the last 24 candles of the HA SEQUENCE frame — per
  candle: [`ha_body_atr`, `ha_upwick_atr`, `ha_lowick_atr`,
  `ha_range_atr`, clip(streak/12, −1, 1), HA close 1-bar
  return] — every sequence ENDS at t, TB labels.
  **Fallback (c′)**: if (c) raises or yields non-finite
  probabilities in any fold or the final fit, substitute
  the flattened-lag variant (the 6 sequence series at
  lags 0…23, lag-major, 144 inputs, run-82 LightGBM
  config) and record the substitution + reason.
- **(d)** HA frame (58) + LightGBM (run-82 `LGBM_PARAMS`),
  TB labels.

**Judge sub-selection (for Part II):** the Shift Trader's
judge is the better TB flat arm — argmax OOF balanced
accuracy over arms (a) and (d) (tie → AUC, then a) —
regardless of which arm wins the eyes zoo.

**Final model**: the selected arm refit on ALL common
training rows (for (c): same protocol, validation = the
last 15% of the training rows). Persisted under
`data/ha_deep/` with its holdout votes.

### Grading (frozen) — TWO bars, both required

1. **Exam.** On the 768 exam rows, identical timestamps,
   predictions = vote ≥ 0.5, against the SELECTED arm's
   label definition (its PRIMARY exam): the new eyes
   (accuracy, precision, up-calls) vs run 73 re-scored —
   the fitted champion on its own pick2_98 frame
   (`champion_holdout_frame`), same rows, same labels.
   **Secondary diagnostic**: both eyes ALSO scored on the
   OTHER label definition over the same 768 timestamps
   (if the primary was TB, the next-candle exam, and vice
   versa). Vote spread per eye over all 792 holdout
   votes: min, max, % outside [0.45, 0.55], % outside
   [0.40, 0.60]. **Exam win = new accuracy STRICTLY >
   run 73's re-scored accuracy on the PRIMARY labels.**
   Anatomy (up-calls, precision, label base rate) is
   reported with the verdict so a degenerate win is
   visible (the run-63 lesson).
2. **Money test.** The standard 3-window duel (nightly
   `evaluate_windows`, next-candle-open fills), PPO
   run 44 frozen, on the same 3 × 264 windows: challenger
   consumes the new eyes' 792 votes; champion consumes
   run 73's fitted votes. **Money win = challenger
   aggregate score > champion aggregate score.**
   **Two-way pin**: the fresh champion arm must reproduce
   runs 82–86's recorded champion numbers on this
   byte-identical holdout (aggregate −210.17, P&L
   −$305.28) within 0.011 or the study ABORTS and nothing
   registers.
3. **Walk-forward corroboration (PRIMARY corroboration —
   the regime study's lesson, and this holdout's SIXTH
   grading use).** The training span (sup positions
   [0, split)) is split into 5 contiguous near-equal
   segments by `np.array_split` (segment 0 = burn-in,
   fit-only); folds = segments 1–4. Per fold: the
   SELECTED recipe is refit on common rows with position
   < the fold's first position AND position + 24 < the
   fold's first position (full purge), then scored
   (accuracy + balanced accuracy against its own labels)
   on the fold's common rows. The COMPARATOR — the
   incumbent recipe (`build_pick_pipeline(42)`) on the
   pick2_98 frame with the SAME label definition, same
   fit rows, same graded rows — is scored identically.
   Report per fold: both accuracies, both balanced
   accuracies, the fold base rate, and the graded-row
   count. This corroboration does not gate the verdict;
   it is reported as the primary corroborating evidence
   alongside the sixth-use holdout exam, and any conflict
   between them is stated plainly.

**Verdict accounting (frozen).** Both bars pass → **WOULD
PROMOTE — PENDING KASS APPROVAL** (registry
pending_approval=1, promoted=0). Exam win only → recorded
as an exam win with pending_approval=0 (the eyes-duel
lesson). Money win only → also not promotable. Neither →
a failed eyes attempt. A component study (run 84/85
accounting): a failure is NOT numbered among the program
nulls. Under NO outcome does this study install, crown,
or move any model file or seat.

**Registration**: one run, kind `stacker`, family
`ha_deep` (compact family code 20 — the registry additions
in `src/dashboards/compact.py`, the meta_label/eyes_v2/
regime_swing precedent), display label "HA-Deep Eyes",
metrics in the nightly stacker shape (challenger_acc /
champion_acc = the two PRIMARY exam accuracies, precision
/ champion_precision, challenger_wf_acc = the selected
arm's OOF accuracy, folds = 4, pending_approval per the
verdict, reason) plus an `ha_deep` sub-record carrying:
the design string, the frozen feature list with block
tags, label stats, the zoo OOF table, the judge
sub-selection, the block ablation + importance shares,
the selected arm, both exam tables (primary + secondary,
both eyes) incl. spread, the walk-forward corroboration
table, and the duel table. Lab note appended. Machine
record in `data/ha_deep/results.json` (ledgers and the
judge log under `data/ha_deep/`).

## Part II — Kass's HA Shift Trader (Amendments 2 + 3)

A deterministic rules system, no ML in the signal. The
ML appears only as the JUDGE in the judged form (below).

### Signal (frozen)

On the 1h HA series of Part I (full-series colors and
runs, computed once — runs are NEVER restarted at span
boundaries; a span that begins mid-run inherits the true
run state):

- A run **confirms** when its length first reaches L
  (L = 2 graded; L = 3 the reported sensitivity). A
  confirmation is an event only if the run is a **shift**:
  its color differs from the immediately preceding
  nonzero run's color. (Colors alternate among nonzero
  runs except across neutral candles: a same-color run
  after neutral candles is NOT a shift and fires nothing —
  the literal reading of Kass's rule, pinned in tests.)
  Each run confirms at most once. A span that starts
  mid-run whose confirmation candle lies before the span
  sees no event for that run.
- The confirmation is evaluated at the confirming
  candle's CLOSE; the action fills at the NEXT candle's
  OPEN (the program's convention), carrying
  atr_pct = production `atr_pct_raw` at the confirming
  candle for the broker's stop sizing.

### Actions — BUY / SHORT / HOLD only (Amendment 3B)

| state | green shift confirmed | red shift confirmed |
|---|---|---|
| flat | BUY | SHORT |
| long | HOLD (BUY while long = no-op) | SHORT (flip) |
| short | BUY (flip) | HOLD (SHORT while short = no-op) |

- BUY from flat opens a long; BUY while short CLOSES the
  short and opens a long (a flip). SHORT mirrors. A flip
  is ONE decision executed as two broker fills at the
  same next open, close-then-open in that order
  (`sell`/`cover` pct=1.0, then `buy`/`short_open`), each
  leg paying its own side fee — the honest cost of a
  flip, reconciled by the ledger re-sum.
- Sizing: single position, 25% tranche of current equity
  per entry (broker `max_position_pct=0.25`), 75%
  exposure cap (broker-enforced), NO adds.
- Between signals the system only HOLDs. Position exits
  happen ONLY via (1) a confirmed opposing shift (the
  flip above) or (2) the production PaperBroker chassis
  — ATR stop-loss + trailing stop, production constants,
  wick-triggered (`check_stop` on every candle of the
  span with that candle's close/low/high, the fill candle
  included, after its open fill — the production env's
  ordering). After a stop exit the system is flat until
  the NEXT confirmed shift; the stopped position's run
  can never re-fire (its confirmation is spent by
  construction), and no other re-entry exists.
- Per-candle ordering inside a span (frozen): (1) execute
  the pending action at the candle's open; (2)
  `check_stop` on the candle; (3) mark equity at the
  candle's close; (4) evaluate the candle's confirmation
  event against the post-stop position and set the
  pending action. An event on a span's last candle is a
  counted decision whose fill never lands inside the span
  (the regime-study convention).
- Span accounting (folds and duel windows alike): flat
  $100,000 start; equity marked at every candle close;
  span P&L = final marked equity − $100,000 (open
  positions marked, never force-closed); max drawdown =
  worst peak-to-trough on the marked curve (starting at
  $100,000); score = P&L − 0.5 × max drawdown; fees = the
  ledger's fee sum; trades = fills; time-in-position =
  share of the span's closes with btc ≠ 0.

### The two graded forms + sensitivity

- **Pure form (CONTROL)**: every confirmed shift trades,
  exactly as above.
- **Judged form (Kass's full system — the REGISTERED
  one)**: at each confirmed shift, before the action is
  set, the JUDGE reads the full 58-feature HA frame at
  the confirming candle and estimates the coming move in
  the SHIFT direction: P_lot = judge P(TB label = 1) for
  a green shift; P_lot = 1 − judge P(TB label = 1) for a
  red shift (the TB label's down class). **"A lot" =
  P_lot ≥ 0.55** (the program's line). The action fires
  only on "a lot"; on "a little" the decision is HOLD —
  for entries AND flips alike (a judged-little opposing
  shift does not close the position either; the position
  then exits only via the chassis). The judgment is read
  ONCE, at the confirmation candle; a skipped streak
  never re-fires. The judge is the Part-I judge
  sub-selection winner (better OOF of arms (a)/(d)),
  fit on training-region common rows ONLY — in
  walk-forward, refit per fold on common rows before the
  fold with the t + 24 purge; for the duel, fit on ALL
  common training rows. **Judge log**: every confirmed
  shift in every graded span is logged with (span,
  candle stamp, direction, P_lot, threshold, verdict
  taken/skipped, context entry/flip) — this system's
  veto-log equivalent, persisted at
  `data/ha_deep/shift_judge_log.json` and summarized in
  the run record.
- **Sensitivity (reported, not graded)**: the PURE form
  with L = 3, full walk-forward + duel numbers in the
  record.

### Grading (frozen)

- **Walk-forward (PRIMARY)**: the Part-I segmentation
  (5 near-equal segments over sup positions [0, split);
  folds = segments 1–4). Each fold runs the fold's 1h
  candles (the sup rows of the segment — consecutive df
  candles; the two known 6h holes in the raw feed are
  crossed positionally, the program-wide convention),
  flat at fold start. Per fold and total, per form:
  P&L, max drawdown, score, trades, fees,
  time-in-position, buy-and-hold over the same span
  (100k × (last close / previous close − 1), previous =
  the candle immediately before the span). Totals: P&L
  summed, max drawdown = worst fold, total score =
  total P&L − 0.5 × worst-fold drawdown (the regime
  study's total convention).
- **Holdout duel (SECONDARY — SIXTH use of this
  holdout)**: the standard 3 windows (hold_sup rows
  [k·264, (k+1)·264)), flat at each window start, vs the
  fresh-recomputed 44+73 champion pair — **pin:
  aggregate −210.17 / P&L −$305.28 within 0.011 or the
  study aborts** (shared with Part I; one pin serves
  both). The judged form's record is built in the `_agg`
  shape (score = mean of window scores, total_pnl = sum,
  windows = per-window score/P&L/bench/dd, bench_pnl =
  sum of window benches, action_mix = the LAST window's
  decision mix over its 264 closes in the BUY/SHORT/HOLD
  vocabulary {hold, buy, short}). **Verdict (judged form
  only)**: formula = judged aggregate > champion
  aggregate AND ≥ 2/3 windows won (on window scores);
  then `gate_decision()`'s two-part floor on the judged
  record. Formula AND floor pass → **WOULD PROMOTE —
  PENDING KASS APPROVAL** (pending_approval=1,
  promoted=0). Anything less → FAIL with the failing half
  named. The pure form's duel is computed identically and
  reported (its own formula/floor outcome recorded in the
  metrics, but it sets no pending flag — the judged form
  is the registered system; run-82 sensitivity-in-record
  precedent).
- **Two-way checks**: (i) the champion pin; (ii) EVERY
  span ledger (4 folds × 3 forms + 3 windows × 3 forms)
  re-summed from its persisted DB alone (the meta-label
  arithmetic, audited per fill against the broker's
  recorded cash_after/btc_after, open position marked at
  the span's final close) must agree with the runner's
  P&L to < $0.01 or the study aborts.

**Registration**: ONE run for the trader (registered
after the eyes run; the registry takes the next id),
kind `ppo`, family `ha_shift` (compact family code 21),
display label "HA Shift Trader", metrics in the nightly
PPO shape (challenger = the JUDGED form's duel record,
champion, reason, deep=False, timesteps=0,
pending_approval per the judged verdict) plus an
`ha_shift` sub-record carrying: the design string, the
signal/judge specification, the judge's identity (arm +
its OOF line), the full walk-forward records for all
three forms, the full duel records for all three forms,
the sensitivity (pure L3) record, judge-log summary
counts (shifts seen / taken / skipped per span set), and
the two-way check results. Lab note appended.

## Guards (both parts)

sha256 of the live seat, shadow seat, gated seat (incl.
the Book #3 gate model + manifest), lab champions and all
three live-state files, taken before and after, must be
identical (the eyes_v2 guard set, reused verbatim). The
paper books are information-only entries (their own loops
write them). The study never opens a paper book DB for
writing, never imports the live loops, and writes only
under `data/ha_deep/`, the registry, the lab notebook,
and its own new files: `src/training/ha_deep.py`,
`scripts/run_ha_deep_study.py`, `tests/test_ha_deep.py`,
plus the family-registry additions in
`src/dashboards/compact.py`. No production module is
modified; tonight's retrain must run unaffected.

## Holdout-reuse caveat (recorded per the mandate)

This is the SIXTH grading use of the same 33-day holdout
(runs 82, 83, 84, 85, 86, now 87/88). Criteria are frozen
and all model/judge selection happens on training-region
data only, which keeps the comparisons honest, but
cross-study selection pressure on these 792 candles is by
now severe: a pass here is weak evidence and is not
forward evidence. That is precisely why Part I carries
the training-region walk-forward as its primary
corroboration and Part II is graded walk-forward-first.
The clean tests remain forward ones (Book #3's gated
forward test is running).

## What would make this a failure

For the eyes: the zoo's OOF stays at chance on both label
definitions (HA structure, given the full suite, separates
winners from losers no better than the classic frame
did), or the selected eyes win neither bar, or the
walk-forward contradicts a holdout exam win. For the
trader: the walk-forward loses money across folds in both
forms (the visible HA runs don't pay after fees and
stops), or the judged duel fails formula or floor. Any of
those is recorded plainly, with the full tables, win or
lose — including the verdict on Kass's hypothesis itself:
whether the red/green structure he can see carries
information a model or a rule can spend.


## Results (2026-10-30, runs 87/88)

**Both parts failed their frozen bars. Neither run pends;
the promotion lock was never approached.** Machine record:
`data/ha_deep/results.json`; ledgers + judge log under
`data/ha_deep/`. Registry max is now 88; tonight's retrain
takes the ids after 88.

### Part I — the eyes (run 87, HA-Deep Eyes, FAIL)

Common training rows: 12,055 (TB base rate 0.490, next-
candle base rate 0.504). Zoo OOF (TimeSeriesSplit(4),
9,644 graded rows per arm, each arm against its own
labels):

| arm | balanced acc | AUC | accuracy |
|---|---|---|---|
| (a) HA frame + stacker, TB | 0.4982 | 0.4982 | 0.4978 |
| (b) HA frame + stacker, next-candle | **0.5024** | 0.5092 | 0.5025 |
| (c) HA sequence LSTM, TB | 0.5005 | 0.5079 | 0.5010 |
| (d) HA frame + LightGBM, TB | 0.4993 | 0.4985 | 0.4999 |

Every arm sits on the chance line. Selection by the frozen
rule picked arm (b) — Kass's original HA intuition in its
classic form (next-candle direction). Judge sub-selection
(better TB flat arm): (d), the LightGBM, 0.4993 vs 0.4982.

Block ablation (stacker recipe, TB labels, same OOF
protocol): full 58 = 0.4982; B1 anatomy/streaks = 0.5048;
B2 main-chart = 0.4944; B3 sub-chart = 0.5076; B1+B2 =
0.4919; B1+B3 = 0.5001; B2+B3 = 0.4971. No block, and no
combination, separates from chance. Importance shares from
arm (a)'s final fit: B1 0.343 / B2 0.374 / B3 0.283 —
spread evenly, as expected when there is nothing to find.

Exam (768 identical timestamps, PRIMARY = next-candle
labels, the selected arm's definition; exam base rate
0.4935): HA-Deep eyes accuracy **0.5208** (precision
0.5133, 413 up-calls) vs run 73 re-scored **0.5273**
(precision 0.5188, 426 up-calls). **Exam bar FAILED.** Secondary diagnostic
(triple-barrier labels, same rows): new eyes 0.4818 vs
run 73's 0.5117 — behind on the other definition too.
Vote spread over the 792 holdout votes: new eyes span
0.4628–0.5480, 0.0% outside [0.45, 0.55] — the same
compressed, expressionless profile as run 73 (0.4615–
0.5421, 0.0%).

Walk-forward corroboration (next-candle labels, selected
recipe vs the pick2_98 incumbent recipe, same fit/graded
rows; segments over the common-row span — see the
deviation note in Verification):

| fold | new acc / BA | comparator acc / BA | base rate | n |
|---|---|---|---|---|
| 1 | 0.5174 / 0.5166 | 0.5008 / 0.5024 | 0.5050 | 2,416 |
| 2 | 0.4988 / 0.4988 | 0.4917 / 0.4921 | 0.4938 | 2,416 |
| 3 | 0.5145 / 0.5145 | 0.5145 / 0.5145 | 0.4988 | 2,416 |
| 4 | 0.5215 / 0.5218 | 0.5111 / 0.5126 | 0.5065 | 2,391 |

The HA recipe edges the incumbent in 3 of 4 folds (fold 3
an exact tie) — inside the training region the HA frame is
no worse than the classic frame at next-candle direction.
But every number on both sides is at the chance line
(0.49–0.52), and the frozen exam bar on the holdout went
to run 73. Corroboration cannot rescue a failed bar; it
only says the two frames are equally uninformative here.

Money test: the fresh champion arm reproduced the pin
exactly (aggregate −210.17, P&L −$305.28), and the
challenger duel came back **identical to the cent**
(aggregate −210.17, P&L −$305.28) — run 44's argmax never
flipped under the new eyes' compressed votes. Fourth
confirmation of the eyes-duel pattern (runs 82–85).
**Money bar FAILED.** Verdict: FAIL, pending_approval=0 —
a failed component attempt, not a numbered null.

### Part II — Kass's HA Shift Trader (run 88, FAIL)

Judge: arm (d) LightGBM on the 58-frame, TB labels (OOF
balanced accuracy 0.4993 — chance; the judge had nothing
to give, and the judged form's numbers below show it).

Walk-forward (PRIMARY; per fold each form restarts flat
at $100k; B&H is buy-and-hold over the same span):

| form | fold P&Ls | total P&L | worst DD | total score | fills | fees | B&H total |
|---|---|---|---|---|---|---|---|
| pure L=2 | −25,774 / −26,696 / −32,371 / −30,178 | **−115,019.25** | 32,739.86 | −131,389.18 | 3,102 | 99,474.74 | −18,688.31 |
| judged | −11,918 / −10,140 / −10,654 / −7,224 | **−39,936.58** | 14,087.82 | −46,980.49 | 683 | 24,531.75 | −18,688.31 |
| pure L=3 (sensitivity) | −16,662 / −21,373 / −20,913 / −20,265 | **−79,212.46** | 25,272.80 | −91,848.86 | 1,988 | 67,208.36 | −18,688.31 |

Every form lost in every fold. The anatomy of the pure
form's loss is the story: **fees of $99,474.74 on $100k
books** — at the 1h timeframe the HA color flips so often
(3,102 fills across four ~100-day folds, ~98% of closes in
position) that the fee bleed alone nearly equals the
entire loss; the gross trading result before fees was
≈ −$15.5k. The visible red/green runs are mostly 1–3
candles long at 1h — noise, traded at full size. L=3
confirmation halves the churn and still loses $79k. The
judge vetoed 72% of actionable shifts and cut the loss by
65% versus pure — damage control, not signal: its taken
trades still lost in all four folds. Both forms also
trailed doing nothing by wide margins (pure 6.2× the B&H
loss; judged 2.1×).

Holdout duel (SECONDARY): judged aggregate **−1,549.59**
vs champion −210.17, 0/3 windows won, P&L −$2,539.66
(windows −767.20 / +146.58 / −1,919.04; B&H −$8,534.12)
— **formula FAIL**, floor never reached. Pure form:
aggregate −4,061.72, P&L −$7,326.46. Pure L=3: aggregate
−4,850.61, P&L −$8,956.37. Verdict (judged form): FAIL on
the formula, pending_approval=0.

Judge log (`data/ha_deep/shift_judge_log.json`): 2,058
confirmed shifts logged across all graded spans — 365
taken, 932 skipped, 761 no-ops (same-direction events
while positioned). The 0.55 line was applied exactly
(taken P_lot range 0.550–0.988; the highest skipped read
was 0.549998). How the taken ones did: they ARE the
judged form's trades — −$39,936.58 walk-forward and
−$2,539.66 in the duel. The judge's best reads lost too,
because its OOF was chance: there was no "a lot vs a
little" signal in the frame for it to read.

### The verdict on Kass's hypothesis

The Heikin-Ashi structure he can see — the smoothed
red/green runs — was given the fullest test the program
has ever given a feature family: its own 58-feature
indicator suite, four model classes, two label
definitions, a walk-forward, an exam, a money duel, and
his own trading rule in three forms. The answer is the
same at every level: **the HA transformation neither
adds nor destroys information relative to the raw frame —
it is a smoothing of the same candles, and the candles
are the bottleneck.** As eyes, HA matches the classic
frame fold-for-fold and loses the holdout exam narrowly.
As a trading rule, the 1h HA runs are too short to pay
their own fees: the pattern that reads as trend on a
chart is, at this timeframe, mostly alternating noise,
and trading every confirmed shift is a fee-generation
machine (−$99.5k of fees in the pure form). The judge —
Kass's "a lot or just a little" layer — could not rescue
it because the underlying frame carries no move-size
signal for it to read (OOF 0.4993). Recorded as failed
component attempts (runs 87/88), not numbered nulls. The
genuinely untested lever is unchanged: new information
(a funding-rate / open-interest feed). Nothing was
installed, crowned, or moved; the live books were never
touched.

## Verification

- **Adversarial tests first**: `tests/test_ha_deep.py`
  (20 tests) — HA recursion on hand-computed candles
  incl. the zero-body rule, run/shift state machine on
  crafted sequences, causality of all 58 features + the
  sequence series under future garble, bucket-END
  stamping of the 4h/1d state (values change only at
  boundary decision rows), the common-row set, the judge
  direction mapping, the full Shift Trader state machine
  (truth table, flip = close+open at one next-open,
  judged skip, judged-little opposing shift does not
  close, stop-out → flat until the next shift, spent
  streaks never re-fire, last-candle decisions counted
  not filled), next-open fill timing, ledger re-sum
  parity, and the compact family registry (codes 20/21).
  Trader tests watched failing on the stub
  (NotImplementedError) before `run_shift_span` was
  implemented; two of the author's own hand-traced
  expectations (prev-run color at a neutral row; an L=2
  confirmation index) were wrong against the frozen spec
  and were corrected to the spec, with the traces
  re-derived. All 20 pass.
- **Frame-layer repairs at build time** (the layer was
  delivered verified under a different import style):
  relative imports made dual-style (package + the
  project's src-on-path convention), and the guard
  delegation fixed to the eyes_v2 guard function's actual
  no-argument signature. No frame math was touched; the
  Excel export's independent HA recomputation (0.0 diff)
  stands.
- **Two-way checks**: (i) champion pin — the fresh 44+73
  arm reproduced aggregate −210.17 / P&L −$305.28 exactly
  before either money test ran; (ii) all 21 Shift Trader
  span ledgers (4 folds × 3 forms + 3 windows × 3 forms)
  re-summed from their persisted DBs alone, per-fill
  audited, agreeing with the runner's P&L to < $0.01 —
  enforced in the runner (abort on mismatch); all passed.
- **One disclosed deviation from the literal pre-reg
  text**, decided before any walk-forward number existed:
  the pre-reg segments sup positions [0, split) into 5;
  the common rows start at position 4,176 (the pick2_98
  warmup — the same 4,176-row head run 82 recorded), so
  literal segment 0 holds zero common rows and fold 1
  would have no purged fit rows at all. The segmentation
  was applied to the common-row span [4,176, split)
  instead (segments [4176,6591] / [6592,9007] /
  [9008,11423] / [11424,13839] / [13840,16254]), preserving
  5 segments, burn-in first, folds 1–4, and the t + 24
  purge verbatim. Used identically for the Part I
  corroboration and the Part II walk-forward.
- **Dashboards**: compact training + gate feeds generated
  from `src/dashboards/compact.py` and pushed verbatim
  (Training Lab: push_snapshot + push_gate; Live desk:
  push_gate, and its trading snapshot re-pushed verbatim
  from the compact trading feed after an initial
  training-feed push_snapshot to the Live slug was
  answered with default state — the Live desk's snapshot
  slot is its trading feed per the standing convention).
  Read back via get_dashboard on both: Training models[0]
  = run 88 "HA Shift Trader" (family 21, FAIL), models[1]
  = run 87 "HA-Deep Eyes" (family 20, 0.5208, FAIL); gate
  pending still run 83 on both dashboards (neither new run
  pends); declined list unchanged [61, 63, 67, 78]; Live
  trading state intact (equity $100,702.90, LONG — the bot
  opened a position on its own 22:00 UTC loop tick, after
  this study's guard window closed).
- **Guards**: the eyes_v2 guard set (live/shadow/gated
  seats incl. Book #3's gate model + manifest, lab
  champions, all three live-state files) hashed before
  (frames phase) and after (registration) — all strict
  hashes identical; paper-book DBs information-only.
- **Suite**: 432 passed / 5 failed. The session's
  pre-study baseline measured 411 passed / 6 failed (the
  same 5 documented registry-state pins — feed-size
  sentinel, audit/telemetry registry pins, run-75
  detail-window pin, gate-history strip pin — plus the
  intermittent 1-ulp shadow-vote flake, which passed on
  this run). 411 + 20 new HA tests + the flake passing =
  432 exactly; no new failures, no rebaseline taken (the
  pins are the dashboard/fix workstream's; the protected
  values were verified unaffected — both dashboards read
  back correctly with runs 87/88 present).
