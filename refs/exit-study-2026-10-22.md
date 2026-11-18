# Exit & Transition Study — 2026-10-22

Kass's request: *"play with the stop loss and take profit metrics…
tell me and pick the best way for it to close or open positions"* —
should a flip close to exactly flat first, flip through flat into
the opposite position in one motion, or stage it?

Every experiment before this one varied the **signal** and froze the
exits at the v4 recipe. This study does the opposite: the signal is
held fixed and only the exit + transition machinery moves.

## Design

- **Signal (fixed):** GBM Lag F (training.db run 26) — the most
  active of the recent graded policies. Same cached probabilities,
  same 0.40/0.40 thresholds, same rule policy, bit-identical across
  all 44 cells.
- **Grading:** the standard duel (last 792h, 3×264h windows,
  next-candle-open fills, 0.15% fees, 25% tranches / 75% cap,
  10×24h episodes per window, score = P&L − 0.5 × worst episode
  drawdown, aggregate = mean window score) **plus a robustness set**:
  the preceding 792h, same layout. Robustness-window probabilities
  come from the same arm-F recipe fit only on rows *before* that
  region (the duel model trained on rows containing the region, so
  its probabilities there would be in-sample).
- **Parametrization:** `ExitConfig` in `src/models/exit_study.py`;
  production defaults are never edited — a `StudyBroker` shadows the
  broker's class constants per instance and a `StudyEnv` adds the
  transition seam. Default config ≡ production recipe, proven by
  the identity check below.
- **Take-profit (new, config-gated, off by default):** fixed %
  targets and R-multiples (TP = entry ± k × initial-stop distance),
  mirrored for shorts. Levels derive from the current weighted entry
  and entry ATR, exactly as the stop level does. Wick-triggered
  symmetrically with stops (long TP watches the candle high, short
  TP the low); the fill is the level itself when the close stayed on
  the entry side of it, otherwise the close (gap-through rule,
  mirroring stops). A candle spanning **both** stop and TP fills the
  **stop** (conservative intra-candle ordering).
- **Transitions** (on an active opposite signal; a band "signal
  gone" flatten is an exit, never a flip):
  - **T0 through-flat (production):** the flip closes to flat; the
    new side may open on a later step. (Production is asymmetric: a
    long is closed in full by one flip signal, but a short is only
    covered one tranche per buy.)
  - **T1 one-motion reversal:** close the whole position **and**
    open the opposite tranche in the same step; fees on both legs'
    actual volume.
  - **T2 staged:** close the whole position now (including a full
    cover of a short); the open happens only if the signal persists
    into the next candle — i.e., production open behavior from flat,
    with the tranche-grind removed from the close.
- **Trip accounting:** "Env trips" is the duel counter (flat-return
  steps not caused by a stop). Win/loss and exit-reason stats use
  ledger segmentation (flat→flat segments; a flip is two segments);
  a segment's P&L is its closing fill's cash minus the cash before
  its opening fill, fees included. Long/short split attributes each
  episode's mark-to-market remainder to the still-open segment's
  side, so the split sums exactly to total P&L.

## Identity check (load-bearing)

The default-config cell, driven through the study harness,
reproduces run 26's recorded duel **exactly**: aggregate
**−1,871.18**, total P&L **−$4,416.98**, **40 round trips**, B&H
+$717.54 (pinned by `tests/test_exit_study.py::
test_default_config_reproduces_run26_duel`). Whatever the grid says,
it says about the exits — the signal path is untouched.

## What the grid actually found

### 1. Stage 1 collapsed: the exit knobs mostly never bind

The 36 exit cells produce exactly **two** distinct outcomes. All 24
stop-1.5 and stop-2.0 cells are bit-identical on both window sets,
and all 12 stop-3.0 cells are identical on the standard windows and
slightly *worse* on robustness (−$23,300 vs −$23,167). Exit counts
across every Stage-1 cell: **0 take-profit hits, 0 trailing exits**,
1 stop-out (standard) / 5–6 (robustness); everything else is a
signal exit.

An independent excursion diagnostic (per position life, both sets)
explains why:

- Median entry ATR% is **0.54%** (standard) / **0.68%**
  (robustness). 1.5× and 2.0× that are 0.8–1.4% — **both clip up to
  the 2.5% stop floor in 97–100% of position lives**. The stop
  distance actually in force is 2.5% in essentially every trade:
  at 1h, "2×ATR" is nominal; the floor is the stop.
- **No position life in either set (0 of 42 / 0 of 88) ever reached
  +3% favorable excursion** — median max favorable excursion is
  **+0.36% / +0.32%**. The trailing stop therefore never arms, and
  TP at +4%, +8%, or 2R (≈+5% at the floor) is never within reach.
  Lag F's trades are scalps that live inside a ±0.5% band and die by
  signal (median adverse excursion −0.27% / −0.45%).
- The handful of robustness stop-outs exceed the diagnostic's
  first-entry MAE count because tranche adds re-average the entry;
  the stop distance is 2.5% from the *current* entry either way.

So on this signal, stop multiple (within the tested range),
take-profit, and trailing mode are **decorative**: the signal's own
flatten rule does ~98% of the exiting. The one knob that changed
anything, a 3.0× stop, only let one robustness loser run further.

### 2. Transitions are real, and through-flat wins

Unlike the exit knobs, the transition rules produce genuinely
different fills (default exit recipe):

| Rule | Std agg | Std P&L | Rob agg | Rob P&L | Closed trips (std/rob) |
|---|---|---|---|---|---|
| **T0 through-flat** | **−1,871.18** | **−$4,416.98** | **−9,527.32** | **−$23,167.08** | 41 / 81 |
| T1 one-motion | −2,117.04 | −$5,043.91 | −10,233.01 | −$25,204.82 | 45 / 105 |
| T2 staged | −1,958.17 | −$4,677.95 | −9,592.58 | −$23,372.05 | 41 / 87 |

- **T1 (flip through flat in one motion) is the worst of the
  three** on both sets: −$627 standard / −$2,038 robustness vs T0.
  It trades more (105 closed segments on robustness vs 81) and pays
  for it — every flip pays both legs' fees immediately *and* puts
  the book on the new side a candle earlier, which on this signal
  usually means entering the side that is about to die too.
- **T2 (staged)** is closer but still behind T0 on both sets
  (−$261 / −$205): closing a short in full at the first flip signal
  is worse than T0's tranche-by-tranche cover, which acts as a
  built-in confirmation delay — by the time T0 finishes covering,
  a flip that was noise has often already died.
- **Answer to Kass's question:** close to exactly flat first
  (T0). Do not extend through flat in one motion; do not stage with
  a full close either. The pause at flat — and on the short side,
  the tranche-paced cover — is worth real money on this signal.

### 3. Where the money actually went (default cell)

Standard windows: 41 closed trips, **7 wins / 34 losses**; avg win
+$196.81 vs avg loss −$167.61. The signal doesn't lose on
win/loss size — it loses on hit rate, and no exit rule in the grid
changes hit rate. Split: longs −$276, **shorts −$4,141**.
Robustness (a strong rally, B&H +$27,402): longs −$3,053, **shorts
−$20,114** — Lag F kept shorting into a rising market and the exits
could only ration the damage.

## Full grid

Every pre-registered cell, both window sets. Identical rows are
identical outcomes (see finding 1) — they are printed in full
because the grid was pre-registered that way.


### Stage 1 — exits under T0 — Standard windows (last 792h)

| Cell | Agg score | Total P&L | B&H | Env trips | Closed trips | Avg win | Avg loss | Max ep. DD | Stop-outs | Trail exits | TP hits | Signal exits | P&L long | P&L short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop1.5·TPoff·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TPoff·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TPoff·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP4%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP4%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP4%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP8%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP8%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP8%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP2R·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP2R·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TP2R·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TPoff·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TPoff·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TPoff·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP4%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP4%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP4%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP8%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP8%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP8%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP2R·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP2R·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TP2R·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TPoff·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TPoff·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TPoff·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP4%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP4%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP4%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP8%·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP8%·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP8%·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP2R·trail-current·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP2R·trail-off·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop3·TP2R·trail-tight·T0 | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |

### Stage 2 — transitions — Standard windows (last 792h)

| Cell | Agg score | Total P&L | B&H | Env trips | Closed trips | Avg win | Avg loss | Max ep. DD | Stop-outs | Trail exits | TP hits | Signal exits | P&L long | P&L short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop2·TPoff·trail-current·T0 (control) | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop2·TPoff·trail-current·T1 | -2,117.04 | -5,043.91 | +717.54 | 39 | 45 | +191.22 | -171.02 | 1,174.34 | 1 | 0 | 0 | 44 | -481.56 | -4,562.36 |
| stop2·TPoff·trail-current·T2 | -1,958.17 | -4,677.95 | +717.54 | 40 | 41 | +196.81 | -175.29 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,401.62 |
| stop1.5·TPoff·trail-current·T0 (control) | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TPoff·trail-current·T1 | -2,117.04 | -5,043.91 | +717.54 | 39 | 45 | +191.22 | -171.02 | 1,174.34 | 1 | 0 | 0 | 44 | -481.56 | -4,562.36 |
| stop1.5·TPoff·trail-current·T2 | -1,958.17 | -4,677.95 | +717.54 | 40 | 41 | +196.81 | -175.29 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,401.62 |
| stop1.5·TPoff·trail-off·T0 (control) | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TPoff·trail-off·T1 | -2,117.04 | -5,043.91 | +717.54 | 39 | 45 | +191.22 | -171.02 | 1,174.34 | 1 | 0 | 0 | 44 | -481.56 | -4,562.36 |
| stop1.5·TPoff·trail-off·T2 | -1,958.17 | -4,677.95 | +717.54 | 40 | 41 | +196.81 | -175.29 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,401.62 |
| stop1.5·TPoff·trail-tight·T0 (control) | -1,871.18 | -4,416.98 | +717.54 | 40 | 41 | +196.81 | -167.61 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,140.64 |
| stop1.5·TPoff·trail-tight·T1 | -2,117.04 | -5,043.91 | +717.54 | 39 | 45 | +191.22 | -171.02 | 1,174.34 | 1 | 0 | 0 | 44 | -481.56 | -4,562.36 |
| stop1.5·TPoff·trail-tight·T2 | -1,958.17 | -4,677.95 | +717.54 | 40 | 41 | +196.81 | -175.29 | 1,174.34 | 1 | 0 | 0 | 40 | -276.34 | -4,401.62 |

### Stage 1 — exits under T0 — Robustness windows (preceding 792h)

| Cell | Agg score | Total P&L | B&H | Env trips | Closed trips | Avg win | Avg loss | Max ep. DD | Stop-outs | Trail exits | TP hits | Signal exits | P&L long | P&L short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop1.5·TPoff·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TPoff·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TPoff·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP4%·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP4%·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP4%·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP8%·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP8%·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP8%·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP2R·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP2R·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TP2R·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TPoff·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TPoff·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TPoff·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP4%·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP4%·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP4%·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP8%·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP8%·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP8%·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP2R·trail-current·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP2R·trail-off·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TP2R·trail-tight·T0 | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop3·TPoff·trail-current·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TPoff·trail-off·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TPoff·trail-tight·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP4%·trail-current·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP4%·trail-off·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP4%·trail-tight·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP8%·trail-current·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP8%·trail-off·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP8%·trail-tight·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP2R·trail-current·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP2R·trail-off·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |
| stop3·TP2R·trail-tight·T0 | -9,593.88 | -23,300.19 | +27,401.66 | 75 | 80 | +101.28 | -331.12 | 6,969.25 | 5 | 0 | 0 | 75 | -3,186.49 | -20,113.70 |

### Stage 2 — transitions — Robustness windows (preceding 792h)

| Cell | Agg score | Total P&L | B&H | Env trips | Closed trips | Avg win | Avg loss | Max ep. DD | Stop-outs | Trail exits | TP hits | Signal exits | P&L long | P&L short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop2·TPoff·trail-current·T0 (control) | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop2·TPoff·trail-current·T1 | -10,233.01 | -25,204.82 | +27,401.66 | 67 | 105 | +71.04 | -310.85 | 6,911.42 | 5 | 0 | 0 | 100 | -4,423.91 | -20,780.91 |
| stop2·TPoff·trail-current·T2 | -9,592.58 | -23,372.05 | +27,401.66 | 82 | 87 | +80.37 | -329.42 | 6,836.14 | 5 | 0 | 0 | 82 | -3,665.58 | -19,706.46 |
| stop1.5·TPoff·trail-current·T0 (control) | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TPoff·trail-current·T1 | -10,233.01 | -25,204.82 | +27,401.66 | 67 | 105 | +71.04 | -310.85 | 6,911.42 | 5 | 0 | 0 | 100 | -4,423.91 | -20,780.91 |
| stop1.5·TPoff·trail-current·T2 | -9,592.58 | -23,372.05 | +27,401.66 | 82 | 87 | +80.37 | -329.42 | 6,836.14 | 5 | 0 | 0 | 82 | -3,665.58 | -19,706.46 |
| stop1.5·TPoff·trail-off·T0 (control) | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TPoff·trail-off·T1 | -10,233.01 | -25,204.82 | +27,401.66 | 67 | 105 | +71.04 | -310.85 | 6,911.42 | 5 | 0 | 0 | 100 | -4,423.91 | -20,780.91 |
| stop1.5·TPoff·trail-off·T2 | -9,592.58 | -23,372.05 | +27,401.66 | 82 | 87 | +80.37 | -329.42 | 6,836.14 | 5 | 0 | 0 | 82 | -3,665.58 | -19,706.46 |
| stop1.5·TPoff·trail-tight·T0 (control) | -9,527.32 | -23,167.08 | +27,401.66 | 75 | 81 | +101.28 | -336.11 | 6,836.14 | 6 | 0 | 0 | 75 | -3,053.38 | -20,113.70 |
| stop1.5·TPoff·trail-tight·T1 | -10,233.01 | -25,204.82 | +27,401.66 | 67 | 105 | +71.04 | -310.85 | 6,911.42 | 5 | 0 | 0 | 100 | -4,423.91 | -20,780.91 |
| stop1.5·TPoff·trail-tight·T2 | -9,592.58 | -23,372.05 | +27,401.66 | 82 | 87 | +80.37 | -329.42 | 6,836.14 | 5 | 0 | 0 | 82 | -3,665.58 | -19,706.46 |

## Recommendation

**Keep the production recipe, unchanged:**

- **Transition: T0 through-flat.** A flip closes to exactly flat;
  the opposite side opens only as a fresh decision on a later step.
  Never extend through flat in one motion (T1: −$627 std /
  −$2,038 rob), and don't "fix" the short side's tranche cover into
  a staged full close (T2: −$261 std / −$205 rob).
- **Stop: 2.0×ATR% clipped [2.5%, 8%]** — knowing that at 1h the
  2.5% floor is what actually fires (median entry ATR 0.54–0.68%).
  1.5× is the identical recipe under the clips; 3.0× is worse.
- **Trailing: current (arm +3%, trail 1.5×ATR% clipped [2%, 6%]).**
  It never fired in 44 cells — it costs nothing and remains the
  right insurance for a signal whose winners actually run.
- **Take-profit: none.** Untestable on this signal (no position
  ever reached +3%, let alone +4–8%); adding one would be
  decoration. The TP mechanism is built, tested, and config-gated
  off if a future signal holds long enough to use it.

**Improvement over the current default: $0 — because nothing beat
it.** The best-scoring cells tie the default to the cent (they are
mechanically the same recipe under the ATR clips); every genuinely
different variant is worse. Ranked by mean aggregate across both
window sets, the production recipe (and its clip-identical twins)
sits at the top of all 44 cells at **−5,699.25**.

### Honesty box

- The fixed signal is a known null (run 26 lost money under every
  recipe). **The best cell still loses to buy-and-hold on both
  window sets** (standard: −$4,417 vs +$718; robustness: −$23,167
  vs +$27,402). Exits reshape losses; they do not manufacture edge,
  and this study found no recipe that changes the verdict on Lag F.
- What the recipe is worth anyway: the study clears the exit
  machinery as a suspect. When a signal with a real edge arrives,
  the standing recipe already (a) caps per-trade damage at the
  effective 2.5% stop, (b) exits ~98% by signal rather than by
  stop, so a good signal's timing is not overridden, and (c) has
  the transition discipline (through-flat) that was worth
  +$627/+$2,038 against the most tempting alternative here.
- One untested lever, flagged not recommended: the 2.5% stop
  *floor* itself. The grid varied the ATR multiple per the
  pre-registration; at 1h ATR levels the floor decides nearly
  every stop. A floor study (e.g. 1.5% / 2.0%) would be a new
  pre-registered grid, and on a signal this choppy it cuts both
  ways — tighter floors mean more stop-outs of trades that
  currently die by signal inside ±0.5%.

## Registration & integrity

- Registered as training.db runs **28–36**, family `exit_study`,
  all promoted=0 (adoption is Kass's decision; nothing deployed):
  the 8 Stage-2 transition finalists (runs 28–35) + the top-ranked
  cell (run 36, stop 1.5× T0). Run 36's BEST RECIPE verdict is a
  **tie** with the production default — a correction lab note is
  on file, and the recorded reasons show delta-agg +0. Registry
  records omit `challenger.mean_reward` (it feeds only the
  dashboard learning-curve series); full numbers are in
  `data/exit_study/results.json`.
- Dashboards: family code 5 + display names ("Exit Study — T1 s2
  noTP tr", …) added to `src/dashboards/compact.py` mirroring
  `gbm_lag`. The coverage test caught a real latent bug on the
  way: the training snapshot caps its runs list at 30, so the
  models index silently dropped the oldest runs once the registry
  passed 30 — the index now reads training.db directly and covers
  all 36 runs. **Fuse warning:** the training feed is 7,925B of
  the 8,192B test ceiling and grows ~110–140B per nightly retrain
  (models index + reward series) — the ceiling will need a feed
  diet or a raise within days, independent of this study.
- Tests: `tests/test_exit_study.py` — 19 tests (config≡production,
  TP long/short/%/R/stop-wins, trailing modes, T0/T1/T2 broker
  mechanics incl. band-buy-is-not-a-flip, env-level flip wiring,
  and the run-26 identity check), watched failing before
  implementation. **Full suite: 103 passed.**
- Production untouched: champion files, `data/paper.db`,
  `data/live_state.json`, `src/trading/broker.py`,
  `src/environment/trading_env.py`, `src/training/nightly.py`
  mtimes verified identical to the pre-study baseline; the park
  holds; `src.trading.live` never ran.
