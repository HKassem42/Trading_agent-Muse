# Indicator-Wave Phase Labels — Spec (prepared 2026-10-31, NOT built)

Status: PREPARED ONLY. Kass directed the labeling on 2026-10-30 ("label all the
indicators where it's rising and peaks and dropping at peak") and gave the
middle-line rule the same evening. The traded-system test (EWO 4h wave trader
through the production broker) is a SEPARATE decision and is not authorized
by this spec.

## Kass's rules, verbatim in substance
- Trading reads the indicators' own waves. Example (EWO): red (negative) ->
  buy at the indicator's lowest point; green -> sell at its highest point.
- Every indicator has a base white line in the middle. Above it = buy side,
  below it = sell side.
- Label every indicator's phases: rising / peak / dropping from peak / trough.

## Middle lines (territory)
- EWO: 0 | MACD: 0 | CCI: 0 | RSI: 50 | Stochastic: 50 | Williams %R: -50

## Label definition (causal — no hindsight)
- Pivot confirmation: 2 candles, matching Kass's second-candle principle
  used in the 2026-10-30 wave audit. A trough/peak is labeled only when
  confirmed; the label timestamp is the confirmation candle, never the
  extreme candle (the extreme's timestamp is stored separately as a field).
- Territory gate: a trough counts only if it forms BELOW the middle line;
  a peak counts only if it forms ABOVE the middle line. This is exactly how
  the 2026-10-30 audit was zone-gated, so audit numbers are this rule's
  numbers.
- Phases per indicator, per timeframe: trough_confirmed -> rising ->
  peak_confirmed -> dropping; plus territory (above/below middle line) as
  its own column on every row.
- Timeframes: 1h and 4h (audit ran both; every indicator was better at 4h).
- Indicators: EWO, MACD, RSI, Stochastic, CCI, Williams %R (the six
  oscillators audited 2026-10-30).

## Deliverable if Kass green-lights the build
- One inspectable dataset (Excel/CSV in Your Files + a data/ artifact):
  per candle, per indicator: value, territory, phase, last confirmed
  extreme (type, value, candles-ago).
- Wave summary table per indicator/timeframe/direction: wave count,
  median/mean causal gross move, % of waves beating the 0.30% round-trip
  fee hurdle — recomputed from the labels, checked against the audit's
  published figures (EWO 4h: 98 waves/direction, causal median +1.13%
  long / +0.97% short, ~64-65% beat fees; RSI/Williams %R ~+0.05% at 1h).

## Explicitly NOT in this spec
- No entries, exits, stops, or broker replay. No model training on the
  labels. Either of those is a new, separate Kass decision.
- No change to any live/shadow/gated book, seat, or dashboard data.
