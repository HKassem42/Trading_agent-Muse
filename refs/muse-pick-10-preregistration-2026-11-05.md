# Muse's Pick #10 — LSTM over Heikin-Ashi Sequences — Pre-registration

**Frozen:** 2026-11-05, BEFORE any Pick #10 fit. Ordered by Kass
2026-11-05 13:51 ET: "Try the exact thing but with the HA
sequence I think it will be able to read the red and green
candles better since they form in a row when it's HA."

## Design — Pick #9 with ONE variable changed

Identical to Pick #9 (refs/muse-pick-9-preregistration-2026-11-05.md)
in every respect — same saved frame, **same exam** (the last 792
supervised rows of Pick #9's fresh production-path build, from
2026-10-03 19:00 UTC, majority 0.5088), same champions re-scored
on that exam (run 96: 0.5215, run 73: 0.5189 — computed two ways
each in Pick #9 and reused here because the exam is literally
the same rows), same LSTM (hidden 64, dropout 0.2,
class-weighted BCE, Adam 1e-3, batch 256, ≤40 epochs, early stop
on the last 792 train rows), same seeds 42/7/123 ensembled, same
24-candle windows, same purged 5-fold walk-forward (24-row
embargo), same threshold 0.50, same frozen bar.

**The one change — the input matrix.** Instead of the 21
production features, each candle is represented by 10 values
computed from its **Heikin-Ashi candle** (HA OHLC via the
program's own causal recursion, `models.ha_seq.heikin_ashi`):

1. `ha_dir` — candle color: +1 green / −1 red
2. `ha_body_pct` — (ha_close − ha_open) / ha_open
3. `ha_range_pct` — (ha_high − ha_low) / ha_open
4. `ha_upper_wick` — upper wick share of the HA range
5. `ha_lower_wick` — lower wick share of the HA range
6. `ha_streak` — signed length of the current same-color run,
   capped at ±12, divided by 12 (Kass's "form in a row" thesis,
   given to the model explicitly)
7. `ha_slope` — HA close change vs prior HA close
8. `ha_disp` — (ha_close − raw close) / raw close
9. `raw_ret` — the raw candle's own 1h return
10. `ha_body_rel` — HA body as a share of the HA range (signed)

HA is a strictly causal filter (row t depends only on candles
≤ t); the builder is truncation-tested before fitting. HA
recursion is seeded at the frame start (730d back) — identical
computation for every row, no future data anywhere.

**Control:** the flat Pick recipe on the same 10 HA rows
(sequence-vs-flat read on the HA diet).

**Disclosure:** run 20 (HA-Seq) read HA candle shapes with a
CNN+LSTM at 90-candle windows; runs 87/88 graded flat HA-derived
features and Kass's HA shift rules. Neither is this
configuration: a pure LSTM over 24-candle HA sequences, graded
to the Pick standard. This study isolates the input
representation under Pick #9's exact grading.

## The bar (frozen, identical to Pick #9)

PENDING KASS only if ALL hold: exam acc > run 96 (0.5215) AND
> run 73 (0.5189) AND > majority (0.5088), clean anatomy
(up-calls ≥ 8, precision > 0.05), walk-forward mean acc >
walk-forward majority. BEAT THE BAR noted if also > 0.5354.
Otherwise FAIL. Guards (champion/seat/shadow/gated/run-73
hashes) checked before registration; lab only, promotion lock.
