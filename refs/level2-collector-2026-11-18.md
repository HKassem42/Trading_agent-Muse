# Level 2 Order-Book Collector — launched 2026-11-18

Kass approved connecting the Coinbase Level 2 book on 2026-11-18. This
is a **data-collection runway only**: no trading code, no model, no
dashboard reads it yet. Everything the system has ever been graded on
recycles OHLCV candles; the order book is the first genuinely new
information source (who is stacked on the bid vs the ask, at what
depth, right now). Level 2 has no downloadable history, so the runway
comes first: collect honestly for ~6–8 weeks, then design a
pre-registered Pick on the collected data.

## Source

- Coinbase Advanced Trade WebSocket, public market-data endpoint
  `wss://advanced-trade-ws.coinbase.com` — no authentication for
  market data.
- Channels on ONE connection: `level2` (one snapshot frame, then
  price-level updates; quantity 0 removes a level) and `heartbeats`
  (liveness). Product: BTC-USD only.
- Library: `websockets` (already in the project venv; recorded in
  requirements.txt).

## Components

| Piece | Path |
|---|---|
| Collector (book, sampling, WS supervision) | `src/data/l2_collector.py` |
| Hourly rollup (pure math + DB) | `src/data/l2_features.py` |
| Storage | `data/l2_book.db` (tables `l2_minutes`, `l2_hours`) |
| Liveness status (rewritten every 10s) | `data/l2_status.json` |
| Log | `logs/l2_collector.log` |
| Service | systemd unit `tam-l2-collector` (Restart=always, RestartSec=5, enabled) |
| Tests | `tests/test_l2_collector.py` (9 tests, hand-computed fixtures) |

Service management:

```bash
systemctl status tam-l2-collector
systemctl restart tam-l2-collector
journalctl -u tam-l2-collector -f        # or tail logs/l2_collector.log
```

## What is stored — aggregates only

The raw update stream (thousands of messages/sec) and book snapshots
are never stored. Every 60 seconds the collector samples the CURRENT
book into one `l2_minutes` row (ts = minute start, UTC epoch):

- `mid` = (best_bid + best_ask) / 2
- `spread_bps` = (best_ask − best_bid) / mid × 1e4 (BTC-USD's 1-cent
  spread at ~$65k is ≈ 0.0015 bps — tiny numbers here are correct)
- `micro_dev_bps` = (microprice − mid) / mid × 1e4, where
  microprice = (best_ask×bid_qty + best_bid×ask_qty) / (bid_qty +
  ask_qty) at top of book
- `imb_10bps / imb_50bps / imb_100bps / imb_200bps` — order-book
  imbalance at depth bands ±0.1% / ±0.5% / ±1% / ±2% of mid:
  (bid_qty − ask_qty) / (bid_qty + ask_qty), quantities in BTC summed
  over levels inside the band; always in [−1, 1]
- `depth_1pct_btc` — total BTC within ±1% of mid, both sides
- `best_bid_qty`, `best_ask_qty`, `n_bid_levels`, `n_ask_levels`
  (BTC-USD runs ~14k bid / ~24k ask levels)
- `coverage` — seconds of that minute the book was live and
  consistent (see honesty rules)
- `sampled_at` — when the row was written

On each hour close, `l2_hours` gets one row per completed UTC hour
(hour-start ts, the project's candle convention): mean and
end-of-hour (last valid minute) for spread, microprice deviation and
each imbalance band; mean `depth_1pct_btc`; `minutes_present` (valid
minutes); `coverage_mean`; and `usable`. The rollup for hour H uses
ONLY minute rows inside [H, H+3600) — causal by construction, final
exactly at H's close when the hourly loops decide. Un-rolled
completed hours are backfilled on collector startup (and any time via
`python -m src.data.l2_features`).

Storage growth: ~0.25 MB/day of minute rows (~90 MB/year); hourly
rows are negligible.

## Gap-honesty rules

- The book is trusted only between a fresh snapshot and the next
  disconnect / sequence gap / stale feed (>20s without a message).
  Any of those tears the session down; the book is rebuilt from a new
  snapshot. A stale book is never sampled.
- `coverage` counts true live seconds per minute, banked at exact
  interval boundaries — a minute spanning a reconnect is partial
  (e.g. the 2026-11-18 restart test minute recorded 27.6s), never
  rounded up.
- A minute whose book is not live at sample time is still written,
  with NULL features and its true coverage. A minute the process
  missed entirely (e.g. killed mid-minute by a restart, finalized by
  the successor process) keeps only the successor's live seconds —
  coverage understates rather than invents. Absence and dead minutes
  stay distinguishable from real data.
- `usable` = 1 only if `minutes_present ≥ 50` AND
  `coverage_mean ≥ 55s`. Thin hours are flagged, never silently
  trusted. An hour with rows but zero valid minutes still gets a
  rollup (usable=0, NULL means): "we were down" ≠ "no data".

## Verification at launch (2026-11-18)

- Tests first: 9 new tests (hand-computed fixture book, update
  add/change/remove, coverage splitting across minutes, rollup
  means/end/thresholds/causality) failed on the missing modules, then
  passed; 15/15 with the market-data-cache smoke suite.
- Live: snapshot applied (13,861 bid / 24,371 ask levels); minute
  rows landing with mid within a few dollars of Coinbase spot,
  spread ≈ 0.0015 bps, imbalances in [−1, 1].
- `systemctl restart` test: service reconnected, re-snapshotted and
  resumed appending; the interrupted minute shows coverage 27.6s and
  the following minute is back to 60.0s.
- Rollup path exercised end-to-end against a copy of the live DB.

## Forward plan

Collect. Once ~1,000+ usable hourly rows exist (~6–8 weeks), design
a pre-registered Muse's Pick on the L2 features — the leading
candidate shape is an entry gate in the run-82 meta-label style (the
only other design that has passed here, and the one needing the least
data to prove itself), with a full feature frame as the alternative.
Nothing about the books, champions, or grading changes until that
Pick is pre-registered and graded.

## Trades tape (Time & Sales) — added 2026-11-18

Kass approved adding the tape the same day the book collector
launched. The book shows resting intent; the tape shows money that
actually changed hands, and which side was the aggressor. It rides
the SAME websocket connection as a third subscription:
`market_trades` for BTC-USD (public, no auth). Coinbase batches tape
updates over ~250 ms; prints are bucketed by their TRADE time, never
receipt time, and deduped by trade_id (the subscribe-time snapshot
can overlap the first updates).

### Side convention — the load-bearing detail

Coinbase's Advanced Trade docs state the market_trades `side` field
"refers to the makers side" — the RESTING order's side. The
aggressor (taker) is the opposite:

- print `side=BUY` → a resting buy was hit by a seller →
  **SELL-initiated** volume
- print `side=SELL` → a resting sell was lifted by a buyer →
  **BUY-initiated** volume

All buy/sell accounting uses the TAKER side (`taker_side()` in
`l2_collector.py`, pinned by tests: a fixture print of side "BUY"
must land in `sell_vol`). Getting this backwards would flip the sign
of the single most valuable tape feature, `signed_imbalance`.

### What is stored

Per minute → `l2_trades_minutes` (ts = minute start, UTC):

- `buy_vol` / `sell_vol` — BTC volume by TAKER side; `total_vol`
  (buy + sell, exactly), `n_trades`, `n_buy`, `n_sell`
- `vwap` — volume-weighted price over the minute's prints
- `max_qty` — largest single print (BTC)
- `large_vol` — BTC volume from prints with qty ≥ **0.5 BTC** (the
  "large" threshold)
- `coverage` — seconds of the minute the socket was connected with
  `market_trades` subscribed (own tracker, same banking rules as the
  book's)

Tape minutes finalize ONE MINUTE behind book minutes: the ~250 ms
batching means a print executed at :59.9 can arrive just after the
boundary, and the lag gives every print its full minute to land in
its own bucket. Cost of the lag: a process restart loses the
in-flight tape minute entirely (its accumulator dies with the
process) on top of the partial restart minute — the lost minute is
written NULL with coverage 0 by the successor, never reconstructed.

Per completed hour → `l2_trades_hours` (same hour-start convention,
same `usable` rule: minutes_present ≥ 50 AND coverage_mean ≥ 55):

- `buy_vol`, `sell_vol` — SUMMED over the hour's valid minutes
- `signed_imbalance` = (buy − sell) / (buy + sell) computed **from
  the summed volumes** — never the mean of per-minute imbalances,
  which would weight a 0.01 BTC minute the same as a 10 BTC one
  (pinned by a divergence fixture in the tests); NULL if the hour's
  total volume is 0
- `n_trades`, `vwap` (volume-weighted across minutes), `max_qty`,
  `large_share` = large_vol / total_vol
- `minutes_present` counts live minutes INCLUDING quiet ones (a
  zero-trade minute is valid data); down minutes contribute only
  their coverage

### Gap honesty for a snapshot-less channel

The tape has no snapshot to rebuild from (the subscribe-time
snapshot only carries the last few seconds of prints). Trades during
an outage are simply LOST — never backfilled, never invented; the
minute's coverage number says how much of it we actually saw. Prints
arriving for an already-finalized minute are counted in
`trades_late_dropped` (status file) and dropped, never merged into a
written row. A live minute with zero trades is written with ZEROS
(vwap NULL — 0/0 is undefined); a down minute is written with NULL
aggregates and its true coverage, exactly mirroring the book side.

Status: `data/l2_status.json` now also carries `tape_subscribed`,
`tape_live`, `last_trade_ts`, `trades_current_minute`,
`trades_last_minute`, `trades_total`, `trades_late_dropped`.

### Verification at launch (2026-11-18)

- Tests first: 14 new tests in `tests/test_l2_tape.py` (side
  inversion, hand-computed minute aggregation, inclusive 0.5 BTC
  large threshold, zero-vs-NULL minute rows, trade-time bucketing +
  trade_id dedupe, late-print drops, from-sums imbalance divergence,
  causality, usable thresholds) failed on the missing code, then
  passed; 29/29 with the existing L2 + market-data-cache suites.
- Live: all three subscriptions confirmed on one connection; tape
  minutes landing at ~200–2,500 prints/minute.
- **Independent cross-check:** over the full-coverage minutes
  available at verification (20:29–20:30 UTC), summed tape volume
  matched Coinbase's own 1-minute candle volume EXACTLY — 19.2851
  and 21.8869 BTC, diff +0.000%. (The full-hour version of this
  check awaits the first completed tape hour; the rollup path was
  exercised end-to-end against a copy of the live DB.)
- Restart test (`systemctl restart`): service reconnected,
  re-snapshotted and resumed; book minute of the restart shows
  coverage 34.5s and the tape minute 34.4s; the in-flight tape
  minute (20:31) was written NULL/coverage 0 per the lag rule
  above; the following minutes are back to 60.0s on both sides.

### Environment note (2026-11-18)

The VM was replaced at 20:08 UTC and the systemd unit (installed
under /etc of the previous instance) did not survive it — the
collector had been down since the old instance died (~06:46 UTC;
06:27–06:45 UTC of launch data is intact in the DB). The unit is
recreated from the copy now kept IN the project at
`system/tam-l2-collector.service` (install: `cp system/tam-l2-collector.service
/etc/systemd/system/ && systemctl daemon-reload && systemctl enable
--now tam-l2-collector`). Hours with no collector have no rows at
all — absence stays distinguishable from dead minutes.
