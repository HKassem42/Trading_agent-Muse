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
