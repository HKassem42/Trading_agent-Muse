# Forward Test — Book #3, the Gated Shadow (Run 82 Gate, Live)

Pre-registered 2026-10-30, BEFORE the book's first loop run.
Approved by Kass 2026-10-30 (~13:34 ET): "Go ahead and run the
forward too if u want" — run 82's ruling is thereby resolved as
FORWARD TEST (not crowned, not back-burnered). The gate study
itself is frozen in refs/meta-label-study-2026-10-30.md; its
duel verdict was PASS by the frozen criteria on THIN evidence
(7 holdout entries, 2 vetoes; meta-model OOF balanced accuracy
0.518). This forward test exists because the duel could not
settle whether the gate helps live. Nothing in this document
may be revised after the first processed candle except the
recorded start anchor (bookkeeping, filled at seed time below)
and the checkpoint report itself.

## Hypothesis

Run 82's frozen meta-label gate, applied live to the champion
pair's own proposal stream, improves the pair's net outcome:
entries the gate vetoes would, on average, have lost money,
and Book #3's equity path beats Book #2's (the identical pair,
ungated) over identical windows.

## The book (frozen configuration)

- **Book #3 — Gated Shadow**: the exact twin of Book #2.
  Driver: PPO run 44 (`models/ppo_champion.zip`, sha256
  e610ddf3a6f6ffe7b7b566694482c5930cbed7cbbc7e4501bec00ed489bae4cb).
  Eyes: Stacker run 73, Lab Stacker Champion #4 (pick2_98,
  98 columns; `models/stacker_champion.pkl`, sha256
  89b633945843062a85d71bf8a2c97c238446ce5bd4fdfb783280daa2e75364a7).
  Production PaperBroker rules (25% tranche, 75% exposure cap,
  0.15% fee/side, ATR stops/trailing, stop-anchor ratchet),
  decisions on closed 1h candles, fresh **$100,000 FLAT**.
- **The only difference from Book #2**: every ENTRY attempt is
  evaluated by run 82's gate; P(profitable) < 0.55 → the entry
  is VETOED (logged, not executed). Exits and all position
  management are never gated.
- **Frozen gate identity**: run 82, family `meta_label`,
  LGBMClassifier (300 est, lr 0.05, leaves 31, seed 42) fitted
  on the study's 489 training-region candidates.
  Artifact: `data/meta_label/meta_model.pkl`,
  **sha256 cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a**.
  The file is COPIED into Book #3's own seat
  (`models/gated_seat/gate_model.pkl`) and hash-pinned in
  `data/gated_seat.json`; the engine refuses to trade if the
  copy's hash differs. **Never retrained, never refit.**
  Threshold: **0.55** (the study's frozen GATE_THRESHOLD).
- **Feature vector**: the study's exact 104-vector, built by
  `src/training/meta_label.py`'s `feature_row` — the 98
  pick2_98 scored values at the decision candle in the
  champion's recorded column order, then
  [p_up, |p_up − 0.5|, atr_rel, volatility_21, atr_pct_raw,
  side], where p_up is the seated run-73 stacker's vote on that
  candle and the three context values are that candle's
  `add_features` row (the same source the study's supervised
  frame carried). No re-derived variant is permitted; parity
  is pinned by tests against the study's own saved holdout
  arrays (frame_hold.npy / votes_hold.npy / context.npz).
- **Gate decision point**: on each candle, after the broker's
  protective stop check and before order dispatch — the exact
  point at which Book #2 executes its entries — classified by
  `meta_label.is_entry_attempt(action, broker.btc)` at that
  point (a buy while short is a cover, a short while long is a
  close: management, never gated).

## Start

- Seeded 2026-10-30. **Start timestamp (anchor):
  2026-10-30T16:00:00 UTC** — the newest closed candle at seed
  time (anchor close $77,202.35), recorded in
  `data/gated_seat.json` as `started` and in the book's state;
  candles at or before the anchor are never backfilled. The
  book's first processed candle is the first candle that
  closes after the anchor (the 17:00 UTC candle).

## Evidence (frozen definitions)

- **Primary**: the `gate_decisions` table in
  `data/paper_gated.db` — EVERY entry attempt the pair makes
  in Book #3, recorded by the gated engine itself at decision
  time: candle ts, side (+1 long / −1 short), gate
  probability, threshold, verdict (TAKEN / VETOED). Each VETO
  is scored against the corresponding fill Book #2 actually
  took: a veto at candle T for side S corresponds to Book
  #2's entry fill executed on candle T in the same direction
  (both books run the identical pair on the identical candle
  stream; while their positions agree, the proposal streams
  are identical). **A veto is CORRECT iff that Book #2
  trade's realized net P&L ≤ 0 once the trade is closed**
  (round trip measured from that fill to its closing fill(s)
  in Book #2's ledger, fees included). After the books'
  positions diverge (the first veto changes Book #3's path),
  later proposals may legitimately differ between the books;
  vetoes without a same-candle Book #2 counterpart are
  recorded and reported as unscoreable, never silently
  dropped or force-matched.
- **Secondary**: Book #3 vs Book #2 equity and P&L over
  identical windows (the gated race feed), with Book #1 and
  buy-and-hold as context only.
- **Review checkpoint**: after **10 vetoes OR 30 days** from
  the start, whichever comes first — report the scored vetoes
  and the book-vs-book comparison to Kass. No ruling is
  implied by the checkpoint; promotion, seat changes, or
  ending the test early are Kass's calls.

## No mid-test changes

Threshold (0.55), the gate model file, the pair, the broker
rules, and this evidence definition are frozen for the life
of the test. Any change ends this test; a changed
configuration is a new test with a new pre-registration.
Book #3 lives under the promotion lock like every other
book: it is a paper forward test, never a promotion path by
itself.

## Isolation

Book #3 writes ONLY: `data/paper_gated.db`,
`data/live_state_gated.json`, `models/gated_seat/`,
`data/gated_seat.json`. It reads Book #1/#2 ledgers read-only
for the race comparison. The engine's write paths are
disjoint from Book #1/#2 files by construction (its own
config) and the perimeter is proven in
tests/test_gated_desk.py (Book #1/#2 files bit-identical
across a gated seed + step).

---

## AMENDMENT — Segment 2 (2026-11-03, Kass's direct order)

Appended 2026-11-04T00:17Z. The original pre-registration above
is unchanged; this amendment records a configuration change
Kass ordered in the main chat on 2026-11-03 (~19:14 ET):

> "switch out book three to 99 I wanna see how it does"
> (and on the gate threshold:) "Our gate might be a little too
> conservative if you wanna put it back at 50 percent instead
> of 55 as the veto that might be good too"

Per the "No mid-test changes" clause above, a changed
configuration is a new segment of the test. The book, the
ledger, and the gate_decisions table continue; every decision
row records the threshold it was judged at, so the segments
are separable in the evidence itself.

### Segment 1 — CLOSED (2026-10-30T16:00Z → 2026-11-03T23:00Z)

Configuration: driver PPO run 44 · eyes Stacker run 73 ·
run 82 gate @ 0.55. The segment closes at the last candle
processed under that seat (state last_ts 2026-11-03T23:00Z).

Record: **7 entry attempts, 0 taken, 7 vetoed.** The book
never traded; equity $100,000 throughout.

- Vetoes #1–#2 (longs, 2026-10-30 20:00Z / 21:00Z, probas
  0.1394 / 0.5454): scored **SAVED** — the corresponding
  Books #1/#2 round trip closed net −$62.36 (Book #1) /
  −$62.11 (Book #2) while Book #3 stayed flat.
- Vetoes #3–#7 (shorts, 2026-10-31T17:00Z → 2026-11-01T09:00Z,
  probas 0.1977 / 0.2129 / 0.0829 / 0.2040 / 0.1216): score
  together against the shared Books #1/#2 short opened
  2026-10-31T17:00Z @ $77,559.24 — **still open at the swap**,
  so their scoring remains pending and carries into Segment 2
  unchanged (the daily veto-outcome check stands).

### Segment 2 — OPEN (from the first candle under the new seat)

Configuration: **driver PPO run 99** · eyes Stacker run 73
(unchanged) · **run 82 gate model unchanged** (sha256
cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a,
never retrained or refit) · **gate threshold 0.50**.

Run 99 provenance: the PPO challenger from the 2026-11-03
midnight retrain (registry run 99, ts 2026-11-03T05:24:22Z),
preserved at `models/ppo_run99.zip`, sha256
fa2c1677980bf97fd9cf885669981d90d42b8b1942b8a0fc5a3a22f4ce10fe0b
(mtime and reward-curve cross-checked against the registry).
Its recorded duel verdict is **NOT PROMOTED** (1/3 windows,
aggregate −760.78 vs champion +23.07, P&L +$697.65 vs
+$531.86 — lost on the score formula's drawdown penalty).
It sits in Book #3 by Kass's direct pick, not by promotion;
the promotion lock is untouched and Book #1's live seat is
unchanged.

At the swap the book was FLAT, $100,000, 0 trades. Pre-swap
seat backed up to
`models/archive/20261104T0016_pre_book3_driver99_gate050/`.

### Rationale (Kass's, recorded)

At 0.55 the gate vetoed 7 of 7 proposals across five days:
the forward test produced veto evidence but zero taken-trade
evidence — nothing to measure the gate's discrimination on
the trades it lets through, and nothing to measure run 99's
driving at all. At 0.50, exactly **1 of the 7** past proposals
would have passed (veto #2, the 2026-10-30T21:00Z long at
proba 0.5454); the other six were all below 0.22. Segment 2
therefore tests a modestly less conservative gate, not an
open door.

### Evidence rules from here

- Veto scoring against Book #2's fills continues per
  decision, as pre-registered; Segment-2 vetoes score against
  Book #2's corresponding candles on the same basis.
- The 10-veto review checkpoint counts all recorded vetoes
  (7 at the swap), with Segment-2 decisions reported
  separately alongside.
- Any further change to the driver, the gate model, or the
  threshold is another Kass-ordered segment, appended here —
  never a silent edit.

## Amendment — Segment 3 (2026-11-04, Kass's direct order)

### Segment 2 — CLOSED (2026-11-04T00:00Z → 2026-11-04T21:00Z candles)

Record: **14 entry attempts, 9 taken, 5 vetoed** at threshold
0.50. The book's first trade (2026-11-04T02:00Z long, gate
passed at 0.9386) closed +$32.73; a short stack built the
same day stopped out for **−$2,257.27** (2026-11-04T15:00Z
candle) — nearly the whole equity gap vs Book #2. At the
close the book was SHORT −0.912837 BTC @ $80,280.14 (stop
$82,219.59), equity $97,779.37, 10 trades total.

Scoring completed since the Segment 2 amendment was written:
vetoes #3–#7 scored **SAVED** — the shared Books #1/#2 short
they blocked stopped out 2026-11-04T14:00Z @ $79,498.22 and
Book #2's round trip netted **−$702.89** (Book #1 −$705.69).
Segment 1's gate record is final: **7-for-7 vetoes SAVED,
both blocked round trips losers.** The pre-registered
10-veto checkpoint was reached 2026-11-04T19:00Z (veto #10,
short-side, proba 0.0421) and reported to Kass the same day.
Totals at the Segment 2 close: **21 attempts, 9 taken,
12 vetoed.**

### Segment 3 — OPEN (from the first candle under the restored seat)

Configuration: **driver PPO run 99** (unchanged) · eyes
Stacker run 73 (unchanged) · **run 82 gate model unchanged**
(sha256 cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a,
never retrained or refit) · **gate threshold restored to
0.55**.

Kass's order (main chat, 2026-11-04 17:37 ET, answering the
10-veto checkpoint review): *"Tighten book three back to
.55."* The review put the two segments side by side —
Segment 1 at 0.55 went 7-for-7 SAVED but never traded;
Segment 2 at 0.50 traded, and its taken side is dominated
by the −$2,257.27 stop-out. Kass picked the tighter gate;
the run 99 driver stays, so Segment 3 reads as run 99's
brain under the original 0.55 discipline.

The change landed 2026-11-04T22:39Z, mid-position (the open
short carries over; exits are never gated, so the position
is unaffected — only future entries face 0.55). Pre-change
seat backed up to
`models/archive/20261104T2239_pre_book3_gate055_segment3/`.
Segment 3 begins with the first candle processed under the
restored seat (the 2026-11-04T22:00:00Z candle).

### Evidence rules from here

- Decision rows keep recording the threshold each attempt
  was judged at; segment totals are reported per segment
  alongside the running totals.
- The next review checkpoint is the pre-registered 30-day
  mark (~2026-11-29).
- Run 83's gate remains separately PENDING KASS; nothing in
  this amendment rules on it.
