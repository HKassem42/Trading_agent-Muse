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
