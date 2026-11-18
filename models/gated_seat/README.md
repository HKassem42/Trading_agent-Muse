# Gated Seat — Book #3 (Run 82 Gate Forward Test)

This seat belongs to **Book #3 — the Gated Shadow**, the forward
test pre-registered in `refs/forward-test-gated-book3-2026-10-30.md`
(Kass approved the forward test 2026-10-30). It holds byte-copies
of the pinned artifacts, hash-pinned in `data/gated_seat.json`:

- `ppo.zip` — Driver: PPO run 99 (seated by Kass's direct
  order 2026-11-03; source `models/ppo_run99.zip`, sha256
  fa2c1677980bf97fd9cf885669981d90d42b8b1942b8a0fc5a3a22f4ce10fe0b).
  Segment 1's driver was PPO run 44.
- `stacker.pkl` / `stacker.json` — Eyes: Stacker run 73,
  Lab Stacker Champion #4 (pick2_98, 98 columns)
- `gate_model.pkl` — Gate: run 82's frozen meta-label model
  (LightGBM; source `data/meta_label/meta_model.pkl`,
  sha256 cf7630c0d2e386322f1422a29ee04581f30b723634478c4a875e8b5d9ad3240a),
  threshold 0.50 (Segment 1 ran at 0.55; Kass set 0.50 on
  2026-11-03 — the threshold lives in `data/gated_seat.json`)

The engine (`src/trading/live_gated.py`) verifies every file
against the manifest before trading and fails loudly on any
mismatch. The gate MODEL is NEVER retrained or refit, and
this seat does not follow future lab crownings. Changes to
the driver, the threshold, or the pairing are Kass's calls,
recorded as segments/amendments to the forward test — see
`refs/forward-test-gated-book3-2026-10-30.md` and the
`swap_history` in `data/gated_seat.json`.
