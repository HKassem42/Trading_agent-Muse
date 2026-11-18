# The Driver's Seat

This directory holds the exact model artifacts the **live loop**
(`src/trading/live.py`) trades with. It is deliberately separate
from the Training Lab's champions (`models/stacker_champion.pkl`,
`models/ppo_champion.zip`), which are grading baselines for the
lab's duels (Kass's governance directive, 2026-10-25: *"Keep the
trading and training separate. I want to be able to see all the
champions and numbers myself and then tell you which one I want
in the driver seat of live trading."*).

- A lab crowning never touches this directory.
- There is **no automatic path** into the seat. The seat changes
  only when Kass names a new driver.
- `data/live_seat.json` is the manifest of record: occupant run
  ids, feature sets, and SHA-256 hashes of the seated files.

## Manual driver-change procedure (Kass's pick only)

1. Kass names the lab champion (by run id) to drive.
2. Copy that champion's artifacts over the seat files:
   - stacker: `models/stacker_champion.pkl` → `stacker.pkl`, and
     its metadata sidecar → `stacker.json` (the seated stacker must
     be a production-feature-set model the live loop can read —
     the live feature path is the production 21; a lab champion
     trained on another feature set cannot drive until it is
     productionized, a separate Kass-approved step);
   - PPO: `models/ppo_champion.zip` → `ppo.zip`.
3. Update `data/live_seat.json`: occupant `run_id`, `feature_set`,
   `sha256` (recompute), `seated_by` (e.g. "Kass pick 2026-XX-XX"),
   and `ppo_champion_run_id` when the PPO changes.
4. Let the next 15-minute live-loop run exercise the new seat and
   confirm the dashboard watch log shows WATCH PASS with the book
   (equity / cash / position) continuous with the pre-change state.
   If the loop errors, restore the previous seat files from the
   rollback shelf (`models/archive/`) and investigate before
   retrying.
