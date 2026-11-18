# TradingAgent-Muse — Full Backup Branch

This branch (`full-backup`) is the complete disaster-recovery backup of the
TradingAgent-Muse project. It is an orphan branch: it shares no history
with `main`.

## What's here

Everything needed to rebuild the project from scratch:

- **Code** — `src/`, `tests/`, `scripts/`, `refs/`, `config/`, docs
  (the same tree the `main` branch carries as text snapshots).
- **Trained models** — `models/`: the lab champions
  (`ppo_champion.zip`, `stacker_champion.pkl`), every book's pinned seat
  copies (`live_seat/`, `shadow_seat/`, `gated_seat/`, `ha_seat/`,
  `champ_seat/`), and the champion archive.
- **Records** — `data/training.db` (the run registry), `data/lab_notes.json`,
  decision logs, and the seat manifests (`data/*_seat.json`).
- **Live paper-book state** — `data/paper*.db` and `data/live_state*.json`
  for all five books, as of the last backup run.

## What's NOT here (regenerable)

- `.venv/` — rebuild with `pip install -r requirements.txt`.
- `data/cache/` — the shared candle cache; refetched from Coinbase.
- `data/train_tmp_*.db` — throwaway training scratch databases.
- `data/timeseries_matrix/` — rebuilt by `scripts/build_timeseries_matrix*.py`.
- Log files — diagnostics only.

## Branches

- `main` — code snapshot history (text only, pushed via Muse's GitHub
  connector snapshot route).
- `full-backup` (this branch) — the full backup above, maintained with real
  git over SSH by Muse, updated nightly.

## Restore

1. Clone this branch.
2. Create the venv and install `requirements.txt`.
3. The books, registry, and models are all in `data/` and `models/` —
   the live loops pick up from the state files as they stand.

Last initial backup: 2026-11-17.
