# HA seat (Book #4) — manual seat-change procedure

This seat is PINNED: PPO run 44 (driver) + stacker run 110
(Muse's Pick #11, Lab Stacker Champion #6, eyes). It does NOT
follow future lab crownings. The stacker source is the Pick #11
study artifact itself (`data/muse_pick11/pick11_stacker.pkl`),
so later crownings cannot move this seat by accident.

A seat change is Kass's explicit decision only. Procedure:

1. Archive the current seat: copy `models/ha_seat/` and
   `data/ha_seat.json` into `models/archive/<ts>_pre_ha_seat_change/`.
2. Copy the new pinned artifacts into `models/ha_seat/`
   (`stacker.pkl`, `stacker.json`, `ppo.zip`) from their
   canonical sources; record the sources in the manifest.
3. Regenerate `data/ha_seat.json` (hashes must match the files;
   `src/trading/live_ha.py::build_manifest` is the shape).
4. Run one engine step manually and verify it trades on the new
   seat before the loop's next scheduled run.
5. Record the change (book state at swap time, Kass's order) in
   the manifest and in a lab note.

The engine verifies every seat file against the manifest on
every run and refuses loudly on any mismatch — never edit a
seat file without updating the manifest in the same pass.
