# The shadow seat (Book #2)

This directory holds the exact artifacts the SHADOW loop
(`src/trading/live_shadow.py`) trades Book #2 with:

- `ppo.zip` — PPO **run 44** (the driver; byte-identical copy of
  `models/ppo_champion.zip` as of the seat's pinning, the same
  brain that drives Book #1).
- `stacker.pkl` / `stacker.json` — stacker **run 119**, lab
  Champion #11 (the HA nightly stacker: RF+XGB+LGBM+ExtraTrees
  → LR; byte-identical copies of `models/stacker_champion.pkl` /
  `.json`; the ha10 feature set: the 10 Heikin-Ashi candle
  values — ha_dir, ha_body_pct, ha_range_pct, ha_upper_wick,
  ha_lower_wick, ha_streak, ha_slope, ha_disp, raw_ret,
  ha_body_rel).
- `gate_model.pkl` — **run 82's frozen meta-label gate**
  (LightGBM; byte-identical copy of
  `data/meta_label/meta_model.pkl`, sha256 cf7630c0…; seated
  2026-11-21 by Kass's order — the same gate Book #3 runs,
  at threshold **0.51** here vs Book #3's 0.55).
- `gate_columns.json` — the gate's 98 pick2_98 column order
  (run 73's recorded order, from `models/gated_seat/stacker.json`);
  part of the gate's identity, hash-pinned in the manifest.

The seat is recorded in `data/shadow_seat.json` (run ids, feature
set, SHA-256 hashes, start timestamp, eye-swap history). The loop
verifies every file against those hashes before it trades; a
mismatch is a loud failure, never a fallback.

## The pin

This seat is **pinned**: the stacker stays run 119 and the PPO
stays run 44 even if the Training Lab crowns a Champion #12
tomorrow. The desk exists to give THIS pairing (run 44 driving,
Champion #11's eyes) a clean live track record next to Book #1
(the driver's seat: run 44 + stacker run 40).

Pin history: the desk was seeded 2026-10-25 with run 64
(Champion #2) "pinned permanently". On **2026-10-26 Kass
explicitly overrode that pin** — he crowned run 69 Champion #3
and ordered it seated here ("Put your winning model in the
shadow one and let it start trading"). The swap copied run 69's
artifacts over `stacker.pkl` / `stacker.json` and re-pinned the
manifest; Book #2's ledger and state were NOT re-seeded because
the book was flat ($100,000, 0 fills) under run 64 — there was
no track record to blend, and the race/WFE history stays
attributable: the scored-eyes exam (+$461.81) belongs to the
pairing (44 + run 64) and the Pick #2 duel exam (+$162.12) to
the pairing (44 + run 69); both are recorded as previous-pairing
references in the manifest's `wfe_exam` entry.

On **2026-10-27 Kass delegated the Pick #2 rulings to Muse**
("do whatever you think is best"); Muse crowned run 73 lab
stacker Champion #4 and swapped the eyes 69 → 73. Unlike the
2026-10-26 swap, the book was **MID-POSITION**: SHORT
−0.311631 BTC @ $80,222.97, opened 2026-10-27 under run 69's
vote, equity $99,909.26 at the swap. By that recorded ruling the
book was NOT re-seeded (the alternative the procedure below
prescribes); the position record and the blend it implies ride
in the manifest's `eyes_swapped` entry, and fills from the swap
onward carry the `champion4_run73` context tag, so the track
record stays attributable candle by candle.

On **2026-11-09 Kass ordered the four pending HA stacker
winners crowned and the highest seated here** ("promote all
the champions that won and add the highest one of our numbers
to book 2 — change out the model in book two to our highest
%"); Muse crowned runs 113 / 115 / 117 / 119 in sequence
(Champions #8–#11) and swapped the eyes 73 → 119. The book was
again **MID-POSITION**: SHORT −0.342743 BTC @ blended entry
$74,364.07, stop anchor $75,895.00, equity $100,466.97 at the
swap (20 fills under run 73). By Kass's recorded order the book
was NOT re-seeded; the position record rides in the manifest's
`eyes_swapped` entry, and fills from the swap onward carry the
`champion11_run119` context tag. The swap also changed the
eyes' FRAME (pick2_98 scored percentiles → ha10 HA candle
values): the shadow loop and the shadow/master vision feeds
dispatch on the seat's `feature_set` tag, the nightly grading
doctrine applied to this seat.

## The gate (added 2026-11-21, Kass's order)

Kass ordered Book #3's gate added to this book at 51% ("adding
the gate on book 2 that book 3 has… keep book 3 the way it
is"). The loop now judges ENTRY attempts (opens and presses,
long or short) with run 82's frozen model before dispatch: the
104-vector is the 98 pick2_98 scored values at the candle (a
frame the loop builds for the gate alone, in the order pinned
in `gate_columns.json`) plus [run 119's p_up, |p_up−0.5|,
atr_rel, volatility_21, atr_pct_raw, side]. Below 0.51 the
entry is vetoed (substituted with HOLD) and recorded in
`data/paper_shadow.db`'s `gate_decisions` table with the
threshold it was judged at; exits, covers and stops are never
gated. The book was FLAT ($100,520.11, last candle
2026-11-21T12:00Z) when the gate was seated — no re-seed; the
manifest's `gate_added` entry marks the segment boundary, and
a manifest without a gate block runs the book exactly as
before. Book #3's gate (same model, 0.55) is untouched.

## Changing the shadow seat (Kass only, manual)

A seat change is a decision, not an event — no code path does it
automatically. To change it:

1. Kass names the new driver/eyes in chat (or records a
   delegation of the ruling, as on 2026-10-27).
2. Copy the named artifacts over `ppo.zip` / `stacker.pkl` /
   `stacker.json` here.
3. Update `data/shadow_seat.json` (run ids, champion label,
   hashes, swap record). If the book holds fills or a position
   under the outgoing eyes, re-seed instead: archive
   `data/paper_shadow.db`, delete `data/live_state_shadow.json`,
   and run `python -m src.trading.live_shadow --seed` — a new
   configuration starts a NEW flat $100k book; track records are
   never blended across configurations. (The 2026-10-26 swap is
   the documented flat-book exception; the 2026-10-27 swap is
   the documented mid-position exception, taken by recorded
   ruling with the position provenance written into the
   manifest.)
4. Note the change in the lab notebook with the reason.
