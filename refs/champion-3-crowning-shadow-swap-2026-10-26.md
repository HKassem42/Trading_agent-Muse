# Champion #3 crowning + shadow seat swap — 2026-10-26

Kass's decision (2026-10-26 ~22:19 ET), verbatim:

> "Go ahead and crown it and swap it out for trading dashboard #2
> model. Put your winning model in the shadow one and let it start
> trading, great job muse"

Two moves executed 2026-10-27T02:32:14Z (22:32 ET), ahead of the
midnight retrain. Both mirror the run 64 crowning mechanics of
2026-10-25. The promotion lock is satisfied by Kass's explicit
approval of run 69's pending promotion; it is not lifted
generally.

## Move 1 — run 69 crowned LAB STACKER CHAMPION #3

- Rollback shelf first: `nightly.archive_champions()` →
  `models/archive/20261027T023218_20261026_pre_champion3/`
  (run 64's stacker pkl `c0373d15…`, meta `52a93161…`, PPO zip).
- `models/stacker_champion.pkl` ← `data/muse_pick/stacker_pick.pkl`
  (byte-identical, sha256 `20705159775147d1d81baf25fffd507aac93aa0c7342fd795a8f79b47947cb89`).
- `models/stacker_champion.json` written in run 64's shape, all
  grades verified from training.db run 69's record (not the
  brief): accuracy 0.5265151515151515, precision
  0.5167464114832536, wf_accuracy 0.5021847070506456,
  wf_precision 0.500419523113745, folds 5, feature_set
  "scored102", the 94 scored feature_columns, run_id 69,
  champion_no 3, promoted_scope "lab champion — Champion #3",
  display "Muse's Pick Stacker — Champion #3", crowned_by Kass.
- Registry: run 69 `promoted = 1`; pending resolved in the prior
  decisions' pattern — `pending_decision` "approved",
  `pending_decided_by` "Kass", `pending_decided_at`
  2026-10-27T02:32:14+00:00, reason "Kass approved 2026-10-26 —
  crowned lab stacker Champion #3; by the same decision seated
  as the shadow desk's eyes (Book #2)". Lab note added:
  "Lab champion crowned — run 69 is stacker Champion #3".
- Roster (compact.py `champs`): #1 run 40 (prod21, status 2 —
  former, in live seat), #2 run 64 (scored102, status 4 — former,
  in no seat), #3 run 69 (scored102, status 1 — current). Status 4
  is new: status 2's text ("…IN DRIVER'S SEAT") would have
  mislabeled run 64. Former champions are derived from stacker
  runs carrying Kass's recorded crowning approval, in id order.
- Grading path verified for tonight's retrain:
  `champion_holdout_frame(df, hold_sup, new meta)` resolves the
  scored102 frame (792 × 94, columns identical to the meta), and
  the installed champion file re-scores on the study holdout at
  accuracy 0.5265151515151515, precision 0.5167, 418/792 up-calls
  — an exact reproduction of run 69's recorded grades.

## Move 2 — run 69 seated in the SHADOW desk (Book #2)

Kass explicitly overrides the 2026-10-25 "pinned to run 64
permanently" rule. Recorded in `data/shadow_seat.json`
(`eyes_swapped`: from_run 64, to_run 69, by "Kass's decision
2026-10-26") and in `models/shadow_seat/README.md`.

- Pre-swap check: Book #2 FLAT — data/paper_shadow.db trades 0,
  cash $100,000, btc 0. Swap proceeded on files + manifest only;
  the book and state were NOT re-seeded (the seat README's
  re-seed step exists to avoid blending track records; with zero
  fills there was nothing to blend). Documented as the exception
  in the README.
- `models/shadow_seat/stacker.pkl` / `stacker.json` ← run 69's
  artifacts (sha256 `20705159…` / `d0c1cf7c…`). Driver untouched:
  shadow `ppo.zip` still run 44 (`e610ddf3…`).
- Manifest: desk "TradingAgent Live — Shadow (Champion #3 Eyes)",
  eyes run_id 69, name "Muse's Pick Stacker", champion
  "Champion #3"; new `wfe_exam` record attributing the graded
  exam to the pairing it graded (driver 44, eyes 64).
- Manual step after the swap:
  `.venv/bin/python -m src.trading.live_shadow` → seat verified
  against the new manifest, `{"processed": 0, "cash": 100000.0,
  "btc": 0.0}` (no newly closed candle at 02:37 UTC; Book #2
  unchanged, flat $100,000). Book #2 now trades on run 69's eyes
  from the next closed candle.

## WFE honesty

The +$461.81 / +0.4618% exam graded the OLD pairing (PPO 44 +
run 64 eyes, scored-eyes study runs 65/66). The new pairing
(44 + run 69) has NO graded exam. `compact._wfe_block` now reads
the manifest's `wfe_exam` attribution: when the attributed
pairing ≠ the desk's current pairing, exam fields are null and
the old exam rides as `pexam` (previous pairing, labeled
reference only); ratio stays null. Tests-first: the new
attribution behavior was pinned by failing tests before the feed
code changed.

## Feeds, labels, dashboards

- compact.py: champions roster (formers + status 4), reads block
  follows the reigning champion's study record (`full_indicator_
  stacker` or `muse_pick`), shadow desk/book/race labels derived
  from the seat manifest, WFE pair/pexam. Feeds pushed verbatim
  and read back: Training Lab roster shows #3 = run 69 current,
  #2 = run 64 former; gate pending null on both dashboards; gate
  challenger still run 68; lab stacker stats now 0.5265 / 0.5167;
  shadow desk shows the Champion #3 pairing everywhere; Book #2
  $100,000 flat.
- Artifact edits: shadow desk renamed "TradingAgent Live —
  Shadow (Champion #3 Eyes)" with all identity copy updated
  (the only "Champion #2"/"Run 64" strings left are the WFE
  previous-pairing reference and the About pin-history line);
  Live desk switcher strip updated; Training Lab gained the
  status-4 "FORMER LAB CHAMPION" chip + dimming and a
  champion-generic reads panel.
- `src/trading/live_shadow.py` docstring/constants/pin text and
  `build_manifest` updated to run 69 / Champion #3 (with the
  `wfe_exam` attribution so a future re-seed stays honest).

## Protection check (hashes before → after)

Unchanged, verified: live seat stacker `0dd62fc4…`, seat meta
`4cc2147e…`, seat PPO `e610ddf3…`, lab PPO champion `e610ddf3…`,
shadow PPO `e610ddf3…`, `data/live_seat.json` `9aa7c637…`.
Book #1 untouched: 29 trades, newest the 2026-10-21 SELL @
$84,495.30, still SHORT −0.293605 BTC, equity $100,536.97
(2026-10-27T01:00 UTC row — the live loop's own writes).

## Suite record (corrected — an earlier draft of this note
## overstated the count)

- Baseline before the moves: 274 passed.
- Full run after the moves: 277 passed, 1 failed —
  `tests/test_scored_eyes.py::test_champion_columns_are_
  run64s_recorded_94` pinned the lab champion meta to run 64.
  Fixed the same night: the test now pins run 69 and additionally
  asserts the champion's columns equal run 64's recorded 94 read
  from the registry (the pin's substance — the scored-eyes
  challenger's column set — is unchanged; run 69's list is
  identical to run 64's).
- Full re-run after that fix: 277 passed, 1 failed — a different
  test, `tests/test_shadow_desk.py::test_shadow_vote_matches_
  champion_computed_the_study_way`, which asserts exact float
  equality between two assemblies of the same model's
  predict_proba. It is intermittently flaky at the 1-ulp level
  (observed failing once mid-task on run 64's artifacts with
  0.49615091727697863 vs 0.4961509172769785, before the swap;
  passes standalone 3/3 and passed in the first full run). The
  flake predates and is unrelated to this change — both seat
  copies are byte-identical, so only nondeterministic float
  reduction can differ. Relaxing its `==` to a small tolerance
  is a candidate follow-up, deliberately NOT folded into this
  change.

Known cosmetic leftover: the scheduler job is still named
`trading-agent-shadow-loop-champion2` — a label only; its
instructions are seat-driven and unaffected.
