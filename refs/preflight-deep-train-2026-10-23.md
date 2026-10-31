# Pre-flight — first weekly deep train (Sun 2026-10-25 01:30 ET)

Prepared 2026-10-23 (Fri) ~12:40 ET, ahead of the first-ever run of
`trading-agent-weekly-deep-train`. Read-only checks; nothing was changed.

## Finding 1 — Test suite is NOT green: 2 failures, root cause proven

Full suite run 2026-10-23: **129 passed, 2 failed** (67.7s).

- `tests/test_size_study.py::test_ppo_25_identity_standard`
  obtained agg **-2,117.7096** vs pinned **-2,037.2361**
- `tests/test_size_study.py::test_prefix_broker_reproduces_the_recorded_grades`
  obtained agg **-2,139.0697** vs pinned **-2,062.9741** (PPO arm only;
  the Lag F arm of the same test passes)

**Cause (verified two independent ways, per the house standard):**
`src/models/size_study.py::load_ppo_signal()` builds its stacker votes
from the LIVE `models/stacker_champion.pkl`. The Oct 23 midnight retrain
(run 40, 04:03 UTC) promoted a new stacker champion (holdout 0.4798 >
0.4609), replacing that file — so the "immutable" provenance pins now
grade the PPO champion with a different stacker's votes.

Reproduction check, same windows / same PPO brain / same harness:

| Stacker votes source | agg score | total P&L | bench |
|---|---|---|---|
| Archived pre-promotion champion (`models/archive/20261023T040320_20261023/stacker_champion.pkl`) | **-2,037.2361** | **-$4,420.6411** | $717.5366 |
| Current champion (`models/stacker_champion.pkl`) | -2,117.7096 | -$4,525.1429 | $717.5366 |

The archived stacker reproduces the pinned values **to the cent**; the
current one reproduces the failing values exactly. Window data and the
broker are unchanged (Lag F identity tests on the same windows pass;
bench $717.5366 matches everywhere). This is not a code regression and
not harness drift — it is a test-design flaw: every other signal loader
in the same file uses frozen inputs (Focus B reads cached
`data/gbm_focus/probs_B.npz`, "bit-identical to its recorded grading");
only the PPO loader reads a mutable live artifact.

**Prepared fix (needs Kass's go — no action taken):** freeze the PPO
signal's stacker the same way Focus B is frozen — copy the archived
Oct 22 champion that reproduces the pins into a study fixture location
and point `load_ppo_signal()` at it, with the docstring updated to say
the votes are frozen. Do NOT re-pin the expected numbers to the new
champion: that would silently destroy the provenance check's purpose
(the pins exist to prove the recorded pre-fix duels reproduce).
Expected result after the fix: suite back to 131/131, and future stacker
promotions can no longer break these tests.

Side note: every future midnight stacker promotion will re-break these
two tests until this is fixed — the failures will recur, not heal.

## Finding 2 — Sunday's 300k deep train uses the one pattern already known to die on this box

- The deep train runs `TAM_DEEP=1 .venv/bin/python -m src.training.nightly`,
  which calls `train_ppo(timesteps=300_000, warm_start=True, ...)`.
- `src/models/ppo_train.py::train()` issues a **single uninterrupted**
  `model.learn(total_timesteps=300_000)` — no chunking, no checkpoint
  callback; the challenger zip is saved only at the very end. If the
  process dies at step 299k, everything is lost.
- `refs/diagnosis-2026-10-21.md` (Experiment 3 setup) records that this
  exact shape already failed here: "the runtime SIGKILLed the battery
  process three times mid-arm… one uninterrupted 300k learn was not
  reliably runnable on this box." The diagnosis battery only completed
  its 300k arm by splitting it into **12 checkpointed 25k chunks**
  (`PPO.load` + `learn(reset_num_timesteps=False)`).
- Timing headroom is also thin: on Oct 23 the nightly 60k PPO phase ran
  ~04:03→04:26 UTC (~23 min including cross-fit votes + 3-window
  evaluation of both brains). A 5× training load plausibly lands the
  deep run at ~90–115 min against the cron's **7,200s (120 min)
  timeout** — before any SIGKILL risk.
- Secondary gaps in the deep-train cron body vs the midnight body:
  (a) it pushes only the training snapshot — no `gate` feed push to
  either dashboard, so if the deep challenger were promoted, the Live
  Desk's gate/parked banner would be stale until the next midnight run;
  (b) its text describes the old relative gate only. The code itself is
  fine on the gate: `gate_decision()` (the two-part floor) is called in
  `nightly.py::main()` for every run, deep included — verified by
  reading the code 2026-10-23.

**Prepared fix (needs Kass's go — no action taken):** make the deep
path chunked + checkpointed exactly like the diagnosis battery that
succeeded (e.g. 12 × 25k `learn(reset_num_timesteps=False)` segments,
saving the challenger after each segment so a kill loses ≤25k steps),
and add the gate-feed push to the deep-train cron body to match the
midnight body. Alternatively, accept the risk knowingly: worst case
Sunday reports a failed deep train and the park/nightly loop are
unaffected (the live loop is parked and independent; a dead deep train
cannot trade or promote anything).

## What is NOT at risk either way

- The bot stays parked at $99,232.84 / 0 BTC / 28 trades regardless —
  a failed or killed deep train promotes nothing.
- The midnight retrain continues on its own schedule.
- The two failing tests affect only the size-study provenance checks;
  they do not feed tonight's retrain or any promotion decision.
