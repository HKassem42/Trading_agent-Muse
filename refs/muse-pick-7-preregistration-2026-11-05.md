# Muse's Pick #7 — "Slow Clock" — Pre-registration (2026-11-05)

Kass's order (2026-11-05 12:14 ET, main chat): "I want you to try
muses pick the next one for training a model. Pick whatever u want
and give it another shot."

Frozen BEFORE the candidate is fitted or graded. Lab only under
the promotion lock: a win is recorded PENDING KASS APPROVAL;
nothing is crowned, seated, or installed by this study.

Status note: the 2026-11-04 picks #4/#5/#6 study build died in a
runtime restart during Pick #4's walk-forward folds — no pick was
graded or registered (registry max stayed the nightly's 105).
Those three designs were all next-1h variations on the same eyes;
this pick supersedes them as the next attempt rather than
re-running them.

## The lever

Horizon × information. The derivatives-positioning columns
(funding, open interest, long/short ratios, taker flow, basis —
plus ETH cross-asset) were tested exactly once, against
next-1-hour direction (Next Try #1/#2, runs 101/102: FAIL, 0.4899
and 0.5101). Positioning is a slow variable — funding settles
every 8h, open interest builds over days — and the program's own
record flags the same signals as genuinely untested at a 24-hour
horizon. Pick #7 fits the champion Pick recipe to next-24h
direction on the full 120-column frame. Everything else about
the frame at 1h is a closed door and stays closed.

## Frame, target, split

- Frame: `data/muse_next_try/frame120.pkl` reused as-is — Pick
  #2's 98 scored eyes + the 22 causal new-information columns.
  Its causality was truncation-tested in the Next Try study
  (max abs diff 0.0 at cuts 9000 and 17018); features at row t
  read only data through candle t's close.
- Target: y24 = 1 if close[t+24] > close[t] else 0, from
  `data/muse_next_try/raw_ohlcv.pkl` close on the frame index
  (labels are labels only; the label construction is pinned by
  tests/test_muse_pick7.py).
- Rows: the Next Try supervised set (finite on the full frame;
  train anchor 12,045 + holdout 792 on the prod_sup index)
  intersected with rows whose y24 is defined.
- Holdout: the standard exam window — prod_sup's last 792 rows
  from 2026-10-02 06:00 UTC. Exam rows = the holdout rows with
  defined y24 (expected 768: the data ends 2026-11-04 05:00 UTC,
  so the final 24 holdout rows have no t+24 close). Every model
  below is graded on exactly those rows; the count is reported.
- Train: supervised rows before the holdout, MINUS a 24-row
  purge at the boundary — any train row whose label reads a
  close at or after the holdout's first position is dropped, so
  no train label touches the exam period.
- The frame index must verify as contiguous hourly; the build
  asserts it (purge/embargo arithmetic is positional).

AMENDMENT (2026-11-05, before any fitting — the build's setup
assert caught this on its first run): the raw series is NOT
contiguous hourly — it carries the program's two documented 6h
yfinance holes (2025-10-25, 2026-05-08), so that assert was
miscalibrated against known data. The purge/embargo arithmetic
is positional (labels are positional shifts — the same
convention the champions' own exams use, holes included), and
the invariant that actually protects the exam is positional:
no train label's endpoint position reaches the first exam
position. The build now asserts THAT, in its strong form: the
supervised grid must be one contiguous slice of the label grid
(verified via index positions), and the holes are counted into
the study metadata. Target, split sizes, purge width, bar —
unchanged.

## Recipe and controls

- Candidate: `build_pick_pipeline` (Pick recipe: scaler → SMOTE
  → RF + XGB + LGBM + ExtraTrees → LR meta, internal cv=3),
  seed 42, fitted on all 120 columns. Threshold fixed at 0.50.
  No frame selection, no threshold tuning, no knob search
  (Next Try #1's tuned-theta degenerate anatomy is the lesson).
- Controls — identical target, rows, split, and recipe:
  C1 = the 21 production features (prod_sup); C2 = the 98 scored
  eyes alone. The bar is set by the controls on THIS exam, not
  by the 1h exams' numbers.
- Diagnostic ablation (reported, never registered, carries no
  verdict): A1 = the 22 new-information columns alone.
- Walk-forward: 3-fold TimeSeriesSplit on the purged train set
  with a 24-row embargo at every fold boundary (last 24
  train-fold rows and first 24 validation rows dropped), so no
  fold's labels straddle its boundary. WF accuracy = mean fold
  accuracy; WF majority = mean fold majority rate.

## The bar

- Clean anatomy (run-63 rule): up-calls ≥ 8 AND precision > 0.05
  on the exam. A higher accuracy with degenerate anatomy is a
  FAIL, recorded as such.
- PENDING KASS: exam accuracy strictly greater than BOTH
  controls AND the exam majority baseline, with clean anatomy,
  AND WF accuracy > WF majority. Recorded pending_approval
  under the promotion lock; champion kept.
- BEAT THE BAR: PENDING conditions met AND exam accuracy >
  0.5354 (all-time clean-era best — a cross-exam reference,
  named as such, not the operative bar).
- Otherwise FAIL.
- Overlap honesty: 24h labels on hourly rows overlap 23/24, so
  the exam's effective independent sample is ≈ 768/24 = 32. A
  non-overlapping diagnostic (every 24th exam row) is computed
  and reported with the verdict; it informs the evidence weight
  and does not carry the verdict either way.
- Holdout grades computed twice: once in the build, once by an
  independent verifier re-reading the saved exam probabilities.
- Champion / live-seat / shadow-seat / gated-seat files are
  sha256-guarded before and after; registration aborts if any
  guard moves.

## Registration

kind=stacker via `training.nightly.log_run`, family `muse_pick7`
(compact.py family code 27, wired in the same pass — the
2026-11-04 lesson). Metrics mirror the picks456 shape, with the
family sub-dict carrying label/design/verdict/beat_bar, both
control accuracies, the exam majority, exam row count,
horizon=24, window accuracies (3 × 256), up-calls, the
non-overlap diagnostic, and the vote histogram via
`nightly.vote_histogram`. The fitted artifact and the exam
probabilities are saved under `data/muse_pick7/` for the
verifier. A lab note records the outcome either way.
