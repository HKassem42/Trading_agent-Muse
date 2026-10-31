# Two-Step Hierarchy — results (2026-10-24)

Executed **exactly** as pre-registered in
`refs/hierarchy-plan-2026-10-24.md` (Drawing-Board Option C,
Kass's order 2026-10-24: A → B → C). Machinery:
`src/models/hierarchy.py`, driver `scripts/build_hierarchy.py`,
tests `tests/test_hierarchy.py` (7 tests, written before the
build), raw records in `data/hierarchy/results.json`.
Registrations: training.db runs **55–56** (family=`hierarchy`,
both promoted=0) + one lab note.

**Verdict up front: NULL — the ninth independent null on 1h BTC,
and the last of Kass's A→B→C sequence.** Both arms fail all three
pre-registered verdict rules. Per the plan, the design is not
re-tuned for a second shot: no different gate threshold, no
different stage split, no third arm.

## Which champion graded the duel

`models/ppo_champion.zip` as it stands now: **run 44's PPO brain,
installed 2026-10-24 by Kass's manual override** (sha256
`e610ddf3…`; the recorded run-44 gate verdict remains NOT
PROMOTED). Graded fresh through the same harness as the arms:

| Champion (run 44 brain) | Standard windows | Robustness windows |
|---|---:|---:|
| Aggregate score | −662.43 | −4,805.59 |
| P&L net / gross / fees | −$994.49 / −$318.83 / $675.65 | −$10,750.15 / — / — |
| Buy-and-hold, same windows | +$717.54 | +$27,401.66 |
| Round trips / non-hold share | 4 / 45.1% | — |

Harness proof: the duel mirror
(`composite_screen.evaluate_windows_composite`) reproduces
`training.nightly.evaluate_windows` on this champion **exactly**
(aggregate diff 0.0000, P&L diff 0.0000) before any arm number was
trusted — the same figure the composite screen's fresh grade
produced (−662.43 / −$994.49).

## The design, in one paragraph

Each arm fits two LightGBM binary stages (gbm_focus hyperparameters,
class weights = inverse frequency within each stage's own training
rows). **Stage 1 (gate):** TRADE vs NO-TRADE, where TRADE = the
matrix label is BUY or SHORT (|next-bar return| > 0.30%). **Stage
2 (direction):** UP vs DOWN, trained only on true TRADE rows.
Combined by the fixed rule: gate fires at P(TRADE) ≥ 0.50,
direction by stage-2 argmax (tie → UP). Arms: **Hierarchy-Focus**
(focus-11) and **Hierarchy-All** (the all-60 feature set).

## Walk-forward — end-to-end (TimeSeriesSplit 5 folds, 15,961 pre-duel rows; 13,300 graded rows)

Accuracy next to the 0.6690 majority baseline (recomputed on the
graded rows), balanced accuracy next to 1/3 chance:

| Arm | Accuracy | vs 0.6690 | Balanced acc | Fold BA spread | Predicted mix B/H/S |
|---|---:|---:|---:|---|---|
| Hierarchy-Focus | 0.5558 | −0.1132 | 0.4133 | 0.380 – 0.433 | 20.0 / 62.1 / 17.9 % |
| Hierarchy-All | 0.5531 | −0.1159 | 0.4159 | 0.393 – 0.428 | 18.2 / 60.9 / 21.0 % |

The balanced-accuracy half of rule 1 passes comfortably for both
arms (pooled > 1/3, all 10 folds > 1/3) — and the accuracy half
fails for both, by ~11 points. Like every weighted flat model
before them, the hierarchies call far more trades than exist
(activity 38–39% vs a 33.1% true TRADE rate, on top of a 66.9%
HOLD majority), and the extra calls are wrong often enough that
raw accuracy lands well below a machine that always says HOLD.

## Stage diagnostics — where the hierarchy loses

Pooled over the 13,300 graded rows (per-fold in results.json):

| Diagnostic | Hierarchy-Focus | Hierarchy-All | Reading |
|---|---:|---:|---|
| Gate balanced accuracy | 0.6130 | 0.6204 | real, consistent (folds 0.56–0.65) |
| True TRADE base rate | 0.331 | 0.331 | — |
| Gate fire rate | 0.3787 | 0.3914 | fires a bit hot |
| **Stage-2 accuracy on true-TRADE rows** | **0.5014** | **0.4952** | **a coin flip** |
| Stage-2 accuracy on fired true-TRADE rows | 0.5135 | 0.5062 | still a coin flip |
| End-to-end activity share | 0.3787 | 0.3914 | — |

**The hierarchy loses at stage 2, not at the gate.** Kass's
premise is half-vindicated by measurement: tradeability *is* the
learnable part — the gate separates tradeable from untradeable
hours with balanced accuracy ~0.61–0.62 in every fold of both
arms, the strongest single-stage separation any of the nine
studies has produced. But conditional on a row being truly
tradeable, direction is 0.495–0.501 pooled, and **no fold of
either arm escapes 0.47–0.53**. Restricting stage 2 to TRADE rows
did not give it a cleaner problem: on those rows, up-vs-down over
the next hour is simply not in these features. The gate's modest
skill is then spent on trades whose direction is chosen by
flipping a coin — which is exactly what the duel shows.

## Duel — money metrics (standard 3×264h windows)

Final fits on all 15,961 pre-duel rows (stage-2 fits consumed
5,567 TRADE rows: 2,822 UP / 2,745 DOWN). Next-candle-open fills,
25% tranches, 75% cap, 0.15% fees, the verified exit recipe.

| Arm | Aggregate | P&L net | P&L gross | Fees | Trips | W/L | Non-hold | Won vs champ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hierarchy-Focus | −1,429.72 | **−$3,054.80** | **+$3,986.30** | $7,041.10 | 63 | 16/47 | 29.2% | 0/3 |
| Hierarchy-All | −2,320.70 | **−$5,644.46** | **+$1,137.51** | $6,781.97 | 54 | 13/41 | 26.9% | 0/3 |
| Champion (run 44) | −662.43 | −$994.49 | −$318.83 | $675.65 | 4 | — | 45.1% | — |

Buy-and-hold over the same windows: **+$717.54** — it beat both
arms and the champion. Windows, Hierarchy-Focus (score / P&L /
B&H): −3,553.96 / −$2,956.08 / +$3,202.97 · +55.13 / +$313.78 /
−$1,148.69 · −790.33 / −$412.50 / −$1,336.74. Hierarchy-All:
−2,730.32 / −$2,132.44 · −1,072.21 / −$801.63 · −3,159.57 /
−$2,710.40 (same B&H per window). Long/short split: Focus
−$790 / −$2,265; All −$590 / −$5,054 — shorts did most of the
damage in both.

Two honest credits before the verdict lands. First,
Hierarchy-Focus's **gross +$3,986 is the largest gross P&L of any
GBM signal arm on the standard windows** (Focus+Regime +$2,766,
Focus B ≈ +$880 current-code, GBM-4 composites +$585, Lag F
≈ +$4): the gate's selectivity does add gross value over the
flat models. Second, its net loss (−$3,055) is the smallest of
any fully-active GBM arm on these windows. Neither credit
survives contact with the fee bill — $7,041 of fees on 63 round
trips is 1.8× the entire gross — and neither is the bar. The bar
was absolute, and it is below.

## Robustness (preceding 792h, refit out-of-sample)

| Arm | Aggregate | P&L net | P&L gross | Fees | Trips | Non-hold |
|---|---:|---:|---:|---:|---:|---:|
| Hierarchy-Focus | −7,724.89 | −$18,969.20 | −$9,266.88 | $9,702.31 | 67 | 39.2% |
| Hierarchy-All | −7,105.33 | −$17,208.41 | −$7,696.71 | $9,511.70 | 50 | 43.2% |

Buy-and-hold on the robustness windows: **+$27,401.66**. Here
the arms are negative **before** fees: the coin-flip direction
stage shorted into a strong rally (short-side P&L −$16,202
Focus / −$13,065 All). The standard-window gross credit does not
generalize one window-set back.

## Verdict, rule by rule (as pre-registered)

| Rule | Hierarchy-Focus | Hierarchy-All |
|---|---|---|
| 1. WF: BA > 1/3 across folds **and** acc ≥ 0.6690 | **FAIL** — BA ✓ (0.4133, 5/5 folds), acc 0.5558 ✗ | **FAIL** — BA ✓ (0.4159, 5/5 folds), acc 0.5531 ✗ |
| 2. Duel: P&L > $0 **and** > B&H **and** activity ≥ 10% | **FAIL** — activity ✓ (29.2%), P&L −$3,055 ✗, B&H +$718 ✗ | **FAIL** — activity ✓ (26.9%), P&L −$5,644 ✗, B&H ✗ |
| 3. Robustness P&L ≥ $0 | **FAIL** — −$18,969 | **FAIL** — −$17,208 |
| **Verdict** | **FAIL (NULL)** | **FAIL (NULL)** |

## Hierarchy vs flat, on identical features

| | WF acc | WF balanced | Duel P&L (net) | Duel gross |
|---|---:|---:|---:|---:|
| Hierarchy-Focus | 0.5558 | 0.4133 | −$3,054.80 | +$3,986.30 |
| Focus B (flat, run 22) | 0.5508 | 0.4145 | −$8,138.37 ¹ | ≈ +$880 ¹ |
| Hierarchy-All | 0.5531 | 0.4159 | −$5,644.46 | +$1,137.51 |
| Focus D (flat, run 24) | 0.5668 | 0.4163 | −$3,178.41 ² | — |

¹ Focus B as re-graded under current production code by the size
study (its recorded-at-the-time duel was −$8,432.41, graded
before the stop-anchor fix; gross = net + its recorded fees).
² Focus D as recorded in run 24; no current-code re-grade exists.

Reading: on walk-forward the split is a **wash** — end-to-end
balanced accuracy 0.413/0.416 vs the flat models' 0.4145/0.4163,
accuracy within ±0.014 either way. The two-stage wiring neither
gains nor loses ranking power; it relocates the same information.
On money, the focus hierarchy clearly improves on its flat twin
(net −$3.1k vs −$8.1k, gross +$4.0k vs +$0.9k, same ~60-trip
activity) — the gate's selectivity is worth real gross dollars.
The all-features hierarchy moves the other way (net −$5.6k vs
flat D's −$3.2k): with 60 features the gate fires hotter
(39–43% activity) and the extra direction coin-flips cost more
than the selectivity saves. Neither version makes money net,
beats buy-and-hold, or survives the robustness windows — so
neither is a challenger, and the comparison itself is the
finding: **the bottleneck was never the decision structure. It
is the direction information, and it is not there.**

## Integrity

- Plan executed as written; **deviations: none** (the only
  judgment calls were made in the plan itself, before results:
  the binary-objective adaptation of the shared hyperparameters,
  and the +1/−1/0 conviction encoding borrowed from the
  composite screen's rule arm).
- Fit audits: final stage-1 fits consumed exactly 15,961 rows
  (gate classes 10,394 NO-TRADE / 5,567 TRADE); stage-2 fits
  consumed only those rows' TRADE subset (2,822 UP / 2,745
  DOWN). Robustness fits: 15,169 rows strictly before the
  region. Poison tests in `tests/test_hierarchy.py` pin the
  consumption boundary.
- Fee identity holds per duel (gross = net + fees, to the cent,
  in every cell above).
- Trading-feed size sentinel: the 2 hierarchy index rows took
  the feed to 7,716B at 56 runs, tripping the 7,400B diet
  sentinel; re-baselined to ≤ 7,850B per that test's own
  documented pattern (measured value + headroom, dated comment
  in the test), disclosed here. The models index renders both
  runs correctly: "Two-Step Hierarchy — Focus 11" / "Two-Step
  Hierarchy — All Features", family code 10, verdicts read from
  the runs' own records — verified in the built feed. No
  dashboard push was made (the 15-min job syncs).
- Champion files untouched (mtimes verified identical at the
  end); paper.db / live_state.json never opened for writing —
  all duel brokers used scratch DBs in /tmp; the live loop was
  never run by this study.
- **Full suite: 172 passed** (165 prior + 7 new hierarchy
  tests), including the re-baselined sentinel.

## What this closes

The A→B→C drawing-board sequence is complete: 4h timeframe
(NULL), daily regime data (NULL), the two-step hierarchy (NULL)
— nine independent nulls on this data in total. The hierarchy
was the strongest of the three drawing-board ideas and it
produced the best diagnostics of the nine: it isolated *the*
missing piece with unusual clarity. Tradeability is partially
predictable (gate BA ≈ 0.61–0.62). Next-hour direction, on
tradeable rows, is not predictable from these features at all
(0.495–0.501). Any future direction must supply direction
information this feature set does not contain, or stop asking
for next-hour direction. That is a drawing-board decision for
Kass; nothing here promotes, and the live bot (run-44 champion)
is untouched.
