# Muse's Pick #13 — HA + Bollinger Position + Relative Volume — Pre-registration

Written 2026-11-12 (~22:45 ET), BEFORE any candidate was fitted.
Commissioned by Kass on 2026-11-12 after asking why Bollinger
position and Relative Volume are display-only on the dashboards:
they were never model inputs in any frame. This study settles
whether they add anything as features. Order of operations is
Kass's: Book #5's time-display fix first (done and verified),
then this study. LAB ONLY — win or lose, no seat, book, or
nightly line changes on the strength of this document alone;
a PASS is ruled on by Muse under Kass's 2026-11-12 standing
crowning delegation, applying the standard below.

## The candidate

The exact Pick recipe (`build_pick_pipeline`, seed 42: scaler →
SMOTE → RF + XGB + LGBM + ExtraTrees → LR meta) on **12 columns**:
the 10 HA candle values of Picks #10/#11 (`build_ha_matrix`,
unchanged) plus exactly two new columns:

- **ha_boll_pos** — Bollinger position of the HA close.
  sma20/std20 = rolling-20 mean/std of the HA close;
  position p = (hc − (sma20 − 2·std20)) / (4·std20)
  (0 = lower band, 0.5 = the mean, 1 = upper band);
  feature = (2p − 1) clipped to [−1, 1] — identically
  (hc − sma20) / (2·std20) before clipping, so ±1 is a ±2σ
  excursion. Same definition as the Book #5 indicators feed's
  Bollinger panel, squeezed to the feature contract.
- **rel_vol** — relative volume on REAL volume:
  r = volume / (rolling-24 mean of volume), the same window as
  the dashboards' Relative Volume panel;
  feature = (r − 1) clipped to [−1, 1] (0 at average volume,
  +1 at ≥ 2× average, → −1 as volume → 0).

Both columns are trailing-window constructions: row t depends
only on raw candles ≤ t. Causality is pinned by truncation tests
(tests/test_muse_pick13.py) before any fit is trusted, and each
column is spot-checked against a hand computation by a second
method.

Feature set tag: `ha12`. Family: `muse_pick13`.

## The exam (identical to Picks #9–#12)

Saved frame `data/muse_pick9/raw_ohlcv.pkl` (17,510 rows) +
`prod_sup.pkl` (17,050 supervised rows); split at 16,258; the
exam is the last 792 supervised rows from 2026-10-03 19:00 UTC;
majority baseline 0.5088. Setup gates, run before any candidate
fit: the rebuilt ha10 block must equal Pick #10's saved matrix
exactly, and Pick #11's artifact must reproduce 0.5341 on the
exam — otherwise the study aborts.

## Champion re-scores on this exam

The sitting Lab Stacker Champion (run 126, Champion #13) and
the former champion (run 123) are re-scored on the identical
exam rows from their installed/archived artifacts. Caveat,
stated in advance and carried into the report: both were
trained on data covering this exam window, so their re-scores
are flattered; the candidate never saw the window. The bar
below is nevertheless "beat the re-scored champion," the same
frozen standard as Pick #12 — it is not moved after the fact.

## The bar (frozen)

PASS requires ALL of:
1. Exam accuracy > run 126 re-scored on this exam.
2. Exam accuracy > Pick #11's 0.5341 and > the exam majority.
3. Clean anatomy (the Pick framework's `is_clean`: no
   degenerate one-sided calling).
4. Purged walk-forward agreement: 5-fold purged WF mean
   accuracy (24-row embargo) above its majority baseline.

Anything else is a FAIL. On FAIL: nothing changes — the HA
nightly line (Champion #13's recipe) keeps retraining exactly
as now, no artifact is installed, and Bollinger position +
Relative Volume stay display-only with a tested basis.

On PASS: the run is registered as WOULD PROMOTE and Muse
rules on the LAB crown under the standing delegation (crown
only a legitimate fresh-exam win with sane precision/WF
anatomy). A crowning, if ruled, follows the standard crowning
pass — including the Book #5 seat re-pin (Era 2) under Kass's
follows-the-crown order for that book. Books #1–#4 seats are
never touched by this study either way.

## Guards

SHA-256 of every champion/seat artifact (lab, live seat,
shadow, gated, HA, champ seats) is recorded before the study
and asserted unchanged after registration. Registration writes
exactly one registry row (family `muse_pick13`) and one lab
note.
