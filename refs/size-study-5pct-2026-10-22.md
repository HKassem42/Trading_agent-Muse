# Tranche-Size Study — 25% vs 5% (2026-10-22)

Kass's request (2026-10-22 ~21:51 ET): *"lower the 25 percent of total
portfolio value to 5 and then re run the numbers."*

The reference signals were re-graded at tranche = **5% of equity**
against tranche = **25%**, everything else frozen: same signals,
same production exit recipe (stop 2×ATR% clipped [2.5%, 8%],
trailing arm +3% / trail 1.5×ATR% clipped [2%, 6%], no take-profit,
T0 through-flat, stop-anchor ratchet active), same 75% exposure cap,
same 0.15% fee per side, same $10 minimum ticket, next-candle-open
fills, same duel windows (standard = last 792h as 3×264h;
robustness = the preceding 792h, Lag F only). Scoring is the gate's:
window score = total P&L − 0.5 × worst episode drawdown; agg = mean
of window scores. Book is the $100k duel book throughout.

**Lab only.** Production `max_position_pct=0.25` is unchanged; the
5% runs went through a broker constructor parameter in a mirror of
the production evaluation loop (`src/models/size_study.py`, the
exit-study pattern). Registered as training.db runs 37–39,
family `size_study`, promoted=0. Park stays.

## The baseline question — read this first

Three of the four recorded 25% duels were graded **before** the
stop-anchor fix landed on the evening of 2026-10-22 (the
position-adding audit: adds used to re-derive the stop from the
blended entry, dragging it away when averaging down). Those records
are pre-fix numbers, and this study is specified to run with the
ratchet active. The harness therefore pins its 25% baselines under
**current production code**, and the tests prove the provenance
both ways (`tests/test_size_study.py`):

- **Lag F, standard windows** — recorded (run 26): agg −1,871.18 /
  P&L −$4,416.98 / 40 trips. Reproduces to the cent under current
  code (and under a pre-fix emulation). Baseline = recorded.
- **Lag F, robustness** — exit-study record: −9,527.32 /
  −$23,167.08. Current code: **−8,587.86 / −$21,288.15**.
- **PPO champion** — focus-build re-grade: −2,062.97 / −$4,484.72.
  Current code: **−2,037.24 / −$4,420.64**.
- **Focus B** — recorded (run 22): −3,476.80 / −$8,432.41.
  Current code: **−3,329.80 / −$8,138.37**.

A pre-fix emulation broker (ratchet neutralized, anchor cleared —
stop re-derived from the blend, first-tranche ATR frozen) driven
through this same harness reproduces **all three** divergent
records to the cent, trips included. The gap is the production
fix, not harness drift. Every 25% → 5% delta below is same-code,
same-harness, same-signal.

Two signal caveats, stated once:

- The GBM rule policies' actions depend on conviction and position
  *sign* only, so their action streams are identical across sizes
  (verified: action mixes match exactly on the standard windows).
  The PPO champion's observation vector contains portfolio state
  (cash fraction, signed position fraction, unrealized %), so its
  action stream legitimately differs at 5% — "frozen" for the PPO
  means same brain, same windows, same stacker votes.
- Exits are not perfectly symmetric to entries (broker code,
  verified): entries are gradual (one tranche per action), a long
  closes in ONE fill (SELL = 100%; a SHORT action while long also
  sells 100% through flat), and every protective exit (stop /
  trailing) closes or covers 100% in one fill on both sides — **but
  a signal-driven cover of a short is tranche-paced**: `buy` while
  short buys back at most one tranche per action. At 5% a
  full-size short unwinds over up to 15 buy actions, vs 3 at 25%.
  This pacing is visible in the robustness results below.

## Results — standard windows

### GBM Lag F (run 26) — standard

| Metric | 25% | 5% |
|---|---:|---:|
| Aggregate score | −1,871.18 | **−408.01** |
| Total P&L | −$4,416.98 | **−$947.02** |
| Buy & hold | +$717.54 | +$717.54 |
| Window scores | −3,265 / −1,583 / −765 | −754 / −317 / −153 |
| Round trips (env) / closed segments | 40 / 41 | 40 / 41 |
| Wins / losses | 7 / 34 | 7 / 34 |
| Avg win / avg loss | $196.81 / −$167.61 | $39.44 / −$35.41 |
| Max episode drawdown | $1,174.34 | $310.29 |
| P&L long / short | −$276.34 / −$4,140.64 | −$54.75 / −$892.27 |
| Fees paid | $4,420.72 | $915.19 |
| Adds per segment: mean / median / max | 0.43 / 0 / 4 | 0.45 / 0 / 5 |
| Adds histogram (0/1/2/3/4/5+) | 29/10/2/0/1/0 | 29/10/2/0/0/1 |
| Add attempts / filled / cap-limited / cap-blocked | 19 / 18 / 3 / 1 | 19 / 19 / 0 / 0 |
| Avg deployed capital (all steps / in-position) | $3,197 / $25,295 | $738 / $5,716 |
| P&L % of avg deployed (all / in-position) | −138.2% / −17.5% | −128.3% / −16.6% |

Same episodes, same wins and losses, same trip count. The loss
shrinks to 21.4% of its 25% size while in-position deployed capital
shrinks to 22.6% — a near-exact scale-down, fractionally better per
deployed dollar (−16.6% vs −17.5%).

### GBM Lag F — robustness windows (the preceding 792h; B&H +$27,401.66)

| Metric | 25% | 5% |
|---|---:|---:|
| Aggregate score | −8,587.86 | **−2,159.57** |
| Total P&L | −$21,288.15 | **−$5,505.77** |
| Window scores | −14,255 / −6,277 / −5,232 | −3,950 / −1,426 / −1,103 |
| Round trips (env) / closed segments | 75 / 82 | 71 / 79 |
| Wins / losses | 15 / 67 | 15 / 64 |
| Avg win / avg loss | $101.29 / −$302.83 | $20.32 / −$81.04 |
| Max episode drawdown | $4,957.21 | $1,114.85 |
| P&L long / short | −$3,081.86 / −$18,206.29 | −$642.81 / −$4,862.96 |
| Fees paid | $10,757.75 | $2,401.21 |
| Adds per segment: mean / median / max | 0.72 / 0 / 5 | 0.93 / 0 / 13 |
| Add attempts / filled / cap-limited / cap-blocked | 78 / 65 / 17 / 13 | 81 / 81 / 0 / 0 |
| Avg deployed capital (all steps / in-position) | $10,440 / $31,715 | $3,182 / $9,274 |
| P&L % of avg deployed (all / in-position) | −203.9% / −67.1% | −173.0% / −59.4% |

This is the one cell where the **path itself changes**: 75 → 71
round trips, 82 → 79 closed segments, action mix shifts slightly
(buy 21.7% → 22.1%). Cause: tranche-paced short covers (up to 15
steps to unwind at 5%) plus smaller per-step exposure while a
position is being built or unwound. The result is better than pure
scaling — P&L falls to 25.9% while in-position deployed falls only
to 29.3%, and the per-deployed-dollar loss improves ~8 points
(−67.1% → −59.4%). The catastrophic first window (score −14,255 at
25%) is where most of the rescue happens (−3,950 at 5%).

### PPO champion (run 17's brain) — standard

| Metric | 25% | 5% |
|---|---:|---:|
| Aggregate score | −2,037.24 | **−603.42** |
| Total P&L | −$4,420.64 | **−$1,204.59** |
| Buy & hold | +$717.54 | +$717.54 |
| Window scores | −2,762 / −1,720 / −1,630 | −689 / −709 / −412 |
| Round trips (env) / closed segments | 0 / 1 | 0 / 1 |
| Segments left open at episode end | 17 | 17 |
| Wins / losses (closed segments) | 0 / 1 (−$935.15) | 0 / 1 (−$254.99) |
| Max episode drawdown | $1,493.47 | $688.80 |
| P&L long / short | $0 / −$4,420.64 | $0 / −$1,204.59 |
| Fees paid | $1,536.00 | $495.12 |
| Adds per segment: mean / median / max | 1.33 / 1 / 4 | 2.56 / 2 / 9 |
| Add attempts / filled / cap-limited / cap-blocked | 40 / 24 / 10 / 16 | 46 / 46 / 0 / 0 |
| Avg deployed capital (all steps / in-position) | $20,215 / $50,537 | $5,483 / $13,708 |
| P&L % of avg deployed (all / in-position) | −21.9% / −8.7% | −22.0% / −8.8% |
| Action mix (hold/buy/sell/short) | .450 / .000 / .463 / .087 | .508 / .000 / .383 / .108 |

The champion is a perma-short on these windows (buy share 0.0% at
both sizes; every dollar of P&L is short-side; only 1 of 18
segments ever closes inside its episode — the rest ride to the
episode end). Its per-deployed-dollar loss is **identical** at both
sizes (−21.9% vs −22.0% of average deployed): for the champion, 5%
is a pure scale-down, nothing more. Two behavioral notes: at 25%
the champion repeatedly tried to add into a full cap (16 of 40 add
attempts refused outright, 10 more shrunk) — it wants more than
75% exposure; at 5% it adds more often (46 attempts, all filled,
up to 9 adds in one segment) yet never reaches the cap.

### GBM Focus B (run 22) — standard

| Metric | 25% | 5% |
|---|---:|---:|
| Aggregate score | −3,329.80 | **−932.39** |
| Total P&L | −$8,138.37 | **−$2,185.06** |
| Buy & hold | +$717.54 | +$717.54 |
| Window scores | −4,375 / −3,541 / −2,074 | −1,591 / −790 / −416 |
| Round trips (env) / closed segments | 66 / 68 | 66 / 68 |
| Wins / losses | 17 / 51 | 17 / 51 |
| Avg win / avg loss | $144.09 / −$189.88 | $27.12 / −$48.33 |
| Max episode drawdown | $1,797.08 | $788.12 |
| P&L long / short | −$2,501.02 / −$5,637.35 | −$555.00 / −$1,630.06 |
| Fees paid | $9,017.70 | $2,000.90 |
| Adds per segment: mean / median / max | 0.73 / 0 / 5 | 0.86 / 0 / 9 |
| Add attempts / filled / cap-limited / cap-blocked | 63 / 53 / 10 / 10 | 63 / 63 / 0 / 0 |
| Avg deployed capital (all steps / in-position) | $8,770 / $33,059 | $2,186 / $8,114 |
| P&L % of avg deployed (all / in-position) | −92.8% / −24.6% | −99.9% / −26.9% |

Identical episodes, wins, losses and trips — but per deployed
dollar Focus B is slightly **worse** at 5% (−26.9% vs −24.6%
in-position; −99.9% vs −92.8% all-steps). Its entries arrive over
more steps (max 9 adds in a segment vs 5), so the average fill is
later; on this fast signal that costs a little. The worst-episode
drawdown also falls less than proportionally (×0.44 vs P&L ×0.27).

## Comparison — 25% → 5% (standard windows; Lag F robustness in italics row)

| Signal | Δ agg | Δ P&L | P&L ratio 5%/25% | Δ fees | P&L % deployed (in-pos) 25% → 5% |
|---|---:|---:|---:|---:|---:|
| Lag F | +1,463.17 | +$3,469.96 | 0.214 | −$3,505.53 | −17.5% → −16.6% |
| *Lag F (robustness)* | *+6,428.29* | *+$15,782.38* | *0.259* | *−$8,356.54* | *−67.1% → −59.4%* |
| PPO champion | +1,433.82 | +$3,216.05 | 0.272 | −$1,040.88 | −8.7% → −8.8% |
| Focus B | +2,397.41 | +$5,953.31 | 0.268 | −$7,016.80 | −24.6% → −26.9% |

Cap behavior, all cells: at 25% the 75% cap limited 40 add attempts
and refused 40 more across the four cells. At 5% the cap **never
bound once** — no limited fills, no refusals, in any cell (the
deepest build was 14 tranches = 70% in one Lag F robustness
segment). At 5% the cap is effectively decorative; the tranche
itself is the exposure control.

## Analysis — scaling, or a different shape?

Mostly scaling, with two genuine shape changes:

1. **PPO champion: pure scaling.** P&L % of deployed capital is
   unchanged to 0.1pt. The brain deploys 3.7× less capital and
   loses 3.7× fewer dollars. Its loss rate per dollar at risk is a
   property of the brain, not the tranche size.
2. **Lag F standard + Focus B: same trades, rescaled tickets.**
   Identical action streams, identical win/loss counts; avg wins
   and losses shrink roughly in proportion to deployed capital.
   Lag F lands fractionally better per deployed dollar, Focus B
   fractionally worse (slower average entry on a fast signal).
   Neither change is large; neither flips a sign.
3. **Lag F robustness: real shape change.** The trade path itself
   differs (trips 75 → 71) because short covers unwind over up to
   15 steps at 5%, and the bleed concentrates in the oldest window
   where smaller, slower positions cut the worst episode drawdown
   from $4,957 to $1,115. Per deployed dollar the loss improves
   ~8 points. This is the only cell where gradualism demonstrably
   dodged damage rather than just resizing it.
4. **The fee story.** Fee drag per dollar *deployed* is identical
   by construction (0.15% of whatever fills). But in absolute terms
   the fee bill at 25% is startling: on the standard windows it
   **exceeds the entire net loss** for both GBM signals (Lag F:
   $4,421 of fees vs −$4,417 net — gross of fees the signal was
   ≈ flat, +$4; Focus B: $9,018 of fees vs −$8,138 net — gross
   ≈ +$879). At 5% the same accounting gives gross ≈ −$32 (Lag F)
   and ≈ −$184 (Focus B) — the GBM signals' standard-window losses
   are, to first order, a fee bill at either size (gross ≈ net +
   fees; without fees, buy quantities would be marginally larger).
   The PPO champion's loss is not a fee story (gross ≈ −$2,885 at
   25%, ≈ −$709 at 5%).
5. **Nothing flips.** All eight cells lose money; buy-and-hold
   beats both sizes on every window set (standard +$717.54;
   robustness +$27,401.66 — the robustness windows contain a rally
   the short-heavy signals fought at both sizes: short-side P&L is
   −$18,206 of Lag F's −$21,288 at 25%, −$4,863 of −$5,506 at 5%).

## Bottom line — what adopting 5% would mean (Kass's decision)

- **Risk per idea:** the first tranche on the $100k book drops
  from $25,000 to $5,000, and a signal must persist for 15 hourly
  actions (vs 3) to reach the full 75% position. Most segments
  never get there — median adds per segment is 0 in every
  rule-signal cell at both sizes — so the *typical* position at 5%
  is 1–2 tranches: 5–10% of equity, with worst-episode drawdowns
  2.3–4.4× smaller in dollars.
- **Scaling behavior:** losses shrink roughly in proportion to
  deployed capital (P&L ratios 0.21–0.27 against deployed-capital
  ratios 0.23–0.29). Per dollar actually at work, outcomes are
  the same (PPO), a touch better (Lag F), or a touch worse
  (Focus B). 5% sizing resizes these signals; it does not repair
  them — the loss *rate* is the signal's, not the tranche's.
- **Fee reality:** the absolute fee bill falls 3.1–4.8× (least for
  the PPO, whose fill count nearly doubles at 5% — 46 adds filled
  vs 24). Fee drag per deployed dollar does not change. Any future
  signal whose gross edge is thinner than its fee bill loses at
  both sizes; on the standard windows that is already the GBM
  signals' situation.
- **Mechanical side effects to expect if adopted:** the 75% cap
  stops binding (exposure is governed by how long a signal
  persists, not by the cap); short positions unwind slowly on
  signal covers (up to 15 steps for a full-size short), which
  changed real trade paths in the robustness windows; and the
  champion's demonstrated appetite for adding into a full cap
  disappears because it can no longer reach the cap inside an
  episode.

Adoption is not recommended or opposed here — the numbers above
are the decision input.

## Integrity

- Production untouched: `paper.db`, `live_state.json`, champion
  files, `broker.py`, `trading_env.py`, `nightly.py` mtimes
  verified unchanged by the build's integrity phase. The only
  production-adjacent edit is the dashboards feed
  (`compact.py`): family code 6 + display names + verdict branch
  for `size_study` — display plumbing, no trading code. Dashboards
  were not pushed (the hourly job syncs).
- Tests: suite 120 → **131 passed** (11 new: tranche dollar math,
  cap binding at 15 tranches long + short, stop-anchor ratchet in
  the study broker, four 25% identity pins, one pre-fix
  reproduction pinning all three pre-fix records to the cent).
- Registrations: runs 37 (Lag F), 38 (PPO Champion), 39 (Focus B),
  family `size_study`, promoted=0, verdict "SIZE STUDY", reasons
  state "5% tranche study — NOT a production change". One lab note.
- Feed: training feed 6,971B at the 39-run registry (was 6,679B at
  36); the size-regression sentinel was rebaselined 6,800 → 7,100B
  per its documented pattern, far under the 16,384B budget.
- Raw cell data: `data/size_study/results.json`;
  harness: `src/models/size_study.py`; build:
  `scripts/build_size_study.py`.
