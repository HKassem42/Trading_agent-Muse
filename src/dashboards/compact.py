"""Compact snapshot builders for dashboard pushes.

The artifact push actions reject large payloads (observed ceiling roughly
1.5-2.5KB depending on the dashboard), so the sync pipeline sends several
small, machine-generated feeds instead of one big snapshot. NEVER hand-type
these payloads -- always generate them here and pass the JSON verbatim.

Feeds (python -m src.dashboards.compact <feed>):
  trading        -> push_snapshot on tradingagent-trading-dashboard.
                    Core stats, position, trade stats, calendar, activity,
                    equity curve, plus dd_now/dd_max/dd_max_t: current and
                    worst drawdown in % (worst with its timestamp) as data,
                    not only as a frontend-derived strip. Activity also
                    carries candles_total/candles_today (hours the bot
                    evaluated) and streak (signed run of same-sign closed
                    trades, newest first). Position stop fields (2026-10-24):
                    stop_price is the broker's EFFECTIVE stop --
                    PaperBroker.stop_level() on the restored live state
                    (the locked anchor until the trailing arm activates,
                    then the tighter trailing level) -- with stop_anchor
                    (the locked initial stop) and stop_trailing (0/1)
                    carried alongside so no information is lost.
  ledger         -> push_ledger on tradingagent-trading-dashboard.
                    {"ledger": [[ts, side, price, qty, fee], ...] newest
                    first (last 24 fills), "closed": [[close_ts, qty, pnl],
                    ...] last 12 FIFO round trips}
  market0        -> push_market on tradingagent-trading-dashboard.
                    First (older) 48 of the last 96 closed 1h candles:
                    {"start": epoch_s, "step": 3600, "candles": [[o,h,l,c],
                    ...]} with integer prices.
  market1        -> push_market, newest 48 candles, plus
                    {"markers": [[epoch_s, "B"|"S", price, qty], ...],
                     "entry_price": ..., "stop_price": ...}
  vision         -> push_vision on tradingagent-trading-dashboard
                    (added 2026-10-20, features v3 / 1h era).
                    {"ts": iso, "p_up": stacker P(up) 0..1,
                     "f": [[key, label, value, raw], ...]} -- the latest
                    closed candle's feature values (21 since features
                    v5; v3 contract: all in [-1, 1]) with a plain-English
                    label per feature, so the dashboard can show what
                    the bot "sees". Keys are the FEATURE_COLUMNS names.
                    raw (2026-10-26) is the same signal in plain human
                    units -- RSI as its 0-100 level, distances and
                    ranges as %, the volume-flow shares as % -- see
                    _raw_frame (recomputed from the same candle frame,
                    never inverted from the clipped squeezed value),
                    or null when no honest raw exists for that row.
                    The dashboard's grouped heat grid renders raw in
                    the cell and heats by value. The shadow desk's
                    f21 rows carry the same 4th element.
  training       -> push_snapshot on tradingagent-training-dashboard.
                    Short-key run objects (legend below), stacker champion,
                    reward_series (real PPO mean-reward per run, ascending),
                    plus models: a slim index of EVERY registered run
                    (legend below). NOTE: no live-trading data here -- the
                    old live-derived reward_history was removed as
                    duplication (2026-10-20).
  gate           -> push_gate on BOTH dashboards (added 2026-10-21).
                    Feed JSON at TOP level (same pattern as push_ledger);
                    stored as the dashboard's "gate" section. The
                    promotion-gate state the parked era revolves around:
                    parked flag + frozen park snapshot, the reigning PPO
                    champion, the latest challenger duel, and the stacker
                    champion's accuracy. Keys: parked (0/1), park
                    {eq, cash, btc, trades, since} from data/park_status.json
                    (null when not parked), champ {i=run id, t=ts,
                    s=aggregate score, win=[3 window scores], pl=holdout
                    P&L, dd=max drawdown} or null, chall (same shape, the
                    challenger side of the latest duel) or null, duel
                    {won=windows the challenger won, delta=challenger minus
                    champion aggregate on the SAME windows, cs/cwin/cpl=the
                    champion side of that duel, pass (0/1) = the RECORDED
                    gate outcome -- 1 only when the score formula passed
                    AND training.db recorded the challenger promoted
                    (display truth, 2026-10-22: a formula pass the
                    registry did not promote must never render as PASSED),
                    fpass (0/1) = the raw score-formula result kept as
                    data, rec = the recorded verdict text ("PROMOTED" /
                    "NOT PROMOTED"), newchamp (1 when the challenger won
                    and took the crown -- note champ then IS that
                    challenger, and the duel fields describe the champion
                    it dethroned)} or null,
                    sacc/sprec = stacker champion accuracy/precision.
                    chall/duel are null until a challenger duels a REAL
                    champion (migration-seated champions carry a -1e9
                    sentinel score and count as no duel).
                    pending (added 2026-10-25, the PROMOTION LOCK --
                    Kass: no challenger advances to champion without
                    his explicit agreement) = {i=run id, k=kind
                    ("ppo"|"stacker"), t=ts, why=recorded reason} of
                    the newest run recorded pending_approval that no
                    later same-kind promotion has resolved, else null.
                    When the pending run IS the gate duel's
                    challenger, duel also carries pend=1 so the
                    verdict area can show WOULD PROMOTE -- AWAITING
                    KASS. pass/fpass/rec keep their recorded meanings:
                    a pending challenger was NOT promoted, and rec
                    still says so.

Training run short keys:
  i=id, t=ts, k=kind ("ppo"|"stacker"), p=promoted (0/1),
  s=score (challenger duel score; stacker: challenger accuracy),
  cs=champion score (stacker: champion accuracy),
  pl=challenger holdout P&L, cpl=champion holdout P&L,
  dd=challenger max drawdown, cdd=champion max drawdown,
  r=challenger mean reward,
  ep=evaluation episodes (sample size), mix=challenger action mix,
  cmix=champion action mix when recorded -- each mix is serialized
  with the run's OWN action vocabulary intact: directional brains
  record {"hold", "buy", "sell", "short"} (v5), the exposure
  manager (Muse's Pick #3, run 75) records {"stand_aside",
  "half", "full", "hold"}. Consumers must render the keys present
  and never assume the directional set (a fixed-key read of an
  exposure mix yields NaN -- the 2026-10-27 watch defect),
  why=one-line verdict reason (stored on the run since
  2026-10-20; reconstructed from the run's own notes for older runs),
  steps=recorded training timesteps, deep=1 for a weekly deep train,
  fam=family code (same codes as the models index below), cfg=the
  family's recorded config/design note (lab families) -- each present
  only when the run itself recorded it (added 2026-10-25; absent,
  never fabricated, for runs that predate the field).
  The feed carries the last 4 runs (was 8 before the 2026-10-23 size
  diet, 6 until the 2026-11-05 diet; the models index below is the
  complete record of every run).

Training models index (snapshot.models, added 2026-10-22 -- Kass's
"display EVERY model tried" directive): one row per registered run,
newest first (the detailed runs list above stays the last 4). Since
2026-11-05 the shipped index is trimmed under Kass's directive of
that day: failed runs 16-50 (never promoted, no seat or crown) are
omitted from the feed to keep it inside its size budget, and the
feed carries the count as snapshot.models_omitted; the COMPLETE
record of every run stays in training.db, the Champions &
Challengers workbook, and the Model Atlas. _models_index itself
still covers ALL runs (the champs roster derives names from it).
Row = [i, d, n, f, p, s, pl, v, x]:
  i=run id, d=date "MM-DD", n=display name (derived in compact.py from
  the run's own record: metrics.family for lab runs -- "HA-Seq v1",
  "GBM Focus <arm> -- <label>", "GBM Lag <arm> -- <label>",
  "Exit Study -- <config label>", "Size Study 5% -- <signal>"; nightly
  runs "<PPO|Stacker> v<recipe>"
  from the version the run itself attests, inherited forward across
  its era ALONG THE PRODUCTION LINEAGE ONLY -- lab-family records
  attest study labels, not feature versions, and never update the
  inherited version (2026-10-25 fix; the composite screen's "V3"
  study name had renamed the newest nightly runs "PPO v3");
  "(early)" for the pre-v2 seed duels), f=family code (0=ppo,
  1=stacker, 2=ha_seq, 3=gbm_focus, 4=gbm_lag, 5=exit_study,
  6=size_study, 7=composite_screen, 8=tf4h, 9=regime_data,
  10=hierarchy, 11=features_v6, 12=full_indicator_stacker,
  13=scored_eyes_ppo, 14=muse_pick, 15=muse_pick2, 16=muse_pick3,
  17=meta_label, 18=eyes_v2, 19=regime_swing, 20=ha_deep,
  21=ha_shift, 22=muse_next_try, 23=muse_next_try2,
  24=muse_pick4, 25=muse_pick5, 26=muse_pick6, 27=muse_pick7,
  28=muse_pick8, 29=muse_pick9,
  30=muse_pick10,
  31=muse_pick11,
  32=muse_pick12, 33=muse_pick13, 34=muse_pick14),
  p=promoted (0/1), s=headline
  (PPO-family: duel aggregate score; stacker: holdout accuracy),
  pl=duel holdout P&L in whole dollars (null for the stacker; exact
  cents live in the detail rows / gate feed / training.db),
  v=recorded lab verdict
  ("PASS" / "FAIL" / "FAIL (vacuous)"; null for nightly runs),
  x=action mix (Action Mix section, Kass 2026-11-23): for PPO runs
  {"v": "d" directional | "x" exposure, "c": [hold, buy, sell,
  short] decision counts, "n": decisions in the last grading
  window} when the counts recover exactly from the recorded
  3-decimal shares (see _recover_mix_counts), {"v": ..., "f":
  2-decimal shares, "c": null, "n": null} when they don't; null
  for stackers (they vote, they don't act) and PPO runs with no
  recorded challenger mix. Counts are the last grading window's
  decisions only -- the same window the detail runs' mix/cmix
  describe -- never a whole-history aggregate.

Training recipe panel (snapshot.recipe, added 2026-10-25 -- Kass:
"upgrade it even more so I can tweek anything we train or try"): the
CURRENT training configuration, read from the code at build time so
it cannot silently drift. Nightly-only literals are parsed strictly
from src/training/nightly.py (a pattern that stops matching fails
the build loudly, never guesses); module constants are imported
(broker sizing/fee/stop/trail, PPO chunking, feature count); feat_v
comes from the registry's own version attestation over the
production lineage (the walk the models index names runs with,
restricted to nightly runs -- lab records carry study labels, not
feature versions; no FEATURE_VERSION constant exists in code).
Keys: steps (nightly PPO timesteps), deep_steps
(weekly deep train), chunk (PPO checkpoint segment), warm (1 =
challengers warm-start from the champion), windows (gate windows),
holdout_d (holdout length in days, shared by the stacker rule and
the PPO gate), episodes (duel episodes per window), fills
("next_open": next-candle-open fills), dd_pen (score = P&L -
dd_pen x max drawdown), gate_wins (windows the challenger must win),
floor [non-hold min share, P&L must exceed $, must beat buy-and-hold
0/1] (the two-part floor, Kass 2026-10-22), stacker ("strict_win":
promotion needs a strict holdout-accuracy win), tranche (per-trade
position cap), exposure (total exposure cap), fee (per side),
stop [ATR mult, min pct, max pct], trail [activate pct, ATR mult,
min pct, max pct], feat_n, feat_v. Display only -- the panel
changes nothing.

Training gate history (snapshot.gate_hist, added 2026-10-25): the
last 6 RECORDED gate duels -- nightly PPO runs (lab studies dueled
study baselines, never the crown) where both sides recorded a real
score. Row = [i, d, won, s, pl, cs, cpl, bpl, fpass, floor, rec]
plus an optional 12th element: i=run id, d=date "MM-DD",
won=windows the challenger won (derived from the recorded window
scores exactly as the gate feed derives them), s/pl=challenger
aggregate (1dp) / duel P&L (whole $), cs/cpl=champion aggregate /
P&L graded on the SAME windows, bpl=buy-and-hold P&L over those
windows, fpass=raw score-formula result (aggregate beats champion
AND won >= gate_wins), floor=1/0 when the formula passed (nightly
promotes only through gate_decision, so the recorded promotion
flag is the floor's verdict), null when the formula failed and
the floor never decided, rec=the RECORDED verdict text
("PROMOTED" / "NOT PROMOTED" -- from the registry's promoted flag,
never recomputed). The optional 12th element is the recorded
floor-block reason, present only when floor=0 and the run's own
recorded reason carries one.

Champions roster (snapshot.champs, added 2026-10-25 -- Kass's
lab vs driver's-seat separation): the title-holders per kind in
coronation order, with the grades from their registry records,
plus the driver's-seat line from data/live_seat.json naming what
live trading actually runs. See build_champions for the shape and
the status codes. Null until the separation exists. The vision
feed's p_up likewise reads the SEAT stacker
(models/live_seat/stacker.pkl) from the same date -- the Live
dashboard shows the driver, the Training Lab shows the lab.

What the brains read (snapshot.reads, added 2026-10-26 -- Kass:
the Training Lab should show how the models use his indicators):
{"p21": [[key, short plain-English name, is_kass], ...] -- the
exact prod21 feature set the seat stacker, the nightly stacker
challengers and the PPO read, with Kass's five v5 indicators
flagged (the PPO additionally reads the stacker's vote and three
readings of its own book); "c2": Champion #2's recorded study
data -- {"run", "n" (input count), "comp" (composite scores among
them), "top" (the ranked top-15 feature NAMES its registry record
stored at grading time -- names in recorded order only; no
numeric weights were stored, so none are carried), "kin"/"kout"
(which of Kass's five indicator families are / are not among its
scored inputs, matched against the run's recorded feature list)}
or null when no champion record carries study data.

Research incorporation (2026-10-26, from Kass's "Automated
Trading Dashboard Design" doc + Muse's RL research notes):
ledger gains "friction" {"fees_all", "fees_30d", "drift_bps",
"drift_usd", "n", "excl"} and "risk" {"peak", "floor_gap",
"floor_gap_pct", "exp", "cap"} (production book only; see
_friction_and_risk). vision gains "drift" {"psi" (21, FEATURE_
COLUMNS order), "bands" [stable, moderate, significant,
no-baseline], "base_end", "live_h", "base_h"} -- PSI of the last
168 closed hours vs the 792-hour grading holdout through the
production pipeline (features/drift.py; doc bands <0.1 /
0.1-0.25 / >=0.25). fillctx fills gain "top" (top-5 [key,
squeezed, plain-unit, direction] ranked by |value| x the seat
stacker's global importance -- the sanctioned non-SHAP fallback,
see _fill_top) and the payload gains "imp" (those importances).
The shadow snapshot gains "drift" {"worst" (top-10 [name, psi]),
"bands", "nb", "n", "base_end"} over Champion #2's scored set,
and the race gains "wfe" {"pair" [driver_run, eyes_run],
"exam_pnl", "exam_pct", "exam_days", "pexam"|null, "live_pnl",
"live_pct", "days", "fills", "ratio"|null} -- the exam read
from the newest scored_eyes_ppo registry record and attributed
to the pairing it graded via the seat manifest's "wfe_exam"
record (null exam fields + a "pexam" reference when that
pairing is not the desk's current one), the ratio carried only
once it can mean something (see _wfe_block).
The training snapshot gains "audit" (the trial ledger + best-of-N
champion context; see build_audit -- deliberately NO Deflated
Sharpe Ratio: the registry stores no per-period return series)
and "telemetry" {"from", "rows"} (SB3 training telemetry read
verbatim from run records; pre-capture runs are absent, never
reconstructed -- capture is logging-only in ppo_train and is
proven outcome-neutral by tests/test_ppo_telemetry.py). The
training snapshot also gains "votes" (the stacker champion's
stored holdout vote distribution -- data/meta_label/
votes_hold.npy, sha-guarded to the reigning champion artifact;
null when the crown has moved on) and "weights" (per-layer
weight stats of the live-seat PPO read from its zip's
policy.pth; null on any mismatch/failure) -- see _votes_block
(src "grading" from the champion's own registry record once
grading-time vote logging covers it, else src "study" from the
meta-label study's stored array) and _policy_weights_block.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.dashboards import snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _downsample_curve(curve: list[dict]) -> list[dict]:
    """Keep full resolution where anything changed. Only the *leading* flat
    pre-live run at exactly $100,000.0 is collapsed to its first and last
    point (lossless); later rows can pass back through $100,000.0 and must
    not be reclassified as flat. The tail uses an adaptive stride so the
    point count stays bounded as the curve grows."""
    if not curve:
        return curve
    out: list[dict] = []
    i = 0
    n = len(curve)
    while i < n and curve[i].get("equity") == 100000.0:
        i += 1
    if i > 0:
        out.append(curve[0])
        if i > 1:
            out.append(curve[i - 1])
    active = curve[i:]
    m = len(active)
    recent = active[-8:] if m > 8 else active
    tail = active[:-8] if m > 8 else []
    stride = max(1, (len(tail) + 7) // 8)
    for j in range(0, len(tail), stride):
        out.append(tail[j])
    out.extend(recent)
    if out and out[-1] != curve[-1]:
        out.append(curve[-1])
    return [{"t": p["t"], "equity": round(p["equity"], 2)} for p in out]


# Feed bound (2026-11-17 size fix): the calendar heatmap ships
# only the newest 371 days (53 weeks). snapshot.daily_calendar
# spans every day since the book's first closed trade -- one
# entry per day, forever -- which made every book snapshot feed
# grow ~72B/day without limit (+26KB/yr). The full history stays
# in the book's DB; the desk's heatmap window is a year + a week.
CALENDAR_FEED_DAYS = 371


def _feed_calendar(s: dict) -> list:
    return (s.get("daily_calendar") or [])[-CALENDAR_FEED_DAYS:]


def _drawdown_stats(s: dict) -> dict:
    """Current + worst drawdown as NUMBERS (viz research L1): the strip is
    derived in the frontend, but the two values a reader actually quotes --
    how far underwater are we now, and what's the worst it's been -- must
    exist as data, with the date of the worst point."""
    curve = s.get("drawdown_curve") or []
    if not curve:
        return {"dd_now": None, "dd_max": None, "dd_max_t": None}
    worst = min(curve, key=lambda p: p["dd_pct"])
    return {"dd_now": curve[-1]["dd_pct"], "dd_max": worst["dd_pct"],
            "dd_max_t": worst["t"]}


def _benchmark(curve_pts: list[dict], db_path=None) -> dict:
    """Buy-and-hold benchmark aligned to the equity curve (v4): what the
    $100K would be worth if it had simply bought BTC at the first equity
    row and held. `bench` is a bare number array zipped by index with
    equity_curve (kept tiny for the payload ceiling); bench_pct is the
    benchmark's total return % over the same window."""
    if not curve_pts:
        return {"bench": [], "bench_pct": None}
    con = snapshot._paper_db(db_path)
    if con is None:
        return {"bench": [], "bench_pct": None}
    rows = con.execute(
        "SELECT ts, price FROM equity ORDER BY ts ASC").fetchall()
    con.close()
    if len(rows) < 2:
        return {"bench": [], "bench_pct": None}
    p0 = rows[0][1]
    price_by_ts = {ts: px for ts, px in rows}
    vals = []
    last_px = p0
    for pt in curve_pts:
        px = price_by_ts.get(pt["t"], last_px)
        last_px = px
        vals.append(round(100_000.0 * px / p0, 2))
    bench_pct = round(100.0 * (rows[-1][1] / p0 - 1.0), 2)
    return {"bench": vals, "bench_pct": bench_pct}


def build_compact_trading() -> dict:
    s = snapshot.build_trading_snapshot()
    # Belt-and-braces (2026-10-20): a fresh account must never crash the
    # dashboard push with a KeyError on missing fields. Any non-live
    # snapshot (e.g. "not_started" when the DB file is missing entirely)
    # degrades to the minimal valid payload the dashboard expects.
    live_ok = s.get("status") == "live"
    if not live_ok:
        fresh = snapshot._fresh_account_snapshot(
            {"as_of": s.get("as_of")})
        s = fresh
    curve = _downsample_curve(s["equity_curve"])
    bench = _benchmark(curve) if live_ok else {"bench": [], "bench_pct": None}
    return {
        "as_of": s["as_of"],
        "status": s["status"],
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "stop_anchor": s["stop_anchor"],
        "stop_trailing": s["stop_trailing"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4) if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": _feed_calendar(s),
        "activity": s["activity"],
        "equity_curve": curve,
        "bench": bench["bench"],
        "bench_pct": bench["bench_pct"],
        **_drawdown_stats(s),
    }


def _seat_stacker_importances() -> list[float] | None:
    """Global feature importances of the DRIVER'S-SEAT stacker:
    the mean of its three base learners' fitted feature_importances_
    (each learner's vector is normalized to sum 1 first), in
    FEATURE_COLUMNS order. Extracted from the fitted artifact the
    seat actually trades with -- never recomputed, never the lab
    champion's. None when the seat artifact can't be read (the
    fill panel then keeps its unweighted ranking)."""
    try:
        from models.stacker import load as load_stacker

        path = ROOT / "models" / "live_seat" / "stacker.pkl"
        if not path.exists():
            return None
        model = load_stacker(path)
        stack = model.named_steps["stack"]
        vecs = []
        for est in stack.estimators_:
            v = getattr(est, "feature_importances_", None)
            if v is None:
                return None
            v = [float(x) for x in v]
            tot = sum(v)
            if tot <= 0:
                return None
            vecs.append([x / tot for x in v])
        n = len(vecs[0])
        if n == 0 or any(len(v) != n for v in vecs):
            return None
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]
    except Exception:
        return None


def _fill_top(f: list, imp: list[float], raws, ts) -> list:
    """The honest attribution fallback (2026-10-26; the design
    doc's SHAP panel, adapted to what is computable): the top-5
    features at this fill ranked by |squeezed value| x the seat
    stacker's global importance -- [key, squeezed, plain-unit
    string or None, direction], direction = the sign of
    value x importance (+1 pushed the stacker's up-vote up, -1
    pushed it down). Labeled on the panel as "strongest signals",
    NOT SHAP and not causal: the stacker's meta-learner composes
    three base learners, so no exact per-feature Shapley split of
    the vote exists to report."""
    import pandas as pd

    from features.technical_indicators import FEATURE_COLUMNS

    order = sorted(range(len(f)),
                   key=lambda i: -abs(f[i] * imp[i]))[:5]
    out = []
    stamp = pd.Timestamp(ts)
    for i in order:
        s = f[i] * imp[i]
        out.append([FEATURE_COLUMNS[i], round(float(f[i]), 3),
                    _raw_str(raws, stamp, FEATURE_COLUMNS[i]),
                    1 if s > 0 else -1 if s < 0 else 0])
    return out


def build_fillctx() -> dict:
    """Per-fill decision context (v4): for the newest fills that carry a
    stored feature snapshot, what the bot saw when it acted -- the
    feature values (FEATURE_COLUMNS order; 21 since v5) + stacker P(up).
    2026-10-26: each fill also carries "top" (see _fill_top) and the
    payload carries "imp", the seat stacker's global importances the
    ranking used -- both only when the seat artifact reads cleanly."""
    con = snapshot._paper_db()
    if con is None:
        return {"fills": []}
    cols = [r[1] for r in con.execute("PRAGMA table_info(trades)")]
    if "context" not in cols:
        # Pre-v4 ledger: the broker adds the column on its next init;
        # until then there is simply no context to show.
        con.close()
        return {"fills": []}
    rows = con.execute(
        "SELECT ts, side, context FROM trades WHERE context IS NOT NULL "
        "ORDER BY id DESC LIMIT 8").fetchall()
    con.close()
    fills = []
    for ts, side, ctx in rows:
        try:
            parsed = json.loads(ctx)
        except Exception:
            continue
        fills.append({"ts": ts, "side": side,
                      "f": parsed.get("f"), "p": parsed.get("p")})
    imp = _seat_stacker_importances()
    if imp is not None and fills:
        raws = None
        try:
            from data.market_data import fetch_klines

            raws = _raw_frame(fetch_klines("BTC-USD", interval="1h",
                                           period="60d"))
        except Exception:
            raws = None  # plain units are garnish; squeezed still ranks
        for fl in fills:
            f = fl.get("f")
            if isinstance(f, list) and len(f) == len(imp):
                fl["top"] = _fill_top(f, imp, raws, fl["ts"])
        return {"fills": fills, "imp": [round(x, 3) for x in imp]}
    return {"fills": fills}


def _friction_and_risk(con) -> dict:
    """Friction + risk proximity for the Live desk (2026-10-26
    research incorporation; the design doc's Part III question
    "what is friction costing?"). friction: total fees all-time
    and trailing-30d (wall clock), and the average decision-to-fill
    drift per fill -- the fill price vs the decision candle's close
    (the equity ledger's price at the fill's own timestamp IS that
    close: the live loop snapshots the same close it fills at).
    Fills with no recorded decision price are excluded and counted
    in "excl". The paper broker fills AT the decision close, so
    drift is 0 by construction today; the panel says so, and the
    number is here so a future execution change shows up in it.
    risk: all-time equity peak, the cushion above the $95,000
    alert floor, and current exposure vs the 75% cap."""
    from datetime import datetime, timedelta, timezone

    fills = con.execute(
        "SELECT ts, side, price, qty, fee FROM trades ORDER BY id ASC"
    ).fetchall()
    fees_all = round(sum(f[4] for f in fills), 2)
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    fees_30d = round(sum(f[4] for f in fills if f[0] >= cutoff), 2)
    closes = dict(con.execute("SELECT ts, price FROM equity").fetchall())
    bps: list[float] = []
    usd: list[float] = []
    excl = 0
    for ts, side, price, qty, _fee in fills:
        dc = closes.get(ts)
        if not dc:
            excl += 1
            continue
        d = price - dc
        bps.append(d / dc * 10000.0)
        # Signed as a COST: paying above the decision close on a
        # buy, or receiving below it on a sell/short, is positive.
        usd.append(d * qty if side == "BUY" else -d * qty)
    n = len(bps)
    friction = {"fees_all": fees_all, "fees_30d": fees_30d,
                "drift_bps": round(sum(bps) / n, 2) if n else 0.0,
                "drift_usd": round(sum(usd) / n, 2) if n else 0.0,
                "n": n, "excl": excl}
    risk = None
    eq = con.execute(
        "SELECT equity, btc, price FROM equity ORDER BY ts ASC").fetchall()
    if eq:
        peak = max(r[0] for r in eq)
        equity, btc, price = eq[-1]
        gap = round(equity - 95000.0, 2)
        risk = {"peak": round(peak, 2), "floor_gap": gap,
                "floor_gap_pct": round(100.0 * gap / 95000.0, 2),
                "exp": round(100.0 * abs(btc) * price / equity, 2)
                if equity else 0.0,
                "cap": 75}
    return {"friction": friction, "risk": risk}


def build_ledger(db_path=None) -> dict:
    con = snapshot._paper_db(db_path)
    if con is None:
        return {"ledger": [], "closed": []}
    rows = con.execute(
        "SELECT ts, side, price, qty, fee FROM trades ORDER BY id DESC LIMIT 24"
    ).fetchall()
    ledger = [[r[0], r[1], round(r[2], 2), round(r[3], 6), round(r[4], 2)]
              for r in rows]
    closed = snapshot.closed_trades(con)
    out = {
        "ledger": ledger,
        "closed": [[c["close_ts"], round(c["qty"], 6), round(c["pnl"], 2)]
                   for c in closed[-12:]],
    }
    if db_path is None:
        # Live desk only: the shadow ledger keeps its original
        # shape (Book #2's own friction is the race/WFE story).
        out.update(_friction_and_risk(con))
    con.close()
    return out


def _market_candles():
    """Last 96 closed 1h candles as (start_epoch, [[o,h,l,c,v]...], df).
    Volume rides as a 5th element (v4 premium dashboard renders a volume
    strip when present; Coinbase volume is real BTC units)."""
    from data.market_data import fetch_klines
    df = fetch_klines("BTC-USD", interval="1h", period="5d").iloc[:-1]
    df = df.tail(96)
    start = int(df.index[0].timestamp())
    candles = [[int(round(r["open"])), int(round(r["high"])),
                int(round(r["low"])), int(round(r["close"])),
                int(round(r["volume"]))]
               for _, r in df.iterrows()]
    return start, candles, df


def build_market(part: int, db_path=None, state_path=None,
                 ha: bool = False) -> dict:
    start, candles, df = _market_candles()
    if ha:
        # Book #4's chart is the Heikin-Ashi candle stream
        # (Kass's order, 2026-11-05): HA OHLC computed over the
        # full 5d window (the recursion's seed warmth dies within
        # the lead-in rows), volume stays the real volume, and
        # markers/entry/stop below stay on real prices -- the
        # trading itself is on real prices; only the candle
        # rendering is HA.
        from data.market_data import fetch_klines
        from models.ha_seq import heikin_ashi

        closed_full = fetch_klines("BTC-USD", interval="1h",
                                   period="5d").iloc[:-1]
        full_df = closed_full.tail(96)
        h = heikin_ashi(closed_full).tail(96)
        candles = [[int(round(h["ha_open"].iloc[i])),
                    int(round(h["ha_high"].iloc[i])),
                    int(round(h["ha_low"].iloc[i])),
                    int(round(h["ha_close"].iloc[i])),
                    int(round(full_df["volume"].iloc[i]))]
                   for i in range(len(full_df))]
        start = int(full_df.index[0].timestamp())
    half = 48
    if part == 0:
        return {"start": start, "step": 3600, "candles": candles[:half]}
    out = {"start": start + half * 3600, "step": 3600, "candles": candles[half:]}
    # Markers: fills inside the candle window + current entry/stop lines.
    # Fill timestamps in paper.db are UTC wall-clock strings (written from
    # the yfinance index); parse them as UTC explicitly -- a naive
    # .timestamp() would read them as local time and shift markers by the
    # UTC offset (bug caught 2026-10-20: markers landed +4h off the grid).
    from datetime import datetime, timezone
    win_start_epoch = start
    con = snapshot._paper_db(db_path)
    markers = []
    entry = stop = None
    if con is not None:
        rows = con.execute(
            "SELECT ts, side, price, qty FROM trades ORDER BY id ASC").fetchall()
        for ts, side, price, qty in rows:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            ep = int(t.timestamp())
            if ep >= win_start_epoch:
                markers.append([ep, side[0], round(price, 2), round(qty, 6)])
        con.close()
    markers = markers[-20:]
    state_file = (Path(state_path) if state_path is not None
                  else ROOT / "data" / "live_state.json")
    if state_file.exists():
        live = json.loads(state_file.read_text())
        entry = live.get("last_entry_price")
        stop = _live_stop_level(live) if entry else None
    out.update({"markers": markers, "entry_price": entry, "stop_price": stop})
    return out


def _live_stop_level(live: dict) -> float | None:
    """The stop the market feed draws for the live position: the
    broker's EFFECTIVE stop level -- PaperBroker.stop_level() on
    the persisted state, computed by snapshot._stop_display (the
    trading feed's mirror), so every display agrees with the exit
    logic itself and not just with each other. The persisted
    anchor alone headlined here until 2026-10-24: the anchor is
    only the locked INITIAL stop, and once the trailing arm
    activates the level that fires is tighter (live short: anchor
    $86,607.68 vs effective $81,285.08). Flat (persisted btc == 0)
    -> None, even if a stale last_entry_price lingers. States
    that predate the anchor get the broker's own entry/ATR
    fallback from the same stop_level call -- there is no local
    formula left to keep in sync."""
    btc = live.get("btc")
    if btc is not None and btc == 0:
        return None
    return snapshot._stop_display(live, btc or 0.0)[0]


# Plain-English label per feature for the vision feed ("what the bot sees").
FEATURE_LABELS = {
    "body_pct": "Candle body: close vs open this hour",
    "upper_wick": "Upper wick: sellers pushed price down from the high",
    "lower_wick": "Lower wick: buyers pushed price up from the low",
    "range_pct": "Candle range: high-to-low size this hour",
    "rsi_c": "RSI momentum: -1 oversold, +1 overbought",
    "macd_hist_rel": "MACD histogram: trend momentum",
    "dist_vwap": "Distance from VWAP (average traded price)",
    "dist_sma50": "Distance from the 50-hour average price",
    "atr_rel": "ATR: typical hourly move size (volatility)",
    "price_change": "Last hour's price change",
    "volatility_21": "Volatility over the last 21 hours",
    "volume_change": "Trading volume change vs previous hour",
    "h4_rsi_c": "4-hour RSI momentum: -1 oversold, +1 overbought",
    "h4_dist_sma": "Distance from the 4-hour 20-period average price",
    "d1_dist_sma": "Distance from the daily 20-day average price (big trend)",
    "h4_atr_rel": "4-hour ATR: typical 4-hour move size (volatility)",
    "ha_dir": "Heikin-Ashi direction: smoothed trend, + up / - down (ATR units)",
    "obv_slope": "On-Balance Volume slope: volume flowing in (+) or out (-) over 24h",
    "adl_slope": "Accumulation/Distribution slope: buying (+) vs selling (-) pressure over 24h",
    "netvol_rel": "Net volume: signed share of the last 24h of volume",
    "ewo": "Elliott Wave Oscillator: fast vs slow momentum, + bullish / - bearish",
}


# ---------------------------------------------------------------------
# Plain-unit readings behind the squeezed vision values (2026-10-26,
# Kass: "do you think it's smart to keep every single number -1 thru
# +1? ... some of those things look like they don't make sense in
# that format"). The brain reads the squeezed inputs; the dashboard
# ALSO shows the same signal in human units beside it (RSI as its
# 0-100 level, distances as %, volume-flow shares as %). Raws are
# RECOMPUTED from the same candle frame the features come from,
# mirroring add_features step for step -- never inverted from the
# squeezed value, because the v3 squeeze CLIPS at the rails and a
# clipped value cannot be un-clipped. tests/test_vision_raws.py pins
# the mirror: un-squeezing each raw with the feature's own transform
# reproduces the brain's actual input on every sampled row.
#
# EMISSION GATE (2026-10-26): the dashboards' vision schema
# originally capped f/f21 rows at 3 elements -- emitting the 4th
# made the live loop's push_vision fail validation every cycle --
# so emission was gated off while the bot's-eyes panel shape was
# under redesign discussion. Kass decided the shape the same day
# (a grouped HEAT GRID whose cells carry the plain-unit value,
# heat = the squeezed lean), the push_vision schema was widened to
# 4-element rows in the same change set, and the gate is OPEN:
# the feeds emit [key, label, squeezed, raw]. If a dashboard schema
# ever caps rows again, close this flag instead of touching the
# computation -- the raws stay computed and test-pinned either way.
_VISION_RAWS_EMITTED = True

def _raw_frame(full):
    """Raw (pre-squeeze) value of every FEATURE_COLUMNS feature for
    each row of an OHLCV frame, in the feature's natural unit:
    fractions for the price-relative features (formatted as %),
    0-100 for the RSIs, ATR units for ha_dir, [-1, 1] volume shares
    for the v5 flow trio (formatted as %). NaN where a warmup has
    not completed."""
    import numpy as np
    import pandas as pd

    from features import technical_indicators as ti

    df = full
    o, h, l, c, v = (df["open"], df["high"], df["low"],
                     df["close"], df["volume"])
    out = pd.DataFrame(index=df.index)
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    out["body_pct"] = (c - o) / o
    out["upper_wick"] = (h - top) / o
    out["lower_wick"] = (bot - l) / o
    out["range_pct"] = (h - l) / o
    pv = c * v
    vwap = pv.rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    macd = (c.ewm(span=12, adjust=False).mean()
            - c.ewm(span=26, adjust=False).mean())
    signal = macd.ewm(span=9, adjust=False).mean()
    out["rsi_c"] = ti._rsi(c, 14)
    out["macd_hist_rel"] = (macd - signal) / c
    out["dist_vwap"] = (c - vwap) / c
    out["dist_sma50"] = (c - c.rolling(50).mean()) / c
    atr = ti._atr(df, 14)
    out["atr_rel"] = atr / c
    out["price_change"] = c.pct_change()
    roll = c.rolling(21)
    out["volatility_21"] = roll.std() / roll.mean()
    out["volume_change"] = v.replace(0, np.nan).pct_change()
    ha_open, ha_close = ti._heikin_ashi(df)
    out["ha_dir"] = (ha_close - ha_open) / atr.replace(0, np.nan)
    vol24 = v.rolling(24).sum().replace(0, np.nan)
    obv = (v * np.sign(c.diff().fillna(0.0))).cumsum()
    out["obv_slope"] = obv.diff(24) / vol24
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    adl = (mfm.fillna(0.0) * v).cumsum()
    out["adl_slope"] = adl.diff(24) / vol24
    out["netvol_rel"] = (v * np.sign(c - o)).rolling(24).sum() / vol24
    out["ewo"] = ((c.ewm(span=5, adjust=False).mean()
                   - c.ewm(span=35, adjust=False).mean()) / c)
    # Multi-timeframe raws: the same completed-bucket merge as
    # add_features (bucket-END stamps, values-based, backward asof
    # onto decision time = row start + 1h), carrying the unscaled
    # intermediates add_features scales and then drops.
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    h4 = df.resample("4h", closed="left", label="left").agg(agg).dropna()
    h4_raw = pd.DataFrame({
        "h4_rsi_c": ti._rsi(h4["close"], 14).to_numpy(),
        "h4_dist_sma": ((h4["close"] - h4["close"].rolling(20).mean())
                        / h4["close"]).to_numpy(),
        "h4_atr_rel": (ti._atr(h4, 14) / h4["close"]).to_numpy(),
    }, index=h4.index + pd.Timedelta(hours=4))
    d1 = df.resample("1d", closed="left", label="left").agg(agg).dropna()
    d1_raw = pd.DataFrame({
        "d1_dist_sma": ((d1["close"] - d1["close"].rolling(20).mean())
                        / d1["close"]).to_numpy(),
    }, index=d1.index + pd.Timedelta(days=1))
    base = pd.DataFrame({"decision_ts": df.index + pd.Timedelta(hours=1)},
                        index=df.index)
    for feats in (h4_raw, d1_raw):
        for col in feats.columns:
            out[col] = np.nan
        if len(feats):
            merged = pd.merge_asof(
                base.sort_values("decision_ts"), feats.sort_index(),
                left_on="decision_ts", right_index=True,
                direction="backward")
            merged.index = base.index
            for col in feats.columns:
                out[col] = merged[col].values
    return out.reindex(columns=ti.FEATURE_COLUMNS).replace(
        [np.inf, -np.inf], np.nan)


def _format_raw(key, x) -> str | None:
    """One raw reading as a compact display string in its natural
    unit, or None when the reading does not exist (warmup / no
    data) -- the dashboard then shows that row squeezed-only."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if key in ("rsi_c", "h4_rsi_c"):
        return f"{x:.0f}"
    if key == "ha_dir":
        return f"{x:+.2f} ATR"
    if key == "volume_change":
        return f"{x * 100:+.0f}%"
    if key in ("obv_slope", "adl_slope", "netvol_rel"):
        return f"{x * 100:+.0f}%"
    if key in ("upper_wick", "lower_wick", "range_pct", "atr_rel",
               "volatility_21", "h4_atr_rel"):
        return f"{x * 100:.2f}%"
    return f"{x * 100:+.2f}%"


def _raw_str(raws, ts, col) -> str | None:
    if raws is None or ts not in raws.index:
        return None
    return _format_raw(col, raws.loc[ts, col])


def _grading_base_end() -> str | None:
    """The newest registry run's timestamp -- the anchor for every
    drift baseline below. The grading holdout is by construction
    the 792 closed candles ending at the last closed candle at or
    before the most recent grading, so the baseline window is the
    exam window itself, recomputed through the same pipeline."""
    try:
        runs = _all_runs_desc()
        return runs[0]["ts"] if runs else None
    except Exception:
        return None


def _drift_windows(feat, base_end_ts):
    """(baseline, live) row windows over a feature frame of closed
    candles: live = the last 168 rows (7 days); baseline = the 792
    rows ending at the last row at/before base_end_ts (the grading
    holdout), falling back to the 792 rows just before the live
    window when no grading anchor exists."""
    import pandas as pd

    live = feat.tail(168)
    base = None
    if base_end_ts is not None:
        bound = pd.Timestamp(base_end_ts)
        if bound.tzinfo is not None:
            # Registry timestamps are tz-aware; candle frames are
            # UTC wall-clock (tz-naive) by the adapter contract.
            bound = bound.tz_convert("UTC").tz_localize(None)
        upto = feat.loc[:bound]
        if len(upto) >= 200:
            base = upto.tail(792)
    if base is None:
        base = feat.iloc[-960:-168] if len(feat) > 960 else feat.iloc[:-168]
    return base, live


def _drift_block(fetch=None) -> dict | None:
    """PSI drift of the 21 production features (2026-10-26 research
    incorporation; design doc Part III): this week's live feature
    distribution vs the grading-holdout baseline, per feature, in
    FEATURE_COLUMNS order. {"psi": [21 x value|None], "bands":
    [stable, moderate, significant, no-baseline], "base_end",
    "live_h": 168, "base_h": 792}. None on any failure -- the
    panel hides rather than showing a stale or fabricated grid
    (the block is recomputed on every vision push)."""
    try:
        from data.market_data import fetch_klines
        from features.drift import psi_frame
        from features.technical_indicators import (FEATURE_COLUMNS,
                                                   add_features)

        fetch = fetch or fetch_klines
        raw = fetch("BTC-USD", interval="1h", period="730d")
        closed = raw.iloc[:-1]
        feat = add_features(closed).dropna(subset=FEATURE_COLUMNS)
        if len(feat) < 400:
            return None
        base, live = _drift_windows(feat, _grading_base_end())
        psi = psi_frame(base, live, FEATURE_COLUMNS)
        vals: list = []
        bands = [0, 0, 0, 0]
        for c in FEATURE_COLUMNS:
            v = psi[c]
            vals.append(round(v, 3) if v is not None else None)
            if v is None:
                bands[3] += 1
            elif v < 0.1:
                bands[0] += 1
            elif v < 0.25:
                bands[1] += 1
            else:
                bands[2] += 1
        return {"psi": vals, "bands": bands,
                "base_end": str(base.index[-1].date()),
                "live_h": 168, "base_h": 792}
    except Exception:
        return None


def build_vision(state_path=None, fetch=None) -> dict:
    """What the bot saw on its OWN last processed candle: all 21
    features (v5; v3 contract, all in [-1, 1]) + the DRIVER'S-SEAT
    stacker's P(up) -- the seat artifact the live loop actually
    trades with (seat separation, 2026-10-25), never the Training
    Lab's champion, which may read an entirely different feature
    set. The candle and the vote are the live loop's OWN recorded
    ones (live_state.json last_ts / last_p_up, see _own_candle_pin):
    until 2026-11-03 this panel recomputed on the newest closed
    candle, which could run ahead of the loop and disagree with
    the desk's own snapshot vote. Each f row's 4th element is the
    same signal in plain units (see _raw_frame), or null when no
    honest raw exists for that row.

    `fetch` is an optional market-fetch callable (the master build
    passes its one shared memoized fetch, 2026-11-04); None keeps
    the standalone behavior -- fetch_klines direct, including
    inside the drift block."""
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features
    from models.stacker import load as load_stacker

    fetch = fetch or fetch_klines
    full = fetch("BTC-USD", interval="1h", period="60d").iloc[:-1]
    feat = add_features(full)
    feat = feat.dropna(subset=FEATURE_COLUMNS)
    own_ts, own_p = _own_candle_pin(
        state_path if state_path is not None
        else ROOT / "data" / "live_state.json")
    pinned = own_ts is not None and own_ts in feat.index
    if pinned:
        ts, row = own_ts, feat.loc[own_ts]
    else:
        ts, row = feat.index[-1], feat.iloc[-1]
    p_up = own_p if pinned else None
    champ = ROOT / "models" / "live_seat" / "stacker.pkl"
    if p_up is None and champ.exists():
        stacker = load_stacker(champ)
        p_up = round(float(stacker.predict_proba(
            row[FEATURE_COLUMNS].to_frame().T)[0, 1]), 4)
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(full)
        except Exception:
            raws = None  # display-only: fall back to squeezed-only
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(row[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    out = {
        "ts": pd.Timestamp(ts).isoformat(),
        "p_up": p_up,
        "f": rows,
    }
    drift = _drift_block(fetch)
    if drift is not None:
        out["drift"] = drift
    return out


def _n(x):
    """Serialize an integral float as an int (-2117.0 -> -2117): same
    number to any JSON consumer, 2 fewer bytes on the wire. Part of the
    2026-10-23 training-feed diet (see build_compact_training)."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _mix(m: dict) -> dict:
    """Action mix at 2 decimals -- the dashboard renders mixes as bars /
    percents, so 1% resolution is display-identical and trims the
    3-decimal tails (0.858 -> 0.86)."""
    return {k: _n(round(v, 2)) for k, v in m.items()}


# Action-mix vocabularies, in canonical order: the directional brain's
# four actions, and the exposure manager's (Muse's Pick #3, run 75).
_MIX_DIRECTIONAL = ("hold", "buy", "sell", "short")
_MIX_EXPOSURE = ("stand_aside", "half", "full", "hold")


def _mix_vocab(raw: dict) -> list[str] | None:
    """The canonical key order for a recorded mix, or None for an
    unknown vocabulary (never guessed from a partial key set)."""
    keys = set(raw)
    if keys <= set(_MIX_EXPOSURE) and keys - {"hold"}:
        return list(_MIX_EXPOSURE)
    if keys <= set(_MIX_DIRECTIONAL):
        return list(_MIX_DIRECTIONAL)
    return None


def _recover_mix_counts(mix: dict, episodes) -> tuple | None:
    """Exact decision counts behind a recorded action mix, or None.

    Action Mix — every model (Kass, 2026-11-23): the grading env
    runs 24-decision episodes (TradingEnv episode_len = 1 day x 24
    hourly candles), so a standard duel window is 10 episodes x 24
    = 240 decisions, and a recorded 3-decimal share recovers
    exactly when every share has one integer preimage at that
    total and the counts sum to it. The total comes from the run's
    own recorded episodes (total across the 3-window duel / 3)
    when present, else the 10-episode default -- and the exact-sum
    check rejects a wrong total rather than guessing. Mixes stored
    as exact fractions (never rounded, e.g. run 88's 256/264)
    recover from their own denominator instead. Returns
    (vocab, counts, total) or None when the counts cannot be
    recovered exactly: the caller ships shares only, never
    interpolated counts."""
    vocab = _mix_vocab(mix)
    if vocab is None:
        return None
    # Exact-fraction mixes: recover from their own denominator --
    # but only when it cannot be a 3-decimal rounding (denominator
    # not dividing 1000), else a per-mille reading would masquerade
    # as counts (run 153's 0.742/0.004/0.129/0.125 would read as
    # 742/4/129/125 over 1000).
    from fractions import Fraction
    import math
    fracs = {}
    exact = True
    for k in vocab:
        v = mix.get(k, 0.0)
        if not isinstance(v, (int, float)):
            return None
        f = Fraction(v).limit_denominator(10000)
        if float(f) != float(v):
            exact = False
            break
        fracs[k] = f
    if exact:
        den = 1
        for f in fracs.values():
            den = den * f.denominator // math.gcd(den, f.denominator)
        if den > 1 and 1000 % den != 0:
            counts = [int(fracs[k] * den) for k in vocab]
            if sum(counts) == den:
                return (vocab, counts, den)
    # Rounded-share recovery at the window total.
    if isinstance(episodes, int) and episodes > 0 and episodes % 3 == 0:
        total = (episodes // 3) * 24
    else:
        total = 240
    counts = []
    for k in vocab:
        v = mix.get(k, 0.0)
        if not isinstance(v, (int, float)):
            return None
        cands = [c for c in range(total + 1) if round(c / total, 3) == v]
        if len(cands) != 1:
            return None
        counts.append(cands[0])
    if sum(counts) != total:
        return None
    return (vocab, counts, total)


def _index_mix(r: dict):
    """The models-index action-mix element (9th row element) for the
    Action Mix section: {"v": "d"|"x", "c": counts, "n": total}
    when the counts recover exactly, {"v": ..., "f": 2-decimal
    shares} when they don't, None for stackers (they vote, they
    don't act) and PPO runs with no recorded challenger mix."""
    if r["kind"] != "ppo":
        return None
    raw = (r.get("metrics") or {}).get("challenger", {}).get("action_mix")
    if not raw:
        return None
    vocab = _mix_vocab(raw)
    if vocab is None:
        return None
    got = _recover_mix_counts(
        raw, (r.get("metrics") or {}).get("challenger", {}).get("episodes"))
    tag = "x" if vocab == list(_MIX_EXPOSURE) else "d"
    if got is None:
        return {"v": tag, "f": [_mix(raw).get(k, 0) for k in vocab],
                "c": None, "n": None}
    _, counts, total = got
    return {"v": tag, "c": counts, "n": total}


def _verdict_reason_raw(r: dict, m: dict) -> str | None:
    """The run's verdict reason UNBOUNDED, exactly as recorded
    (explicit `reason` / `promotion_reason`, else reconstructed from
    what the run recorded -- champion notes, promotion fields, or
    the duel outcome itself)."""
    reason = m.get("reason") or m.get("promotion_reason")
    if not reason and r["kind"] == "ppo":
        h = m.get("champion") or {}
        reason = h.get("note")
        if not reason and h.get("policy"):
            reason = str(h["policy"])
    if not reason:
        if r["kind"] == "ppo" and not m.get("champion"):
            reason = ("challenger scored below the do-nothing baseline (0)"
                      if not r["promoted"] else "first recorded duel")
        elif r["promoted"]:
            reason = "challenger won the duel"
        else:
            reason = "champion kept the crown"
    return str(reason) if reason else None


def _verdict_reason(r: dict, m: dict) -> str | None:
    """One-line verdict for a run card (viz research T1/T2: the reason a
    brain was promoted or kept out must live on the dashboard, not only in
    training.db), word/clause-bounded at the 100-char preview cap,
    never a raw slice: the [:100] cut landed mid-number on run 88's
    reason ("...duel agg -1550 vs champion -" | "210, ...") in the
    2026-10-31 dashboard review. When the cap trims, the feed ALSO
    ships the complete text as why_full (2026-11-03 audit: a preview
    whose remainder exists nowhere in the payload is information
    destroyed, not shortened)."""
    raw = _verdict_reason_raw(r, m)
    if raw is None:
        return None
    return _clause_bounded(raw, limit=100, word_fallback=True,
                           ellipsis=True)


def _duel_run(r: dict) -> dict:
    m = r.get("metrics") or {}
    out = {"i": r["id"], "t": r["ts"], "k": r["kind"], "p": int(r["promoted"])}
    if r["kind"] == "ppo":
        c, h = m.get("challenger") or {}, m.get("champion") or {}
        if isinstance(c.get("score"), (int, float)):
            out["s"] = _n(round(c["score"], 1))
        if isinstance(h.get("score"), (int, float)):
            out["cs"] = _n(round(h["score"], 1))
        if isinstance(c.get("total_pnl"), (int, float)):
            out["pl"] = round(c["total_pnl"], 2)
        if isinstance(h.get("total_pnl"), (int, float)):
            out["cpl"] = round(h["total_pnl"], 2)
        if isinstance(c.get("max_drawdown"), (int, float)):
            out["dd"] = round(c["max_drawdown"], 2)
        if isinstance(h.get("max_drawdown"), (int, float)):
            out["cdd"] = round(h["max_drawdown"], 2)
        if isinstance(c.get("mean_reward"), (int, float)):
            out["r"] = round(c["mean_reward"], 3)
        if c.get("episodes"):
            out["ep"] = c["episodes"]
        elif h.get("episodes"):
            out["ep"] = h["episodes"]
        if c.get("action_mix"):
            out["mix"] = _mix(c["action_mix"])
        if h.get("action_mix"):
            out["cmix"] = _mix(h["action_mix"])
        if isinstance(c.get("bench_pnl"), (int, float)):
            # Buy-and-hold P&L over the same gate windows both brains were
            # graded on -- identical for challenger and champion.
            out["bpl"] = round(c["bench_pnl"], 2)
        if c.get("windows"):
            out["win"] = [round(w["score"], 1) for w in c["windows"]]
        if h.get("windows"):
            out["cwin"] = [round(w["score"], 1) for w in h["windows"]]
    else:  # stacker
        if isinstance(m.get("challenger_acc"), (int, float)):
            out["s"] = round(m["challenger_acc"], 4)
        if isinstance(m.get("champion_acc"), (int, float)):
            out["cs"] = round(m["champion_acc"], 4)
    # Recorded settings (2026-10-25): runs compare by settings, not
    # only scores. The registry records training steps + the deep
    # flag on the nightly/lab PPO path and a config record per lab
    # family; surface exactly what was recorded -- absent (never
    # fabricated) for runs that predate the field (older runs, the
    # stacker, whose runs record no step count).
    if isinstance(m.get("timesteps"), int):
        out["steps"] = m["timesteps"]
    if isinstance(m.get("deep"), bool):
        out["deep"] = 1 if m["deep"] else 0
    fam = m.get("family")
    if fam in _FAMILY_CODES:
        out["fam"] = _FAMILY_CODES[fam]
        sub = m.get(fam) or {}
        cfg = sub.get("design")
        if not isinstance(cfg, str) and isinstance(sub.get("config"), dict):
            cfg = ", ".join(f"{k}={_n(v)}"
                            for k, v in sub["config"].items()
                            if v is not None)
        if not isinstance(cfg, str) and fam == "size_study" \
                and isinstance(sub.get("tranche_pct"), (int, float)):
            cfg = f"tranche {_n(sub['tranche_pct'])}%"
        if isinstance(cfg, str) and cfg:
            # Word/clause-bounded at the same 80-char cap, never a
            # raw slice: the [:80] cut landed mid-word on the HA /
            # regime cfg texts ("...sub-cha", "...deterministic
            # swi") in the 2026-10-31 dashboard review.
            out["cfg"] = _clause_bounded(cfg, limit=80,
                                         word_fallback=True,
                                         ellipsis=True)
    why = _verdict_reason(r, m)
    if why:
        out["why"] = why
        raw = _verdict_reason_raw(r, m)
        if raw is not None and raw != why:
            # The preview was trimmed: ship the complete recorded
            # reason alongside it (2026-11-03 audit).
            out["why_full"] = raw
    return out


def _lab_notes_raw() -> list[dict]:
    """The full lab notebook as stored (oldest first)."""
    path = ROOT / "data" / "lab_notes.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _lab_notes() -> list[dict]:
    """Newest lab-notebook entries (v4), compact form for the feed."""
    notes = _lab_notes_raw()
    # Word/clause-bounded at the same 90-char cap, never a raw
    # slice: the [:90] cut landed mid-number on the HA-Deep note
    # ("...exam acc 0.5" | "208 ...") in the 2026-10-31 review.
    # When the cap trims, the COMPLETE stored detail rides along as
    # "full" (2026-11-03 audit: the preview was the only copy in the
    # payload, so the two long Kass-directive notes -- the 1,218-char
    # trailing-stop ruling and the 795-char run-96 crowning -- could
    # not be read past ~90 chars anywhere on the dashboard).
    out = []
    # Newest 3 (was 4 until the 2026-11-05 size diet -- see the SIZE
    # DIET note in build_compact_training; every shipped note still
    # carries its complete text as "full" when its preview is trimmed).
    for n in notes[-3:]:
        full = str(n["detail"])
        d = _clause_bounded(full, limit=90, word_fallback=True,
                            ellipsis=True)
        item = {"t": n["ts"], "h": n["title"], "d": d}
        if d != full:
            item["full"] = full
        out.append(item)
    return out[::-1]


_FAMILY_CODES = {"ppo": 0, "stacker": 1, "ha_seq": 2, "gbm_focus": 3,
                 "gbm_lag": 4, "exit_study": 5, "size_study": 6,
                 "composite_screen": 7, "tf4h": 8, "regime_data": 9,
                 "hierarchy": 10, "features_v6": 11,
                 "full_indicator_stacker": 12,
                 "scored_eyes_ppo": 13,
                 "muse_pick": 14,
                 "muse_pick2": 15,
                 "muse_pick3": 16,
                 "meta_label": 17,
                 "eyes_v2": 18,
                 "regime_swing": 19,
                 "ha_deep": 20,
                 "ha_shift": 21,
                 "muse_next_try": 22,
                 "muse_next_try2": 23,
                 "muse_pick4": 24,
                 "muse_pick5": 25,
                 "muse_pick6": 26,
                 "muse_pick7": 27,
                 "muse_pick8": 28,
                 "muse_pick9": 29,
                 "muse_pick10": 30,
                 "muse_pick11": 31,
                 "muse_pick12": 32,
                 "muse_pick13": 33,
                 "muse_pick14": 34}
_GBM_ARM_LABELS = {"A": "Plain", "B": "Class-Weighted", "C": "SMOTE",
                   "D": "All-60 Features"}
_GBM_LAG_ARM_LABELS = {"E": "Lag Class-Weighted", "F": "Lag SMOTE",
                       "G": "Lag All-60 Features"}
_VERSION_RE = re.compile(r"v(\d+(?:\.\d+)?)")


def _version_candidates(r: dict, note: dict | None) -> list[str]:
    """Recipe versions a run itself attests to, from its own record: the
    features marker (v3 era), a `vN_duel` metric key, its reason /
    promotion_reason / recovery text, its champion's note, or the lab
    note written at the run's exact timestamp (migration notes title the
    recipe in force, e.g. "Features v5.1 migration")."""
    m = r.get("metrics") or {}
    cands: list[str] = []
    feats = m.get("features")
    if isinstance(feats, str):
        hit = _VERSION_RE.search(feats)
        if hit:
            cands.append(hit.group(1))
    for key in m:
        hit = re.match(r"v(\d+)_duel$", key)
        if hit:
            cands.append(hit.group(1))
    texts = [m.get("reason"), m.get("promotion_reason"), m.get("recovery"),
             (m.get("champion") or {}).get("note")]
    if note:
        texts += [note.get("title"), note.get("detail")]
    for text in texts:
        if isinstance(text, str):
            hit = _VERSION_RE.search(text)
            if hit:
                cands.append(hit.group(1))
    return cands


def _pick_version(cands: list[str]) -> str | None:
    """Most specific marker wins ("5.1" over "5"): a migration run's own
    reason can still name the outgoing recipe while its lab note titles
    the new one (run 16: reason "v5 stacker", note "Features v5.1")."""
    if not cands:
        return None
    return max(cands, key=lambda v: ("." in v,
                                     tuple(int(x) for x in v.split("."))))


# Kass's Pick-lineage rename (2026-11-04, Kass's direct order): runs
# 101/102/103 are called Muse's Pick #4/#5/#6, continuing Pick #1
# (run 69), Pick #2 (run 73) and Pick #3 (run 75). Runs 101/102 were
# built as the "Muse's Next Try" studies and run 103 is the
# 2026-11-04 nightly PPO challenger; this is a display-name decision
# only -- families, registry records and recorded verdicts
# (FAIL / FAIL / NOT PROMOTED) are untouched.
_PICK_RENAMES = {101: "Muse's Pick #4", 102: "Muse's Pick #5",
                 103: "Muse's Pick #6"}


def _display_name(r: dict, version: str | None) -> str:
    """Human-readable model name for the all-runs index. Lab families
    carry their own names (metrics.family); nightly runs are
    "<Brain> v<recipe>", with "(early)" for the pre-v2 seed duels that
    recorded no version anywhere."""
    m = r.get("metrics") or {}
    fam = m.get("family") or r["kind"]
    pick_name = _PICK_RENAMES.get(r.get("id"))
    if pick_name:
        return pick_name
    if fam == "ha_seq":
        return "HA-Seq v1"
    if fam == "gbm_focus":
        g = m.get("gbm_focus") or {}
        arm = g.get("arm") or "?"
        label = _GBM_ARM_LABELS.get(arm) or g.get("design") or ""
        return f"GBM Focus {arm} — {label}" if label else f"GBM Focus {arm}"
    if fam == "gbm_lag":
        g = m.get("gbm_lag") or {}
        arm = g.get("arm") or "?"
        label = _GBM_LAG_ARM_LABELS.get(arm) or g.get("design") or ""
        return f"GBM Lag {arm} — {label}" if label else f"GBM Lag {arm}"
    if fam == "exit_study":
        label = (m.get("exit_study") or {}).get("label") or ""
        return f"Exit Study — {label}" if label else "Exit Study"
    if fam == "size_study":
        label = (m.get("size_study") or {}).get("label") or ""
        return (f"Size Study 5% — {label}" if label else "Size Study 5%")
    if fam == "composite_screen":
        label = (m.get("composite_screen") or {}).get("label") or ""
        return (f"Composite Screen — {label}" if label
                else "Composite Screen")
    if fam == "tf4h":
        label = (m.get("tf4h") or {}).get("label") or ""
        return f"4H Timeframe — {label}" if label else "4H Timeframe"
    if fam == "regime_data":
        label = (m.get("regime_data") or {}).get("label") or ""
        return f"Regime Data — {label}" if label else "Regime Data"
    if fam == "hierarchy":
        label = (m.get("hierarchy") or {}).get("label") or ""
        return (f"Two-Step Hierarchy — {label}" if label
                else "Two-Step Hierarchy")
    if fam == "features_v6":
        label = (m.get("features_v6") or {}).get("label") or ""
        return f"Features v6 — {label}" if label else "Features v6"
    if fam == "full_indicator_stacker":
        label = (m.get("full_indicator_stacker") or {}).get("label") or ""
        return (f"Full-Indicator Stacker — {label}" if label
                else "Full-Indicator Stacker")
    if fam == "scored_eyes_ppo":
        label = (m.get("scored_eyes_ppo") or {}).get("label") or ""
        return (f"Scored-Eyes PPO — {label}" if label
                else "Scored-Eyes PPO")
    if fam == "muse_pick":
        # Kass (2026-10-26): the challenger Muse picked from his
        # RL/ML sheet is labeled exactly that -- "Muse's Pick
        # Stacker" / "Muse's Pick PPO" -- never a study codename.
        label = (m.get("muse_pick") or {}).get("label") or ""
        return f"Muse's Pick {label}" if label else "Muse's Pick"
    if fam == "muse_pick2":
        # Kass (2026-10-27): the second stab at Muse's own crowned
        # champion is labeled "Muse's Pick #2 Stacker" / "Muse's
        # Pick #2 PPO" -- the #2 is part of the name he ordered.
        label = (m.get("muse_pick2") or {}).get("label") or ""
        return f"Muse's Pick #2 {label}" if label else "Muse's Pick #2"
    if fam == "muse_pick3":
        # Kass's 2026-10-27 delegation (the exposure-manager brain)
        # is labeled "Muse's Pick #3 PPO" the same way. Without
        # this case the index fell back to a nightly name and
        # mislabeled run 75 "Stacker v5.1" (caught on the feed
        # before its first push, 2026-10-27).
        label = (m.get("muse_pick3") or {}).get("label") or ""
        return f"Muse's Pick #3 {label}" if label else "Muse's Pick #3"
    if fam in ("muse_pick4", "muse_pick5", "muse_pick6",
               "muse_pick7", "muse_pick8",
               "muse_pick9", "muse_pick10",
               "muse_pick11", "muse_pick12",
               "muse_pick13", "muse_pick14"):
        # Kass's 2026-11-04 order ("Name the 3 models you did,
        # Muse's Pick #4 #5 #6"): three stacker challengers, each
        # labeled by its study label (e.g. "Stacker — Three
        # Clocks"). Without this case the index would fall back to
        # a nightly name and mislabel the runs. muse_pick7 joined
        # 2026-11-05 (Kass: "try muses pick the next one").
        label = (m.get(fam) or {}).get("label") or ""
        num = fam[len("muse_pick"):]  # "4".."10" (fam[-1] broke at pick10)
        return (f"Muse's Pick #{num} {label}" if label
                else f"Muse's Pick #{num}")
    if fam == "meta_label":
        # Pick A (2026-10-30, Kass's delegated new-system pick):
        # the tradeability gate run is labeled by its study label
        # ("Meta-Label Gate — Pick A").
        label = (m.get("meta_label") or {}).get("label") or ""
        return f"Meta-Label Gate — {label}" if label else "Meta-Label Gate"
    if fam == "eyes_v2":
        # Eyes v2 (2026-10-30, Kass's order): the tradeable-move
        # stacker is labeled by its study label ("Eyes v2 —
        # Tradeable-Move Stacker").
        label = (m.get("eyes_v2") or {}).get("label") or ""
        return f"Eyes v2 — {label}" if label else "Eyes v2"
    if fam == "regime_swing":
        # Regime-Swing (2026-10-30, Kass's full delegation for
        # the last build before the retrain): the 4h regime
        # system is labeled by its study label ("Regime-Swing
        # — 4h Swing").
        label = (m.get("regime_swing") or {}).get("label") or ""
        return f"Regime-Swing — {label}" if label else "Regime-Swing"
    if fam == "ha_deep":
        # HA-Deep Eyes (2026-10-30, Kass's call): the 58-feature
        # Heikin-Ashi stacker is labeled by its study label
        # ("HA-Deep Eyes").
        label = (m.get("ha_deep") or {}).get("label") or ""
        return label or "HA-Deep Eyes"
    if fam == "ha_shift":
        # HA Shift Trader (2026-10-30, Kass's design): the
        # confirmed-shift trader is labeled by its study label
        # ("HA Shift Trader").
        label = (m.get("ha_shift") or {}).get("label") or ""
        return label or "HA Shift Trader"
    if fam == "muse_next_try":
        # Muse's Next Try (2026-11-04, Kass's full delegation in
        # the model-pick chat -- "Label it as ur next try"): the
        # stacker is labeled by its study label ("Muse's Next
        # Try"). Without this case the index fell back to a
        # nightly name and mislabeled the run.
        label = (m.get("muse_next_try") or {}).get("label") or ""
        return label or "Muse's Next Try"
    if fam == "muse_next_try2":
        # Muse's Next Try #2 (2026-11-04, same Kass order): the
        # recipe-selected follow-on to Next Try #1, labeled by its
        # study label ("Muse's Next Try #2").
        label = (m.get("muse_next_try2") or {}).get("label") or ""
        return label or "Muse's Next Try #2"
    base = "PPO" if fam == "ppo" else "Stacker"
    return f"{base} v{version}" if version else f"{base} (early)"


def _model_entry(r: dict, name: str) -> list:
    """One row of the all-runs index: [id, MM-DD, display name, family
    code, promoted, headline, P&L, recorded lab verdict, action mix].
    Headline is the duel aggregate score for PPO-family runs and holdout
    accuracy for the stacker; P&L and verdict are null where the family
    does not record them. Element 9 is the Action Mix section's data
    (Kass, 2026-11-23): see _index_mix -- counts + window total when
    exactly recovered, shares when not, null for stackers and runs
    with no recorded mix."""
    m = r.get("metrics") or {}
    fam = m.get("family") or r["kind"]
    c = m.get("challenger") or {}
    # features_v6 registers one run per brain (kind stacker + kind
    # ppo) under one family: the headline follows the run's own kind,
    # the family code stays 11 for both. full_indicator_stacker runs
    # are all kind=stacker under family code 12 -- same rule.
    headline_kind = (r["kind"] if fam in ("features_v6",
                                          "full_indicator_stacker",
                                          "muse_pick", "muse_pick2",
                                          "muse_pick3", "eyes_v2",
                                          "ha_deep", "muse_next_try",
                                          "muse_next_try2", "muse_pick4",
                                          "muse_pick5", "muse_pick6",
                                          "muse_pick7", "muse_pick8",
                                          "muse_pick9", "muse_pick10",
                                          "muse_pick11", "muse_pick12",
                                          "muse_pick13", "muse_pick14")
                     else fam)
    if headline_kind == "stacker":
        acc = m.get("challenger_acc")
        headline = round(acc, 4) if isinstance(acc, (int, float)) else None
        pnl = None
    else:
        score = c.get("score")
        headline = (_n(round(score, 1))
                    if isinstance(score, (int, float)) else None)
        tp = c.get("total_pnl")
        # Index P&L is whole dollars (2026-10-23 diet): the exact cents
        # stay in the detail rows, the gate feed, and training.db; the
        # all-runs index is a summary line, not the money record.
        pnl = round(tp) if isinstance(tp, (int, float)) else None
    verdict = None
    if fam == "ha_seq":
        h = m.get("ha_seq") or {}
        verdict = ("FAIL (vacuous)" if h.get("vacuous_no_trades")
                   else "PASS" if h.get("gate_pass") else "FAIL")
    elif fam == "gbm_focus":
        verdict = (m.get("gbm_focus") or {}).get("verdict")
    elif fam == "gbm_lag":
        verdict = (m.get("gbm_lag") or {}).get("verdict")
    elif fam == "exit_study":
        verdict = (m.get("exit_study") or {}).get("verdict")
    elif fam == "size_study":
        verdict = (m.get("size_study") or {}).get("verdict")
    elif fam == "composite_screen":
        verdict = (m.get("composite_screen") or {}).get("verdict")
    elif fam == "tf4h":
        verdict = (m.get("tf4h") or {}).get("verdict")
    elif fam == "regime_data":
        verdict = (m.get("regime_data") or {}).get("verdict")
    elif fam == "hierarchy":
        verdict = (m.get("hierarchy") or {}).get("verdict")
    elif fam == "features_v6":
        verdict = (m.get("features_v6") or {}).get("verdict")
    elif fam == "full_indicator_stacker":
        verdict = (m.get("full_indicator_stacker") or {}).get("verdict")
    elif fam == "scored_eyes_ppo":
        verdict = (m.get("scored_eyes_ppo") or {}).get("verdict")
    elif fam == "muse_pick":
        verdict = (m.get("muse_pick") or {}).get("verdict")
    elif fam == "muse_pick2":
        verdict = (m.get("muse_pick2") or {}).get("verdict")
    elif fam == "muse_pick3":
        verdict = (m.get("muse_pick3") or {}).get("verdict")
    elif fam == "meta_label":
        verdict = (m.get("meta_label") or {}).get("verdict")
    elif fam == "eyes_v2":
        verdict = (m.get("eyes_v2") or {}).get("verdict")
    elif fam == "regime_swing":
        verdict = (m.get("regime_swing") or {}).get("verdict")
    elif fam == "ha_deep":
        verdict = (m.get("ha_deep") or {}).get("verdict")
    elif fam == "ha_shift":
        verdict = (m.get("ha_shift") or {}).get("verdict")
    elif fam == "muse_next_try":
        verdict = (m.get("muse_next_try") or {}).get("verdict")
    elif fam == "muse_next_try2":
        verdict = (m.get("muse_next_try2") or {}).get("verdict")
    elif fam in ("muse_pick4", "muse_pick5", "muse_pick6",
                 "muse_pick7", "muse_pick8", "muse_pick9",
                 "muse_pick10", "muse_pick11", "muse_pick12",
                 "muse_pick13", "muse_pick14"):
        verdict = (m.get(fam) or {}).get("verdict")
    return [r["id"], r["ts"][5:10], name, _FAMILY_CODES[fam],
            int(r["promoted"]), headline, pnl, verdict, _index_mix(r)]


def _all_runs_desc() -> list[dict]:
    """ALL registered runs, newest first, read straight from
    training.db. The training snapshot caps its runs list at 30 (a
    display window for the detailed sections), but the models index
    is Kass's complete record -- it must not inherit that cap (the
    index silently dropped the oldest runs once the registry passed
    30, caught by the coverage test at run 36)."""
    import sqlite3

    tdb = ROOT / "data" / "training.db"
    if not tdb.exists():
        return []
    con = sqlite3.connect(tdb)
    con.row_factory = sqlite3.Row
    runs = [
        {"id": r["id"], "ts": r["ts"], "kind": r["kind"],
         "metrics": json.loads(r["metrics"]),
         "promoted": bool(r["promoted"])}
        for r in con.execute("SELECT * FROM runs ORDER BY id DESC")
    ]
    con.close()
    return runs


# Kass's feed-trim directive (2026-11-05): when the Training feed
# needs room, the older FAILED models come out of the feed's models
# index -- runs at or under this id that were never promoted and
# hold no seat or crown. The complete record stays in training.db,
# the Champions & Challengers workbook, and the Model Atlas; only
# the Lab feed's index is trimmed. Runs 1-15 are out of scope (the
# pre-fix set, already display-eliminated since 2026-10-30 and
# joined by id through the pfx flags). Run 44 is kept by id: it is
# the lab PPO champion and the live-seat driver, installed by
# Kass's manual override, so its promoted flag is 0 by design while
# its recorded verdict stays NOT PROMOTED.
_FEED_TRIM_MAX_ID = 50
_FEED_TRIM_KEEP_IDS = {44}


def _models_index_feed(runs_desc: list[dict]) -> tuple[list[list], int]:
    """The models index AS SHIPPED in the training feed: the full
    _models_index minus the failed runs Kass directed out of the
    way (see _FEED_TRIM_MAX_ID). Returns (rows, omitted_count);
    the feed carries the count as models_omitted so the trim reads
    as design, never as silent loss."""
    rows = _models_index(runs_desc)
    promoted = {r["id"] for r in runs_desc if r.get("promoted")}
    kept = [row for row in rows
            if row[0] < 16 or row[0] > _FEED_TRIM_MAX_ID
            or row[0] in promoted or row[0] in _FEED_TRIM_KEEP_IDS]
    return kept, len(rows) - len(kept)


def _models_index(runs_desc: list[dict]) -> list[list]:
    """Every registered run, newest first, with a derived display name.
    Versions inherit forward: a nightly run that attests no version of
    its own ran the recipe in force (the newest explicit marker at or
    before it), which names the v4-era and v5.1-era retrain runs. The
    inheritance walk runs over the PRODUCTION lineage only -- the same
    restriction _current_feat_version applies: lab-family runs attest
    study labels, not feature versions (run 45's record attests "3"
    via its lab note "Matrix v3 composite screen" -- the study's NAME),
    and letting one update the inherited version renamed every later
    nightly run "PPO v3" (defect fixed 2026-10-25; runs 57-60 showed
    v3 while running the v5.1 recipe). Lab runs are named from their
    own family records and never read the inherited version, so
    skipping them here changes only nightly names."""
    notes = {n["ts"]: n for n in _lab_notes_raw()}
    version = None
    named = {}
    for r in reversed(runs_desc):  # ascending: oldest first
        if not (r.get("metrics") or {}).get("family"):
            own = _pick_version(_version_candidates(r, notes.get(r["ts"])))
            if own:
                version = own
        named[r["id"]] = _display_name(r, version)
    return [_model_entry(r, named[r["id"]]) for r in runs_desc]


def _current_feat_version(runs_desc: list[dict]) -> str | None:
    """The feature-recipe version in force, from the registry's own
    attestation -- the same ascending walk the models index names
    runs with (a run that attests no version ran the recipe in
    force), restricted to the PRODUCTION lineage (nightly runs).
    Lab-family runs are excluded on purpose: their records carry
    study labels, not feature versions -- run 45's note attests "3"
    because the study was NAMED "V3 composite screen", and letting
    it into the walk would label today's v5.1 feature set "v3".
    No FEATURE_VERSION constant exists in the feature code, so the
    production run records are the honest source."""
    notes = {n["ts"]: n for n in _lab_notes_raw()}
    version = None
    for r in reversed(runs_desc):  # ascending: oldest first
        if (r.get("metrics") or {}).get("family"):
            continue
        own = _pick_version(_version_candidates(r, notes.get(r["ts"])))
        if own:
            version = own
    return version


def build_recipe(runs_desc: list[dict] | None = None) -> dict:
    """The CURRENT training configuration for the recipe panel (see
    module docstring). Every value is read from the code at build
    time: module constants are imported, and the literals that exist
    only inside nightly.py's main()/gate_decision() are parsed
    strictly -- a pattern that stops matching raises, so a recipe
    change can never leave the panel silently stale."""
    import inspect

    from features.technical_indicators import FEATURE_COLUMNS
    from models import ppo_train
    from trading.broker import FEE_RATE, PaperBroker

    src = (ROOT / "src" / "training" / "nightly.py").read_text()

    def _req(pattern: str, name: str) -> re.Match:
        m = re.search(pattern, src)
        if not m:
            raise ValueError(
                f"recipe source drift: {name} not found in nightly.py")
        return m

    m = _req(r"timesteps = (\d[\d_]*) if deep else (\d[\d_]*)", "PPO steps")
    deep_steps = int(m.group(1).replace("_", ""))
    steps = int(m.group(2).replace("_", ""))
    warm = 1 if _req(r"train_ppo\(\s*timesteps=timesteps, "
                     r"warm_start=(True|False)", "warm-start").group(1) \
        == "True" else 0
    holdout_candles = int(_req(r"split = len\(sup\) - (\d+)",
                               "holdout split").group(1))
    windows = int(_req(r"windows_sup = \[hold_sup\.iloc\[k \* \d+:"
                       r"\(k \+ 1\) \* \d+\] for k in range\((\d+)\)\]",
                       "gate windows").group(1))
    episodes = int(_req(r"def evaluate_ppo\(model_path: Path, sup, p_up, "
                        r"episodes: int = (\d+)", "duel episodes").group(1))
    fills = _req(r'fill_mode="(\w+)"', "fill mode").group(1)
    dd_pen = float(_req(r'"score": total - ([\d.]+) \* max_dd',
                        "drawdown penalty").group(1))
    gate_wins = int(_req(r'\(chall\["score"\] > champ\["score"\] '
                         r'and wins >= (\d+)\)', "gate formula").group(1))
    nonhold = float(_req(r"if non_hold < ([\d.]+):",
                         "floor activity").group(1))
    _req(r"if pnl <= 0\.0:", "floor P&L rule")
    _req(r"if pnl <= bench:", "floor buy-and-hold rule")
    _req(r"stacker_promoted = chall_acc > champ_acc", "stacker rule")
    sig = inspect.signature(PaperBroker.__init__).parameters
    if runs_desc is None:
        runs_desc = _all_runs_desc()
    return {
        "steps": steps,
        "deep_steps": deep_steps,
        "chunk": ppo_train.CHUNK_SIZE,
        "warm": warm,
        "windows": windows,
        "holdout_d": holdout_candles // 24,
        "episodes": episodes,
        "fills": fills,
        "dd_pen": _n(dd_pen),
        "gate_wins": gate_wins,
        "floor": [_n(nonhold), 0, 1],
        "stacker": "strict_win",
        "tranche": sig["max_position_pct"].default,
        "exposure": sig["max_exposure_pct"].default,
        "fee": FEE_RATE,
        "stop": [_n(PaperBroker.STOP_ATR_MULT),
                 PaperBroker.STOP_MIN_PCT, PaperBroker.STOP_MAX_PCT],
        "trail": [PaperBroker.TRAIL_ACTIVATE_PCT,
                  _n(PaperBroker.TRAIL_ATR_MULT),
                  PaperBroker.TRAIL_MIN_PCT, PaperBroker.TRAIL_MAX_PCT],
        "feat_n": len(FEATURE_COLUMNS),
        "feat_v": _current_feat_version(runs_desc),
    }


def _gate_history(runs_desc: list[dict]) -> list[list]:
    """The last 6 recorded gate duels (see module docstring). The
    population mirrors build_gate's: nightly PPO runs only (lab
    studies dueled study baselines, never the crown) where both
    sides recorded a real score -- read from the FULL registry, not
    the snapshot's 30-run window."""
    out: list[list] = []
    for r in runs_desc:  # newest first
        if r["kind"] != "ppo":
            continue
        m = r.get("metrics") or {}
        if m.get("family"):
            continue
        c_raw, h_raw = m.get("challenger") or {}, m.get("champion") or {}
        c, h = _gate_side(c_raw), _gate_side(h_raw)
        if c is None or h is None or "win" not in c or "win" not in h:
            continue
        won = sum(1 for a, b in zip(c["win"], h["win"]) if a > b)
        fpass = 1 if (c["s"] > h["s"] and won >= 2) else 0
        recorded = 1 if r["promoted"] else 0
        # The floor only decides when the formula passed; nightly
        # promotes solely through gate_decision, so then the
        # recorded promotion flag IS the floor's verdict. A failed
        # formula leaves the floor unreached (null).
        floor = (1 if recorded else 0) if fpass else None
        bench = c_raw.get("bench_pnl")
        row = [r["id"], r["ts"][5:10], won,
               _n(c["s"]), round(c_raw["total_pnl"]),
               _n(h["s"]), round(h_raw["total_pnl"]),
               round(bench) if isinstance(bench, (int, float)) else None,
               fpass, floor,
               "PROMOTED" if recorded else "NOT PROMOTED"]
        if floor == 0:
            fm = re.search(r"floor: ([^;]+)", str(m.get("reason") or ""))
            if fm:
                row.append(fm.group(1).strip())
        out.append(row)
        if len(out) == 6:
            break
    return out


def _seat_manifest() -> dict | None:
    """The driver's-seat manifest (data/live_seat.json): which runs'
    artifacts live trading actually runs, seeded at the 2026-10-25
    separation and updated only by Kass's explicit driver pick (the
    manual procedure lives in models/live_seat/README.md)."""
    path = ROOT / "data" / "live_seat.json"
    if not path.exists():
        return None
    try:
        man = json.loads(path.read_text())
        return man if isinstance(man, dict) else None
    except Exception:
        return None


def build_champions(runs_desc: list[dict] | None = None) -> dict | None:
    """The Champions roster for the Training Lab (seat separation,
    Kass 2026-10-25: "Keep the trading and training separate. I want
    to be able to see all the champions and numbers myself and then
    tell you which one I want in the driver seat of live trading").

    The Lab holds the crowns; the driver's seat is a separate slot
    that changes only on Kass's pick. Sources, all records (never
    recomputed): the lab stacker champion's identity + feature set
    from its metadata sidecar (models/stacker_champion.json), every
    grade from the registry run that earned it, the seat occupants
    and the PPO champion's run id from the seat manifest (the PPO
    champion has no metadata sidecar; the manifest is its ledger --
    run 44 was seated by Kass's 2026-10-24 manual override and the
    manifest records that lineage). None before the separation
    exists (no manifest / no run-tagged champion metadata).

    Shape: {"seat": {"ppo_i", "stk_i", "stk_feat"},
            "stk": [[no, run_id, name, feature_set, holdout_acc,
                     wf_acc, precision, status], ...],
            "ppo": [[no, run_id, name, duel_score, duel_pnl,
                     status], ...]}
    Rows are in coronation order (numbered from 1). Status codes:
    1 = current lab champion; 2 = former lab champion, still in
    the driver's seat; 3 = current lab champion AND in the
    driver's seat; 4 = former lab champion, in no seat (added at
    the Champion #3 crowning, 2026-10-26: run 64 was displaced by
    run 69 and sits in neither seat -- the in-seat status 2 would
    mislabel it). Former champions beyond the seat stacker are
    the stacker runs carrying Kass's recorded crowning decision
    (metrics.pending_decision == "approved", the pattern of runs
    64 and 69) that no longer hold the crown. Names come from the
    models index, so the roster can never disagree with the
    all-runs list."""
    man = _seat_manifest()
    if man is None:
        return None
    champ_file = ROOT / "models" / "stacker_champion.json"
    if not champ_file.exists():
        return None
    try:
        champ_meta = json.loads(champ_file.read_text())
    except Exception:
        return None
    lab_stk_run = champ_meta.get("run_id")
    if not isinstance(lab_stk_run, int):
        return None
    if runs_desc is None:
        runs_desc = _all_runs_desc()
    by_id = {r["id"]: r for r in runs_desc}
    names = {row[0]: row[2] for row in _models_index(runs_desc)}
    seat = man.get("seat") or {}
    seat_stk = seat.get("stacker") or {}
    seat_ppo = seat.get("ppo") or {}
    seat_stk_run = seat_stk.get("run_id")

    def _stk_grades(rid: int):
        m = (by_id.get(rid) or {}).get("metrics") or {}

        def _r(key):
            v = m.get(key)
            return round(v, 4) if isinstance(v, (int, float)) else None

        return _r("challenger_acc"), _r("challenger_wf_acc"), \
            _r("precision")

    stk_rows: list[list] = []
    if isinstance(seat_stk_run, int) and seat_stk_run != lab_stk_run:
        acc, wf, prec = _stk_grades(seat_stk_run)
        stk_rows.append([len(stk_rows) + 1, seat_stk_run,
                         names.get(seat_stk_run),
                         seat_stk.get("feature_set") or "prod21",
                         acc, wf, prec, 2])

    def _former_feat(m: dict) -> str:
        """A former champion's feature-set tag, from its own
        recorded study block (94 = the scored102 frame of
        Champions #2/#3; 98 = Pick #2's extension of it, the
        pick2_98 frame of Champion #4). The HA champions carry
        their frame in their record directly: Champion #6
        (run 110) by its muse_pick11 family tag, the nightly
        HA-line champions (run 111 on) by their recorded reason
        (added 2026-11-06 at the Champion #7 crowning, when the
        fallback below mislabeled run 110 as prod21)."""
        if m.get("feature_set") == "ha10" or \
                m.get("family") in ("muse_pick10", "muse_pick11"):
            return "ha10"
        if str(m.get("reason") or "").startswith("HA stacker challenger"):
            return "ha10"
        for blk in m.values():
            if isinstance(blk, dict):
                nf = blk.get("n_features")
                feats = blk.get("features")
                n = nf if isinstance(nf, int) else (
                    len(feats) if isinstance(feats, list) else None)
                if n == 94:
                    return "scored102"
                if n == 98:
                    return "pick2_98"
        return "prod21"

    former = [r for r in runs_desc
              if r["kind"] == "stacker"
              and (r.get("metrics") or {}).get(
                  "pending_decision") == "approved"
              and r["id"] not in (lab_stk_run, seat_stk_run)]
    for r in sorted(former, key=lambda r: r["id"]):
        acc, wf, prec = _stk_grades(r["id"])
        stk_rows.append([len(stk_rows) + 1, r["id"],
                         names.get(r["id"]),
                         _former_feat(r.get("metrics") or {}),
                         acc, wf, prec, 4])
    acc, wf, prec = _stk_grades(lab_stk_run)
    stk_rows.append([len(stk_rows) + 1, lab_stk_run,
                     names.get(lab_stk_run),
                     champ_meta.get("feature_set") or "prod21",
                     acc, wf, prec,
                     3 if seat_stk_run == lab_stk_run else 1])
    ppo_rows: list[list] = []
    lab_ppo_run = man.get("ppo_champion_run_id")
    if isinstance(lab_ppo_run, int):
        m = (by_id.get(lab_ppo_run) or {}).get("metrics") or {}
        c = m.get("challenger") or {}
        score, pnl = c.get("score"), c.get("total_pnl")
        ppo_rows.append([1, lab_ppo_run, names.get(lab_ppo_run),
                         _n(round(score, 1))
                         if isinstance(score, (int, float)) else None,
                         round(pnl, 2)
                         if isinstance(pnl, (int, float)) else None,
                         3 if seat_ppo.get("run_id") == lab_ppo_run
                         else 1])
    return {
        "seat": {"ppo_i": seat_ppo.get("run_id"),
                 "stk_i": seat_stk_run,
                 "stk_feat": seat_stk.get("feature_set") or "prod21"},
        "stk": stk_rows,
        "ppo": ppo_rows,
    }


def _fix_boundary_ts(runs_desc: list[dict]) -> str | None:
    """When the v5.1 look-ahead fix went live, derived from the
    registry itself: the earliest production-lineage run whose own
    record attests recipe version 5.1 (run 16 -- its lab note titles
    the migration, and its recorded holdout 0.4596 is the first clean
    grade). No hard-coded date: if the registry ever loses the
    attestation, the boundary is None and nothing is flagged."""
    notes = {n["ts"]: n for n in _lab_notes_raw()}
    stamps = []
    for r in runs_desc:
        if (r.get("metrics") or {}).get("family"):
            continue  # lab runs attest study labels, not recipes
        if _pick_version(_version_candidates(
                r, notes.get(r["ts"]))) == "5.1":
            stamps.append(r["ts"])
    return min(stamps) if stamps else None


def _prefix_run_ids(runs_desc: list[dict]) -> list[int]:
    """Ids of runs graded BEFORE the v5.1 fix: their multi-timeframe
    features still carried future data, so those grades are history,
    not edge (run 10's 0.6364, run 15's +7,513.9). The dashboard
    flags them PRE-FIX ERA -- family bests, index rows, detail
    cards -- joined by run id."""
    boundary = _fix_boundary_ts(runs_desc)
    if boundary is None:
        return []
    return sorted(r["id"] for r in runs_desc if r["ts"] < boundary)


def _declined_run_ids(runs_desc: list[dict]) -> list[int]:
    """Ids of runs Kass decided against under the promotion lock:
    the registry records the decision on the run itself
    (metrics.pending_decision), and the dashboard must not keep
    reading those rows as plain PASSes."""
    return sorted(r["id"] for r in runs_desc
                  if (r.get("metrics") or {}).get(
                      "pending_decision") == "declined")


# Kass's own indicator set (features v5, 2026-10-20): the five
# features in the prod21 set that came from his Drive library.
KASS_FEATURES = ["ha_dir", "obv_slope", "adl_slope", "netvol_rel", "ewo"]

# Short plain-English roster names for the prod21 features (the
# Training Lab's "what the brains read" panel; the full descriptive
# labels live in FEATURE_LABELS for the Live desk's vision panel).
PROD21_SHORT = {
    "body_pct": "Candle body",
    "upper_wick": "Upper wick",
    "lower_wick": "Lower wick",
    "range_pct": "Candle range",
    "rsi_c": "RSI - 1h momentum",
    "macd_hist_rel": "MACD histogram",
    "dist_vwap": "Distance from VWAP",
    "dist_sma50": "Distance from 50h average",
    "atr_rel": "ATR - hourly volatility",
    "price_change": "Last hour's price change",
    "volatility_21": "Volatility - last 21h",
    "volume_change": "Volume change vs prior hour",
    "h4_rsi_c": "RSI - 4h momentum",
    "h4_dist_sma": "Distance from 4h average",
    "d1_dist_sma": "Distance from daily average",
    "h4_atr_rel": "ATR - 4h volatility",
    "ha_dir": "Heikin-Ashi direction",
    "obv_slope": "OBV slope",
    "adl_slope": "A/D slope",
    "netvol_rel": "Net volume share",
    "ewo": "Elliott Wave Oscillator",
}

# Kass's five families and the name patterns that identify their
# counterparts in Champion #2's scored indicator set. Presence is
# DERIVED from the champion run's own recorded feature list below,
# never assumed: a family with no scored counterpart lands in kout.
_KASS_SCORED = [
    ("Heikin-Ashi", ("heikin",)),
    ("OBV", ("on-balance volume", "(obv)")),
    ("A/D line", ("accumulation/distribution",)),
    ("Net volume", ("net volume",)),
    ("Elliott Wave Osc.", ("elliott",)),
]


def _reads_block(runs_desc: list[dict]) -> dict:
    """What each brain reads (Training Lab panel, Kass 2026-10-26:
    "shouldn't we be having [Kass's indicators] in the training lab
    as well, to see how the models/bots are using them when they
    train"). p21 = the prod21 roster every production brain reads
    (seat stacker, nightly stacker challengers, and the PPO -- which
    additionally reads the stacker's vote and its own book), with
    Kass's five flagged. c2 = Champion #2's recorded study data
    from its registry record: input count, composite-score count,
    and the ranked top-15 feature list stored at grading time --
    NAMES IN RECORDED ORDER ONLY: the registry stored no numeric
    importance weights, so the feed carries none and the panel must
    not draw bar lengths. Kass-family presence in the scored set is
    matched against the run's recorded feature list. c2 is null
    when no champion record carries study data."""
    from features.technical_indicators import FEATURE_COLUMNS

    p21 = [[c, PROD21_SHORT[c], 1 if c in KASS_FEATURES else 0]
           for c in FEATURE_COLUMNS]
    c2 = None
    champ_file = ROOT / "models" / "stacker_champion.json"
    meta = None
    if champ_file.exists():
        try:
            meta = json.loads(champ_file.read_text())
        except Exception:
            meta = None
    rid = (meta or {}).get("run_id")
    rec = next((r for r in runs_desc if r["id"] == rid), None)
    rec_metrics = (rec or {}).get("metrics") or {}
    # The reigning champion's recorded study block, under whichever
    # family key its study used: run 64's full-indicator study,
    # run 69's Muse's Pick study, or run 73's Muse's Pick #2
    # study (the scored frame and its Pick #2 extension, same
    # fields).
    fi = rec_metrics.get("full_indicator_stacker") or rec_metrics.get(
        "muse_pick") or rec_metrics.get("muse_pick2")
    if fi and fi.get("top_features"):
        feats = fi.get("features") or (meta or {}).get(
            "feature_columns") or []
        low = [str(f).lower() for f in feats]
        kin, kout = [], []
        for label, pats in _KASS_SCORED:
            hit = any(any(p in f for p in pats) for f in low)
            (kin if hit else kout).append(label)
        c2 = {"run": rid,
              "n": fi.get("n_features") or len(feats),
              "comp": sum(1 for f in feats
                          if str(f).startswith("score_")),
              "top": [str(x) for x in fi["top_features"][:15]],
              "kin": kin, "kout": kout}
    return {"p21": p21, "c2": c2}


def build_compact_training() -> dict:
    # SIZE DIET (2026-10-23): the feed hit 7,925B at 36 registered runs
    # against the old 8,192B test ceiling, growing ~120B/night (two new
    # models-index rows per retrain). Cuts, none of which remove a run,
    # a name, a verdict, or a headline from the dashboard: the detailed
    # runs window is the newest 6 (the models index below remains the
    # complete record), index P&L is whole dollars, action mixes are
    # 2-decimal, integral floats serialize as ints, note/reason
    # previews are slightly tighter, and reward_series is capped at the
    # newest 20 points. Dieted size 6,679B at 36 runs; the models index
    # is the only unbounded section (rows average ~58B; the nightly
    # stacker+PPO rows ~44B), so the feed grows ~90-118B/night and
    # stays under the 16,384B budget (2x the largest payload this
    # dashboard has verifiably accepted -- 7,925B pushed and read back
    # intact 2026-10-23; 6,433B and 3,307B before it; no push to this
    # dashboard has ever been rejected) for 60+ nights (~13.8KB even
    # at the worst-case row rate).
    # 2026-10-25 upgrade (Kass: see/compare everything tweakable): the
    # recipe panel (+309B, fixed size) and gate history (+426B, fixed
    # 6 rows) join the feed, and detail runs carry the settings the
    # registry recorded (~+281B in the current window) -- 8,642B at
    # 60 runs, sentinel re-baselined to 8,800B in
    # tests/test_compact_feeds.py with the basis recorded there.
    # 2026-11-05 diet: the feed hit 17,776B at 110 registered runs --
    # over the 16,384B budget -- after the Pick #7-#11 study burst put
    # five long stacker records in the detail window and the run-110
    # crowning added a 1,031-char lab note. Cuts, same doctrine as
    # 2026-10-23 (no run, name, verdict, headline, roster entry, or
    # gate fact leaves the dashboard): the detail window is the newest
    # 4 (the models index below remains the complete record; runs
    # 105/106 keep their index rows and run 105's duel stays in
    # gate_hist) and the lab-notes window is the newest 3 (every
    # shipped note still carries its full text when trimmed). Dieted
    # size 16,089B at 110 runs; sentinel re-baselined to 16,200B in
    # tests/test_compact_feeds.py with the basis recorded there.
    s = snapshot.build_training_snapshot()
    runs = [_duel_run(r) for r in s["runs"][:4]]
    ch = s.get("champion") or {}
    # reward_series and rcurve read the FULL registry, not the
    # snapshot's 30-run window: the window silently dropped older PPO
    # runs from the learning chart once the registry passed 30 (14
    # points shown where 20 are recorded -- 2026-10-24 audit; same
    # cap class as the models-index fix below).
    registry = _all_runs_desc()
    reward_series = [
        {"t": r["ts"], "r": round((r["metrics"]["challenger"]["mean_reward"]), 3)}
        for r in reversed(registry)
        if r["kind"] == "ppo"
        and isinstance((r.get("metrics") or {}).get("challenger", {}).get("mean_reward"), (int, float))
    ][-20:]
    # Learning curve of the newest PPO run that recorded one (v4).
    # Diet (2026-10-25): a 300k deep run records ~288 curve points
    # (~4.1KB) -- when run 66 landed the feed hit 13,706B. The chart
    # gets a stride-downsample to <= 64 points, first and last
    # recorded points kept exactly; every shown point is a recorded
    # point (no interpolation, no invented values).
    rcurve = None
    for r in registry:
        if r["kind"] == "ppo":
            rc = (r.get("metrics") or {}).get("reward_curve")
            if rc:
                rcurve = [[p[0], _n(round(p[1], 2))]
                          if isinstance(p[1], (int, float)) else p
                          for p in rc]
                if len(rcurve) > 64:
                    stride = -(-(len(rcurve) - 1) // 62)
                    sampled = rcurve[::stride]
                    if sampled[-1] != rcurve[-1]:
                        sampled.append(rcurve[-1])
                    rcurve = sampled
                break
    _feed_models, _feed_omitted = _models_index_feed(_all_runs_desc())
    return {
        "as_of": s["as_of"],
        "runs": runs,
        # PAGINATION (2026-11-17 size fix): the models index no
        # longer rides inside this snapshot. It was the feed's one
        # unbounded section (~55B/row, ~90-118B/night forever);
        # the snapshot now carries only the counts, and the full
        # index ships via build_models_pages() + the dashboard's
        # push_models_page action (see MODELS_PAGE_SIZE below).
        # The index itself is NOT trimmed beyond Kass's
        # 2026-11-05 directive (_models_index_feed): every kept
        # row still reaches the desk, one page at a time.
        "models_total": len(_feed_models),
        "models_omitted": _feed_omitted,
        "champion": {
            "accuracy": round(ch["accuracy"], 4),
            "precision": round(ch["precision"], 4),
            "folds": ch.get("folds"),
        } if ch else None,
        "reward_series": reward_series,
        "rcurve": rcurve,
        "notes": _lab_notes(),
        "next_retrain": s["next_retrain"],
        "recipe": build_recipe(registry),
        "gate_hist": _gate_history(registry),
        "champs": build_champions(registry),
        # Truth-in-labeling (2026-10-26): two run-id sets the
        # dashboard joins onto the models index, detail cards, and
        # family bests -- grades from before the v5.1 look-ahead
        # fix (pfx) and runs Kass declined under the lock (dec).
        # ~70B total, versus ~10B per flagged row inline.
        "pfx": _prefix_run_ids(registry),
        "dec": _declined_run_ids(registry),
        # What each brain reads (2026-10-26, Kass): the prod21
        # roster with his five indicators flagged + Champion #2's
        # recorded study data. Fixed-size block (~1.5KB); see
        # _reads_block and the size sentinel's re-baseline note.
        "reads": _reads_block(registry),
        # Overfitting auditor + PPO telemetry (2026-10-26 research
        # incorporation): two fixed-size blocks (~450B together).
        "audit": build_audit(registry),
        "telemetry": _telemetry_block(registry),
        # Phase 3 (2026-10-31): the champion stacker's stored
        # holdout vote distribution and the seat PPO's per-layer
        # weight stats -- both derived from stored artifacts only
        # (see the builders); either is None (key present, value
        # null) when its source no longer describes the reigning
        # champion / seat, never stale data under a current label.
        "votes": _votes_block(),
        "weights": _policy_weights_block(),
    }


# Models-index pagination (2026-11-17 size fix): the Training
# Lab's models index is Kass's complete record -- the full list
# IS the point, so it is never sliced to a tail. Instead it ships
# in fixed-size pages through the dashboard's push_models_page
# action ({page, pages, total, omitted, rows}); the desk assembles
# the full index from the stored pages. 40 rows/page at ~78B/row
# is ~3.1KB per push -- comfortably under the action-arg ceiling
# class this dashboard has always accepted (its largest verified
# snapshot acceptance is 7,925B) -- and the page COUNT is the only
# thing that grows with the registry (300 runs = 8 pages), so no
# single push ever grows again. (Row size rose from ~55B when the
# 9th element -- the action mix, Kass 2026-11-23 -- joined the row;
# the <4000B page test below is the guard.)
MODELS_PAGE_SIZE = 40


def build_models_pages(runs_desc: list[dict] | None = None) -> list[dict]:
    """The full feed models index as a list of page payloads for
    push_models_page, in page order (page 0 = newest runs). Each
    page carries the same total/omitted so the desk can assemble
    and label the complete index from any complete page set."""
    if runs_desc is None:
        runs_desc = _all_runs_desc()
    rows, omitted = _models_index_feed(runs_desc)
    pages = max(1, -(-len(rows) // MODELS_PAGE_SIZE))
    out = []
    for i in range(pages):
        out.append({
            "page": i,
            "pages": pages,
            "total": len(rows),
            "omitted": omitted,
            "rows": rows[i * MODELS_PAGE_SIZE:(i + 1) * MODELS_PAGE_SIZE],
        })
    return out


def build_models_manifest() -> dict:
    """Page-count manifest for the pushing jobs: how many
    push_models_page calls the current registry needs, so a job
    can loop `training_models_page <i>` for i in range(pages)
    without parsing a payload."""
    pages = build_models_pages()
    return {"pages": len(pages),
            "total": pages[0]["total"] if pages else 0,
            "omitted": pages[0]["omitted"] if pages else 0,
            "page_size": MODELS_PAGE_SIZE}


def build_audit(runs_desc: list[dict]) -> dict:
    """The Overfitting Auditor block (2026-10-26 research
    incorporation; design doc Part I): the trial ledger + a
    best-of-N context for the reigning champions.

    {"n": total graded runs,
     "ppo": [tries, promoted, declined_by_kass, pending_kass],
     "stk": [same for stacker],
     "cs": [stacker champion run, its holdout accuracy, median
            challenger accuracy, best-of-the-rest accuracy],
     "cp": [PPO champion run, its duel score, median challenger
            duel score, best-of-the-rest duel score]}

    Counts cover the FULL registry (pre-fix runs included: they
    consumed research degrees of freedom, so the multiple-testing
    count keeps them). The score DISTRIBUTIONS (median / best of
    the rest) cover post-v5.1-fix runs only -- pre-fix grades were
    earned with leaked multi-timeframe features and are not
    comparable draws. No Deflated Sharpe Ratio is computed: the
    registry stores duel totals and holdout accuracies, not the
    per-period return series (nor per-trial Sharpe ratios) the
    DSR formula requires; the method note lives in
    refs/dashboard-incorporation-2026-10-26.md."""
    import statistics

    pfx = set(_prefix_run_ids(runs_desc))
    champs = build_champions(runs_desc)
    pend = _pending_promotion(runs_desc)

    def _counts(kind: str) -> list:
        rs = [r for r in runs_desc if r["kind"] == kind]
        return [len(rs),
                sum(1 for r in rs if r["promoted"]),
                sum(1 for r in rs
                    if (r.get("metrics") or {}).get(
                        "pending_decision") == "declined"),
                1 if pend is not None and pend["k"] == kind else 0]

    def _ctx(kind: str, champ_id, extract, nd: int) -> list:
        pairs = []
        for r in runs_desc:
            if r["kind"] != kind or r["id"] in pfx:
                continue
            v = extract(r)
            if v is not None:
                pairs.append((r["id"], v))
        allv = [v for _i, v in pairs]
        rest = [v for i, v in pairs if i != champ_id]
        champ_v = next((v for i, v in pairs if i == champ_id), None)
        return [champ_id,
                round(champ_v, nd) if champ_v is not None else None,
                round(statistics.median(allv), nd) if allv else None,
                round(max(rest), nd) if rest else None]

    def _stk_acc(r):
        v = (r.get("metrics") or {}).get("challenger_acc")
        return float(v) if isinstance(v, (int, float)) else None

    def _ppo_score(r):
        c = (r.get("metrics") or {}).get("challenger") or {}
        v = c.get("score")
        # The -1e9 sentinel marks an un-runnable side, not a grade.
        return float(v) if isinstance(v, (int, float)) and v > -1e8 \
            else None

    stk_champ = champs["stk"][-1][1] if champs and champs.get("stk") \
        else None
    ppo_champ = champs["ppo"][0][1] if champs and champs.get("ppo") \
        else None
    return {"n": len(runs_desc),
            "ppo": _counts("ppo"), "stk": _counts("stacker"),
            "cs": _ctx("stacker", stk_champ, _stk_acc, 4),
            "cp": _ctx("ppo", ppo_champ, _ppo_score, 1)}


def _telemetry_block(runs_desc: list[dict]) -> dict:
    """PPO training telemetry for the Lab panel (2026-10-26):
    {"from": first run id carrying telemetry | None,
     "rows": [[run_id, ent_start, ent_end, ent_min, kl_end,
               clip_end, ev_end, vl_end], ...]} -- newest first,
    at most 4 runs, values read VERBATIM from the run records'
    "telemetry" (the logging-only SB3 capture in ppo_train,
    lifted by nightly). Runs graded before the capture shipped
    have no telemetry key and never appear here: the panel reads
    their cells as "Not recorded", with no reconstruction."""
    rows: list[list] = []
    first = None
    for r in reversed(runs_desc):
        if r["kind"] == "ppo" and isinstance(
                (r.get("metrics") or {}).get("telemetry"), dict):
            first = r["id"]
            break
    for r in runs_desc:  # newest first
        if r["kind"] != "ppo":
            continue
        t = (r.get("metrics") or {}).get("telemetry")
        if not isinstance(t, dict):
            continue

        def _g(key, _t=t):
            v = _t.get(key)
            return round(v, 4) if isinstance(v, (int, float)) else None

        rows.append([r["id"], _g("ent_start"), _g("ent_end"),
                     _g("ent_min"), _g("kl_end"), _g("clip_end"),
                     _g("ev_end"), _g("vl_end")])
        if len(rows) == 4:
            break
    return {"from": first, "rows": rows}


def _votes_block() -> dict | None:
    """Stacker champion vote distribution: {"run", "sha" (champion
    artifact sha256[:8]), "src" ("grading" | "study"), "n", "from",
    "to", "bins" (20 bins over [0, 1]), "min", "max", "med",
    "mid" (% of votes within 0.45-0.55), "tight" (% within
    0.48-0.52)}. Preference order (grading-time vote logging,
    Kass-approved 2026-10-31): (1) the reigning champion's OWN
    grading-stored summary -- since 2026-10-31 the nightly grader
    persists metrics["votes"] on every stacker run it grades
    (src/training/nightly.py vote_histogram), so a champion crowned
    from a graded challenger carries its holdout vote histogram in
    its registry record; serve that, stamped with the run's grading
    timestamp; (2) the meta-label study's stored array (below),
    which is run 73's only vote record -- its grading predates the
    logging. Self-invalidating in both modes: a champion whose
    record carries no stored votes and no matching study array gets
    no block at all, never a former champion's votes. The feed
    never re-runs a brain. Any failure returns None: a display
    extra must never break the feed."""
    block = _grading_votes_block()
    if block is not None:
        return block
    return _study_votes_block()


def _grading_votes_block() -> dict | None:
    """The reigning lab stacker champion's grading-stored vote
    summary (metrics["votes"] on its own registry record, written
    by the nightly grader at grading time). Identity comes from
    models/stacker_champion.json's run_id -- the same record the
    champions roster reads -- and the summary is served verbatim
    (shape-validated: 20 bins summing to n) with the run's grading
    timestamp in from/to. None when the champion's record predates
    vote logging or carries no summary."""
    try:
        import hashlib

        meta = json.loads(
            (ROOT / "models" / "stacker_champion.json").read_text())
        run_id = meta.get("run_id")
        if not isinstance(run_id, int):
            return None
        run = next((r for r in _all_runs_desc()
                    if r["id"] == run_id), None)
        if run is None or run["kind"] != "stacker":
            return None
        vh = (run.get("metrics") or {}).get("votes")
        if not isinstance(vh, dict):
            return None
        bins = vh.get("bins")
        if (not isinstance(bins, list) or len(bins) != 20
                or not all(isinstance(x, int) for x in bins)):
            return None
        n = vh.get("n")
        if not isinstance(n, int) or n <= 0 or sum(bins) != n:
            return None
        stats = {}
        for key in ("min", "max", "med"):
            x = vh.get(key)
            if not isinstance(x, (int, float)):
                return None
            stats[key] = _n(round(float(x), 4))
        for key in ("mid", "tight"):
            x = vh.get(key)
            if not isinstance(x, (int, float)):
                return None
            stats[key] = _n(round(float(x), 1))
        sha = hashlib.sha256(
            (ROOT / "models" / "stacker_champion.pkl")
            .read_bytes()).hexdigest()
        ts = str(run["ts"])[:16]
        return {
            "run": run_id,
            "sha": sha[:8],
            "src": "grading",
            "n": n,
            "from": ts,
            "to": ts,
            "bins": [int(x) for x in bins],
            "min": stats["min"],
            "max": stats["max"],
            "med": stats["med"],
            "mid": stats["mid"],
            "tight": stats["tight"],
        }
    except Exception:
        return None


def _study_votes_block() -> dict | None:
    """The meta-label study's stored vote array (Phase 3,
    2026-10-31): a histogram + summary of the per-candle holdout
    probabilities STORED at data/meta_label/votes_hold.npy, which
    scripts/run_meta_label_study.py generated with predict_proba
    from models/stacker_champion.pkl over its 792-candle holdout
    frame (2026-09-27 16:00 -> 2026-10-30 15:00 UTC). Emitted ONLY
    while the live champion artifact's sha256 still equals the sha
    the study pinned in its results.json AND the champion metadata
    names the same run: the moment a new stacker is crowned, the
    stored votes describe a former champion and the block
    disappears rather than going stale."""
    try:
        import hashlib

        import numpy as np

        lab = ROOT / "data" / "meta_label"
        study = json.loads((lab / "results.json").read_text())
        champ = study["frames"]["champion"]
        meta = json.loads(
            (ROOT / "models" / "stacker_champion.json").read_text())
        sha = hashlib.sha256(
            (ROOT / "models" / "stacker_champion.pkl")
            .read_bytes()).hexdigest()
        if (sha != champ["stacker_sha256"]
                or meta.get("run_id") != champ["stacker_run"]):
            return None
        votes = np.load(lab / "votes_hold.npy")
        frames = study["frames"]
        if len(votes) != frames["holdout_rows"]:
            return None
        bins, _ = np.histogram(votes, bins=20, range=(0.0, 1.0))
        return {
            "run": champ["stacker_run"],
            "sha": sha[:8],
            "src": "study",
            "n": int(len(votes)),
            "from": frames["holdout_first"][:16],
            "to": frames["holdout_last"][:16],
            "bins": [int(x) for x in bins],
            "min": _n(round(float(votes.min()), 4)),
            "max": _n(round(float(votes.max()), 4)),
            "med": _n(round(float(np.median(votes)), 4)),
            "mid": _n(round(100.0 * float(
                np.mean((votes >= 0.45) & (votes <= 0.55))), 1)),
            "tight": _n(round(100.0 * float(
                np.mean((votes >= 0.48) & (votes <= 0.52))), 1)),
        }
    except Exception:
        return None


_POLICY_LAYER_SPEC = [
    ("mlp_extractor.policy_net.0.weight", "pi1"),
    ("mlp_extractor.policy_net.2.weight", "pi2"),
    ("mlp_extractor.value_net.0.weight", "vf1"),
    ("mlp_extractor.value_net.2.weight", "vf2"),
    ("action_net.weight", "act"),
    ("value_net.weight", "val"),
]


def _policy_weights_block() -> dict | None:
    """Live-seat PPO policy weight statistics (Phase 3, 2026-10-31):
    {"run", "sha" (seat artifact sha256[:8]),
    "layers": [[name, mean, std, min, max], ...]} -- per-layer
    weight stats (population std) of the driver's-seat PPO
    artifact (models/live_seat/ppo.zip; run id + expected sha from
    data/live_seat.json), read straight out of the zip's
    policy.pth via torch.load -- no stable-baselines3 model
    construction, no environment, no inference; read-only.
    Cost: one lazy torch import + the state-dict read (~3s),
    paid only by the training feed build (hourly / retrain
    pushes); no other feed imports torch. Emitted only while the
    file's sha256 matches the seat manifest; any failure returns
    None so a display extra can never break the feed."""
    try:
        import hashlib
        import io
        import zipfile

        import torch

        seat = json.loads((ROOT / "data" / "live_seat.json").read_text())
        ppo = seat["seat"]["ppo"]
        path = ROOT / "models" / "live_seat" / "ppo.zip"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != ppo["sha256"]:
            return None
        with zipfile.ZipFile(path) as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")),
                            map_location="cpu", weights_only=True)
        layers = []
        for key, short in _POLICY_LAYER_SPEC:
            f = sd[key].double()
            layers.append([short,
                           _n(round(float(f.mean()), 4)),
                           _n(round(float(f.std(unbiased=False)), 4)),
                           _n(round(float(f.min()), 4)),
                           _n(round(float(f.max()), 4))])
        return {"run": ppo["run_id"], "sha": sha[:8], "layers": layers}
    except Exception:
        return None


def _gate_side(d: dict | None) -> dict | None:
    """One side of a PPO duel in gate-feed shape, or None when that side
    recorded no real score. Migration-seated champions carry a -1e9
    sentinel (the old champion could not run the new recipe), which is
    NOT a grade and must never reach the dashboard as one."""
    if not isinstance(d, dict):
        return None
    s = d.get("score")
    if not isinstance(s, (int, float)) or s <= -1e8:
        return None
    out: dict = {"s": round(s, 1)}
    wins = [w.get("score") for w in d.get("windows") or []]
    if len(wins) == 3 and all(isinstance(w, (int, float)) for w in wins):
        out["win"] = [round(w, 1) for w in wins]
    if isinstance(d.get("total_pnl"), (int, float)):
        out["pl"] = round(d["total_pnl"], 2)
    if isinstance(d.get("max_drawdown"), (int, float)):
        out["dd"] = round(d["max_drawdown"], 2)
    return out


def _clause_bounded(text: str, limit: int = 200, *,
                    word_fallback: bool = False,
                    ellipsis: bool = False) -> str:
    """A stored reason shortened for a compact feed WITHOUT ever
    cutting mid-word or mid-clause (2026-10-30 dashboard audit: the
    gate pending.why was sliced at [:100] and arrived on both
    dashboards as "...P&L +237 vs " -- the champion comparison it
    referenced never made it). Text at/under the limit rides whole.
    Longer text is trimmed at the LAST clause boundary at or before
    the limit (a ";" clause separator or an " -- " clause break,
    delimiter excluded); if no boundary fits inside the limit, the
    text is extended to the FIRST boundary past it -- completeness
    beats the cap -- and if the text has no clause boundary at all
    it rides whole, because any cut would be mid-clause. The limit
    (200) is sized for the gate feed's <1,024B test budget: run 83's
    stored 566-char reason trims at its 198-char clause boundary,
    keeping the feed ~690B.

    word_fallback (2026-10-31, training-feed previews): when NO
    clause boundary fits inside the limit, trim at the last WORD
    boundary inside the limit instead of extending past the cap.
    The training previews carry tighter caps (run why 100, run cfg
    80, note detail 90) inside a size-budgeted feed, so extending
    past the cap is not free there the way it is in the <1,024B
    gate feed -- run 87's cfg would otherwise ride 153 chars
    against its 80 cap. A word-boundary cut still never lands
    mid-word or mid-number (the 2026-10-31 review found the raw
    slices ending "...sub-cha", "...deterministic swi" and a note
    preview ending "...exam acc 0.5"). The gate keeps the default
    (word_fallback=False): its stored reasons are verdict
    sentences whose comparison clause must ride complete.

    ellipsis (2026-11-03, training-feed previews): a word-boundary
    cut with NO marker still reads as a mid-clause cut on the
    dashboard -- the trailing-stop note preview arrived as
    "...options tap 'Arm trailing at 2%" (83 chars, full text
    1,218), a complete word but a sentence cut in half with no
    sign anything was dropped. With ellipsis=True, every ACTUAL
    trim appends a single "…" to the bounded prefix, so a preview
    can grow past its cap by that one character and no more.
    Untrimmed text (at/under the limit, or riding whole for lack
    of any boundary) never gains one. The gate keeps the default
    (ellipsis=False): its pending why is pinned as a verbatim
    clause-complete prefix of the stored reason."""
    if len(text) <= limit:
        return text

    def _mark(out: str) -> str:
        return out + "…" if ellipsis and len(out) < len(text) else out

    bounds = [m.start() for m in re.finditer(r";| — ", text)]
    fits = [b for b in bounds if b <= limit]
    if fits:
        return _mark(text[:fits[-1]].rstrip())
    if word_fallback:
        cut = text.rfind(" ", 0, limit + 1)
        if cut > 0:
            out = text[:cut].rstrip()
            if out.endswith(("—", "-")):
                # A lone dash token at the cut is a dangling
                # separator, not a word -- drop it too.
                out = out[:-1].rstrip()
            return _mark(out)
    if bounds:
        return _mark(text[:bounds[0]].rstrip())
    return text


def _pending_promotion(runs_desc: list[dict]) -> dict | None:
    """The newest unresolved PENDING KASS APPROVAL record, or None
    (promotion lock, 2026-10-25). A run pends when its metrics carry
    pending_approval: the challenger passed its gate but Kass has not
    agreed to install it, so it was recorded, not crowned. A pending
    record is resolved when a LATER run of the SAME kind is recorded
    promoted (an install Kass approved out of band), or when Kass's
    decision is recorded on the run itself (metrics.pending_decision,
    e.g. "declined" -- run 61, 2026-10-25); the pending flag itself
    stays in the registry as history either way. Kinds resolve
    independently -- a PPO install does not clear a stacker pending."""
    newest_promoted: dict[str, int] = {}
    for r in runs_desc:  # newest first: first sighting is the max id
        if r["promoted"]:
            newest_promoted.setdefault(r["kind"], r["id"])
    for r in runs_desc:  # newest first
        m = r.get("metrics") or {}
        if not m.get("pending_approval"):
            continue
        if m.get("pending_decision"):
            continue  # Kass decided this one; decision is on the record
        sup = newest_promoted.get(r["kind"])
        if sup is not None and sup > r["id"]:
            continue  # resolved by a later install of the same kind
        return {"i": r["id"], "k": r["kind"], "t": r["ts"],
                # Clause-bounded, never a raw [:100] slice: that cap
                # cut run 83's reason mid-clause ("...P&L +237 vs ")
                # on both dashboards (2026-10-30 audit).
                "why": _clause_bounded(str(m.get("reason") or ""))}
    return None


def build_gate() -> dict:
    """Promotion-gate state for both dashboards (see module docstring).

    champ = the newest PROMOTED PPO run, shown with the grade it was
    seated on (sourced from the full registry when the snapshot's
    30-run window no longer reaches it). chall/duel = the newest
    nightly (non-lab) PPO run's head-to-head, when that
    run dueled a real champion: the duel's own champion side rides along
    (cs/cwin) because the gate windows slide nightly -- the only honest
    comparison is challenger vs champion graded on the SAME windows.
    When the newest run is the champion itself and won a real duel,
    newchamp=1 marks that the challenger took the crown."""
    s = snapshot.build_training_snapshot()
    runs = [r for r in s["runs"] if r["kind"] == "ppo"]
    registry = None

    def _registry_ppo():
        nonlocal registry
        if registry is None:
            registry = [r for r in _all_runs_desc() if r["kind"] == "ppo"]
        return registry

    champ = None
    champ_run = next((r for r in runs if r["promoted"]), None)
    if champ_run is None:
        # The snapshot window caps at the newest 30 runs; once the
        # registry outgrew it, the window held no promoted PPO at all
        # and the gate rendered champ/chall/duel all null while a
        # champion reigned (2026-10-24 audit). Fall back to the full
        # registry -- the champion is a registry fact, not a window
        # fact (same cap class as the models-index fix).
        champ_run = next((r for r in _registry_ppo() if r["promoted"]),
                         None)
    if champ_run is not None:
        side = _gate_side((champ_run.get("metrics") or {}).get("challenger"))
        if side is not None:
            champ = {"i": champ_run["id"], "t": champ_run["ts"], **side}
    chall = duel = None
    if champ is not None:
        # The gate's challenger is the newest run that dueled for the
        # crown: a nightly run (no lab `family` in its metrics). Lab
        # studies dueled study baselines, never the champion, so they
        # are not gate challengers even when they are the newest
        # PPO-kind rows in the registry.
        # The duel is a registry fact, not a window fact (same cap
        # class as the champion fallback above): the window-first
        # read paired the registry champion with a STALE duel
        # whenever the crowning run itself was capped out of the
        # snapshot window -- exposed 2026-11-06 when Kass crowned
        # run 112 and a promoted-run-capped window's newest duel
        # was run 105's, graded against the previous champion.
        # Always read the newest nightly duel from the registry.
        gate_runs = [r for r in _registry_ppo()
                     if not (r.get("metrics") or {}).get("family")]
        latest = gate_runs[0] if gate_runs else None
        m = (latest.get("metrics") or {}) if latest else {}
        c_side = _gate_side(m.get("challenger"))
        h_side = _gate_side(m.get("champion"))
        if c_side is not None and h_side is not None:
            won = None
            if "win" in c_side and "win" in h_side:
                won = sum(1 for a, b in zip(c_side["win"], h_side["win"])
                          if a > b)
            delta = round(c_side["s"] - h_side["s"], 1)
            formula = 1 if delta > 0 and (won is None or won >= 2) else 0
            # Display truth (2026-10-22): the headline chip is the
            # RECORDED outcome -- a challenger the registry did not
            # promote (e.g. a lab run that only passed the formula, or
            # run 20's vacuous zero-trade "pass") must never render as
            # PASSED. The formula result stays visible as data (fpass).
            # This changes no promotion rule, threshold, or scorer.
            recorded = 1 if latest["promoted"] else 0
            chall = {"i": latest["id"], "t": latest["ts"], **c_side}
            duel = {"won": won, "delta": delta,
                    "pass": 1 if (formula and recorded) else 0,
                    "fpass": formula,
                    "rec": "PROMOTED" if recorded else "NOT PROMOTED",
                    "newchamp": recorded,
                    "cs": h_side["s"]}
            if "win" in h_side:
                duel["cwin"] = h_side["win"]
            if "pl" in h_side:
                duel["cpl"] = h_side["pl"]
    parked, park = 0, None
    park_file = ROOT / "data" / "park_status.json"
    if park_file.exists():
        try:
            pj = json.loads(park_file.read_text())
        except Exception:
            pj = {}
        if pj.get("parked"):
            parked = 1
            park = {"eq": pj.get("equity"), "cash": pj.get("cash"),
                    "btc": pj.get("btc"), "trades": pj.get("trades"),
                    "since": pj.get("since")}
    ch = s.get("champion") or {}
    # Promotion lock (2026-10-25): surface the newest unresolved
    # PENDING KASS APPROVAL from the FULL registry (either kind --
    # a stacker pending matters even though the duel is PPO), and
    # flag the duel itself when its challenger is the pending run.
    pending = _pending_promotion(_all_runs_desc())
    if (pending is not None and duel is not None and chall is not None
            and pending["k"] == "ppo" and pending["i"] == chall["i"]):
        duel["pend"] = 1
    # Stacker champion identity (2026-11-13 dashboard review): sacc/
    # sprec alone left Book #1's gate panel unable to NAME the sitting
    # stacker champion without hard-coding it. schamp = the champions
    # roster's current stacker row (status 1/3), registry-derived:
    # {"i": run id, "n": crown number}. None when no crown exists.
    schamp = None
    roster = build_champions(_all_runs_desc())
    if roster and roster.get("stk"):
        current = [row for row in roster["stk"] if row[7] in (1, 3)]
        if len(current) == 1:
            schamp = {"i": current[0][1], "n": current[0][0]}
    return {
        "parked": parked,
        "park": park,
        "champ": champ,
        "chall": chall,
        "duel": duel,
        "pending": pending,
        "sacc": round(ch["accuracy"], 4) if isinstance(ch.get("accuracy"), (int, float)) else None,
        "sprec": round(ch["precision"], 4) if isinstance(ch.get("precision"), (int, float)) else None,
        "schamp": schamp,
    }


# ---------------------------------------------------------------
# Champion #2 SHADOW DESK feeds (2026-10-25, Kass: Champion #2 gets
# its own trading dashboard). The shadow desk is a SEPARATE
# artifact (tradingagent-live-champion2); these feeds read ONLY
# the shadow book (data/paper_shadow.db + data/
# live_state_shadow.json) plus, for the race, Book #1's ledger
# READ-ONLY. They never write anything. Feed names:
#   shadow         -> push_snapshot on the shadow desk: the Live
#                     Desk snapshot shape for Book #2, plus a
#                     "desk" identity block (book / driver / eyes
#                     / start) sourced from data/shadow_seat.json
#                     so every label on the desk comes from the
#                     seat record itself.
#   shadow_ledger  -> push_ledger: {"ledger", "closed"} for Book #2.
#   shadow_market0/1 -> push_market: the shared market candles;
#                     markers/entry/stop from Book #2's own ledger
#                     and state.
#   shadow_vision0/1 -> push_vision (two parts, the market0/1
#                     pattern): Champion #2's eyes on the newest
#                     closed candle -- part 0: the vote p_up, the
#                     master score, the 3 family composite scores,
#                     and the trend + momentum indicator groups;
#                     part 1: the volume + other groups and the raw
#                     production 21 (continuity with Book #1's
#                     vision; its rows carry the same plain-unit
#                     raw string as the Live vision feed's 4th
#                     element). Scores are percentiles in [0, 100];
#                     the 90 indicators are grouped by the matrix-v3
#                     SPEC family (trend / momentum / volume, else
#                     "other") and the four groups partition the 90
#                     exactly.
#   shadow_race    -> push_race: the head-to-head since the shadow
#                     start -- Book #2 equity, Book #1 equity
#                     (rebased to ITS value at the race start, a
#                     fair race), and buy-and-hold $100k -- all on
#                     one aligned timeline, with P&L headlines and
#                     delta = Book #2 P&L - Book #1 P&L over the
#                     window.

DATA = ROOT / "data"
SHADOW_DB = DATA / "paper_shadow.db"
SHADOW_STATE = DATA / "live_state_shadow.json"
SHADOW_MANIFEST = DATA / "shadow_seat.json"
SHADOW_SEAT_DIR = ROOT / "models" / "shadow_seat"
SHADOW_RACE_B1_DB = DATA / "paper.db"

B1_LABEL = "Book #1 — Driver's Seat"

# Book #3 (Gated Shadow, 2026-10-30): Book #2's exact twin plus
# run 82's frozen meta-label gate on entries (forward test,
# refs/forward-test-gated-book3-2026-10-30.md). Its feeds mirror
# the shadow feed set plus the veto log — the test's primary
# evidence, read from the gated engine's own gate_decisions
# table in paper_gated.db.
GATED_DB = DATA / "paper_gated.db"
GATED_STATE = DATA / "live_state_gated.json"
GATED_MANIFEST = DATA / "gated_seat.json"
GATED_SEAT_DIR = ROOT / "models" / "gated_seat"
GATED_RACE_B2_DB = DATA / "paper_shadow.db"
GATED_RACE_B1_DB = DATA / "paper.db"

# Book #4 (HA Desk, 2026-11-05): PPO run 44 driving with Pick
# #11's HA-candle eyes (Champion #6, run 110) -- Kass's order the
# day he crowned it. Its feeds mirror the shadow feed set; its
# market feed renders the Heikin-Ashi candle stream (Kass's
# explicit requirement), while markers/entry/stop stay on real
# prices, as the trading itself does.
HA_DB = DATA / "paper_ha.db"
HA_STATE = DATA / "live_state_ha.json"
HA_MANIFEST = DATA / "ha_seat.json"
HA_SEAT_DIR = ROOT / "models" / "ha_seat"
HA_RACE_B3_DB = DATA / "paper_gated.db"
HA_RACE_B2_DB = DATA / "paper_shadow.db"
HA_RACE_B1_DB = DATA / "paper.db"

# Book #5 (Champions, 2026-11-12): Muse's book -- Kass's
# full-freedom order. The sitting LAB champions trade as a pair
# (PPO champion drives, stacker champion is the eyes); the seat
# follows the crown via crowning-time re-pins (see
# src/trading/live_champ.py). Its snapshot carries a "crown"
# block (the pair's crowning records, the eyes' grading votes,
# the seat-era lineage, and the bar the next challenger must
# beat) and a "decisions" tail (Book #5's own per-candle log) --
# panels no other desk has, by Muse's design.
CHAMP_DB = DATA / "paper_champ.db"
CHAMP_STATE = DATA / "live_state_champ.json"
CHAMP_MANIFEST = DATA / "champ_seat.json"
CHAMP_SEAT_DIR = ROOT / "models" / "champ_seat"
CHAMP_DECISIONS = DATA / "decisions" / "book5.jsonl"


def _shadow_manifest() -> dict:
    if SHADOW_MANIFEST.exists():
        try:
            man = json.loads(SHADOW_MANIFEST.read_text())
            return man if isinstance(man, dict) else {}
        except Exception:
            return {}
    return {}


def _shadow_desk_block(man: dict | None = None) -> dict:
    """The shadow desk's identity, from the seat manifest itself:
    which book, which driver, which eyes, when it started. The
    dashboard renders its banner and panel captions from this
    block, so a label can never drift from the seat record."""
    man = _shadow_manifest() if man is None else man
    drv = man.get("driver") or {}
    eyes = man.get("eyes") or {}
    d_brain = drv.get("brain", "PPO")
    d_run = drv.get("run_id", 44)
    e_brain = eyes.get("brain", "Stacker")
    e_run = eyes.get("run_id", 64)
    e_champ = eyes.get("champion", "Champion #2")
    e_n = eyes.get("n_features", 94)
    e_set = eyes.get("feature_set", "scored102")
    e_units = "HA candle values" if e_set == "ha10" else "scored indicators"
    block = {
        "book": "Book #2",
        "book_name": f"Book #2 — {e_champ} Shadow",
        "desk": man.get("desk") or f"TradingAgent Live — {e_champ} (Shadow)",
        "driver": {"brain": d_brain, "run": d_run,
                   "label": f"Driver: {d_brain} · Run {d_run}"},
        "eyes": {"brain": e_brain, "run": e_run, "champion": e_champ,
                 "feature_set": e_set,
                 "n_features": e_n,
                 "label": f"Eyes: {e_brain} {e_champ} · Run {e_run} · "
                          f"{e_n} {e_units}"},
        "started": man.get("started"),
        "starting_balance": man.get("starting_balance", 100000),
        "paper": 1,
    }
    gate = man.get("gate") or {}
    if gate:
        # The gate joined this seat on Kass's 2026-11-21 order
        # (run 82's frozen model, threshold from the manifest --
        # 0.51 here vs Book #3's 0.55). Label derives from the
        # seat record like every other identity on this desk.
        g_run = gate.get("run_id", 82)
        g_thr = gate.get("threshold", 0.51)
        block["gate"] = {
            "brain": gate.get("brain", "Meta-Label"),
            "run": g_run, "threshold": g_thr,
            "label": f"Gate: Meta-Label · Run {g_run} · "
                     f"veto entries below {g_thr}"}
    return block


def _shadow_race_label(desk: dict) -> str:
    """Book #2's race label, with the gate named when the seat
    carries one -- the same "Meta-Label Run {run} gate @
    {threshold}" suffix Book #3's race label carries (Kass's
    2026-11-21 order: the 0.51 gate must be named anywhere the
    book is described). Derives from the desk block, never
    hard-coded; a seat with no gate keeps the plain label."""
    label = (f"{desk['book_name']} (PPO Run {desk['driver']['run']} "
             f"driver · {desk['eyes']['champion']} Run "
             f"{desk['eyes']['run']} eyes)")
    gate = desk.get("gate") or {}
    if gate:
        label = (label[:-1] + f" · Meta-Label Run {gate['run']} "
                 f"gate @ {gate['threshold']})")
    return label


def _shadow_gate_decision_rows(limit: int | None = None) -> list:
    """Book #2's recorded gate decisions, newest first, as
    (ts, side, proba, verdict) -- [] when the book has none yet
    (or the gate_decisions table does not exist yet)."""
    import sqlite3

    if not SHADOW_DB.exists():
        return []
    try:
        con = sqlite3.connect(SHADOW_DB)
        try:
            q = ("SELECT ts, side, proba, verdict FROM "
                 "gate_decisions ORDER BY id DESC")
            if limit:
                q += f" LIMIT {int(limit)}"
            return con.execute(q).fetchall()
        finally:
            con.close()
    except Exception:
        return []


def _shadow_gate_summary() -> dict | None:
    """The gate block on Book #2's snapshot: run/threshold from
    the seat manifest, decision counts + the latest decision
    from the engine's own gate_decisions table in
    paper_shadow.db. None when the seat carries no gate."""
    desk_gate = _shadow_desk_block().get("gate")
    if desk_gate is None:
        return None
    rows = _shadow_gate_decision_rows()
    taken = sum(1 for r in rows if r[3] == "TAKEN")
    vetoed = sum(1 for r in rows if r[3] == "VETOED")
    last = None
    if rows:
        ts, side, proba, verdict = rows[0]
        last = {"ts": ts, "side": side, "proba": round(proba, 4),
                "verdict": verdict}
    return {"run": desk_gate["run"], "threshold": desk_gate["threshold"],
            "label": desk_gate["label"], "attempts": len(rows),
            "taken": taken, "vetoed": vetoed, "last": last}


def build_shadow_veto() -> dict:
    """Book #2's Veto Log feed (the gate joined this book
    2026-11-21): every gate decision on an entry attempt, newest
    first, capped at 200 rows with shown/total counts -- the
    same contract as Book #3's veto feed, read from Book #2's
    own gate_decisions table. Counts always cover the FULL
    table; the full table stays in the DB."""
    rows = _shadow_gate_decision_rows(limit=200)
    total_rows = _shadow_gate_decision_rows()
    taken = sum(1 for r in total_rows if r[3] == "TAKEN")
    vetoed = sum(1 for r in total_rows if r[3] == "VETOED")
    desk = _shadow_desk_block()
    gate = desk.get("gate") or {"run": 82, "threshold": 0.51,
                                "label": "Gate: Meta-Label · Run 82"}
    return {
        "desk": desk,
        "gate": {"run": gate["run"],
                 "threshold": gate["threshold"],
                 "label": gate["label"]},
        "total": len(total_rows), "taken": taken, "vetoed": vetoed,
        "shown": len(rows),
        "rows": [[ts, side, round(proba, 4), verdict]
                 for ts, side, proba, verdict in rows],
    }


def _shadow_drift_block(fetch=None, scored_builder=None) -> dict | None:
    """PSI drift summary for the shadow seat's eyes (2026-10-26;
    frame follows the seat — the ha10 columns since 2026-11-09):
    the same windows as the Live drift grid (live = last 168 closed
    hours; baseline = the 792-hour grading holdout ending at the
    newest grading) over the shadow seat's recorded feature set.
    The full 90+ grid is too heavy for this feed, so the block
    carries the worst-10 drifted features + band counts, scope
    labeled by "n". {"worst": [[name, psi], ...], "bands":
    [stable, moderate, significant], "nb": no-baseline count,
    "n": features compared, "base_end"}. None on any failure."""
    try:
        from data.market_data import fetch_klines
        from features.drift import psi_for_samples

        fetch = fetch or fetch_klines
        raw = fetch("BTC-USD", interval="1h", period="730d")
        closed = raw.iloc[:-1]
        if scored_builder is None:
            scored_builder = _shadow_frame_builder()
        frame, _cols = scored_builder(raw)
        frame = frame.loc[frame.index.isin(closed.index)]
        _model, meta = _shadow_seat_stacker()
        columns = list(meta["feature_columns"])
        base, live = _drift_windows(frame, _grading_base_end())
        scored: list[tuple[str, float]] = []
        bands = [0, 0, 0]
        nb = 0
        for c in columns:
            v = psi_for_samples(base[c].to_numpy(), live[c].to_numpy())
            if v is None:
                nb += 1
                continue
            scored.append((c, v))
            if v < 0.1:
                bands[0] += 1
            elif v < 0.25:
                bands[1] += 1
            else:
                bands[2] += 1
        scored.sort(key=lambda kv: -kv[1])
        return {"worst": [[c, round(v, 3)] for c, v in scored[:10]],
                "bands": bands, "nb": nb, "n": len(columns),
                "base_end": str(base.index[-1].date())}
    except Exception:
        return None


def build_shadow_compact_trading() -> dict:
    """Book #2's snapshot: the Live Desk snapshot shape computed on
    the shadow book/state, plus the desk identity block."""
    s = snapshot.build_trading_snapshot(db_path=SHADOW_DB,
                                        state_path=SHADOW_STATE)
    live_ok = s.get("status") == "live"
    if not live_ok:
        s = snapshot._fresh_account_snapshot({"as_of": s.get("as_of")})
    curve = _downsample_curve(s["equity_curve"])
    bench = (_benchmark(curve, db_path=SHADOW_DB) if live_ok
             else {"bench": [], "bench_pct": None})
    out = {
        "as_of": s["as_of"],
        "status": s["status"],
        "desk": _shadow_desk_block(),
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "stop_anchor": s["stop_anchor"],
        "stop_trailing": s["stop_trailing"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4)
        if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": _feed_calendar(s),
        "activity": s["activity"],
        "equity_curve": curve,
        "bench": bench["bench"],
        "bench_pct": bench["bench_pct"],
        **_drawdown_stats(s),
    }
    gate_summary = _shadow_gate_summary()
    if gate_summary is not None:
        out["gate"] = gate_summary
    drift = _shadow_drift_block()
    if drift is not None:
        out["drift"] = drift
    return out


def build_shadow_ledger() -> dict:
    return build_ledger(db_path=SHADOW_DB)


def build_shadow_market(part: int) -> dict:
    return build_market(part, db_path=SHADOW_DB, state_path=SHADOW_STATE)


def _shadow_seat_stacker():
    """(model, meta) for the SHADOW seat stacker -- the pinned
    copy in models/shadow_seat/, never the lab champion file (a
    future crowning must not move this desk's eyes)."""
    from models.stacker import load as load_stacker

    meta = json.loads((SHADOW_SEAT_DIR / "stacker.json").read_text())
    return load_stacker(SHADOW_SEAT_DIR / "stacker.pkl"), meta


def _shadow_frame_builder():
    """The frame builder for the SHADOW seat stacker's own
    feature_set tag: scored102 for Champions #2/#3, pick2_98 (98
    columns) for Champion #4, and the ha10 HA candle matrix since
    Champion #11 (run 119) took the seat on 2026-11-09 (Kass's
    order: the highest of the crowned winners to Book #2). Scored
    tags dispatch through features.pick2_features.builder_for_tag."""
    meta = json.loads((SHADOW_SEAT_DIR / "stacker.json").read_text())
    if meta.get("feature_set") == "ha10":
        return _ha_frame_builder()
    from features import pick2_features

    return pick2_features.builder_for_tag(meta.get("feature_set"))


def _scored_groups(frame_row, columns: list[str]) -> dict:
    """The 90 scored indicators of one candle, grouped by the
    matrix-v3 SPEC family: [[name, score], ...] per group, scores
    rounded to 1 decimal (percentiles). The four groups partition
    the 90 exactly."""
    from features import scored_features as _sf

    spec = _sf._b3.SPEC
    groups: dict[str, list] = {"trend": [], "momentum": [],
                               "volume": [], "other": []}
    for name in columns:
        if name.startswith("score_"):
            continue  # composites headline the feed, not a group
        fam = spec[name][3] if name in spec else ""
        groups[fam if fam in groups else "other"].append(
            [name, round(float(frame_row[name]), 1)])
    return groups


def _own_candle_pin(state_path):
    """(ts, p_up) a book's OWN loop recorded in its state file, or
    (None, None) when the state is missing/unreadable. The vision
    builders pin to this candle (2026-11-03 audit): a vision feed
    recomputed on the newest closed candle can run AHEAD of the
    book's own loop, and for Books #2/#3 -- byte-identical run-73
    eyes -- that recomputation returns the OTHER book's recorded
    vote (Book #2's vision showed 0.4797 @21:00, Book #3's vote,
    while Book #2's own state said 0.4712 @20:00). Every p_up on a
    book's desk is that book's recorded vote or it is not shown as
    the book's vote at all; the master feed learned this the same
    day (see _master_book)."""
    try:
        st = json.loads(Path(state_path).read_text())
        ts = st.get("last_ts")
        p = st.get("last_p_up")
        if ts is None or p is None:
            return None, None
        import pandas as pd

        return pd.Timestamp(ts), round(float(p), 4)
    except Exception:
        return None, None


def _shadow_vision_ha(part: int, fetch=None, ha_builder=None) -> dict:
    """What the SHADOW seat's HA eyes (Champion #11, run 119,
    ha10 — seated 2026-11-09) read on the book's OWN last
    processed candle: build_ha_vision's payload shape on the
    shadow seat/state/desk — the 10 HA values at that candle, the
    book's recorded vote (see _own_candle_pin), and (part 1) the
    driver's production-21 table for the same candle."""
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features

    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    if ha_builder is None:
        ha_builder = _ha_frame_builder()
    frame, _cols = ha_builder(raw)
    model, meta = _shadow_seat_stacker()
    columns = list(meta["feature_columns"])
    complete = frame.loc[frame.index.isin(closed.index), columns].dropna()
    if complete.empty:
        raise RuntimeError("HA frame has no complete closed row")
    own_ts, own_p = _own_candle_pin(SHADOW_STATE)
    pinned = own_ts is not None and own_ts in complete.index
    ts = own_ts if pinned else complete.index[-1]
    srow = complete.loc[ts]
    ha_rows = [[c, HA_LABELS[c], round(float(srow[c]), 4)]
               for c in columns]
    if part == 0:
        return {
            "part": 0,
            "ts": pd.Timestamp(ts).isoformat(),
            "desk": _shadow_desk_block(),
            "p_up": own_p if pinned else round(float(
                model.predict_proba(frame.loc[[ts], columns])[0, 1]), 4),
            "ha_rows": ha_rows,
        }
    feat = add_features(raw)
    frow = feat.loc[ts]
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(raw)
        except Exception:
            raws = None
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(frow[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    return {
        "part": 1,
        "ts": pd.Timestamp(ts).isoformat(),
        "ha_rows": ha_rows,
        "f21": rows,
    }


def build_shadow_vision(part: int, fetch=None, scored_builder=None,
                        ha_builder=None) -> dict:
    """What the shadow seat's eyes read on the book's OWN last
    processed candle -- the candle its loop recorded in
    live_state_shadow.json (see _own_candle_pin for why the panel
    no longer recomputes on the newest closed candle). The values
    come from the SHADOW seat artifact on its own frame restricted
    to the seat stacker's recorded columns -- the same inputs the
    shadow loop trades on. Since Champion #11 (ha10) took the
    seat on 2026-11-09 the payload is the HA shape (see
    _shadow_vision_ha); the scored shape below serves the retired
    scored seats' record."""
    _meta = json.loads((SHADOW_SEAT_DIR / "stacker.json").read_text())
    if _meta.get("feature_set") == "ha10":
        return _shadow_vision_ha(part, fetch=fetch,
                                 ha_builder=ha_builder)
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features

    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    if scored_builder is None:
        scored_builder = _shadow_frame_builder()
    frame, _cols = scored_builder(raw)
    model, meta = _shadow_seat_stacker()
    columns = list(meta["feature_columns"])
    complete = frame.loc[frame.index.isin(closed.index), columns].dropna()
    if complete.empty:
        raise RuntimeError("scored frame has no complete closed row")
    own_ts, own_p = _own_candle_pin(SHADOW_STATE)
    pinned = own_ts is not None and own_ts in complete.index
    ts = own_ts if pinned else complete.index[-1]
    srow = complete.loc[ts]
    desk = _shadow_desk_block()
    if part == 0:
        groups = _scored_groups(srow, columns)
        return {
            "part": 0,
            "ts": pd.Timestamp(ts).isoformat(),
            "desk": desk,
            "p_up": own_p if pinned else round(float(model.predict_proba(
                frame.loc[[ts], columns])[0, 1]), 4),
            "master": round(float(srow["score_master"]), 1),
            "fam": {"trend": round(float(srow["score_trend"]), 1),
                    "momentum": round(float(srow["score_momentum"]), 1),
                    "volume": round(float(srow["score_volume"]), 1)},
            "groups": {"trend": groups["trend"],
                       "momentum": groups["momentum"]},
        }
    groups = _scored_groups(srow, columns)
    feat = add_features(raw)
    frow = feat.loc[ts]
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(raw)
        except Exception:
            raws = None  # display-only: fall back to squeezed-only
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(frow[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    return {
        "part": 1,
        "ts": pd.Timestamp(ts).isoformat(),
        "groups": {"volume": groups["volume"], "other": groups["other"]},
        "f21": rows,
    }


def _race_series(rows, timeline):
    """Forward-fill (ts, value) rows onto the shared timeline."""
    out, last, i = [], None, 0
    rows = sorted(rows)
    for t in timeline:
        while i < len(rows) and rows[i][0] <= t:
            last = rows[i][1]
            i += 1
        out.append([t, round(last, 2) if last is not None else None])
    return out


def _compress_flat_runs(pairs: list) -> list:
    """Drop interior points of constant-value runs from a
    [ts, value] pair curve (2026-11-02 master diet round 2).
    Keeps the first point, the last point, and both edges of every
    flat run, so a step/linear rendering of the compressed curve
    is identical to the original everywhere; only the repeated
    interior points of a run are spent bytes. A fully flat curve
    collapses to its two endpoints."""
    if len(pairs) <= 2:
        return pairs
    out = [pairs[0]]
    for i in range(1, len(pairs) - 1):
        if pairs[i][1] != pairs[i - 1][1] or \
                pairs[i][1] != pairs[i + 1][1]:
            out.append(pairs[i])
    out.append(pairs[-1])
    return out


def _downsample_pairs(curve: list, cap: int = 40) -> list:
    if len(curve) <= cap:
        return curve
    recent = curve[-8:]
    tail = curve[:-8]
    stride = max(1, (len(tail) + (cap - 9)) // (cap - 8))
    out = [tail[j] for j in range(0, len(tail), stride)] + recent
    if out[0] != curve[0]:
        out.insert(0, curve[0])
    if out[-1] != curve[-1]:
        out.append(curve[-1])
    return out


def build_shadow_race() -> dict:
    """The head-to-head since the shadow start (see section
    header). Book #1 is read READ-ONLY from its own ledger; its
    race P&L is measured from ITS equity at the race start so the
    comparison is return-vs-return over the same window, not
    balance-vs-balance."""
    man = _shadow_manifest()
    desk = _shadow_desk_block(man)
    start = man.get("started")
    base_capital = float(man.get("starting_balance", 100000))

    def _rows(db):
        con = snapshot._paper_db(db)
        if con is None:
            return []
        try:
            return con.execute(
                "SELECT ts, equity, price FROM equity ORDER BY ts ASC"
            ).fetchall()
        except Exception:
            return []
        finally:
            con.close()

    rows2 = _rows(SHADOW_DB)
    # Book #1's race base is its equity AT the race start: both
    # books snapshot on candle closes, so its first in-window row
    # is the shared anchor -- rows before the start never enter.
    rows1 = [r for r in _rows(SHADOW_RACE_B1_DB)
             if start is None or r[0] >= start]
    timeline = sorted({r[0] for r in rows2} | {r[0] for r in rows1})
    if start is not None:
        timeline = [t for t in timeline if t >= start]
    b2_curve = _downsample_pairs(
        _race_series([(r[0], r[1]) for r in rows2], timeline))
    b1_curve = _downsample_pairs(
        _race_series([(r[0], r[1]) for r in rows1], timeline))
    # Buy-and-hold: $100k of BTC bought at the race-start close.
    px_rows = [(r[0], r[2]) for r in rows2] or [(r[0], r[2]) for r in rows1]
    p0 = None
    for t, px in px_rows:
        if start is None or t >= start:
            p0 = px
            break
    bh_curve = []
    if p0:
        bh_curve = _downsample_pairs(_race_series(
            [(t, base_capital * px / p0) for t, px in px_rows
             if start is None or t >= start], timeline))

    def _headline(curve, base):
        eq = next((v for _t, v in reversed(curve) if v is not None), base)
        pnl = round(eq - base, 2)
        return {"eq": round(eq, 2), "pnl": pnl,
                "pct": round(100.0 * pnl / base, 2) if base else None}

    b2_base = base_capital
    b1_base = next((v for _t, v in b1_curve if v is not None), base_capital)
    b2_h = _headline(b2_curve, b2_base)
    b1_h = _headline(b1_curve, b1_base)
    bh_h = _headline(bh_curve, base_capital)
    wfe = _wfe_block(man, b2_h)
    return {
        "start": start,
        "desk": desk,
        "b2": {"label": _shadow_race_label(desk),
               **b2_h, "curve": b2_curve},
        "b1": {"label": f"{B1_LABEL} (PPO Run 44 driver · "
                        f"Stacker Run 40 prod21 eyes)",
               **b1_h, "curve": b1_curve},
        "bh": {"label": "Buy & Hold BTC ($100k at race start)",
               **bh_h, "curve": bh_curve},
        "delta": round(b2_h["pnl"] - b1_h["pnl"], 2),
        "delta_pct": round((b2_h["pct"] or 0.0) - (b1_h["pct"] or 0.0), 2),
        "wfe": wfe,
    }


def _wfe_block(man: dict, b2_h: dict) -> dict | None:
    """Walk-Forward Efficiency for Book #2 (2026-10-26 research
    incorporation; design doc Part II): live return vs the graded
    exam return of the pairing the desk CURRENTLY trades. The
    exam numbers are read from the registry, never typed in: the
    champion side of the newest scored_eyes_ppo study run (runs
    65/66 graded the pairing PPO run 44 + Champion #2 run 64 eyes
    on the 33-day holdout; champion total_pnl +$461.81).

    Which pairing that exam belongs to is a seat record: at the
    2026-10-26 shadow seat swap (Kass seated Champion #3's eyes,
    run 69) the manifest gained "wfe_exam" {driver_run, eyes_run}
    attributing the registry exam to the pairing it graded. When
    the attributed pairing is not the desk's current pairing, the
    exam fields are null -- the current pairing has NO graded
    exam -- and the old exam rides along as "pexam", a labeled
    previous-pairing reference only. A manifest with no "wfe_exam"
    (the pre-swap shape) attributes the exam to its own pairing.
    The ratio is only carried once it can mean something -- at
    least one live fill AND at least 7 live days AND a positive
    exam for the CURRENT pairing (a ratio against a non-positive
    or absent exam is undefined, not zero); until then "ratio"
    is None. Bands (the doc's): >80% exceptional, 50-80% proceed,
    30-50% borderline/halt, <30% reject."""
    from datetime import datetime, timezone

    # The attributed exam's values: when the manifest's wfe_exam
    # record CARRIES them (pnl/pct/days -- the 2026-10-27 swap
    # recorded the retired 44 + run 69 pairing's exam from the
    # Pick #2 duel record, which lives in a muse_pick2 run, not
    # in any scored_eyes_ppo run), they are the exam of the
    # attributed pairing. Otherwise the exam is read from the
    # registry as before (the scored_eyes_ppo champion side).
    attr = man.get("wfe_exam") or {}
    exam = None
    exam_days = 33
    if isinstance(attr.get("pnl"), (int, float)):
        exam = {"pnl": round(float(attr["pnl"]), 2),
                "pct": round(float(attr.get(
                    "pct", float(attr["pnl"]) / 1000.0)), 4)}
        if isinstance(attr.get("days"), (int, float)):
            exam_days = int(attr["days"])
    if exam is None:
        for r in _all_runs_desc():
            m = r.get("metrics") or {}
            if m.get("family") == "scored_eyes_ppo":
                c = m.get("champion") or {}
                if isinstance(c.get("total_pnl"), (int, float)):
                    exam = {"pnl": round(c["total_pnl"], 2),
                            "pct": round(c["total_pnl"] / 1000.0, 4)}
                    break
    if exam is None:
        return None
    drv = man.get("driver") or {}
    eyes = man.get("eyes") or {}
    pair = [drv.get("run_id", 44), eyes.get("run_id", 64)]
    exam_pair = [attr.get("driver_run", pair[0]),
                 attr.get("eyes_run", pair[1])]
    pexam = None
    if exam_pair != pair:
        pexam = {"driver": exam_pair[0], "eyes": exam_pair[1],
                 "pnl": exam["pnl"], "pct": exam["pct"],
                 "days": exam_days}
        exam = None
    fills = 0
    con = snapshot._paper_db(SHADOW_DB)
    if con is not None:
        try:
            fills = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        except Exception:
            fills = 0
        con.close()
    days = 0.0
    started = man.get("started")
    if started:
        try:
            t0 = datetime.fromisoformat(str(started))
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            days = round(max(
                0.0, (datetime.now(timezone.utc) - t0).total_seconds()
                / 86400.0), 1)
        except Exception:
            days = 0.0
    live_pct = b2_h.get("pct")
    ratio = None
    if (exam is not None and fills >= 1 and days >= 7
            and exam["pct"] > 0
            and isinstance(live_pct, (int, float))):
        ratio = round(live_pct / exam["pct"], 3)
    return {"pair": pair,
            "exam_pnl": exam["pnl"] if exam else None,
            "exam_pct": exam["pct"] if exam else None,
            "exam_days": exam_days if exam else None,
            "pexam": pexam,
            "live_pnl": b2_h.get("pnl"), "live_pct": live_pct,
            "days": days, "fills": int(fills), "ratio": ratio}


# ------------------------------------------------------------ Book #3
# The Gated Shadow desk (2026-10-30): snapshot / ledger / market /
# vision mirror the shadow desk's shapes on Book #3's own files,
# with the desk identity + gate carried from data/gated_seat.json.
# The veto feed renders the gate_decisions table (every entry
# attempt: ts, side, gate probability, TAKEN/VETOED). The race
# aligns FOUR lines on the gated start: Book #3, Book #2 (rebased
# to its equity at the gated start), Book #1 (same rebase), and
# buy-and-hold; delta = Book #3 P&L - Book #2 P&L (the forward
# test's secondary evidence).


def _gated_manifest() -> dict:
    if GATED_MANIFEST.exists():
        try:
            man = json.loads(GATED_MANIFEST.read_text())
            return man if isinstance(man, dict) else {}
        except Exception:
            return {}
    return {}


def _gated_desk_block(man: dict | None = None) -> dict:
    """Book #3's identity, from its own seat manifest: the
    pairing PLUS the gate (run 82's frozen model; threshold from
    the manifest), so every panel label derives from the seat
    record, never from typed text."""
    man = _gated_manifest() if man is None else man
    drv = man.get("driver") or {}
    eyes = man.get("eyes") or {}
    gate = man.get("gate") or {}
    d_brain = drv.get("brain", "PPO")
    d_run = drv.get("run_id", 44)
    e_brain = eyes.get("brain", "Stacker")
    e_run = eyes.get("run_id", 73)
    e_champ = eyes.get("champion", "Champion #4")
    e_n = eyes.get("n_features", 98)
    g_run = gate.get("run_id", 82)
    g_thr = gate.get("threshold", 0.55)
    return {
        "book": "Book #3",
        "book_name": "Book #3 — Run 82 Gated Shadow",
        "desk": man.get("desk")
        or "TradingAgent Live — Gated (Book #3 · Run 82 Gate)",
        "driver": {"brain": d_brain, "run": d_run,
                   "label": f"Driver: {d_brain} · Run {d_run}"},
        "eyes": {"brain": e_brain, "run": e_run, "champion": e_champ,
                 "feature_set": eyes.get("feature_set", "pick2_98"),
                 "n_features": e_n,
                 "label": f"Eyes: {e_brain} {e_champ} · Run {e_run} · "
                          f"{e_n} scored indicators"},
        "gate": {"brain": gate.get("brain", "Meta-Label"),
                 "run": g_run, "threshold": g_thr,
                 "label": f"Gate: Meta-Label · Run {g_run} · "
                          f"veto entries below {g_thr}"},
        "started": man.get("started"),
        "starting_balance": man.get("starting_balance", 100000),
        "paper": 1,
    }


def _gate_decision_rows(limit: int | None = None) -> list:
    """Every recorded gate decision, newest first, as
    (ts, side, proba, verdict) — [] when the book has none yet."""
    import sqlite3

    if not GATED_DB.exists():
        return []
    try:
        con = sqlite3.connect(GATED_DB)
        try:
            q = ("SELECT ts, side, proba, verdict FROM "
                 "gate_decisions ORDER BY id DESC")
            if limit:
                q += f" LIMIT {int(limit)}"
            return con.execute(q).fetchall()
        finally:
            con.close()
    except Exception:
        return []


def _gate_summary() -> dict:
    """The gate block on Book #3's snapshot: run/threshold from
    the manifest, decision counts + the latest decision from the
    engine's own gate_decisions table."""
    desk_gate = _gated_desk_block()["gate"]
    rows = _gate_decision_rows()
    taken = sum(1 for r in rows if r[3] == "TAKEN")
    vetoed = sum(1 for r in rows if r[3] == "VETOED")
    last = None
    if rows:
        ts, side, proba, verdict = rows[0]
        last = {"ts": ts, "side": side, "proba": round(proba, 4),
                "verdict": verdict}
    return {"run": desk_gate["run"], "threshold": desk_gate["threshold"],
            "label": desk_gate["label"], "attempts": len(rows),
            "taken": taken, "vetoed": vetoed, "last": last}


def build_gated_veto() -> dict:
    """The Veto Log feed: every gate decision on an entry attempt
    (newest first, capped at 200 rows with shown/total counts).
    This log is the forward test's PRIMARY evidence — it renders
    the gated engine's own gate_decisions table verbatim. The
    200-row window is a display bound (2026-11-17 size fix: the
    old 500-row cap plateaued at ~23KB in one push); total /
    taken / vetoed below always count the FULL table, and the
    full table stays in the DB."""
    rows = _gate_decision_rows(limit=200)
    total_rows = _gate_decision_rows()
    taken = sum(1 for r in total_rows if r[3] == "TAKEN")
    vetoed = sum(1 for r in total_rows if r[3] == "VETOED")
    desk = _gated_desk_block()
    return {
        "desk": desk,
        "gate": {"run": desk["gate"]["run"],
                 "threshold": desk["gate"]["threshold"],
                 "label": desk["gate"]["label"]},
        "total": len(total_rows), "taken": taken, "vetoed": vetoed,
        "shown": len(rows),
        "rows": [[ts, side, round(proba, 4), verdict]
                 for ts, side, proba, verdict in rows],
    }


def _gated_seat_stacker():
    """(model, meta) for the GATED seat stacker — the pinned copy
    in models/gated_seat/, never the lab champion file."""
    from models.stacker import load as load_stacker

    meta = json.loads((GATED_SEAT_DIR / "stacker.json").read_text())
    return load_stacker(GATED_SEAT_DIR / "stacker.pkl"), meta


def _gated_frame_builder():
    from features import pick2_features

    meta = json.loads((GATED_SEAT_DIR / "stacker.json").read_text())
    return pick2_features.builder_for_tag(meta.get("feature_set"))


def _gated_drift_block(fetch=None, scored_builder=None) -> dict | None:
    """PSI drift summary for Book #3's eyes — the same windows and
    shape as the shadow desk's block, computed on the gated
    seat's recorded feature set (identical eyes to Book #2, so
    identical drift by construction; computed from this desk's
    own seat record regardless)."""
    try:
        from data.market_data import fetch_klines
        from features.drift import psi_for_samples

        fetch = fetch or fetch_klines
        raw = fetch("BTC-USD", interval="1h", period="730d")
        closed = raw.iloc[:-1]
        if scored_builder is None:
            scored_builder = _gated_frame_builder()
        frame, _cols = scored_builder(raw)
        frame = frame.loc[frame.index.isin(closed.index)]
        _model, meta = _gated_seat_stacker()
        columns = list(meta["feature_columns"])
        base, live = _drift_windows(frame, _grading_base_end())
        scored: list[tuple[str, float]] = []
        bands = [0, 0, 0]
        nb = 0
        for c in columns:
            v = psi_for_samples(base[c].to_numpy(), live[c].to_numpy())
            if v is None:
                nb += 1
                continue
            scored.append((c, v))
            if v < 0.1:
                bands[0] += 1
            elif v < 0.25:
                bands[1] += 1
            else:
                bands[2] += 1
        scored.sort(key=lambda kv: -kv[1])
        return {"worst": [[c, round(v, 3)] for c, v in scored[:10]],
                "bands": bands, "nb": nb, "n": len(columns),
                "base_end": str(base.index[-1].date())}
    except Exception:
        return None


def build_gated_compact_trading() -> dict:
    """Book #3's snapshot: the Live Desk snapshot shape computed
    on the gated book/state, plus the desk identity block and the
    gate summary (decision counts + latest decision)."""
    s = snapshot.build_trading_snapshot(db_path=GATED_DB,
                                        state_path=GATED_STATE)
    live_ok = s.get("status") == "live"
    if not live_ok:
        s = snapshot._fresh_account_snapshot({"as_of": s.get("as_of")})
    curve = _downsample_curve(s["equity_curve"])
    bench = (_benchmark(curve, db_path=GATED_DB) if live_ok
             else {"bench": [], "bench_pct": None})
    out = {
        "as_of": s["as_of"],
        "status": s["status"],
        "desk": _gated_desk_block(),
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "stop_anchor": s["stop_anchor"],
        "stop_trailing": s["stop_trailing"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4)
        if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": _feed_calendar(s),
        "activity": s["activity"],
        "equity_curve": curve,
        "bench": bench["bench"],
        "bench_pct": bench["bench_pct"],
        "gate": _gate_summary(),
        **_drawdown_stats(s),
    }
    drift = _gated_drift_block()
    if drift is not None:
        out["drift"] = drift
    return out


def build_gated_ledger() -> dict:
    return build_ledger(db_path=GATED_DB)


def build_gated_market(part: int) -> dict:
    return build_market(part, db_path=GATED_DB, state_path=GATED_STATE)


def build_gated_vision(part: int, fetch=None, scored_builder=None) -> dict:
    """What Book #3's eyes (run 73, the gated seat copy) read on
    the book's OWN last processed candle — the shadow vision shape
    on the gated seat, pinned to live_state_gated.json exactly as
    the shadow builder pins to its state (see _own_candle_pin).
    The scores come from the GATED seat artifact on the scored
    frame restricted to its recorded columns."""
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features

    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    if scored_builder is None:
        scored_builder = _gated_frame_builder()
    frame, _cols = scored_builder(raw)
    model, meta = _gated_seat_stacker()
    columns = list(meta["feature_columns"])
    complete = frame.loc[frame.index.isin(closed.index), columns].dropna()
    if complete.empty:
        raise RuntimeError("scored frame has no complete closed row")
    own_ts, own_p = _own_candle_pin(GATED_STATE)
    pinned = own_ts is not None and own_ts in complete.index
    ts = own_ts if pinned else complete.index[-1]
    srow = complete.loc[ts]
    desk = _gated_desk_block()
    if part == 0:
        groups = _scored_groups(srow, columns)
        return {
            "part": 0,
            "ts": pd.Timestamp(ts).isoformat(),
            "desk": desk,
            "p_up": own_p if pinned else round(float(model.predict_proba(
                frame.loc[[ts], columns])[0, 1]), 4),
            "master": round(float(srow["score_master"]), 1),
            "fam": {"trend": round(float(srow["score_trend"]), 1),
                    "momentum": round(float(srow["score_momentum"]), 1),
                    "volume": round(float(srow["score_volume"]), 1)},
            "groups": {"trend": groups["trend"],
                       "momentum": groups["momentum"]},
        }
    groups = _scored_groups(srow, columns)
    feat = add_features(raw)
    frow = feat.loc[ts]
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(raw)
        except Exception:
            raws = None
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(frow[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    return {
        "part": 1,
        "ts": pd.Timestamp(ts).isoformat(),
        "groups": {"volume": groups["volume"], "other": groups["other"]},
        "f21": rows,
    }


def build_gated_race() -> dict:
    """The 4-way head-to-head since the gated start. Books #2 and
    #1 are read READ-ONLY from their own ledgers and rebased to
    THEIR equity at the gated start (return-vs-return over the
    same window); Book #3 races from its fresh $100k; buy-and-hold
    is $100k of BTC at the gated-start close."""
    man = _gated_manifest()
    desk = _gated_desk_block(man)
    sdesk = _shadow_desk_block()
    start = man.get("started")
    base_capital = float(man.get("starting_balance", 100000))

    def _rows(db):
        con = snapshot._paper_db(db)
        if con is None:
            return []
        try:
            return con.execute(
                "SELECT ts, equity, price FROM equity ORDER BY ts ASC"
            ).fetchall()
        except Exception:
            return []
        finally:
            con.close()

    rows3 = _rows(GATED_DB)
    rows2 = [r for r in _rows(GATED_RACE_B2_DB)
             if start is None or r[0] >= start]
    rows1 = [r for r in _rows(GATED_RACE_B1_DB)
             if start is None or r[0] >= start]
    timeline = sorted({r[0] for r in rows3} | {r[0] for r in rows2}
                      | {r[0] for r in rows1})
    if start is not None:
        timeline = [t for t in timeline if t >= start]
    b3_curve = _downsample_pairs(
        _race_series([(r[0], r[1]) for r in rows3], timeline))
    b2_curve = _downsample_pairs(
        _race_series([(r[0], r[1]) for r in rows2], timeline))
    b1_curve = _downsample_pairs(
        _race_series([(r[0], r[1]) for r in rows1], timeline))
    px_rows = ([(r[0], r[2]) for r in rows3]
               or [(r[0], r[2]) for r in rows2]
               or [(r[0], r[2]) for r in rows1])
    p0 = None
    for t, px in px_rows:
        if start is None or t >= start:
            p0 = px
            break
    bh_curve = []
    if p0:
        bh_curve = _downsample_pairs(_race_series(
            [(t, base_capital * px / p0) for t, px in px_rows
             if start is None or t >= start], timeline))

    def _headline(curve, base):
        eq = next((v for _t, v in reversed(curve) if v is not None), base)
        pnl = round(eq - base, 2)
        return {"eq": round(eq, 2), "pnl": pnl,
                "pct": round(100.0 * pnl / base, 2) if base else None}

    b3_h = _headline(b3_curve, base_capital)
    b2_base = next((v for _t, v in b2_curve if v is not None),
                   base_capital)
    b2_h = _headline(b2_curve, b2_base)
    b1_base = next((v for _t, v in b1_curve if v is not None),
                   base_capital)
    b1_h = _headline(b1_curve, b1_base)
    bh_h = _headline(bh_curve, base_capital)
    return {
        "start": start,
        "desk": desk,
        "b3": {"label": "Book #3 — Run 82 Gated Shadow (PPO Run "
                        f"{desk['driver']['run']} driver · "
                        f"{desk['eyes']['champion']} Run "
                        f"{desk['eyes']['run']} eyes · Meta-Label "
                        f"Run {desk['gate']['run']} gate @ "
                        f"{desk['gate']['threshold']})",
               **b3_h, "curve": b3_curve},
        "b2": {"label": _shadow_race_label(sdesk),
               **b2_h, "curve": b2_curve},
        "b1": {"label": f"{B1_LABEL} (PPO Run 44 driver · "
                        f"Stacker Run 40 prod21 eyes)",
               **b1_h, "curve": b1_curve},
        "bh": {"label": "Buy & Hold BTC ($100k at race start)",
               **bh_h, "curve": bh_curve},
        "delta": round(b3_h["pnl"] - b2_h["pnl"], 2),
        "delta_pct": round((b3_h["pct"] or 0.0) - (b2_h["pct"] or 0.0), 2),
        "delta_b1": round(b3_h["pnl"] - b1_h["pnl"], 2),
    }


# ------------------------------------------------------------ Book #4 (HA)

HA_LABELS = {
    "ha_dir": "HA direction (green +1 / red −1)",
    "ha_body_pct": "HA body %",
    "ha_range_pct": "HA range %",
    "ha_upper_wick": "HA upper wick share",
    "ha_lower_wick": "HA lower wick share",
    "ha_streak": "HA streak (signed run, ÷12)",
    "ha_slope": "HA slope (close change)",
    "ha_disp": "HA displacement vs raw close",
    "raw_ret": "Raw candle 1h return",
    "ha_body_rel": "HA body share of range",
}


def _ha_manifest() -> dict:
    if HA_MANIFEST.exists():
        try:
            man = json.loads(HA_MANIFEST.read_text())
            return man if isinstance(man, dict) else {}
        except Exception:
            return {}
    return {}


def _ha_desk_block(man: dict | None = None) -> dict:
    """The HA desk's identity, from its seat manifest itself --
    the same doctrine as the shadow/gated desk blocks: labels
    derive from the seat record, never from a hard-coded string."""
    man = _ha_manifest() if man is None else man
    drv = man.get("driver") or {}
    eyes = man.get("eyes") or {}
    d_brain = drv.get("brain", "PPO")
    d_run = drv.get("run_id", 44)
    e_brain = eyes.get("brain", "Stacker")
    e_run = eyes.get("run_id", 110)
    e_champ = eyes.get("champion", "Champion #6")
    e_n = eyes.get("n_features", 10)
    return {
        "book": "Book #4",
        "book_name": f"Book #4 — {e_champ} HA Desk",
        "desk": man.get("desk") or f"TradingAgent Live — {e_champ} (HA Desk)",
        "driver": {"brain": d_brain, "run": d_run,
                   "label": f"Driver: {d_brain} · Run {d_run}"},
        "eyes": {"brain": e_brain, "run": e_run, "champion": e_champ,
                 "feature_set": eyes.get("feature_set", "ha10"),
                 "n_features": e_n,
                 "label": f"Eyes: {e_brain} {e_champ} · Run {e_run} · "
                          f"{e_n} HA candle values"},
        "started": man.get("started"),
        "starting_balance": man.get("starting_balance", 100000),
        "paper": 1,
    }


def _ha_seat_stacker():
    """(model, meta) for the HA seat stacker -- the pinned copy
    in models/ha_seat/, never the lab champion file."""
    from models.stacker import load as load_stacker

    meta = json.loads((HA_SEAT_DIR / "stacker.json").read_text())
    return load_stacker(HA_SEAT_DIR / "stacker.pkl"), meta


def _ha_frame_builder():
    """The HA frame builder: (raw) -> (HA matrix, HA_COLUMNS)."""
    from models.muse_pick10 import HA_COLUMNS, build_ha_matrix

    def _build(raw):
        frame = build_ha_matrix(raw)
        return frame[HA_COLUMNS], list(HA_COLUMNS)

    return _build


def _ha_drift_block(fetch=None, ha_builder=None) -> dict | None:
    """PSI drift summary for the HA seat's eyes: the shadow
    drift block's windows and bands over the 10 HA columns.
    None on any failure."""
    try:
        from data.market_data import fetch_klines
        from features.drift import psi_for_samples

        fetch = fetch or fetch_klines
        raw = fetch("BTC-USD", interval="1h", period="730d")
        closed = raw.iloc[:-1]
        if ha_builder is None:
            ha_builder = _ha_frame_builder()
        frame, _cols = ha_builder(raw)
        frame = frame.loc[frame.index.isin(closed.index)]
        _model, meta = _ha_seat_stacker()
        columns = list(meta["feature_columns"])
        base, live = _drift_windows(frame, _grading_base_end())
        scored: list[tuple[str, float]] = []
        bands = [0, 0, 0]
        nb = 0
        for c in columns:
            v = psi_for_samples(base[c].to_numpy(), live[c].to_numpy())
            if v is None:
                nb += 1
                continue
            scored.append((c, v))
            if v < 0.1:
                bands[0] += 1
            elif v < 0.25:
                bands[1] += 1
            else:
                bands[2] += 1
        scored.sort(key=lambda kv: -kv[1])
        return {"worst": [[c, round(v, 3)] for c, v in scored[:10]],
                "bands": bands, "nb": nb, "n": len(columns),
                "base_end": str(base.index[-1].date())}
    except Exception:
        return None


def build_ha_compact_trading() -> dict:
    """Book #4's snapshot: the Live Desk snapshot shape computed
    on the HA book/state, plus the desk identity block."""
    s = snapshot.build_trading_snapshot(db_path=HA_DB,
                                        state_path=HA_STATE)
    live_ok = s.get("status") == "live"
    if not live_ok:
        s = snapshot._fresh_account_snapshot({"as_of": s.get("as_of")})
    curve = _downsample_curve(s["equity_curve"])
    bench = (_benchmark(curve, db_path=HA_DB) if live_ok
             else {"bench": [], "bench_pct": None})
    out = {
        "as_of": s["as_of"],
        "status": s["status"],
        "desk": _ha_desk_block(),
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "stop_anchor": s["stop_anchor"],
        "stop_trailing": s["stop_trailing"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4)
        if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": _feed_calendar(s),
        "activity": s["activity"],
        "equity_curve": curve,
        "bench": bench["bench"],
        "bench_pct": bench["bench_pct"],
        **_drawdown_stats(s),
    }
    drift = _ha_drift_block()
    if drift is not None:
        out["drift"] = drift
    return out


def build_ha_ledger() -> dict:
    return build_ledger(db_path=HA_DB)


def build_ha_market(part: int) -> dict:
    return build_market(part, db_path=HA_DB, state_path=HA_STATE,
                        ha=True)


def build_ha_vision(part: int, fetch=None, ha_builder=None) -> dict:
    """What Pick #11's eyes read on the book's OWN last processed
    candle (pinned via _own_candle_pin, the audit doctrine): the
    10 HA values at that candle, the book's recorded vote, and
    (part 1) the driver's production-21 table for the same
    candle -- the driver reads prod21 exactly like the other
    books; only the eyes are HA."""
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features

    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    if ha_builder is None:
        ha_builder = _ha_frame_builder()
    frame, _cols = ha_builder(raw)
    model, meta = _ha_seat_stacker()
    columns = list(meta["feature_columns"])
    complete = frame.loc[frame.index.isin(closed.index), columns].dropna()
    if complete.empty:
        raise RuntimeError("HA frame has no complete closed row")
    own_ts, own_p = _own_candle_pin(HA_STATE)
    pinned = own_ts is not None and own_ts in complete.index
    ts = own_ts if pinned else complete.index[-1]
    srow = complete.loc[ts]
    ha_rows = [[c, HA_LABELS[c], round(float(srow[c]), 4)]
               for c in columns]
    if part == 0:
        return {
            "part": 0,
            "ts": pd.Timestamp(ts).isoformat(),
            "desk": _ha_desk_block(),
            "p_up": own_p if pinned else round(float(
                model.predict_proba(frame.loc[[ts], columns])[0, 1]), 4),
            "ha_rows": ha_rows,
        }
    feat = add_features(raw)
    frow = feat.loc[ts]
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(raw)
        except Exception:
            raws = None
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(frow[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    return {
        "part": 1,
        "ts": pd.Timestamp(ts).isoformat(),
        "ha_rows": ha_rows,
        "f21": rows,
    }


def build_ha_race() -> dict:
    """The 5-way head-to-head since the HA start. Books #1/#2/#3
    are read READ-ONLY from their own ledgers and rebased to
    THEIR equity at the HA start (return-vs-return over the same
    window); Book #4 races from its fresh $100k; buy-and-hold is
    $100k of BTC at the HA-start close."""
    man = _ha_manifest()
    desk = _ha_desk_block(man)
    g3 = _gated_desk_block()
    sdesk = _shadow_desk_block()
    start = man.get("started")
    base_capital = float(man.get("starting_balance", 100000))

    def _rows(db):
        con = snapshot._paper_db(db)
        if con is None:
            return []
        try:
            return con.execute(
                "SELECT ts, equity, price FROM equity ORDER BY ts ASC"
            ).fetchall()
        except Exception:
            return []
        finally:
            con.close()

    rows4 = _rows(HA_DB)
    rows3 = [r for r in _rows(HA_RACE_B3_DB)
             if start is None or r[0] >= start]
    rows2 = [r for r in _rows(HA_RACE_B2_DB)
             if start is None or r[0] >= start]
    rows1 = [r for r in _rows(HA_RACE_B1_DB)
             if start is None or r[0] >= start]
    timeline = sorted({r[0] for r in rows4} | {r[0] for r in rows3}
                      | {r[0] for r in rows2} | {r[0] for r in rows1})
    if start is not None:
        timeline = [t for t in timeline if t >= start]
    curves = {}
    for key, rows in (("b4", rows4), ("b3", rows3), ("b2", rows2),
                      ("b1", rows1)):
        curves[key] = _downsample_pairs(
            _race_series([(r[0], r[1]) for r in rows], timeline))
    px_rows = ([(r[0], r[2]) for r in rows4]
               or [(r[0], r[2]) for r in rows3]
               or [(r[0], r[2]) for r in rows2]
               or [(r[0], r[2]) for r in rows1])
    p0 = None
    for t, px in px_rows:
        if start is None or t >= start:
            p0 = px
            break
    bh_curve = []
    if p0:
        bh_curve = _downsample_pairs(_race_series(
            [(t, base_capital * px / p0) for t, px in px_rows
             if start is None or t >= start], timeline))

    def _headline(curve, base):
        eq = next((v for _t, v in reversed(curve) if v is not None), base)
        pnl = round(eq - base, 2)
        return {"eq": round(eq, 2), "pnl": pnl,
                "pct": round(100.0 * pnl / base, 2) if base else None}

    def _base(curve):
        return next((v for _t, v in curve if v is not None),
                    base_capital)

    b4_h = _headline(curves["b4"], base_capital)
    b3_h = _headline(curves["b3"], _base(curves["b3"]))
    b2_h = _headline(curves["b2"], _base(curves["b2"]))
    b1_h = _headline(curves["b1"], _base(curves["b1"]))
    bh_h = _headline(bh_curve, base_capital)
    return {
        "start": start,
        "desk": desk,
        "b4": {"label": f"{desk['book_name']} (PPO Run "
                        f"{desk['driver']['run']} driver · "
                        f"{desk['eyes']['champion']} Run "
                        f"{desk['eyes']['run']} eyes)",
               **b4_h, "curve": curves["b4"]},
        "b3": {"label": f"Book #3 — Run 82 Gated Shadow (PPO Run "
                        f"{g3['driver']['run']} driver · "
                        f"{g3['eyes']['champion']} Run "
                        f"{g3['eyes']['run']} eyes · Meta-Label "
                        f"Run {g3['gate']['run']} gate @ "
                        f"{g3['gate']['threshold']})",
               **b3_h, "curve": curves["b3"]},
        "b2": {"label": _shadow_race_label(sdesk),
               **b2_h, "curve": curves["b2"]},
        "b1": {"label": f"{B1_LABEL} (PPO Run 44 driver · "
                        f"Stacker Run 40 prod21 eyes)",
               **b1_h, "curve": curves["b1"]},
        "bh": {"label": "Buy & Hold BTC ($100k at race start)",
               **bh_h, "curve": bh_curve},
        "delta": round(b4_h["pnl"] - b2_h["pnl"], 2),
        "delta_pct": round((b4_h["pct"] or 0.0) - (b2_h["pct"] or 0.0), 2),
        "delta_b1": round(b4_h["pnl"] - b1_h["pnl"], 2),
    }


# ------------------------------------------------------------ Book #5

def _champ_manifest() -> dict:
    if CHAMP_MANIFEST.exists():
        try:
            man = json.loads(CHAMP_MANIFEST.read_text())
            return man if isinstance(man, dict) else {}
        except Exception:
            return {}
    return {}


def _champ_desk_block(man: dict | None = None) -> dict:
    """Book #5's identity, from its seat manifest itself -- the
    labels-follow-the-seat doctrine, like every other desk."""
    man = _champ_manifest() if man is None else man
    drv = man.get("driver") or {}
    eyes = man.get("eyes") or {}
    d_brain = drv.get("brain", "PPO")
    d_run = drv.get("run_id", 112)
    e_brain = eyes.get("brain", "Stacker")
    e_run = eyes.get("run_id", 126)
    e_champ = eyes.get("champion", "Champion #13")
    e_n = eyes.get("n_features", 10)
    return {
        "book": "Book #5",
        "book_name": "Book #5 — Champions (Follows the Crown)",
        "desk": man.get("desk") or "TradingAgent Live — Champions (Book #5)",
        "driver": {"brain": d_brain, "run": d_run,
                   "label": f"Driver: {d_brain} · Run {d_run} · "
                            f"Lab PPO Champion"},
        "eyes": {"brain": e_brain, "run": e_run, "champion": e_champ,
                 "feature_set": eyes.get("feature_set", "ha10"),
                 "n_features": e_n,
                 "label": f"Eyes: {e_brain} {e_champ} · Run {e_run} · "
                          f"{eyes.get('feature_set', 'ha10')} frame"},
        "started": man.get("started"),
        "starting_balance": man.get("starting_balance", 100000),
        "seat_policy": man.get("seat_policy", "follows_the_crown"),
        "paper": 1,
    }


def _champ_eyes_votes(run_id) -> dict | None:
    """The sitting stacker champion's grading vote histogram,
    read verbatim from its registry record (the same votes the
    Training Lab serves for the sitting champion)."""
    import sqlite3

    try:
        con = sqlite3.connect(DATA / "training.db")
        try:
            row = con.execute("SELECT metrics FROM runs WHERE id=?",
                              (run_id,)).fetchone()
        finally:
            con.close()
        if not row:
            return None
        v = (json.loads(row[0]) or {}).get("votes")
        if not v:
            return None
        return {"run": run_id, "n": v.get("n"), "bins": v.get("bins"),
                "min": v.get("min"), "max": v.get("max"),
                "med": v.get("med"), "mid": v.get("mid"),
                "tight": v.get("tight")}
    except Exception:
        return None


def _champ_crown_block(man: dict) -> dict:
    """The crown block (Muse's design for Book #5): who holds
    the crowns this book trades, the records they won them with,
    the seat-era lineage, and the bar the next challengers must
    beat at the next retrain."""
    g = man.get("grading_record") or {}
    eyes = man.get("eyes") or {}
    drv = man.get("driver") or {}
    return {
        "driver": {
            "run": drv.get("run_id"),
            "brain": "PPO",
            "title": "Lab PPO Champion",
            "duel_score": g.get("driver_duel_score"),
            "duel_pnl": g.get("driver_duel_pnl"),
            "duel_windows": g.get("driver_duel_windows"),
            "bench_pnl": g.get("driver_duel_bench_pnl"),
            "reason": g.get("driver_reason"),
        },
        "eyes": {
            "run": eyes.get("run_id"),
            "brain": "Stacker",
            "champion": eyes.get("champion"),
            "title": f"Lab Stacker {eyes.get('champion') or ''}".strip(),
            "feature_set": eyes.get("feature_set"),
            "acc": g.get("eyes_exam_acc"),
            "precision": g.get("eyes_precision"),
            "wf": g.get("eyes_wf_acc"),
            "votes": _champ_eyes_votes(eyes.get("run_id")),
        },
        # Feed bound (2026-11-17 size fix): the lineage grows one
        # era per crowning (~119B/era, unbounded over years); the
        # feed ships the newest 24 eras and the true total count.
        "lineage": (man.get("lineage") or [])[-24:],
        "lineage_total": len(man.get("lineage") or []),
        "era": (man.get("lineage") or [{}])[-1].get("era"),
        "bar": {
            "note": "What the next challengers must beat",
            "stacker_acc": g.get("eyes_exam_acc"),
            "ppo": "Full gate pass: beat the champion's duel "
                   "aggregate, win at least 2 of 3 windows, and "
                   "clear the floor (P&L > $0 and > buy-and-hold, "
                   "non-hold at least 10%)",
        },
    }


def _champ_decisions(limit: int = 14) -> list:
    """Book #5's own per-candle decision tail (newest last), from
    data/decisions/book5.jsonl -- what the pair thought and did,
    candle by candle."""
    try:
        lines = CHAMP_DECISIONS.read_text().strip().splitlines()
        out = []
        for ln in lines[-limit:]:
            r = json.loads(ln)
            out.append({
                "ts": r.get("ts"), "close": r.get("close"),
                "p_up": r.get("p_up"), "stance": r.get("stance"),
                "action": r.get("action"), "taken": r.get("taken"),
                "reason": r.get("reason"),
                "pos_before": r.get("pos_before"),
                "pos_after": r.get("pos_after"),
                "equity": r.get("equity"), "stop": r.get("stop_level"),
                "exposure_pct": r.get("exposure_pct"),
            })
        return out
    except Exception:
        return []


def _champ_drift_block(fetch=None) -> dict | None:
    """PSI drift summary for the champ seat's eyes -- the same
    windows and shape as the shadow block (live = last 168 closed
    hours; baseline = the 792-hour grading holdout ending at the
    newest grading), computed over the seated champion's own
    frame via live_champ.eyes_frame, so the frame follows the
    crown exactly like the seat does. {"worst": [[name, psi],
    ...], "bands": [stable, moderate, significant], "nb":
    no-baseline count, "n": features compared, "base_end"}.
    None on any failure -- the panel hides."""
    try:
        from data.market_data import fetch_klines
        from features.drift import psi_for_samples
        from trading.live_champ import (eyes_frame, seat_columns,
                                        seat_feature_set)

        fetch = fetch or fetch_klines
        raw = fetch("BTC-USD", interval="1h", period="730d")
        closed = raw.iloc[:-1]
        fset = seat_feature_set(CHAMP_SEAT_DIR)
        frame, _cols = eyes_frame(raw, fset)
        frame = frame.loc[frame.index.isin(closed.index)]
        columns = seat_columns(CHAMP_SEAT_DIR)
        base, live = _drift_windows(frame, _grading_base_end())
        scored: list[tuple[str, float]] = []
        bands = [0, 0, 0]
        nb = 0
        for c in columns:
            v = psi_for_samples(base[c].to_numpy(), live[c].to_numpy())
            if v is None:
                nb += 1
                continue
            scored.append((c, v))
            if v < 0.1:
                bands[0] += 1
            elif v < 0.25:
                bands[1] += 1
            else:
                bands[2] += 1
        scored.sort(key=lambda kv: -kv[1])
        return {"worst": [[c, round(v, 3)] for c, v in scored[:10]],
                "bands": bands, "nb": nb, "n": len(columns),
                "base_end": str(base.index[-1].date())}
    except Exception:
        return None


def build_champ_compact_trading() -> dict:
    """Book #5's snapshot: the Live Desk snapshot shape computed
    on the champ book/state, plus the desk identity, the crown
    block, and the decision tail."""
    s = snapshot.build_trading_snapshot(db_path=CHAMP_DB,
                                        state_path=CHAMP_STATE)
    live_ok = s.get("status") == "live"
    if not live_ok:
        s = snapshot._fresh_account_snapshot({"as_of": s.get("as_of")})
    curve = _downsample_curve(s["equity_curve"])
    bench = (_benchmark(curve, db_path=CHAMP_DB) if live_ok
             else {"bench": [], "bench_pct": None})
    man = _champ_manifest()
    out = {
        "as_of": s["as_of"],
        "status": s["status"],
        "desk": _champ_desk_block(man),
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "stop_anchor": s["stop_anchor"],
        "stop_trailing": s["stop_trailing"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4)
        if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": _feed_calendar(s),
        "activity": s["activity"],
        "equity_curve": curve,
        "bench": bench["bench"],
        "bench_pct": bench["bench_pct"],
        **_drawdown_stats(s),
        "crown": _champ_crown_block(man),
        "decisions": _champ_decisions(),
    }
    # Decision-mode fields (the desk labels the equity mark from
    # these two, so the number never wears the wrong label).
    # Open-decision mode (Kass's order, 2026-11-13, ended by his
    # order 2026-11-17): after a decision, this snapshot's
    # btc_price was the decision candle's OPEN (the price the
    # book just traded at), and the last decision candle is the
    # state's last_open_ts. In close mode (the production
    # default again), the mark is a candle close and the last
    # decision candle is the state's last_ts.
    try:
        from trading.live_champ import DECISION_MODE as _DMODE
    except Exception:
        _DMODE = "close"
    try:
        _chst = json.loads(Path(CHAMP_STATE).read_text())
    except Exception:
        _chst = {}
    out["decision_mode"] = _DMODE
    out["last_decision_ts"] = (_chst.get("last_open_ts")
                               if _DMODE == "open"
                               else _chst.get("last_ts"))
    drift = _champ_drift_block()
    if drift is not None:
        out["drift"] = drift
    return out


def build_champ_ledger() -> dict:
    return build_ledger(db_path=CHAMP_DB)


def build_champ_market(part: int) -> dict:
    # The chart renders the Heikin-Ashi candle stream while the
    # seated eyes read the ha10 frame (Book #4's doctrine). If a
    # future stacker champion's frame is not ha10, revisit: the
    # chart should always show the candles the eyes read.
    return build_market(part, db_path=CHAMP_DB, state_path=CHAMP_STATE,
                        ha=True)


def _champ_seat_stacker():
    """(model, meta) for the champ seat stacker -- the pinned
    copy in models/champ_seat/, never the lab champion file."""
    from models.stacker import load as load_stacker

    meta = json.loads((CHAMP_SEAT_DIR / "stacker.json").read_text())
    return load_stacker(CHAMP_SEAT_DIR / "stacker.pkl"), meta


def build_champ_vision(part: int, fetch=None) -> dict:
    """What the seated stacker champion read on Book #5's OWN
    last processed candle (pinned via _own_candle_pin, the audit
    doctrine): the eyes' own frame values at that candle, the
    book's recorded vote, and (part 1) the driver's production-21
    table for the same candle."""
    import pandas as pd

    from data.market_data import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features
    from trading.live_champ import eyes_frame, seat_columns, seat_feature_set

    fetch = fetch or fetch_klines
    raw = fetch("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    feat = add_features(raw)
    fset = seat_feature_set(CHAMP_SEAT_DIR)
    frame, _cols = eyes_frame(raw, fset, feat=feat)
    model, meta = _champ_seat_stacker()
    columns = seat_columns(CHAMP_SEAT_DIR)
    complete = frame.loc[frame.index.isin(closed.index), columns].dropna()
    if complete.empty:
        raise RuntimeError("champ eyes frame has no complete closed row")
    own_ts, own_p = _own_candle_pin(CHAMP_STATE)
    pinned = own_ts is not None and own_ts in complete.index
    ts = own_ts if pinned else complete.index[-1]
    srow = complete.loc[ts]
    labels = HA_LABELS if fset == "ha10" else FEATURE_LABELS
    eyes_rows = [[c, labels.get(c, c), round(float(srow[c]), 4)]
                 for c in columns]
    if part == 0:
        return {
            "part": 0,
            "ts": pd.Timestamp(ts).isoformat(),
            "desk": _champ_desk_block(),
            "feature_set": fset,
            "p_up": own_p if pinned else round(float(
                model.predict_proba(frame.loc[[ts], columns])[0, 1]), 4),
            "eyes_rows": eyes_rows,
        }
    frow = feat.loc[ts]
    raws = None
    if _VISION_RAWS_EMITTED:
        try:
            raws = _raw_frame(raw)
        except Exception:
            raws = None
    rows = []
    for c in FEATURE_COLUMNS:
        r = [c, FEATURE_LABELS[c], round(float(frow[c]), 3)]
        if _VISION_RAWS_EMITTED:
            r.append(_raw_str(raws, ts, c))
        rows.append(r)
    return {
        "part": 1,
        "ts": pd.Timestamp(ts).isoformat(),
        "eyes_rows": eyes_rows,
        "f21": rows,
    }


def build_champ_indicators() -> dict:
    """The HA stream, measured: indicator series over the same
    96 closed hourly candles the price chart draws, computed ON
    the Heikin-Ashi candles with the production transforms (the
    Pick #12 suite machinery -- the same math the model family
    uses), so the panels show the market the way this book's
    eyes actually read it. Series are positional: value i is at
    start + i*step (epoch seconds). rsi is in plain 0-100 units
    (un-squeezed from ha_rsi_c); the signed series keep their
    production [-1, 1] squeeze. boll (Bollinger position on HA
    closes, 0 = lower band, 1 = upper band) and relvol (volume
    vs its 24h mean) are DISPLAY-ONLY, labeled as such in the
    feed's basis line -- they are not model inputs. "now"
    carries each series' last value; "ts_last" is the newest
    closed candle's timestamp."""
    import pandas as pd

    from data.market_data import fetch_klines
    from models.ha_seq import heikin_ashi
    from models.muse_pick12 import build_ha_suite_matrix

    raw = fetch_klines("BTC-USD", interval="1h", period="730d")
    closed = raw.iloc[:-1]
    suite = build_ha_suite_matrix(closed)
    ha = heikin_ashi(closed)
    hc = ha["ha_close"]
    vol = closed["volume"].astype(float)
    sma20 = hc.rolling(20).mean()
    std20 = hc.rolling(20).std()
    boll = ((hc - (sma20 - 2.0 * std20)) / (4.0 * std20)).clip(-0.25, 1.25)
    relvol = vol / vol.rolling(24).mean()
    frame = pd.DataFrame({
        "rsi": suite["ha_rsi_c"] * 50.0 + 50.0,
        "macd": suite["ha_macd_hist_rel"],
        "ewo": suite["ha_ewo"],
        "obv": suite["ha_obv_slope"],
        "adl": suite["ha_adl_slope"],
        "netvol": suite["ha_netvol_rel"],
        "atr": suite["ha_atr_rel"],
        "sma50": suite["ha_dist_sma50"],
        "boll": boll,
        "relvol": relvol,
    }).dropna().tail(96)
    if len(frame) < 48:
        raise RuntimeError("champ indicators window too short")
    nd = {"rsi": 1, "macd": 4, "ewo": 4, "obv": 4, "adl": 4,
          "netvol": 4, "atr": 4, "sma50": 4, "boll": 3, "relvol": 2}
    series = {c: [round(float(v), nd[c]) for v in frame[c]]
              for c in frame.columns}
    now = {c: series[c][-1] for c in frame.columns}
    return {
        "start": int(frame.index[0].timestamp()),
        "step": 3600,
        "ts_last": pd.Timestamp(frame.index[-1]).isoformat(),
        "basis": "Heikin-Ashi stream (the seated eyes' candles), "
                 "production transforms; Bollinger position and "
                 "Relative Volume are display-only.",
        "series": series,
        "now": now,
    }


def build_champ_race() -> dict:
    """The 6-way head-to-head since Book #5's start: all five
    books plus buy-and-hold. The other books are read READ-ONLY
    from their own ledgers and rebased to their equity at Book
    #5's start (return-vs-return over the same window); Book #5
    races from its fresh $100k; buy-and-hold is $100k of BTC at
    Book #5's start close."""
    man = _champ_manifest()
    desk = _champ_desk_block(man)
    hadesk = _ha_desk_block()
    g3 = _gated_desk_block()
    sdesk = _shadow_desk_block()
    start = man.get("started")
    base_capital = float(man.get("starting_balance", 100000))

    def _rows(db):
        con = snapshot._paper_db(db)
        if con is None:
            return []
        try:
            return con.execute(
                "SELECT ts, equity, price FROM equity ORDER BY ts ASC"
            ).fetchall()
        except Exception:
            return []
        finally:
            con.close()

    def _since(db):
        return [r for r in _rows(db) if start is None or r[0] >= start]

    rows5 = _rows(CHAMP_DB)
    rows4 = _since(HA_DB)
    rows3 = _since(HA_RACE_B3_DB)
    rows2 = _since(HA_RACE_B2_DB)
    rows1 = _since(HA_RACE_B1_DB)
    timeline = sorted({r[0] for r in rows5} | {r[0] for r in rows4}
                      | {r[0] for r in rows3} | {r[0] for r in rows2}
                      | {r[0] for r in rows1})
    if start is not None:
        timeline = [t for t in timeline if t >= start]
    curves = {}
    for key, rows in (("b5", rows5), ("b4", rows4), ("b3", rows3),
                      ("b2", rows2), ("b1", rows1)):
        curves[key] = _downsample_pairs(
            _race_series([(r[0], r[1]) for r in rows], timeline))
    px_rows = ([(r[0], r[2]) for r in rows5]
               or [(r[0], r[2]) for r in rows4]
               or [(r[0], r[2]) for r in rows3]
               or [(r[0], r[2]) for r in rows2]
               or [(r[0], r[2]) for r in rows1])
    p0 = None
    for t, px in px_rows:
        if start is None or t >= start:
            p0 = px
            break
    bh_curve = []
    if p0:
        bh_curve = _downsample_pairs(_race_series(
            [(t, base_capital * px / p0) for t, px in px_rows
             if start is None or t >= start], timeline))

    def _headline(curve, base):
        eq = next((v for _t, v in reversed(curve) if v is not None), base)
        pnl = round(eq - base, 2)
        return {"eq": round(eq, 2), "pnl": pnl,
                "pct": round(100.0 * pnl / base, 2) if base else None}

    def _base(curve):
        return next((v for _t, v in curve if v is not None),
                    base_capital)

    b5_h = _headline(curves["b5"], base_capital)
    b4_h = _headline(curves["b4"], _base(curves["b4"]))
    b3_h = _headline(curves["b3"], _base(curves["b3"]))
    b2_h = _headline(curves["b2"], _base(curves["b2"]))
    b1_h = _headline(curves["b1"], _base(curves["b1"]))
    bh_h = _headline(bh_curve, base_capital)
    return {
        "start": start,
        "desk": desk,
        "b5": {"label": f"{desk['book_name']} (PPO Run "
                        f"{desk['driver']['run']} driver · "
                        f"{desk['eyes']['champion']} Run "
                        f"{desk['eyes']['run']} eyes)",
               **b5_h, "curve": curves["b5"]},
        "b4": {"label": f"{hadesk['book_name']} (PPO Run "
                        f"{hadesk['driver']['run']} driver · "
                        f"{hadesk['eyes']['champion']} Run "
                        f"{hadesk['eyes']['run']} eyes)",
               **b4_h, "curve": curves["b4"]},
        "b3": {"label": f"Book #3 — Run 82 Gated Shadow (PPO Run "
                        f"{g3['driver']['run']} driver · "
                        f"{g3['eyes']['champion']} Run "
                        f"{g3['eyes']['run']} eyes · Meta-Label "
                        f"Run {g3['gate']['run']} gate @ "
                        f"{g3['gate']['threshold']})",
               **b3_h, "curve": curves["b3"]},
        "b2": {"label": _shadow_race_label(sdesk),
               **b2_h, "curve": curves["b2"]},
        "b1": {"label": f"{B1_LABEL} (PPO Run 44 driver · "
                        f"Stacker Run 40 prod21 eyes)",
               **b1_h, "curve": curves["b1"]},
        "bh": {"label": "Buy & Hold BTC ($100k at race start)",
               **bh_h, "curve": bh_curve},
        "delta": round(b5_h["pnl"] - b1_h["pnl"], 2),
        "delta_pct": round((b5_h["pct"] or 0.0) - (b1_h["pct"] or 0.0), 2),
        "delta_bh": round(b5_h["pnl"] - bh_h["pnl"], 2),
    }


# ------------------------------------------------------------ Master desk
# ONE combined payload for the master live desk (2026-10-31): all
# three paper books in a single feed (five since Book #5 joined
# 2026-11-12), assembled by REUSING the
# per-book machinery -- snapshot.build_trading_snapshot for every
# book's state, snapshot.trade_stats / closed_trades /
# activity_stats for the journal stats, the desk blocks for
# identity, the gated engine's gate_decisions rows for Book #3's
# vetoes, and the shadow race builder for the buy-and-hold line.
# No math is forked here: a master number is the per-book number,
# passed through or summed. READ-ONLY over the books' files.
#
# Sign conventions (mirroring the per-book feeds): cur_drawdown
# is the drawdown feed's dd_now (0 at the equity peak, negative
# underwater); max_dd_pct is the positive magnitude of the worst
# recorded dd_pct; max_drawdown is that worst peak-to-trough drop
# in dollars. Stats a book has not earned yet are null, never 0:
# profit_factor / expectancy with no closed trades, avg_win with
# no wins, avg_loss with no losses, payoff without both.


def _master_shared_fetch():
    """ONE memoized market fetch for a whole master build
    (2026-11-04). A 730d fetch is 59 sequential chunk requests;
    build_master's three vision paths used to fetch the series
    independently (~182 requests, ~676s at the day's ~3.6s/request
    proxy latency -- the 15-min master-feed job timed out three
    runs in a row). Every consumer reads closed candles
    (iloc[:-1]) and add_features copies its input, so sharing the
    fetched frame changes no values -- at most the forming
    candle's fetch-time skew the build already exhibits between
    books disappears."""
    from data.market_data import fetch_klines

    cache: dict = {}

    def fetch(symbol, interval="1h", period="730d"):
        key = (symbol, interval, period)
        if key not in cache:
            cache[key] = fetch_klines(symbol, interval=interval,
                                      period=period)
        return cache[key]

    return fetch


def _master_vision_book1(fetch=None):
    """(p_up, vision_ts, top_gauges) from Book #1's own vision
    feed: the 8 of its 21 squeezed features farthest from their
    neutral midpoint (0). `fetch` is the master build's shared
    memoized fetch; None keeps the standalone direct fetch."""
    v = build_vision(fetch=fetch)
    rows = sorted(v.get("f") or [], key=lambda r: -abs(r[2]))[:8]
    return v.get("p_up"), v.get("ts"), [
        {"name": r[0], "value": r[2]} for r in rows]


def _master_vision_scored(vision_fn, frame_builder, fetch=None):
    """(p_up, vision_ts, top_gauges) for a scored-eyes book (Book
    #2 / #3): both vision parts over ONE memoized market fetch
    and ONE scored-frame build (each part otherwise refetches
    730d and rebuilds the frame). Gauges are the scored
    indicators (0-100 percentile scores) farthest from their
    neutral midpoint (50). `fetch` is the master build's shared
    memoized fetch; None mints a per-call one (the pre-2026-11-04
    behavior, kept for standalone callers/tests)."""
    if fetch is None:
        fetch = _master_shared_fetch()

    raw = fetch("BTC-USD", interval="1h", period="730d")
    frame_tuple = frame_builder()(raw)
    p0 = vision_fn(0, fetch=fetch,
                   scored_builder=lambda _raw: frame_tuple)
    p1 = vision_fn(1, fetch=fetch,
                   scored_builder=lambda _raw: frame_tuple)
    gauges = []
    for part in (p0, p1):
        for group in (part.get("groups") or {}).values():
            gauges.extend(group)
    top = sorted(gauges, key=lambda g: -abs(g[1] - 50.0))[:8]
    return p0.get("p_up"), p0.get("ts"), [
        {"name": g[0], "value": g[1]} for g in top]


def _master_vision_ha(vision_fn, fetch=None):
    """(p_up, vision_ts, top_gauges) for an HA-eyes book (Book
    #4; Book #2 since Champion #11 took its seat 2026-11-09):
    gauges are the HA readings farthest from their neutral
    0 (the bounded HA values -- direction, streak, body share --
    dominate honestly; the ratio readings ride along), read from
    the book's own vision part 0 over the shared fetch."""
    v = vision_fn(0, fetch=fetch)
    rows = sorted(v.get("ha_rows") or [], key=lambda r: -abs(r[2]))[:8]
    return v.get("p_up"), v.get("ts"), [
        {"name": r[0], "value": r[2]} for r in rows]


def _master_vision_champ(fetch=None):
    """(p_up, vision_ts, top_gauges) for Book #5 (Champions):
    gauges are the seated eyes' own readings (build_champ_vision
    part 0 eyes_rows) farthest from their neutral 0. The eyes
    frame is the seated stacker's own -- ha10 today, prod21
    squeezes read identically -- so one path serves every era
    of the follows-the-crown seat."""
    v = build_champ_vision(0, fetch=fetch)
    rows = sorted(v.get("eyes_rows") or [], key=lambda r: -abs(r[2]))[:8]
    return v.get("p_up"), v.get("ts"), [
        {"name": r[1], "value": r[2]} for r in rows]


def _master_identity_book1() -> dict:
    """Book #1's identity from the driver's-seat manifest
    (data/live_seat.json): PPO run + stacker run/feature set;
    indicators_n is the prod21 feature count the seat stacker
    reads (features.technical_indicators.FEATURE_COLUMNS)."""
    from features.technical_indicators import FEATURE_COLUMNS

    seat = (_seat_manifest() or {}).get("seat") or {}
    ppo = seat.get("ppo") or {}
    stk = seat.get("stacker") or {}
    return {"brain_run": ppo.get("run_id"),
            "brain_label": f"PPO · Run {ppo.get('run_id')}",
            "eyes_run": stk.get("run_id"),
            "eyes_label": f"Stacker · Run {stk.get('run_id')} · "
                          f"{stk.get('feature_set')}",
            "feature_set": stk.get("feature_set"),
            "indicators_n": len(FEATURE_COLUMNS),
            "gate_run": None, "gate_threshold": None}


DECISIONS_DIR = DATA / "decisions"


def _decision_log_block(book_id, limit: int = 8):
    """One book's per-candle decision log for the master feed:
    (decision_log, latest_thought) read from the loop-written
    data/decisions/<book_id>.jsonl (src/trading/decision_log.py).
    decision_log is the last `limit` records OLDEST to NEWEST
    with the features object stripped (it would repeat per
    record); latest_thought is the newest record whole, with
    its features DIETED to the strongest 12 by absolute value
    (2026-11-01 transport diet; 16 in diet round 6, 2026-11-12;
    12 in diet round 7, 2026-11-17 -- the Master panel displays
    the strongest 12, so the trim is invisible) when the
    loop recorded them. A book with no file yet (logging starts
    with its next processed candle; history is never backfilled)
    gets ([], None). Reads a bounded tail -- the file grows one
    line per candle forever, and 8 records are always well
    inside it. Diet round 7 also cut the depth 12 -> 8 (Kass's
    2026-11-17 order: stop hitting size limits for good); the
    full log stays on disk and on each book's own desk."""
    path = DECISIONS_DIR / f"{book_id}.jsonl"
    if not path.exists():
        return [], None
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > 1_000_000:
                fh.seek(size - 1_000_000)
            raw = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return [], None
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue  # partial first line of the tail window
    if not records:
        return [], None

    def _feed_copy(r):
        # Feed-presentation diet ONLY (the JSONL on disk keeps full
        # precision): feature values round to 4dp and positions to
        # 6dp -- full float reprs of the 98 scored features alone
        # cost ~4.6KB per book in latest_thought.
        # Diet round 3 (2026-11-05): price/money floats round to
        # 2dp -- the precision the desk renders prices at, so every
        # rendered value is identical, while a full float repr
        # (stop 82132.24855655871, 17 chars) otherwise costs ~8
        # extra bytes on each of 16 rows x 3 books plus
        # latest_thought. Pinned by test_decision_log's
        # price-precision test.
        out = dict(r)
        for k in ("pos_before", "pos_after"):
            if isinstance(out.get(k), float):
                out[k] = round(out[k], 6)
        for k in ("stop_level", "close", "equity"):
            if isinstance(out.get(k), float):
                out[k] = round(out[k], 2)
        feats = out.get("features")
        if isinstance(feats, dict):
            out["features"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in feats.items()}
        return out

    latest = _feed_copy(records[-1])
    # Transport diet (2026-11-01): the full master payload hit
    # ~37.7KB and push_master rejected it in transport (a 34.3KB
    # build had succeeded earlier the same night). The two
    # 98-feature latest_thought blocks cost ~3.6KB each, so the
    # feed now carries only the STRONGEST 24 features by absolute
    # value (the Master panel displays the strongest 12 -- no
    # visible change; diet round 6, 2026-11-12, carried the
    # strongest 16; diet round 7, 2026-11-17, carries exactly
    # the displayed 12) plus "features_total", the full recorded
    # count, so the panel's "Strongest 12 of N" caption can stay
    # truthful. The JSONL on disk keeps ALL features at full
    # precision; this trim is feed-only, like the rounding above.
    feats = latest.get("features")
    if isinstance(feats, dict):
        latest["features_total"] = len(feats)
        if len(feats) > 12:
            def _strength(kv):
                v = kv[1]
                return (0, -abs(v)) if isinstance(v, (int, float)) \
                    else (1, 0.0)

            latest["features"] = dict(
                sorted(feats.items(), key=_strength)[:12])
    # Diet round 2 (2026-11-02, record slimming named in
    # test_master_feed.py's sentinel comment): feed rows also omit
    # None-valued keys -- an ungated book's rows otherwise repeat
    # gate_proba/gate_threshold/gate_verdict/stop_level nulls 16
    # times per book. Non-null values (a real Book #3 veto row's
    # gate fields) pass through verbatim, latest_thought above
    # stays the newest record WHOLE, and the on-disk JSONL is
    # untouched, as with every feed-only diet here.
    log = [{k: v for k, v in r.items()
            if k != "features" and v is not None}
           for r in (_feed_copy(r) for r in records[-int(limit):])]
    return log, latest


# Diet round 5 (2026-11-05, Kass's 2026-11-04 standing
# directive): the master's ROW sections ship as POSITIONAL
# ARRAYS. Book #4 (added 2026-11-05) took the with-vision
# payload to ~34.5KB with its sections still nearly empty;
# filling Book #4 to the same caps as a mature book projects
# ~40.9KB -- ~5.6KB OVER the ~35,360B push-transport truncation
# point -- and the repeated per-row key names are the cost:
# 64 decision rows x ~130B of keys, 48 fills, 48 decisions,
# 10 vetoes, 32 gauges. The keys now ride ONCE in these column
# lists, mirrored by the Master client's feed.ts (DECISION_LOG_
# COLS / FILL_COLS / DECISION_COLS / VETO_COLS / GAUGE_COLS in
# client/src/feed.ts, expanded back to objects at read time;
# the client also passes object rows through unchanged, so the
# two formats can deploy in either order). Every value is
# carried verbatim -- expansion reproduces the dict rows
# exactly (None where the round-2 elision dropped a null key;
# the client already treats absent == null). Measured on the
# 2026-11-05 17:07 ET payload: 34,502B -> 25,859B; on a
# plateau simulation (Book #4 filled to caps): 40,929B ->
# 30,306B, ~5.1KB under the truncation point. Pinned by
# test_master_feed's round-5 tests (positional shape, exact
# expansion vs _decision_log_block / snapshot / trades table /
# gate_decisions, unknown-key fallback). Display-changing
# levers stayed untouched: decision_log depth is still 16 and
# latest_thought still carries the strongest 24 features.
MASTER_DECISION_LOG_COLS = (
    "ts", "close", "p_up", "stance", "action", "taken", "reason",
    "gate_proba", "gate_threshold", "gate_verdict", "pos_before",
    "pos_after", "equity", "stop_level", "exposure_pct")
MASTER_FILL_COLS = ("ts", "action", "price", "qty", "reason")
MASTER_DECISION_COLS = ("ts", "action", "reason", "p_up")
MASTER_VETO_COLS = ("ts", "proba", "threshold", "verdict")
MASTER_GAUGE_COLS = ("name", "value")


def _master_row_array(row, cols):
    """One master feed row as a positional array in `cols` order
    (None for absent keys). A row carrying a key outside `cols`
    passes through as the original dict: a future field degrades
    that one row to the old format instead of losing data."""
    if not isinstance(row, dict) or not set(row) <= set(cols):
        return row
    return [row.get(c) for c in cols]


def _master_book(book_id, db_path, state_path, identity, *,
                 kind, label, start_equity, include_vision=True,
                 vision_fn=None, drift_bps=None, vetoes=None) -> dict:
    """One book's master block from its own snapshot + ledger.
    db_path/state_path None selects Book #1's production files
    (snapshot's own defaults), exactly as the per-book compact
    builders call it."""
    if db_path is None:
        s = snapshot.build_trading_snapshot()
    else:
        s = snapshot.build_trading_snapshot(db_path=db_path,
                                            state_path=state_path)
    if s.get("status") != "live":
        s = snapshot._fresh_account_snapshot({"as_of": s.get("as_of")})
    con = snapshot._paper_db(db_path)
    if con is not None:
        con.row_factory = snapshot.sqlite3.Row
    closed = snapshot.closed_trades(con) if con is not None else []
    stats = (snapshot.trade_stats(con) if con is not None
             else s["trade_stats"])
    streak = (snapshot.activity_stats(con)["streak"]
              if con is not None else s["activity"]["streak"])
    fills = []
    if con is not None:
        fills = con.execute(
            "SELECT ts, side, price, qty, fee, reason, context "
            "FROM trades ORDER BY id ASC").fetchall()
        con.close()
    pnls = [c["pnl"] for c in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    # Position age: the fill that took the book off flat for the
    # position it still holds (signed walk, same semantics as
    # snapshot.closed_trades' trip definition).
    pos = 0.0
    run_start = None
    for ts, side, _px, qty, _fee, _r, _c in fills:
        delta = qty if side == "BUY" else -qty
        if pos == 0.0 and delta != 0:
            run_start = ts
        pos += delta
        if abs(pos) <= 1e-10:
            pos = 0.0
            run_start = None
    equity = s["equity"]
    mark = round(s["btc_price"], 2) if s["btc_price"] is not None else None
    side = s["position"]
    entry = s["entry_price"]
    stop = s["stop_price"]
    stop_dist = None
    if side != "FLAT" and stop is not None and mark:
        stop_dist = round(
            100.0 * ((mark - stop) if side == "LONG" else (stop - mark))
            / mark, 2)
    dd = _drawdown_stats(s)
    curve_full = s["equity_curve"]
    peak = 0.0
    max_dd_usd = 0.0
    for pt in curve_full:
        peak = max(peak, pt["equity"])
        max_dd_usd = max(max_dd_usd, peak - pt["equity"])
    p_up = (round(s["stacker_p_up"], 4)
            if s["stacker_p_up"] is not None else None)
    vision_ts = s["last_candle"]
    top_gauges: list = []
    if include_vision and vision_fn is not None:
        try:
            _v_p, _v_ts, v_g = vision_fn()
            # p_up / vision_ts are NOT taken from the recomputed
            # vision (2026-11-03 fix): they stay this book's OWN
            # recorded vote -- snapshot's stacker_p_up from this
            # book's own state file, which is the value its loop
            # recorded in its decision log for vision_ts. The
            # recomputation is built at master-build time on the
            # newest closed candle and, for Books #2/#3 (identical
            # run-73 eyes, byte-identical seat artifacts), returns
            # the SAME value for both books; when it ran a candle
            # ahead of Book #2's loop (whose loop processes later
            # than Book #3's), Book #2's headline carried Book
            # #3's recorded vote instead of its own (saved payload
            # 2026-11-03_1307: book2 p_up 0.4875 == book3's
            # latest_thought @17:00, book2's own @16:00 = 0.4906).
            # The recomputed vision here supplies only top_gauges.
            top_gauges = v_g or []
        except Exception:
            top_gauges = []  # vision is garnish; the book still reports
    decisions = []
    for ts, fside, _px, _q, _fee, reason, ctx in reversed(fills[-10:]):
        p = None
        if ctx:
            try:
                p = json.loads(ctx).get("p")
            except Exception:
                p = None
        decisions.append({"ts": ts, "action": fside, "reason": reason,
                          "p_up": p})
    decisions = [_master_row_array(d, MASTER_DECISION_COLS)
                 for d in decisions]
    last = fills[-1] if fills else None
    dec_log, latest_thought = _decision_log_block(book_id)
    book = {
        "id": book_id,
        "label": label,
        "kind": kind,
        "identity": identity,
        "start_equity": start_equity,
        "equity": equity,
        "cash": s["cash"],
        "pnl": round(equity - start_equity, 2),
        "pnl_pct": round(100.0 * (equity - start_equity) / start_equity, 2)
        if start_equity else None,
        "btc_held": s["btc"],
        "side": side,
        "exposure_pct": round(100.0 * abs(s["btc_value"]) / equity, 2)
        if equity else 0.0,
        "entry_price": entry,
        "mark_price": mark,
        "unrealized": round(s["btc"] * (mark - entry), 2)
        if s["btc"] and entry and mark else 0.0,
        "stop_level": stop,
        "stop_anchor": s["stop_anchor"],
        "stop_distance_pct": stop_dist,
        "trailing_active": bool(s["stop_trailing"]),
        "position_since": run_start if side != "FLAT" else None,
        "last_action": last[1] if last else None,
        "last_action_ts": last[0] if last else None,
        "last_reason": last[5] if last else None,
        "stance": s["agent_stance"],
        "p_up": p_up,
        "vision_ts": vision_ts,
        "closed_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": stats["win_pct"],
        "profit_factor": stats["profit_factor"] if n else None,
        "expectancy": stats["expectancy"] if n else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "payoff": (round((sum(wins) / len(wins))
                         / abs(sum(losses) / len(losses)), 2)
                   if wins and losses else None),
        "best_trade": round(max(pnls), 2) if pnls else None,
        "worst_trade": round(min(pnls), 2) if pnls else None,
        "streak": streak,
        "fees_total": round(sum(f[4] for f in fills), 2),
        "drift_bps": drift_bps,
        "max_drawdown": round(max_dd_usd, 2) if curve_full else None,
        "max_dd_pct": (round(-dd["dd_max"], 2) + 0.0
                       if dd["dd_max"] is not None else None),
        "cur_drawdown": dd["dd_now"],
        "dd_recovered": (dd["dd_now"] == 0
                         if dd["dd_now"] is not None else None),
        "equity_curve": [[p["t"], p["equity"]]
                         for p in _downsample_curve(curve_full)],
        "dd_curve": _downsample_pairs(
            _compress_flat_runs(
                [[p["t"], p["dd_pct"]] for p in s["drawdown_curve"]]),
            # Diet round 2 (2026-11-02): cap 120 -> 64 -- the
            # dd-curve stride the test_master_feed sentinel named.
            # _downsample_pairs keeps the newest 8 points full-res;
            # 64 points over the whole book history is ~7px/point
            # at phone width, past what the chart can resolve.
            # Diet round 3 (2026-11-05): cap 64 -> 40, the module's
            # own race-curve cap -- the round-2 steady state still
            # outgrew the push transport (with-vision payload
            # 35,360B rejected twice; see the sentinel comment in
            # test_master_feed.py). Retained points stay
            # verbatim points of the flat-run-compressed series
            # (pinned by test_master_dd_curve_stride_diet_keeps_
            # points_verbatim); at 40 points the strip is
            # ~10px/point.
            # Diet round 4 (2026-11-05, later the same day): cap
            # 40 -> 32. Section-by-section measurement against the
            # round-3 pushed payload showed the dd curve was the
            # ONE master section whose size still depended on
            # flat-run compression luck: as BTC dropped on
            # 2026-11-05, Book #3's dd rode 30 -> 39 points
            # (917B -> 1,193B) because a deepening drawdown has
            # no flat runs to compress, and the no-vision build
            # drifted 32,241B -> 32,507B onto the sentinel. At
            # cap 32 the bound is deterministic -- worst case
            # 34 points x <= 31B (19-char ts, 3dp value) no matter
            # how volatile the series -- and the strip is
            # ~12px/point at phone width, past chart resolution,
            # same stride class as rounds 2-3. Pinned by
            # test_master_dd_curve_worst_case_volatile_series_
            # bounded (a zero-flat-run series, where compression
            # is the identity and the cap is the only protection).
            # Diet round 6 (2026-11-12): cap 32 -> 24 when Book
            # #5 joined the master feed (Kass's approval) -- the
            # five-book no-vision build measured 30,673B against
            # the 31,000 sentinel with Book #5 still nearly
            # empty; at cap 24 the deterministic worst case is
            # 26 points/book and the strip is ~16px/point at
            # phone width over a book's full history.
            # Diet round 7 (2026-11-17, Kass: stop hitting size
            # limits for good): cap 24 -> 20 (worst case 22
            # points/book, ~18px/point) as part of the structural
            # round that makes the five-book PLATEAU fit the
            # 31,000 sentinel with margin -- see the round-7 note
            # in test_master_feed.py.
            cap=20),
        "recent_fills": [
            _master_row_array(
                {"ts": r["ts"], "action": r["side"],
                 "price": r["price"], "qty": r["qty"],
                 "reason": r["reason"]}, MASTER_FILL_COLS)
            for r in s["recent_trades"][:10]],
        "recent_decisions": decisions,
        "decision_log": [
            _master_row_array(r, MASTER_DECISION_LOG_COLS)
            for r in dec_log],
        "latest_thought": latest_thought,
        "top_gauges": [
            _master_row_array(g, MASTER_GAUGE_COLS)
            for g in top_gauges],
        "health": {"last_candle_ts": s["last_candle"],
                   "feed_as_of": s["as_of"]},
    }
    if vetoes is not None:
        book["vetoes"] = [
            _master_row_array(v, MASTER_VETO_COLS) for v in vetoes]
    return book


def _master_gate_vetoes(db_path, limit: int = 8) -> list:
    """A gated book's newest gate decisions from its engine's
    own gate_decisions table (threshold carried per row as
    stored -- Book #3's rows carry 0.55, Book #2's carry 0.51
    since the gate joined its seat on 2026-11-21)."""
    import sqlite3

    if not db_path.exists():
        return []
    try:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                "SELECT ts, proba, threshold, verdict FROM "
                "gate_decisions ORDER BY id DESC LIMIT ?",
                (int(limit),)).fetchall()
        finally:
            con.close()
    except Exception:
        return []
    return [{"ts": ts, "proba": round(p, 4), "threshold": thr,
             "verdict": v} for ts, p, thr, v in rows]


def _master_book3_vetoes(limit: int = 8) -> list:
    """Book #3's newest gate decisions (the same rows
    build_gated_veto renders)."""
    return _master_gate_vetoes(GATED_DB, limit)


def _master_book2_vetoes(limit: int = 8) -> list:
    """Book #2's newest gate decisions (the same rows
    build_shadow_veto renders)."""
    return _master_gate_vetoes(SHADOW_DB, limit)


def build_master(include_vision: bool = True) -> dict:
    """The master live-desk feed: Books #1/#2/#3/#4/#5 in one
    payload (Book #5 joined 2026-11-12 on Kass's approval).
    include_vision=False skips the (network + frame) vision
    builders (top_gauges is then empty). Either way, each book's
    p_up/vision_ts are that book's OWN recorded loop state -- the
    same values its compact trading feed and decision log carry
    -- never the recomputed vision's vote (2026-11-03 fix in
    _master_book: the recomputation made Book #2 carry Book
    #3's recorded vote whenever it ran ahead of Book #2's loop).
    With vision on, all three books' vision paths share ONE
    memoized market fetch (_master_shared_fetch, 2026-11-04):
    the 730d series is fetched once per build, not three times."""
    desk2 = _shadow_desk_block()
    desk3 = _gated_desk_block()
    desk4 = _ha_desk_block()
    man2 = _shadow_manifest()
    man3 = _gated_manifest()
    man4 = _ha_manifest()
    desk5 = _champ_desk_block()
    man5 = _champ_manifest()
    drift1 = None
    try:
        drift1 = build_ledger().get("friction", {}).get("drift_bps")
    except Exception:
        drift1 = None
    shared_fetch = _master_shared_fetch() if include_vision else None
    books = [
        _master_book(
            "book1", None, None, _master_identity_book1(),
            kind="driver", label=B1_LABEL, start_equity=100000.0,
            include_vision=include_vision,
            vision_fn=lambda: _master_vision_book1(fetch=shared_fetch),
            drift_bps=drift1),
        _master_book(
            "book2", SHADOW_DB, SHADOW_STATE,
            {"brain_run": desk2["driver"]["run"],
             "brain_label": desk2["driver"]["label"],
             "eyes_run": desk2["eyes"]["run"],
             "eyes_label": desk2["eyes"]["label"],
             "feature_set": desk2["eyes"]["feature_set"],
             "indicators_n": desk2["eyes"]["n_features"],
             "gate_run": (desk2.get("gate") or {}).get("run"),
             "gate_threshold": (desk2.get("gate") or {}).get(
                 "threshold")},
            kind="shadow", label=desk2["book_name"],
            start_equity=float(man2.get("starting_balance", 100000)),
            include_vision=include_vision,
            vetoes=_master_book2_vetoes(),
            # Book #2's vision follows its seat's frame: scored
            # gauges while a scored champion held the seat, the HA
            # gauges since Champion #11 (ha10) took it 2026-11-09.
            vision_fn=(lambda: _master_vision_ha(
                build_shadow_vision, fetch=shared_fetch))
            if desk2["eyes"]["feature_set"] == "ha10" else
            (lambda: _master_vision_scored(
                build_shadow_vision, _shadow_frame_builder,
                fetch=shared_fetch))),
        _master_book(
            "book3", GATED_DB, GATED_STATE,
            {"brain_run": desk3["driver"]["run"],
             "brain_label": desk3["driver"]["label"],
             "eyes_run": desk3["eyes"]["run"],
             "eyes_label": desk3["eyes"]["label"],
             "feature_set": desk3["eyes"]["feature_set"],
             "indicators_n": desk3["eyes"]["n_features"],
             "gate_run": desk3["gate"]["run"],
             "gate_threshold": desk3["gate"]["threshold"]},
            kind="gated", label=desk3["book_name"],
            start_equity=float(man3.get("starting_balance", 100000)),
            include_vision=include_vision,
            vision_fn=lambda: _master_vision_scored(
                build_gated_vision, _gated_frame_builder,
                fetch=shared_fetch),
            vetoes=_master_book3_vetoes()),
        _master_book(
            "book4", HA_DB, HA_STATE,
            {"brain_run": desk4["driver"]["run"],
             "brain_label": desk4["driver"]["label"],
             "eyes_run": desk4["eyes"]["run"],
             "eyes_label": desk4["eyes"]["label"],
             "feature_set": desk4["eyes"]["feature_set"],
             "indicators_n": desk4["eyes"]["n_features"],
             "gate_run": None, "gate_threshold": None},
            kind="ha", label=desk4["book_name"],
            start_equity=float(man4.get("starting_balance", 100000)),
            include_vision=include_vision,
            vision_fn=lambda: _master_vision_ha(
                build_ha_vision, fetch=shared_fetch)),
        _master_book(
            "book5", CHAMP_DB, CHAMP_STATE,
            {"brain_run": desk5["driver"]["run"],
             "brain_label": desk5["driver"]["label"],
             "eyes_run": desk5["eyes"]["run"],
             "eyes_label": desk5["eyes"]["label"],
             "feature_set": desk5["eyes"]["feature_set"],
             "indicators_n": desk5["eyes"]["n_features"],
             "gate_run": None, "gate_threshold": None},
            kind="champ", label=desk5["book_name"],
            start_equity=float(man5.get("starting_balance", 100000)),
            include_vision=include_vision,
            vision_fn=lambda: _master_vision_champ(
                fetch=shared_fetch)),
    ]
    race_bh = None
    try:
        # Diet round 4 (2026-11-05): the master's copy of the
        # buy-and-hold race curve is strided to cap 32 -- the
        # same deterministic bound as the master dd curves.
        # build_shadow_race's own cap-40 curve is untouched (it
        # also serves the shadow desk at its own resolution);
        # the stride keeps verbatim points of that curve only,
        # endpoints and newest 8 at full resolution.
        # Diet round 7 (2026-11-17): cap 32 -> 24, matching the
        # round-7 dd-curve class so the plateau fits with margin.
        race_bh = _downsample_pairs(
            build_shadow_race()["bh"]["curve"], cap=24) or None
    except Exception:
        race_bh = None
    combined = {
        "equity": round(sum(b["equity"] for b in books), 2),
        "pnl": round(sum(b["pnl"] for b in books), 2),
        "exposure_usd": round(
            sum(abs(b["btc_held"] * (b["mark_price"] or 0.0))
                for b in books), 2),
        "books_in_profit": sum(1 for b in books if b["pnl"] > 0),
        "books_in_drawdown": sum(
            1 for b in books
            if b["cur_drawdown"] is not None and b["cur_drawdown"] < 0),
    }
    return {
        "as_of": books[0]["health"]["feed_as_of"],
        "btc": {"price": books[0]["mark_price"],
                "candle_ts": books[0]["health"]["last_candle_ts"]},
        "books": books,
        "combined": combined,
        "race_bh": race_bh,
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "trading"
    builders = {
        "trading": build_compact_trading,
        "ledger": build_ledger,
        "market0": lambda: build_market(0),
        "market1": lambda: build_market(1),
        "vision": build_vision,
        "fillctx": build_fillctx,
        "training": build_compact_training,
        "training_models": build_models_manifest,
        "gate": build_gate,
        "shadow": build_shadow_compact_trading,
        "shadow_ledger": build_shadow_ledger,
        "shadow_market0": lambda: build_shadow_market(0),
        "shadow_market1": lambda: build_shadow_market(1),
        "shadow_vision0": lambda: build_shadow_vision(0),
        "shadow_vision1": lambda: build_shadow_vision(1),
        "shadow_race": build_shadow_race,
        "shadow_veto": build_shadow_veto,
        "gated": build_gated_compact_trading,
        "gated_ledger": build_gated_ledger,
        "gated_market0": lambda: build_gated_market(0),
        "gated_market1": lambda: build_gated_market(1),
        "gated_vision0": lambda: build_gated_vision(0),
        "gated_vision1": lambda: build_gated_vision(1),
        "gated_race": build_gated_race,
        "gated_veto": build_gated_veto,
        "ha": build_ha_compact_trading,
        "ha_ledger": build_ha_ledger,
        "ha_market0": lambda: build_ha_market(0),
        "ha_market1": lambda: build_ha_market(1),
        "ha_vision0": lambda: build_ha_vision(0),
        "ha_vision1": lambda: build_ha_vision(1),
        "ha_race": build_ha_race,
        "champ": build_champ_compact_trading,
        "champ_ledger": build_champ_ledger,
        "champ_market0": lambda: build_champ_market(0),
        "champ_market1": lambda: build_champ_market(1),
        "champ_vision0": lambda: build_champ_vision(0),
        "champ_vision1": lambda: build_champ_vision(1),
        "champ_indicators": build_champ_indicators,
        "champ_race": build_champ_race,
        "master": build_master,
    }
    if which == "training_models_page":
        # Page index rides argv[2]: `compact.py training_models_page 0`.
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        out = build_models_pages()[idx]
    else:
        out = builders[which]()
    text = json.dumps(out, separators=(",", ":"))
    sys.stderr.write(f"bytes={len(text)}\n")
    print(text)
