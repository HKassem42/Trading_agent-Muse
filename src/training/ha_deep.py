"""HA-Deep study machinery (runs 87/88) — pre-registered in
refs/ha-deep-study-2026-10-30.md.

HA-native feature frame + streak/shift state machine + Kass's HA Shift
Trader spans (judged and pure forms). The HA series extends the production
recursion in src/features/technical_indicators.py (imported, never
re-derived); all multi-timeframe state is bucket-END stamped and
values-based (the v5.1 lesson). Lab-only: nothing here is imported by a
live loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # package context (src.training.ha_deep)
    from ..features.technical_indicators import _atr, _heikin_ashi, _rsi
    from ..trading.broker import PaperBroker
except ImportError:  # project convention: src/ on sys.path
    from features.technical_indicators import _atr, _heikin_ashi, _rsi
    from trading.broker import PaperBroker

FAMILY = "ha_deep"
LABEL = "HA-Deep Eyes"
SHIFT_FAMILY = "ha_shift"
SHIFT_LABEL = "HA Shift Trader"

HORIZON = 24
SEQ_LEN = 24
JUDGE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Frozen 58-feature frame: Block 1 anatomy/streaks (22), Block 2 main-chart
# on HA (21), Block 3 sub-chart on HA (15). Order is the registry contract.
# ---------------------------------------------------------------------------
BLOCK1_COLUMNS = [
    "ha_body_atr", "ha_body_ratio", "ha_upwick_atr", "ha_lowick_atr",
    "ha_range_atr", "ha_color", "ha_streak", "ha_streak_body",
    "ha_streak_rel", "ha_gap_ma6", "ha_ret6", "ha_ret24", "ha_ret96",
    "ha_dist_sma50", "ha_dist_sma200", "ha_sma50_slope", "ha_raw_gap_atr",
    "ha_raw_agree", "ha4_color", "ha4_streak", "ha1d_color", "ha1d_streak",
]
BLOCK2_COLUMNS = [
    "ha_dist_ema9", "ha_dist_ema21", "ha_dist_ema50", "ha_dist_ema200",
    "ha_ema9_slope", "ha_ema21_slope", "ha_ema50_slope", "ha_ema200_slope",
    "ha_dist_sma9", "ha_dist_sma21", "ha_sma9_slope", "ha_sma21_slope",
    "ha_bb_pos", "ha_bb_width_atr", "ha_ichi_tk_gap", "ha_ichi_close_kijun",
    "ha_ichi_vs_cloud", "ha_ichi_cloud_thick", "ha_ichi_in_cloud",
    "ha_keltner_pos", "ha_dist_vwap24",
]
BLOCK3_COLUMNS = [
    "ha_rsi14", "ha_macd", "ha_macd_signal", "ha_macd_hist", "ha_stoch_k",
    "ha_stoch_d", "ha_adx", "ha_di_gap", "ha_cci", "ha_willr", "ha_ewo",
    "ha_atr_pct", "ha_obv_slope", "ha_adl_slope", "ha_netvol_rel",
]
FEATURE_COLUMNS_HA = BLOCK1_COLUMNS + BLOCK2_COLUMNS + BLOCK3_COLUMNS
BLOCK_OF = ({c: 1 for c in BLOCK1_COLUMNS} | {c: 2 for c in BLOCK2_COLUMNS}
            | {c: 3 for c in BLOCK3_COLUMNS})

SEQ_SERIES = ["s_body_atr", "s_upwick_atr", "s_lowick_atr", "s_range_atr",
              "s_streak", "s_haret"]


def guard_hashes(root=None) -> dict[str, str]:
    """SHA-256 of every file the study must never move (the eyes_v2 guard
    set: live/shadow/gated seats + lab champions + all three live-state
    files strict; the three paper DBs information-only). The eyes_v2
    guard set is path-absolute; `root` is accepted for caller
    compatibility and ignored."""
    try:
        from . import eyes_v2
    except ImportError:
        from training import eyes_v2
    return eyes_v2.guard_hashes()


# ---------------------------------------------------------------------------
# HA series
# ---------------------------------------------------------------------------
def ha_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Full HA OHLC on the raw frame. Open/close come from the production
    `_heikin_ashi`; high/low use the standard completion (extremes of the
    raw candle and the HA open/close)."""
    ha_open, ha_close = _heikin_ashi(df)
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        "ha_open": ha_open, "ha_high": ha_high,
        "ha_low": ha_low, "ha_close": ha_close,
    }, index=df.index)


def ha_colors(ha: pd.DataFrame) -> np.ndarray:
    """+1 green (body > 0), -1 red (body < 0), 0 neutral (body == 0.0
    exactly — a neutral candle ends the current run and starts none)."""
    body = (ha["ha_close"] - ha["ha_open"]).to_numpy(dtype=float)
    return np.sign(body).astype(int)


def run_state(colors: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row run structure: `streak` (signed length of the run up to and
    including t; 0 on neutral rows), `run_id` (-1 on neutral rows), and
    `prev_run_color` (color of the nonzero run immediately before the run
    containing t — for neutral rows, the last completed nonzero run's
    color; 0 when no nonzero run has completed/started yet)."""
    n = len(colors)
    streak = np.zeros(n, dtype=int)
    run_id = np.full(n, -1, dtype=int)
    prev_run_color = np.zeros(n, dtype=int)
    cur_color = 0
    cur_len = 0
    cur_id = -1
    next_id = 0
    last_completed_color = 0
    for t in range(n):
        c = int(colors[t])
        if c == 0:
            if cur_color != 0:
                last_completed_color = cur_color
            cur_color = 0
            cur_len = 0
            cur_id = -1
            prev_run_color[t] = last_completed_color
            continue
        if c == cur_color:
            cur_len += 1
        else:
            if cur_color != 0:
                last_completed_color = cur_color
            cur_color = c
            cur_len = 1
            cur_id = next_id
            next_id += 1
        streak[t] = cur_color * cur_len
        run_id[t] = cur_id
        prev_run_color[t] = last_completed_color
    return {"streak": streak, "run_id": run_id,
            "prev_run_color": prev_run_color}


def shift_events(colors: np.ndarray, confirm_len: int = 2) -> np.ndarray:
    """Confirmed-shift events: at row t, the event value is the run's color
    (+1/-1) if the run containing t has just reached length `confirm_len`
    AND the run is a shift (its color differs from the immediately
    preceding nonzero run's color). 0 everywhere else. The first-ever run
    is never a shift; a run repeating the previous nonzero color after a
    neutral gap is not a shift. Fires at most once per run."""
    st = run_state(colors)
    streak, prev = st["streak"], st["prev_run_color"]
    events = np.zeros(len(colors), dtype=int)
    for t in range(len(colors)):
        s = int(streak[t])
        if s == 0 or abs(s) != confirm_len:
            continue
        color = 1 if s > 0 else -1
        if prev[t] != 0 and prev[t] != color:
            events[t] = color
    return events


# ---------------------------------------------------------------------------
# Multi-timeframe HA state (bucket-END stamped, values-based; the v5.1
# construction copied from technical_indicators.py's h4 pattern).
# ---------------------------------------------------------------------------
def _mtf_ha_state(df: pd.DataFrame, rule: str, bucket: pd.Timedelta):
    agg = df.resample(rule, closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    ha_b = ha_candles(agg)
    colors_b = ha_colors(ha_b)
    streak_b = run_state(colors_b)["streak"]
    ends = agg.index + bucket  # bucket-END stamps (decision-time axes)
    return ends, colors_b.astype(float), streak_b.astype(float)


def _merge_mtf(df: pd.DataFrame, rule: str, bucket: pd.Timedelta):
    ends, col_b, str_b = _mtf_ha_state(df, rule, bucket)
    right = pd.DataFrame({
        "decision_ts": ends, "color": col_b, "streak": str_b,
    })
    base = pd.DataFrame({"decision_ts": df.index + pd.Timedelta(hours=1)})
    merged = pd.merge_asof(base.sort_values("decision_ts"), right,
                           on="decision_ts", direction="backward")
    return (merged["color"].to_numpy(dtype=float),
            merged["streak"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# The frozen 58-feature frame
# ---------------------------------------------------------------------------
def build_ha_frame(df: pd.DataFrame):
    """Build the HA-native frame over the full raw frame `df`.

    Returns (frame58, seq_frame, aux):
      frame58   — DataFrame on df.index with FEATURE_COLUMNS_HA (58 cols);
      seq_frame — DataFrame on df.index with the 6 sequence series;
      aux       — dict with the HA candles, colors, signed streak,
                  confirmed-shift event arrays (L=2 and L=3), the
                  production raw atr_pct series, and the HA ATR series.
    """
    idx = df.index
    ha = ha_candles(df)
    hao, hah, hal, hac = (ha["ha_open"], ha["ha_high"], ha["ha_low"],
                          ha["ha_close"])
    body = hac - hao
    ha_atr = _atr(pd.DataFrame({"high": hah, "low": hal, "close": hac},
                               index=idx), 14)
    colors = ha_colors(ha)
    st = run_state(colors)
    streak = st["streak"].astype(float)
    run_id = st["run_id"]
    vol = df["volume"]

    f: dict[str, pd.Series] = {}

    # ---- Block 1: anatomy & streaks (22) ----
    body_atr = body / ha_atr
    rng = hah - hal
    f["ha_body_atr"] = body_atr
    f["ha_body_ratio"] = (body / rng).where(rng != 0, 0.0)
    f["ha_upwick_atr"] = (hah - pd.concat([hao, hac], axis=1).max(axis=1)) / ha_atr
    f["ha_lowick_atr"] = (pd.concat([hao, hac], axis=1).min(axis=1) - hal) / ha_atr
    f["ha_range_atr"] = rng / ha_atr
    f["ha_color"] = pd.Series(colors.astype(float), index=idx)
    f["ha_streak"] = pd.Series(streak, index=idx)
    # streak cumulative body (sum of ha_body_atr within the current run)
    ba = body_atr.to_numpy(dtype=float)
    ba_ok = np.isfinite(ba)
    streak_body = np.full(len(df), np.nan)
    run_sums: dict[int, float] = {}
    for t in range(len(df)):
        if run_id[t] < 0:
            streak_body[t] = 0.0
        elif not ba_ok[t]:
            streak_body[t] = np.nan
            run_sums[run_id[t]] = np.nan
        else:
            run_sums[run_id[t]] = run_sums.get(run_id[t], 0.0) + ba[t]
            streak_body[t] = run_sums[run_id[t]]
    f["ha_streak_body"] = pd.Series(streak_body, index=idx)
    # streak length vs trailing-500 completed runs
    n = len(df)
    end_len = np.zeros(n)
    end_cnt = np.zeros(n)
    for t in range(n):
        if run_id[t] >= 0 and (t == n - 1 or run_id[t + 1] != run_id[t]):
            end_len[t] = abs(streak[t])
            end_cnt[t] = 1.0
    cs_len = np.concatenate([[0.0], np.cumsum(end_len)])
    cs_cnt = np.concatenate([[0.0], np.cumsum(end_cnt)])
    streak_rel = np.ones(n)
    for t in range(n):
        lo = max(0, t - 500)
        s = cs_len[t] - cs_len[lo]
        c = cs_cnt[t] - cs_cnt[lo]
        if c > 0:
            streak_rel[t] = abs(streak[t]) / (s / c)
    f["ha_streak_rel"] = pd.Series(streak_rel, index=idx)
    f["ha_gap_ma6"] = body_atr.rolling(6).mean()
    f["ha_ret6"] = hac / hac.shift(6) - 1
    f["ha_ret24"] = hac / hac.shift(24) - 1
    f["ha_ret96"] = hac / hac.shift(96) - 1
    sma50 = hac.rolling(50).mean()
    sma200 = hac.rolling(200).mean()
    f["ha_dist_sma50"] = hac / sma50 - 1
    f["ha_dist_sma200"] = hac / sma200 - 1
    f["ha_sma50_slope"] = sma50 / sma50.shift(6) - 1
    f["ha_raw_gap_atr"] = (df["close"] - hac) / ha_atr
    body_sign_full = np.sign(body.to_numpy(dtype=float))
    raw_sign = np.sign((df["close"] - df["open"]).to_numpy(dtype=float))
    agree = np.where((raw_sign == body_sign_full) & (raw_sign != 0), 1.0,
                     np.where((raw_sign != 0) & (body_sign_full != 0),
                              -1.0, 0.0))
    f["ha_raw_agree"] = pd.Series(agree, index=idx)
    c4, s4 = _merge_mtf(df, "4h", pd.Timedelta(hours=4))
    c1, s1 = _merge_mtf(df, "1D", pd.Timedelta(days=1))
    f["ha4_color"] = pd.Series(c4, index=idx)
    f["ha4_streak"] = pd.Series(s4, index=idx)
    f["ha1d_color"] = pd.Series(c1, index=idx)
    f["ha1d_streak"] = pd.Series(s1, index=idx)

    # ---- Block 2: main-chart indicators on HA (21) ----
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    for span in (9, 21, 50, 200):
        e = _ema(hac, span)
        f[f"ha_dist_ema{span}"] = hac / e - 1
        f[f"ha_ema{span}_slope"] = e / e.shift(6) - 1
    for span in (9, 21):
        m = hac.rolling(span).mean()
        f[f"ha_dist_sma{span}"] = hac / m - 1
        f[f"ha_sma{span}_slope"] = m / m.shift(6) - 1
    mid = hac.rolling(20).mean()
    sd = hac.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    width = upper - lower
    f["ha_bb_pos"] = ((hac - lower) / width).where(width != 0, 0.5)
    f["ha_bb_width_atr"] = width / ha_atr
    tenkan = (hah.rolling(9).max() + hal.rolling(9).min()) / 2
    kijun = (hah.rolling(26).max() + hal.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hah.rolling(52).max() + hal.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    cloud_mid = (cloud_top + cloud_bot) / 2
    f["ha_ichi_tk_gap"] = (tenkan - kijun) / ha_atr
    f["ha_ichi_close_kijun"] = (hac - kijun) / ha_atr
    f["ha_ichi_vs_cloud"] = (hac - cloud_mid) / ha_atr
    f["ha_ichi_cloud_thick"] = (cloud_top - cloud_bot) / ha_atr
    in_cloud = ((hac >= cloud_bot) & (hac <= cloud_top)).astype(float)
    f["ha_ichi_in_cloud"] = in_cloud.where(cloud_top.notna())
    ema20 = _ema(hac, 20)
    k_up, k_lo = ema20 + 2 * ha_atr, ema20 - 2 * ha_atr
    k_width = k_up - k_lo
    f["ha_keltner_pos"] = ((hac - k_lo) / k_width).where(k_width != 0, 0.5)
    tp = (hah + hal + hac) / 3
    vwap = (tp * vol).rolling(24).sum() / vol.rolling(24).sum().replace(0, np.nan)
    f["ha_dist_vwap24"] = (hac - vwap) / ha_atr

    # ---- Block 3: sub-chart indicators on HA (15) ----
    f["ha_rsi14"] = ((_rsi(hac, 14) - 50) / 50).clip(-1, 1)
    macd_line = _ema(hac, 12) - _ema(hac, 26)
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    f["ha_macd"] = macd_line / hac
    f["ha_macd_signal"] = macd_sig / hac
    f["ha_macd_hist"] = (macd_line - macd_sig) / hac
    ll14 = hal.rolling(14).min()
    hh14 = hah.rolling(14).max()
    stoch_rng = hh14 - ll14
    k_pct = ((hac - ll14) / stoch_rng).where(stoch_rng != 0, 0.5)
    f["ha_stoch_k"] = k_pct
    f["ha_stoch_d"] = k_pct.rolling(3).mean()
    # ADX (Wilder smoothing via EWM alpha=1/14)
    prev_hac = hac.shift(1)
    tr_ha = pd.concat([hah - hal, (hah - prev_hac).abs(),
                       (hal - prev_hac).abs()], axis=1).max(axis=1)
    up_move = hah.diff()
    dn_move = -hal.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0),
                                 up_move.to_numpy(dtype=float), 0.0),
                        index=idx).where(up_move.notna())
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0),
                                  dn_move.to_numpy(dtype=float), 0.0),
                         index=idx).where(dn_move.notna())
    atr_s = tr_ha.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False,
                            min_periods=14).mean() / atr_s
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False,
                             min_periods=14).mean() / atr_s
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).where((pdi + mdi) != 0, 0.0)
    adx = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["ha_adx"] = adx / 100
    f["ha_di_gap"] = (pdi - mdi) / 100
    sma_tp = tp.rolling(20).mean()
    mad = (tp - sma_tp).abs().rolling(20).mean()
    f["ha_cci"] = ((tp - sma_tp) / (0.015 * mad)).where(mad != 0, 0.0) / 100
    f["ha_willr"] = (-(hh14 - hac) / stoch_rng).where(stoch_rng != 0, -0.5)
    f["ha_ewo"] = (_ema(hac, 5) - _ema(hac, 35)) / hac
    f["ha_atr_pct"] = ha_atr / hac
    vol24 = vol.rolling(24).sum().replace(0, np.nan)
    obv = pd.Series(np.cumsum(vol.to_numpy(dtype=float) * body_sign_full),
                    index=idx)
    f["ha_obv_slope"] = (obv.diff(24) / vol24).clip(-1, 1)
    mfm = (((hac - hal) - (hah - hac)) / rng).where(rng != 0, 0.0).fillna(0.0)
    adl = (mfm * vol).cumsum()
    f["ha_adl_slope"] = (adl.diff(24) / vol24).clip(-1, 1)
    net = pd.Series(vol.to_numpy(dtype=float) * body_sign_full,
                    index=idx).rolling(24).sum()
    f["ha_netvol_rel"] = (net / vol24).clip(-1, 1)

    frame = pd.DataFrame({c: f[c] for c in FEATURE_COLUMNS_HA}, index=idx)

    seq = pd.DataFrame({
        "s_body_atr": f["ha_body_atr"],
        "s_upwick_atr": f["ha_upwick_atr"],
        "s_lowick_atr": f["ha_lowick_atr"],
        "s_range_atr": f["ha_range_atr"],
        "s_streak": pd.Series(np.clip(streak / 12.0, -1.0, 1.0), index=idx),
        "s_haret": hac / hac.shift(1) - 1,
    }, index=idx)

    atr_raw = _atr(df, 14)
    aux = {
        "ha": ha,
        "colors": colors,
        "streak": st["streak"],
        "events2": shift_events(colors, 2),
        "events3": shift_events(colors, 3),
        "atr_pct_raw": atr_raw / df["close"],
        "ha_atr": ha_atr,
    }
    return frame, seq, aux


# ---------------------------------------------------------------------------
# Shared row/label helpers (spec in the pre-registration)
# ---------------------------------------------------------------------------
def common_training_rows(frame58: pd.DataFrame, pick2_finite: np.ndarray,
                         tb_labels: np.ndarray, nc_labels: np.ndarray,
                         split: int) -> np.ndarray:
    """Positions usable for training ALL arms fairly: rows 24.., < split,
    with all 58 HA features finite at t and across the trailing 24 rows,
    the Pick-2 frame finite at t, both labels defined, and the barrier
    window (t+24) strictly inside the training region."""
    x = frame58.to_numpy(dtype=float)
    finite_t = np.isfinite(x).all(axis=1)
    # trailing-25-row window (t-24..t) all-finite, via a cumsum of the
    # per-row non-finite counts:
    nonfinite_per_row = (~np.isfinite(x)).sum(axis=1)
    cs = np.concatenate([[0], np.cumsum(nonfinite_per_row)])
    bad = cs[SEQ_LEN + 1:] - cs[:len(frame58) - SEQ_LEN]
    window_ok = np.zeros(len(frame58), dtype=bool)
    window_ok[SEQ_LEN:] = bad == 0
    pos = np.arange(len(frame58))
    ok = (finite_t & window_ok & pick2_finite
          & np.isfinite(tb_labels) & np.isfinite(nc_labels)
          & (pos + HORIZON < split) & (pos >= SEQ_LEN))
    return pos[ok]


def judge_lot(p_up: float, event: int) -> float:
    """P(tradeable move in the shift direction): for a green shift the TB
    probability itself; for a red shift its complement (the TB label's
    0-class is the down-barrier-first outcome)."""
    return float(p_up) if event > 0 else float(1.0 - p_up)


def _shift_action(side: int, event: int) -> tuple[str, str]:
    """The frozen BUY/SHORT/HOLD truth table. `side` is the position
    sign AFTER the candle's stop check (+1 long / -1 short / 0 flat);
    `event` is the confirmed shift color (+1 green / -1 red). Returns
    (action, context): action in {"buy", "short", "hold"}; context in
    {"entry", "flip", "noop"} ("noop" = same-direction event while
    positioned — BUY while long / SHORT while short do nothing)."""
    if event > 0:  # green shift
        if side == 0:
            return "buy", "entry"
        if side < 0:
            return "buy", "flip"
        return "hold", "noop"
    # red shift
    if side == 0:
        return "short", "entry"
    if side > 0:
        return "short", "flip"
    return "hold", "noop"


def run_shift_span(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, atr_pct: np.ndarray,
                   events: np.ndarray, lot: np.ndarray, stamps,
                   db_path: Path, judged: bool) -> dict:
    """Run Kass's HA Shift Trader over one span on the production
    PaperBroker (flat $100k, 25% tranche, 75% cap, 0.15%/side).

    Per-candle ordering (frozen): (1) execute the pending action at
    the candle's OPEN; (2) broker.check_stop on the candle (wick
    trigger, close/low/high); (3) mark equity at the close; (4)
    evaluate the candle's confirmed-shift event against the
    post-stop position and set the pending action. In the JUDGED
    form, an actionable event fires only when lot[i] >=
    JUDGE_THRESHOLD ("a lot"); a skipped opposing shift does NOT
    close the position. An event on the span's last candle is a
    counted decision whose fill never lands inside the span.

    `events` / `lot` are full-length arrays aligned to the span's
    candles (event color +1/-1 or 0; lot = P(tradeable move in the
    shift direction), read only at event candles in judged spans).
    Returns the span record: pnl, max_drawdown, score, trades,
    fees, time_in_position, action_counts {hold, buy, short},
    judge_log (judged spans only), fills, final_equity."""
    import sqlite3

    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    atr = np.asarray(atr_pct, dtype=float)
    ev = np.asarray(events, dtype=int)
    lt = np.asarray(lot, dtype=float)
    n = len(c)
    assert len(o) == len(h) == len(lo) == len(ev) == len(lt) == n
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    broker = PaperBroker(db_path, starting_balance=100_000,
                         max_position_pct=0.25, stop_loss_pct=0.05)
    pending: tuple[str, float] | None = None
    action_counts = {"hold": 0, "buy": 0, "short": 0}
    judge_log: list[dict] = []
    curve = [100_000.0]
    tip = 0

    def _exec(action: str, price: float, ts: str, atr_d) -> None:
        if action == "buy":
            if broker.btc < 0:
                broker.cover(price, pct=1.0, reason="cover", ts=ts)
                broker.buy(price, reason="signal", ts=ts, atr_pct=atr_d)
            elif broker.btc == 0:
                broker.buy(price, reason="signal", ts=ts, atr_pct=atr_d)
        elif action == "short":
            if broker.btc > 0:
                broker.sell(price, pct=1.0, reason="signal", ts=ts)
                broker.short_open(price, reason="signal", ts=ts,
                                  atr_pct=atr_d)
            elif broker.btc == 0:
                broker.short_open(price, reason="signal", ts=ts,
                                  atr_pct=atr_d)

    for i in range(n):
        ts = str(stamps[i])
        # (1) pending action at the open
        if pending is not None:
            action, atr_d = pending
            pending = None
            _exec(action, float(o[i]), ts,
                  None if not np.isfinite(atr_d) else atr_d)
        # (2) protective chassis on this candle
        broker.check_stop(float(c[i]), ts=ts, low=float(lo[i]),
                          high=float(h[i]))
        # (3) mark at the close
        curve.append(broker.equity(float(c[i])))
        tip += int(broker.btc != 0)
        # (4) confirmation event -> next pending action
        decision = "hold"
        e = int(ev[i])
        if e != 0:
            side = 0 if broker.btc == 0 else (1 if broker.btc > 0
                                              else -1)
            action, context = _shift_action(side, e)
            if judged:
                p_lot = float(lt[i]) if np.isfinite(lt[i]) else None
                if action == "hold":
                    verdict = "noop"
                elif p_lot is not None and p_lot >= JUDGE_THRESHOLD:
                    verdict = "taken"
                else:
                    verdict = "skipped"
                judge_log.append({
                    "candle": ts, "direction": e,
                    "p_lot": p_lot, "threshold": JUDGE_THRESHOLD,
                    "verdict": verdict, "context": context})
                if verdict != "taken":
                    action = "hold"
            if action != "hold":
                pending = (action, float(atr[i]))
                decision = action
        action_counts[decision] += 1

    curve_arr = np.asarray(curve, dtype=float)
    pnl = float(curve_arr[-1] - 100_000.0)
    peak = np.maximum.accumulate(curve_arr)
    max_dd = float(np.max(peak - curve_arr)) if len(curve_arr) else 0.0
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT ts, side, price, qty, fee, reason FROM trades "
            "ORDER BY id").fetchall()
    fills = [{"ts": r[0], "side": r[1], "price": float(r[2]),
              "qty": float(r[3]), "fee": float(r[4]), "reason": r[5]}
             for r in rows]
    return {"pnl": pnl, "max_drawdown": max_dd,
            "score": pnl - 0.5 * max_dd,
            "trades": len(fills),
            "fees": float(sum(f["fee"] for f in fills)),
            "time_in_position": float(tip / n) if n else 0.0,
            "action_counts": action_counts,
            "judge_log": judge_log,
            "fills": fills,
            "final_equity": float(curve_arr[-1])}
