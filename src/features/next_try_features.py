"""Muse's Next Try features -- LAB ONLY (Kass's order, 2026-11-03).

The one lever every prior clean-era study left untested: NEW
INFORMATION. All 14 information series below come from outside
BTC's own spot candles -- Binance USDT-perpetual derivatives
positioning (funding, open interest, long/short ratios, taker flow,
basis; official Binance Vision archive) and ETH cross-asset prices
(Coinbase via the project's own market-data adapter). Four sequence
columns (the Master Score's own trajectory + the prior candle's
return) and four calendar columns (hour / day-of-week cyclicals)
complete the 22 new columns; the frame is Pick #2's 98 plus these
22 = 120 columns.

Causality -- ONE rule for every external series: a feature at row
t (hour-start index) may use only external observations
timestamped < t + 1h, i.e. through the close of candle t -- the
same footing the spot features already use (they read candle t in
full). Funding level uses the last settlement <= t; the funding
24h mean uses settlements in (t-24h, t]. Every raw series is then
percentile-scored through the project's own trailing-4,320-bar
machinery (the matrix-v3 builder's rolling_scores, the identical
function that scores every champion column), at the same scored
positions discipline as Pick #2's builder. Calendar columns enter
directly as (sin/cos + 1) * 50 on the same 0..100 scale.

Pre-registration: refs/muse-next-try-preregistration-2026-11-04.md
(frozen before any grading). Production never imports this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import pick2_features, scored_features

WINDOW = scored_features.WINDOW  # 4320 trailing bars
FRAME_TAG = "next_try_120"

# Raw (unscored) information series, in frame order.
INFO_RAW = [
    "funding_level_raw", "funding_mean24_raw",
    "oi_level_raw", "oi_chg1_raw", "oi_chg24_raw",
    "ls_all_raw", "ls_top_acct_raw", "ls_top_pos_raw",
    "taker_ls_raw", "basis_raw", "taker_buy_share_raw",
    "perp_vol_ratio_raw", "eth_ret_raw", "ethbtc_chg24_raw",
]
INFO_COLUMNS = [
    "Funding Rate (last settled)", "Funding Rate 24h Mean",
    "Open Interest Level", "Open Interest Change 1h %",
    "Open Interest Change 24h %",
    "Long/Short Ratio (all accounts)",
    "Top Trader Account L/S Ratio", "Top Trader Position L/S Ratio",
    "Taker L/S Volume Ratio (1h)", "Perp Basis %",
    "Perp Taker Buy Share", "Perp/Spot Volume Ratio",
    "ETH Return 1h %", "ETH/BTC Ratio Change 24h %",
]
PRIOR_RET_RAW = "prior_ret_raw"
PRIOR_RET_COLUMN = "Prior Candle Return %"
SEQUENCE_COLUMNS = ["Master Score Lag 6h", "Master Score Lag 24h",
                    "Master Score Change 24h", PRIOR_RET_COLUMN]
CALENDAR_COLUMNS = ["Hour of Day Sin", "Hour of Day Cos",
                    "Day of Week Sin", "Day of Week Cos"]
NEW_COLUMNS = INFO_COLUMNS + SEQUENCE_COLUMNS + CALENDAR_COLUMNS

_NS = 1_000_000_000


def _ns(times) -> np.ndarray:
    # pandas 3 date_range defaults to microsecond resolution; the
    # alignment math below is written in nanoseconds, so normalize
    # every timestamp to ns before taking asi8 (a unit mismatch here
    # silently shifted every window -- pinned by the hand-computed
    # tests in tests/test_muse_next_try.py).
    return pd.DatetimeIndex(times).astype("datetime64[ns]").asi8


# ------------------------------------------------------------ funding
def funding_series(index, times, rates):
    """(level, mean24) raw funding arrays aligned to `index`.

    level[t]   = last rate with calc_time <= t (NaN before the
                 first settlement -- a future settlement is never
                 read);
    mean24[t]  = mean of rates with calc_time in (t-24h, t].
    """
    ft = _ns(times)
    r = np.asarray(rates, dtype=float)
    t = _ns(index)
    hi = np.searchsorted(ft, t, side="right")
    level = np.where(hi > 0, r[np.maximum(hi - 1, 0)], np.nan)
    lo = np.searchsorted(ft, t - 24 * 3600 * _NS, side="right")
    mean24 = np.full(len(t), np.nan)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    cnt = hi - lo
    ok = cnt > 0
    mean24[ok] = (csum[hi[ok]] - csum[lo[ok]]) / cnt[ok]
    return level, mean24


# ------------------------------------------------------------ metrics
def metrics_series(index, metrics: pd.DataFrame) -> dict:
    """Raw derivatives-metrics arrays aligned to `index` from the
    5-minute Binance metrics frame (create_time + the five ratio /
    OI columns). One rule: row t reads observations < t + 1h only;
    the taker ratio is the mean inside candle t, everything else is
    the last observation before t + 1h (changes compare against the
    last observation before t + 1h - 1h and t + 1h - 24h)."""
    mt = _ns(metrics["create_time"])
    t = _ns(index)
    ends = t + 3600 * _NS
    i_end = np.searchsorted(mt, ends, side="left") - 1
    i_1h = np.searchsorted(mt, ends - 3600 * _NS, side="left") - 1
    i_24h = np.searchsorted(mt, ends - 24 * 3600 * _NS, side="left") - 1

    def at(col, idxs):
        vals = metrics[col].to_numpy(dtype=float)
        out = np.full(len(t), np.nan)
        ok = idxs >= 0
        out[ok] = vals[idxs[ok]]
        return out

    # As-of readings carry the last known value across publication
    # gaps (a field missing from one 5-minute snapshot, or the
    # archive's ~30h live-edge lag): forward-fill is causal -- it
    # only ever carries PAST values forward.
    oi = _ffill(at("sum_open_interest", i_end))
    oi_1h = _ffill(at("sum_open_interest", i_1h))
    oi_24h = _ffill(at("sum_open_interest", i_24h))
    with np.errstate(invalid="ignore", divide="ignore"):
        chg1 = np.where(oi_1h > 0, oi / oi_1h - 1.0, np.nan)
        chg24 = np.where(oi_24h > 0, oi / oi_24h - 1.0, np.nan)
    taker = metrics["sum_taker_long_short_vol_ratio"].to_numpy(float)
    lo = np.searchsorted(mt, t, side="left")
    hi = np.searchsorted(mt, ends, side="left")
    csum = np.concatenate([[0.0], np.nancumsum(
        np.where(np.isfinite(taker), taker, 0.0))])
    cnt = np.array([np.isfinite(taker[a:b]).sum()
                    for a, b in zip(lo, hi)])
    taker_mean = np.where(cnt > 0,
                          (csum[hi] - csum[lo]) / np.maximum(cnt, 1),
                          np.nan)
    # An hour with no snapshot at all (the live-edge lag) reads the
    # last known taker ratio -- the same as-of rule as every other
    # metrics feature.
    taker_mean = np.where(np.isfinite(taker_mean), taker_mean,
                          at("sum_taker_long_short_vol_ratio", i_end))
    return {
        "oi_level_raw": oi,
        "oi_chg1_raw": _ffill(chg1),
        "oi_chg24_raw": _ffill(chg24),
        "ls_all_raw": _ffill(at("count_long_short_ratio", i_end)),
        "ls_top_acct_raw": _ffill(
            at("count_toptrader_long_short_ratio", i_end)),
        "ls_top_pos_raw": _ffill(
            at("sum_toptrader_long_short_ratio", i_end)),
        "taker_ls_raw": _ffill(taker_mean),
    }


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs with the last finite value. Positions
    before the first finite value stay NaN (their index points at
    a[0], itself NaN). Causal by construction."""
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


# ------------------------------------------------------------ perp / eth
def perp_series(index, perp: pd.DataFrame, spot_close: pd.Series,
                spot_volume: pd.Series) -> dict:
    """Perp-market raw arrays from the hourly perp klines aligned
    to the spot index (same hour-start semantics; candle t's perp
    close is known at candle t's close, like the spot candle)."""
    p = perp.reindex(index)
    sc = spot_close.reindex(index).to_numpy(dtype=float)
    sv = spot_volume.reindex(index).to_numpy(dtype=float)
    pc = p["close"].to_numpy(dtype=float)
    pv = p["volume"].to_numpy(dtype=float)
    tbv = p["taker_buy_volume"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        basis = np.where(sc > 0, pc / sc - 1.0, np.nan)
        share = np.where(pv > 0, tbv / pv, np.nan)
        vol_ratio = np.where(sv > 0, pv / sv, np.nan)
    return {"basis_raw": basis, "taker_buy_share_raw": share,
            "perp_vol_ratio_raw": vol_ratio}


def eth_series(index, eth_close: pd.Series,
               spot_close: pd.Series) -> dict:
    """ETH cross-asset raw arrays: ETH's own 1h return and the
    ETH/BTC ratio's 24h change. ETH closes are aligned to the spot
    index and forward-filled (a missing ETH hour carries the last
    known close -- causal, never a future one)."""
    eth = eth_close.reindex(index).ffill()
    e = eth.to_numpy(dtype=float)
    sc = spot_close.reindex(index).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(np.isfinite(e), e, np.nan)
        ret = np.concatenate([[np.nan], ret[1:] / ret[:-1] - 1.0])
        ratio = np.where(sc > 0, e / sc, np.nan)
        chg24 = np.concatenate([
            np.full(24, np.nan), ratio[24:] / ratio[:-24] - 1.0])
    return {"eth_ret_raw": ret, "ethbtc_chg24_raw": chg24}


# ------------------------------------------------------------ calendar
def calendar_series(index) -> pd.DataFrame:
    """Hour-of-day / day-of-week cyclicals on the 0..100 scale:
    (sin + 1) * 50 and (cos + 1) * 50."""
    idx = pd.DatetimeIndex(index)
    h = idx.hour.to_numpy(dtype=float)
    d = idx.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame({
        "hour_sin": (np.sin(2 * np.pi * h / 24.0) + 1.0) * 50.0,
        "hour_cos": (np.cos(2 * np.pi * h / 24.0) + 1.0) * 50.0,
        "dow_sin": (np.sin(2 * np.pi * d / 7.0) + 1.0) * 50.0,
        "dow_cos": (np.cos(2 * np.pi * d / 7.0) + 1.0) * 50.0,
    }, index=idx)


# ------------------------------------------------------------ combined raw
def raw_external_series(index, funding, metrics, perp, spot_close,
                        spot_volume, eth_close) -> pd.DataFrame:
    """All 14 raw information series aligned to `index`, as a
    DataFrame with INFO_RAW columns (unscored). `funding` is a
    (times, rates) tuple."""
    ftimes, frates = funding
    level, mean24 = funding_series(index, ftimes, frates)
    out = {"funding_level_raw": level, "funding_mean24_raw": mean24}
    out.update(metrics_series(index, metrics))
    out.update(perp_series(index, perp, spot_close, spot_volume))
    out.update(eth_series(index, eth_close, spot_close))
    return pd.DataFrame(out, index=index)[INFO_RAW]


# ------------------------------------------------------------ the frame
def build_next_try_frame(df: pd.DataFrame, funding, metrics,
                         perp: pd.DataFrame, eth_close: pd.Series):
    """(frame120, cols120, evidence): Pick #2's 98-column frame
    plus the 22 new columns, indexed exactly like `df`.

    Scored positions follow Pick #2's discipline: p0 is the latest
    warmup demand across the base frame, every scored raw series
    (last non-finite row + WINDOW), and the 24-hour sequence lags.
    New columns are NaN outside their scored positions."""
    b3 = scored_features._b3
    frame98, cols98, ev98 = pick2_features.build_pick2_frame(df)
    assert cols98 == list(scored_features.scored_columns()) + list(
        pick2_features.NEW_COLUMNS)
    raw = raw_external_series(df.index, funding, metrics, perp,
                              df["close"], df["volume"], eth_close)
    logret = np.log(df["close"] / df["close"].shift(1))
    raw[PRIOR_RET_RAW] = logret.shift(1).to_numpy(dtype=float)
    scored_raw_names = INFO_RAW + [PRIOR_RET_RAW]
    scored_col_names = INFO_COLUMNS + [PRIOR_RET_COLUMN]

    p0 = int(ev98["pick2_p0"]) + 24  # sequence lags need t-24 finite
    qa_map: dict[str, np.ndarray] = {}
    for name in scored_raw_names:
        qa = raw[name].to_numpy(dtype=float)
        isn = ~np.isfinite(qa)
        last_bad = int(np.max(np.nonzero(isn))) if isn.any() else -1
        p0 = max(p0, last_bad + WINDOW)
        qa_map[name] = qa
    positions = np.arange(p0, len(df))
    assert len(positions) > 0, "no fully-warmed position for next-try frame"

    frame = frame98.copy()
    for raw_name, col_name in zip(scored_raw_names, scored_col_names):
        s = b3.rolling_scores(qa_map[raw_name], positions)
        assert s.min() >= 0.0 and s.max() <= 100.0, col_name
        full = np.full(len(df), np.nan)
        full[positions] = s
        frame[col_name] = full

    master = frame98["score_master"].to_numpy(dtype=float)
    for col_name, vals in (
        ("Master Score Lag 6h", master[np.maximum(positions - 6, 0)]),
        ("Master Score Lag 24h", master[positions - 24]),
        ("Master Score Change 24h", np.clip(
            (master[positions] - master[positions - 24] + 100.0) / 2.0,
            0.0, 100.0)),
    ):
        full = np.full(len(df), np.nan)
        full[positions] = vals
        frame[col_name] = full

    cal = calendar_series(df.index)
    for col_name, key in zip(CALENDAR_COLUMNS,
                             ("hour_sin", "hour_cos", "dow_sin", "dow_cos")):
        full = np.full(len(df), np.nan)
        full[positions] = cal[key].to_numpy(dtype=float)[positions]
        frame[col_name] = full

    cols = list(cols98) + list(NEW_COLUMNS)
    coverage = {}
    pos_frame = frame.iloc[p0:]
    for c in NEW_COLUMNS:
        coverage[c] = float(np.isfinite(
            pos_frame[c].to_numpy(dtype=float)).mean())
    evidence = {
        "pick2_p0": int(ev98["pick2_p0"]),
        "next_try_p0": int(p0),
        "scored_positions": int(len(positions)),
        "new_column_coverage_from_p0": coverage,
        "raw_warmup_last_bad": {
            name: int(np.max(np.nonzero(
                ~np.isfinite(qa_map[name]))))
            if (~np.isfinite(qa_map[name])).any() else -1
            for name in scored_raw_names},
    }
    return frame[cols], cols, evidence
