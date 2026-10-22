"""A-Z indicator library for the 2026-10-22 screening program.

One function per indicator family member, uniform signature
``builder(df) -> pd.Series`` over an OHLCV frame (columns
open/high/low/close/volume, hourly DatetimeIndex), returning the RAW
indicator series aligned to ``df.index``. Every builder is causal:
the value at row t is a function of rows <= t only (trailing rolling /
EWM windows, recursive filters seeded from the first rows, or
completed-bucket constructions built from values, never from a labeled
Series carried across a shifted index -- the v5.1 look-ahead trap).

Scaling to the project's [-1, 1] contract happens in a second stage so
the statistics can be fitted on the TRAIN split only:

- ``natural``: the indicator is mathematically bounded on [lo, hi];
  mapped affinely to [-1, 1] and clipped. No fitted statistics.
- ``unit``: already in [-1, 1] by construction; clipped only.
- ``robust``: (x - train_median) / train_scale with
  train_scale = (p99.5 - p0.5) / 2 over the train split, clipped to
  [-1, 1]. Fitted on train rows only -- mutating holdout rows cannot
  change any train value (pinned by tests/test_az_indicators.py).

Warmup: ratio/rolling indicators carry NaN until their window fills
(explicit, never backfilled). Recursive filters (Laguerre, McGinley,
Klinger, HA-style smoothers) are seeded from the first rows per their
canonical definitions, exactly like production's Heikin-Ashi.

Route (pre-flight 2026-10-22): TA-Lib is not installed; pandas_ta
0.4.71b0 installed into the venv (it pinned numpy 2.5.4 -> 2.2.6; the
full suite still passes 31/31 and torch/SB3 import cleanly). Most
builders wrap pandas_ta with explicit canonical parameters; the rest
are hand-implemented where pandas_ta lacks the indicator (Keltner is
present as `kc`; mass index, Chaikin volatility, Ulcer, Parkinson,
Garman-Klass, DPO, Ichimoku, Alligator, pivots, DSS, Laguerre,
Klinger, McGinley, PFE, Beyers RVI) or where its default is a
non-causal variant (DPO's centered form is never used).

Deliberate positive controls: az_rsi_14 and az_atr_pct duplicate
production features (rsi_c / atr_rel) up to scaling; the screening
program's redundancy detector must flag them, which validates the
detector on known ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------


def _col(res: pd.DataFrame, prefix: str) -> pd.Series:
    """First column of a pandas_ta result whose name starts with prefix."""
    for c in res.columns:
        if str(c).startswith(prefix):
            return res[c]
    raise KeyError(f"no column starting with {prefix!r} in {list(res.columns)}")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _dist(ma: pd.Series, close: pd.Series) -> pd.Series:
    """Canonical MA feature form: signed distance of price from the MA."""
    return (close - ma) / close


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


# ------------------------------------------------------------------
# pandas_ta-backed builders (explicit canonical parameters)
# ------------------------------------------------------------------


def _rsi14(df):
    import pandas_ta as ta
    return ta.rsi(df["close"], length=14)


def _stoch_k(df):
    import pandas_ta as ta
    return _col(ta.stoch(df["high"], df["low"], df["close"],
                         k=14, d=3, smooth_k=3), "STOCHk_")


def _stoch_d(df):
    import pandas_ta as ta
    return _col(ta.stoch(df["high"], df["low"], df["close"],
                         k=14, d=3, smooth_k=3), "STOCHd_")


def _stochrsi_k(df):
    import pandas_ta as ta
    return _col(ta.stochrsi(df["close"], length=14, rsi_length=14,
                            k=3, d=3), "STOCHRSIk_")


def _cci20(df):
    import pandas_ta as ta
    return ta.cci(df["high"], df["low"], df["close"], length=20)


def _willr14(df):
    import pandas_ta as ta
    return ta.willr(df["high"], df["low"], df["close"], length=14)


def _roc(n):
    def b(df):
        import pandas_ta as ta
        return ta.roc(df["close"], length=n)
    return b


def _ao(df):
    import pandas_ta as ta
    return ta.ao(df["high"], df["low"], fast=5, slow=34)


def _ac(df):
    ao = _ao(df)
    return ao - _sma(ao, 5)


def _cmo14(df):
    import pandas_ta as ta
    return ta.cmo(df["close"], length=14)


def _tsi(df):
    """Blau TSI: double-smoothed momentum / double-smoothed |momentum|."""
    d = df["close"].diff()
    num = _ema(_ema(d, 25), 13)
    den = _ema(_ema(d.abs(), 25), 13)
    return 100.0 * _safe_div(num, den)


def _uo(df):
    import pandas_ta as ta
    return ta.uo(df["high"], df["low"], df["close"],
                 fast=7, medium=14, slow=28)


def _trix15(df):
    import pandas_ta as ta
    return _col(ta.trix(df["close"], length=15), "TRIX_")


def _fisher9(df):
    import pandas_ta as ta
    return _col(ta.fisher(df["high"], df["low"], length=9), "FISHERT_")


def _kst(df):
    import pandas_ta as ta
    return _col(ta.kst(df["close"]), "KST_")


def _apo_rel(df):
    c = df["close"]
    return (_ema(c, 12) - _ema(c, 26)) / c


def _ppo(df):
    import pandas_ta as ta
    return ta.ppo(df["close"], fast=12, slow=26, signal=9).iloc[:, 0]


def _crsi(df):
    import pandas_ta as ta
    return ta.crsi(df["close"], length=3, roc_length=100)


def _smi(df):
    import pandas_ta as ta
    return ta.smi(df["close"]).iloc[:, 0]


def _stc(df):
    import pandas_ta as ta
    return ta.stc(df["close"]).iloc[:, 0]


def _qqe_line(df):
    import pandas_ta as ta
    return _col(ta.qqe(df["close"]), "QQEl")


def _coppock(df):
    import pandas_ta as ta
    return ta.coppock(df["close"])


def _aroon_up(df):
    import pandas_ta as ta
    return _col(ta.aroon(df["high"], df["low"], length=14), "AROONU_")


def _aroon_down(df):
    import pandas_ta as ta
    return _col(ta.aroon(df["high"], df["low"], length=14), "AROOND_")


def _adx(df):
    import pandas_ta as ta
    return _col(ta.adx(df["high"], df["low"], df["close"], length=14),
                "ADX_")


def _plus_di(df):
    import pandas_ta as ta
    return _col(ta.adx(df["high"], df["low"], df["close"], length=14),
                "DMP_")


def _minus_di(df):
    import pandas_ta as ta
    return _col(ta.adx(df["high"], df["low"], df["close"], length=14),
                "DMN_")


def _vortex_diff(df):
    import pandas_ta as ta
    r = ta.vortex(df["high"], df["low"], df["close"], length=14)
    return _col(r, "VTXP_") - _col(r, "VTXM_")


def _inertia(df):
    import pandas_ta as ta
    return ta.inertia(df["open"], df["high"], df["low"], df["close"],
                      df["volume"])


def _cg10(df):
    import pandas_ta as ta
    return ta.cg(df["close"], length=10)


def _rvgi(df):
    import pandas_ta as ta
    return ta.rvi(df["open"], df["high"], df["low"], df["close"],
                  length=14)


def _thermo(df):
    import pandas_ta as ta
    return ta.thermo(df["high"], df["low"], df["close"]).iloc[:, 0]


def _squeeze_mom(df):
    import pandas_ta as ta
    return ta.squeeze(df["high"], df["low"], df["close"],
                      df["volume"]).iloc[:, 0]


def _supertrend(df):
    import pandas_ta as ta
    return ta.supertrend(df["high"], df["low"], df["close"],
                         length=10, multiplier=3.0)


def _supertrend_dir(df):
    return _col(_supertrend(df), "SUPERTd_")


def _supertrend_dist(df):
    st = _col(_supertrend(df), "SUPERT_")
    return (df["close"] - st) / df["close"]


def _psar_dist(df):
    import pandas_ta as ta
    r = ta.psar(df["high"], df["low"], df["close"])
    sar = _col(r, "PSARl_").fillna(_col(r, "PSARs_"))
    return (df["close"] - sar) / df["close"]


def _bb(df):
    import pandas_ta as ta
    return ta.bbands(df["close"], length=20, std=2.0)


def _bb_pctb(df):
    return _col(_bb(df), "BBP_")


def _bb_bw(df):
    return _col(_bb(df), "BBB_")


def _donchian(df):
    import pandas_ta as ta
    return ta.donchian(df["high"], df["low"], length=20)


def _donchian_pos(df):
    r = _donchian(df)
    lo, up = _col(r, "DCL_"), _col(r, "DCU_")
    return _safe_div(df["close"] - lo, up - lo)


def _donchian_width(df):
    r = _donchian(df)
    return _safe_div(_col(r, "DCU_") - _col(r, "DCL_"), df["close"])


def _keltner(df):
    import pandas_ta as ta
    return ta.kc(df["high"], df["low"], df["close"],
                 length=20, scalar=2.0)


def _keltner_pos(df):
    r = _keltner(df)
    mid = _col(r, "KCBe_")
    half = (_col(r, "KCUe_") - _col(r, "KCLe_")) / 2.0
    return _safe_div(df["close"] - mid, half) / 2.0 + 0.5


def _keltner_width(df):
    r = _keltner(df)
    return _safe_div(_col(r, "KCUe_") - _col(r, "KCLe_"), df["close"])


def _atr_pct(df):
    import pandas_ta as ta
    return ta.atr(df["high"], df["low"], df["close"],
                  length=14) / df["close"]


def _cmf20(df):
    import pandas_ta as ta
    return ta.cmf(df["high"], df["low"], df["close"], df["volume"],
                  length=20)


def _mfi14(df):
    import pandas_ta as ta
    return ta.mfi(df["high"], df["low"], df["close"], df["volume"],
                  length=14)


def _adosc(df):
    import pandas_ta as ta
    return ta.adosc(df["high"], df["low"], df["close"], df["volume"],
                    fast=3, slow=10)


def _eom14(df):
    import pandas_ta as ta
    return ta.eom(df["high"], df["low"], df["close"], df["volume"],
                  length=14)


def _force_index(df):
    import pandas_ta as ta
    return ta.efi(df["close"], df["volume"], length=13)


def _bop(df):
    import pandas_ta as ta
    return ta.bop(df["open"], df["high"], df["low"], df["close"])


def _qstick10(df):
    import pandas_ta as ta
    return ta.qstick(df["open"], df["close"], length=10)


def _linreg(df):
    import pandas_ta as ta
    return ta.linreg(df["close"], length=20)


def _slope20(df):
    import pandas_ta as ta
    return ta.slope(df["close"], length=20) / df["close"]


def _linreg_dev(df):
    return _safe_div(df["close"] - _linreg(df), df["close"])


def _tsf_dev(df):
    import pandas_ta as ta
    tsf = ta.linreg(df["close"], length=20) + ta.slope(df["close"],
                                                      length=20)
    return _safe_div(df["close"] - tsf, df["close"])


def _ma_dist(kind, length):
    def b(df):
        import pandas_ta as ta
        fn = getattr(ta, kind)
        return _dist(fn(df["close"], length=length), df["close"])
    return b


def _ma_cross(df):
    import pandas_ta as ta
    return (ta.sma(df["close"], length=50)
            - ta.sma(df["close"], length=200)) / df["close"]


def _rsi_sma(df):
    return _sma(_rsi14(df), 14)


def _kdj_j(df):
    return 3.0 * _stoch_k(df) - 2.0 * _stoch_d(df)


def _obv_mom(df):
    import pandas_ta as ta
    obv = ta.obv(df["close"], df["volume"])
    return _safe_div(obv - obv.shift(10), df["volume"].rolling(10).sum())


def _pvt_mom(df):
    import pandas_ta as ta
    pvt = ta.pvt(df["close"], df["volume"])
    return _safe_div(pvt - pvt.shift(10), df["volume"].rolling(10).sum())


def _nvi_ret(df):
    import pandas_ta as ta
    nvi = ta.nvi(df["close"], df["volume"])
    return nvi.pct_change(20)


def _pvi_ret(df):
    import pandas_ta as ta
    pvi = _col(ta.pvi(df["close"], df["volume"]), "PVI")
    return pvi.pct_change(20)


def _vwma_dist(df):
    import pandas_ta as ta
    return _dist(ta.vwma(df["close"], df["volume"], length=20),
                 df["close"])


# ------------------------------------------------------------------
# hand-implemented builders (pandas_ta lacks them, or its default
# variant is non-causal)
# ------------------------------------------------------------------


def _dpo20(df):
    """Detrended Price Oscillator, causal form: the value known at
    row t is close[t-k] vs SMA20[t-k] with k = 20//2 + 1 (the classic
    construction evaluates the detrended deviation k bars back; it is
    never centered on future data)."""
    k = 20 // 2 + 1
    c = df["close"]
    return c.shift(k) - _sma(c, 20).shift(k)


def _disparity14(df):
    c = df["close"]
    return _safe_div(c - _sma(c, 14), _sma(c, 14))


def _mass_index(df):
    rng = df["high"] - df["low"]
    e1 = _ema(rng, 9)
    e2 = _ema(e1, 9)
    return _safe_div(e1, e2).rolling(25).sum()


def _rel_vol_index(df):
    """Beyers Relative Volatility Index: share of return dispersion
    coming from up moves, rolling-std formulation."""
    d = df["close"].diff()
    sd_up = d.clip(lower=0).rolling(14).std()
    sd_dn = (-d.clip(upper=0)).rolling(14).std()
    return 100.0 * _safe_div(sd_up, sd_up + sd_dn)


def _laguerre_rsi(df, gamma: float = 0.8):
    """Ehlers Laguerre RSI from the 4-tap Laguerre filter (loop:
    the filter is recursive by definition)."""
    c = df["close"].to_numpy(dtype=float)
    L = np.zeros((len(c), 4))
    for t in range(len(c)):
        prev = L[t - 1] if t > 0 else np.full(4, c[t])
        L[t, 0] = (1 - gamma) * c[t] + gamma * prev[0]
        L[t, 1] = -gamma * L[t, 0] + prev[0] + gamma * prev[1]
        L[t, 2] = -gamma * L[t, 1] + prev[1] + gamma * prev[2]
        L[t, 3] = -gamma * L[t, 2] + prev[2] + gamma * prev[3]
    d = np.diff(L, axis=0, prepend=L[:1])
    cu = np.where(d > 0, d, 0.0).sum(axis=1)
    cd = np.where(d < 0, -d, 0.0).sum(axis=1)
    tot = cu + cd
    rsi = np.where(tot > 0, 100.0 * cu / np.where(tot > 0, tot, 1.0),
                   50.0)
    return pd.Series(rsi, index=df.index)


def _dss10(df):
    """Double-Smoothed Stochastic (Bressert): two stages of
    stochastic + EMA(3) smoothing."""
    h, l, c = df["high"], df["low"], df["close"]
    hh, ll = h.rolling(10).max(), l.rolling(10).min()
    x = 100.0 * _safe_div(c - ll, hh - ll)
    s1 = _ema(x, 3)
    hh2, ll2 = s1.rolling(10).max(), s1.rolling(10).min()
    y = 100.0 * _safe_div(s1 - ll2, hh2 - ll2)
    return _ema(y, 3)


def _pfe10(df):
    """Polarized Fractal Efficiency (Hanulla): straight-line distance
    over 10 bars vs the path actually travelled, signed by direction."""
    c = df["close"]
    step = np.sqrt(c.diff() ** 2 + 1.0)
    path = step.rolling(10).sum()
    dist = np.sqrt((c - c.shift(10)) ** 2 + 100.0)
    return np.sign(c - c.shift(10)) * 100.0 * _safe_div(dist, path)


def _klinger(df):
    """Klinger Oscillator minus its signal line (hand implementation
    of Klinger's volume-force construction)."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    sv = (h + l + c).to_numpy(dtype=float)
    dm = (h - l).to_numpy(dtype=float)
    vol = v.to_numpy(dtype=float)
    trend = np.where(np.diff(sv, prepend=sv[0]) >= 0, 1.0, -1.0)
    cm = np.zeros(len(df))
    vf = np.zeros(len(df))
    for t in range(1, len(df)):
        cm[t] = cm[t - 1] + dm[t] if trend[t] == trend[t - 1] \
            else dm[t - 1] + dm[t]
        if cm[t] > 0:
            vf[t] = vol[t] * abs(2.0 * dm[t] / cm[t] - 1.0) * trend[t]
    vf_s = pd.Series(vf, index=df.index)
    kvo = _ema(vf_s, 34) - _ema(vf_s, 55)
    out = kvo - _ema(kvo, 13)
    out.iloc[:68] = np.nan  # explicit warmup until the 55-EMA is seeded
    return out


def _mcginley20(df):
    c = df["close"].to_numpy(dtype=float)
    mg = np.empty(len(c))
    mg[0] = c[0]
    for t in range(1, len(c)):
        ratio = c[t] / mg[t - 1] if mg[t - 1] > 0 else 1.0
        mg[t] = mg[t - 1] + (c[t] - mg[t - 1]) / (20.0 * ratio ** 4)
    return pd.Series((df["close"].to_numpy() - mg) / df["close"].to_numpy(),
                     index=df.index)


def _ichimoku(kind):
    def b(df):
        h, l, c = df["high"], df["low"], df["close"]
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        if kind == "tenkan":
            ref = tenkan
        elif kind == "kijun":
            ref = kijun
        elif kind == "span_a":
            ref = (tenkan + kijun) / 2.0
        else:  # span_b: 52-period midpoint, UNSHIFTED (causal form)
            ref = (h.rolling(52).max() + l.rolling(52).min()) / 2.0
        return (c - ref) / c
    return b


def _alligator(kind):
    n = {"jaw": 13, "teeth": 8, "lips": 5}[kind]

    def b(df):
        import pandas_ta as ta
        med = (df["high"] + df["low"]) / 2.0
        # canonical Alligator lines are PLOT-shifted forward; the values
        # themselves are trailing SMMAs -- the unshifted line at row t
        # is the causal form (same information, no future plot offset).
        return _dist(ta.smma(med, length=n), df["close"])
    return b


def _starc_pos(df):
    mid = _sma(df["close"], 20)
    import pandas_ta as ta
    atr = ta.atr(df["high"], df["low"], df["close"], length=10)
    return _safe_div(df["close"] - mid, 2.0 * atr)


def _chaikin_vol(df):
    e = _ema(df["high"] - df["low"], 10)
    return 100.0 * _safe_div(e - e.shift(10), e.shift(10))


def _hist_vol20(df):
    lr = np.log(df["close"] / df["close"].shift(1))
    return lr.rolling(20).std() * 100.0


def _ulcer14(df):
    c = df["close"]
    mx = c.rolling(14).max()
    dd = 100.0 * (c - mx) / mx
    return np.sqrt((dd ** 2).rolling(14).mean())


def _parkinson20(df):
    x = (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()
    return np.sqrt(x / (4.0 * np.log(2.0))) * 100.0


def _garman_klass20(df):
    gk = (0.5 * np.log(df["high"] / df["low"]) ** 2
          - (2.0 * np.log(2.0) - 1.0) * np.log(df["close"] / df["open"]) ** 2)
    return np.sqrt(gk.rolling(20).mean().clip(lower=0)) * 100.0


def _bull_power(df):
    return (df["high"] - _ema(df["close"], 13)) / df["close"]


def _bear_power(df):
    return (df["low"] - _ema(df["close"], 13)) / df["close"]


def _ibs(df):
    return _safe_div(df["close"] - df["low"], df["high"] - df["low"])


def _bwmfi(df):
    return _safe_div(df["high"] - df["low"], df["volume"])


def _vol_osc(df):
    v = df["volume"]
    return _safe_div(_ema(v, 5) - _ema(v, 20), _ema(v, 20))


def _vroc14(df):
    return df["volume"].pct_change(14)


def _typical_dev(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    return (df["close"] - _sma(tp, 20)) / df["close"]


def _medprice_dev(df):
    mp = (df["high"] + df["low"]) / 2.0
    return (df["close"] - _sma(mp, 14)) / df["close"]


def _wclose_dev(df):
    wc = (df["high"] + df["low"] + 2.0 * df["close"]) / 4.0
    return (df["close"] - _sma(wc, 14)) / df["close"]


def _zscore(n):
    def b(df):
        c = df["close"]
        return _safe_div(c - _sma(c, n), c.rolling(n).std())
    return b


def _ret_skew20(df):
    return df["close"].pct_change().rolling(20).skew()


def _ret_kurt20(df):
    return df["close"].pct_change().rolling(20).kurt()


def _acf(lag):
    def b(df):
        r = df["close"].pct_change()
        return r.rolling(50).corr(r.shift(lag))
    return b


def _pv_corr20(df):
    return df["close"].pct_change().rolling(20).corr(
        df["volume"].pct_change())


def _er10(df):
    c = df["close"]
    change = (c - c.shift(10)).abs()
    vol = c.diff().abs().rolling(10).sum()
    return _safe_div(change, vol)


def _vhf28(df):
    rng = df["high"].rolling(28).max() - df["low"].rolling(28).min()
    path = df["close"].diff().abs().rolling(28).sum()
    return _safe_div(rng, path)


# -- completed-day constructions (pivots / ADR) --------------------
# Built from VALUES on day stamps, stamped onto the FOLLOWING day --
# the same values-not-labels discipline as the v5.1 fix: a row on
# day D only ever sees statistics of days that completed before D.


def _daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()


def _stamp_next_day(df: pd.DataFrame, daily_values: np.ndarray,
                    day_index: pd.DatetimeIndex) -> pd.Series:
    """daily_values[i] was computed from days < day_index[i]; stamp it
    on every hourly row of day_index[i] via merge_asof backward."""
    feats = pd.DataFrame({"v": np.asarray(daily_values, dtype=float)},
                         index=day_index)
    base = pd.DataFrame({"ts": df.index}, index=df.index)
    merged = pd.merge_asof(base.sort_values("ts"), feats.sort_index(),
                           left_on="ts", right_index=True,
                           direction="backward")
    merged.index = base.index
    return merged["v"]


def _pivot(kind):
    def b(df):
        d = _daily_frame(df)
        P = (d["high"] + d["low"] + d["close"]) / 3.0
        if kind == "p":
            raw = P
        elif kind == "r1":
            raw = 2.0 * P - d["low"]
        else:
            raw = 2.0 * P - d["high"]
        vals = raw.to_numpy()
        nxt = np.concatenate([[np.nan], vals[:-1]])  # prior day's level
        stamped = _stamp_next_day(df, nxt, d.index)
        return (df["close"] - stamped) / df["close"]
    return b


def _adr_rel(df):
    d = _daily_frame(df)
    day_rng = ((d["high"] - d["low"]) / d["close"]).to_numpy()
    # mean of the prior 14 completed days, stamped on the next day
    out = np.full(len(day_rng), np.nan)
    for i in range(14, len(day_rng)):
        out[i] = day_rng[i - 14:i].mean()
    stamped = _stamp_next_day(df, out, d.index)
    return stamped


# -- candlestick pattern flags (canonical OHLC definitions) ---------


def _body_rng(df):
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return body, rng


def _pat_doji(df):
    body, rng = _body_rng(df)
    return ((body <= 0.1 * rng) & rng.notna()).astype(float)


def _pat_hammer(df):
    body, rng = _body_rng(df)
    lower_w = np.minimum(df["open"], df["close"]) - df["low"]
    upper_w = df["high"] - np.maximum(df["open"], df["close"])
    return ((lower_w >= 2.0 * body) & (upper_w <= body)
            & (body > 0)).astype(float)


def _pat_shooting_star(df):
    body, rng = _body_rng(df)
    lower_w = np.minimum(df["open"], df["close"]) - df["low"]
    upper_w = df["high"] - np.maximum(df["open"], df["close"])
    return -((upper_w >= 2.0 * body) & (lower_w <= body)
             & (body > 0)).astype(float)


def _pat_engulfing(df):
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    bull = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    return bull.astype(float) - bear.astype(float)


def _pat_marubozu(df):
    body, rng = _body_rng(df)
    return (np.sign(df["close"] - df["open"])
            * ((body >= 0.9 * rng) & rng.notna()).astype(float))


def _pat_three_soldiers(df):
    o, c = df["open"], df["close"]
    body, rng = _body_rng(df)
    strong = body >= 0.5 * rng
    up = (c > o) & strong
    dn = (c < o) & strong
    rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    soldiers = (up & up.shift(1).fillna(False) & up.shift(2).fillna(False)
                & rising)
    crows = (dn & dn.shift(1).fillna(False) & dn.shift(2).fillna(False)
             & falling)
    return soldiers.astype(float) - crows.astype(float)


# -- calendar features (encyclopedia section 1) ---------------------


def _hour_sin(df):
    return np.sin(2 * np.pi * df.index.hour / 24.0)


def _hour_cos(df):
    return np.cos(2 * np.pi * df.index.hour / 24.0)


def _dow_sin(df):
    return np.sin(2 * np.pi * df.index.dayofweek / 7.0)


def _dow_cos(df):
    return np.cos(2 * np.pi * df.index.dayofweek / 7.0)


# ------------------------------------------------------------------
# registry: name -> builder + scaling spec + provenance
# scale kinds: ("natural", lo, hi) | ("unit",) | ("robust",)
# ------------------------------------------------------------------

NAT = lambda lo, hi: ("natural", float(lo), float(hi))  # noqa: E731
UNIT = ("unit",)
ROBUST = ("robust",)

REGISTRY: dict[str, dict] = {}


def _reg(name, build, scale, family, source):
    REGISTRY[name] = {"build": build, "scale": scale,
                      "family": family, "source": source}


# momentum / oscillators
_reg("az_rsi_14", _rsi14, NAT(0, 100), "momentum", "RSI(14), pandas_ta [positive control: production rsi_c]")
_reg("az_rsi_sma_14", _rsi_sma, NAT(0, 100), "momentum", "SMA(14) of RSI(14)")
_reg("az_stoch_k", _stoch_k, NAT(0, 100), "momentum", "Stochastic %K(14,3,3), pandas_ta")
_reg("az_stoch_d", _stoch_d, NAT(0, 100), "momentum", "Stochastic %D(14,3,3), pandas_ta")
_reg("az_stochrsi_k", _stochrsi_k, NAT(0, 100), "momentum", "StochRSI %K(14,14,3,3), pandas_ta")
_reg("az_kdj_j", _kdj_j, ROBUST, "momentum", "KDJ J line = 3K - 2D of stochastic(14,3,3)")
_reg("az_cci_20", _cci20, ROBUST, "momentum", "CCI(20), pandas_ta")
_reg("az_willr_14", _willr14, NAT(-100, 0), "momentum", "Williams %R(14), pandas_ta")
_reg("az_roc_6", _roc(6), ROBUST, "momentum", "ROC(6) %, pandas_ta")
_reg("az_roc_12", _roc(12), ROBUST, "momentum", "ROC(12) %, pandas_ta")
_reg("az_roc_24", _roc(24), ROBUST, "momentum", "ROC(24) %, pandas_ta")
_reg("az_ao", _ao, ROBUST, "momentum", "Awesome Oscillator(5,34), pandas_ta")
_reg("az_ac_osc", _ac, ROBUST, "momentum", "Accelerator Oscillator = AO - SMA5(AO) (Bill Williams)")
_reg("az_cmo_14", _cmo14, NAT(-100, 100), "momentum", "Chande Momentum Oscillator(14), pandas_ta")
_reg("az_tsi", _tsi, NAT(-100, 100), "momentum", "True Strength Index (25,13) double-smoothed, hand")
_reg("az_uo", _uo, NAT(0, 100), "momentum", "Ultimate Oscillator(7,14,28), pandas_ta")
_reg("az_trix_15", _trix15, ROBUST, "momentum", "TRIX(15) %, pandas_ta")
_reg("az_fisher_9", _fisher9, ROBUST, "momentum", "Fisher Transform(9), pandas_ta")
_reg("az_kst", _kst, ROBUST, "momentum", "Know Sure Thing line, pandas_ta")
_reg("az_apo_rel", _apo_rel, ROBUST, "momentum", "APO = (EMA12-EMA26)/close (MACD line, price-relative)")
_reg("az_ppo", _ppo, ROBUST, "momentum", "Percentage Price Oscillator(12,26,9), pandas_ta")
_reg("az_crsi", _crsi, NAT(0, 100), "momentum", "Connors RSI(3,2,100), pandas_ta")
_reg("az_laguerre_rsi", _laguerre_rsi, NAT(0, 100), "momentum", "Laguerre RSI (gamma 0.8), Ehlers, hand")
_reg("az_smi", _smi, NAT(-100, 100), "momentum", "Stochastic Momentum Index, pandas_ta")
_reg("az_dss_10", _dss10, NAT(0, 100), "momentum", "Double-Smoothed Stochastic (Bressert), hand")
_reg("az_dpo_20", _dpo20, ROBUST, "momentum", "Detrended Price Oscillator(20), causal form, hand")
_reg("az_disparity_14", _disparity14, ROBUST, "momentum", "Disparity Index(14) = (close-SMA14)/SMA14")
_reg("az_mass_index", _mass_index, ROBUST, "volatility", "Mass Index(9,9,25), hand")
# az_rvgi DROPPED at build time: this fork's rvi(close, high, low)
# signature does not match the canonical Relative Vigor Index (which
# requires open); excluded rather than screened under a wrong label.
# The Beyers Relative Volatility Index is covered by az_rel_vol_index.
_reg("az_rel_vol_index", _rel_vol_index, NAT(0, 100), "momentum", "Relative Volatility Index (Beyers), rolling-std form, hand")
_reg("az_stc", _stc, NAT(0, 100), "momentum", "Schaff Trend Cycle, pandas_ta")
# az_qqe_line DROPPED at build time: this fork's QQE long line is NaN
# on ~50% of rows with interior gaps up to 34 bars -- an implementation
# defect of pandas_ta 0.4.71b0, not a property of QQE. Excluded from
# the screen rather than graded on a defective series (see report).
_reg("az_coppock", _coppock, ROBUST, "momentum", "Coppock Curve(11,14,10), pandas_ta")
_reg("az_inertia", _inertia, NAT(0, 100), "momentum", "Inertia(20, RVI-based), pandas_ta")
_reg("az_cg_10", _cg10, ROBUST, "momentum", "Center of Gravity(10), Ehlers, pandas_ta")
_reg("az_pfe_10", _pfe10, NAT(-100, 100), "momentum", "Polarized Fractal Efficiency(10), hand")

# trend strength / direction
_reg("az_aroon_up", _aroon_up, NAT(0, 100), "trend", "Aroon Up(14), pandas_ta")
_reg("az_aroon_down", _aroon_down, NAT(0, 100), "trend", "Aroon Down(14), pandas_ta")
_reg("az_adx_14", _adx, NAT(0, 100), "trend", "ADX(14), pandas_ta")
_reg("az_plus_di", _plus_di, NAT(0, 100), "trend", "+DI(14), pandas_ta")
_reg("az_minus_di", _minus_di, NAT(0, 100), "trend", "-DI(14), pandas_ta")
_reg("az_vortex_diff", _vortex_diff, ROBUST, "trend", "Vortex VI+ - VI- (14), pandas_ta")
_reg("az_supertrend_dir", _supertrend_dir, UNIT, "trend", "SuperTrend(10,3) direction, pandas_ta")
_reg("az_supertrend_dist", _supertrend_dist, ROBUST, "trend", "(close - SuperTrend line)/close, pandas_ta")
_reg("az_psar_dist", _psar_dist, ROBUST, "trend", "Signed (close - Parabolic SAR)/close, pandas_ta")
_reg("az_er_10", _er10, NAT(0, 1), "trend", "Kaufman Efficiency Ratio(10), hand")
_reg("az_vhf_28", _vhf28, ROBUST, "trend", "Vertical Horizontal Filter(28), hand")

# moving-average distances (canonical MA family, distance form)
for _kind, _ln, _nm in [
        ("sma", 20, "az_sma_20_dist"), ("sma", 200, "az_sma_200_dist"),
        ("ema", 20, "az_ema_20_dist"), ("ema", 50, "az_ema_50_dist"),
        ("ema", 200, "az_ema_200_dist"), ("wma", 20, "az_wma_20_dist"),
        ("hma", 20, "az_hma_20_dist"), ("dema", 20, "az_dema_20_dist"),
        ("tema", 20, "az_tema_20_dist"), ("kama", 10, "az_kama_10_dist"),
        ("alma", 20, "az_alma_20_dist"), ("vidya", 14, "az_vidya_14_dist"),
        ("zlma", 20, "az_zlema_20_dist"), ("smma", 20, "az_smma_20_dist"),
        ("trima", 20, "az_trima_20_dist"), ("t3", 10, "az_t3_10_dist")]:
    _reg(_nm, _ma_dist(_kind, _ln), ROBUST, "ma_distance",
         f"(close - { _kind.upper() }({_ln}))/close, pandas_ta")
_reg("az_mcg_20_dist", _mcginley20, ROBUST, "ma_distance",
     "McGinley Dynamic(20) distance, hand")
_reg("az_ma_cross_spread", _ma_cross, ROBUST, "ma_distance",
     "(SMA50 - SMA200)/close, pandas_ta")

# channels / structure
_reg("az_ichimoku_tenkan", _ichimoku("tenkan"), ROBUST, "structure",
     "(close - Tenkan-sen 9)/close, hand")
_reg("az_ichimoku_kijun", _ichimoku("kijun"), ROBUST, "structure",
     "(close - Kijun-sen 26)/close, hand")
_reg("az_ichimoku_span_a", _ichimoku("span_a"), ROBUST, "structure",
     "(close - Senkou Span A, unshifted)/close, hand")
_reg("az_ichimoku_span_b", _ichimoku("span_b"), ROBUST, "structure",
     "(close - Senkou Span B 52, unshifted)/close, hand")
_reg("az_alligator_jaw", _alligator("jaw"), ROBUST, "structure",
     "(close - SMMA13 of median price)/close, unshifted, pandas_ta")
_reg("az_alligator_teeth", _alligator("teeth"), ROBUST, "structure",
     "(close - SMMA8 of median price)/close, unshifted, pandas_ta")
_reg("az_alligator_lips", _alligator("lips"), ROBUST, "structure",
     "(close - SMMA5 of median price)/close, unshifted, pandas_ta")
_reg("az_donchian_pos", _donchian_pos, NAT(0, 1), "structure",
     "Position of close in Donchian channel(20), pandas_ta")
_reg("az_donchian_width", _donchian_width, ROBUST, "structure",
     "Donchian channel width / close, pandas_ta")
_reg("az_keltner_pos", _keltner_pos, NAT(0, 1), "structure",
     "Position of close in Keltner channel(20,2), pandas_ta kc")
_reg("az_keltner_width", _keltner_width, ROBUST, "structure",
     "Keltner channel width / close, pandas_ta kc")
_reg("az_starc_pos", _starc_pos, ROBUST, "structure",
     "(close - SMA20) / (2*ATR10), STARC position, hand")
_reg("az_bb_pctb", _bb_pctb, NAT(0, 1), "structure",
     "Bollinger %B(20,2), pandas_ta (clipped outside the bands)")
_reg("az_bb_bw", _bb_bw, ROBUST, "volatility",
     "Bollinger BandWidth(20,2), pandas_ta")

# volatility
_reg("az_atr_pct", _atr_pct, ROBUST, "volatility",
     "ATR(14)/close, pandas_ta [positive control: production atr_rel]")
_reg("az_chaikin_vol", _chaikin_vol, ROBUST, "volatility",
     "Chaikin Volatility(10,10), hand")
_reg("az_hist_vol_20", _hist_vol20, ROBUST, "volatility",
     "Historical volatility: rolling std(20) of log returns, hand")
_reg("az_ulcer_14", _ulcer14, ROBUST, "volatility",
     "Ulcer Index(14), hand")
_reg("az_chop_14", lambda df: __import__("pandas_ta").chop(
    df["high"], df["low"], df["close"], length=14), NAT(0, 100),
    "volatility", "Choppiness Index(14), pandas_ta")
_reg("az_parkinson_20", _parkinson20, ROBUST, "volatility",
     "Parkinson OHLC volatility estimator(20), hand")
_reg("az_garman_klass_20", _garman_klass20, ROBUST, "volatility",
     "Garman-Klass OHLC volatility estimator(20), hand")
_reg("az_thermo", _thermo, ROBUST, "volatility",
     "Elder Thermometer, pandas_ta")

# volume
_reg("az_cmf_20", _cmf20, UNIT, "volume", "Chaikin Money Flow(20), pandas_ta")
_reg("az_mfi_14", _mfi14, NAT(0, 100), "volume", "Money Flow Index(14), pandas_ta")
_reg("az_adosc", _adosc, ROBUST, "volume",
     "Chaikin Oscillator / A-D Oscillator(3,10), pandas_ta")
_reg("az_obv_mom_10", _obv_mom, ROBUST, "volume",
     "OBV 10-bar change / 10-bar volume, pandas_ta OBV")
_reg("az_pvt_mom_10", _pvt_mom, ROBUST, "volume",
     "PVT 10-bar change / 10-bar volume, pandas_ta PVT")
_reg("az_klinger_diff", _klinger, ROBUST, "volume",
     "Klinger Oscillator(34,55) minus signal EMA(13), hand")
_reg("az_eom_14", _eom14, ROBUST, "volume", "Ease of Movement(14), pandas_ta")
_reg("az_force_index", _force_index, ROBUST, "volume",
     "Elder Force Index(13), pandas_ta efi")
_reg("az_nvi_ret_20", _nvi_ret, ROBUST, "volume",
     "Negative Volume Index 20-bar % change, pandas_ta")
_reg("az_pvi_ret_20", _pvi_ret, ROBUST, "volume",
     "Positive Volume Index 20-bar % change, pandas_ta")
_reg("az_vol_osc", _vol_osc, ROBUST, "volume",
     "Volume Oscillator = (EMA5-EMA20)/EMA20 of volume, hand")
_reg("az_vroc_14", _vroc14, ROBUST, "volume",
     "Volume Rate of Change(14), hand")
_reg("az_vwma_dist", _vwma_dist, ROBUST, "volume",
     "(close - VWMA20)/close, pandas_ta")
_reg("az_bwmfi", _bwmfi, ROBUST, "volume",
     "Market Facilitation Index = (high-low)/volume (Bill Williams), hand")

# price / statistical
_reg("az_typical_dev", _typical_dev, ROBUST, "statistical",
     "(close - SMA20 of typical price)/close, hand")
_reg("az_medprice_dev", _medprice_dev, ROBUST, "statistical",
     "(close - SMA14 of median price)/close, hand")
_reg("az_wclose_dev", _wclose_dev, ROBUST, "statistical",
     "(close - SMA14 of weighted close)/close, hand")
_reg("az_zscore_20", _zscore(20), ROBUST, "statistical",
     "Price z-score(20) = (close-SMA)/SD, hand")
_reg("az_zscore_50", _zscore(50), ROBUST, "statistical",
     "Price z-score(50), hand")
_reg("az_ret_skew_20", _ret_skew20, ROBUST, "statistical",
     "Rolling skew(20) of returns, hand")
_reg("az_ret_kurt_20", _ret_kurt20, ROBUST, "statistical",
     "Rolling kurtosis(20) of returns, hand")
_reg("az_acf_1_50", _acf(1), UNIT, "statistical",
     "Autocorrelation of returns, lag 1, window 50, hand")
_reg("az_acf_5_50", _acf(5), UNIT, "statistical",
     "Autocorrelation of returns, lag 5, window 50, hand")
_reg("az_linreg_slope_20", _slope20, ROBUST, "statistical",
     "Linear regression slope(20)/close, pandas_ta")
_reg("az_linreg_dev", _linreg_dev, ROBUST, "statistical",
     "(close - linear regression value(20))/close, pandas_ta")
_reg("az_tsf_dev", _tsf_dev, ROBUST, "statistical",
     "(close - Time Series Forecast(20))/close, pandas_ta linreg+slope")
_reg("az_pv_corr_20", _pv_corr20, UNIT, "statistical",
     "Rolling corr(20) of price returns vs volume returns, hand")
_reg("az_ibs", _ibs, NAT(0, 1), "statistical",
     "Internal Bar Strength = (close-low)/(high-low), hand")
_reg("az_bop", _bop, UNIT, "statistical",
     "Balance of Power = (close-open)/(high-low), pandas_ta")
_reg("az_bull_power", _bull_power, ROBUST, "statistical",
     "Elder Bull Power = (high - EMA13)/close, hand")
_reg("az_bear_power", _bear_power, ROBUST, "statistical",
     "Elder Bear Power = (low - EMA13)/close, hand")
_reg("az_qstick_10", _qstick10, ROBUST, "statistical",
     "QStick(10) = SMA of (close-open), pandas_ta")
_reg("az_squeeze_mom", _squeeze_mom, ROBUST, "statistical",
     "TTM Squeeze momentum (BB/KC), pandas_ta")

# levels (completed prior day)
_reg("az_pivot_dist", _pivot("p"), ROBUST, "levels",
     "(close - classic Pivot Point of prior completed day)/close, hand")
_reg("az_pivot_r1_dist", _pivot("r1"), ROBUST, "levels",
     "(close - prior-day R1)/close, hand")
_reg("az_pivot_s1_dist", _pivot("s1"), ROBUST, "levels",
     "(close - prior-day S1)/close, hand")
_reg("az_adr_rel", _adr_rel, ROBUST, "levels",
     "Mean daily range/close over prior 14 completed days, hand")

# candlestick patterns
_reg("az_pat_doji", _pat_doji, NAT(0, 1), "pattern",
     "Doji flag: body <= 10% of range")
_reg("az_pat_hammer", _pat_hammer, NAT(0, 1), "pattern",
     "Hammer flag: lower wick >= 2x body, upper <= body")
_reg("az_pat_shooting_star", _pat_shooting_star, UNIT, "pattern",
     "Shooting star flag (encoded -1): upper wick >= 2x body")
_reg("az_pat_engulfing", _pat_engulfing, UNIT, "pattern",
     "Engulfing pattern, signed +1 bull / -1 bear")
_reg("az_pat_marubozu", _pat_marubozu, UNIT, "pattern",
     "Marubozu: body >= 90% of range, signed by direction")
_reg("az_pat_three", _pat_three_soldiers, UNIT, "pattern",
     "Three white soldiers (+1) / three black crows (-1), strong-body form")

# calendar (encyclopedia section 1)
_reg("az_hour_sin", _hour_sin, UNIT, "calendar", "sin(2*pi*hour/24)")
_reg("az_hour_cos", _hour_cos, UNIT, "calendar", "cos(2*pi*hour/24)")
_reg("az_dow_sin", _dow_sin, UNIT, "calendar", "sin(2*pi*dayofweek/7)")
_reg("az_dow_cos", _dow_cos, UNIT, "calendar", "cos(2*pi*dayofweek/7)")

AZ_COLUMNS = sorted(REGISTRY)


# ------------------------------------------------------------------
# compute + scaling
# ------------------------------------------------------------------


def compute_raw(df: pd.DataFrame) -> pd.DataFrame:
    """All registered raw indicator series for an OHLCV frame."""
    out = {}
    for name in AZ_COLUMNS:
        s = REGISTRY[name]["build"](df)
        s = pd.Series(np.asarray(s, dtype=float), index=df.index)
        out[name] = s.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=df.index)


def fit_scalers(raw_train: pd.DataFrame) -> dict:
    """Train-split statistics for the robust-scaled indicators."""
    scalers = {}
    for name in AZ_COLUMNS:
        if REGISTRY[name]["scale"] != ROBUST:
            continue
        x = raw_train[name].dropna()
        if len(x) < 50:
            scalers[name] = {"center": 0.0, "scale": 1.0,
                             "degenerate": True}
            continue
        center = float(x.median())
        scale = float((x.quantile(0.995) - x.quantile(0.005)) / 2.0)
        scalers[name] = {"center": center,
                         "scale": scale if scale > 1e-12 else 1.0,
                         "degenerate": bool(scale <= 1e-12)}
    return scalers


def apply_scalers(raw: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Map every raw series into [-1, 1] per its scaling spec."""
    out = pd.DataFrame(index=raw.index)
    for name in AZ_COLUMNS:
        spec = REGISTRY[name]["scale"]
        x = raw[name]
        if spec == UNIT:
            y = x.clip(-1.0, 1.0)
        elif spec == ROBUST:
            sc = scalers[name]
            y = ((x - sc["center"]) / sc["scale"]).clip(-1.0, 1.0)
        else:  # natural
            _, lo, hi = spec
            y = (2.0 * (x - lo) / (hi - lo) - 1.0).clip(-1.0, 1.0)
        out[name] = y
    return out


def compute_az(df: pd.DataFrame, train_end: int):
    """Raw + scaled AZ frame; scalers fitted on df.iloc[:train_end]
    rows only. Returns (scaled, raw, scalers)."""
    raw = compute_raw(df)
    scalers = fit_scalers(raw.iloc[:train_end])
    return apply_scalers(raw, scalers), raw, scalers
