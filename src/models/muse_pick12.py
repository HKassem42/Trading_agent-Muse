"""Muse's Pick #12 — HA Suite feature matrix.

Pre-registered in refs/muse-pick-12-preregistration-2026-11-11.md.
The 10 Heikin-Ashi candle values of Pick #10/#11
(`build_ha_matrix`, unchanged) PLUS 9 suite columns: Kass's
indicator set computed ON the Heikin-Ashi candles, using
production's own transforms (features.technical_indicators'
_rsi / _atr / _scale with the FROZEN FEATURE_SCALES constants —
nothing is fit to the study data).

Suite columns (all in [-1, 1] by the production contract):
- ha_rsi_c:         RSI(14) on HA close, centered (rsi-50)/50
- ha_macd_hist_rel: MACD(12,26,9) histogram on HA close / HA close
- ha_ewo:           (EMA5 - EMA35) on HA close / HA close
- ha_obv_slope:     OBV signed by HA-close moves, 24h change / 24h vol
- ha_adl_slope:     ADL from the HA candle's money-flow multiplier
                    x real volume, 24h change / 24h vol
- ha_netvol_rel:    24h signed-volume share, bars signed by the
                    HA bar's own direction (close vs open)
- ha_atr_rel:       ATR(14) on HA candles / HA close
- ha_dist_sma50:    (HA close - HA SMA50) / HA close
- ha_volatility_21: rolling std/mean of HA close over 21

Every column at row t depends only on raw candles <= t
(values-based construction; pinned by tests/test_muse_pick12.py
truncation checks, the v5.1 lesson).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical_indicators import _atr, _rsi, _scale
from models.ha_seq import heikin_ashi
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix

SUITE_COLUMNS = ["ha_rsi_c", "ha_macd_hist_rel", "ha_ewo",
                 "ha_obv_slope", "ha_adl_slope", "ha_netvol_rel",
                 "ha_atr_rel", "ha_dist_sma50", "ha_volatility_21"]
SUITE_ALL_COLUMNS = HA_COLUMNS + SUITE_COLUMNS


def build_ha_suite_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = build_ha_matrix(df)
    ha = heikin_ashi(df)
    ho = ha["ha_open"]
    hh = ha["ha_high"]
    hl = ha["ha_low"]
    hc = ha["ha_close"]
    vol = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["ha_rsi_c"] = ((_rsi(hc, 14) - 50.0) / 50.0).clip(-1.0, 1.0)
    macd = (hc.ewm(span=12, adjust=False).mean()
            - hc.ewm(span=26, adjust=False).mean())
    signal = macd.ewm(span=9, adjust=False).mean()
    out["ha_macd_hist_rel"] = _scale((macd - signal) / hc,
                                     "macd_hist_rel")
    ema5 = hc.ewm(span=5, adjust=False).mean()
    ema35 = hc.ewm(span=35, adjust=False).mean()
    out["ha_ewo"] = _scale((ema5 - ema35) / hc, "ewo")

    vol24 = vol.rolling(24).sum().replace(0, np.nan)
    obv = (vol * np.sign(hc.diff().fillna(0.0))).cumsum()
    out["ha_obv_slope"] = (obv.diff(24) / vol24).clip(-1.0, 1.0)
    rng = (hh - hl).replace(0, np.nan)
    mfm = ((hc - hl) - (hh - hc)) / rng
    adl = (mfm.fillna(0.0) * vol).cumsum()
    out["ha_adl_slope"] = (adl.diff(24) / vol24).clip(-1.0, 1.0)
    signed_bar = vol * np.sign(hc - ho)
    out["ha_netvol_rel"] = (signed_bar.rolling(24).sum()
                            / vol24).clip(-1.0, 1.0)

    ha_df = pd.DataFrame({"high": hh, "low": hl, "close": hc})
    out["ha_atr_rel"] = _scale(_atr(ha_df, 14) / hc, "atr_rel")
    sma50 = hc.rolling(50).mean()
    out["ha_dist_sma50"] = _scale((hc - sma50) / hc, "dist_sma50")
    roll = hc.rolling(21)
    out["ha_volatility_21"] = _scale(roll.std() / roll.mean(),
                                     "volatility_21")
    return pd.concat([base, out[SUITE_COLUMNS]], axis=1)[
        SUITE_ALL_COLUMNS]
