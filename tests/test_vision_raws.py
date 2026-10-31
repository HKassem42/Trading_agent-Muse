"""Vision-feed raw readings (2026-10-26, Kass): the Live desk's
"What the Bot Sees" keeps the brain's squeezed [-1, 1] inputs AND
shows the same signal in plain human units beside it (RSI as its
0-100 level, distances as %, volume shares as %). The raw readings
are recomputed from the same candle frame the features come from --
never inverted from the squeezed value, because the v3 squeeze
CLIPS at the rails and a clipped value cannot be un-clipped. The
consistency test below is the honesty pin: for every row of a long
synthetic frame, un-squeezing the raw reading with the feature's
own transform must reproduce the brain's actual input.

EMISSION (2026-10-26): emission was briefly gated off while the
dashboard vision schema still capped rows at 3 elements; Kass then
decided the bot's-eyes shape (a grouped heat grid whose cells
carry the plain-unit value), the push_vision schema was widened to
4-element rows, and _VISION_RAWS_EMITTED is True. The two
feed-shape tests pin the emitted 4-element shape."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dashboards import compact  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, FEATURE_SCALES, add_features)

_SCALED = [c for c in FEATURE_COLUMNS if c in FEATURE_SCALES]
_RSI = ["rsi_c", "h4_rsi_c"]
_SHARES = ["obv_slope", "adl_slope", "netvol_rel"]


def _ohlcv(n=1600, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.006, n)
    close = 100_000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    vol = rng.uniform(50, 500, n) * (1 + (rng.random(n) > 0.95) * 3)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_raw_frame_reproduces_every_squeezed_input():
    full = _ohlcv()
    feat = add_features(full)
    raws = compact._raw_frame(full)
    assert list(raws.columns) == FEATURE_COLUMNS
    complete = feat.dropna(subset=FEATURE_COLUMNS)
    assert len(complete) > 500  # the fixture actually exercises rows
    checked = 0
    for ts in complete.index[::37]:  # sample across the whole frame
        row, rrow = feat.loc[ts], raws.loc[ts]
        for c in _SCALED:
            expect = float(np.clip(rrow[c] / FEATURE_SCALES[c], -1, 1))
            assert abs(expect - float(row[c])) < 1e-8, (ts, c)
        for c in _RSI:  # (RSI-50)/50 by construction, never clipped
            assert abs(rrow[c] - (float(row[c]) * 50 + 50)) < 1e-6, (ts, c)
        for c in _SHARES:  # bounded shares: raw IS the squeezed value
            assert abs(rrow[c] - float(row[c])) < 1e-9, (ts, c)
        assert abs(float(np.clip(rrow["volume_change"], -1, 1))
                   - float(row["volume_change"])) < 1e-9, (ts, "volume_change")
        checked += 1
    assert checked >= 10


def test_format_raw_units():
    f = compact._format_raw
    assert f("rsi_c", 62.4) == "62"
    assert f("h4_rsi_c", 41.5) == "42" or f("h4_rsi_c", 41.5) == "41"
    assert f("dist_sma50", 0.01244) == "+1.24%"
    assert f("dist_vwap", -0.004) == "-0.40%"
    assert f("range_pct", 0.0057) == "0.57%"
    assert f("atr_rel", 0.0083) == "0.83%"
    assert f("volume_change", 2.5) == "+250%"
    assert f("volume_change", -0.35) == "-35%"
    assert f("obv_slope", 0.34) == "+34%"
    assert f("netvol_rel", -0.05) == "-5%"
    assert f("ha_dir", 0.83).endswith("ATR") and f("ha_dir", 0.83).startswith("+")
    assert f("ewo", -0.0021) == "-0.21%"
    assert f("rsi_c", float("nan")) is None
    assert f("dist_sma50", None) is None
    assert f("price_change", float("inf")) is None


def test_vision_feed_rows_carry_raw_strings(monkeypatch):
    """EMISSION (2026-10-26): the gate is OPEN -- the dashboard's
    push_vision schema was widened to 4-element rows in the same
    change set as the heat-grid bot's-eyes panel, so the feed MUST
    emit [key, label, squeezed, raw] with an honest plain-unit
    string on every row (the fixture's 66 days warm all 21), and
    each emitted string must equal the raws pipeline's own output
    for the very candle the feed describes. If the emission gate is
    ever closed again, this test must be rewritten to pin the
    gated shape -- never just loosened."""
    import data.market_data as md

    assert compact._VISION_RAWS_EMITTED is True  # the gate itself
    frame = _ohlcv(1600)
    monkeypatch.setattr(md, "fetch_klines", lambda *a, **k: frame.copy())
    out = compact.build_vision()
    assert [r[0] for r in out["f"]] == FEATURE_COLUMNS
    for r in out["f"]:
        assert len(r) == 4, r
        assert isinstance(r[3], str) and r[3], r
    payload = json.dumps(out, separators=(",", ":"))
    print(f"vision feed bytes={len(payload)}")
    assert len(payload) < 3072, "vision feed must stay push-sized"
    feat = add_features(frame.iloc[:-1])
    ts = feat.dropna(subset=FEATURE_COLUMNS).index[-1]
    raws = compact._raw_frame(frame.iloc[:-1])
    by_key = {r[0]: r[3] for r in out["f"]}
    for c in FEATURE_COLUMNS:
        assert by_key[c] == compact._format_raw(c, raws.loc[ts, c]), c


def test_shadow_vision_f21_rows_carry_raw_strings(monkeypatch):
    """Same emission as the Live vision feed (see
    test_vision_feed_rows_carry_raw_strings): f21 rows are
    4-element, raw a string or null where no honest raw exists."""
    import data.market_data as md  # noqa: F401  (parity with desk tests)
    from tests.test_shadow_desk import _ohlcv as desk_ohlcv  # noqa: E402

    meta = json.loads(
        (ROOT / "models" / "shadow_seat" / "stacker.json").read_text())
    columns = list(meta["feature_columns"])
    raw = desk_ohlcv(300, start="2026-10-20")

    def fake_builder(df):
        frame = pd.DataFrame(63.2, index=df.index, columns=columns)
        return frame, list(columns)

    class _Vote:
        def predict_proba(self, X):
            return np.array([[0.4877, 0.5123]])

    monkeypatch.setattr(compact, "_shadow_seat_stacker",
                        lambda: (_Vote(), meta))
    v1 = compact.build_shadow_vision(
        1, fetch=lambda *a, **kw: raw, scored_builder=fake_builder)
    assert [r[0] for r in v1["f21"]] == FEATURE_COLUMNS
    for r in v1["f21"]:
        assert len(r) == 4, r
        assert r[3] is None or isinstance(r[3], str), r
    by_key = {r[0]: r[3] for r in v1["f21"]}
    assert isinstance(by_key["rsi_c"], str)  # warmed well before row 300
