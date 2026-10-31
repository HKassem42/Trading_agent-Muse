"""Position adding / averaging audit tests (2026-10-22, Kass's question:
"can our agent buy or short and then buy again to average up or down?").

What the audit established and these tests pin:

- Adds EXECUTE in the broker (long and short): a further BUY while long
  / SHORT while short fills another tranche of 25% of CURRENT equity,
  bounded by the 75% exposure cap, available cash (longs), and the $10
  minimum ticket. Nothing anywhere treats buy-while-long as a no-op.
- The blended entry is the size-weighted average of the tranches.
- DEFECT (fixed with this file): the initial stop was re-derived from
  the blended entry on every add using the FIRST tranche's ATR, so
  averaging down silently moved the stop AWAY from price -- the shares
  already held lost protection they were promised at entry, while the
  position doubled in size. The fix: an explicit stop anchor set at
  first entry, recomputed on adds from the blended entry and the
  CURRENT ATR, that only ever ratchets TIGHTER (long: never down;
  short: never up). Adds also refresh entry_atr_pct to the current
  ATR context. The dashboard mirror (compact._live_stop_level) reads
  the persisted anchor, and live.py persists it across restarts.

Run: .venv/bin/python -m pytest tests/test_position_adds.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboards.compact import _live_stop_level  # noqa: E402
from environment.trading_env import TradingEnv  # noqa: E402
from features.technical_indicators import FEATURE_COLUMNS, add_features  # noqa: E402
from trading.broker import PaperBroker  # noqa: E402

ATR = 0.01  # base_pct = clip(2*1%, 2.5%, 8%) = 2.5% (floor binds)


# ---------------------------------------------------------------- adds work
def test_long_adds_execute_and_blend_entry(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    f1 = b.buy(100.0, ts="t0", atr_pct=ATR)
    f2 = b.buy(100.0, ts="t1", atr_pct=ATR)
    assert f1 is not None and f2 is not None
    assert abs(f1.qty - 249.625) < 1e-6          # 25k spend, fee haircut
    assert b.btc > f1.qty                        # second tranche added
    assert abs(b.entry_price - 100.0) < 1e-9     # same price -> same blend
    f3 = b.buy(110.0, ts="t2", atr_pct=ATR)      # averaging up blends up
    q1, q2, q3 = f1.qty, f2.qty, f3.qty
    expect = (100.0 * (q1 + q2) + 110.0 * q3) / (q1 + q2 + q3)
    assert abs(b.entry_price - expect) < 1e-9


def test_short_adds_execute_and_blend_entry(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    f1 = b.short_open(100.0, ts="t0", atr_pct=ATR)
    f2 = b.short_open(90.0, ts="t1", atr_pct=ATR)  # add as price falls
    assert f1 is not None and f2 is not None and b.btc < 0
    expect = (100.0 * f1.qty + 90.0 * f2.qty) / (f1.qty + f2.qty)
    assert abs(b.entry_price - expect) < 1e-9


def test_adds_bounded_by_exposure_cap_and_dust_rule(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    fills = [b.buy(100.0, ts=f"t{i}", atr_pct=ATR) for i in range(3)]
    assert all(f is not None for f in fills)
    assert b.btc * 100.0 <= 75_000 + 1           # at the 75% cap
    # Residual headroom at the cap is ~$110 of equity: the next BUY can
    # only fill that sliver (>= the $10 ticket), never another tranche.
    f4 = b.buy(100.0, ts="t3", atr_pct=ATR)
    assert f4 is not None and f4.qty * 100.0 < 200.0
    assert b.btc * 100.0 <= 75_000 + 1           # still capped after top-up


def test_nothing_distinguishes_averaging_up_from_down(tmp_path):
    # Factual pin for the audit answer: the broker applies identical
    # tranche/cap math to an add below entry (down) and above (up).
    down = PaperBroker(tmp_path / "d.db")
    down.buy(100.0, ts="t0", atr_pct=ATR)
    f_down = down.buy(95.0, ts="t1", atr_pct=ATR)
    up = PaperBroker(tmp_path / "u.db")
    up.buy(100.0, ts="t0", atr_pct=ATR)
    f_up = up.buy(105.0, ts="t1", atr_pct=ATR)
    assert f_down is not None and f_up is not None
    # Both are full tranche-sized fills (25% of current equity, modulo
    # mark-to-market): no gate, no size penalty for adding underwater.
    assert f_down.qty * 95.0 > 20_000 and f_up.qty * 105.0 > 20_000


# ------------------------------------------------- DEFECT: stop anchoring
def test_long_averaging_down_must_not_loosen_stop(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=ATR)
    lvl, _ = b.stop_level()
    assert abs(lvl - 97.50) < 1e-9               # promised stop at entry
    b.buy(98.0, ts="t1", atr_pct=ATR)            # average down, size ~doubles
    assert b.entry_price < 100.0                 # blend really moved
    lvl, trailing = b.stop_level()
    assert not trailing
    assert abs(lvl - 97.50) < 1e-9, (
        f"averaging down loosened the stop to {lvl:.4f}: the original "
        "tranche's protection was widened retroactively")


def test_long_averaging_down_into_stop_still_stops_out(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=ATR)
    b.buy(98.0, ts="t1", atr_pct=ATR)
    # 97.40 is above the loosened legacy level (~96.55) but below the
    # promised 97.50: the position must exit here.
    f = b.check_stop(97.40, ts="t2", low=97.40)
    assert f is not None and f.reason == "stop_loss" and b.btc == 0


def test_long_averaging_up_tightens_stop_to_blended_candidate(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    f1 = b.buy(100.0, ts="t0", atr_pct=ATR)
    f2 = b.buy(110.0, ts="t1", atr_pct=ATR)
    blend = (100.0 * f1.qty + 110.0 * f2.qty) / (f1.qty + f2.qty)
    # The anchor ratchets UP to the blended candidate (the trailing
    # stop, armed off the 110 peak, sits even higher -- checked via the
    # effective level being >= the anchor).
    assert abs(b.stop_anchor - blend * 0.975) < 1e-6
    assert b.stop_anchor > 97.50                  # tightened, not loosened
    lvl, _ = b.stop_level()
    assert lvl >= b.stop_anchor


def test_short_averaging_down_must_not_loosen_stop(tmp_path):
    # For a short, "averaging down" = adding as price RISES (worse entry).
    b = PaperBroker(tmp_path / "t.db")
    b.short_open(100.0, ts="t0", atr_pct=ATR)
    lvl, _ = b.stop_level()
    assert abs(lvl - 102.50) < 1e-9
    b.short_open(102.0, ts="t1", atr_pct=ATR)
    assert b.entry_price > 100.0
    lvl, trailing = b.stop_level()
    assert not trailing
    assert abs(lvl - 102.50) < 1e-9, (
        f"short add loosened the stop to {lvl:.4f}")


def test_short_averaging_up_tightens_stop(tmp_path):
    # For a short, adding as price FALLS improves the blend; the stop
    # (above entry) must ratchet DOWN toward price, never stay stale.
    b = PaperBroker(tmp_path / "t.db")
    f1 = b.short_open(100.0, ts="t0", atr_pct=ATR)
    f2 = b.short_open(90.0, ts="t1", atr_pct=ATR)
    blend = (100.0 * f1.qty + 90.0 * f2.qty) / (f1.qty + f2.qty)
    assert abs(b.stop_anchor - blend * 1.025) < 1e-6
    assert b.stop_anchor < 102.50                 # ratcheted DOWN (tighter)
    lvl, _ = b.stop_level()
    assert lvl <= b.stop_anchor                   # trailing may sit lower


def test_add_recomputes_stop_with_current_atr(tmp_path):
    # Enter in calm vol (ATR 4% -> base 8% cap -> stop 92.00), then add
    # in quiet vol (ATR 0.4% -> base 2.5% floor). The blended candidate
    # must use the CURRENT 0.4%, not the stale first-tranche 4%.
    b = PaperBroker(tmp_path / "t.db")
    f1 = b.buy(100.0, ts="t0", atr_pct=0.04)
    lvl, _ = b.stop_level()
    assert abs(lvl - 92.00) < 1e-9
    f2 = b.buy(110.0, ts="t1", atr_pct=0.004)
    blend = (100.0 * f1.qty + 110.0 * f2.qty) / (f1.qty + f2.qty)
    assert abs(b.stop_anchor - blend * 0.975) < 1e-6, (
        f"anchor {b.stop_anchor} does not reflect the add's current "
        "ATR context (the stale first-tranche ATR would give "
        f"{blend * 0.92:.4f})")
    assert b.entry_atr_pct == 0.004              # context refreshed


def test_wider_atr_on_add_cannot_loosen_either(tmp_path):
    # Enter calm (stop 97.50); vol triples by the add (ATR 3% -> base
    # 6%): the candidate off the blend would sit BELOW 97.50 -- the
    # ratchet must refuse it.
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=ATR)
    b.buy(101.0, ts="t1", atr_pct=0.03)
    lvl, _ = b.stop_level()
    assert abs(lvl - 97.50) < 1e-9


def test_anchor_clears_when_flat_and_resets_next_position(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=ATR)
    b.buy(98.0, ts="t1", atr_pct=ATR)
    assert b.stop_anchor is not None
    b.sell(99.0, ts="t2")
    assert b.btc == 0 and b.stop_anchor is None
    b.buy(50.0, ts="t3", atr_pct=ATR)             # fresh position, fresh anchor
    lvl, _ = b.stop_level()
    assert abs(lvl - 48.75) < 1e-9


def test_partial_cover_keeps_anchor(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.short_open(100.0, ts="t0", atr_pct=ATR)
    b.short_open(100.0, ts="t1", atr_pct=ATR)
    b.buy(99.0, ts="t2")                          # tranche cover (partial)
    assert b.btc < 0
    lvl, _ = b.stop_level()
    assert abs(lvl - 102.50) < 1e-9               # remaining size keeps its stop


# ------------------------------------------------------- whole-life peak
def test_trailing_peak_tracks_whole_position_life_across_adds(tmp_path):
    b = PaperBroker(tmp_path / "t.db")
    b.buy(100.0, ts="t0", atr_pct=ATR)
    assert b.check_stop(106.0, ts="t1", low=105.0) is None   # peak 106
    b.buy(104.0, ts="t2", atr_pct=ATR)                       # add below peak
    assert b.high_water == 106.0                             # peak not reset
    lvl, trailing = b.stop_level()
    assert trailing and abs(lvl - 106.0 * (1 - 0.02)) < 1e-6  # trail off 106


# ------------------------------------------------- live persistence/mirror
def test_live_state_anchor_roundtrip_and_dashboard_mirror(tmp_path,
                                                          monkeypatch):
    import trading.live as live
    state_file = tmp_path / "live_state.json"
    monkeypatch.setattr(live, "STATE_FILE", state_file)
    live.save_state({"cash": 50_000.0, "btc": 0.5, "entry_price": 99.0,
                     "entry_atr_pct": ATR, "high_water": 100.0,
                     "low_water": None, "stop_anchor": 97.50,
                     "last_ts": None})
    state = live.load_state()
    assert state["stop_anchor"] == 97.50
    # The dashboard mirror must honor the persisted anchor even though
    # the blended entry (99.0) would imply 96.53 on the legacy formula.
    live_dict = {"last_entry_price": 99.0, "entry_atr_pct": ATR,
                 "btc": 0.5, "high_water": 100.0, "stop_anchor": 97.50}
    assert _live_stop_level(live_dict) == 97.50
    # ...and without an anchor (legacy state) it keeps the old formula.
    live_dict.pop("stop_anchor")
    assert _live_stop_level(live_dict) == round(99.0 * 0.975, 2)


# ----------------------------------------------------- env-level consistency
def _synthetic_df(n=600):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = close * np.exp(rng.normal(0, 0.001, n))
    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    df = pd.DataFrame({
        "open": open_,
        "high": top * (1 + rng.uniform(0, 0.003, n)),
        "low": bot * (1 - rng.uniform(0, 0.003, n)),
        "close": close, "volume": rng.uniform(50, 150, n)}, index=idx)
    return add_features(df)


def test_env_consecutive_buys_add_in_both_fill_modes(tmp_path):
    df = _synthetic_df(600).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    p = np.full(len(df), 0.5)
    for mode in ("close", "next_open"):
        broker = PaperBroker(tmp_path / f"e_{mode}.db")
        env = TradingEnv(df, p, broker, fill_mode=mode)
        env.reset(seed=0, options={"start": 30})
        env.step(1)  # BUY
        q1 = broker.btc
        env.step(1)  # BUY again while long -> ADD, not a no-op
        assert q1 > 0 and broker.btc > q1, mode
        env.step(3)  # SHORT while long -> closes through flat only
        assert broker.btc == 0, mode
        env.step(3)  # SHORT -> opens short
        q2 = broker.btc
        env.step(3)  # SHORT again while short -> ADD
        assert q2 < 0 and broker.btc < q2, mode
