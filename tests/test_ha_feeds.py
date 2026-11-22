"""Book #4 (HA desk) compact-feed tests.

Pins: the desk identity derives from the HA seat manifest; the
market feed's candles ARE Heikin-Ashi candles (Kass's explicit
requirement) while volume stays real; the race carries Book #4
from its fresh $100k; the vision part 0 carries the 10 HA
values and the book's own vote.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboards import compact
from models.ha_seq import heikin_ashi


def _frame(n=130, seed=11):
    rng = np.random.default_rng(seed)
    close = 75000 * np.exp(np.cumsum(rng.normal(0.002, 0.008, n)))
    idx = pd.date_range("2026-10-31", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": close * 0.998, "high": close * 1.013,
        "low": close * 0.991,
        "close": close, "volume": np.full(n, 5.0)}, index=idx)


def test_ha_desk_block_identity_from_manifest():
    desk = compact._ha_desk_block()
    assert desk["book"] == "Book #4"
    assert desk["eyes"]["run"] == 110
    assert desk["eyes"]["champion"] == "Champion #6"
    assert desk["eyes"]["feature_set"] == "ha10"
    assert desk["driver"]["run"] == 44


def test_ha_market_candles_are_heikin_ashi(monkeypatch):
    frame = _frame()
    import data.market_data as md
    monkeypatch.setattr(md, "fetch_klines",
                        lambda *a, **k: frame.copy())
    out0 = compact.build_ha_market(0)
    out1 = compact.build_ha_market(1)
    candles = out0["candles"] + out1["candles"]
    closed = frame.iloc[:-1].tail(96)
    ha = heikin_ashi(frame.iloc[:-1]).tail(96)
    assert len(candles) == 96
    for i in range(96):
        assert candles[i][0] == int(round(ha["ha_open"].iloc[i]))
        assert candles[i][1] == int(round(ha["ha_high"].iloc[i]))
        assert candles[i][2] == int(round(ha["ha_low"].iloc[i]))
        assert candles[i][3] == int(round(ha["ha_close"].iloc[i]))
        assert candles[i][4] == int(round(closed["volume"].iloc[i]))
    # and they are NOT the raw candles (HA smoothing moves them)
    raw_diff = sum(1 for i in range(96)
                   if candles[i][3] != int(round(closed["close"].iloc[i])))
    assert raw_diff > 48


def test_ha_vision_part0_carries_the_ten_ha_values():
    big = _frame(900)
    out = compact.build_ha_vision(0, fetch=lambda *a, **k: big.copy())
    assert out["desk"]["book"] == "Book #4"
    assert len(out["ha_rows"]) == 10
    assert 0.0 <= out["p_up"] <= 1.0
    names = [r[0] for r in out["ha_rows"]]
    assert "ha_streak" in names and "ha_dir" in names


def test_ha_race_carries_book4_from_its_fresh_100k():
    import sqlite3

    race = compact.build_ha_race()
    # From its fresh $100k: the race curve STARTS at the seed.
    # The headline is the book's own latest equity — Book #4 has
    # traded since the 2026-11-05 seed, so it is reconciled
    # against paper_ha.db, never hard-coded to the seed value.
    assert race["b4"]["curve"][0][1] == 100000.0
    con = sqlite3.connect(compact.HA_DB)
    try:
        latest = con.execute(
            "SELECT equity FROM equity ORDER BY ts DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()
    assert race["b4"]["eq"] == round(float(latest), 2)
    assert "Champion #6" in race["b4"]["label"]
    # Book #2's label must derive from the shadow seat manifest —
    # it hard-coded Champion #4 / Run 73 after the 2026-11-09 swap
    # to Champion #11 / Run 119 (regression found 2026-11-10).
    desk2 = compact._shadow_desk_block()
    expected_b2 = (f"{desk2['book_name']} (PPO Run {desk2['driver']['run']} "
                   f"driver · {desk2['eyes']['champion']} Run "
                   f"{desk2['eyes']['run']} eyes)")
    # Since the 2026-11-21 gate addition the seat carries the
    # run-82 gate at 0.51, and the race label must name it like
    # Book #3's label names its gate — derived, not hard-coded.
    gate2 = desk2.get("gate") or {}
    if gate2:
        expected_b2 = (expected_b2[:-1] +
                       f" · Meta-Label Run {gate2['run']} "
                       f"gate @ {gate2['threshold']})")
    assert race["b2"]["label"] == expected_b2
    for key in ("b1", "b2", "b3", "bh"):
        assert key in race
