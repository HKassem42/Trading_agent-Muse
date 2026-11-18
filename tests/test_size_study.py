"""Tranche-size study mechanics (2026-10-22, Kass's request: "lower
the 25 percent of total portfolio value to 5 and then re run the
numbers").

Pins: (1) the 5% tranche dollar math on a fixture book, (2) position
adds at 5% -- they fill, they count, and the 75% exposure cap binds
(15 full tranches in, the next ones are cap-limited, then refused),
(3) the production stop-anchor ratchet (2026-10-22 audit fix) is
active inside the study broker, long and short, and (4) the
load-bearing identity checks. One wrinkle the checks document:
three of the four recorded 25% duels were graded BEFORE the
stop-anchor fix landed the same evening, so they are pre-fix
numbers. Lag F standard (run 26: agg -1,871.18 / P&L -$4,416.98 /
40 trips) reproduces to the cent under CURRENT code. The other
three are pinned twice: under current code (the study's true 25%
baselines) and under a pre-fix emulation broker, which reproduces
the recorded duels to the cent (Lag F robustness -9,527.32 /
-$23,167.08, PPO champion -2,062.97 / -$4,484.72, Focus B
-3,476.80 / -$8,432.41) -- proof the gap is the production fix,
not harness drift. Only with the 25% pins green do the 5% numbers
mean anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import models.size_study as size_study  # noqa: E402
from models.size_study import (SizeBroker, evaluate_windows_size,  # noqa: E402
                               load_focusb_signal, load_lagf_signal,
                               load_model, load_ppo_signal, segment_adds)


def _broker(tmp_path, tranche=0.05, name="b.db"):
    return SizeBroker(tmp_path / name, starting_balance=100_000,
                      max_position_pct=tranche, stop_loss_pct=0.05)


# ------------------------------------------------------- tranche math
def test_tranche_dollar_amount_at_5pct(tmp_path):
    """First buy on a $100k book at 5%: exactly $5,000 leaves cash,
    the 0.15% side fee is $7.50, and an OPEN is not counted as an add."""
    b = _broker(tmp_path)
    fill = b.buy(100.0)
    assert fill is not None
    assert b.cash == pytest.approx(95_000.0)
    assert fill.fee == pytest.approx(7.50)
    assert fill.qty == pytest.approx(50.0 * (1 - 0.0015))
    assert b.tele["add_attempts"] == 0 and b.tele["adds_filled"] == 0


def test_adds_fill_and_cap_binds_at_5pct(tmp_path):
    """Constant price: 5% tranches stack toward the 75% cap (15
    tranches of headroom). Adds are counted as they fill; once the
    residual headroom is the binding constraint the fills shrink
    (cap_limited) and finally a buy is refused outright (cap_blocked).
    Position value never exceeds 75% of equity."""
    b = _broker(tmp_path)
    assert b.buy(100.0) is not None  # the open
    refused = 0
    for _ in range(30):
        eq = b.equity(100.0)  # equity at THIS fill's price, pre-fill
        if b.buy(100.0) is None:
            refused += 1
            if refused >= 2:
                break
        # the fill's notional never exceeded the headroom it saw
        assert abs(b.btc) * 100.0 <= 0.75 * eq + 1e-6
    assert refused >= 1, "the cap never refused an add"
    t = b.tele
    assert t["add_attempts"] > t["adds_filled"] >= 14
    assert t["cap_limited"] >= 1, "no add was ever shrunk by the cap"
    assert t["cap_blocked"] >= 1, "no add was ever refused by the cap"
    # ~15 tranches of 5% fill the 75% cap: the book ends up ~75% in.
    assert abs(b.btc) * 100.0 >= 0.70 * b.equity(100.0)


def test_short_adds_mirror_the_cap(tmp_path):
    b = _broker(tmp_path)
    assert b.short_open(100.0) is not None
    for _ in range(30):
        eq = b.equity(100.0)
        if b.short_open(100.0) is None:
            break
        # NB: shorts book the FULL notional as quantity (only the
        # proceeds are fee-discounted), so vs POST-fee equity the
        # ratio can sit a fill's fee above 75% -- the cap itself is
        # enforced pre-fill, which is what this asserts.
        assert abs(b.btc) * 100.0 <= 0.75 * eq + 1e-6
    assert b.tele["adds_filled"] >= 14
    assert b.tele["cap_blocked"] >= 1


# ------------------------------------------------- stop anchor active
def test_stop_anchor_ratchet_active_on_long_adds(tmp_path):
    """The 2026-10-22 audit fix must be live in the study broker: an
    add BELOW entry (averaging down) recomputes the anchor off the
    blend but may only ratchet it TIGHTER -- the promised stop of the
    shares already held (97.50 on a 100 entry at ATR 1%) survives."""
    b = _broker(tmp_path)
    b.buy(100.0, atr_pct=0.01)
    assert b.stop_anchor == pytest.approx(97.50)
    b.buy(90.0, atr_pct=0.01)  # average down: blend < 100
    assert b.entry_price < 100.0
    assert b.stop_anchor == pytest.approx(97.50)


def test_stop_anchor_ratchet_active_on_short_adds(tmp_path):
    b = _broker(tmp_path)
    b.short_open(100.0, atr_pct=0.01)
    assert b.stop_anchor == pytest.approx(102.50)
    b.short_open(110.0, atr_pct=0.01)  # averaging a short up in price
    assert b.entry_price > 100.0
    assert b.stop_anchor == pytest.approx(102.50)


# ------------------------------------------------------- segment adds
def test_segment_adds_counting():
    def seg(btcs):
        return {"fills": [{"btc_after": x} for x in btcs]}
    assert segment_adds(seg([0.5, 0.5])) == 0            # open only
    assert segment_adds(seg([0.5, 1.0, 1.5, 0.0])) == 2  # open + 2 adds
    assert segment_adds(seg([-0.5, -1.0, -0.4, -0.9, 0.0])) == 2
    # open, add, partial cover (decrease), add again -> 2 adds
    assert segment_adds(seg([0.5, 1.0, 0.6, 1.1, 0.0])) == 2


# ------------------------------------------------------------ identity
def test_lagf_25_identity_standard():
    """The 25% pins come FIRST: the new harness must reproduce run
    26's recorded standard-window duel before its 5% numbers count."""
    windows, p_ups = load_lagf_signal("standard")
    res = evaluate_windows_size(windows, p_ups, load_model("lagf"), 0.25)
    assert res["total_round_trips"] == 40
    assert res["agg_score"] == pytest.approx(-1871.1787, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-4416.9757, abs=0.01)
    assert res["bench_pnl"] == pytest.approx(717.5366, abs=0.01)


def test_lagf_25_identity_robustness():
    """Current-code 25% baseline. The exit study's recorded
    robustness numbers (agg -9,527.32 / P&L -$23,167.08) were graded
    BEFORE the 2026-10-22 stop-anchor fix -- see the pre-fix
    reproduction test below. With the ratchet active (this study's
    specified configuration) the same signal grades -8,587.86."""
    windows, p_ups = load_lagf_signal("robustness")
    res = evaluate_windows_size(windows, p_ups, load_model("lagf"), 0.25)
    assert res["total_round_trips"] == 75
    assert res["agg_score"] == pytest.approx(-8587.8592, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-21288.1523, abs=0.01)
    assert res["bench_pnl"] == pytest.approx(27401.6614, abs=0.01)


def test_ppo_25_identity_standard():
    """PPO champion (run 17's brain) on the standard windows,
    current-code baseline. The focus build's recorded re-grade
    (agg -2,062.97 / P&L -$4,484.72) predates the stop-anchor fix;
    with the ratchet active the champion grades -2,037.24 /
    -$4,420.64. B&H (+$717.54) is broker-independent and matches."""
    windows, p_ups = load_ppo_signal()
    res = evaluate_windows_size(windows, p_ups, load_model("ppo"), 0.25)
    assert res["agg_score"] == pytest.approx(-2037.2361, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-4420.6411, abs=0.01)
    assert res["bench_pnl"] == pytest.approx(717.5366, abs=0.01)


def test_focusb_25_identity_standard():
    """GBM Focus B (run 22, the most active arm: 66 round trips),
    current-code baseline. Recorded pre-fix: agg -3,476.80 /
    P&L -$8,432.41; with the ratchet active: -3,329.80 / -$8,138.37."""
    windows, p_ups = load_focusb_signal()
    res = evaluate_windows_size(windows, p_ups, load_model("focusb"),
                                0.25)
    assert res["total_round_trips"] == 66
    assert res["agg_score"] == pytest.approx(-3329.8011, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-8138.3720, abs=0.01)
    assert res["bench_pnl"] == pytest.approx(717.5366, abs=0.01)


def test_ppo_signal_votes_are_frozen_not_live(tmp_path, monkeypatch):
    """The PPO signal's stacker votes are FROZEN to the champion
    that graded the recorded duels (pre-flight Finding 1, fixed
    2026-10-24): load_ppo_signal must read the study fixture under
    data/size_study/, never the live models/stacker_champion.pkl.
    Simulated future promotion: the module's MODELS dir is pointed
    at a scratch dir whose stacker_champion.pkl is a DIFFERENT
    model -- the loader's votes must still equal the fixture's
    votes exactly, and must differ from the live driver's votes.

    2026-10-25 (seat separation): the contrast voter is now the
    DRIVER'S-SEAT stacker (models/live_seat/stacker.pkl) -- the
    model live trading actually uses, a production-21 brain from a
    later lineage than the fixture. The lab champion can no longer
    play that role: Champion #2 (run 64) reads the scored-102
    feature set and cannot vote on these windows at all. It serves
    as the scratch decoy instead -- if the loader ever read the
    path again, it would crash on the feature mismatch, so a pass
    proves the freeze, it doesn't assume it."""
    import joblib
    import shutil

    import numpy as np

    from features.technical_indicators import FEATURE_COLUMNS

    windows, p_ups = load_ppo_signal()
    frozen = joblib.load(size_study.FROZEN_STACKER)
    expect = frozen.predict_proba(windows[0][FEATURE_COLUMNS])[:, 1]
    # atol, not array_equal: two loads of the same stacker pickle can
    # differ by ~1e-16 on a row or two (parallel tree reductions);
    # the freeze claim is about WHICH model, not bit identity.
    assert np.allclose(p_ups[0], expect, atol=1e-9)

    live_now = joblib.load(size_study.MODELS / "live_seat" / "stacker.pkl")
    live_votes = live_now.predict_proba(windows[0][FEATURE_COLUMNS])[:, 1]
    assert np.abs(live_votes - expect).max() > 1e-3  # a genuinely different brain
    scratch_models = tmp_path / "models"
    scratch_models.mkdir()
    shutil.copy2(size_study.MODELS / "stacker_champion.pkl",
                 scratch_models / "stacker_champion.pkl")
    monkeypatch.setattr(size_study, "MODELS", scratch_models)
    _, p_ups2 = load_ppo_signal()
    assert np.allclose(p_ups2[0], expect, atol=1e-9)


def test_ppo_brain_is_frozen_not_live(tmp_path, monkeypatch):
    """The PPO brain under test is FROZEN to run 17's champion (the
    FROZEN_STACKER discipline applied to the brain itself, fixed
    2026-10-24): load_model('ppo') must load the study fixture
    under data/size_study/, never the live models/ppo_champion.zip.
    Simulated champion change: the module's MODELS dir is pointed
    at a scratch dir whose ppo_champion.zip is a DIFFERENT brain
    (a copy of the current live champion -- run 44's, installed by
    Kass's manual override) -- the loaded brain's policy parameters
    must still equal the fixture's exactly."""
    import numpy as np
    from stable_baselines3 import PPO

    def params(model):
        return {k: v.detach().cpu().numpy().copy()
                for k, v in model.policy.state_dict().items()}

    frozen_params = params(PPO.load(size_study.FROZEN_PPO))

    size_study._MODELS_CACHE.clear()
    brain_params = params(load_model("ppo"))
    assert brain_params.keys() == frozen_params.keys()
    assert all(np.array_equal(brain_params[k], frozen_params[k])
               for k in frozen_params)

    scratch_models = tmp_path / "models"
    scratch_models.mkdir()
    live_zip = size_study.MODELS / "ppo_champion.zip"
    (scratch_models / "ppo_champion.zip").write_bytes(
        live_zip.read_bytes())
    live_params = params(PPO.load(scratch_models / "ppo_champion.zip"))
    # a genuinely different brain: at least one parameter differs
    assert any(not np.array_equal(live_params[k], frozen_params[k])
               for k in frozen_params)
    monkeypatch.setattr(size_study, "MODELS", scratch_models)
    size_study._MODELS_CACHE.clear()
    try:
        brain2_params = params(load_model("ppo"))
        assert all(np.array_equal(brain2_params[k], frozen_params[k])
                   for k in frozen_params)
    finally:
        size_study._MODELS_CACHE.clear()


# ------------------------------------------- pre-fix recorded grades
class PreFixBroker(SizeBroker):
    """The broker as it was when the focus/lag builds recorded their
    duels (before the 2026-10-22 stop-anchor fix): no anchor -- the
    stop re-derived from the blended entry on every add, and the
    first tranche's ATR stayed frozen (adds never refreshed it).
    Emulated by neutralizing the ratchet hook and clearing the
    anchor after every entry, which returns stop_level to its
    entry-derived fallback permanently."""

    def _ratchet_anchor_on_add(self, is_long, old_entry, old_atr,
                               new_atr):
        pass

    def buy(self, price, reason="signal", ts=None, atr_pct=None,
            context=None):
        fill = super().buy(price, reason=reason, ts=ts,
                           atr_pct=atr_pct, context=context)
        self.stop_anchor = None
        return fill

    def short_open(self, price, reason="signal", ts=None, atr_pct=None,
                   context=None):
        fill = super().short_open(price, reason=reason, ts=ts,
                                  atr_pct=atr_pct, context=context)
        self.stop_anchor = None
        return fill


def test_prefix_broker_reproduces_the_recorded_grades():
    """The load-bearing provenance check: driven by the PRE-FIX
    broker, this harness reproduces the recorded 25% duels TO THE
    CENT -- Lag F robustness (-9,527.32 / -$23,167.08 / 75 trips,
    exit-study record), the PPO champion's focus-build re-grade
    (-2,062.97 / -$4,484.72) and Focus B (-3,476.80 / -$8,432.41 /
    66 trips). That is the evidence that the current-code baselines
    above differ from the records ONLY because production's stop
    changed in between -- not because this harness drifts."""
    windows, p_ups = load_lagf_signal("robustness")
    res = evaluate_windows_size(windows, p_ups, load_model("lagf"),
                                0.25, broker_cls=PreFixBroker)
    assert res["total_round_trips"] == 75
    assert res["agg_score"] == pytest.approx(-9527.3229, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-23167.0798, abs=0.01)

    windows, p_ups = load_ppo_signal()
    res = evaluate_windows_size(windows, p_ups, load_model("ppo"),
                                0.25, broker_cls=PreFixBroker)
    assert res["agg_score"] == pytest.approx(-2062.9741, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-4484.7216, abs=0.01)

    windows, p_ups = load_focusb_signal()
    res = evaluate_windows_size(windows, p_ups, load_model("focusb"),
                                0.25, broker_cls=PreFixBroker)
    assert res["total_round_trips"] == 66
    assert res["agg_score"] == pytest.approx(-3476.7968, abs=0.01)
    assert res["total_pnl"] == pytest.approx(-8432.4099, abs=0.01)
