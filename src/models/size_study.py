"""Tranche-size study (2026-10-22, Kass's request: "lower the 25
percent of total portfolio value to 5 and then re run the numbers").

Re-grades the reference signals at tranche = 5% of equity against
their recorded 25% duels, everything else frozen: same signals,
same production exit recipe (the exit study's default ExitConfig --
stop 2xATR% clipped [2.5%, 8%], trailing arm +3% / trail 1.5xATR%
clipped [2%, 6%], no take-profit, T0 through-flat, and the
stop-anchor ratchet the 2026-10-22 averaging audit put into
PaperBroker), same 75% exposure cap, same 0.15% side fee, same $10
minimum ticket, next-candle-open fills.

The production default (PaperBroker max_position_pct=0.25) is NOT
changed by this module: the tranche is a broker constructor argument,
threaded through a mirror of nightly.evaluate_ppo's loop exactly the
way models.exit_study mirrored it for exit configs. Tests pin the
25% cells to the cent -- with one documented wrinkle: the recorded
duels for the PPO champion, Focus B and Lag F robustness were graded
BEFORE the 2026-10-22 stop-anchor fix, under a broker whose stop
re-derived from the blended entry on every add. The tests prove
this (a pre-fix emulation broker reproduces all three recorded
numbers exactly) and pin the study's actual 25% baselines under
CURRENT production code -- ratchet active, as this study is
specified to run. Lag F standard is the control: its recorded
numbers reproduce under BOTH broker regimes (verified), so they are
also its current-code numbers. The 5% cells are graded through this
same harness, so every 25% -> 5% delta is same-code, same-harness.

Entry/exit asymmetry at any tranche size (broker code, stated for
the report): entries are GRADUAL -- each buy/short action adds one
tranche, so a position scales in over up to cap/tranche actions
(3 at 25%, 15 at 5%). Exits are asymmetric by side: a long is
closed in ONE fill (SELL sells 100%; a SHORT action while long also
sells 100% through flat), and every protective exit (stop-loss,
trailing stop) closes/covers 100% in one fill on both sides -- but a
SIGNAL cover of a short is tranche-paced: broker.buy while short
routes to _cover_tranche, which buys back at most one tranche
(max_position_pct of equity) per action. At 5% a full-size short
therefore unwinds over as many as 15 buy actions, where at 25% it
took 3.

Telemetry the production scorer doesn't keep, added here: fills are
segmented flat->flat (models.exit_study.segment_trips) to count
ADDS per position episode (fills after the opener that grow the
position), total fees actually paid, how often the 75% cap was the
binding constraint on an add (cap_limited: the add filled but the
cap, not the tranche budget, set its size; cap_blocked: the add was
refused because residual headroom fell under the $10 ticket), and
the time-averaged deployed capital (mean |position value| over all
steps, and over in-position steps only) so P&L can be normalized by
the capital actually at work.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models.exit_study import (EPISODES, STARTING_BALANCE, ExitConfig,
                               StudyBroker, StudyEnv, _conv_for_rows,
                               _read_new_fills, _windows_from,
                               load_lagf_signal, segment_trips)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

# The stacker whose votes graded the PPO champion's recorded duels:
# the champion archived by the 2026-10-23 midnight retrain just
# before it promoted its successor (holdout 0.4798 > 0.4609). Frozen
# as a study fixture so future promotions can never move the pins
# (pre-flight Finding 1, fixed 2026-10-24) -- the same discipline
# Focus B's cached probs_B.npz already had.
FROZEN_STACKER = DATA / "size_study" / "frozen_stacker_champion.pkl"

# The PPO brain this study grades: run 17's champion, archived by
# Kass's 2026-10-24 manual override just before run 44's challenger
# was installed as the live champion. The loader used to read the
# LIVE models/ppo_champion.zip, so the override silently swapped
# the signal under test and broke the pinned identity tests
# (-2,037.2361 -> -662.43); the frozen fixture reproduces the pins
# to the cent, and no future champion change can move them again --
# the same discipline as FROZEN_STACKER above.
FROZEN_PPO = DATA / "size_study" / "frozen_ppo_champion.zip"

__all__ = ["SizeBroker", "evaluate_windows_size", "segment_adds",
           "load_lagf_signal", "load_focusb_signal", "load_ppo_signal",
           "load_model", "FROZEN_STACKER", "FROZEN_PPO"]


# ----------------------------------------------------------------- broker
class SizeBroker(StudyBroker):
    """StudyBroker (default ExitConfig == production exits, stop
    anchor included via PaperBroker) with add/cap telemetry.

    The overrides only OBSERVE: they classify the add attempt from
    the same budget/headroom arithmetic PaperBroker is about to run,
    then pass straight through. Behavior is production behavior at
    whatever max_position_pct the constructor received."""

    def __init__(self, db_path, starting_balance=STARTING_BALANCE,
                 max_position_pct=0.05, stop_loss_pct=0.05,
                 max_exposure_pct=0.75, cfg: ExitConfig | None = None):
        super().__init__(db_path, starting_balance=starting_balance,
                         max_position_pct=max_position_pct,
                         stop_loss_pct=stop_loss_pct,
                         max_exposure_pct=max_exposure_pct,
                         cfg=cfg or ExitConfig())
        self.tele = {"add_attempts": 0, "adds_filled": 0,
                     "cap_limited": 0, "cap_blocked": 0}

    def _cap_state(self, price: float, is_long: bool) -> tuple[bool, bool]:
        """(cap_binds, cap_blocks) for an add attempt at `price`,
        computed with PaperBroker's own spend arithmetic: the cap
        binds when the residual headroom is the min term AND is below
        the tranche budget (the tranche, not the cap, is then not
        what sized the fill); it blocks when that spend falls under
        the $10 minimum ticket."""
        eq = self.equity(price)
        budget = eq * self.max_position_pct
        head = max(eq * self.max_exposure_pct - abs(self.btc) * price,
                   0.0)
        terms = [budget, head] + ([self.cash] if is_long else [])
        spend = min(terms)
        binds = head <= spend + 1e-9 and head < budget - 1e-9
        return binds, bool(binds and spend < 10.0)

    def _note_add(self, state, fill) -> None:
        """Tally one add attempt. `state` is the (binds, blocked) pair
        computed BEFORE the fill landed (post-fill headroom would
        misclassify); cap_limited counts adds that FILLED but were
        sized by the cap instead of the tranche budget, cap_blocked
        counts adds the cap refused outright."""
        binds, blocked = state
        self.tele["add_attempts"] += 1
        if fill is not None:
            self.tele["adds_filled"] += 1
            if binds:
                self.tele["cap_limited"] += 1
        elif blocked:
            self.tele["cap_blocked"] += 1

    def buy(self, price, reason="signal", ts=None, atr_pct=None,
            context=None):
        is_add = self.btc > 0  # btc < 0 routes to a cover, not an add
        state = self._cap_state(price, True) if is_add else None
        fill = super().buy(price, reason=reason, ts=ts, atr_pct=atr_pct,
                           context=context)
        if is_add:
            self._note_add(state, fill)
        return fill

    def short_open(self, price, reason="signal", ts=None, atr_pct=None,
                   context=None):
        is_add = self.btc < 0  # btc > 0 is a through-flat close
        state = self._cap_state(price, False) if is_add else None
        fill = super().short_open(price, reason=reason, ts=ts,
                                  atr_pct=atr_pct, context=context)
        if is_add:
            self._note_add(state, fill)
        return fill


# ---------------------------------------------------------------- segments
def segment_adds(seg: dict) -> int:
    """Add fills inside one flat->flat segment: fills after the
    opener that GROW |position| (a re-add after a partial cover
    counts too -- it is capital committed after the open)."""
    inc, prev = 0, 0.0
    for f in seg["fills"]:
        if abs(f["btc_after"]) > abs(prev) + 1e-9:
            inc += 1
        prev = f["btc_after"]
    return max(inc - 1, 0)


def _adds_hist(adds: list[int]) -> dict:
    hist = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for a in adds:
        hist[str(a) if a < 5 else "5+"] += 1
    return hist


# ------------------------------------------------------------------ driver
def evaluate_windows_size(windows, p_ups, model,
                          tranche_pct: float,
                          broker_cls=SizeBroker) -> dict:
    """evaluate_ppo/evaluate_windows with a tranche parameter and
    size telemetry: same 10x24 next-open episodes, same scoring
    (window score = total P&L - 0.5 x worst episode drawdown, agg =
    mean of window scores), plus per-cell fees, adds, cap and
    deployed-capital accounting segmented from the ledger.
    `broker_cls` exists for the tests' pre-fix emulation broker; the
    study itself always runs the default (current production code)."""
    per = []
    all_closed: list[dict] = []
    all_adds: list[int] = []
    tele_total = {"add_attempts": 0, "adds_filled": 0,
                  "cap_limited": 0, "cap_blocked": 0}
    fees_total = 0.0
    dep_all_steps: list[float] = []
    dep_inpos_steps: list[float] = []
    for w_i, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        tmp = Path("/tmp") / f"tam_size_{os.getpid()}_{w_i}.db"
        if tmp.exists():
            tmp.unlink()
        try:
            broker = broker_cls(tmp, starting_balance=STARTING_BALANCE,
                                max_position_pct=tranche_pct,
                                stop_loss_pct=0.05, cfg=ExitConfig())
            env = StudyEnv(w_sup, w_pup, broker, fill_mode="next_open")
            closes_pos = w_sup["close"].to_numpy(dtype=float)
            rewards, pnls, trips_env, ep_dds = [], [], [], []
            action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
            offset = 0
            pnl_long = pnl_short = 0.0
            open_left = 0
            w_fees = 0.0
            w_adds: list[int] = []
            tele0 = dict(broker.tele)
            for ep, start in enumerate([k * 24 for k in range(EPISODES)]):
                obs, _ = env.reset(seed=ep, options={"start": start})
                start_eq = env._day_start_equity
                curve = [start_eq]
                done, ep_r = False, 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    action_counts[["hold", "buy", "sell",
                                   "short"][int(action)]] += 1
                    obs, r, done, _, _ = env.step(int(action))
                    ep_r += r
                    curve.append(env._prev_eq)
                    ci = min(env._cursor, len(closes_pos) - 1)
                    dep = abs(broker.btc) * float(closes_pos[ci])
                    dep_all_steps.append(dep)
                    if dep > 1e-9:
                        dep_inpos_steps.append(dep)
                ep_pnl = env._prev_eq - start_eq
                rewards.append(ep_r)
                pnls.append(ep_pnl)
                trips_env.append(env._round_trips)
                c = np.array(curve)
                ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
                fills, offset = _read_new_fills(tmp, offset)
                w_fees += float(sum(f["fee"] for f in fills))
                segs = segment_trips(fills, STARTING_BALANCE)
                closed_pnl = 0.0
                open_seg = None
                for s in segs:
                    w_adds.append(segment_adds(s))
                    if s["closed"]:
                        closed_pnl += s["pnl"]
                        all_closed.append(s)
                        if s["side"] > 0:
                            pnl_long += s["pnl"]
                        else:
                            pnl_short += s["pnl"]
                    else:
                        open_seg = s
                if open_seg is not None:
                    # The episode's mark-to-market remainder belongs
                    # to the still-open segment (exit-study rule), so
                    # the long/short split sums to the window P&L.
                    residual = ep_pnl - closed_pnl
                    if open_seg["side"] > 0:
                        pnl_long += residual
                    else:
                        pnl_short += residual
                    open_left += 1
            total = float(np.sum(pnls))
            max_dd = float(np.max(ep_dds)) if ep_dds else 0.0
            total_actions = sum(action_counts.values()) or 1
            closes = w_sup["close"].to_numpy(dtype=float)
            bench = (float(STARTING_BALANCE * (closes[-1] / closes[0] - 1.0))
                     if len(closes) > 1 else 0.0)
            w_tele = {k: broker.tele[k] - tele0[k] for k in broker.tele}
            for k in tele_total:
                tele_total[k] += w_tele[k]
            fees_total += w_fees
            all_adds.extend(w_adds)
            per.append({
                "mean_reward": float(np.mean(rewards)),
                "total_pnl": total,
                "max_drawdown": max_dd,
                "score": total - 0.5 * max_dd,
                "bench_pnl": bench,
                "round_trips_env": int(sum(trips_env)),
                "action_mix": {k: round(v / total_actions, 3)
                               for k, v in action_counts.items()},
                "pnl_long": pnl_long, "pnl_short": pnl_short,
                "open_left": open_left,
                "fees": w_fees,
                "tele": w_tele,
                "segments": len(w_adds),
            })
        finally:
            tmp.unlink(missing_ok=True)
    wins = [s for s in all_closed if s["pnl"] > 0]
    losses = [s for s in all_closed if s["pnl"] <= 0]
    avg_dep_all = float(np.mean(dep_all_steps)) if dep_all_steps else 0.0
    avg_dep_inpos = (float(np.mean(dep_inpos_steps))
                     if dep_inpos_steps else 0.0)
    total_pnl = float(np.sum([w["total_pnl"] for w in per]))
    return {
        "windows": per,
        "agg_score": float(np.mean([w["score"] for w in per])),
        "total_pnl": total_pnl,
        "bench_pnl": float(np.sum([w["bench_pnl"] for w in per])),
        "max_drawdown": float(max(w["max_drawdown"] for w in per)),
        "total_round_trips": int(sum(w["round_trips_env"] for w in per)),
        "closed_trips": len(all_closed),
        "wins": len(wins), "losses": len(losses),
        "avg_win": float(np.mean([s["pnl"] for s in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([s["pnl"] for s in losses])) if losses else 0.0,
        "pnl_long": float(sum(w["pnl_long"] for w in per)),
        "pnl_short": float(sum(w["pnl_short"] for w in per)),
        "open_left": int(sum(w["open_left"] for w in per)),
        "action_mix": per[-1]["action_mix"],
        "fees_total": fees_total,
        "segments": int(sum(w["segments"] for w in per)),
        "adds_mean": float(np.mean(all_adds)) if all_adds else 0.0,
        "adds_median": float(np.median(all_adds)) if all_adds else 0.0,
        "adds_max": int(max(all_adds)) if all_adds else 0,
        "adds_hist": _adds_hist(all_adds),
        "tele": tele_total,
        "avg_deployed_all": avg_dep_all,
        "avg_deployed_inpos": avg_dep_inpos,
        "pnl_pct_deployed_all": (100.0 * total_pnl / avg_dep_all
                                 if avg_dep_all > 0 else 0.0),
        "pnl_pct_deployed_inpos": (100.0 * total_pnl / avg_dep_inpos
                                   if avg_dep_inpos > 0 else 0.0),
    }


# ------------------------------------------------------------------ signals
def load_focusb_signal():
    """(windows, p_ups) for GBM Focus B (run 22) on the standard
    windows, from the focus build's cached final-model probabilities
    (probs_B.npz) -- bit-identical to its recorded grading."""
    sup = pd.read_pickle(DATA / "ha_seq" / "sup.pkl")
    windows = _windows_from(sup, len(sup) - 792)
    z = np.load(DATA / "gbm_focus" / "probs_B.npz")
    prob_by_ts = {pd.Timestamp(t): p for t, p in zip(z["ts"], z["proba"])}
    p_ups = [(_conv_for_rows(prob_by_ts, w.index) + 1.0) / 2.0
             for w in windows]
    return windows, p_ups


def load_ppo_signal():
    """(windows, stacker votes) for the PPO champion on the standard
    windows -- exactly the focus build's champion re-grade inputs:
    P(up) over each window's feature rows, the frame its recorded
    -2,062.97 re-grade was computed on.

    The votes are FROZEN to the stacker champion that graded those
    recorded duels -- the pre-promotion champion archived by the
    2026-10-23 midnight retrain (holdout 0.4609), copied to
    data/size_study/frozen_stacker_champion.pkl. The loader used to
    read the LIVE models/stacker_champion.pkl, so the Oct 23
    promotion (0.4798) silently changed this signal's inputs and
    broke the pinned identity tests (pre-flight Finding 1); the
    frozen fixture reproduces the pins to the cent, and no future
    promotion can move them again."""
    import joblib

    from features.technical_indicators import FEATURE_COLUMNS

    sup = pd.read_pickle(DATA / "ha_seq" / "sup.pkl")
    windows = _windows_from(sup, len(sup) - 792)
    stacker_model = joblib.load(FROZEN_STACKER)
    p_ups = [stacker_model.predict_proba(w[FEATURE_COLUMNS])[:, 1]
             for w in windows]
    return windows, p_ups


_MODELS_CACHE: dict = {}


def load_model(which: str):
    """The drivable brain for a signal key: 'lagf' / 'focusb' are the
    GBM rule zips (conviction rule over the p_up channel); 'ppo' is
    run 17's champion brain, loaded from the FROZEN study fixture
    (FROZEN_PPO) -- never the live champion, whose identity changes
    with promotions and manual overrides."""
    if which not in _MODELS_CACHE:
        from stable_baselines3 import PPO

        if which == "lagf":
            from models.exit_study import _rule_model
            _MODELS_CACHE[which] = _rule_model()
        elif which == "focusb":
            _MODELS_CACHE[which] = PPO.load(
                DATA / "gbm_focus" / "gbm_rule_model.zip")
        elif which == "ppo":
            _MODELS_CACHE[which] = PPO.load(FROZEN_PPO)
        else:
            raise ValueError(f"unknown signal {which!r}")
    return _MODELS_CACHE[which]
