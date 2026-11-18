"""Exit & transition study (2026-10-22, Kass's request).

Every experiment so far varied the SIGNAL and froze the exits at the
v4 recipe (initial stop 2xATR% clipped [2.5%, 8%], trailing arms at
+3% / trails 1.5xATR% clipped [2%, 6%], NO take-profit, flips only
through flat across separate steps). This module holds the signal
fixed -- the GBM Lag F rule policy (training.db run 26) -- and varies
ONLY the exit + transition machinery:

- ExitConfig: stop ATR multiple, take-profit (fixed % or R-multiple,
  both new), trailing mode (current / off / tight). Defaults are
  EXACTLY the production recipe: a default-config StudyBroker is the
  production broker with a TP switch that is off.
- Transitions (how a position closes/opens when the signal flips):
  T0 through-flat (production), T1 one-motion reversal (close +
  opposite open in the same step), T2 staged (close fully now; the
  open, if the signal persists, happens on a later step -- which is
  production open behavior once flat). Transitions live in the
  broker's buy/short_open seam: production already closes a long
  fully on a flip signal (short_open while long sells 100%) but only
  covers a short one tranche per buy, so the rules differ exactly
  where production is asymmetric. A flip is an ACTIVE opposite
  signal (the env writes the decision candle's conviction direction
  into broker.signal_dir each step); a band BUY that merely flattens
  a short is an exit, never a flip, under every rule.

Take-profit semantics (documented choice): the level derives from
the position's CURRENT weighted entry and entry ATR, exactly like
the stop level derives in PaperBroker.stop_level -- fixed % is
entry x (1 +/- tp), R form is entry +/- R x (initial-stop distance).
Triggers are symmetric with stops: long TP watches the candle HIGH,
short TP the candle LOW (wick touch counts); the fill is the level
itself when the close held beyond it on the safe side... precisely:
fill = level when the close did NOT cross the level, else the close
(the market ran through the target and kept going -- the duel's
next-candle accounting takes the close, mirroring the stop's
gap-through rule). A candle whose wicks span BOTH the stop and the
TP fills the STOP (conservative intra-candle ordering; the path
inside one hourly candle is unknowable).

Lab only: nothing here writes production artifacts. The duel driver
mirrors training.nightly.evaluate_ppo's loop line for line (same
episodes, fills, scoring) with the study broker/env swapped in;
tests/test_exit_study.py pins the default config to run 26's
recorded numbers as the identity proof.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from environment.trading_env import TradingEnv
from models.gbm_policy import THRESH, conviction_np
from trading.broker import PaperBroker

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "data"

STARTING_BALANCE = 100_000.0
EPISODES = 10


# ----------------------------------------------------------------- config
_TRAIL_PRESETS = {
    #            (activate_pct, atr_mult, min_pct, max_pct)
    "current": (0.02, 1.5, 0.02, 0.06),   # the production recipe (activation 2% since Kass 2026-11-02; the Oct 2026 exit study ran it at 3%)
    "tight": (0.02, 1.0, 0.02, 0.06),     # arms earlier, trails closer
    "off": (float("inf"), 1.5, 0.02, 0.06),  # never arms
}


@dataclass
class ExitConfig:
    stop_mult: float = 2.0
    stop_min_pct: float = 0.025
    stop_max_pct: float = 0.08
    trail: str = "current"               # current | tight | off
    tp_mode: str | None = None           # None | "pct" | "R"
    tp_value: float = 0.0                # 0.04 / 0.08, or 2.0 for 2R
    transition: str = "T0"               # T0 | T1 | T2
    # resolved trailing numbers (set from `trail` in __post_init__)
    trail_activate_pct: float = 0.02  # placeholder — __post_init__ resolves it from _TRAIL_PRESETS[trail]
    trail_atr_mult: float = 1.5
    trail_min_pct: float = 0.02
    trail_max_pct: float = 0.06

    def __post_init__(self):
        assert self.trail in _TRAIL_PRESETS, f"bad trail mode {self.trail}"
        assert self.transition in ("T0", "T1", "T2")
        assert self.tp_mode in (None, "pct", "R")
        (self.trail_activate_pct, self.trail_atr_mult,
         self.trail_min_pct, self.trail_max_pct) = _TRAIL_PRESETS[self.trail]

    def label(self) -> str:
        tp = "TPoff" if self.tp_mode is None else (
            f"TP{self.tp_value * 100:.0f}%" if self.tp_mode == "pct"
            else f"TP{self.tp_value:g}R")
        return (f"stop{self.stop_mult:g}·{tp}·trail-{self.trail}"
                f"·{self.transition}")

    def short_label(self) -> str:
        """Compact label for the dashboards' model index (the feed's
        8KB ceiling keeps these terse: 'T1 s2 noTP tr')."""
        tp = "noTP" if self.tp_mode is None else (
            f"TP{self.tp_value * 100:.0f}%" if self.tp_mode == "pct"
            else f"TP{self.tp_value:g}R")
        tr = {"current": "tr", "tight": "tr-tight", "off": "tr-off"}[
            self.trail]
        return f"{self.transition} s{self.stop_mult:g} {tp} {tr}"


# ----------------------------------------------------------------- broker
class StudyBroker(PaperBroker):
    """PaperBroker with config-driven exit rules + flip transitions.

    Production constants are shadowed per-instance from the config;
    with a default ExitConfig every shadow equals the class constant
    and every override below passes straight through, so behavior is
    production behavior (pinned by the identity test)."""

    def __init__(self, db_path, starting_balance=STARTING_BALANCE,
                 max_position_pct=0.25, stop_loss_pct=0.05,
                 max_exposure_pct=0.75, cfg: ExitConfig | None = None):
        super().__init__(db_path, starting_balance=starting_balance,
                         max_position_pct=max_position_pct,
                         stop_loss_pct=stop_loss_pct,
                         max_exposure_pct=max_exposure_pct)
        self._cfg = cfg or ExitConfig()
        self.STOP_ATR_MULT = self._cfg.stop_mult
        self.STOP_MIN_PCT = self._cfg.stop_min_pct
        self.STOP_MAX_PCT = self._cfg.stop_max_pct
        self.TRAIL_ACTIVATE_PCT = self._cfg.trail_activate_pct
        self.TRAIL_ATR_MULT = self._cfg.trail_atr_mult
        self.TRAIL_MIN_PCT = self._cfg.trail_min_pct
        self.TRAIL_MAX_PCT = self._cfg.trail_max_pct
        # Direction of the ACTIVE signal on the decision candle
        # (+1 long / -1 short / 0 band), written by StudyEnv each step.
        self.signal_dir = 0

    # -- take-profit --
    def _base_stop_pct(self) -> float:
        """The initial-stop distance as a fraction of entry, derived
        exactly as PaperBroker.stop_level derives it."""
        atr = self.entry_atr_pct
        if atr:
            return min(max(self.STOP_ATR_MULT * atr,
                           self.STOP_MIN_PCT), self.STOP_MAX_PCT)
        return self.stop_loss_pct

    def tp_level(self) -> float | None:
        if self.btc == 0 or not self.entry_price or self._cfg.tp_mode is None:
            return None
        if self._cfg.tp_mode == "pct":
            dist = self._cfg.tp_value
        else:  # "R": multiple of the initial-stop distance
            dist = self._cfg.tp_value * self._base_stop_pct()
        if self.btc > 0:
            return self.entry_price * (1 + dist)
        return self.entry_price * (1 - dist)

    def check_stop(self, price, ts=None, low=None, high=None,
                   context=None):
        # Stop (incl. trailing) first: a candle spanning both levels
        # fills the stop -- conservative ordering, documented above.
        fill = super().check_stop(price, ts=ts, low=low, high=high,
                                  context=context)
        if fill is not None or self.btc == 0:
            return fill
        lvl = self.tp_level()
        if lvl is None:
            return None
        if self.btc > 0:
            trigger = high if high is not None else price
            if trigger >= lvl:
                fill_price = lvl if price <= lvl else price
                return self.sell(fill_price, pct=1.0,
                                 reason="take_profit", ts=ts,
                                 context=context)
            return None
        trigger = low if low is not None else price
        if trigger <= lvl:
            fill_price = lvl if price >= lvl else price
            return self.cover(fill_price, pct=1.0, reason="take_profit",
                              ts=ts, context=context)
        return None

    # -- transitions --
    def buy(self, price, reason="signal", ts=None, atr_pct=None,
            context=None):
        if (self.btc < 0 and self.signal_dir > 0
                and self._cfg.transition in ("T1", "T2")):
            # Active flip signal against the short: close it ALL now
            # (production would cover one tranche per step).
            cover_fill = self.cover(price, pct=1.0, reason="cover",
                                    ts=ts, context=context)
            if self._cfg.transition == "T1":
                # One motion: open the long tranche in the same step.
                return super().buy(price, reason=reason, ts=ts,
                                   atr_pct=atr_pct, context=context)
            return cover_fill  # T2 staged: the open is a later step
        return super().buy(price, reason=reason, ts=ts, atr_pct=atr_pct,
                           context=context)

    def short_open(self, price, reason="signal", ts=None, atr_pct=None,
                   context=None):
        if (self.btc > 0 and self.signal_dir < 0
                and self._cfg.transition == "T1"):
            # Production closes the long fully here but opens nothing;
            # T1 completes the reversal in the same step.
            self.sell(price, pct=1.0, reason=reason, ts=ts,
                      context=context)
            return super().short_open(price, reason=reason, ts=ts,
                                      atr_pct=atr_pct, context=context)
        # T0: production (close only). T2: also production here -- the
        # close IS the full close, and the open waits for a later step.
        return super().short_open(price, reason=reason, ts=ts,
                                  atr_pct=atr_pct, context=context)


# -------------------------------------------------------------------- env
class StudyEnv(TradingEnv):
    """TradingEnv that tells the broker the decision candle's active
    signal direction before each step (from the same conviction
    channel the rule policy reads), so the broker can tell a genuine
    flip from a band flatten. Nothing else changes."""

    def reset(self, *, seed=None, options=None):
        self.broker.signal_dir = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        i = self._cursor
        if i < len(self.stacker_p_up):
            c = 2.0 * float(self.stacker_p_up[i]) - 1.0
            self.broker.signal_dir = (1 if c >= THRESH
                                      else -1 if c <= -THRESH else 0)
        return super().step(action)


# ------------------------------------------------------------------ signal
_RULE_MODEL = None
_ROBUST_CACHE: dict = {}


def _rule_model():
    global _RULE_MODEL
    if _RULE_MODEL is None:
        from stable_baselines3 import PPO
        _RULE_MODEL = PPO.load(DATA / "gbm_lag" / "gbm_lag_rule_model.zip")
    return _RULE_MODEL


def _windows_from(sup: pd.DataFrame, start: int) -> list[pd.DataFrame]:
    return [sup.iloc[start + k * 264: start + (k + 1) * 264]
            for k in range(3)]


def _conv_for_rows(prob_by_ts: dict, frame_index) -> np.ndarray:
    conv = np.zeros(len(frame_index))
    for i, ts in enumerate(frame_index):
        p = prob_by_ts.get(ts)
        if p is not None:
            conv[i] = conviction_np(p[0:1], p[2:3])[0]
    return conv


def load_lagf_signal(which: str):
    """(windows, p_ups) for GBM Lag F (run 26), exactly as the lag
    build graded them. `which` = "standard" (last 792h, cached final-
    model probabilities) or "robustness" (the PRECEDING 792h; the
    probabilities come from the same arm-F recipe fit ONLY on rows
    before that region, so the signal there is out-of-sample too --
    the final duel model trained on rows that include this region,
    so its cached probabilities would be in-sample here)."""
    from models import gbm_focus as gf
    from models import gbm_lag as gl

    sup = pd.read_pickle(DATA / "ha_seq" / "sup.pkl")
    n = len(sup)
    if which == "standard":
        windows = _windows_from(sup, n - 792)
        z = np.load(DATA / "gbm_lag" / "probs_F.npz")
        prob_by_ts = {pd.Timestamp(t): p for t, p in zip(z["ts"], z["proba"])}
        p_ups = [( _conv_for_rows(prob_by_ts, w.index) + 1.0) / 2.0
                 for w in windows]
        return windows, p_ups
    if which == "robustness":
        if "p_ups" in _ROBUST_CACHE:
            return _ROBUST_CACHE["windows"], _ROBUST_CACHE["p_ups"]
        windows = _windows_from(sup, n - 1584)
        region = sup.index[n - 1584: n - 792]
        matrix = gf.load_matrix(
            DATA / "timeseries_matrix" / "btc_usd_1h_matrix.csv")
        frame = gl.drop_warmup(gl.arm_frame(matrix, "F"))
        start_ts = region[0]
        pre = frame.loc[frame.index < start_ts]
        y_pre = matrix.loc[pre.index, gf.TARGET_COL].to_numpy()
        model, _audit = gf.fit_arm_final(
            pre.to_numpy(dtype=float), y_pre, gl.ARM_TREATMENT["F"])
        rows = frame.loc[(frame.index >= region[0])
                          & (frame.index <= region[-1])]
        proba = model.predict_proba(rows.to_numpy(dtype=float))
        prob_by_ts = {ts: p for ts, p in zip(rows.index, proba)}
        p_ups = [(_conv_for_rows(prob_by_ts, w.index) + 1.0) / 2.0
                 for w in windows]
        _ROBUST_CACHE.update(windows=windows, p_ups=p_ups)
        return windows, p_ups
    raise ValueError(f"unknown window set {which!r}")


# ------------------------------------------------------------------ driver
def _read_new_fills(db_path: Path, offset: int) -> tuple[list[dict], int]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT side, price, qty, fee, cash_after, btc_after, reason"
            " FROM trades ORDER BY id LIMIT -1 OFFSET ?", (offset,)
        ).fetchall()
    fills = [{"side": s, "price": p, "qty": q, "fee": f, "cash_after": ca,
              "btc_after": ba, "reason": r}
             for s, p, q, f, ca, ba, r in rows]
    return fills, offset + len(fills)


def segment_trips(fills: list[dict], start_cash: float) -> list[dict]:
    """Split one episode's fills into flat->flat segments (trips).
    A segment's P&L is its closing fill's cash minus the cash before
    its opening fill -- exact, fees included. A segment still open at
    the episode's end is returned with closed=False and pnl=None."""
    segs, cur, prev_cash = [], None, start_cash
    for f in fills:
        if cur is None and abs(f["btc_after"]) > 1e-9:
            cur = {"side": 1 if f["btc_after"] > 0 else -1,
                   "start_cash": prev_cash, "fills": [f]}
        elif cur is not None:
            cur["fills"].append(f)
        if cur is not None and abs(f["btc_after"]) <= 1e-9:
            cur["pnl"] = f["cash_after"] - cur["start_cash"]
            cur["exit_reason"] = f["reason"]
            cur["closed"] = True
            segs.append(cur)
            cur = None
        prev_cash = f["cash_after"]
    if cur is not None:
        cur["pnl"] = None
        cur["exit_reason"] = None
        cur["closed"] = False
        segs.append(cur)
    return segs


_EXIT_BUCKETS = ("stop_loss", "trailing_stop", "take_profit", "signal")


def _bucket(reason: str) -> str:
    return reason if reason in _EXIT_BUCKETS[:3] else "signal"


def evaluate_windows_study(windows, p_ups, cfg: ExitConfig,
                           model=None) -> dict:
    """evaluate_windows (nightly) with study exits/transitions: same
    10x24 episodes, next-open fills, scoring and benchmark. Adds
    per-cell trip statistics segmented from the ledger."""
    model = model or _rule_model()
    per = []
    all_closed: list[dict] = []
    for w_i, (w_sup, w_pup) in enumerate(zip(windows, p_ups)):
        tmp = Path("/tmp") / f"tam_exit_{os.getpid()}_{w_i}.db"
        if tmp.exists():
            tmp.unlink()
        try:
            broker = StudyBroker(tmp, starting_balance=STARTING_BALANCE,
                                 max_position_pct=0.25, stop_loss_pct=0.05,
                                 cfg=cfg)
            env = StudyEnv(w_sup, w_pup, broker, fill_mode="next_open")
            rewards, pnls, trips_env, ep_dds = [], [], [], []
            action_counts = {"hold": 0, "buy": 0, "sell": 0, "short": 0}
            offset = 0
            pnl_long = pnl_short = 0.0
            open_left = 0
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
                ep_pnl = env._prev_eq - start_eq
                rewards.append(ep_r)
                pnls.append(ep_pnl)
                trips_env.append(env._round_trips)
                c = np.array(curve)
                ep_dds.append(float(np.max(np.maximum.accumulate(c) - c)))
                fills, offset = _read_new_fills(tmp, offset)
                segs = segment_trips(fills, STARTING_BALANCE)
                closed_pnl = 0.0
                open_seg = None
                for s in segs:
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
                    # The episode's mark-to-market remainder belongs to
                    # the still-open segment's side, so the long/short
                    # split sums exactly to the window P&L.
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
            })
        finally:
            tmp.unlink(missing_ok=True)
    wins = [s for s in all_closed if s["pnl"] > 0]
    losses = [s for s in all_closed if s["pnl"] <= 0]
    exit_counts = {b: 0 for b in _EXIT_BUCKETS}
    for s in all_closed:
        exit_counts[_bucket(s["exit_reason"])] += 1
    return {
        "windows": per,
        "agg_score": float(np.mean([w["score"] for w in per])),
        "total_pnl": float(np.sum([w["total_pnl"] for w in per])),
        "bench_pnl": float(np.sum([w["bench_pnl"] for w in per])),
        "max_drawdown": float(max(w["max_drawdown"] for w in per)),
        "total_round_trips": int(sum(w["round_trips_env"] for w in per)),
        "closed_trips": len(all_closed),
        "wins": len(wins), "losses": len(losses),
        "avg_win": float(np.mean([s["pnl"] for s in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([s["pnl"] for s in losses])) if losses else 0.0,
        "exit_counts": exit_counts,
        "pnl_long": float(sum(w["pnl_long"] for w in per)),
        "pnl_short": float(sum(w["pnl_short"] for w in per)),
        "open_left": int(sum(w["open_left"] for w in per)),
        "action_mix": per[-1]["action_mix"],
    }
