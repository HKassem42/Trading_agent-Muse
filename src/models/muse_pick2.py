"""Muse's Pick #2 challenger -- LAB ONLY (Kass's order, 2026-10-27).

Kass: "You go ahead and try to beat your own champion first muse,
take another stab at it and call it Muse model #2 or whatever u
called the first one but #2."

Pick #1 (runs 69/70) split: its stacker WON and was crowned Lab
Stacker Champion #3 (run 69, by Kass's approval); its PPO lost --
the post-mortem's named confound: reward shaping penalized
drawdown increases inside 24-candle training episodes while the
duel's damage compounds across a 264-candle window the reward
never saw. Pick #2 is the fair retest plus one new-eyes lever,
pre-registered in refs/muse-pick-2-challenger-2026-10-27.md:

- Eyes -- "Muse's Pick #2 Stacker": Pick #1's exact recipe
  (scaler -> SMOTE -> RF + XGBoost + LightGBM + ExtraTrees -> LR,
  imported from models.muse_pick unchanged) on Champion #3's
  scored-102 frame PLUS four new causal features from the
  archive-mining candidates (src/features/pick2_features.py):
  rolling Hurst exponent (100 bars), rolling return skew /
  kurtosis / lag-1 autocorrelation (100 bars), each percentile-
  scored through the project's trailing-4,320-bar machinery.
  98 feature columns. Graded against Champion #3 (run 69)
  re-scored on identical holdout timestamps: strict accuracy
  win, precision and up-call activity on the record -- a win
  with degenerate anatomy (<= 7 up-calls or precision <= 0.05)
  counts as a LOSS (the run 63 rule, as pre-registered).

- Brain -- "Muse's Pick #2 PPO": Pick #1's shaped reward
  (-0.5 x step drawdown increase, ShapedTradingEnv unchanged)
  trained under the scored-eyes protocol (fresh brain, 60,000
  steps, seed 42, ent_coef 0.05) on the Pick #2 stacker's
  cross-fitted votes -- with ONE design change answering Pick
  #1's confound: training episodes span 264 candles (11 days x
  24), the duel-window length, so the shaping's running peak
  and drawdown live at the horizon the gate grades. Everything
  else -- the dense reward, the terminal +1/-1 (now on the
  264-candle episode's P&L sign), the activity penalty, the
  duel itself -- is the production standard, unchanged.

The episode length lives in DuelLengthShapedTradingEnv, a
ShapedTradingEnv subclass used only by the study builder (which
swaps it into ppo_train's env slot for the study's train call
only). The production env, the nightly retrain, and both seats
are never touched. A study win installs nothing (the promotion
lock, 2026-10-25): it is recorded pending_approval for Kass.
"""

from __future__ import annotations

from models.muse_pick import ShapedTradingEnv

FAMILY = "muse_pick2"
EPISODE_DAYS = 11
STEPS_PER_DAY = 24
EPISODE_CANDLES = EPISODE_DAYS * STEPS_PER_DAY  # 264


class DuelLengthShapedTradingEnv(ShapedTradingEnv):
    """ShapedTradingEnv with duel-length training episodes:
    264 candles (11 x 24h) instead of the production 24. Only the
    episode length changes -- state, fills, broker, shaping,
    terminal verdict and activity penalty are inherited
    unchanged, so Pick #1's shaping gets its first fair test at
    the horizon it is graded on."""

    def __init__(self, df, stacker_p_up, broker, **kwargs):
        kwargs["episode_days"] = EPISODE_DAYS
        kwargs["steps_per_day"] = STEPS_PER_DAY
        super().__init__(df, stacker_p_up, broker, **kwargs)
        assert self.episode_len == EPISODE_CANDLES
