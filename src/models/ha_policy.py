"""HA-sequence policy wrapper: turns the classifier's P(up) into duel
actions through the SAME observation contract the PPO brains use.

At grading time the classifier's probabilities ride into the standard
duel scorer (training.nightly.evaluate_windows) through its `p_up`
argument, so the env's obs[21] carries 2*p-1 (centered) exactly where
the stacker vote sits for the champion. This policy reads obs[21] and
the signed position fraction obs[23] and emits actions under the
broker's position-aware semantics:

  p >= 0.55 -> BUY   (open/add long; covers a short tranche-by-tranche)
  p <= 0.45 -> SHORT (open/add short; closes a long through flat)
  between   -> flatten: SELL when long, BUY (cover) when short,
               HOLD when already flat

Thresholds are symmetric around 0.5 (dead band 0.45..0.55) so the rule
cannot be accused of a directional tilt. The class is a genuine SB3
ActorCriticPolicy (verified: a saved zip reloads via PPO.load in a
fresh process), so evaluate_windows consumes it through its normal
model_path argument -- identical env, broker, fills, fees, stops and
scoring as every PPO duel, no reimplementation.
"""

from __future__ import annotations

import numpy as np

P_IDX = 21    # centered probability slot in the 25-dim duel obs
POS_IDX = 23  # signed position value fraction (btc*price/eq)
HI_C = 0.10   # centered form of p >= 0.55
LO_C = -0.10  # centered form of p <= 0.45
_EPS = 1e-6

HOLD, BUY, SELL, SHORT = 0, 1, 2, 3


def decide(p_centered: float, pos_frac: float) -> int:
    """Scalar rule (see module docstring)."""
    if p_centered >= HI_C:
        return BUY
    if p_centered <= LO_C:
        return SHORT
    if pos_frac > _EPS:
        return SELL
    if pos_frac < -_EPS:
        return BUY
    return HOLD


def rule_actions_np(p_c: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Vectorized decide() over batches of (p_centered, pos_frac)."""
    p_c = np.asarray(p_c, dtype=float)
    pos = np.asarray(pos, dtype=float)
    a = np.zeros(p_c.shape, dtype=np.int64)
    a = np.where(p_c >= HI_C, BUY, a)
    a = np.where(p_c <= LO_C, SHORT, a)
    band = (p_c > LO_C) & (p_c < HI_C)
    a = np.where(band & (pos > _EPS), SELL, a)
    a = np.where(band & (pos < -_EPS), BUY, a)
    return a


def build_rule_model(obs_dim: int = 25):
    """A saveable SB3 PPO whose policy is the threshold rule above.

    The network weights are never trained and never read: the policy's
    distribution is computed directly from obs[P_IDX]/obs[POS_IDX].
    The policy class lives at module level in models.ha_rule_policy --
    SB3 reloads policy classes by reference, so a local class would
    not survive PPO.load in a fresh process."""
    from gymnasium import spaces
    from stable_baselines3 import PPO

    from models.ha_rule_policy import HaRulePolicy

    import gymnasium as gym

    class Dummy(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(4)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            return (np.zeros(obs_dim, dtype=np.float32),
                    0.0, True, False, {})

    return PPO(HaRulePolicy, Dummy(), verbose=0, seed=42)
