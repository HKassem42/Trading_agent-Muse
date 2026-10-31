"""GBM focus-list policy wrapper: turns a 3-class LightGBM probability
pair (P(BUY), P(SHORT)) into duel actions through the SAME observation
contract the PPO brains and the HA-sequence rule use.

Two probabilities cannot ride the env's single vote channel (obs[21]),
so the pair is encoded OFFLINE as a signed conviction:

  c = +P(BUY)   if P(BUY)  >= 0.40 and P(BUY)  >= P(SHORT)
  c = -P(SHORT) if P(SHORT) >= 0.40 and P(SHORT) >  P(BUY)
  c = 0         otherwise

and passed to the duel scorer as p_up = (c + 1) / 2, so obs[21] == c
exactly (the env centers the channel: 2*p_up - 1). The encoding is
lossless FOR THIS RULE: c >= +0.40 fires exactly when the LONG branch
of the pre-registered rule fires, c <= -0.40 exactly when the SHORT
branch fires, and every other case lands in the flat band -- where the
action mapping mirrors ha_policy.decide (signal gone or flipped ->
flatten: SELL when long, BUY/cover when short, HOLD when flat).

PRE-REGISTERED thresholds (fixed in the experiment brief 2026-10-22,
before any grading): LONG at P(BUY) >= 0.40, SHORT at P(SHORT) >= 0.40,
ties at/above the bar go LONG (the rule lists LONG first). These are
probability bars on the classifier, NOT the centered-channel values;
on the channel the bars are +0.40 / -0.40 of the signed conviction.
"""

from __future__ import annotations

import numpy as np

P_IDX = 21    # conviction slot in the 25-dim duel obs (env centers it)
POS_IDX = 23  # signed position value fraction (btc*price/eq)
THRESH = 0.40  # pre-registered probability bar, both directions
_EPS = 1e-6

HOLD, BUY, SELL, SHORT = 0, 1, 2, 3


def conviction(p_buy: float, p_short: float) -> float:
    """Scalar pre-registered rule -> signed conviction (see docstring)."""
    if p_buy >= THRESH and p_buy >= p_short:
        return float(p_buy)
    if p_short >= THRESH and p_short > p_buy:
        return -float(p_short)
    return 0.0


def conviction_np(p_buy: np.ndarray, p_short: np.ndarray) -> np.ndarray:
    """Vectorized conviction() over probability arrays."""
    pb = np.asarray(p_buy, dtype=float)
    ps = np.asarray(p_short, dtype=float)
    c = np.zeros(pb.shape, dtype=float)
    long_fire = (pb >= THRESH) & (pb >= ps)
    short_fire = (ps >= THRESH) & (ps > pb)
    c = np.where(long_fire, pb, c)
    c = np.where(short_fire, -ps, c)
    return c


def decide(c: float, pos_frac: float) -> int:
    """Action for one step, given the conviction channel value c (as it
    appears in obs[21]) and the signed position fraction obs[23].
    Mirrors ha_policy.decide's position-aware exit semantics."""
    if c >= THRESH:
        return BUY
    if c <= -THRESH:
        return SHORT
    if pos_frac > _EPS:
        return SELL
    if pos_frac < -_EPS:
        return BUY
    return HOLD


def rule_actions_np(c: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Vectorized decide() over batches of (conviction, pos_frac)."""
    c = np.asarray(c, dtype=float)
    pos = np.asarray(pos, dtype=float)
    a = np.zeros(c.shape, dtype=np.int64)
    a = np.where(c >= THRESH, BUY, a)
    a = np.where(c <= -THRESH, SHORT, a)
    band = (c > -THRESH) & (c < THRESH)
    a = np.where(band & (pos > _EPS), SELL, a)
    a = np.where(band & (pos < -_EPS), BUY, a)
    return a


def build_rule_model(obs_dim: int = 25):
    """A saveable SB3 PPO whose policy is the conviction rule above.

    Mirror of ha_policy.build_rule_model: the network weights are never
    trained and never read -- the policy's distribution is computed
    directly from obs[P_IDX]/obs[POS_IDX]. The policy class lives at
    module level in models.gbm_rule_policy because SB3 reloads policy
    classes by reference."""
    from gymnasium import spaces
    from stable_baselines3 import PPO

    from models.gbm_rule_policy import GbmRulePolicy

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

    return PPO(GbmRulePolicy, Dummy(), verbose=0, seed=42)
