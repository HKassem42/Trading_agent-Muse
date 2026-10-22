"""SB3 policy class for the HA-sequence threshold rule.

Defined at MODULE level on purpose: SB3 serializes the policy class by
reference (module + qualname), so a function-local class cannot be
reloaded by PPO.load in a fresh process (verified empirically
2026-10-22 -- AttributeError on a <locals> qualname). Any process with
the project src/ on sys.path can import models.ha_rule_policy and load
a saved rule model -- including the duel scorer's evaluate_ppo.

The distribution is computed directly from obs[21] (centered P(up)) and
obs[23] (signed position fraction) via ha_policy.rule_actions_np; the
underlying network weights exist only to satisfy the SB3 save/load
contract and are never read.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from models.ha_policy import P_IDX, POS_IDX, rule_actions_np

_BIG = 30.0


class HaRulePolicy(ActorCriticPolicy):
    def get_distribution(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        want = torch.from_numpy(rule_actions_np(
            obs[:, P_IDX].detach().cpu().numpy(),
            obs[:, POS_IDX].detach().cpu().numpy())).to(obs.device)
        logits = torch.full((obs.shape[0], 4), -_BIG,
                            dtype=torch.float32, device=obs.device)
        logits.scatter_(1, want.unsqueeze(1), _BIG)
        return CategoricalDistribution(4).proba_distribution(
            action_logits=logits)
