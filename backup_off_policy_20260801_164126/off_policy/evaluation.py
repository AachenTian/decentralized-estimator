from __future__ import annotations

from typing import Any

from sender_marl.envs.base import MultiAgentEnvAdapter
from sender_marl.off_policy.independent import (
    make_independent_sac_actor_apply,
)
from sender_marl.on_policy.evaluation import make_policy_evaluator


def make_independent_sac_evaluator(
    adapter: MultiAgentEnvAdapter,
    actor: Any,
    *,
    num_envs: int,
    horizon: int,
    jit: bool = True,
):
    """Reuse the deterministic oracle-state evaluator for independent SAC."""

    independent_apply = make_independent_sac_actor_apply(
        actor.apply,
        adapter.spec.num_agents,
    )
    return make_policy_evaluator(
        adapter,
        independent_apply,
        num_envs=num_envs,
        horizon=horizon,
        jit=jit,
    )
