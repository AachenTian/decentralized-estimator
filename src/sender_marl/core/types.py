from __future__ import annotations

from typing import Any, Mapping

import jax
from flax import struct


Array = jax.Array


@struct.dataclass
class EnvFeatures:
    """Environment-independent semantic features.

    Leading batch dimensions are allowed. The final dimensions follow:

    private_obs:    (..., num_agents, private_obs_dim)
    public_context: (..., num_agents, public_context_dim)
    local_states:   (..., num_agents, local_state_dim)
    critic_state:   (..., critic_obs_dim)
    model_state:    (..., model_state_dim)
    alive:          (..., num_agents)
    """

    private_obs: Array
    public_context: Array
    local_states: Array
    critic_state: Array
    model_state: Array
    alive: Array


@struct.dataclass
class EnvStep:
    """Standard step output consumed by algorithm-independent collectors."""

    env_state: Any
    features: EnvFeatures
    rewards: Array
    dones: Array
    episode_done: Array
    metrics: Mapping[str, Array]
