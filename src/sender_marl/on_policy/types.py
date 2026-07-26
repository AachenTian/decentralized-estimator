from __future__ import annotations

from typing import Any, Mapping

import jax
from flax import struct

from sender_marl.core.types import EnvFeatures


Array = jax.Array


@struct.dataclass
class RolloutBatch:
    """Time-major real-environment rollout.

    Expected shapes after `jax.lax.scan`:

    actor_obs:       (T, E, N, O)
    critic_obs:      (T, E, N, C)
    actions:         (T, E, N, A)
    pre_tanh_actions:(T, E, N, A)
    log_probs:       (T, E, N)
    values:          (T, E, N)
    rewards:         (T, E, N)
    dones:           (T, E, N)
    episode_dones:   (T, E)
    local_states:    (T, E, N, X)
    next_local_states:(T, E, N, X)
    alive:           (T, E, N)
    metrics:         static-key mapping with arrays beginning in (T, E, ...)
    """

    actor_obs: Array
    critic_obs: Array
    actions: Array
    pre_tanh_actions: Array
    log_probs: Array
    values: Array
    rewards: Array
    dones: Array
    episode_dones: Array
    local_states: Array
    next_local_states: Array
    alive: Array
    metrics: Mapping[str, Array]


@struct.dataclass
class CollectorCarry:
    env_state: Any
    features: EnvFeatures
    common_state_source_state: Any
    key: Array


@struct.dataclass
class RolloutOutput:
    batch: RolloutBatch
    final_env_state: Any
    final_features: EnvFeatures
    final_common_state_source_state: Any
    final_actor_obs: Array
    final_critic_obs: Array
    final_values: Array
