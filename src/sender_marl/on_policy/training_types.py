from __future__ import annotations

from typing import Any, Mapping

import jax
from flax import struct


Array = jax.Array


@struct.dataclass
class PPOTrainingBatch:
    """Flattened samples used by one PPO optimization update."""

    actor_obs: Array
    critic_obs: Array
    pre_tanh_actions: Array
    old_log_probs: Array
    old_values: Array
    advantages: Array
    returns: Array
    alive: Array


@struct.dataclass
class PPOUpdateOutput:
    actor_state: Any
    value_state: Any
    metrics: Mapping[str, Array]
    key: Array
