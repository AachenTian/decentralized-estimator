from __future__ import annotations

from typing import Any

import jax
from flax import struct


Array = jax.Array


@struct.dataclass
class JointDynamicsRolloutBatch:
    """Real joint transitions for independent joint-conditioned models.

    Shapes:
        model_states:             (T, E, S)
        joint_actions:            (T, E, N, A)
        local_states:             (T, E, N, D)
        next_local_states:        (T, E, N, D)
        episode_dones:            (T, E)
    """

    model_states: Array
    joint_actions: Array
    local_states: Array
    next_local_states: Array
    episode_dones: Array


@struct.dataclass
class JointDynamicsRolloutOutput:
    env_state: Any
    features: Any
    batch: JointDynamicsRolloutBatch


@struct.dataclass
class JointDynamicsBatch:
    """One batch of complete transitions shared by all local model owners.

    Every model receives the same full state and joint action, but model i only
    predicts the next local state of agent i. Collision labels are evaluation
    metadata and are not network inputs.

    Shapes:
        model_states:             (B, S)
        joint_actions:            (B, N, A)
        local_states:             (B, N, D)
        next_local_states:        (B, N, D)
        agent_collision_flags:    (B, N), bool at current state x_t
    """

    model_states: Array
    joint_actions: Array
    local_states: Array
    next_local_states: Array
    agent_collision_flags: Array
