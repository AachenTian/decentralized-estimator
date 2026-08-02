from __future__ import annotations

from typing import Any

import jax
from flax import struct


Array = jax.Array


@struct.dataclass
class SACBatch:
    """One joint transition batch sampled for an independent SAC learner.

    Each learner samples independently, but the critic receives the complete
    Markov state and joint action. Shapes:

        observations:       (B, N, O)
        actions:            (B, N, A)
        rewards:            (B,)
        next_observations:  (B, N, O)
        dones:              (B,)
        model_states:       (B, M)
        next_model_states:  (B, M)

    The focal agent is selected by the SAC update closure. Its actor consumes
    observations[:, agent_id], while its private twin critics consume
    model_states and all N actions.
    """

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    model_states: Array
    next_model_states: Array


@struct.dataclass
class RealRolloutBatch:
    """Time-major transitions collected from the real joint environment.

    Shapes:
        observations:          (T, E, N, O)
        actions:               (T, E, N, A)
        rewards:               (T, E, N)
        next_observations:     (T, E, N, O)
        dones:                 (T, E, N)
        model_states:          (T, E, M)
        next_model_states:     (T, E, M)
        episode_dones:         (T, E)
        pair_collision_rates:  (T, E)
    """

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    model_states: Array
    next_model_states: Array
    episode_dones: Array
    pair_collision_rates: Array


@struct.dataclass
class RealRolloutOutput:
    env_state: Any
    features: Any
    batch: RealRolloutBatch
