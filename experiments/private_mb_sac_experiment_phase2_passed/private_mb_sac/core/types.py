"""Shared immutable data containers used by collection and replay."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax


Array = jax.Array


class ObservationNormalizer(NamedTuple):
    """Frozen actor-observation normalizer shared by all execution owners."""

    mean: Array
    std: Array
    clip: Array


class DynamicsNormalizer(NamedTuple):
    """One owner-private frozen dynamics standardizer."""

    state_mean: Array
    state_std: Array
    action_mean: Array
    action_std: Array
    delta_mean: Array
    delta_std: Array
    clip: Array
    sample_count: Array


class ActorSnapshotBank(NamedTuple):
    """Immutable tuple of one actor parameter tree per agent."""

    params_by_agent: tuple[Any, ...]
    synchronization_round: int


class EnvironmentKeySchedule(NamedTuple):
    step_keys: Array       # (T, E, 2)
    reset_keys: Array      # (T, E, 2)


class OwnerPolicyKeySchedule(NamedTuple):
    action_keys: Array     # (T, 2)
    random_action_keys: Array  # (T, 2)


class RealRolloutBatch(NamedTuple):
    """Time-major complete joint transitions.

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
      min_pair_distances:    (T, E)
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
    min_pair_distances: Array


class RealRolloutOutput(NamedTuple):
    env_state: Any
    features: Any
    batch: RealRolloutBatch


class ReplayBatch(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    model_states: Array
    next_model_states: Array
    pair_collision_rates: Array
    min_pair_distances: Array
