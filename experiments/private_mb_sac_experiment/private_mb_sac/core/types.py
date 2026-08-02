"""Shared immutable data containers used by collection, replay, and models."""

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
    """Time-major complete joint transitions."""
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
    episode_steps: Array | None = None
    next_episode_steps: Array | None = None


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
    episode_steps: Array | None = None
    next_episode_steps: Array | None = None


class DynamicsPrediction(NamedTuple):
    member_delta_means: Array
    member_delta_variances: Array
    mean_delta: Array
    aleatoric_variance: Array
    epistemic_variance: Array
    total_variance: Array


class ModelRolloutOutput(NamedTuple):
    """Time-major synthetic transitions plus validity/termination diagnostics."""
    batch: RealRolloutBatch
    valid_mask: Array                 # (H, B)
    trajectory_lengths: Array         # (B,)
    termination_reasons: Array        # (B,), integer code
    epistemic_scores: Array           # (H, B)
    landmark_drift: Array             # (H, B)



class SACBatch(NamedTuple):
    """One focal learner's private SAC minibatch.

    `rewards` and `dones` are already sliced for the focal owner, while
    observations/actions/states retain the complete joint transition.
    """

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    model_states: Array
    next_model_states: Array
