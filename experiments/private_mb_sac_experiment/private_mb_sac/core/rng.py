"""Forced seed-40 environment keys and owner-private policy keys.

Environment behavior in this experiment is intentionally diagnostic:

- every reset receives exactly ``jax.random.PRNGKey(40)``;
- every vectorized environment therefore begins from the same layout;
- every owner receives the same environment key schedule;
- every synchronization round repeats the same environment key schedule;
- step keys are deterministic folds of seed 40 by time index only.

Policy exploration, replay sampling, dynamics, SAC, and optimizer RNG streams
remain separate and are not forced to 40.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from private_mb_sac.core.types import (
    EnvironmentKeySchedule,
    OwnerPolicyKeySchedule,
)


Array = jax.Array
FORCED_ENVIRONMENT_SEED = 40


def _fold_in_many(key: Array, values: Sequence[int]) -> Array:
    result = key
    for value in values:
        result = jax.random.fold_in(result, int(value))
    return result


def forced_environment_key() -> Array:
    return jax.random.PRNGKey(FORCED_ENVIRONMENT_SEED)


def _repeat_key(key: Array, *shape: int) -> Array:
    if any(int(value) < 1 for value in shape):
        raise ValueError("Repeated-key dimensions must be positive.")
    return jnp.broadcast_to(
        key,
        tuple(int(value) for value in shape) + (2,),
    )


def make_initial_reset_keys(
    environment_seed: int,
    *,
    num_envs: int,
) -> Array:
    """Return identical PRNGKey(40) resets for every vector environment."""
    del environment_seed
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    return _repeat_key(forced_environment_key(), num_envs)


def make_environment_key_schedule(
    environment_seed: int,
    *,
    synchronization_round: int,
    rollout_length: int,
    num_envs: int,
) -> EnvironmentKeySchedule:
    """Repeat the same seed-40 environment schedule in every round."""
    del environment_seed
    if synchronization_round < 0:
        raise ValueError("synchronization_round must be non-negative.")
    if rollout_length < 1 or num_envs < 1:
        raise ValueError("rollout_length and num_envs must be positive.")

    root = forced_environment_key()
    step_time_keys = jax.vmap(
        lambda time_index: jax.random.fold_in(root, time_index + 1)
    )(jnp.arange(rollout_length, dtype=jnp.uint32))
    step_keys = jnp.broadcast_to(
        step_time_keys[:, None, :],
        (rollout_length, num_envs, 2),
    )
    reset_keys = _repeat_key(
        root,
        rollout_length,
        num_envs,
    )
    return EnvironmentKeySchedule(
        step_keys=step_keys,
        reset_keys=reset_keys,
    )


def make_owner_policy_key_schedule(
    policy_seed: int,
    *,
    synchronization_round: int,
    owner_id: int,
    rollout_length: int,
) -> OwnerPolicyKeySchedule:
    """Keep stochastic actor exploration distinct across owners and rounds."""
    if synchronization_round < 0 or owner_id < 0:
        raise ValueError("round and owner_id must be non-negative.")
    if rollout_length < 1:
        raise ValueError("rollout_length must be positive.")

    owner_root = _fold_in_many(
        jax.random.PRNGKey(policy_seed),
        (0x97531, synchronization_round, owner_id),
    )
    return OwnerPolicyKeySchedule(
        action_keys=jax.random.split(
            jax.random.fold_in(owner_root, 0),
            rollout_length,
        ),
        random_action_keys=jax.random.split(
            jax.random.fold_in(owner_root, 1),
            rollout_length,
        ),
    )
