"""Deterministic common environment keys and owner-private policy keys."""

from __future__ import annotations

from collections.abc import Sequence

import jax

from private_mb_sac.core.types import (
    EnvironmentKeySchedule,
    OwnerPolicyKeySchedule,
)


Array = jax.Array


def _fold_in_many(key: Array, values: Sequence[int]) -> Array:
    result = key
    for value in values:
        result = jax.random.fold_in(result, int(value))
    return result


def make_initial_reset_keys(environment_seed: int, *, num_envs: int) -> Array:
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    root = _fold_in_many(
        jax.random.PRNGKey(environment_seed),
        (0x13579, 0),
    )
    return jax.random.split(root, num_envs)


def make_environment_key_schedule(
    environment_seed: int,
    *,
    synchronization_round: int,
    rollout_length: int,
    num_envs: int,
) -> EnvironmentKeySchedule:
    if synchronization_round < 0:
        raise ValueError("synchronization_round must be non-negative.")
    if rollout_length < 1 or num_envs < 1:
        raise ValueError("rollout_length and num_envs must be positive.")

    round_root = _fold_in_many(
        jax.random.PRNGKey(environment_seed),
        (0x24680, synchronization_round),
    )
    step_root = jax.random.fold_in(round_root, 0)
    reset_root = jax.random.fold_in(round_root, 1)
    step_time_keys = jax.random.split(step_root, rollout_length)
    reset_time_keys = jax.random.split(reset_root, rollout_length)
    return EnvironmentKeySchedule(
        step_keys=jax.vmap(lambda key: jax.random.split(key, num_envs))(
            step_time_keys
        ),
        reset_keys=jax.vmap(lambda key: jax.random.split(key, num_envs))(
            reset_time_keys
        ),
    )


def make_owner_policy_key_schedule(
    policy_seed: int,
    *,
    synchronization_round: int,
    owner_id: int,
    rollout_length: int,
) -> OwnerPolicyKeySchedule:
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
            jax.random.fold_in(owner_root, 0), rollout_length
        ),
        random_action_keys=jax.random.split(
            jax.random.fold_in(owner_root, 1), rollout_length
        ),
    )
