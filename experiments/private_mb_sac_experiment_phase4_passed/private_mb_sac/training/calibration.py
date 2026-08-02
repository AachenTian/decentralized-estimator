"""Public random-policy calibration for the shared frozen actor normalizer."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from private_mb_sac.agents.snapshots import exchange_actor_snapshots
from private_mb_sac.core.rng import (
    make_environment_key_schedule,
    make_initial_reset_keys,
    make_owner_policy_key_schedule,
)
from private_mb_sac.dynamics.normalization import (
    fit_actor_observation_normalizer,
)
from private_mb_sac.envs.adapter import identity_actor_normalizer


def calibrate_common_actor_normalizer(
    *,
    adapter: Any,
    collector: Any,
    actor_params: tuple[Any, ...],
    environment_seed: int,
    policy_seed: int,
    num_envs: int,
    rollout_length: int,
    rounds: int,
    clip: float,
    minimum_std: float,
):
    """Collect separate public random-policy data and fit shared statistics."""
    if rounds < 1:
        raise ValueError("Calibration rounds must be positive.")

    reset_keys = make_initial_reset_keys(environment_seed, num_envs=num_envs)
    env_state, features = jax.vmap(adapter.reset)(reset_keys)
    identity = identity_actor_normalizer(adapter, clip=clip)
    snapshots = exchange_actor_snapshots(
        actor_params,
        synchronization_round=0,
    )
    chunks: list[np.ndarray] = []

    for round_index in range(1, rounds + 1):
        environment_schedule = make_environment_key_schedule(
            environment_seed,
            synchronization_round=round_index,
            rollout_length=rollout_length,
            num_envs=num_envs,
        )
        policy_schedule = make_owner_policy_key_schedule(
            policy_seed,
            synchronization_round=round_index,
            owner_id=0,
            rollout_length=rollout_length,
        )
        output = collector(
            snapshots,
            identity,
            env_state,
            features,
            environment_schedule,
            policy_schedule,
            jnp.asarray(True),
        )
        jax.block_until_ready(output.batch.observations)
        env_state, features = output.env_state, output.features
        observation_dim = output.batch.observations.shape[-1]
        chunks.append(
            np.asarray(output.batch.observations).reshape(-1, observation_dim)
        )
        chunks.append(
            np.asarray(output.batch.next_observations).reshape(
                -1,
                observation_dim,
            )
        )

    observations = np.concatenate(chunks, axis=0)
    normalizer = fit_actor_observation_normalizer(
        observations,
        clip=clip,
        minimum_std=minimum_std,
    )
    metrics = {
        "sample_count": int(len(observations)),
        "environment_steps": int(rounds * num_envs * rollout_length),
        "std_min": float(np.min(np.asarray(normalizer.std))),
        "std_max": float(np.max(np.asarray(normalizer.std))),
    }
    return normalizer, metrics
