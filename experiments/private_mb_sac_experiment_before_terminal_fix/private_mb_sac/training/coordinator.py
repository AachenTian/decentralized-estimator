"""Phase-one three-owner collection coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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


@dataclass
class OwnerRuntime:
    owner_id: int
    env_state: Any
    features: Any
    real_replay: Any
    env_steps_total: int = 0


def copy_pytree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.copy(jnp.asarray(leaf)), tree)


def initialize_owner_runtimes(
    *,
    adapter: Any,
    replays: tuple[Any, ...],
    environment_seed: int,
    num_envs: int,
) -> list[OwnerRuntime]:
    reset_keys = make_initial_reset_keys(environment_seed, num_envs=num_envs)
    env_state, features = jax.vmap(adapter.reset)(reset_keys)
    return [
        OwnerRuntime(
            owner_id=owner_id,
            env_state=copy_pytree(env_state),
            features=copy_pytree(features),
            real_replay=replays[owner_id],
        )
        for owner_id in range(len(replays))
    ]


def collect_private_round(
    *,
    round_index: int,
    live_actor_params: tuple[Any, ...],
    owner_runtimes: list[OwnerRuntime],
    collector: Any,
    normalizer: Any,
    environment_seed: int,
    policy_seed: int,
    num_envs: int,
    rollout_length: int,
    use_random_actions: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Collect one round for all owners and append to private replays."""
    snapshots = exchange_actor_snapshots(
        live_actor_params,
        synchronization_round=round_index,
    )
    environment_schedule = make_environment_key_schedule(
        environment_seed,
        synchronization_round=round_index,
        rollout_length=rollout_length,
        num_envs=num_envs,
    )

    owner_metrics: list[dict[str, float]] = []
    round_start = perf_counter()
    expected_steps = num_envs * rollout_length

    for runtime in owner_runtimes:
        start = perf_counter()
        policy_schedule = make_owner_policy_key_schedule(
            policy_seed,
            synchronization_round=round_index,
            owner_id=runtime.owner_id,
            rollout_length=rollout_length,
        )
        output = collector(
            snapshots,
            normalizer,
            runtime.env_state,
            runtime.features,
            environment_schedule,
            policy_schedule,
            jnp.asarray(use_random_actions),
        )
        jax.block_until_ready(output.batch.actions)
        collection_seconds = perf_counter() - start

        runtime.env_state = output.env_state
        runtime.features = output.features
        runtime.real_replay.add_rollout(output.batch)
        runtime.env_steps_total += expected_steps

        rewards = np.asarray(output.batch.rewards)
        team_rewards = rewards.mean(axis=-1)
        episode_dones = np.asarray(output.batch.episode_dones)
        collision = np.asarray(output.batch.pair_collision_rates)
        actions = np.asarray(output.batch.actions)

        owner_metrics.append(
            {
                "real/env_steps_round": float(expected_steps),
                "real/env_steps_total": float(runtime.env_steps_total),
                "real/episodes_round": float(episode_dones.sum()),
                "real/return_mean": float(team_rewards.sum(axis=0).mean()),
                "real/collision_rate": float(collision.mean()),
                "real/replay_size": float(len(runtime.real_replay)),
                "actor/action_saturation_rate": float(
                    (np.abs(actions) > 0.95).mean()
                ),
                "timing/collection_seconds": float(collection_seconds),
            }
        )

    system_steps_round = expected_steps * len(owner_runtimes)
    system_steps_total = sum(runtime.env_steps_total for runtime in owner_runtimes)
    system_metrics = {
        "real_env_steps_round": float(system_steps_round),
        "real_env_steps_total": float(system_steps_total),
        "owner_env_steps_min": float(
            min(runtime.env_steps_total for runtime in owner_runtimes)
        ),
        "owner_env_steps_max": float(
            max(runtime.env_steps_total for runtime in owner_runtimes)
        ),
        "round_seconds": float(perf_counter() - round_start),
        "real_steps_per_second": float(
            system_steps_round / max(perf_counter() - round_start, 1.0e-9)
        ),
    }
    return owner_metrics, system_metrics
