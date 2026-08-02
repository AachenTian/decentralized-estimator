"""Deterministic joint evaluation for the three independently trained actors."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from private_mb_sac.agents.snapshots import exchange_actor_snapshots
from private_mb_sac.core.rng import (
    FORCED_ENVIRONMENT_SEED,
    forced_environment_key,
)
from private_mb_sac.rollout.real_collector import (
    normalize_actor_observations,
    resolve_episode_done,
)


def _final_landmark_distances(model_states, *, num_agents: int):
    states = jnp.asarray(model_states, dtype=jnp.float32)
    agent_positions = states[..., : num_agents * 4].reshape(
        states.shape[:-1] + (num_agents, 4)
    )[..., :2]
    landmark_positions = states[..., num_agents * 4 :].reshape(
        states.shape[:-1] + (num_agents, 2)
    )
    distances = jnp.linalg.norm(
        agent_positions[..., :, None, :]
        - landmark_positions[..., None, :, :],
        axis=-1,
    )
    return jnp.min(distances, axis=-2)


def make_deterministic_evaluator(
    adapter: Any,
    actor_apply_fn: Any,
    *,
    num_envs: int,
    horizon: int,
    jit: bool = True,
):
    if num_envs < 1 or horizon < 1:
        raise ValueError("num_envs and horizon must be positive.")

    num_agents = int(adapter.spec.num_agents)

    def evaluate(actor_params, normalizer, reset_keys, step_keys):
        snapshot_bank = exchange_actor_snapshots(
            actor_params,
            synchronization_round=0,
        )
        env_state, features = jax.vmap(adapter.reset)(reset_keys)
        active = jnp.ones((num_envs,), dtype=jnp.bool_)
        returns = jnp.zeros((num_envs,), dtype=jnp.float32)
        collision_sum = jnp.zeros((num_envs,), dtype=jnp.float32)
        active_steps = jnp.zeros((num_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_envs,), dtype=jnp.int32)

        def one_step(carry, keys):
            (
                current_state,
                current_features,
                active_mask,
                return_sum,
                collision_total,
                transition_count,
                lengths,
            ) = carry

            raw_observations = adapter.build_actor_observations(
                current_features,
                current_features.local_states,
            )
            actor_observations = normalize_actor_observations(
                raw_observations,
                normalizer,
            )
            means, _ = actor_apply_fn(
                {"params": snapshot_bank.params_by_agent},
                actor_observations,
            )
            actions = jnp.tanh(means)
            output = jax.vmap(adapter.step)(
                keys,
                current_state,
                actions,
            )
            episode_done, _, _ = resolve_episode_done(adapter, output)

            team_reward = jnp.mean(output.rewards, axis=-1)
            collision_rate = output.metrics.get(
                "pair_collision_rate",
                jnp.zeros((num_envs,), dtype=jnp.float32),
            )
            active_float = active_mask.astype(jnp.float32)
            next_returns = return_sum + active_float * team_reward
            next_collision = (
                collision_total + active_float * collision_rate
            )
            next_transition_count = transition_count + active_float
            next_lengths = lengths + active_mask.astype(jnp.int32)
            next_active = active_mask & ~episode_done

            return (
                output.env_state,
                output.features,
                next_active,
                next_returns,
                next_collision,
                next_transition_count,
                next_lengths,
            ), None

        final, _ = jax.lax.scan(
            one_step,
            (
                env_state,
                features,
                active,
                returns,
                collision_sum,
                active_steps,
                episode_lengths,
            ),
            step_keys,
            length=horizon,
        )
        (
            final_state,
            final_features,
            final_active,
            final_returns,
            final_collision_sum,
            final_active_steps,
            final_lengths,
        ) = final
        del final_state

        landmark_min_distances = _final_landmark_distances(
            final_features.model_state,
            num_agents=num_agents,
        )
        collision_mean = final_collision_sum / jnp.maximum(
            final_active_steps,
            1.0,
        )
        return {
            "returns": final_returns,
            "collision_rates": collision_mean,
            "episode_lengths": final_lengths,
            "completed": ~final_active,
            "landmark_min_distances": landmark_min_distances,
        }

    return jax.jit(evaluate) if jit else evaluate


def evaluate_live_actors(
    evaluator,
    *,
    actor_params,
    normalizer,
    environment_seed: int,
    num_envs: int,
    horizon: int,
) -> dict[str, float]:
    # The argument remains for metadata/API compatibility. This experiment
    # forces every evaluation environment to the exact seed-40 layout.
    del environment_seed
    root = forced_environment_key()
    reset_keys = jnp.broadcast_to(root, (num_envs, 2))
    step_time_keys = jax.vmap(
        lambda time_index: jax.random.fold_in(root, time_index + 1)
    )(jnp.arange(horizon, dtype=jnp.uint32))
    step_keys = jnp.broadcast_to(
        step_time_keys[:, None, :],
        (horizon, num_envs, 2),
    )

    output = evaluator(
        actor_params,
        normalizer,
        reset_keys,
        step_keys,
    )
    jax.block_until_ready(output["returns"])

    returns = np.asarray(output["returns"], dtype=np.float64)
    collisions = np.asarray(
        output["collision_rates"],
        dtype=np.float64,
    )
    lengths = np.asarray(
        output["episode_lengths"],
        dtype=np.float64,
    )
    completed = np.asarray(output["completed"], dtype=np.float64)
    landmark_distances = np.asarray(
        output["landmark_min_distances"],
        dtype=np.float64,
    )

    worst_count = max(1, int(np.ceil(0.10 * len(returns))))
    worst_returns = np.sort(returns)[:worst_count]

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_median": float(np.median(returns)),
        "return_worst10_mean": float(np.mean(worst_returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "collision_rate_mean": float(np.mean(collisions)),
        "episode_length_mean": float(np.mean(lengths)),
        "completion_rate": float(np.mean(completed)),
        "landmark_mean_min_distance": float(
            np.mean(landmark_distances)
        ),
        "landmark_max_min_distance": float(
            np.max(landmark_distances)
        ),
        "environment_steps": float(np.sum(lengths)),
    }
