from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from sender_marl.core.distributions import deterministic_squashed_action
from sender_marl.envs.base import MultiAgentEnvAdapter


Array = jax.Array
ApplyFn = Callable[..., Any]


@struct.dataclass
class EvaluationResult:
    episode_returns: Array
    episode_lengths: Array
    completed: Array
    episode_pair_collision_rates: Array


def make_policy_evaluator(
    adapter: MultiAgentEnvAdapter,
    actor_apply_fn: ApplyFn,
    *,
    num_envs: int,
    horizon: int,
    jit: bool = True,
):
    """Build a deterministic, oracle-state evaluator for one episode."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    def evaluate(key: Array, actor_params: Any) -> EvaluationResult:
        reset_key, scan_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        env_state, features = jax.vmap(adapter.reset)(reset_keys)

        active = jnp.ones((num_envs,), dtype=jnp.bool_)
        returns = jnp.zeros((num_envs,), dtype=jnp.float32)
        lengths = jnp.zeros((num_envs,), dtype=jnp.int32)
        collision_sums = jnp.zeros((num_envs,), dtype=jnp.float32)

        def one_step(carry, _):
            (
                env_state,
                features,
                active,
                returns,
                lengths,
                collision_sums,
                key,
            ) = carry
            actor_obs = adapter.build_actor_observations(
                features,
                features.local_states,
            )
            mean, _log_std = actor_apply_fn(
                {"params": actor_params},
                actor_obs,
            )
            actions = deterministic_squashed_action(mean)

            next_key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_envs)
            step_output = jax.vmap(adapter.step)(
                step_keys,
                env_state,
                actions,
            )

            active_float = active.astype(jnp.float32)
            team_reward = jnp.mean(step_output.rewards, axis=-1)
            returns = returns + active_float * team_reward
            lengths = lengths + active.astype(jnp.int32)

            pair_collision_rate = step_output.metrics.get(
                "pair_collision_rate",
                jnp.zeros_like(team_reward),
            )
            collision_sums = (
                collision_sums
                + active_float * pair_collision_rate
            )

            active = jnp.logical_and(
                active,
                jnp.logical_not(step_output.episode_done),
            )
            return (
                step_output.env_state,
                step_output.features,
                active,
                returns,
                lengths,
                collision_sums,
                next_key,
            ), None

        final_carry, _ = jax.lax.scan(
            one_step,
            (
                env_state,
                features,
                active,
                returns,
                lengths,
                collision_sums,
                scan_key,
            ),
            xs=None,
            length=horizon,
        )
        (
            _,
            _,
            active,
            returns,
            lengths,
            collision_sums,
            _,
        ) = final_carry

        safe_lengths = jnp.maximum(lengths, 1).astype(jnp.float32)
        return EvaluationResult(
            episode_returns=returns,
            episode_lengths=lengths,
            completed=jnp.logical_not(active),
            episode_pair_collision_rates=(
                collision_sums / safe_lengths
            ),
        )

    return jax.jit(evaluate) if jit else evaluate
