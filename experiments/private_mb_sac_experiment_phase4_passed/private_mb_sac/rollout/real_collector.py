"""Independent real-environment collection using round-frozen actors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from private_mb_sac.core.types import (
    ActorSnapshotBank,
    EnvironmentKeySchedule,
    ObservationNormalizer,
    OwnerPolicyKeySchedule,
    RealRolloutBatch,
    RealRolloutOutput,
)


ApplyFn = Callable[..., Any]
Array = jax.Array


def resolve_episode_done(adapter: Any, output: Any) -> tuple[Array, Array, Array]:
    """Combine native environment termination with the configured time limit.

    Some JaxMARL Simple-Spread versions expose the terminal state at
    ``state.step == max_steps`` but do not raise ``__all__`` until another
    environment step is requested. A rollout of exactly 25 transitions would
    therefore otherwise contain zero terminal flags.

    Returns:
        episode_done: native_done OR timeout_done.
        native_done: termination reported by the adapter/environment.
        timeout_done: termination inferred from next_state.step.
    """
    native_done = jnp.asarray(output.episode_done, dtype=jnp.bool_)
    next_step = getattr(output.env_state, "step", None)
    if next_step is None:
        timeout_done = jnp.zeros_like(native_done)
    else:
        max_steps = int(adapter.config.max_steps)
        timeout_done = jnp.asarray(next_step) >= max_steps
        timeout_done = jnp.asarray(timeout_done, dtype=jnp.bool_)
    episode_done = jnp.logical_or(native_done, timeout_done)
    return episode_done, native_done, timeout_done


def _tree_where_batch(mask: Array, when_true: Any, when_false: Any) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)

    def select(true_leaf, false_leaf):
        true_array = jnp.asarray(true_leaf)
        false_array = jnp.asarray(false_leaf)
        if false_array.ndim == 0:
            return jnp.where(jnp.any(mask), true_array, false_array)
        expanded = mask.reshape((mask.shape[0],) + (1,) * (false_array.ndim - 1))
        return jnp.where(expanded, true_array, false_array)

    return jax.tree_util.tree_map(select, when_true, when_false)


def normalize_actor_observations(
    observations: Array,
    normalizer: ObservationNormalizer,
) -> Array:
    std = jnp.maximum(normalizer.std, 1.0e-3)
    return jnp.clip(
        (jnp.asarray(observations, dtype=jnp.float32) - normalizer.mean) / std,
        -normalizer.clip,
        normalizer.clip,
    )


def make_private_real_collector(
    adapter: Any,
    actor_apply_fn: ApplyFn,
    *,
    num_envs: int,
    rollout_length: int,
    jit: bool = True,
):
    """Build a collector used independently by each rollout owner."""
    if num_envs < 1 or rollout_length < 1:
        raise ValueError("num_envs and rollout_length must be positive.")

    num_agents = int(adapter.spec.num_agents)
    action_dim = int(adapter.spec.policy_action_dim)

    def collect(
        snapshot_bank: ActorSnapshotBank,
        normalizer: ObservationNormalizer,
        env_state: Any,
        features: Any,
        environment_schedule: EnvironmentKeySchedule,
        policy_schedule: OwnerPolicyKeySchedule,
        use_random_actions: Array,
    ) -> RealRolloutOutput:
        if len(snapshot_bank.params_by_agent) != num_agents:
            raise ValueError("Snapshot bank has the wrong number of actors.")
        scan_inputs = (
            environment_schedule.step_keys,
            environment_schedule.reset_keys,
            policy_schedule.action_keys,
            policy_schedule.random_action_keys,
        )

        def one_step(carry, xs):
            current_env_state, current_features = carry
            step_keys, reset_keys, action_key, random_action_key = xs

            raw_observations = adapter.build_actor_observations(
                current_features,
                current_features.local_states,
            )
            actor_observations = normalize_actor_observations(
                raw_observations,
                normalizer,
            )
            means, log_stds = actor_apply_fn(
                {"params": snapshot_bank.params_by_agent},
                actor_observations,
            )
            noise = jax.random.normal(action_key, means.shape)
            policy_actions = jnp.tanh(means + jnp.exp(log_stds) * noise)
            random_actions = jax.random.uniform(
                random_action_key,
                shape=(num_envs, num_agents, action_dim),
                minval=-1.0,
                maxval=1.0,
            )
            actions = jnp.where(
                jnp.asarray(use_random_actions, dtype=jnp.bool_),
                random_actions,
                policy_actions,
            )

            output = jax.vmap(adapter.step)(step_keys, current_env_state, actions)
            episode_done, _native_done, _timeout_done = resolve_episode_done(
                adapter,
                output,
            )
            next_observations = adapter.build_actor_observations(
                output.features,
                output.features.local_states,
            )
            dones = jnp.broadcast_to(
                episode_done[..., None],
                (num_envs, num_agents),
            ).astype(jnp.float32)

            reset_env_state, reset_features = jax.vmap(adapter.reset)(reset_keys)
            carry_env_state = _tree_where_batch(
                episode_done,
                reset_env_state,
                output.env_state,
            )
            carry_features = _tree_where_batch(
                episode_done,
                reset_features,
                output.features,
            )

            current_episode_steps = jnp.asarray(
                getattr(current_env_state, "step"), dtype=jnp.int32
            )
            next_episode_steps = jnp.asarray(
                getattr(output.env_state, "step"), dtype=jnp.int32
            )

            transition = RealRolloutBatch(
                observations=raw_observations,
                actions=actions,
                rewards=output.rewards,
                next_observations=next_observations,
                dones=dones,
                model_states=current_features.model_state,
                next_model_states=output.features.model_state,
                episode_dones=episode_done,
                pair_collision_rates=output.metrics.get(
                    "pair_collision_rate",
                    jnp.zeros((num_envs,), dtype=jnp.float32),
                ),
                min_pair_distances=output.metrics.get(
                    "min_pair_distance",
                    jnp.full((num_envs,), jnp.nan, dtype=jnp.float32),
                ),
                episode_steps=current_episode_steps,
                next_episode_steps=next_episode_steps,
            )
            return (carry_env_state, carry_features), transition

        (final_env_state, final_features), batch = jax.lax.scan(
            one_step,
            (env_state, features),
            scan_inputs,
            length=rollout_length,
        )
        return RealRolloutOutput(final_env_state, final_features, batch)

    return jax.jit(collect) if jit else collect
