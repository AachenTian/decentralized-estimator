from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

from sender_marl.core.distributions import sample_squashed_gaussian
from sender_marl.core.types import EnvFeatures
from sender_marl.envs.base import MultiAgentEnvAdapter
from sender_marl.on_policy.state_source import (
    CommonStateSource,
    OracleCommonStateSource,
)
from sender_marl.on_policy.types import (
    CollectorCarry,
    RolloutBatch,
    RolloutOutput,
)


Array = jax.Array
CriticMode = Literal["local", "centralized"]
ApplyFn = Callable[..., Any]


def _broadcast_centralized_critic_observation(
    critic_observation: Array,
    num_agents: int,
) -> Array:
    return jnp.broadcast_to(
        critic_observation[..., None, :],
        critic_observation.shape[:-1]
        + (num_agents, critic_observation.shape[-1]),
    )


def _tree_where_batch(mask: Array, when_true: Any, when_false: Any) -> Any:
    """Select batched PyTree leaves using a mask over the first dimension."""

    mask = jnp.asarray(mask, dtype=jnp.bool_)

    def select(true_leaf, false_leaf):
        true_array = jnp.asarray(true_leaf)
        false_array = jnp.asarray(false_leaf)
        if false_array.ndim == 0:
            return jnp.where(jnp.any(mask), true_array, false_array)
        expanded_mask = mask.reshape(
            (mask.shape[0],) + (1,) * (false_array.ndim - 1)
        )
        return jnp.where(expanded_mask, true_array, false_array)

    return jax.tree_util.tree_map(select, when_true, when_false)


def make_rollout_collector(
    adapter: MultiAgentEnvAdapter,
    actor_apply_fn: ApplyFn,
    value_apply_fn: ApplyFn,
    *,
    num_envs: int,
    rollout_length: int,
    critic_mode: CriticMode = "local",
    common_state_source: CommonStateSource | None = None,
    jit: bool = True,
):
    """Build an environment-independent, time-major rollout collector.

    Args:
        adapter: Concrete environment adapter implementing the common contract.
        actor_apply_fn: Flax actor `.apply` function.
        value_apply_fn: Flax value-network `.apply` function.
        num_envs: Number of vectorized environments.
        rollout_length: Number of real environment transitions to collect.
        critic_mode: `local` for IPPO/DPO, `centralized` for MAPPO.
        common_state_source: Provider for remote-agent states. Defaults to an
            oracle provider while validating the policy optimizer pipeline.
        jit: Return a JIT-compiled collector when true.

    Returns:
        Callable `(key, actor_params, value_params) -> RolloutOutput`.
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    if rollout_length < 1:
        raise ValueError("rollout_length must be positive.")
    if critic_mode not in ("local", "centralized"):
        raise ValueError(f"Unsupported critic_mode={critic_mode!r}.")

    source = common_state_source or OracleCommonStateSource()
    num_agents = adapter.spec.num_agents

    def build_observations(
        features: EnvFeatures,
        source_state: Any,
    ) -> tuple[Array, Array]:
        common_local_states = source.read(source_state)
        actor_obs = adapter.build_actor_observations(
            features,
            common_local_states,
        )
        if critic_mode == "local":
            critic_obs = actor_obs
        else:
            centralized = adapter.build_critic_observations(features)
            critic_obs = _broadcast_centralized_critic_observation(
                centralized,
                num_agents,
            )
        return actor_obs, critic_obs

    def collect(key: Array, actor_params: Any, value_params: Any) -> RolloutOutput:
        key, reset_key, scan_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, num_envs)
        env_state, features = jax.vmap(adapter.reset)(reset_keys)
        source_state = source.initialize(features)

        initial_carry = CollectorCarry(
            env_state=env_state,
            features=features,
            common_state_source_state=source_state,
            key=scan_key,
        )

        def one_step(carry: CollectorCarry, _):
            actor_obs, critic_obs = build_observations(
                carry.features,
                carry.common_state_source_state,
            )
            mean, log_std = actor_apply_fn(
                {"params": actor_params},
                actor_obs,
            )
            values = value_apply_fn(
                {"params": value_params},
                critic_obs,
            )

            next_key, action_key, env_key, source_key, reset_key = (
                jax.random.split(carry.key, 5)
            )
            sample = sample_squashed_gaussian(action_key, mean, log_std)
            env_keys = jax.random.split(env_key, num_envs)
            step_output = jax.vmap(adapter.step)(
                env_keys,
                carry.env_state,
                sample.action,
            )

            next_source_state, source_metrics = source.update(
                source_key,
                carry.common_state_source_state,
                carry.features,
                sample.action,
                step_output,
            )

            transition_metrics = {
                **step_output.metrics,
                **source_metrics,
            }
            transition = RolloutBatch(
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                actions=sample.action,
                pre_tanh_actions=sample.pre_tanh_action,
                log_probs=sample.log_prob,
                values=values,
                rewards=step_output.rewards,
                dones=step_output.dones,
                episode_dones=step_output.episode_done,
                local_states=carry.features.local_states,
                next_local_states=step_output.features.local_states,
                alive=carry.features.alive,
                metrics=transition_metrics,
            )

            # Auto-reset completed vectorized environments. Resetting all E
            # candidates is JAX-friendly; tree selection keeps only completed
            # environment resets in the carry.
            reset_keys = jax.random.split(reset_key, num_envs)
            reset_env_state, reset_features = jax.vmap(adapter.reset)(reset_keys)
            reset_source_state = source.initialize(reset_features)

            next_env_state = _tree_where_batch(
                step_output.episode_done,
                reset_env_state,
                step_output.env_state,
            )
            next_features = _tree_where_batch(
                step_output.episode_done,
                reset_features,
                step_output.features,
            )
            next_source_state = _tree_where_batch(
                step_output.episode_done,
                reset_source_state,
                next_source_state,
            )

            next_carry = CollectorCarry(
                env_state=next_env_state,
                features=next_features,
                common_state_source_state=next_source_state,
                key=next_key,
            )
            return next_carry, transition

        final_carry, batch = jax.lax.scan(
            one_step,
            initial_carry,
            xs=None,
            length=rollout_length,
        )
        final_actor_obs, final_critic_obs = build_observations(
            final_carry.features,
            final_carry.common_state_source_state,
        )
        final_values = value_apply_fn(
            {"params": value_params},
            final_critic_obs,
        )
        return RolloutOutput(
            batch=batch,
            final_env_state=final_carry.env_state,
            final_features=final_carry.features,
            final_common_state_source_state=(
                final_carry.common_state_source_state
            ),
            final_actor_obs=final_actor_obs,
            final_critic_obs=final_critic_obs,
            final_values=final_values,
        )

    return jax.jit(collect) if jit else collect
