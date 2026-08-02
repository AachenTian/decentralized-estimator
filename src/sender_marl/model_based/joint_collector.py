from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from sender_marl.envs.base import MultiAgentEnvAdapter
from sender_marl.model_based.joint_types import (
    JointDynamicsRolloutBatch,
    JointDynamicsRolloutOutput,
)


Array = jax.Array


def _tree_where_batch(mask: Array, when_true: Any, when_false: Any) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)

    def select(true_leaf, false_leaf):
        true_array = jnp.asarray(true_leaf)
        false_array = jnp.asarray(false_leaf)
        if false_array.ndim == 0:
            return jnp.where(jnp.any(mask), true_array, false_array)
        expanded = mask.reshape(
            (mask.shape[0],) + (1,) * (false_array.ndim - 1)
        )
        return jnp.where(expanded, true_array, false_array)

    return jax.tree_util.tree_map(select, when_true, when_false)


def make_random_joint_dynamics_collector(
    adapter: MultiAgentEnvAdapter,
    *,
    num_envs: int,
    rollout_length: int,
    jit: bool = True,
):
    """Collect full-state/joint-action transitions with random actions."""

    if num_envs < 1 or rollout_length < 1:
        raise ValueError("num_envs and rollout_length must be positive.")

    num_agents = adapter.spec.num_agents
    action_dim = adapter.spec.policy_action_dim

    def collect(
        key: Array,
        env_state: Any,
        features: Any,
    ) -> JointDynamicsRolloutOutput:
        def one_step(carry, _):
            env_state, features, key = carry
            key, action_key, step_key, reset_key = jax.random.split(key, 4)
            actions = jax.random.uniform(
                action_key,
                shape=(num_envs, num_agents, action_dim),
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float32,
            )
            step_keys = jax.random.split(step_key, num_envs)
            step_output = jax.vmap(adapter.step)(
                step_keys,
                env_state,
                actions,
            )

            reset_keys = jax.random.split(reset_key, num_envs)
            reset_env_state, reset_features = jax.vmap(adapter.reset)(reset_keys)
            carry_env_state = _tree_where_batch(
                step_output.episode_done,
                reset_env_state,
                step_output.env_state,
            )
            carry_features = _tree_where_batch(
                step_output.episode_done,
                reset_features,
                step_output.features,
            )

            transition = JointDynamicsRolloutBatch(
                model_states=features.model_state,
                joint_actions=actions,
                local_states=features.local_states,
                next_local_states=step_output.features.local_states,
                episode_dones=step_output.episode_done,
            )
            return (
                carry_env_state,
                carry_features,
                key,
            ), transition

        final_carry, batch = jax.lax.scan(
            one_step,
            (env_state, features, key),
            xs=None,
            length=rollout_length,
        )
        final_env_state, final_features, _ = final_carry
        return JointDynamicsRolloutOutput(
            env_state=final_env_state,
            features=final_features,
            batch=batch,
        )

    return jax.jit(collect) if jit else collect
