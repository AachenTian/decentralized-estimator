from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from sender_marl.core.distributions import sample_squashed_gaussian
from sender_marl.envs.base import MultiAgentEnvAdapter
from sender_marl.off_policy.types import RealRolloutBatch, RealRolloutOutput


Array = jax.Array
ApplyFn = Callable[..., Any]


def _tree_where_batch(mask: Array, when_true: Any, when_false: Any) -> Any:
    """Select reset values for vector environments that terminated."""

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


def make_real_rollout_collector(
    adapter: MultiAgentEnvAdapter,
    actor_apply_fn: ApplyFn,
    *,
    num_envs: int,
    rollout_length: int,
    jit: bool = True,
):
    """Collect joint real transitions with fully independent actors.

    Oracle remote states are intentionally used in this optimizer-validation
    stage. The sender-triggered estimator will replace this state source only
    after the independent SAC and model-rollout pipelines are validated.
    """

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    if rollout_length < 1:
        raise ValueError("rollout_length must be positive.")

    num_agents = adapter.spec.num_agents
    action_dim = adapter.spec.policy_action_dim

    def collect(
        key: Array,
        actor_params: Any,
        env_state: Any,
        features: Any,
        use_random_actions: Array,
    ) -> RealRolloutOutput:
        def one_step(carry, _):
            env_state, features, key = carry
            key, policy_key, random_key, step_key, reset_key = (
                jax.random.split(key, 5)
            )

            observations = adapter.build_actor_observations(
                features,
                features.local_states,
            )
            means, log_stds = actor_apply_fn(
                {"params": actor_params},
                observations,
            )

            agent_policy_keys = jax.random.split(
                policy_key,
                num_agents,
            )
            policy_actions = []
            for agent_id in range(num_agents):
                sample_i = sample_squashed_gaussian(
                    agent_policy_keys[agent_id],
                    means[..., agent_id, :],
                    log_stds[..., agent_id, :],
                )
                policy_actions.append(sample_i.action)
            policy_actions = jnp.stack(policy_actions, axis=-2)

            random_actions = jax.random.uniform(
                random_key,
                shape=(num_envs, num_agents, action_dim),
                minval=-1.0,
                maxval=1.0,
            )
            actions = jnp.where(
                jnp.asarray(use_random_actions, dtype=jnp.bool_),
                random_actions,
                policy_actions,
            )

            step_keys = jax.random.split(step_key, num_envs)
            step_output = jax.vmap(adapter.step)(
                step_keys,
                env_state,
                actions,
            )

            next_observations = adapter.build_actor_observations(
                step_output.features,
                step_output.features.local_states,
            )
            team_reward = jnp.mean(step_output.rewards, axis=-1)
            rewards = jnp.broadcast_to(
                team_reward[..., None],
                (num_envs, num_agents),
            )
            dones = jnp.broadcast_to(
                step_output.episode_done[..., None],
                (num_envs, num_agents),
            ).astype(jnp.float32)
            pair_collision_rate = step_output.metrics.get(
                "pair_collision_rate",
                jnp.zeros_like(team_reward),
            )

            reset_keys = jax.random.split(reset_key, num_envs)
            reset_env_state, reset_features = jax.vmap(adapter.reset)(
                reset_keys
            )
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

            transition = RealRolloutBatch(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                dones=dones,
                model_states=features.model_state,
                next_model_states=step_output.features.model_state,
                episode_dones=step_output.episode_done,
                pair_collision_rates=pair_collision_rate,
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
        return RealRolloutOutput(
            env_state=final_env_state,
            features=final_features,
            batch=batch,
        )

    return jax.jit(collect) if jit else collect
