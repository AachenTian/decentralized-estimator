from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from sender_marl.core.distributions import sample_squashed_gaussian
from sender_marl.envs.base import MultiAgentEnvAdapter
from sender_marl.off_policy.types import RealRolloutBatch, RealRolloutOutput


Array = jax.Array
ApplyFn = Callable[..., Any]


class FixedEnvironmentKeySchedule(NamedTuple):
    """Stateless environment RNG schedule shared by all rollout owners.

    Shapes:
        step_keys:  (T, E, 2)
        reset_keys: (T, E, 2)

    Every owner receives exactly the same keys for the same synchronization
    round. Policy/action sampling uses a separate owner-specific schedule.
    """

    step_keys: Array
    reset_keys: Array


class OwnerPolicyKeySchedule(NamedTuple):
    """Deterministic owner-specific action-sampling RNG schedule."""

    policy_keys: Array
    random_action_keys: Array


def _fold_in_many(key: Array, values: Sequence[int]) -> Array:
    result = key
    for value in values:
        result = jax.random.fold_in(result, int(value))
    return result


def make_fixed_initial_reset_keys(
    environment_seed: int,
    *,
    num_envs: int,
) -> Array:
    """Create the common initial reset keys used by every rollout owner."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    root = _fold_in_many(
        jax.random.PRNGKey(environment_seed),
        (0x13579, 0),
    )
    return jax.random.split(root, num_envs)


def make_fixed_environment_key_schedule(
    environment_seed: int,
    *,
    synchronization_round: int,
    rollout_length: int,
    num_envs: int,
) -> FixedEnvironmentKeySchedule:
    """Create reproducible common random numbers for one rollout round.

    The mapping from

        (environment_seed, synchronization_round, time, env_index)

    to a PRNG key is deterministic. Calling this function again with the same
    arguments produces the same schedule. All three rollout owners must use
    this same returned schedule.
    """

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
    step_keys = jax.vmap(lambda key: jax.random.split(key, num_envs))(
        step_time_keys
    )
    reset_keys = jax.vmap(lambda key: jax.random.split(key, num_envs))(
        reset_time_keys
    )
    return FixedEnvironmentKeySchedule(
        step_keys=step_keys,
        reset_keys=reset_keys,
    )


def make_owner_policy_key_schedule(
    policy_seed: int,
    *,
    synchronization_round: int,
    owner_agent_id: int,
    rollout_length: int,
) -> OwnerPolicyKeySchedule:
    """Create a reproducible policy RNG stream for one rollout owner.

    Owner-specific streams make the three private trajectories distinct while
    the environment randomness remains common and fixed.
    """

    if synchronization_round < 0 or owner_agent_id < 0:
        raise ValueError("round and owner_agent_id must be non-negative.")
    if rollout_length < 1:
        raise ValueError("rollout_length must be positive.")

    owner_root = _fold_in_many(
        jax.random.PRNGKey(policy_seed),
        (0x97531, synchronization_round, owner_agent_id),
    )
    policy_root = jax.random.fold_in(owner_root, 0)
    random_root = jax.random.fold_in(owner_root, 1)
    return OwnerPolicyKeySchedule(
        policy_keys=jax.random.split(policy_root, rollout_length),
        random_action_keys=jax.random.split(
            random_root, rollout_length
        ),
    )


def _tree_where_batch(mask: Array, when_true: Any, when_false: Any) -> Any:
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


def make_fixed_rng_oracle_rollout_collector(
    adapter: MultiAgentEnvAdapter,
    actor_apply_fn: ApplyFn,
    *,
    num_envs: int,
    rollout_length: int,
    jit: bool = True,
):
    """Collect one owner's private real-environment joint trajectory.

    Actor parameters are frozen snapshots supplied by the caller. Environment
    step/reset keys are supplied explicitly and are therefore independent of
    policy-sampling RNG consumption.
    """

    if num_envs < 1 or rollout_length < 1:
        raise ValueError("num_envs and rollout_length must be positive.")

    num_agents = adapter.spec.num_agents
    action_dim = adapter.spec.policy_action_dim

    def collect(
        actor_params: Any,
        env_state: Any,
        features: Any,
        environment_schedule: FixedEnvironmentKeySchedule,
        policy_schedule: OwnerPolicyKeySchedule,
        use_random_actions: Array,
    ) -> RealRolloutOutput:
        if environment_schedule.step_keys.shape[:2] != (
            rollout_length,
            num_envs,
        ):
            raise ValueError("Incorrect environment step-key schedule shape.")
        if environment_schedule.reset_keys.shape[:2] != (
            rollout_length,
            num_envs,
        ):
            raise ValueError("Incorrect environment reset-key schedule shape.")
        if policy_schedule.policy_keys.shape[0] != rollout_length:
            raise ValueError("Incorrect policy-key schedule length.")

        scan_inputs = (
            environment_schedule.step_keys,
            environment_schedule.reset_keys,
            policy_schedule.policy_keys,
            policy_schedule.random_action_keys,
        )

        def one_step(carry, xs):
            env_state, features = carry
            step_keys, reset_keys, policy_key, random_action_key = xs

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
            return (carry_env_state, carry_features), transition

        (final_env_state, final_features), batch = jax.lax.scan(
            one_step,
            (env_state, features),
            scan_inputs,
            length=rollout_length,
        )
        return RealRolloutOutput(
            env_state=final_env_state,
            features=final_features,
            batch=batch,
        )

    return jax.jit(collect) if jit else collect
