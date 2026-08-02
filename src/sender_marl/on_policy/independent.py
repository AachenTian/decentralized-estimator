from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sender_marl.on_policy.train_state import (
    PPOConfig,
    create_ppo_train_states,
)
from sender_marl.on_policy.update import make_ppo_update


Array = jax.Array
ApplyFn = Callable[..., Any]


def make_independent_actor_apply(
    actor_apply_fn: ApplyFn,
    num_agents: int,
) -> ApplyFn:
    """Wrap one actor module so each agent uses its own parameter tree.

    The wrapped function keeps the same Flax-style call signature used by the
    existing rollout collector and evaluator:

        apply({"params": tuple_of_agent_params}, observations)

    Observations must have an agent axis at position -2, e.g.:
        (num_agents, obs_dim)
        (num_envs, num_agents, obs_dim)
        (time, num_envs, num_agents, obs_dim)
    """

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    def apply(variables: dict[str, Any], observations: Array):
        params_by_agent = variables["params"]
        if len(params_by_agent) != num_agents:
            raise ValueError(
                "Expected one actor parameter tree per agent; got "
                f"{len(params_by_agent)} for {num_agents} agents."
            )

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.shape[-2] != num_agents:
            raise ValueError(
                "Actor observations must contain the agent axis at -2; "
                f"got shape={observations.shape}."
            )

        means = []
        log_stds = []
        for agent_id in range(num_agents):
            mean_i, log_std_i = actor_apply_fn(
                {"params": params_by_agent[agent_id]},
                observations[..., agent_id, :],
            )
            means.append(mean_i)
            log_stds.append(log_std_i)

        # Insert the agent axis immediately before the action dimension.
        return (
            jnp.stack(means, axis=-2),
            jnp.stack(log_stds, axis=-2),
        )

    return apply


def make_independent_value_apply(
    value_apply_fn: ApplyFn,
    num_agents: int,
) -> ApplyFn:
    """Wrap one local critic module so every agent has independent parameters."""

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    def apply(variables: dict[str, Any], observations: Array):
        params_by_agent = variables["params"]
        if len(params_by_agent) != num_agents:
            raise ValueError(
                "Expected one value parameter tree per agent; got "
                f"{len(params_by_agent)} for {num_agents} agents."
            )

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.shape[-2] != num_agents:
            raise ValueError(
                "Critic observations must contain the agent axis at -2; "
                f"got shape={observations.shape}."
            )

        values = []
        for agent_id in range(num_agents):
            value_i = value_apply_fn(
                {"params": params_by_agent[agent_id]},
                observations[..., agent_id, :],
            )
            values.append(value_i)

        # Scalar value outputs receive the agent axis at the end.
        return jnp.stack(values, axis=-1)

    return apply


def create_independent_ppo_train_states(
    key: Array,
    actor: Any,
    value_network: Any,
    *,
    num_agents: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    config: PPOConfig,
):
    """Create one actor TrainState and one critic TrainState per agent."""

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    dummy_actor_obs = jnp.zeros((actor_obs_dim,), dtype=jnp.float32)
    dummy_critic_obs = jnp.zeros((critic_obs_dim,), dtype=jnp.float32)
    agent_keys = jax.random.split(key, num_agents)

    actor_states = []
    value_states = []
    for agent_key in agent_keys:
        actor_state, value_state = create_ppo_train_states(
            agent_key,
            actor,
            value_network,
            dummy_actor_obs,
            dummy_critic_obs,
            config,
        )
        actor_states.append(actor_state)
        value_states.append(value_state)

    return tuple(actor_states), tuple(value_states)


def independent_params(states: tuple[Any, ...]) -> tuple[Any, ...]:
    """Extract a tuple containing one parameter tree per agent."""

    return tuple(state.params for state in states)


def _select_agent_rollout(rollout, agent_id: int):
    """Keep one agent axis so the existing PPO update can be reused.

    The shared PPO implementation expects arrays beginning with (T, E, N).
    We therefore slice agent i as i:i+1 rather than removing the N axis.
    """

    batch = rollout.batch.replace(
        actor_obs=rollout.batch.actor_obs[..., agent_id : agent_id + 1, :],
        critic_obs=rollout.batch.critic_obs[
            ..., agent_id : agent_id + 1, :
        ],
        pre_tanh_actions=rollout.batch.pre_tanh_actions[
            ..., agent_id : agent_id + 1, :
        ],
        log_probs=rollout.batch.log_probs[
            ..., agent_id : agent_id + 1
        ],
        values=rollout.batch.values[..., agent_id : agent_id + 1],
        rewards=rollout.batch.rewards[..., agent_id : agent_id + 1],
        dones=rollout.batch.dones[..., agent_id : agent_id + 1],
        alive=rollout.batch.alive[..., agent_id : agent_id + 1],
    )

    return rollout.replace(
        batch=batch,
        final_values=rollout.final_values[
            ..., agent_id : agent_id + 1
        ],
    )


@dataclass(frozen=True)
class IndependentPPOUpdateOutput:
    actor_states: tuple[Any, ...]
    value_states: tuple[Any, ...]
    metrics: dict[str, Array]


def make_independent_ppo_update(
    config: PPOConfig,
    *,
    num_agents: int,
    jit_single_agent_update: bool = True,
):
    """Build an independent-IPPO update from the existing PPO update.

    Each agent:
      - receives only its own trajectory samples,
      - owns separate actor/critic parameters,
      - owns separate Adam optimizer states,
      - applies gradient clipping independently.

    The returned metrics are means across agents.
    """

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    single_agent_update = make_ppo_update(
        config,
        jit=jit_single_agent_update,
    )

    def update(
        key: Array,
        actor_states: tuple[Any, ...],
        value_states: tuple[Any, ...],
        rollout,
    ) -> IndependentPPOUpdateOutput:
        if len(actor_states) != num_agents:
            raise ValueError("Incorrect number of actor states.")
        if len(value_states) != num_agents:
            raise ValueError("Incorrect number of value states.")

        agent_keys = jax.random.split(key, num_agents)
        outputs = []

        # The loop is over a small, static number of agents. Each underlying
        # single-agent PPO update is JIT compiled and reused across agents.
        for agent_id in range(num_agents):
            agent_rollout = _select_agent_rollout(rollout, agent_id)
            output_i = single_agent_update(
                agent_keys[agent_id],
                actor_states[agent_id],
                value_states[agent_id],
                agent_rollout,
            )
            outputs.append(output_i)

        new_actor_states = tuple(
            output.actor_state for output in outputs
        )
        new_value_states = tuple(
            output.value_state for output in outputs
        )

        metrics = jax.tree_util.tree_map(
            lambda *values: jnp.mean(jnp.stack(values, axis=0), axis=0),
            *(output.metrics for output in outputs),
        )

        return IndependentPPOUpdateOutput(
            actor_states=new_actor_states,
            value_states=new_value_states,
            metrics=metrics,
        )

    return update
