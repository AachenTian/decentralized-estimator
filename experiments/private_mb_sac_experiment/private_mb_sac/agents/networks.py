"""Independent Gaussian actors and owner-private centralized twin critics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class SACActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    minimum_log_std: float = -5.0
    maximum_log_std: float = 2.0

    @nn.compact
    def __call__(self, observations):
        x = jnp.asarray(observations, dtype=jnp.float32)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"hidden_{index}")(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim, name="mean")(x)
        log_std = nn.Dense(self.action_dim, name="log_std")(x)
        return mean, jnp.clip(
            log_std,
            self.minimum_log_std,
            self.maximum_log_std,
        )


class QNetwork(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, model_states, joint_actions):
        states = jnp.asarray(model_states, dtype=jnp.float32)
        actions = jnp.asarray(joint_actions, dtype=jnp.float32)

        if actions.shape[-2:] == (3, 2):
            actions = actions.reshape(actions.shape[:-2] + (6,))
        if states.shape[:-1] != actions.shape[:-1]:
            raise ValueError(
                "State and joint-action leading shapes differ: "
                f"{states.shape} versus {actions.shape}."
            )

        x = jnp.concatenate([states, actions], axis=-1)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"hidden_{index}")(x)
            x = nn.relu(x)
        value = nn.Dense(1, name="value")(x)
        return jnp.squeeze(value, axis=-1)


class TwinQNetwork(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, model_states, joint_actions):
        q1 = QNetwork(self.hidden_dims, name="q1")(
            model_states,
            joint_actions,
        )
        q2 = QNetwork(self.hidden_dims, name="q2")(
            model_states,
            joint_actions,
        )
        return q1, q2


def initialize_independent_actor_params(
    *,
    seed: int,
    num_agents: int,
    observation_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int],
) -> tuple[SACActor, tuple[Any, ...]]:
    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    actor = SACActor(
        action_dim=action_dim,
        hidden_dims=tuple(int(value) for value in hidden_dims),
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_agents)
    dummy_observation = jnp.zeros(
        (observation_dim,),
        dtype=jnp.float32,
    )
    params = tuple(
        actor.init(key, dummy_observation)["params"]
        for key in keys
    )
    return actor, params


def make_independent_actor_apply(actor: SACActor, *, num_agents: int):
    """Apply one actor module using one independent parameter tree per agent."""

    def apply(variables: dict[str, Any], observations):
        params_by_agent = variables["params"]
        if len(params_by_agent) != num_agents:
            raise ValueError(
                f"Expected {num_agents} actor parameter trees, got "
                f"{len(params_by_agent)}."
            )

        observations_array = jnp.asarray(
            observations,
            dtype=jnp.float32,
        )
        if observations_array.shape[-2] != num_agents:
            raise ValueError(
                "Observations must contain the agent axis at -2; "
                f"got {observations_array.shape}."
            )

        means = []
        log_stds = []
        for agent_id in range(num_agents):
            mean, log_std = actor.apply(
                {"params": params_by_agent[agent_id]},
                observations_array[..., agent_id, :],
            )
            means.append(mean)
            log_stds.append(log_std)

        return (
            jnp.stack(means, axis=-2),
            jnp.stack(log_stds, axis=-2),
        )

    return apply
