"""Phase-one actor network and independent parameter initialization."""

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
        hidden_dims=tuple(int(v) for v in hidden_dims),
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_agents)
    dummy = jnp.zeros((observation_dim,), dtype=jnp.float32)
    params = tuple(actor.init(key, dummy)["params"] for key in keys)
    return actor, params


def make_independent_actor_apply(actor: SACActor, *, num_agents: int):
    """Return an apply function accepting one parameter tree per agent."""

    def apply(variables: dict[str, Any], observations):
        params_by_agent = variables["params"]
        if len(params_by_agent) != num_agents:
            raise ValueError(
                f"Expected {num_agents} actor parameter trees, got "
                f"{len(params_by_agent)}."
            )
        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.shape[-2] != num_agents:
            raise ValueError(
                "Observations must contain the agent axis at -2; "
                f"got {observations.shape}."
            )
        means = []
        log_stds = []
        for agent_id in range(num_agents):
            mean, log_std = actor.apply(
                {"params": params_by_agent[agent_id]},
                observations[..., agent_id, :],
            )
            means.append(mean)
            log_stds.append(log_std)
        return jnp.stack(means, axis=-2), jnp.stack(log_stds, axis=-2)

    return apply
