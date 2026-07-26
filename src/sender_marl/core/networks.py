from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import linen as nn


HiddenDims = Sequence[int]


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(scale)


class ContinuousActor(nn.Module):
    """Environment-independent shared continuous actor.

    Dense layers operate on the final feature dimension, so the same module
    accepts `(obs_dim,)`, `(num_agents, obs_dim)`, or arbitrary leading batch
    dimensions such as `(num_envs, num_agents, obs_dim)`.
    """

    action_dim: int
    hidden_dims: HiddenDims = (128, 128)
    initial_log_std: float = -0.5
    minimum_log_std: float = -5.0
    maximum_log_std: float = 2.0

    @nn.compact
    def __call__(self, observations):
        x = jnp.asarray(observations, dtype=jnp.float32)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=_orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)

        mean = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        log_std_parameter = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.action_dim,),
        )
        log_std_parameter = jnp.clip(
            log_std_parameter,
            self.minimum_log_std,
            self.maximum_log_std,
        )
        log_std = jnp.broadcast_to(log_std_parameter, mean.shape)
        return mean, log_std


class LocalValueNetwork(nn.Module):
    """Value function over each agent's decentralized actor observation."""

    hidden_dims: HiddenDims = (128, 128)

    @nn.compact
    def __call__(self, observations):
        x = jnp.asarray(observations, dtype=jnp.float32)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=_orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)
        value = nn.Dense(
            1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)
        return jnp.squeeze(value, axis=-1)


class CentralizedValueNetwork(nn.Module):
    """Value function over a centralized critic observation.

    The rollout collector broadcasts one global critic observation across the
    agent axis. A later MAPPO extension may append agent IDs before applying
    this network when agent-specific centralized values are desired.
    """

    hidden_dims: HiddenDims = (128, 128)

    @nn.compact
    def __call__(self, observations):
        x = jnp.asarray(observations, dtype=jnp.float32)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=_orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)
        value = nn.Dense(
            1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)
        return jnp.squeeze(value, axis=-1)
