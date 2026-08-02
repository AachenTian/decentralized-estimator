from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import linen as nn


HiddenDims = Sequence[int]


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(scale)


class SACActor(nn.Module):
    """State-dependent decentralized tanh-Gaussian actor."""

    action_dim: int
    hidden_dims: HiddenDims = (256, 256)
    minimum_log_std: float = -5.0
    maximum_log_std: float = 2.0

    @nn.compact
    def __call__(self, observations):
        x = jnp.asarray(observations, dtype=jnp.float32)
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                kernel_init=_orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
                name=f"hidden_{index}",
            )(x)
            x = nn.relu(x)

        mean = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="mean",
        )(x)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="log_std",
        )(x)
        return mean, jnp.clip(
            log_std,
            self.minimum_log_std,
            self.maximum_log_std,
        )


class QNetwork(nn.Module):
    """Centralized-input Q-network owned by one independent learner.

    `model_states` contains all agent states plus environment context. The joint
    action may be `(B, N, A)` or already flattened `(B, N*A)`.
    """

    hidden_dims: HiddenDims = (256, 256)

    @nn.compact
    def __call__(self, model_states, joint_actions):
        model_states = jnp.asarray(model_states, dtype=jnp.float32)
        joint_actions = jnp.asarray(joint_actions, dtype=jnp.float32)
        if joint_actions.ndim == model_states.ndim + 1:
            joint_actions = joint_actions.reshape(
                joint_actions.shape[:-2] + (-1,)
            )
        if joint_actions.ndim != model_states.ndim:
            raise ValueError(
                "joint_actions must be flattened or have one agent axis; "
                f"got state={model_states.shape}, action={joint_actions.shape}."
            )

        x = jnp.concatenate([model_states, joint_actions], axis=-1)
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                hidden_dim,
                kernel_init=_orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
                name=f"hidden_{index}",
            )(x)
            x = nn.relu(x)

        value = nn.Dense(
            1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="value",
        )(x)
        return jnp.squeeze(value, axis=-1)


class TwinQNetwork(nn.Module):
    """Two private Q-functions over joint state and joint action."""

    hidden_dims: HiddenDims = (256, 256)

    @nn.compact
    def __call__(self, model_states, joint_actions):
        q1 = QNetwork(self.hidden_dims, name="q1")(
            model_states, joint_actions
        )
        q2 = QNetwork(self.hidden_dims, name="q2")(
            model_states, joint_actions
        )
        return q1, q2
