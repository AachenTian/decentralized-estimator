from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from flax import linen as nn


class SingleDeltaDynamics(nn.Module):
    """One probabilistic local delta-state model."""

    hidden_dims: Sequence[int]
    output_dim: int
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    @nn.compact
    def __call__(self, inputs):
        x = jnp.asarray(inputs, dtype=jnp.float32)
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = nn.relu(x)
        delta_mean = nn.Dense(self.output_dim, name="delta_mean")(x)
        delta_logvar = nn.Dense(self.output_dim, name="delta_logvar")(x)
        delta_logvar = jnp.clip(
            delta_logvar,
            self.min_logvar,
            self.max_logvar,
        )
        return delta_mean, delta_logvar


class SharedFactorizedDynamicsEnsemble(nn.Module):
    """Shared ensemble used independently for every homogeneous agent.

    Input must be ensemble-first: ``(K, B, local_state_dim + action_dim)``.
    Each member owns separate parameters and receives its own bootstrap batch.
    """

    ensemble_size: int
    hidden_dims: Sequence[int]
    output_dim: int
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def setup(self):
        self.members = nn.vmap(
            SingleDeltaDynamics,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
            out_axes=0,
            axis_size=self.ensemble_size,
        )(
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

    def __call__(self, ensemble_inputs):
        return self.members(ensemble_inputs)
