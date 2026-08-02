"""Five-member probabilistic joint delta-dynamics ensemble."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


class ProbabilisticDynamicsMember(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    minimum_logvar: float = -6.0
    maximum_logvar: float = 0.5

    @nn.compact
    def __call__(self, inputs):
        x = jnp.asarray(inputs, dtype=jnp.float32)
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(int(width), name=f"hidden_{index}")(x)
            x = nn.relu(x)
        mean = nn.Dense(self.output_dim, name="delta_mean")(x)
        raw_logvar = nn.Dense(self.output_dim, name="delta_logvar")(x)
        logvar = jnp.clip(
            raw_logvar,
            self.minimum_logvar,
            self.maximum_logvar,
        )
        return mean, logvar


class EnsembleJointDynamics(nn.Module):
    ensemble_size: int
    hidden_dims: Sequence[int]
    output_dim: int
    minimum_logvar: float = -6.0
    maximum_logvar: float = 0.5

    def setup(self):
        self.members = nn.vmap(
            ProbabilisticDynamicsMember,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
            out_axes=0,
            axis_size=self.ensemble_size,
        )(
            hidden_dims=tuple(int(v) for v in self.hidden_dims),
            output_dim=int(self.output_dim),
            minimum_logvar=float(self.minimum_logvar),
            maximum_logvar=float(self.maximum_logvar),
        )

    def __call__(self, inputs):
        return self.members(inputs)


def initialize_independent_dynamics_states(
    *,
    seed: int,
    num_owners: int,
    ensemble_size: int,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    learning_rate: float,
    maximum_gradient_norm: float,
    minimum_logvar: float,
    maximum_logvar: float,
) -> tuple[EnsembleJointDynamics, tuple[TrainState, ...]]:
    if num_owners < 1:
        raise ValueError("num_owners must be positive.")
    if ensemble_size != 5:
        raise ValueError("This experiment requires exactly five members.")

    model = EnsembleJointDynamics(
        ensemble_size=ensemble_size,
        hidden_dims=tuple(int(v) for v in hidden_dims),
        output_dim=output_dim,
        minimum_logvar=minimum_logvar,
        maximum_logvar=maximum_logvar,
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_owners)
    dummy = jnp.zeros(
        (ensemble_size, 1, input_dim),
        dtype=jnp.float32,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(maximum_gradient_norm)),
        optax.adam(float(learning_rate)),
    )
    states = tuple(
        TrainState.create(
            apply_fn=model.apply,
            params=model.init(key, dummy)["params"],
            tx=optimizer,
        )
        for key in keys
    )
    return model, states


def tile_for_ensemble(inputs: Any, ensemble_size: int) -> jnp.ndarray:
    array = jnp.asarray(inputs, dtype=jnp.float32)
    return jnp.broadcast_to(array[None, ...], (ensemble_size,) + array.shape)
