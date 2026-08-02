from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from sender_marl.model_based.dynamics import SharedFactorizedDynamicsEnsemble


Array = jax.Array


@dataclass(frozen=True)
class SharedDynamicsConfig:
    ensemble_size: int = 5
    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    max_gradient_norm: float = 10.0
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def __post_init__(self) -> None:
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive.")
        if self.min_logvar >= self.max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar.")


@struct.dataclass
class DynamicsStandardizer:
    state_mean: Array
    state_std: Array
    action_mean: Array
    action_std: Array
    delta_mean: Array
    delta_std: Array

    @classmethod
    def fit(
        cls,
        local_states,
        actions,
        next_local_states,
        *,
        minimum_std: float = 1e-6,
    ) -> "DynamicsStandardizer":
        states = np.asarray(local_states, dtype=np.float32)
        acts = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_local_states, dtype=np.float32)
        if states.ndim != 2 or acts.ndim != 2 or next_states.shape != states.shape:
            raise ValueError("Invalid arrays for DynamicsStandardizer.fit().")
        if states.shape[0] != acts.shape[0]:
            raise ValueError("State and action batch sizes differ.")
        delta = next_states - states

        def moments(array):
            mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
            std = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
            std = np.maximum(std, np.float32(minimum_std))
            return jnp.asarray(mean), jnp.asarray(std)

        state_mean, state_std = moments(states)
        action_mean, action_std = moments(acts)
        delta_mean, delta_std = moments(delta)
        return cls(
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
        )

    def normalize_state(self, value: Array) -> Array:
        return (value - self.state_mean) / self.state_std

    def normalize_action(self, value: Array) -> Array:
        return (value - self.action_mean) / self.action_std

    def normalize_delta(self, value: Array) -> Array:
        return (value - self.delta_mean) / self.delta_std

    def denormalize_delta(self, value: Array) -> Array:
        return value * self.delta_std + self.delta_mean

    def to_serializable(self) -> dict[str, Any]:
        return {
            "state_mean": np.asarray(jax.device_get(self.state_mean)),
            "state_std": np.asarray(jax.device_get(self.state_std)),
            "action_mean": np.asarray(jax.device_get(self.action_mean)),
            "action_std": np.asarray(jax.device_get(self.action_std)),
            "delta_mean": np.asarray(jax.device_get(self.delta_mean)),
            "delta_std": np.asarray(jax.device_get(self.delta_std)),
        }


def create_shared_dynamics_train_state(
    key: Array,
    *,
    local_state_dim: int,
    action_dim: int,
    config: SharedDynamicsConfig,
) -> tuple[SharedFactorizedDynamicsEnsemble, TrainState]:
    model = SharedFactorizedDynamicsEnsemble(
        ensemble_size=config.ensemble_size,
        hidden_dims=config.hidden_dims,
        output_dim=local_state_dim,
        min_logvar=config.min_logvar,
        max_logvar=config.max_logvar,
    )
    dummy = jnp.zeros(
        (config.ensemble_size, 1, local_state_dim + action_dim),
        dtype=jnp.float32,
    )
    params = model.init(key, dummy)["params"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )
    return model, TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
