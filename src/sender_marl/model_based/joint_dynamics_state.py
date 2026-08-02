from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from sender_marl.model_based.joint_dynamics import (
    IndependentJointConditionedDynamicsEnsemble,
)


Array = jax.Array


@dataclass(frozen=True)
class IndependentJointDynamicsConfig:
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
class JointDynamicsStandardizer:
    """Normalizer owned by one local dynamics model.

    Full-state and joint-action statistics are numerically identical across
    agents when fit on the same data, while delta statistics are agent-specific.
    Keeping one complete object per agent makes checkpoint exchange explicit.
    """

    model_state_mean: Array
    model_state_std: Array
    joint_action_mean: Array
    joint_action_std: Array
    delta_mean: Array
    delta_std: Array

    @classmethod
    def fit(
        cls,
        model_states,
        joint_actions,
        local_states,
        next_local_states,
        *,
        minimum_std: float = 1e-6,
    ) -> "JointDynamicsStandardizer":
        states = np.asarray(model_states, dtype=np.float32)
        actions = np.asarray(joint_actions, dtype=np.float32)
        local = np.asarray(local_states, dtype=np.float32)
        next_local = np.asarray(next_local_states, dtype=np.float32)

        if states.ndim != 2:
            raise ValueError("model_states must have shape (B, S).")
        if actions.ndim != 3:
            raise ValueError("joint_actions must have shape (B, N, A).")
        if local.ndim != 2 or next_local.shape != local.shape:
            raise ValueError(
                "local_states and next_local_states must have shape (B, D)."
            )
        if states.shape[0] != actions.shape[0] or states.shape[0] != local.shape[0]:
            raise ValueError("Standardizer batch sizes differ.")

        actions_flat = actions.reshape(actions.shape[0], -1)
        delta = next_local - local

        def moments(array):
            mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
            std = np.std(array, axis=0, dtype=np.float64).astype(np.float32)
            std = np.maximum(std, np.float32(minimum_std))
            return jnp.asarray(mean), jnp.asarray(std)

        state_mean, state_std = moments(states)
        action_mean, action_std = moments(actions_flat)
        delta_mean, delta_std = moments(delta)
        return cls(
            model_state_mean=state_mean,
            model_state_std=state_std,
            joint_action_mean=action_mean,
            joint_action_std=action_std,
            delta_mean=delta_mean,
            delta_std=delta_std,
        )

    def normalize_model_state(self, value: Array) -> Array:
        return (value - self.model_state_mean) / self.model_state_std

    def normalize_joint_action(self, value: Array) -> Array:
        return (value - self.joint_action_mean) / self.joint_action_std

    def normalize_delta(self, value: Array) -> Array:
        return (value - self.delta_mean) / self.delta_std

    def denormalize_delta(self, value: Array) -> Array:
        return value * self.delta_std + self.delta_mean

    def to_serializable(self) -> dict[str, Any]:
        return {
            "model_state_mean": np.asarray(
                jax.device_get(self.model_state_mean)
            ),
            "model_state_std": np.asarray(
                jax.device_get(self.model_state_std)
            ),
            "joint_action_mean": np.asarray(
                jax.device_get(self.joint_action_mean)
            ),
            "joint_action_std": np.asarray(
                jax.device_get(self.joint_action_std)
            ),
            "delta_mean": np.asarray(jax.device_get(self.delta_mean)),
            "delta_std": np.asarray(jax.device_get(self.delta_std)),
        }


def create_independent_joint_dynamics_train_states(
    key: Array,
    *,
    num_agents: int,
    model_state_dim: int,
    action_dim: int,
    local_state_dim: int,
    config: IndependentJointDynamicsConfig,
) -> tuple[
    IndependentJointConditionedDynamicsEnsemble,
    list[TrainState],
]:
    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    model = IndependentJointConditionedDynamicsEnsemble(
        ensemble_size=config.ensemble_size,
        hidden_dims=config.hidden_dims,
        output_dim=local_state_dim,
        min_logvar=config.min_logvar,
        max_logvar=config.max_logvar,
    )
    input_dim = model_state_dim + num_agents * action_dim
    dummy = jnp.zeros(
        (config.ensemble_size, 1, input_dim),
        dtype=jnp.float32,
    )
    keys = jax.random.split(key, num_agents)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )
    train_states: list[TrainState] = []
    for agent_key in keys:
        params = model.init(agent_key, dummy)["params"]
        train_states.append(
            TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optimizer,
            )
        )
    return model, train_states
