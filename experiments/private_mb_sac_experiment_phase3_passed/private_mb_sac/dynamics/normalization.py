"""Frozen common actor normalization and private dynamics normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from private_mb_sac.core.types import DynamicsNormalizer, ObservationNormalizer


def _fit_mean_std(
    values: np.ndarray,
    *,
    minimum_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError(
            "Statistics require a non-empty rank-two array, got "
            f"shape={array.shape}."
        )
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, float(minimum_std)).astype(np.float32)
    return mean, std


def fit_actor_observation_normalizer(
    observations: np.ndarray,
    *,
    clip: float = 10.0,
    minimum_std: float = 1.0e-3,
) -> ObservationNormalizer:
    """Fit one public frozen actor-observation normalizer.

    The input is calibration data produced by a separate random-policy rollout,
    not any owner's private training replay.
    """
    mean, std = _fit_mean_std(observations, minimum_std=minimum_std)
    return ObservationNormalizer(
        mean=jnp.asarray(mean),
        std=jnp.asarray(std),
        clip=jnp.asarray(float(clip), dtype=jnp.float32),
    )


def save_actor_observation_normalizer(
    normalizer: ObservationNormalizer,
    path: str | Path,
    *,
    sample_count: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "common_frozen_actor_observation_normalizer",
        "sample_count": int(sample_count),
        "mean": np.asarray(normalizer.mean).tolist(),
        "std": np.asarray(normalizer.std).tolist(),
        "clip": float(np.asarray(normalizer.clip)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_actor_observation_normalizer(
    path: str | Path,
) -> ObservationNormalizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ObservationNormalizer(
        mean=jnp.asarray(payload["mean"], dtype=jnp.float32),
        std=jnp.asarray(payload["std"], dtype=jnp.float32),
        clip=jnp.asarray(payload.get("clip", 10.0), dtype=jnp.float32),
    )


def flatten_joint_actions(actions: Any) -> jnp.ndarray:
    actions = jnp.asarray(actions, dtype=jnp.float32)
    if actions.ndim < 3:
        raise ValueError(
            "Joint actions must end with (num_agents, action_dim), got "
            f"shape={actions.shape}."
        )
    return actions.reshape(actions.shape[:-2] + (-1,))


def agent_state_delta(
    model_states: Any,
    next_model_states: Any,
    *,
    agent_state_dim: int = 12,
) -> jnp.ndarray:
    state = jnp.asarray(model_states, dtype=jnp.float32)
    next_state = jnp.asarray(next_model_states, dtype=jnp.float32)
    if state.shape != next_state.shape:
        raise ValueError(
            f"State shapes differ: {state.shape} and {next_state.shape}."
        )
    if state.shape[-1] < agent_state_dim:
        raise ValueError(
            f"Model state has only {state.shape[-1]} dimensions; "
            f"need at least {agent_state_dim}."
        )
    return next_state[..., :agent_state_dim] - state[..., :agent_state_dim]


def fit_private_dynamics_normalizer(
    replay: Any,
    *,
    minimum_std: float = 1.0e-3,
    clip: float = 10.0,
    agent_state_dim: int = 12,
) -> DynamicsNormalizer:
    """Fit one owner-private frozen normalizer from nonterminal replay data."""
    indices = replay.valid_dynamics_indices
    if len(indices) < 1:
        raise ValueError("No nonterminal replay transitions are available.")

    states = np.asarray(replay.model_states[indices], dtype=np.float32)
    actions = np.asarray(replay.actions[indices], dtype=np.float32).reshape(
        len(indices), -1
    )
    next_states = np.asarray(replay.next_model_states[indices], dtype=np.float32)
    deltas = next_states[:, :agent_state_dim] - states[:, :agent_state_dim]

    state_mean, state_std = _fit_mean_std(
        states,
        minimum_std=minimum_std,
    )
    action_mean, action_std = _fit_mean_std(
        actions,
        minimum_std=minimum_std,
    )
    delta_mean, delta_std = _fit_mean_std(
        deltas,
        minimum_std=minimum_std,
    )

    return DynamicsNormalizer(
        state_mean=jnp.asarray(state_mean),
        state_std=jnp.asarray(state_std),
        action_mean=jnp.asarray(action_mean),
        action_std=jnp.asarray(action_std),
        delta_mean=jnp.asarray(delta_mean),
        delta_std=jnp.asarray(delta_std),
        clip=jnp.asarray(float(clip), dtype=jnp.float32),
        sample_count=jnp.asarray(len(indices), dtype=jnp.int32),
    )


def normalize_dynamics_inputs(
    model_states: Any,
    actions: Any,
    normalizer: DynamicsNormalizer,
) -> jnp.ndarray:
    states = jnp.asarray(model_states, dtype=jnp.float32)
    joint_actions = flatten_joint_actions(actions)
    state_normalized = jnp.clip(
        (states - normalizer.state_mean)
        / jnp.maximum(normalizer.state_std, 1.0e-3),
        -normalizer.clip,
        normalizer.clip,
    )
    action_normalized = jnp.clip(
        (joint_actions - normalizer.action_mean)
        / jnp.maximum(normalizer.action_std, 1.0e-3),
        -normalizer.clip,
        normalizer.clip,
    )
    return jnp.concatenate([state_normalized, action_normalized], axis=-1)


def normalize_delta(
    delta: Any,
    normalizer: DynamicsNormalizer,
) -> jnp.ndarray:
    return jnp.clip(
        (jnp.asarray(delta, dtype=jnp.float32) - normalizer.delta_mean)
        / jnp.maximum(normalizer.delta_std, 1.0e-3),
        -normalizer.clip,
        normalizer.clip,
    )


def denormalize_delta(
    normalized_delta: Any,
    normalizer: DynamicsNormalizer,
) -> jnp.ndarray:
    return (
        jnp.asarray(normalized_delta, dtype=jnp.float32)
        * normalizer.delta_std
        + normalizer.delta_mean
    )
