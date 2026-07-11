"""Trajectory-level data collection for acceleration-controlled diff-drive."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np

from aorpo.envs.accel_diff_drive_multi_agent_env import (
    AccelDiffDriveMultiAgentEnv,
)


Array = jax.Array


class TrajectoryDataset(NamedTuple):
    """Arrays retain the trajectory axis to prevent split leakage.

    Shapes:
        local_states: (K, H + 1, N, 5)
        actions:      (K, H, N, 2)
    """
    local_states: Array
    actions: Array


def collect_piecewise_constant_trajectories(
    env: AccelDiffDriveMultiAgentEnv,
    rng: Array,
    num_trajectories: int,
    horizon: int,
    hold_steps_min: int = 5,
    hold_steps_max: int = 20,
    linear_action_scale: float = 1.0,
    angular_action_scale: float = 1.0,
) -> TrajectoryDataset:
    """Collect trajectories with temporally correlated excitation actions."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if hold_steps_min <= 0 or hold_steps_max < hold_steps_min:
        raise ValueError("Invalid hold-step range.")

    trajectory_states = []
    trajectory_actions = []
    key = rng

    for _ in range(num_trajectories):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        states = [env.local_states(state)]
        actions = []

        current_actions = jnp.zeros(
            (env.num_agents, env.action_dim),
            dtype=jnp.float32,
        )
        remaining = jnp.zeros((env.num_agents,), dtype=jnp.int32)

        for _step in range(horizon):
            key, action_key, hold_key, step_key = jax.random.split(key, 4)

            proposed_actions = env.sample_actions(
                action_key,
                linear_scale=linear_action_scale,
                angular_scale=angular_action_scale,
            )
            proposed_holds = jax.random.randint(
                hold_key,
                shape=(env.num_agents,),
                minval=hold_steps_min,
                maxval=hold_steps_max + 1,
                dtype=jnp.int32,
            )

            resample = remaining <= 0
            current_actions = jnp.where(
                resample[:, None],
                proposed_actions,
                current_actions,
            )
            remaining = jnp.where(
                resample,
                proposed_holds,
                remaining,
            )

            next_state, local_tp1, _done, _info = env.step(
                step_key,
                state,
                current_actions,
            )
            actions.append(current_actions)
            states.append(local_tp1)

            state = next_state
            remaining = remaining - 1

        trajectory_states.append(jnp.stack(states, axis=0))
        trajectory_actions.append(jnp.stack(actions, axis=0))

    return TrajectoryDataset(
        local_states=jnp.stack(trajectory_states, axis=0),
        actions=jnp.stack(trajectory_actions, axis=0),
    )


def flatten_local_transitions(
    dataset: TrajectoryDataset,
    trajectory_indices: np.ndarray | list[int] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Flatten selected trajectories into local supervised transitions.

    Returns:
        local_state_t:   (B, 5)
        local_action_t:  (B, 2)
        local_state_tp1: (B, 5)
        delta_t:         (B, 5)
    """
    states = dataset.local_states
    actions = dataset.actions

    if trajectory_indices is not None:
        idx = jnp.asarray(trajectory_indices, dtype=jnp.int32)
        states = states[idx]
        actions = actions[idx]

    state_t = states[:, :-1]
    state_tp1 = states[:, 1:]

    local_state_t = state_t.reshape(-1, state_t.shape[-1])
    local_action_t = actions.reshape(-1, actions.shape[-1])
    local_state_tp1 = state_tp1.reshape(-1, state_tp1.shape[-1])
    delta_t = local_state_tp1 - local_state_t

    return local_state_t, local_action_t, local_state_tp1, delta_t


def trajectory_split_indices(
    num_trajectories: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split by whole trajectories, never by adjacent transitions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1).")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be < 1.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(num_trajectories)
    n_train = int(round(train_fraction * num_trajectories))
    n_val = int(round(validation_fraction * num_trajectories))

    return {
        "train": order[:n_train],
        "validation": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def save_trajectory_dataset(
    path: str | Path,
    dataset: TrajectoryDataset,
    metadata: dict[str, object],
    splits: dict[str, np.ndarray],
) -> Path:
    """Save trajectory arrays, split indices, and JSON metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        local_states=np.asarray(dataset.local_states),
        actions=np.asarray(dataset.actions),
        train_trajectory_indices=np.asarray(splits["train"]),
        validation_trajectory_indices=np.asarray(splits["validation"]),
        test_trajectory_indices=np.asarray(splits["test"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return output
