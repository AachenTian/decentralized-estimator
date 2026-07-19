#!/usr/bin/env python3
"""Plot a saved acceleration-controlled differential-drive dynamics model.

The script creates:

1. Open-loop true/predicted XY trajectories.
2. Position-error growth over rollout horizon.
3. Per-agent state-component trajectories.
4. Teacher-forced one-step predictive mean and ±2 sigma intervals.

Run this script from the repository root.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from aorpo.agents.independent_dynamics import (
    init_independent_transition_models,
    predict_local_ensemble_next_gaussians,
    predict_local_next,
)


STATE_NAMES = ("p_x", "p_y", "phi", "v", "omega")
STATE_LABELS = (
    "x position",
    "y position",
    "heading",
    "linear velocity",
    "angular velocity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/independent_dynamics_accel_diff_drive.pkl"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_debug.npz"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/figures/accel_diff_drive_dynamics"),
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=-1,
        help=(
            "Dataset trajectory index. The default -1 selects the first "
            "trajectory in the saved test split."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Maximum open-loop horizon to plot.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        result = {
            "local_states": np.asarray(data["local_states"]),
            "actions": np.asarray(data["actions"]),
            "test_indices": np.asarray(
                data["test_trajectory_indices"]
            ),
            "metadata": json.loads(str(data["metadata_json"])),
        }

    states = result["local_states"]
    actions = result["actions"]
    if states.ndim != 4 or actions.ndim != 4:
        raise ValueError(
            "Expected states (K,H+1,N,D) and actions (K,H,N,A)."
        )
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError("There must be one more state than action.")
    return result


def _parse_checkpoint_object(
    loaded: Any,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Accept common checkpoint return/pickle layouts."""
    if isinstance(loaded, dict):
        params = None
        for key in (
            "model_params",
            "saved_model_params",
            "params",
            "model_states",
        ):
            if key in loaded:
                params = loaded[key]
                break

        standardizers = None
        for key in (
            "standardizers",
            "local_standardizers",
        ):
            if key in loaded:
                standardizers = loaded[key]
                break

        metadata = loaded.get("metadata", {})

        if params is None or standardizers is None:
            raise KeyError(
                "Could not find model parameters and standardizers in "
                f"checkpoint keys: {sorted(loaded.keys())}"
            )
        return params, list(standardizers), dict(metadata)

    if isinstance(loaded, (tuple, list)) and len(loaded) == 3:
        params, standardizers, metadata = loaded
        return params, list(standardizers), dict(metadata)

    raise TypeError(
        "Unsupported checkpoint layout. Expected a dict or a 3-tuple."
    )


def load_checkpoint(
    path: Path,
) -> tuple[Any, list[Any], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Prefer the repository utility because it is the source of truth.
    try:
        from aorpo.utils.checkpoints import (
            load_independent_dynamics_checkpoint,
        )

        loaded = load_independent_dynamics_checkpoint(str(path))
        return _parse_checkpoint_object(loaded)
    except Exception as utility_error:
        # Fall back to direct pickle loading and report both errors if needed.
        try:
            with path.open("rb") as file:
                loaded = pickle.load(file)
            return _parse_checkpoint_object(loaded)
        except Exception as pickle_error:
            raise RuntimeError(
                "Unable to load checkpoint through either the repository "
                "utility or pickle.\n"
                f"Repository utility error: {utility_error}\n"
                f"Pickle error: {pickle_error}"
            ) from pickle_error


def make_model_config(metadata: dict[str, Any]) -> Any:
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=int(metadata.get("ensemble_size", 5)),
            hidden_dims=tuple(metadata.get("hidden_dims", [256, 256])),
            min_logvar=float(metadata.get("min_logvar", -10.0)),
            max_logvar=float(metadata.get("max_logvar", 0.5)),
            lr=float(metadata.get("learning_rate", 1e-3)),
        )
    )


def restore_model_states(
    initialized_states: list[Any],
    saved: Any,
) -> list[Any]:
    """Restore parameters while retaining freshly built apply functions."""
    try:
        from aorpo.utils.checkpoints import (
            restore_independent_model_states,
        )

        restored = restore_independent_model_states(
            initialized_states,
            saved,
        )
        return list(restored)
    except Exception:
        saved_items = list(saved)
        if len(saved_items) != len(initialized_states):
            raise ValueError(
                "Checkpoint agent count does not match initialized models: "
                f"{len(saved_items)} versus {len(initialized_states)}."
            )

        restored = []
        for initialized, saved_item in zip(
            initialized_states,
            saved_items,
        ):
            params = (
                saved_item.params
                if hasattr(saved_item, "params")
                else saved_item
            )
            restored.append(initialized.replace(params=params))
        return restored


def aggregate_one_step_gaussian(
    model_state: Any,
    standardizer: Any,
    states: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    raw = predict_local_ensemble_next_gaussians(
        train_state=model_state,
        standardizer=standardizer,
        local_state=states,
        local_action=actions,
    )
    member_means = raw.ensemble_next_means
    member_covariances = raw.ensemble_next_covariances

    mean = jnp.mean(member_means, axis=1)
    aleatoric = jnp.mean(member_covariances, axis=1)

    centered = member_means - mean[:, None, :]
    ensemble_size = int(member_means.shape[1])
    denominator = max(ensemble_size - 1, 1)
    epistemic = (
        jnp.einsum("bed,bef->bdf", centered, centered)
        / denominator
    )
    return mean, aleatoric + epistemic


def open_loop_rollout(
    model_state: Any,
    standardizer: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    predicted = jnp.asarray(initial_state[None, :])
    trajectory = [np.asarray(predicted[0])]

    for action in actions:
        predicted, _ = predict_local_next(
            train_state=model_state,
            standardizer=standardizer,
            local_state=predicted,
            local_action=jnp.asarray(action[None, :]),
            deterministic=True,
        )
        trajectory.append(np.asarray(predicted[0]))

    return np.stack(trajectory, axis=0)


def wrapped_heading_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    difference = predicted - target
    return np.arctan2(np.sin(difference), np.cos(difference))


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_xy(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 7))
    num_agents = true_states.shape[1]

    for agent_id in range(num_agents):
        plt.plot(
            true_states[:, agent_id, 0],
            true_states[:, agent_id, 1],
            label=f"agent {agent_id} true",
        )
        plt.plot(
            predicted_states[:, agent_id, 0],
            predicted_states[:, agent_id, 1],
            linestyle="--",
            label=f"agent {agent_id} predicted",
        )
        plt.scatter(
            true_states[0, agent_id, 0],
            true_states[0, agent_id, 1],
            marker="o",
        )
        plt.scatter(
            true_states[-1, agent_id, 0],
            true_states[-1, agent_id, 1],
            marker="x",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Open-loop trajectory: true vs learned dynamics")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_position_error(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(9, 5))
    num_agents = true_states.shape[1]

    for agent_id in range(num_agents):
        error = np.linalg.norm(
            predicted_states[:, agent_id, :2]
            - true_states[:, agent_id, :2],
            axis=-1,
        )
        plt.plot(error, label=f"agent {agent_id}")

    plt.xlabel("rollout step")
    plt.ylabel("position error")
    plt.title("Open-loop position-error accumulation")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_state_component(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    name: str,
    label: str,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(true_values, label="true")
    plt.plot(predicted_values, linestyle="--", label="open-loop prediction")
    plt.xlabel("rollout step")
    plt.ylabel(label)
    plt.title(f"Open-loop {name}: true vs learned dynamics")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_one_step_uncertainty(
    true_next: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    name: str,
    label: str,
    output: Path,
    dpi: int,
) -> None:
    time = np.arange(true_next.shape[0])
    lower = predicted_mean - 2.0 * predicted_std
    upper = predicted_mean + 2.0 * predicted_std

    plt.figure(figsize=(9, 5))
    plt.plot(time, true_next, label="true next state")
    plt.plot(time, predicted_mean, label="one-step predicted mean")
    plt.fill_between(
        time,
        lower,
        upper,
        alpha=0.25,
        label="predicted ±2σ",
    )
    plt.xlabel("transition index")
    plt.ylabel(label)
    plt.title(f"Teacher-forced one-step uncertainty: {name}")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    local_states = dataset["local_states"]
    actions = dataset["actions"]

    saved_params, standardizers, checkpoint_metadata = load_checkpoint(
        args.checkpoint
    )

    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)

    num_agents = int(local_states.shape[2])
    state_dim = int(local_states.shape[3])
    action_dim = int(actions.shape[3])

    if int(metadata.get("num_agents", num_agents)) != num_agents:
        raise ValueError("Checkpoint and dataset agent counts differ.")
    if int(metadata.get("local_state_dim", state_dim)) != state_dim:
        raise ValueError("Checkpoint and dataset state dimensions differ.")
    if int(metadata.get("action_dim", action_dim)) != action_dim:
        raise ValueError("Checkpoint and dataset action dimensions differ.")

    cfg = make_model_config(metadata)
    _, initialized_states = init_independent_transition_models(
        rng=jax.random.PRNGKey(args.seed),
        num_agents=num_agents,
        act_dim=action_dim,
        cfg=cfg,
        local_state_dim=state_dim,
    )
    model_states = restore_model_states(
        initialized_states,
        saved_params,
    )

    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, got "
            f"{len(standardizers)}."
        )

    trajectory_id = args.trajectory_index
    if trajectory_id < 0:
        test_indices = dataset["test_indices"]
        if len(test_indices) == 0:
            raise ValueError("Dataset test split is empty.")
        trajectory_id = int(test_indices[0])

    if not 0 <= trajectory_id < local_states.shape[0]:
        raise ValueError(
            f"trajectory-index {trajectory_id} is outside "
            f"[0, {local_states.shape[0] - 1}]."
        )

    horizon = min(
        int(args.horizon),
        int(actions.shape[1]),
    )
    true_states = local_states[trajectory_id, : horizon + 1]
    true_actions = actions[trajectory_id, :horizon]

    predicted_by_agent = []
    one_step_means = []
    one_step_covariances = []

    for agent_id in range(num_agents):
        predicted = open_loop_rollout(
            model_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            initial_state=true_states[0, agent_id],
            actions=true_actions[:, agent_id],
        )
        predicted_by_agent.append(predicted)

        one_step_mean, one_step_cov = aggregate_one_step_gaussian(
            model_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            states=jnp.asarray(true_states[:-1, agent_id]),
            actions=jnp.asarray(true_actions[:, agent_id]),
        )
        one_step_means.append(np.asarray(one_step_mean))
        one_step_covariances.append(np.asarray(one_step_cov))

    predicted_states = np.stack(predicted_by_agent, axis=1)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    plot_xy(
        true_states,
        predicted_states,
        args.output_directory / "open_loop_xy.png",
        args.dpi,
    )
    plot_position_error(
        true_states,
        predicted_states,
        args.output_directory / "open_loop_position_error.png",
        args.dpi,
    )

    for agent_id in range(num_agents):
        agent_dir = args.output_directory / f"agent_{agent_id}"
        one_step_std = np.sqrt(
            np.maximum(
                np.diagonal(
                    one_step_covariances[agent_id],
                    axis1=-2,
                    axis2=-1,
                ),
                0.0,
            )
        )

        for dim, (name, label) in enumerate(
            zip(STATE_NAMES, STATE_LABELS)
        ):
            plot_state_component(
                true_values=true_states[:, agent_id, dim],
                predicted_values=predicted_states[:, agent_id, dim],
                name=name,
                label=label,
                output=agent_dir / f"open_loop_{name}.png",
                dpi=args.dpi,
            )
            plot_one_step_uncertainty(
                true_next=true_states[1:, agent_id, dim],
                predicted_mean=one_step_means[agent_id][:, dim],
                predicted_std=one_step_std[:, dim],
                name=name,
                label=label,
                output=agent_dir / f"one_step_uncertainty_{name}.png",
                dpi=args.dpi,
            )

    np.savez_compressed(
        args.output_directory / "plotted_predictions.npz",
        trajectory_index=np.asarray(trajectory_id),
        true_states=true_states,
        actions=true_actions,
        predicted_open_loop_states=predicted_states,
        one_step_predicted_means=np.stack(one_step_means, axis=1),
        one_step_predicted_covariances=np.stack(
            one_step_covariances,
            axis=1,
        ),
    )

    print("Plots saved to:", args.output_directory)
    print("Trajectory index:", trajectory_id)
    print("Horizon:", horizon)
    print("Main figures:")
    print(" - open_loop_xy.png")
    print(" - open_loop_position_error.png")
    print("Per-agent state and uncertainty figures are in agent_* folders.")


if __name__ == "__main__":
    main()
