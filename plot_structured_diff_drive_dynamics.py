#!/usr/bin/env python3
"""Plot a saved structured probabilistic diff-drive dynamics checkpoint.

Outputs:
- Open-loop XY trajectories for ensemble mean and Infoprop CI.
- Position-error growth curves.
- Per-agent state-component open-loop plots.
- Teacher-forced one-step mean and ±2σ uncertainty plots.
- Saved numeric predictions in NPZ format.

Run from the repository root.
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

from aorpo.agents.structured_diff_drive_dynamics import (
    init_structured_transition_models,
    moment_match_structured_prediction,
    predict_structured_ensemble_next_gaussians,
    predict_structured_infoprop,
    predict_structured_next,
)


STATE_NAMES = ("p_x", "p_y", "phi", "v", "omega")
STATE_LABELS = (
    "x position",
    "y position",
    "heading",
    "linear velocity",
    "angular velocity",
)
PREDICTION_MODES = ("ensemble_mean", "infoprop_ci")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/"
            "structured_diff_drive_dynamics_accel_control.pkl"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_500traj.npz"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/figures/structured_diff_drive_dynamics"
        ),
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=-1,
        help=(
            "Dataset trajectory index. -1 uses the first saved test "
            "trajectory."
        ),
    )
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {
            "local_states": np.asarray(data["local_states"]),
            "actions": np.asarray(data["actions"]),
            "test_indices": np.asarray(
                data["test_trajectory_indices"]
            ),
            "metadata": json.loads(str(data["metadata_json"])),
        }


def parse_checkpoint_object(
    loaded: Any,
) -> tuple[Any, list[Any], dict[str, Any]]:
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
        for key in ("standardizers", "local_standardizers"):
            if key in loaded:
                standardizers = loaded[key]
                break

        metadata = loaded.get("metadata", {})
        if params is None or standardizers is None:
            raise KeyError(
                "Checkpoint does not contain recognizable model parameters "
                f"and standardizers. Keys: {sorted(loaded.keys())}"
            )
        return params, list(standardizers), dict(metadata)

    if isinstance(loaded, (tuple, list)) and len(loaded) == 3:
        params, standardizers, metadata = loaded
        return params, list(standardizers), dict(metadata)

    raise TypeError(
        "Unsupported checkpoint layout. Expected dict or length-3 tuple."
    )


def load_checkpoint(
    path: Path,
) -> tuple[Any, list[Any], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    utility_error = None
    try:
        from aorpo.utils.checkpoints import (
            load_independent_dynamics_checkpoint,
        )

        loaded = load_independent_dynamics_checkpoint(str(path))
        return parse_checkpoint_object(loaded)
    except Exception as error:
        utility_error = error

    try:
        with path.open("rb") as file:
            loaded = pickle.load(file)
        return parse_checkpoint_object(loaded)
    except Exception as pickle_error:
        raise RuntimeError(
            "Unable to load checkpoint.\n"
            f"Repository utility error: {utility_error}\n"
            f"Direct pickle error: {pickle_error}"
        ) from pickle_error


def make_model_config(metadata: dict[str, Any]) -> Any:
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=int(metadata.get("ensemble_size", 5)),
            hidden_dims=tuple(
                metadata.get("hidden_dims", [256, 256])
            ),
            min_logvar=float(metadata.get("min_logvar", -6.0)),
            max_logvar=float(metadata.get("max_logvar", 0.5)),
            lr=float(metadata.get("learning_rate", 1.0e-3)),
        )
    )


def restore_model_states(
    initialized_states: list[Any],
    saved: Any,
) -> list[Any]:
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
                "Saved and initialized agent counts differ: "
                f"{len(saved_items)} vs {len(initialized_states)}."
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


def model_kwargs(
    metadata: dict[str, Any],
) -> dict[str, float]:
    return {
        "dt": float(metadata["dt"]),
        "min_v": float(metadata["min_v"]),
        "max_v": float(metadata["max_v"]),
        "max_omega": float(metadata["max_omega"]),
    }


def open_loop_rollout(
    model_state: Any,
    standardizer: Any,
    initial_state: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
    mode: str,
) -> np.ndarray:
    predicted = jnp.asarray(initial_state[None, :])
    result = [np.asarray(predicted[0])]
    kwargs = model_kwargs(metadata)

    for action in actions:
        predicted, _ = predict_structured_next(
            train_state=model_state,
            standardizer=standardizer,
            local_state=predicted,
            local_action=jnp.asarray(action[None, :]),
            prediction_mode=mode,
            epistemic_process_scale=float(
                metadata.get("epistemic_process_scale", 1.0)
            ),
            **kwargs,
        )
        result.append(np.asarray(predicted[0]))

    return np.stack(result, axis=0)


def one_step_predictions(
    model_state: Any,
    standardizer: Any,
    states: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    kwargs = model_kwargs(metadata)

    raw = predict_structured_ensemble_next_gaussians(
        train_state=model_state,
        standardizer=standardizer,
        local_state=jnp.asarray(states),
        local_action=jnp.asarray(actions),
        **kwargs,
    )

    moment_mean, _, _, moment_total = (
        moment_match_structured_prediction(raw)
    )

    infoprop = predict_structured_infoprop(
        train_state=model_state,
        standardizer=standardizer,
        local_state=jnp.asarray(states),
        local_action=jnp.asarray(actions),
        **kwargs,
    )

    return {
        "ensemble_mean": (
            np.asarray(moment_mean),
            np.asarray(moment_total),
        ),
        "infoprop_ci": (
            np.asarray(infoprop.next_mean),
            np.asarray(infoprop.total_predictive_covariance),
        ),
    }


def wrapped_heading_error(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    difference = predicted - truth
    return np.arctan2(np.sin(difference), np.cos(difference))


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_xy(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    mode: str,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8.5, 7.5))
    num_agents = truth.shape[1]

    for agent_id in range(num_agents):
        plt.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            label=f"agent {agent_id} true",
        )
        plt.plot(
            predictions[mode][:, agent_id, 0],
            predictions[mode][:, agent_id, 1],
            linestyle="--",
            label=f"agent {agent_id} predicted",
        )
        plt.scatter(
            truth[0, agent_id, 0],
            truth[0, agent_id, 1],
            marker="o",
        )
        plt.scatter(
            truth[-1, agent_id, 0],
            truth[-1, agent_id, 1],
            marker="x",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Open-loop trajectory: {mode}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_position_error_comparison(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(10, 6))
    num_agents = truth.shape[1]

    for mode in PREDICTION_MODES:
        for agent_id in range(num_agents):
            error = np.linalg.norm(
                predictions[mode][:, agent_id, :2]
                - truth[:, agent_id, :2],
                axis=-1,
            )
            plt.plot(
                error,
                label=f"{mode}, agent {agent_id}",
            )

    plt.xlabel("rollout step")
    plt.ylabel("position error")
    plt.title("Open-loop position-error accumulation")
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_state_component_comparison(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    agent_id: int,
    dim: int,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        truth[:, agent_id, dim],
        label="true",
    )
    for mode in PREDICTION_MODES:
        plt.plot(
            predictions[mode][:, agent_id, dim],
            linestyle="--",
            label=mode,
        )

    plt.xlabel("rollout step")
    plt.ylabel(STATE_LABELS[dim])
    plt.title(
        f"Agent {agent_id} open-loop {STATE_NAMES[dim]}"
    )
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def plot_one_step_uncertainty(
    truth_next: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    agent_id: int,
    dim: int,
    mode: str,
    output: Path,
    dpi: int,
) -> None:
    std = np.sqrt(
        np.maximum(covariance[:, dim, dim], 0.0)
    )
    lower = mean[:, dim] - 2.0 * std
    upper = mean[:, dim] + 2.0 * std
    time = np.arange(truth_next.shape[0])

    plt.figure(figsize=(10, 5.5))
    plt.plot(
        time,
        truth_next[:, dim],
        label="true next state",
    )
    plt.plot(
        time,
        mean[:, dim],
        label="predicted mean",
    )
    plt.fill_between(
        time,
        lower,
        upper,
        alpha=0.25,
        label="predicted ±2σ",
    )
    plt.xlabel("transition index")
    plt.ylabel(STATE_LABELS[dim])
    plt.title(
        f"Agent {agent_id} one-step uncertainty: "
        f"{mode}, {STATE_NAMES[dim]}"
    )
    plt.grid(True)
    plt.legend()
    save_figure(output, dpi)


def print_summary(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    print("\nOpen-loop summary:")
    for mode in PREDICTION_MODES:
        print(f"\n{mode}")
        for agent_id in range(truth.shape[1]):
            position_error = np.linalg.norm(
                predictions[mode][:, agent_id, :2]
                - truth[:, agent_id, :2],
                axis=-1,
            )
            heading_error = wrapped_heading_error(
                predictions[mode][:, agent_id, 2],
                truth[:, agent_id, 2],
            )
            print(
                f"  agent_{agent_id}: "
                f"final position error={position_error[-1]:.6f} | "
                f"mean position error={position_error.mean():.6f} | "
                f"final heading error={abs(heading_error[-1]):.6f}"
            )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    local_states = dataset["local_states"]
    actions = dataset["actions"]

    saved_params, standardizers, checkpoint_metadata = (
        load_checkpoint(args.checkpoint)
    )
    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)

    if metadata.get("model_kind") not in (
        None,
        "structured_diff_drive_delta_velocity_gaussian",
    ):
        raise ValueError(
            "This script expects a structured diff-drive checkpoint, "
            f"got model_kind={metadata.get('model_kind')!r}."
        )

    num_agents = int(local_states.shape[2])
    cfg = make_model_config(metadata)
    _, initialized_states = init_structured_transition_models(
        rng=jax.random.PRNGKey(args.seed),
        num_agents=num_agents,
        cfg=cfg,
    )
    model_states = restore_model_states(
        initialized_states,
        saved_params,
    )

    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, "
            f"got {len(standardizers)}."
        )

    trajectory_id = args.trajectory_index
    if trajectory_id < 0:
        if len(dataset["test_indices"]) == 0:
            raise ValueError("Test split is empty.")
        trajectory_id = int(dataset["test_indices"][0])

    horizon = min(args.horizon, actions.shape[1])
    truth = local_states[
        trajectory_id,
        : horizon + 1,
    ]
    action_sequence = actions[
        trajectory_id,
        :horizon,
    ]

    predictions_by_mode: dict[str, np.ndarray] = {}
    one_step_by_agent: list[
        dict[str, tuple[np.ndarray, np.ndarray]]
    ] = []

    for agent_id in range(num_agents):
        one_step_by_agent.append(
            one_step_predictions(
                model_state=model_states[agent_id],
                standardizer=standardizers[agent_id],
                states=truth[:-1, agent_id],
                actions=action_sequence[:, agent_id],
                metadata=metadata,
            )
        )

    for mode in PREDICTION_MODES:
        per_agent = []
        for agent_id in range(num_agents):
            per_agent.append(
                open_loop_rollout(
                    model_state=model_states[agent_id],
                    standardizer=standardizers[agent_id],
                    initial_state=truth[0, agent_id],
                    actions=action_sequence[:, agent_id],
                    metadata=metadata,
                    mode=mode,
                )
            )
        predictions_by_mode[mode] = np.stack(
            per_agent,
            axis=1,
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    for mode in PREDICTION_MODES:
        plot_xy(
            truth,
            predictions_by_mode,
            mode,
            args.output_directory / f"open_loop_xy_{mode}.png",
            args.dpi,
        )

    plot_position_error_comparison(
        truth,
        predictions_by_mode,
        args.output_directory
        / "open_loop_position_error_comparison.png",
        args.dpi,
    )

    for agent_id in range(num_agents):
        agent_dir = args.output_directory / f"agent_{agent_id}"

        for dim in range(5):
            plot_state_component_comparison(
                truth,
                predictions_by_mode,
                agent_id,
                dim,
                agent_dir
                / f"open_loop_{STATE_NAMES[dim]}_comparison.png",
                args.dpi,
            )

            for mode in PREDICTION_MODES:
                mean, covariance = one_step_by_agent[agent_id][mode]
                plot_one_step_uncertainty(
                    truth_next=truth[1:, agent_id],
                    mean=mean,
                    covariance=covariance,
                    agent_id=agent_id,
                    dim=dim,
                    mode=mode,
                    output=agent_dir
                    / f"one_step_uncertainty_{mode}_"
                    f"{STATE_NAMES[dim]}.png",
                    dpi=args.dpi,
                )

    np.savez_compressed(
        args.output_directory / "structured_predictions.npz",
        trajectory_index=np.asarray(trajectory_id),
        truth=truth,
        actions=action_sequence,
        ensemble_mean_open_loop=(
            predictions_by_mode["ensemble_mean"]
        ),
        infoprop_ci_open_loop=(
            predictions_by_mode["infoprop_ci"]
        ),
        ensemble_mean_one_step=np.stack(
            [
                item["ensemble_mean"][0]
                for item in one_step_by_agent
            ],
            axis=1,
        ),
        ensemble_mean_one_step_covariance=np.stack(
            [
                item["ensemble_mean"][1]
                for item in one_step_by_agent
            ],
            axis=1,
        ),
        infoprop_ci_one_step=np.stack(
            [
                item["infoprop_ci"][0]
                for item in one_step_by_agent
            ],
            axis=1,
        ),
        infoprop_ci_one_step_covariance=np.stack(
            [
                item["infoprop_ci"][1]
                for item in one_step_by_agent
            ],
            axis=1,
        ),
    )

    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Trajectory index:", trajectory_id)
    print("Horizon:", horizon)
    print("Plots saved to:", args.output_directory)
    print_summary(truth, predictions_by_mode)


if __name__ == "__main__":
    main()
