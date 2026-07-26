#!/usr/bin/env python3
"""Evaluate sender-triggered common-belief estimation for diff-drive agents.

This evaluator reuses an already trained structured probabilistic dynamics
checkpoint and an already generated trajectory dataset. It does not retrain the
model and it does not regenerate physical trajectories.

At each step:

1. Propagate one common Gaussian belief with the learned dynamics model.
2. Every agent privately observes its own realized local state.
3. Every agent evaluates the same sender-side trigger against its own block of
   the predicted common belief:

       position error > error threshold
       OR
       position uncertainty radius > covariance threshold

4. All trigger decisions are evaluated from the same pre-broadcast belief.
5. Triggered agents broadcast their complete 5D local state and the common
   belief is corrected with the existing Joseph-form measurement update.

Unlike the older receiver-centric baseline, no agent writes its private ego
observation into the recursively propagated common belief unless it broadcasts.
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
from matplotlib.lines import Line2D

from aorpo.agents.structured_diff_drive_dynamics import (
    init_structured_transition_models,
    predict_structured_next,
)
from aorpo.estimator.global_belief import (
    direct_agent_observation_update,
    extract_agent_covariance,
    initialize_global_belief,
    make_isotropic_covariance,
)
from aorpo.estimator.structured_global_belief import (
    all_agent_position_uncertainty_radii,
    predict_structured_global_belief_oracle_actions,
)
from aorpo.estimator.trigger import sender_position_trigger


PREDICTION_MODES = ("ensemble_mean", "infoprop_ci")
AGENT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
)
TRUE_COLOR = "#111111"
MODEL_COLOR = "#E69F00"
ESTIMATOR_COLOR = "#0072B2"
START_COLOR = "#009E73"
END_COLOR = "#D55E00"
COMMUNICATION_COLOR = "#CC79A7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/structured_diff_drive_dynamics_noise005.pkl"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_noise005.npz"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/figures/structured_sender_position_trigger"
        ),
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=-1,
        help="-1 selects the first trajectory from the saved test split.",
    )
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--sender-error-threshold",
        type=float,
        default=0.10,
        help="Sender-side Euclidean position-error threshold.",
    )
    parser.add_argument(
        "--sender-covariance-radius-threshold",
        type=float,
        default=0.30,
        help="Sender-side position uncertainty-radius threshold.",
    )
    parser.add_argument(
        "--sender-covariance-scale",
        type=float,
        default=2.0,
        help="Sigma multiplier used for the position uncertainty radius.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="ensemble_mean",
    )
    parser.add_argument(
        "--epistemic-process-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--initial-variance", type=float, default=1.0e-6)
    parser.add_argument(
        "--measurement-variance",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        result = {
            "local_states": np.asarray(data["local_states"]),
            "actions": np.asarray(data["actions"]),
            "test_indices": np.asarray(data["test_trajectory_indices"]),
            "metadata": json.loads(str(data["metadata_json"])),
        }

    states = result["local_states"]
    actions = result["actions"]

    if states.ndim != 4 or states.shape[-1] != 5:
        raise ValueError("Expected local_states shape (K, H+1, N, 5).")
    if actions.ndim != 4 or actions.shape[-1] != 2:
        raise ValueError("Expected actions shape (K, H, N, 2).")
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError("There must be one more state than action.")

    return result


def parse_checkpoint_object(
    loaded: Any,
) -> tuple[Any, list[Any], dict[str, Any]]:
    if isinstance(loaded, dict):
        saved_models = None
        for key in (
            "model_params",
            "saved_model_params",
            "params",
            "model_states",
        ):
            if key in loaded:
                saved_models = loaded[key]
                break

        standardizers = None
        for key in ("standardizers", "local_standardizers"):
            if key in loaded:
                standardizers = loaded[key]
                break

        metadata = loaded.get("metadata", {})
        if saved_models is None or standardizers is None:
            raise KeyError(
                "Could not locate model parameters and standardizers. "
                f"Checkpoint keys: {sorted(loaded.keys())}"
            )
        return saved_models, list(standardizers), dict(metadata)

    if isinstance(loaded, (tuple, list)) and len(loaded) == 3:
        saved_models, standardizers, metadata = loaded
        return saved_models, list(standardizers), dict(metadata)

    raise TypeError("Unsupported checkpoint layout.")


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
            hidden_dims=tuple(metadata.get("hidden_dims", [256, 256])),
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

        return list(
            restore_independent_model_states(initialized_states, saved)
        )
    except Exception:
        saved_items = list(saved)
        if len(saved_items) != len(initialized_states):
            raise ValueError(
                "Saved model count does not match initialized model count: "
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


def dynamics_kwargs(metadata: dict[str, Any]) -> dict[str, float]:
    return {
        "dt": float(metadata["dt"]),
        "min_v": float(metadata["min_v"]),
        "max_v": float(metadata["max_v"]),
        "max_omega": float(metadata["max_omega"]),
    }


def model_only_rollout(
    initial_states: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    prediction_mode: str,
    epistemic_process_scale: float,
) -> np.ndarray:
    num_agents = initial_states.shape[0]
    current = jnp.asarray(initial_states)
    trajectory = [np.asarray(current)]

    for action_t in actions:
        next_states = []
        for agent_id in range(num_agents):
            next_state, _ = predict_structured_next(
                train_state=model_states[agent_id],
                standardizer=standardizers[agent_id],
                local_state=current[agent_id:agent_id + 1],
                local_action=jnp.asarray(
                    action_t[agent_id:agent_id + 1]
                ),
                prediction_mode=prediction_mode,
                epistemic_process_scale=epistemic_process_scale,
                **dynamics_kwargs(metadata),
            )
            next_states.append(next_state[0])

        current = jnp.stack(next_states, axis=0)
        trajectory.append(np.asarray(current))

    return np.stack(trajectory, axis=0)


def reshape_belief_mean(
    belief: Any,
    num_agents: int,
    local_state_dim: int,
) -> np.ndarray:
    return np.asarray(
        belief.mean.reshape(1, num_agents, local_state_dim)[0]
    )


def position_errors(
    means: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(means[..., :2] - truth[..., :2], axis=-1)


def run_sender_trigger_estimator(
    truth: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    sender_error_threshold: float,
    sender_covariance_radius_threshold: float,
    sender_covariance_scale: float,
    prediction_mode: str,
    epistemic_process_scale: float,
    initial_variance: float,
    measurement_variance: float,
) -> dict[str, np.ndarray]:
    num_agents = truth.shape[1]
    local_state_dim = truth.shape[2]

    belief = initialize_global_belief(
        initial_local_states=jnp.asarray(truth[0:1]),
        initial_variance=initial_variance,
    )
    measurement_covariance = make_isotropic_covariance(
        batch_size=1,
        dimension=local_state_dim,
        variance=measurement_variance,
        dtype=jnp.asarray(truth).dtype,
    )

    predicted_means = []
    posterior_means = []
    predicted_covariances = []
    posterior_covariances = []
    predicted_position_errors = []
    posterior_position_errors = []
    predicted_position_radii = []
    posterior_position_radii = []
    communication_decisions = []

    for step_index in range(actions.shape[0]):
        predicted_belief, _ = (
            predict_structured_global_belief_oracle_actions(
                belief=belief,
                model_states=model_states,
                standardizers=standardizers,
                joint_actions=jnp.asarray(
                    actions[step_index:step_index + 1]
                ),
                num_agents=num_agents,
                prediction_mode=prediction_mode,
                epistemic_process_scale=epistemic_process_scale,
                **dynamics_kwargs(metadata),
            )
        )

        true_next = jnp.asarray(truth[step_index + 1])
        predicted_local_means = reshape_belief_mean(
            predicted_belief,
            num_agents,
            local_state_dim,
        )
        predicted_radii = np.asarray(
            all_agent_position_uncertainty_radii(
                belief=predicted_belief,
                num_agents=num_agents,
                scale=sender_covariance_scale,
            )[0]
        )
        predicted_errors = position_errors(
            predicted_local_means,
            np.asarray(true_next),
        )

        decisions = np.zeros((num_agents,), dtype=np.bool_)

        # Every sender evaluates its decision from the same predicted common
        # belief, before any broadcast correction is applied.
        for sender_agent_id in range(num_agents):
            sender_covariance = extract_agent_covariance(
                belief=predicted_belief,
                agent_id=sender_agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )
            decisions[sender_agent_id] = bool(
                sender_position_trigger(
                    predicted_local_mean=jnp.asarray(
                        predicted_local_means[
                            sender_agent_id:sender_agent_id + 1
                        ]
                    ),
                    observed_local_state=true_next[
                        sender_agent_id:sender_agent_id + 1
                    ],
                    predicted_local_covariance=sender_covariance,
                    error_threshold=sender_error_threshold,
                    covariance_radius_threshold=(
                        sender_covariance_radius_threshold
                    ),
                    covariance_scale=sender_covariance_scale,
                )[0]
            )

        belief = predicted_belief
        for sender_agent_id in range(num_agents):
            if not decisions[sender_agent_id]:
                continue

            belief, _ = direct_agent_observation_update(
                predicted_belief=belief,
                observed_agent_id=sender_agent_id,
                measurement=true_next[
                    sender_agent_id:sender_agent_id + 1
                ],
                measurement_covariance=measurement_covariance,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

        posterior_local_means = reshape_belief_mean(
            belief,
            num_agents,
            local_state_dim,
        )
        posterior_radii = np.asarray(
            all_agent_position_uncertainty_radii(
                belief=belief,
                num_agents=num_agents,
                scale=sender_covariance_scale,
            )[0]
        )
        posterior_errors = position_errors(
            posterior_local_means,
            np.asarray(true_next),
        )

        predicted_means.append(predicted_local_means)
        posterior_means.append(posterior_local_means)
        predicted_covariances.append(
            np.asarray(predicted_belief.covariance[0])
        )
        posterior_covariances.append(np.asarray(belief.covariance[0]))
        predicted_position_errors.append(predicted_errors)
        posterior_position_errors.append(posterior_errors)
        predicted_position_radii.append(predicted_radii)
        posterior_position_radii.append(posterior_radii)
        communication_decisions.append(decisions)

    return {
        "predicted_means": np.stack(predicted_means),
        "posterior_means": np.stack(posterior_means),
        "predicted_covariances": np.stack(predicted_covariances),
        "posterior_covariances": np.stack(posterior_covariances),
        "predicted_position_errors": np.stack(
            predicted_position_errors
        ),
        "posterior_position_errors": np.stack(
            posterior_position_errors
        ),
        "predicted_position_radii": np.stack(
            predicted_position_radii
        ),
        "posterior_position_radii": np.stack(
            posterior_position_radii
        ),
        "communications": np.stack(communication_decisions),
    }


def save_figure(path: Path, figure: plt.Figure, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_all_agent_routes(
    truth: np.ndarray,
    model_only: np.ndarray,
    posterior: np.ndarray,
    communications: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 8.5))

    for agent_id in range(truth.shape[1]):
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        axis.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            color=color,
            linewidth=2.7,
            linestyle="-",
        )
        axis.plot(
            model_only[:, agent_id, 0],
            model_only[:, agent_id, 1],
            color=color,
            linewidth=1.9,
            linestyle=":",
            alpha=0.75,
        )
        axis.plot(
            posterior[:, agent_id, 0],
            posterior[:, agent_id, 1],
            color=color,
            linewidth=2.2,
            linestyle="--",
            alpha=0.95,
        )

        axis.scatter(
            truth[0, agent_id, 0],
            truth[0, agent_id, 1],
            s=85,
            marker="o",
            facecolor=START_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )
        axis.scatter(
            truth[-1, agent_id, 0],
            truth[-1, agent_id, 1],
            s=105,
            marker="X",
            facecolor=END_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )

        event_steps = np.flatnonzero(communications[:, agent_id]) + 1
        if event_steps.size:
            axis.scatter(
                posterior[event_steps, agent_id, 0],
                posterior[event_steps, agent_id, 1],
                s=48,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.4,
                zorder=8,
            )

    agent_handles = [
        Line2D(
            [0],
            [0],
            color=AGENT_COLORS[i % len(AGENT_COLORS)],
            linewidth=3.0,
            label=f"agent {i}",
        )
        for i in range(truth.shape[1])
    ]
    meaning_handles = [
        Line2D(
            [0], [0], color=TRUE_COLOR, linewidth=2.7,
            linestyle="-", label="true state"
        ),
        Line2D(
            [0], [0], color=TRUE_COLOR, linewidth=1.9,
            linestyle=":", label="model-only open-loop"
        ),
        Line2D(
            [0], [0], color=TRUE_COLOR, linewidth=2.2,
            linestyle="--", label="common-belief posterior"
        ),
        Line2D(
            [0], [0], color=COMMUNICATION_COLOR, marker="D",
            markerfacecolor="none", linestyle="None",
            label="sender broadcast"
        ),
        Line2D(
            [0], [0], color=START_COLOR, marker="o",
            linestyle="None", label="start"
        ),
        Line2D(
            [0], [0], color=END_COLOR, marker="X",
            linestyle="None", label="end"
        ),
    ]

    first_legend = axis.legend(
        handles=agent_handles,
        loc="upper left",
        title="Agent color",
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=meaning_handles,
        loc="lower right",
        title="Trajectory meaning",
    )
    axis.set_title(
        "Sender-triggered common belief: all-agent trajectories"
    )
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.32)
    save_figure(output, figure, dpi)


def plot_agent_panels(
    truth: np.ndarray,
    model_only: np.ndarray,
    posterior: np.ndarray,
    communications: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    num_agents = truth.shape[1]
    figure, axes = plt.subplots(
        1,
        num_agents,
        figsize=(6.0 * num_agents, 5.8),
        squeeze=False,
    )

    for agent_id, axis in enumerate(axes[0]):
        axis.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            color=TRUE_COLOR,
            linewidth=2.6,
            label="true state",
        )
        axis.plot(
            model_only[:, agent_id, 0],
            model_only[:, agent_id, 1],
            color=MODEL_COLOR,
            linewidth=2.0,
            linestyle=":",
            label="model-only",
        )
        axis.plot(
            posterior[:, agent_id, 0],
            posterior[:, agent_id, 1],
            color=ESTIMATOR_COLOR,
            linewidth=2.2,
            linestyle="--",
            label="common posterior",
        )
        axis.scatter(
            truth[0, agent_id, 0],
            truth[0, agent_id, 1],
            s=85,
            marker="o",
            facecolor=START_COLOR,
            edgecolor="white",
            label="start",
            zorder=7,
        )
        axis.scatter(
            truth[-1, agent_id, 0],
            truth[-1, agent_id, 1],
            s=105,
            marker="X",
            facecolor=END_COLOR,
            edgecolor="white",
            label="end",
            zorder=7,
        )
        event_steps = np.flatnonzero(communications[:, agent_id]) + 1
        if event_steps.size:
            axis.scatter(
                posterior[event_steps, agent_id, 0],
                posterior[event_steps, agent_id, 1],
                s=52,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.4,
                label="broadcast",
                zorder=8,
            )

        axis.set_title(f"Agent {agent_id}: sender-triggered")
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.32)
        axis.legend(fontsize=8.5)

    figure.suptitle(
        "True, open-loop, and sender-triggered common-belief routes",
        fontsize=15,
    )
    save_figure(output, figure, dpi)


def plot_trigger_signals(
    predicted_errors: np.ndarray,
    posterior_errors: np.ndarray,
    predicted_radii: np.ndarray,
    posterior_radii: np.ndarray,
    communications: np.ndarray,
    error_threshold: float,
    covariance_threshold: float,
    covariance_scale: float,
    output: Path,
    dpi: int,
) -> None:
    horizon, num_agents = predicted_errors.shape
    time = np.arange(1, horizon + 1)
    figure, (radius_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for agent_id in range(num_agents):
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        radius_axis.plot(
            time,
            predicted_radii[:, agent_id],
            color=color,
            linewidth=2.1,
            label=f"agent {agent_id} pre-broadcast",
        )
        radius_axis.plot(
            time,
            posterior_radii[:, agent_id],
            color=color,
            linewidth=1.7,
            linestyle="--",
            alpha=0.80,
            label=f"agent {agent_id} posterior",
        )
        error_axis.plot(
            time,
            predicted_errors[:, agent_id],
            color=color,
            linewidth=2.1,
            label=f"agent {agent_id} pre-broadcast",
        )
        error_axis.plot(
            time,
            posterior_errors[:, agent_id],
            color=color,
            linewidth=1.7,
            linestyle="--",
            alpha=0.80,
            label=f"agent {agent_id} posterior",
        )

        event_mask = communications[:, agent_id]
        if np.any(event_mask):
            radius_axis.scatter(
                time[event_mask],
                predicted_radii[event_mask, agent_id],
                s=42,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=6,
            )
            error_axis.scatter(
                time[event_mask],
                predicted_errors[event_mask, agent_id],
                s=42,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=6,
            )

    radius_axis.axhline(
        covariance_threshold,
        color=END_COLOR,
        linestyle=":",
        linewidth=2.0,
        label=f"radius threshold = {covariance_threshold:g}",
    )
    error_axis.axhline(
        error_threshold,
        color=END_COLOR,
        linestyle=":",
        linewidth=2.0,
        label=f"error threshold = {error_threshold:g}",
    )
    radius_axis.set_ylabel(
        f"position uncertainty radius ({covariance_scale:g}σ)"
    )
    radius_axis.set_title(
        "Sender trigger signal 1: common-belief position uncertainty"
    )
    error_axis.set_xlabel("time step")
    error_axis.set_ylabel("position prediction error")
    error_axis.set_title(
        "Sender trigger signal 2: private observation residual"
    )

    for axis in (radius_axis, error_axis):
        axis.grid(True, alpha=0.32)
        axis.legend(ncol=2, fontsize=8.5)

    save_figure(output, figure, dpi)


def print_summary(
    truth: np.ndarray,
    model_only: np.ndarray,
    posterior: np.ndarray,
    communications: np.ndarray,
) -> None:
    horizon = communications.shape[0]
    total_messages = int(communications.sum())
    maximum_messages = horizon * communications.shape[1]

    print("\n===== Sender-trigger common-belief summary =====")
    for agent_id in range(truth.shape[1]):
        model_error = np.linalg.norm(
            model_only[:, agent_id, :2] - truth[:, agent_id, :2],
            axis=-1,
        )
        posterior_error = np.linalg.norm(
            posterior[:, agent_id, :2] - truth[:, agent_id, :2],
            axis=-1,
        )
        messages = int(communications[:, agent_id].sum())
        print(
            f"agent_{agent_id}: "
            f"model mean pos error={model_error.mean():.6f} | "
            f"common posterior mean pos error="
            f"{posterior_error.mean():.6f} | "
            f"messages={messages}/{horizon}"
        )

    print(
        f"Broadcasts: {total_messages}/{maximum_messages} | "
        f"broadcast rate={total_messages / maximum_messages:.4f}"
    )


def main() -> None:
    args = parse_args()

    if args.sender_error_threshold < 0.0:
        raise ValueError("sender-error-threshold must be non-negative.")
    if args.sender_covariance_radius_threshold < 0.0:
        raise ValueError(
            "sender-covariance-radius-threshold must be non-negative."
        )
    if args.sender_covariance_scale <= 0.0:
        raise ValueError("sender-covariance-scale must be positive.")
    if args.epistemic_process_scale < 0.0:
        raise ValueError("epistemic-process-scale must be non-negative.")

    dataset = load_dataset(args.dataset)
    saved_models, standardizers, checkpoint_metadata = load_checkpoint(
        args.checkpoint
    )

    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)

    local_states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(local_states.shape[2])

    cfg = make_model_config(metadata)
    _, initialized_states = init_structured_transition_models(
        rng=jax.random.PRNGKey(args.seed),
        num_agents=num_agents,
        cfg=cfg,
    )
    model_states = restore_model_states(
        initialized_states,
        saved_models,
    )

    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, "
            f"got {len(standardizers)}."
        )

    trajectory_id = args.trajectory_index
    if trajectory_id < 0:
        test_indices = dataset["test_indices"]
        if len(test_indices) == 0:
            raise ValueError("Dataset test split is empty.")
        trajectory_id = int(test_indices[0])

    if not 0 <= trajectory_id < local_states.shape[0]:
        raise ValueError(
            f"trajectory-index {trajectory_id} is out of range."
        )

    horizon = min(args.horizon, actions.shape[1])
    truth = local_states[trajectory_id, :horizon + 1]
    action_sequence = actions[trajectory_id, :horizon]

    model_only = model_only_rollout(
        initial_states=truth[0],
        actions=action_sequence,
        model_states=model_states,
        standardizers=standardizers,
        metadata=metadata,
        prediction_mode=args.prediction_mode,
        epistemic_process_scale=args.epistemic_process_scale,
    )

    trace = run_sender_trigger_estimator(
        truth=truth,
        actions=action_sequence,
        model_states=model_states,
        standardizers=standardizers,
        metadata=metadata,
        sender_error_threshold=args.sender_error_threshold,
        sender_covariance_radius_threshold=(
            args.sender_covariance_radius_threshold
        ),
        sender_covariance_scale=args.sender_covariance_scale,
        prediction_mode=args.prediction_mode,
        epistemic_process_scale=args.epistemic_process_scale,
        initial_variance=args.initial_variance,
        measurement_variance=args.measurement_variance,
    )

    posterior_with_initial = np.concatenate(
        [truth[0:1], trace["posterior_means"]],
        axis=0,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    plot_all_agent_routes(
        truth=truth,
        model_only=model_only,
        posterior=posterior_with_initial,
        communications=trace["communications"],
        output=args.output_directory / "all_agents_sender_routes.png",
        dpi=args.dpi,
    )
    plot_agent_panels(
        truth=truth,
        model_only=model_only,
        posterior=posterior_with_initial,
        communications=trace["communications"],
        output=args.output_directory / "sender_routes_by_agent.png",
        dpi=args.dpi,
    )
    plot_trigger_signals(
        predicted_errors=trace["predicted_position_errors"],
        posterior_errors=trace["posterior_position_errors"],
        predicted_radii=trace["predicted_position_radii"],
        posterior_radii=trace["posterior_position_radii"],
        communications=trace["communications"],
        error_threshold=args.sender_error_threshold,
        covariance_threshold=(
            args.sender_covariance_radius_threshold
        ),
        covariance_scale=args.sender_covariance_scale,
        output=args.output_directory / "sender_trigger_signals.png",
        dpi=args.dpi,
    )

    np.savez_compressed(
        args.output_directory / "sender_trigger_estimator_trace.npz",
        trajectory_index=np.asarray(trajectory_id),
        communication_mode=np.asarray("sender_position_triggered"),
        prediction_mode=np.asarray(args.prediction_mode),
        sender_error_threshold=np.asarray(
            args.sender_error_threshold
        ),
        sender_covariance_radius_threshold=np.asarray(
            args.sender_covariance_radius_threshold
        ),
        sender_covariance_scale=np.asarray(
            args.sender_covariance_scale
        ),
        truth=truth,
        actions=action_sequence,
        model_only=model_only,
        estimator_prior=trace["predicted_means"],
        estimator_posterior=posterior_with_initial,
        prior_covariances=trace["predicted_covariances"],
        posterior_covariances=trace["posterior_covariances"],
        pre_message_position_radii=trace[
            "predicted_position_radii"
        ],
        post_message_position_radii=trace[
            "posterior_position_radii"
        ],
        pre_message_position_errors=trace[
            "predicted_position_errors"
        ],
        post_message_position_errors=trace[
            "posterior_position_errors"
        ],
        communications=trace["communications"],
    )

    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Trajectory index:", trajectory_id)
    print("Prediction mode:", args.prediction_mode)
    print(
        "Sender trigger: position error > "
        f"{args.sender_error_threshold:g} OR "
        f"{args.sender_covariance_scale:g}σ radius > "
        f"{args.sender_covariance_radius_threshold:g}"
    )
    print("Outputs:", args.output_directory)
    print_summary(
        truth=truth,
        model_only=model_only,
        posterior=posterior_with_initial,
        communications=trace["communications"],
    )


if __name__ == "__main__":
    main()
