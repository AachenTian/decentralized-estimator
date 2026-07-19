#!/usr/bin/env python3
"""Evaluate covariance-triggered correction for structured diff-drive dynamics.

Three trajectory estimates are compared:

1. True state.
2. Model-only open-loop prediction.
3. Global-belief posterior with position-covariance-triggered communication.

For each remote agent j, the communication rule is:

    r_j = scale * sqrt(lambda_max(P_j[position, position]))

    communicate if r_j > radius_threshold

The trigger uses only the receiver's belief covariance. It does not use the
unknown true prediction error.

The current baseline sends the complete 5D local state after a trigger, while
the trigger statistic itself uses only the 2D position covariance.
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
from aorpo.estimator.trigger import (
    position_radius_threshold_trigger,
)


PREDICTION_MODES = ("ensemble_mean", "infoprop_ci")

# Okabe-Ito-inspired, color-blind-friendly palette.
TRUE_COLOR = "#111111"
MODEL_COLOR = "#E69F00"
ESTIMATOR_COLOR = "#0072B2"
START_COLOR = "#009E73"
END_COLOR = "#D55E00"
COMMUNICATION_COLOR = "#CC79A7"
REMOTE_AGENT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
)


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
            "results/figures/"
            "structured_covariance_trigger_estimator"
        ),
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=-1,
        help="-1 selects the first trajectory in the saved test split.",
    )
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--ego-agent-id", type=int, default=0)

    parser.add_argument(
        "--position-trigger-radius",
        type=float,
        default=0.10,
        help="Trigger threshold in environment position units.",
    )
    parser.add_argument(
        "--position-trigger-scale",
        type=float,
        default=2.0,
        help="Standard-deviation multiplier for the position ellipse.",
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
        default=1.0e-5,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-ego-model-only",
        action="store_true",
        help=(
            "Show the counterfactual open-loop model-only curve for the "
            "ego agent. By default it is hidden because the actual ego "
            "posterior is corrected from its observation at every step."
        ),
    )
    parser.add_argument("--dpi", type=int, default=170)
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

    if states.ndim != 4 or states.shape[-1] != 5:
        raise ValueError(
            "Expected local_states shape (K, H+1, N, 5)."
        )
    if actions.ndim != 4 or actions.shape[-1] != 2:
        raise ValueError(
            "Expected actions shape (K, H, N, 2)."
        )
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

    raise TypeError(
        "Unsupported checkpoint layout. Expected dict or 3-tuple."
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
                local_state=current[
                    agent_id:agent_id + 1
                ],
                local_action=jnp.asarray(
                    action_t[
                        agent_id:agent_id + 1
                    ]
                ),
                prediction_mode=prediction_mode,
                epistemic_process_scale=epistemic_process_scale,
                **dynamics_kwargs(metadata),
            )
            next_states.append(next_state[0])

        current = jnp.stack(next_states, axis=0)
        trajectory.append(np.asarray(current))

    return np.stack(trajectory, axis=0)


def run_covariance_trigger_estimator(
    truth: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    ego_agent_id: int,
    position_trigger_radius: float,
    position_trigger_scale: float,
    prediction_mode: str,
    epistemic_process_scale: float,
    initial_variance: float,
    measurement_variance: float,
) -> dict[str, np.ndarray]:
    """Run the covariance-triggered global Gaussian estimator."""
    num_agents = truth.shape[1]
    local_state_dim = truth.shape[2]
    remote_agent_ids = [
        agent_id
        for agent_id in range(num_agents)
        if agent_id != ego_agent_id
    ]

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

    prior_means = []
    after_ego_means = []
    posterior_means = []

    prior_covariances = []
    posterior_covariances = []

    pre_message_radii = []
    post_message_radii = []
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

        # The ego receiver directly observes its own state every step.
        belief_after_ego, _ = direct_agent_observation_update(
            predicted_belief=predicted_belief,
            observed_agent_id=ego_agent_id,
            measurement=true_next[
                ego_agent_id:ego_agent_id + 1
            ],
            measurement_covariance=measurement_covariance,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
        )

        # All remote decisions use the same pre-message belief.
        pre_remote_belief = belief_after_ego
        belief = belief_after_ego

        pre_radii = all_agent_position_uncertainty_radii(
            belief=pre_remote_belief,
            num_agents=num_agents,
            scale=position_trigger_scale,
        )[0]

        decisions = np.zeros((num_agents,), dtype=np.bool_)

        for remote_agent_id in remote_agent_ids:
            remote_covariance = extract_agent_covariance(
                belief=pre_remote_belief,
                agent_id=remote_agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

            decisions[remote_agent_id] = bool(
                position_radius_threshold_trigger(
                    covariance=remote_covariance,
                    radius_threshold=position_trigger_radius,
                    scale=position_trigger_scale,
                )[0]
            )

        for remote_agent_id in remote_agent_ids:
            if not decisions[remote_agent_id]:
                continue

            # Baseline payload: complete local state after a position-only
            # covariance trigger.
            belief, _ = direct_agent_observation_update(
                predicted_belief=belief,
                observed_agent_id=remote_agent_id,
                measurement=true_next[
                    remote_agent_id:remote_agent_id + 1
                ],
                measurement_covariance=measurement_covariance,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

        post_radii = all_agent_position_uncertainty_radii(
            belief=belief,
            num_agents=num_agents,
            scale=position_trigger_scale,
        )[0]

        prior_means.append(
            np.asarray(
                predicted_belief.mean.reshape(
                    1,
                    num_agents,
                    local_state_dim,
                )[0]
            )
        )
        after_ego_means.append(
            np.asarray(
                belief_after_ego.mean.reshape(
                    1,
                    num_agents,
                    local_state_dim,
                )[0]
            )
        )
        posterior_means.append(
            np.asarray(
                belief.mean.reshape(
                    1,
                    num_agents,
                    local_state_dim,
                )[0]
            )
        )

        prior_covariances.append(
            np.asarray(predicted_belief.covariance[0])
        )
        posterior_covariances.append(
            np.asarray(belief.covariance[0])
        )

        pre_message_radii.append(np.asarray(pre_radii))
        post_message_radii.append(np.asarray(post_radii))
        communication_decisions.append(decisions)

    return {
        "prior_means": np.stack(prior_means, axis=0),
        "after_ego_means": np.stack(after_ego_means, axis=0),
        "posterior_means": np.stack(posterior_means, axis=0),
        "prior_covariances": np.stack(
            prior_covariances,
            axis=0,
        ),
        "posterior_covariances": np.stack(
            posterior_covariances,
            axis=0,
        ),
        "pre_message_position_radii": np.stack(
            pre_message_radii,
            axis=0,
        ),
        "post_message_position_radii": np.stack(
            post_message_radii,
            axis=0,
        ),
        "communications": np.stack(
            communication_decisions,
            axis=0,
        ),
    }


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()



def plot_xy_three_way(
    truth: np.ndarray,
    model_only: np.ndarray,
    estimator: np.ndarray,
    communications: np.ndarray,
    ego_agent_id: int,
    show_ego_model_only: bool,
    output: Path,
    dpi: int,
) -> None:
    """Plot one clear XY panel per agent."""
    num_agents = truth.shape[1]
    figure, axes = plt.subplots(
        1,
        num_agents,
        figsize=(6.0 * num_agents, 5.8),
        squeeze=False,
    )
    axes = axes[0]

    for agent_id, axis in enumerate(axes):
        is_ego = agent_id == ego_agent_id
        role = "ego" if is_ego else "remote"

        axis.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            color=TRUE_COLOR,
            linewidth=2.6,
            linestyle="-",
            label="true state",
            zorder=3,
        )

        if (not is_ego) or show_ego_model_only:
            model_label = (
                "model-only (counterfactual)"
                if is_ego
                else "model-only open-loop"
            )
            axis.plot(
                model_only[:, agent_id, 0],
                model_only[:, agent_id, 1],
                color=MODEL_COLOR,
                linewidth=2.1,
                linestyle="--",
                label=model_label,
                zorder=2,
            )

        estimator_label = (
            "posterior (ego observed each step)"
            if is_ego
            else "posterior estimator"
        )
        axis.plot(
            estimator[:, agent_id, 0],
            estimator[:, agent_id, 1],
            color=ESTIMATOR_COLOR,
            linewidth=2.2,
            linestyle="-.",
            label=estimator_label,
            zorder=4,
        )

        axis.scatter(
            truth[0, agent_id, 0],
            truth[0, agent_id, 1],
            s=95,
            marker="o",
            facecolor=START_COLOR,
            edgecolor="white",
            linewidth=1.2,
            label="start",
            zorder=7,
        )
        axis.scatter(
            truth[-1, agent_id, 0],
            truth[-1, agent_id, 1],
            s=115,
            marker="X",
            facecolor=END_COLOR,
            edgecolor="white",
            linewidth=1.2,
            label="end",
            zorder=7,
        )

        if not is_ego:
            event_steps = (
                np.flatnonzero(communications[:, agent_id]) + 1
            )
            if event_steps.size:
                axis.scatter(
                    estimator[event_steps, agent_id, 0],
                    estimator[event_steps, agent_id, 1],
                    s=58,
                    marker="D",
                    facecolor="none",
                    edgecolor=COMMUNICATION_COLOR,
                    linewidth=1.5,
                    label="communication update",
                    zorder=8,
                )

        title_suffix = (
            "observation update every step"
            if is_ego
            else "covariance-triggered updates"
        )
        axis.set_title(
            f"Agent {agent_id} ({role})\\n{title_suffix}",
            fontsize=12,
        )
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.35)
        axis.legend(fontsize=8.5, loc="best")

    figure.suptitle(
        "True state, open-loop dynamics, and corrected belief",
        fontsize=15,
    )
    save_figure(output, dpi)



def plot_remote_position_errors(
    truth: np.ndarray,
    model_only: np.ndarray,
    estimator: np.ndarray,
    communications: np.ndarray,
    ego_agent_id: int,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(11, 6))

    remote_index = 0
    for agent_id in range(truth.shape[1]):
        if agent_id == ego_agent_id:
            continue

        color = REMOTE_AGENT_COLORS[
            remote_index % len(REMOTE_AGENT_COLORS)
        ]
        remote_index += 1

        model_error = np.linalg.norm(
            model_only[:, agent_id, :2]
            - truth[:, agent_id, :2],
            axis=-1,
        )
        estimator_error = np.linalg.norm(
            estimator[:, agent_id, :2]
            - truth[:, agent_id, :2],
            axis=-1,
        )

        plt.plot(
            model_error,
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.75,
            label=f"agent {agent_id} model-only",
        )
        plt.plot(
            estimator_error,
            color=color,
            linewidth=2.3,
            linestyle="-",
            label=f"agent {agent_id} estimator",
        )

        event_steps = np.flatnonzero(
            communications[:, agent_id]
        ) + 1

        if event_steps.size:
            plt.scatter(
                event_steps,
                estimator_error[event_steps],
                s=55,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.5,
                label=f"agent {agent_id} communication",
                zorder=5,
            )

    plt.xlabel("time step")
    plt.ylabel("position error")
    plt.title("Remote-agent position error and communication updates")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    save_figure(output, dpi)



def plot_position_uncertainty_radii(
    pre_radii: np.ndarray,
    post_radii: np.ndarray,
    communications: np.ndarray,
    threshold: float,
    ego_agent_id: int,
    output: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(11, 6))
    time = np.arange(1, pre_radii.shape[0] + 1)

    remote_index = 0
    for agent_id in range(pre_radii.shape[1]):
        if agent_id == ego_agent_id:
            continue

        color = REMOTE_AGENT_COLORS[
            remote_index % len(REMOTE_AGENT_COLORS)
        ]
        remote_index += 1

        plt.plot(
            time,
            pre_radii[:, agent_id],
            color=color,
            linewidth=2.1,
            linestyle="-",
            label=f"agent {agent_id} prior radius",
        )
        plt.plot(
            time,
            post_radii[:, agent_id],
            color=color,
            linewidth=1.8,
            linestyle=":",
            alpha=0.85,
            label=f"agent {agent_id} posterior radius",
        )

        event_mask = communications[:, agent_id]

        if np.any(event_mask):
            plt.scatter(
                time[event_mask],
                pre_radii[event_mask, agent_id],
                s=55,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.5,
                label=f"agent {agent_id} trigger",
                zorder=5,
            )

    plt.axhline(
        threshold,
        color=END_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=f"trigger threshold = {threshold:g}",
    )
    plt.xlabel("time step")
    plt.ylabel("position uncertainty radius")
    plt.title("Belief position-covariance trigger signal")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    save_figure(output, dpi)


def print_summary(
    truth: np.ndarray,
    model_only: np.ndarray,
    estimator: np.ndarray,
    communications: np.ndarray,
    ego_agent_id: int,
) -> None:
    horizon = communications.shape[0]

    print("\n===== Covariance-trigger estimator summary =====")

    all_model_errors = []
    all_estimator_errors = []
    total_messages = 0
    maximum_messages = horizon * (truth.shape[1] - 1)

    for agent_id in range(truth.shape[1]):
        if agent_id == ego_agent_id:
            continue

        model_error = np.linalg.norm(
            model_only[:, agent_id, :2]
            - truth[:, agent_id, :2],
            axis=-1,
        )
        estimator_error = np.linalg.norm(
            estimator[:, agent_id, :2]
            - truth[:, agent_id, :2],
            axis=-1,
        )
        messages = int(communications[:, agent_id].sum())

        all_model_errors.append(model_error)
        all_estimator_errors.append(estimator_error)
        total_messages += messages

        print(
            f"agent_{agent_id}: "
            f"model mean pos error={model_error.mean():.6f} | "
            f"estimator mean pos error={estimator_error.mean():.6f} | "
            f"model final pos error={model_error[-1]:.6f} | "
            f"estimator final pos error={estimator_error[-1]:.6f} | "
            f"messages={messages}/{horizon}"
        )

    model_errors = np.concatenate(all_model_errors)
    estimator_errors = np.concatenate(all_estimator_errors)

    print(
        "Aggregate remote mean position error: "
        f"model-only={model_errors.mean():.6f} | "
        f"estimator={estimator_errors.mean():.6f}"
    )
    print(
        f"Remote messages: {total_messages}/{maximum_messages} | "
        f"message rate={total_messages / maximum_messages:.4f}"
    )


def main() -> None:
    args = parse_args()

    if args.position_trigger_radius < 0.0:
        raise ValueError(
            "position-trigger-radius must be non-negative."
        )
    if args.position_trigger_scale <= 0.0:
        raise ValueError(
            "position-trigger-scale must be positive."
        )
    if args.epistemic_process_scale < 0.0:
        raise ValueError(
            "epistemic-process-scale must be non-negative."
        )

    dataset = load_dataset(args.dataset)
    saved_models, standardizers, checkpoint_metadata = (
        load_checkpoint(args.checkpoint)
    )

    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)

    local_states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(local_states.shape[2])

    if not 0 <= args.ego_agent_id < num_agents:
        raise ValueError(
            f"ego-agent-id must lie in [0, {num_agents - 1}]."
        )

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

    truth = local_states[
        trajectory_id,
        : horizon + 1,
    ]
    action_sequence = actions[
        trajectory_id,
        :horizon,
    ]

    model_only = model_only_rollout(
        initial_states=truth[0],
        actions=action_sequence,
        model_states=model_states,
        standardizers=standardizers,
        metadata=metadata,
        prediction_mode=args.prediction_mode,
        epistemic_process_scale=args.epistemic_process_scale,
    )

    estimator_trace = run_covariance_trigger_estimator(
        truth=truth,
        actions=action_sequence,
        model_states=model_states,
        standardizers=standardizers,
        metadata=metadata,
        ego_agent_id=args.ego_agent_id,
        position_trigger_radius=args.position_trigger_radius,
        position_trigger_scale=args.position_trigger_scale,
        prediction_mode=args.prediction_mode,
        epistemic_process_scale=args.epistemic_process_scale,
        initial_variance=args.initial_variance,
        measurement_variance=args.measurement_variance,
    )

    estimator_with_initial = np.concatenate(
        [truth[0:1], estimator_trace["posterior_means"]],
        axis=0,
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_xy_three_way(
        truth=truth,
        model_only=model_only,
        estimator=estimator_with_initial,
        communications=estimator_trace["communications"],
        ego_agent_id=args.ego_agent_id,
        show_ego_model_only=args.show_ego_model_only,
        output=args.output_directory / "xy_three_way_by_agent.png",
        dpi=args.dpi,
    )

    plot_remote_position_errors(
        truth=truth,
        model_only=model_only,
        estimator=estimator_with_initial,
        communications=estimator_trace["communications"],
        ego_agent_id=args.ego_agent_id,
        output=(
            args.output_directory
            / "remote_position_error_comparison.png"
        ),
        dpi=args.dpi,
    )

    plot_position_uncertainty_radii(
        pre_radii=(
            estimator_trace["pre_message_position_radii"]
        ),
        post_radii=(
            estimator_trace["post_message_position_radii"]
        ),
        communications=estimator_trace["communications"],
        threshold=args.position_trigger_radius,
        ego_agent_id=args.ego_agent_id,
        output=(
            args.output_directory
            / "position_uncertainty_radius_trigger.png"
        ),
        dpi=args.dpi,
    )

    np.savez_compressed(
        args.output_directory
        / "covariance_trigger_estimator_trace.npz",
        trajectory_index=np.asarray(trajectory_id),
        ego_agent_id=np.asarray(args.ego_agent_id),
        prediction_mode=np.asarray(args.prediction_mode),
        position_trigger_radius=np.asarray(
            args.position_trigger_radius
        ),
        position_trigger_scale=np.asarray(
            args.position_trigger_scale
        ),
        truth=truth,
        actions=action_sequence,
        model_only=model_only,
        estimator_prior=estimator_trace["prior_means"],
        estimator_after_ego=(
            estimator_trace["after_ego_means"]
        ),
        estimator_posterior=estimator_with_initial,
        prior_covariances=(
            estimator_trace["prior_covariances"]
        ),
        posterior_covariances=(
            estimator_trace["posterior_covariances"]
        ),
        pre_message_position_radii=(
            estimator_trace["pre_message_position_radii"]
        ),
        post_message_position_radii=(
            estimator_trace["post_message_position_radii"]
        ),
        communications=estimator_trace["communications"],
    )

    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Trajectory index:", trajectory_id)
    print("Prediction mode:", args.prediction_mode)
    print(
        "Position trigger:",
        f"{args.position_trigger_scale:g}σ radius > "
        f"{args.position_trigger_radius:g}",
    )
    print("Plots saved to:", args.output_directory)
    print(
        "Ego semantics: the estimator first predicts agent 0 with the "
        "dynamics model, then performs a direct observation update at "
        "every step. The ego model-only curve, when shown, is only a "
        "counterfactual open-loop baseline."
    )

    print_summary(
        truth=truth,
        model_only=model_only,
        estimator=estimator_with_initial,
        communications=estimator_trace["communications"],
        ego_agent_id=args.ego_agent_id,
    )


if __name__ == "__main__":
    main()
