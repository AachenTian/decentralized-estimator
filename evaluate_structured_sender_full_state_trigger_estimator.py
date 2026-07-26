#!/usr/bin/env python3
"""Evaluate sender-triggered common-belief estimation for diff-drive agents.

This evaluator reuses an already trained structured probabilistic dynamics
checkpoint and a saved trajectory dataset. It can either evaluate the saved
trajectory directly or replay the saved action sequence from the same initial
state with configurable acceleration-level process noise. It also supports
configurable five-dimensional measurement noise for the private observations
used by the sender trigger and broadcast update. The dynamics model is never
retrained by this script.

At each step:

1. Propagate one common Gaussian belief with the learned dynamics model.
2. Every agent privately receives a noisy measurement of its realized local
   state.
3. Every agent evaluates the same sender-side trigger against its own block of
   the predicted common belief:

       normalized full-state measurement residual > error threshold
       OR
       normalized full-state uncertainty > covariance threshold

4. All trigger decisions are evaluated from the same pre-broadcast belief.
5. Triggered agents broadcast their complete noisy 5D measurement and the
   common belief is corrected with the existing Joseph-form measurement update.

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
)
from aorpo.estimator.structured_global_belief import (
    all_agent_position_uncertainty_radii,
    predict_structured_global_belief_oracle_actions,
)
from aorpo.estimator.normalized_full_state_trigger import (
    normalized_full_state_error_score,
    normalized_full_state_uncertainty_score,
    sender_normalized_full_state_trigger,
)
from aorpo.envs.accel_diff_drive_multi_agent_env import (
    AccelDiffDriveConfig,
    AccelDiffDriveMultiAgentEnv,
)


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
            "results/figures/structured_sender_full_state_trigger"
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
        "--sender-full-state-error-threshold",
        type=float,
        default=0.10,
        help=(
            "Dimensionless normalized full-state RMS-error threshold."
        ),
    )
    parser.add_argument(
        "--sender-full-state-uncertainty-threshold",
        type=float,
        default=0.10,
        help=(
            "Dimensionless normalized full-state uncertainty threshold."
        ),
    )
    parser.add_argument(
        "--sender-uncertainty-sigma-scale",
        type=float,
        default=2.0,
        help="Sigma multiplier in the normalized uncertainty score.",
    )
    parser.add_argument(
        "--state-scales",
        type=float,
        nargs=5,
        default=None,
        metavar=("PX", "PY", "PHI", "V", "OMEGA"),
        help=(
            "Positive normalization scales for "
            "[px, py, phi, v, omega]. "
            "Default: [1, 1, pi, max_abs_v, max_omega]."
        ),
    )
    parser.add_argument(
        "--position-ellipse-scale",
        type=float,
        default=2.0,
        help=(
            "Sigma multiplier used only for the 2D position ellipse "
            "in figures and animation; it does not affect triggering."
        ),
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
        "--measurement-noise-std",
        type=float,
        nargs=5,
        default=(0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("PX", "PY", "PHI", "V", "OMEGA"),
        help=(
            "Standard deviations of additive private measurement noise for "
            "[px, py, phi, v, omega]. The same diagonal covariance is used "
            "for every agent."
        ),
    )
    parser.add_argument(
        "--measurement-variance",
        type=float,
        default=1.0e-9,
        help=(
            "Non-negative diagonal variance floor added to the measurement "
            "covariance for numerical stability."
        ),
    )
    parser.add_argument(
        "--linear-acceleration-process-noise-std",
        type=float,
        default=None,
        help=(
            "If specified, replay the selected action sequence from its "
            "initial state with this acceleration-noise standard deviation. "
            "If both process-noise arguments are omitted, use the trajectory "
            "stored in the dataset without resimulation."
        ),
    )
    parser.add_argument(
        "--angular-acceleration-process-noise-std",
        type=float,
        default=None,
        help=(
            "If specified, replay the selected action sequence with this "
            "angular-acceleration-noise standard deviation."
        ),
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


def resolve_state_scales(
    requested_scales: list[float] | None,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Resolve positive scales for [px, py, phi, v, omega]."""
    if requested_scales is None:
        max_abs_v = max(
            abs(float(metadata["min_v"])),
            abs(float(metadata["max_v"])),
        )
        max_omega = abs(float(metadata["max_omega"]))
        scales = np.asarray(
            [
                1.0,
                1.0,
                np.pi,
                max(max_abs_v, 1.0e-6),
                max(max_omega, 1.0e-6),
            ],
            dtype=np.float32,
        )
    else:
        scales = np.asarray(
            requested_scales,
            dtype=np.float32,
        )

    if scales.shape != (5,):
        raise ValueError(
            "state-scales must contain exactly five values."
        )
    if np.any(scales <= 0.0):
        raise ValueError("Every state scale must be positive.")

    return scales



def resolve_measurement_noise_std(
    requested_std: tuple[float, ...] | list[float],
) -> np.ndarray:
    """Validate measurement-noise standard deviations for the 5D state."""
    std = np.asarray(requested_std, dtype=np.float32)
    if std.shape != (5,):
        raise ValueError(
            "measurement-noise-std must contain exactly five values."
        )
    if np.any(std < 0.0):
        raise ValueError(
            "Every measurement-noise standard deviation must be non-negative."
        )
    return std


def diagonal_measurement_covariance(
    measurement_noise_std: np.ndarray,
    variance_floor: float,
    dtype: Any,
) -> jnp.ndarray:
    """Return R with shape (1, 5, 5)."""
    if variance_floor < 0.0:
        raise ValueError("measurement-variance must be non-negative.")
    variances = measurement_noise_std.astype(np.float64) ** 2
    variances = variances + float(variance_floor)
    covariance = np.diag(variances).astype(np.dtype(dtype))
    return jnp.asarray(covariance[None, ...])


def simulate_process_noisy_truth(
    initial_states: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
    linear_acceleration_noise_std: float,
    angular_acceleration_noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay actions in the analytic environment with process noise.

    The process disturbance is applied to commanded linear and angular
    accelerations before integration, matching AccelDiffDriveMultiAgentEnv.
    """
    if linear_acceleration_noise_std < 0.0:
        raise ValueError(
            "linear-acceleration-process-noise-std must be non-negative."
        )
    if angular_acceleration_noise_std < 0.0:
        raise ValueError(
            "angular-acceleration-process-noise-std must be non-negative."
        )

    config = AccelDiffDriveConfig(
        num_agents=int(initial_states.shape[0]),
        dt=float(metadata["dt"]),
        max_steps=int(actions.shape[0]),
        min_v=float(metadata["min_v"]),
        max_v=float(metadata["max_v"]),
        max_omega=float(metadata["max_omega"]),
        max_linear_acceleration=float(
            metadata.get("max_linear_acceleration", 0.8)
        ),
        max_angular_acceleration=float(
            metadata.get("max_angular_acceleration", 2.0)
        ),
        linear_acceleration_noise_std=float(
            linear_acceleration_noise_std
        ),
        angular_acceleration_noise_std=float(
            angular_acceleration_noise_std
        ),
        wrap_heading=bool(metadata.get("wrap_heading", False)),
    )
    env = AccelDiffDriveMultiAgentEnv(config)
    state = env.state_from_local_states(jnp.asarray(initial_states))
    rng = jax.random.PRNGKey(seed)

    trajectory = [np.asarray(initial_states)]
    sampled_noise = []
    effective_actions = []
    for action_t in actions:
        rng, step_key = jax.random.split(rng)
        state, local_next, _, info = env.step(
            rng=step_key,
            state=state,
            actions=jnp.asarray(action_t),
        )
        trajectory.append(np.asarray(local_next))
        sampled_noise.append(np.asarray(info["acceleration_noise"]))
        effective_actions.append(np.asarray(info["effective_actions"]))

    return (
        np.stack(trajectory, axis=0),
        np.stack(sampled_noise, axis=0),
        np.stack(effective_actions, axis=0),
    )


def sample_measurements(
    truth: np.ndarray,
    measurement_noise_std: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample y_t = x_t + v_t for t >= 1.

    Returns measurements and the sampled noise, both with shape (H, N, 5).
    The initial state is assumed known for estimator initialization and is not
    independently measured here.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0.0,
        scale=measurement_noise_std[None, None, :],
        size=truth[1:].shape,
    ).astype(truth.dtype, copy=False)
    measurements = truth[1:] + noise
    return measurements, noise

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
    measurements: np.ndarray,
    measurement_noise: np.ndarray,
    measurement_noise_std: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    state_scales: np.ndarray,
    sender_full_state_error_threshold: float,
    sender_full_state_uncertainty_threshold: float,
    sender_uncertainty_sigma_scale: float,
    position_ellipse_scale: float,
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
    if measurements.shape != truth[1:].shape:
        raise ValueError(
            "measurements must have shape equal to truth[1:]: "
            f"{measurements.shape} versus {truth[1:].shape}."
        )
    if measurement_noise.shape != measurements.shape:
        raise ValueError(
            "measurement_noise must match measurements shape."
        )

    measurement_covariance = diagonal_measurement_covariance(
        measurement_noise_std=measurement_noise_std,
        variance_floor=measurement_variance,
        dtype=np.asarray(truth).dtype,
    )

    predicted_means = []
    posterior_means = []
    predicted_covariances = []
    posterior_covariances = []

    # Trigger errors are residuals against noisy private measurements.
    predicted_full_state_errors = []
    posterior_full_state_errors = []
    # True errors are diagnostics against the latent physical state.
    predicted_true_full_state_errors = []
    posterior_true_full_state_errors = []
    predicted_full_state_uncertainties = []
    posterior_full_state_uncertainties = []

    # Position radii are retained only for the 2D uncertainty ellipse and
    # optional diagnostics. They are not used by the full-state trigger.
    predicted_position_radii = []
    posterior_position_radii = []
    communication_decisions = []

    state_scales_jax = jnp.asarray(state_scales)

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
        measured_next = jnp.asarray(measurements[step_index])
        predicted_local_means = reshape_belief_mean(
            predicted_belief,
            num_agents,
            local_state_dim,
        )

        decisions = np.zeros((num_agents,), dtype=np.bool_)
        step_error_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )
        step_uncertainty_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )

        for sender_agent_id in range(num_agents):
            sender_mean = jnp.asarray(
                predicted_local_means[
                    sender_agent_id:sender_agent_id + 1
                ]
            )
            sender_observation = measured_next[
                sender_agent_id:sender_agent_id + 1
            ]
            sender_covariance = extract_agent_covariance(
                belief=predicted_belief,
                agent_id=sender_agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

            step_error_scores[sender_agent_id] = float(
                normalized_full_state_error_score(
                    predicted_local_mean=sender_mean,
                    observed_local_state=sender_observation,
                    state_scales=state_scales_jax,
                )[0]
            )
            step_uncertainty_scores[sender_agent_id] = float(
                normalized_full_state_uncertainty_score(
                    predicted_local_covariance=sender_covariance,
                    state_scales=state_scales_jax,
                    sigma_scale=sender_uncertainty_sigma_scale,
                )[0]
            )
            decisions[sender_agent_id] = bool(
                sender_normalized_full_state_trigger(
                    predicted_local_mean=sender_mean,
                    observed_local_state=sender_observation,
                    predicted_local_covariance=sender_covariance,
                    state_scales=state_scales_jax,
                    error_threshold=(
                        sender_full_state_error_threshold
                    ),
                    uncertainty_threshold=(
                        sender_full_state_uncertainty_threshold
                    ),
                    uncertainty_sigma_scale=(
                        sender_uncertainty_sigma_scale
                    ),
                )[0]
            )

        predicted_position_radius = np.asarray(
            all_agent_position_uncertainty_radii(
                belief=predicted_belief,
                num_agents=num_agents,
                scale=position_ellipse_scale,
            )[0]
        )

        belief = predicted_belief
        for sender_agent_id in range(num_agents):
            if not decisions[sender_agent_id]:
                continue

            belief, _ = direct_agent_observation_update(
                predicted_belief=belief,
                observed_agent_id=sender_agent_id,
                measurement=measured_next[
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

        posterior_error_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )
        posterior_uncertainty_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )

        for agent_id in range(num_agents):
            posterior_covariance = extract_agent_covariance(
                belief=belief,
                agent_id=agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )
            posterior_error_scores[agent_id] = float(
                normalized_full_state_error_score(
                    predicted_local_mean=jnp.asarray(
                        posterior_local_means[
                            agent_id:agent_id + 1
                        ]
                    ),
                    observed_local_state=measured_next[
                        agent_id:agent_id + 1
                    ],
                    state_scales=state_scales_jax,
                )[0]
            )
            posterior_uncertainty_scores[agent_id] = float(
                normalized_full_state_uncertainty_score(
                    predicted_local_covariance=posterior_covariance,
                    state_scales=state_scales_jax,
                    sigma_scale=sender_uncertainty_sigma_scale,
                )[0]
            )

        predicted_true_error_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )
        posterior_true_error_scores = np.zeros(
            (num_agents,),
            dtype=np.float32,
        )
        for agent_id in range(num_agents):
            predicted_true_error_scores[agent_id] = float(
                normalized_full_state_error_score(
                    predicted_local_mean=jnp.asarray(
                        predicted_local_means[
                            agent_id:agent_id + 1
                        ]
                    ),
                    observed_local_state=true_next[
                        agent_id:agent_id + 1
                    ],
                    state_scales=state_scales_jax,
                )[0]
            )
            posterior_true_error_scores[agent_id] = float(
                normalized_full_state_error_score(
                    predicted_local_mean=jnp.asarray(
                        posterior_local_means[
                            agent_id:agent_id + 1
                        ]
                    ),
                    observed_local_state=true_next[
                        agent_id:agent_id + 1
                    ],
                    state_scales=state_scales_jax,
                )[0]
            )

        posterior_position_radius = np.asarray(
            all_agent_position_uncertainty_radii(
                belief=belief,
                num_agents=num_agents,
                scale=position_ellipse_scale,
            )[0]
        )

        predicted_means.append(predicted_local_means)
        posterior_means.append(posterior_local_means)
        predicted_covariances.append(
            np.asarray(predicted_belief.covariance[0])
        )
        posterior_covariances.append(
            np.asarray(belief.covariance[0])
        )

        predicted_full_state_errors.append(step_error_scores)
        posterior_full_state_errors.append(
            posterior_error_scores
        )
        predicted_true_full_state_errors.append(
            predicted_true_error_scores
        )
        posterior_true_full_state_errors.append(
            posterior_true_error_scores
        )
        predicted_full_state_uncertainties.append(
            step_uncertainty_scores
        )
        posterior_full_state_uncertainties.append(
            posterior_uncertainty_scores
        )

        predicted_position_radii.append(
            predicted_position_radius
        )
        posterior_position_radii.append(
            posterior_position_radius
        )
        communication_decisions.append(decisions)

    return {
        "predicted_means": np.stack(predicted_means),
        "posterior_means": np.stack(posterior_means),
        "predicted_covariances": np.stack(
            predicted_covariances
        ),
        "posterior_covariances": np.stack(
            posterior_covariances
        ),
        "predicted_full_state_errors": np.stack(
            predicted_full_state_errors
        ),
        "posterior_full_state_errors": np.stack(
            posterior_full_state_errors
        ),
        "predicted_true_full_state_errors": np.stack(
            predicted_true_full_state_errors
        ),
        "posterior_true_full_state_errors": np.stack(
            posterior_true_full_state_errors
        ),
        "measurements": np.asarray(measurements),
        "measurement_noise": np.asarray(measurement_noise),
        "predicted_full_state_uncertainties": np.stack(
            predicted_full_state_uncertainties
        ),
        "posterior_full_state_uncertainties": np.stack(
            posterior_full_state_uncertainties
        ),
        "predicted_position_radii": np.stack(
            predicted_position_radii
        ),
        "posterior_position_radii": np.stack(
            posterior_position_radii
        ),
        "communications": np.stack(
            communication_decisions
        ),
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
    predicted_uncertainties: np.ndarray,
    posterior_uncertainties: np.ndarray,
    communications: np.ndarray,
    error_threshold: float,
    uncertainty_threshold: float,
    uncertainty_sigma_scale: float,
    output: Path,
    dpi: int,
) -> None:
    horizon, num_agents = predicted_errors.shape
    time = np.arange(1, horizon + 1)
    figure, (uncertainty_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for agent_id in range(num_agents):
        color = AGENT_COLORS[
            agent_id % len(AGENT_COLORS)
        ]

        uncertainty_axis.plot(
            time,
            predicted_uncertainties[:, agent_id],
            color=color,
            linewidth=2.1,
            label=f"agent {agent_id} pre-broadcast",
        )
        uncertainty_axis.plot(
            time,
            posterior_uncertainties[:, agent_id],
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
            uncertainty_axis.scatter(
                time[event_mask],
                predicted_uncertainties[
                    event_mask,
                    agent_id,
                ],
                s=42,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=6,
            )
            error_axis.scatter(
                time[event_mask],
                predicted_errors[
                    event_mask,
                    agent_id,
                ],
                s=42,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=6,
            )

    uncertainty_axis.axhline(
        uncertainty_threshold,
        color=END_COLOR,
        linestyle=":",
        linewidth=2.0,
        label=(
            "uncertainty threshold = "
            f"{uncertainty_threshold:g}"
        ),
    )
    error_axis.axhline(
        error_threshold,
        color=END_COLOR,
        linestyle=":",
        linewidth=2.0,
        label=f"error threshold = {error_threshold:g}",
    )

    uncertainty_axis.set_ylabel(
        "normalized full-state uncertainty"
    )
    uncertainty_axis.set_title(
        "Full-state trigger 1: "
        f"{uncertainty_sigma_scale:g}σ normalized RMS uncertainty"
    )
    error_axis.set_xlabel("time step")
    error_axis.set_ylabel(
        "normalized full-state RMS error"
    )
    error_axis.set_title(
        "Full-state trigger 2: private observation residual"
    )

    for axis in (uncertainty_axis, error_axis):
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

    if args.sender_full_state_error_threshold < 0.0:
        raise ValueError(
            "sender-full-state-error-threshold must be non-negative."
        )
    if args.sender_full_state_uncertainty_threshold < 0.0:
        raise ValueError(
            "sender-full-state-uncertainty-threshold "
            "must be non-negative."
        )
    if args.sender_uncertainty_sigma_scale <= 0.0:
        raise ValueError(
            "sender-uncertainty-sigma-scale must be positive."
        )
    if args.position_ellipse_scale <= 0.0:
        raise ValueError(
            "position-ellipse-scale must be positive."
        )
    if args.epistemic_process_scale < 0.0:
        raise ValueError("epistemic-process-scale must be non-negative.")
    if args.measurement_variance < 0.0:
        raise ValueError("measurement-variance must be non-negative.")
    measurement_noise_std = resolve_measurement_noise_std(
        args.measurement_noise_std
    )
    for name, value in (
        (
            "linear-acceleration-process-noise-std",
            args.linear_acceleration_process_noise_std,
        ),
        (
            "angular-acceleration-process-noise-std",
            args.angular_acceleration_process_noise_std,
        ),
    ):
        if value is not None and value < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    dataset = load_dataset(args.dataset)
    saved_models, standardizers, checkpoint_metadata = load_checkpoint(
        args.checkpoint
    )

    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)

    state_scales = resolve_state_scales(
        args.state_scales,
        metadata,
    )

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
    dataset_truth = local_states[trajectory_id, :horizon + 1]
    action_sequence = actions[trajectory_id, :horizon]

    resimulate_process_noise = (
        args.linear_acceleration_process_noise_std is not None
        or args.angular_acceleration_process_noise_std is not None
    )
    if resimulate_process_noise:
        linear_process_std = float(
            args.linear_acceleration_process_noise_std
            if args.linear_acceleration_process_noise_std is not None
            else 0.0
        )
        angular_process_std = float(
            args.angular_acceleration_process_noise_std
            if args.angular_acceleration_process_noise_std is not None
            else 0.0
        )
        truth, sampled_process_noise, effective_actions = (
            simulate_process_noisy_truth(
                initial_states=dataset_truth[0],
                actions=action_sequence,
                metadata=metadata,
                linear_acceleration_noise_std=linear_process_std,
                angular_acceleration_noise_std=angular_process_std,
                seed=args.seed + 101,
            )
        )
    else:
        truth = dataset_truth
        sampled_process_noise = np.zeros_like(action_sequence)
        effective_actions = np.asarray(action_sequence)
        linear_process_std = float(
            metadata.get("linear_acceleration_noise_std", np.nan)
        )
        angular_process_std = float(
            metadata.get("angular_acceleration_noise_std", np.nan)
        )

    measurements, sampled_measurement_noise = sample_measurements(
        truth=truth,
        measurement_noise_std=measurement_noise_std,
        seed=args.seed + 202,
    )

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
        measurements=measurements,
        measurement_noise=sampled_measurement_noise,
        measurement_noise_std=measurement_noise_std,
        actions=action_sequence,
        model_states=model_states,
        standardizers=standardizers,
        metadata=metadata,
        state_scales=state_scales,
        sender_full_state_error_threshold=(
            args.sender_full_state_error_threshold
        ),
        sender_full_state_uncertainty_threshold=(
            args.sender_full_state_uncertainty_threshold
        ),
        sender_uncertainty_sigma_scale=(
            args.sender_uncertainty_sigma_scale
        ),
        position_ellipse_scale=args.position_ellipse_scale,
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
        predicted_errors=trace[
            "predicted_full_state_errors"
        ],
        posterior_errors=trace[
            "posterior_full_state_errors"
        ],
        predicted_uncertainties=trace[
            "predicted_full_state_uncertainties"
        ],
        posterior_uncertainties=trace[
            "posterior_full_state_uncertainties"
        ],
        communications=trace["communications"],
        error_threshold=(
            args.sender_full_state_error_threshold
        ),
        uncertainty_threshold=(
            args.sender_full_state_uncertainty_threshold
        ),
        uncertainty_sigma_scale=(
            args.sender_uncertainty_sigma_scale
        ),
        output=(
            args.output_directory
            / "sender_full_state_trigger_signals.png"
        ),
        dpi=args.dpi,
    )

    np.savez_compressed(
        args.output_directory / "sender_full_state_trigger_estimator_trace.npz",
        trajectory_index=np.asarray(trajectory_id),
        communication_mode=np.asarray(
            "sender_normalized_full_state_triggered"
        ),
        prediction_mode=np.asarray(args.prediction_mode),
        sender_full_state_error_threshold=np.asarray(
            args.sender_full_state_error_threshold
        ),
        sender_full_state_uncertainty_threshold=np.asarray(
            args.sender_full_state_uncertainty_threshold
        ),
        sender_uncertainty_sigma_scale=np.asarray(
            args.sender_uncertainty_sigma_scale
        ),
        position_ellipse_scale=np.asarray(
            args.position_ellipse_scale
        ),
        state_scales=state_scales,
        truth=truth,
        dataset_truth=dataset_truth,
        measurements=trace["measurements"],
        measurement_noise=trace["measurement_noise"],
        measurement_noise_std=measurement_noise_std,
        measurement_variance_floor=np.asarray(
            args.measurement_variance
        ),
        process_noise_resimulated=np.asarray(
            resimulate_process_noise
        ),
        linear_acceleration_process_noise_std=np.asarray(
            linear_process_std
        ),
        angular_acceleration_process_noise_std=np.asarray(
            angular_process_std
        ),
        sampled_process_noise=sampled_process_noise,
        effective_actions=effective_actions,
        actions=action_sequence,
        model_only=model_only,
        estimator_prior=trace["predicted_means"],
        estimator_posterior=posterior_with_initial,
        prior_covariances=trace["predicted_covariances"],
        posterior_covariances=trace["posterior_covariances"],
        pre_message_full_state_uncertainty_scores=trace[
            "predicted_full_state_uncertainties"
        ],
        post_message_full_state_uncertainty_scores=trace[
            "posterior_full_state_uncertainties"
        ],
        pre_message_full_state_error_scores=trace[
            "predicted_full_state_errors"
        ],
        post_message_full_state_error_scores=trace[
            "posterior_full_state_errors"
        ],
        pre_message_true_full_state_error_scores=trace[
            "predicted_true_full_state_errors"
        ],
        post_message_true_full_state_error_scores=trace[
            "posterior_true_full_state_errors"
        ],
        pre_message_position_radii=trace[
            "predicted_position_radii"
        ],
        post_message_position_radii=trace[
            "posterior_position_radii"
        ],
        communications=trace["communications"],
    )

    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Trajectory index:", trajectory_id)
    print("Prediction mode:", args.prediction_mode)
    print(
        "State scales [px, py, phi, v, omega]:",
        state_scales,
    )
    print(
        "Measurement noise std [px, py, phi, v, omega]:",
        measurement_noise_std,
    )
    if resimulate_process_noise:
        print(
            "Resimulated process noise std [linear accel, angular accel]:",
            [linear_process_std, angular_process_std],
        )
    else:
        print(
            "Process trajectory source: saved dataset "
            "(no additional resimulation)."
        )
    print(
        "Sender trigger: normalized full-state RMS measurement residual > "
        f"{args.sender_full_state_error_threshold:g} OR "
        f"{args.sender_uncertainty_sigma_scale:g}σ normalized "
        "RMS uncertainty > "
        f"{args.sender_full_state_uncertainty_threshold:g}"
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
