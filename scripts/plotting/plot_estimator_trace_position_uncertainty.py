# scripts/plotting/plot_estimator_trace_position_uncertainty.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot position uncertainty radius, position error, and "
            "communication events from a saved estimator trace."
        )
    )

    parser.add_argument(
        "--trace-path",
        type=Path,
        required=True,
        help="Path to a .npz estimator trace.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/figures"),
        help="Directory for saved figures.",
    )

    parser.add_argument(
        "--ellipse-scale",
        type=float,
        default=2.0,
        help=(
            "Standard-deviation multiplier for the position uncertainty "
            "radius. Use 2.0 for a 2-sigma radius."
        ),
    )

    return parser.parse_args()


def communication_steps(
    communication_decisions: np.ndarray,
    agent_id: int,
) -> np.ndarray:
    """
    Return zero-based time indices where the given agent communicated.
    """
    return np.flatnonzero(
        communication_decisions[:, agent_id]
    )


def compute_position_error(
    local_means: np.ndarray,
    true_local_states: np.ndarray,
) -> np.ndarray:
    """
    Compute per-agent Euclidean position error.

    Args:
        local_means:
            Estimated local states with shape (T, N, 4).

        true_local_states:
            True local states with shape (T, N, 4).

    Returns:
        Position error with shape (T, N), in environment position units.
    """
    estimated_positions = local_means[:, :, :2]
    true_positions = true_local_states[:, :, :2]

    return np.linalg.norm(
        estimated_positions - true_positions,
        axis=-1,
    )


def compute_position_uncertainty_radius(
    global_covariances: np.ndarray,
    num_agents: int,
    local_state_dim: int,
    ellipse_scale: float,
) -> np.ndarray:
    """
    Compute the major semi-axis of the position uncertainty ellipse.

    Args:
        global_covariances:
            Full global covariance matrices with shape (T, N * D, N * D).

        num_agents:
            Number of agents.

        local_state_dim:
            Local physical state dimension.

        ellipse_scale:
            Standard-deviation multiplier, for example 2.0 for 2-sigma.

    Returns:
        Position uncertainty radius with shape (T, N), in environment
        position units.

    The radius is:
        ellipse_scale * sqrt(lambda_max(P_pos))

    where P_pos is the 2x2 position covariance block.
    """
    num_steps = global_covariances.shape[0]

    radii = np.zeros(
        (num_steps, num_agents),
        dtype=np.float32,
    )

    for step_index in range(num_steps):
        for agent_id in range(num_agents):
            start = agent_id * local_state_dim

            position_covariance = global_covariances[
                step_index,
                start:start + 2,
                start:start + 2,
            ]

            symmetric_position_covariance = 0.5 * (
                position_covariance + position_covariance.T
            )

            eigenvalues = np.linalg.eigvalsh(
                symmetric_position_covariance
            )

            largest_eigenvalue = float(
                np.max(eigenvalues)
            )

            largest_eigenvalue = max(
                largest_eigenvalue,
                0.0,
            )

            radii[step_index, agent_id] = (
                ellipse_scale * np.sqrt(largest_eigenvalue)
            )

    return radii


def compute_scene_scale(
    true_local_states: np.ndarray,
    landmark_positions: np.ndarray,
) -> float:
    """
    Compute a reference position scale from the recorded episode.

    The returned value is the largest span over x/y coordinates, using both
    agent positions and landmark positions.
    """
    agent_positions = true_local_states[:, :, :2].reshape(-1, 2)
    landmark_positions = landmark_positions.reshape(-1, 2)

    all_positions = np.concatenate(
        [agent_positions, landmark_positions],
        axis=0,
    )

    spans = np.ptp(
        all_positions,
        axis=0,
    )

    scene_scale = float(np.max(spans))

    return max(scene_scale, 1.0e-8)


def plot_remote_agent_trace(
    time_indices: np.ndarray,
    pre_message_position_uncertainty: np.ndarray,
    posterior_position_uncertainty: np.ndarray,
    pre_message_position_error: np.ndarray,
    posterior_position_error: np.ndarray,
    communication_decisions: np.ndarray,
    agent_id: int,
    ego_agent_id: int,
    ellipse_scale: float,
    scene_scale: float,
    output_path: Path,
) -> None:
    """
    Plot position uncertainty and position error for one remote agent.
    """
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
    )

    communication_time_indices = communication_steps(
        communication_decisions=communication_decisions,
        agent_id=agent_id,
    )

    communication_times = time_indices[
        communication_time_indices
    ]

    axes[0].plot(
        time_indices,
        pre_message_position_uncertainty[:, agent_id],
        label="Pre-message belief",
    )

    axes[0].plot(
        time_indices,
        posterior_position_uncertainty[:, agent_id],
        label="Posterior belief",
    )

    axes[0].axhline(
        0.10 * scene_scale,
        linestyle="--",
        label="10% scene scale",
    )

    axes[0].axhline(
        0.25 * scene_scale,
        linestyle=":",
        label="25% scene scale",
    )

    for communication_time in communication_times:
        axes[0].axvline(
            communication_time,
            linestyle=":",
            alpha=0.6,
        )

    axes[0].set_ylabel(
        f"{ellipse_scale:.1f}σ position uncertainty radius"
    )

    axes[0].set_title(
        f"Ego agent {ego_agent_id}: remote agent {agent_id}"
    )

    axes[0].legend()

    axes[1].plot(
        time_indices,
        pre_message_position_error[:, agent_id],
        label="Pre-message belief",
    )

    axes[1].plot(
        time_indices,
        posterior_position_error[:, agent_id],
        label="Posterior belief",
    )

    axes[1].axhline(
        0.10 * scene_scale,
        linestyle="--",
        label="10% scene scale",
    )

    axes[1].axhline(
        0.25 * scene_scale,
        linestyle=":",
        label="25% scene scale",
    )

    for communication_time in communication_times:
        axes[1].axvline(
            communication_time,
            linestyle=":",
            alpha=0.6,
        )

    axes[1].set_ylabel("Position error")
    axes[1].legend()

    communication_indicator = communication_decisions[
        :, agent_id
    ].astype(np.int32)

    axes[2].step(
        time_indices,
        communication_indicator,
        where="mid",
        label="Remote message",
    )

    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylabel("Communication")
    axes[2].set_xlabel("Environment step")
    axes[2].legend()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    trace_path = args.trace_path.resolve()
    output_directory = args.output_directory.resolve()

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace file does not exist: {trace_path}"
        )

    if args.ellipse_scale <= 0.0:
        raise ValueError(
            "ellipse_scale must be positive, "
            f"got {args.ellipse_scale}."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(trace_path, allow_pickle=False) as trace:
        time_indices = trace["time_indices"]
        true_local_states = trace["true_local_states"]
        landmark_positions = trace["landmark_positions"]

        after_ego_local_means = trace[
            "after_ego_local_means"
        ]
        posterior_local_means = trace[
            "posterior_local_means"
        ]

        after_ego_global_covariances = trace[
            "after_ego_global_covariances"
        ]
        posterior_global_covariances = trace[
            "posterior_global_covariances"
        ]

        communication_decisions = trace[
            "communication_decisions"
        ]

        ego_agent_id = int(trace["ego_agent_id"])
        num_agents = int(trace["num_agents"])
        local_state_dim = int(trace["local_state_dim"])

    pre_message_position_error = compute_position_error(
        local_means=after_ego_local_means,
        true_local_states=true_local_states,
    )

    posterior_position_error = compute_position_error(
        local_means=posterior_local_means,
        true_local_states=true_local_states,
    )

    pre_message_position_uncertainty = (
        compute_position_uncertainty_radius(
            global_covariances=after_ego_global_covariances,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
            ellipse_scale=args.ellipse_scale,
        )
    )

    posterior_position_uncertainty = (
        compute_position_uncertainty_radius(
            global_covariances=posterior_global_covariances,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
            ellipse_scale=args.ellipse_scale,
        )
    )

    scene_scale = compute_scene_scale(
        true_local_states=true_local_states,
        landmark_positions=landmark_positions,
    )

    print(f"Scene scale: {scene_scale:.6f}")
    print(f"10% scene scale: {0.10 * scene_scale:.6f}")
    print(f"25% scene scale: {0.25 * scene_scale:.6f}")
    print(
        "Position uncertainty radius scale: "
        f"{args.ellipse_scale:.2f} sigma"
    )

    trace_stem = trace_path.stem

    for agent_id in range(num_agents):
        if agent_id == ego_agent_id:
            continue

        output_path = output_directory / (
            f"{trace_stem}_remote_agent_{agent_id}_"
            f"position_uncertainty.png"
        )

        plot_remote_agent_trace(
            time_indices=time_indices,
            pre_message_position_uncertainty=(
                pre_message_position_uncertainty
            ),
            posterior_position_uncertainty=(
                posterior_position_uncertainty
            ),
            pre_message_position_error=pre_message_position_error,
            posterior_position_error=posterior_position_error,
            communication_decisions=communication_decisions,
            agent_id=agent_id,
            ego_agent_id=ego_agent_id,
            ellipse_scale=args.ellipse_scale,
            scene_scale=scene_scale,
            output_path=output_path,
        )

        print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()