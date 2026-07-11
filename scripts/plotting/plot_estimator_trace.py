# scripts/plotting/plot_estimator_trace.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot uncertainty, state-estimation error, and communication "
            "events from a saved estimator trace."
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


def plot_remote_agent_trace(
    time_indices: np.ndarray,
    predicted_covariance_proxy: np.ndarray,
    after_ego_covariance_proxy: np.ndarray,
    posterior_covariance_proxy: np.ndarray,
    predicted_state_mse: np.ndarray,
    after_ego_state_mse: np.ndarray,
    posterior_state_mse: np.ndarray,
    communication_decisions: np.ndarray,
    agent_id: int,
    ego_agent_id: int,
    threshold: float,
    output_path: Path,
) -> None:
    """
    Plot uncertainty and estimation error for one remote agent.
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
        predicted_covariance_proxy[:, agent_id],
        label="Predicted covariance proxy",
    )

    axes[0].plot(
        time_indices,
        after_ego_covariance_proxy[:, agent_id],
        label="After ego observation",
    )

    axes[0].plot(
        time_indices,
        posterior_covariance_proxy[:, agent_id],
        label="After remote communication",
    )

    axes[0].axhline(
        threshold,
        linestyle="--",
        label="Event threshold",
    )

    for communication_time in communication_times:
        axes[0].axvline(
            communication_time,
            linestyle=":",
            alpha=0.6,
        )

    axes[0].set_ylabel("trace(P_jj) / D")
    axes[0].set_title(
        f"Ego agent {ego_agent_id}: uncertainty for remote agent {agent_id}"
    )
    axes[0].legend()

    axes[1].plot(
        time_indices,
        predicted_state_mse[:, agent_id],
        label="Predicted state MSE",
    )

    axes[1].plot(
        time_indices,
        after_ego_state_mse[:, agent_id],
        label="After ego observation MSE",
    )

    axes[1].plot(
        time_indices,
        posterior_state_mse[:, agent_id],
        label="After remote communication MSE",
    )

    for communication_time in communication_times:
        axes[1].axvline(
            communication_time,
            linestyle=":",
            alpha=0.6,
        )

    axes[1].set_ylabel("State MSE")
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

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(trace_path, allow_pickle=False) as trace:
        time_indices = trace["time_indices"]
        communication_decisions = trace["communication_decisions"]

        predicted_covariance_proxy = trace[
            "predicted_covariance_proxy"
        ]
        after_ego_covariance_proxy = trace[
            "after_ego_covariance_proxy"
        ]
        posterior_covariance_proxy = trace[
            "posterior_covariance_proxy"
        ]

        predicted_state_mse = trace["predicted_state_mse"]
        after_ego_state_mse = trace[
            "after_ego_state_mse"
        ]
        posterior_state_mse = trace[
            "posterior_state_mse"
        ]

        ego_agent_id = int(trace["ego_agent_id"])
        num_agents = int(trace["num_agents"])
        threshold = float(trace["event_trigger_threshold"])

    trace_stem = trace_path.stem

    for agent_id in range(num_agents):
        if agent_id == ego_agent_id:
            continue

        output_path = output_directory / (
            f"{trace_stem}_remote_agent_{agent_id}.png"
        )

        plot_remote_agent_trace(
            time_indices=time_indices,
            predicted_covariance_proxy=predicted_covariance_proxy,
            after_ego_covariance_proxy=after_ego_covariance_proxy,
            posterior_covariance_proxy=posterior_covariance_proxy,
            predicted_state_mse=predicted_state_mse,
            after_ego_state_mse=after_ego_state_mse,
            posterior_state_mse=posterior_state_mse,
            communication_decisions=communication_decisions,
            agent_id=agent_id,
            ego_agent_id=ego_agent_id,
            threshold=threshold,
            output_path=output_path,
        )

        print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()