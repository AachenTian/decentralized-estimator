# scripts/plotting/animate_estimator_trace.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an animation from a saved event-triggered estimator trace."
        )
    )

    parser.add_argument(
        "--trace-path",
        type=Path,
        required=True,
        help="Path to the saved .npz trace.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output .gif or .mp4 path.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Output resolution.",
    )

    parser.add_argument(
        "--ellipse-scale",
        type=float,
        default=2.0,
        help="Ellipse radius multiplier in standard deviations.",
    )

    return parser.parse_args()


def covariance_ellipse_parameters(
    covariance: np.ndarray,
    ellipse_scale: float,
) -> tuple[float, float, float]:
    """
    Convert a 2D covariance matrix into ellipse width, height, and angle.
    """
    symmetric_covariance = 0.5 * (
        covariance + covariance.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        symmetric_covariance
    )

    ordering = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]

    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    width = 2.0 * ellipse_scale * np.sqrt(eigenvalues[0])
    height = 2.0 * ellipse_scale * np.sqrt(eigenvalues[1])

    principal_direction = eigenvectors[:, 0]
    angle_degrees = np.degrees(
        np.arctan2(
            principal_direction[1],
            principal_direction[0],
        )
    )

    return float(width), float(height), float(angle_degrees)


def local_position_covariance(
    global_covariance: np.ndarray,
    agent_id: int,
    local_state_dim: int,
) -> np.ndarray:
    """
    Extract the 2x2 position covariance block for one agent.
    """
    start = agent_id * local_state_dim

    return global_covariance[
        start:start + 2,
        start:start + 2,
    ]


def build_writer(
    output_path: Path,
    fps: int,
) -> animation.AbstractMovieWriter:
    """
    Select a Matplotlib writer based on the output extension.
    """
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)

    if suffix == ".mp4":
        return animation.FFMpegWriter(
            fps=fps,
            bitrate=1800,
        )

    raise ValueError(
        "output_path must end with '.gif' or '.mp4', "
        f"got '{output_path}'."
    )


def main() -> None:
    args = parse_arguments()

    trace_path = args.trace_path.resolve()
    output_path = args.output_path.resolve()

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace file does not exist: {trace_path}"
        )

    if args.fps < 1:
        raise ValueError(
            f"fps must be at least one, got {args.fps}."
        )

    if args.ellipse_scale <= 0.0:
        raise ValueError(
            "ellipse_scale must be positive, "
            f"got {args.ellipse_scale}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(trace_path, allow_pickle=False) as trace:
        time_indices = trace["time_indices"]

        true_local_states = trace["true_local_states"]
        landmark_positions = trace["landmark_positions"]

        predicted_local_means = trace[
            "predicted_local_means"
        ]
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

        after_ego_covariance_proxy = trace[
            "after_ego_covariance_proxy"
        ]
        posterior_covariance_proxy = trace[
            "posterior_covariance_proxy"
        ]

        after_ego_state_mse = trace[
            "after_ego_state_mse"
        ]
        posterior_state_mse = trace[
            "posterior_state_mse"
        ]

        communication_decisions = trace[
            "communication_decisions"
        ]

        ego_agent_id = int(trace["ego_agent_id"])
        num_agents = int(trace["num_agents"])
        local_state_dim = int(trace["local_state_dim"])
        threshold = float(trace["event_trigger_threshold"])

    num_steps = len(time_indices)

    if num_steps == 0:
        raise ValueError("The trace contains zero time steps.")

    remote_agent_ids = [
        agent_id
        for agent_id in range(num_agents)
        if agent_id != ego_agent_id
    ]

    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    agent_colors = {
        agent_id: color_cycle[
            agent_id % len(color_cycle)
        ]
        for agent_id in range(num_agents)
    }

    true_positions = true_local_states[:, :, :2]
    predicted_positions = predicted_local_means[:, :, :2]
    after_ego_positions = after_ego_local_means[:, :, :2]
    posterior_positions = posterior_local_means[:, :, :2]

    all_positions = np.concatenate(
        [
            true_positions.reshape(-1, 2),
            landmark_positions.reshape(-1, 2),
        ],
        axis=0,
    )

    position_minimum = np.min(all_positions, axis=0)
    position_maximum = np.max(all_positions, axis=0)

    span = np.maximum(
        position_maximum - position_minimum,
        1.0,
    )

    margin = 0.20 * span

    figure = plt.figure(
        figsize=(15, 8),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.25, 1.0],
    )

    axis_scene = figure.add_subplot(grid[:, 0])
    axis_covariance = figure.add_subplot(grid[0, 1])
    axis_mse = figure.add_subplot(grid[1, 1])

    axis_scene.set_title(
        "Global scene: true states and ego belief"
    )
    axis_scene.set_xlabel("x position")
    axis_scene.set_ylabel("y position")
    axis_scene.set_aspect("equal", adjustable="box")

    axis_scene.set_xlim(
        position_minimum[0] - margin[0],
        position_maximum[0] + margin[0],
    )
    axis_scene.set_ylim(
        position_minimum[1] - margin[1],
        position_maximum[1] + margin[1],
    )

    axis_scene.grid(alpha=0.25)

    axis_scene.scatter(
        landmark_positions[0, :, 0],
        landmark_positions[0, :, 1],
        marker="*",
        s=180,
        color="black",
        label="Landmark",
        zorder=2,
    )

    true_state_artists: dict[int, Line2D] = {}
    posterior_state_artists: dict[int, Line2D] = {}
    pre_message_state_artists: dict[int, Line2D] = {}
    correction_lines: dict[int, Line2D] = {}

    posterior_ellipses: dict[int, Ellipse] = {}
    pre_message_ellipses: dict[int, Ellipse] = {}

    for agent_id in range(num_agents):
        true_state_artist, = axis_scene.plot(
            [],
            [],
            marker="o",
            linestyle="None",
            color=agent_colors[agent_id],
            markersize=9,
            label=f"True agent {agent_id}",
            zorder=5,
        )

        true_state_artists[agent_id] = true_state_artist

    for agent_id in remote_agent_ids:
        posterior_state_artist, = axis_scene.plot(
            [],
            [],
            marker="X",
            linestyle="None",
            color=agent_colors[agent_id],
            markersize=10,
            label=f"Posterior belief agent {agent_id}",
            zorder=7,
        )

        posterior_state_artists[agent_id] = posterior_state_artist

        pre_message_state_artist, = axis_scene.plot(
            [],
            [],
            marker="x",
            linestyle="None",
            color=agent_colors[agent_id],
            markersize=10,
            markeredgewidth=2.0,
            label=f"Pre-message belief agent {agent_id}",
            zorder=8,
        )

        pre_message_state_artists[agent_id] = (
            pre_message_state_artist
        )

        correction_line, = axis_scene.plot(
            [],
            [],
            linestyle="--",
            color=agent_colors[agent_id],
            alpha=0.9,
            linewidth=1.5,
            zorder=6,
        )

        correction_lines[agent_id] = correction_line

        posterior_ellipse = Ellipse(
            xy=(0.0, 0.0),
            width=0.0,
            height=0.0,
            angle=0.0,
            fill=False,
            edgecolor=agent_colors[agent_id],
            linewidth=2.0,
            alpha=0.85,
            zorder=4,
        )

        axis_scene.add_patch(posterior_ellipse)
        posterior_ellipses[agent_id] = posterior_ellipse

        pre_message_ellipse = Ellipse(
            xy=(0.0, 0.0),
            width=0.0,
            height=0.0,
            angle=0.0,
            fill=False,
            edgecolor=agent_colors[agent_id],
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            zorder=3,
        )

        axis_scene.add_patch(pre_message_ellipse)
        pre_message_ellipses[agent_id] = (
            pre_message_ellipse
        )

    status_text = axis_scene.text(
        0.02,
        0.98,
        "",
        transform=axis_scene.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
        zorder=10,
    )

    scene_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            markersize=8,
            label="True state",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="black",
            linestyle="None",
            markersize=8,
            label="Posterior belief mean",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=8,
            label="Pre-message belief mean",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="black",
            label="Communication correction",
        ),
    ]

    axis_scene.legend(
        handles=scene_legend_handles,
        loc="lower left",
        fontsize=8,
    )

    axis_covariance.set_title(
        "Remote uncertainty and threshold"
    )
    axis_covariance.set_ylabel("trace(P_jj) / D")
    axis_covariance.grid(alpha=0.25)

    axis_covariance.axhline(
        threshold,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Event threshold",
    )

    axis_mse.set_title("Remote state-estimation MSE")
    axis_mse.set_xlabel("Environment step")
    axis_mse.set_ylabel("State MSE")
    axis_mse.grid(alpha=0.25)

    for agent_id in remote_agent_ids:
        color = agent_colors[agent_id]

        axis_covariance.plot(
            time_indices,
            after_ego_covariance_proxy[:, agent_id],
            color=color,
            linewidth=2.0,
            label=f"Agent {agent_id}: before message",
        )

        axis_covariance.plot(
            time_indices,
            posterior_covariance_proxy[:, agent_id],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            label=f"Agent {agent_id}: posterior",
        )

        communication_mask = communication_decisions[
            :, agent_id
        ].astype(bool)

        axis_covariance.scatter(
            time_indices[communication_mask],
            after_ego_covariance_proxy[
                communication_mask,
                agent_id,
            ],
            marker="v",
            color=color,
            s=45,
            zorder=5,
        )

        axis_mse.plot(
            time_indices,
            after_ego_state_mse[:, agent_id],
            color=color,
            linewidth=2.0,
            label=f"Agent {agent_id}: before message",
        )

        axis_mse.plot(
            time_indices,
            posterior_state_mse[:, agent_id],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            label=f"Agent {agent_id}: posterior",
        )

    covariance_cursor = axis_covariance.axvline(
        time_indices[0],
        color="black",
        linestyle=":",
        linewidth=1.5,
    )

    mse_cursor = axis_mse.axvline(
        time_indices[0],
        color="black",
        linestyle=":",
        linewidth=1.5,
    )

    axis_covariance.legend(
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    axis_mse.legend(
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    def update(frame_index: int) -> None:
        current_time = time_indices[frame_index]

        sent_message_agent_ids = [
            agent_id
            for agent_id in remote_agent_ids
            if communication_decisions[
                frame_index,
                agent_id,
            ]
        ]

        if sent_message_agent_ids:
            message_description = ", ".join(
                f"agent {agent_id}"
                for agent_id in sent_message_agent_ids
            )
            status = (
                f"Step {current_time}\n"
                f"Communication: {message_description} → "
                f"agent {ego_agent_id}"
            )
        else:
            status = (
                f"Step {current_time}\n"
                "Communication: none"
            )

        status_text.set_text(status)

        for agent_id in range(num_agents):
            current_true_position = true_positions[
                frame_index,
                agent_id,
            ]

            true_state_artists[agent_id].set_data(
                [current_true_position[0]],
                [current_true_position[1]],
            )

        for agent_id in remote_agent_ids:
            current_posterior_position = posterior_positions[
                frame_index,
                agent_id,
            ]

            posterior_state_artists[agent_id].set_data(
                [current_posterior_position[0]],
                [current_posterior_position[1]],
            )

            posterior_covariance = local_position_covariance(
                global_covariance=posterior_global_covariances[
                    frame_index
                ],
                agent_id=agent_id,
                local_state_dim=local_state_dim,
            )

            width, height, angle = covariance_ellipse_parameters(
                covariance=posterior_covariance,
                ellipse_scale=args.ellipse_scale,
            )

            posterior_ellipses[agent_id].center = (
                current_posterior_position[0],
                current_posterior_position[1],
            )
            posterior_ellipses[agent_id].width = width
            posterior_ellipses[agent_id].height = height
            posterior_ellipses[agent_id].angle = angle
            posterior_ellipses[agent_id].set_visible(True)

            is_communication_step = bool(
                communication_decisions[
                    frame_index,
                    agent_id,
                ]
            )

            if is_communication_step:
                current_pre_message_position = (
                    after_ego_positions[
                        frame_index,
                        agent_id,
                    ]
                )

                pre_message_state_artists[
                    agent_id
                ].set_data(
                    [current_pre_message_position[0]],
                    [current_pre_message_position[1]],
                )

                correction_lines[agent_id].set_data(
                    [
                        current_pre_message_position[0],
                        current_posterior_position[0],
                    ],
                    [
                        current_pre_message_position[1],
                        current_posterior_position[1],
                    ],
                )

                pre_message_covariance = (
                    local_position_covariance(
                        global_covariance=after_ego_global_covariances[
                            frame_index
                        ],
                        agent_id=agent_id,
                        local_state_dim=local_state_dim,
                    )
                )

                width, height, angle = (
                    covariance_ellipse_parameters(
                        covariance=pre_message_covariance,
                        ellipse_scale=args.ellipse_scale,
                    )
                )

                pre_message_ellipses[agent_id].center = (
                    current_pre_message_position[0],
                    current_pre_message_position[1],
                )
                pre_message_ellipses[agent_id].width = width
                pre_message_ellipses[agent_id].height = height
                pre_message_ellipses[agent_id].angle = angle
                pre_message_ellipses[
                    agent_id
                ].set_visible(True)
            else:
                pre_message_state_artists[
                    agent_id
                ].set_data([], [])

                correction_lines[agent_id].set_data([], [])

                pre_message_ellipses[
                    agent_id
                ].set_visible(False)

        covariance_cursor.set_xdata(
            [current_time, current_time]
        )

        mse_cursor.set_xdata(
            [current_time, current_time]
        )

    animation_object = animation.FuncAnimation(
        figure,
        update,
        frames=num_steps,
        interval=1000.0 / args.fps,
        repeat=True,
        blit=False,
    )

    writer = build_writer(
        output_path=output_path,
        fps=args.fps,
    )

    animation_object.save(
        output_path,
        writer=writer,
        dpi=args.dpi,
    )

    plt.close(figure)

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()