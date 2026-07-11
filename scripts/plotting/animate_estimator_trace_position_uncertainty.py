from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def read_scalar(data, name: str, default):
    if name not in data.files:
        return default

    value = np.asarray(data[name])

    if value.shape == ():
        return value.item()

    if value.size == 1:
        return value.reshape(-1)[0].item()

    return default


def infer_num_agents(true_local_states: np.ndarray) -> int:
    if true_local_states.ndim != 3:
        raise ValueError(
            "true_local_states must have shape (T, N, D), "
            f"got {true_local_states.shape}."
        )

    return int(true_local_states.shape[1])


def extract_local_covariances(
    global_covariances: np.ndarray,
    num_agents: int,
    local_state_dim: int,
) -> np.ndarray:
    if global_covariances.ndim != 3:
        raise ValueError(
            "global_covariances must have shape (T, N*D, N*D), "
            f"got {global_covariances.shape}."
        )

    local_covariances = []

    for agent_id in range(num_agents):
        start = agent_id * local_state_dim
        end = start + local_state_dim
        local_covariances.append(
            global_covariances[:, start:end, start:end]
        )

    return np.stack(local_covariances, axis=1)


def position_uncertainty_radius(
    local_covariances: np.ndarray,
    scale: float,
) -> np.ndarray:
    position_covariances = local_covariances[..., :2, :2]
    position_covariances = 0.5 * (
        position_covariances
        + np.swapaxes(position_covariances, -1, -2)
    )

    eigenvalues = np.linalg.eigvalsh(position_covariances)
    largest_eigenvalues = np.maximum(
        np.max(eigenvalues, axis=-1),
        0.0,
    )

    return scale * np.sqrt(largest_eigenvalues)


def position_error(
    estimated_local_states: np.ndarray,
    true_local_states: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(
        estimated_local_states[..., :2]
        - true_local_states[..., :2],
        axis=-1,
    )


def communication_for_agent(
    communication_decisions: np.ndarray,
    agent_id: int,
    remote_agent_ids: Sequence[int],
    num_agents: int,
    horizon: int,
) -> np.ndarray:
    if communication_decisions.ndim == 1:
        return np.zeros(horizon, dtype=bool)

    if communication_decisions.shape[1] == num_agents:
        return communication_decisions[:horizon, agent_id].astype(bool)

    if agent_id in remote_agent_ids:
        remote_index = list(remote_agent_ids).index(agent_id)

        if remote_index < communication_decisions.shape[1]:
            return communication_decisions[:horizon, remote_index].astype(bool)

    return np.zeros(horizon, dtype=bool)


def add_covariance_ellipse(
    ax,
    mean: np.ndarray,
    covariance: np.ndarray,
    scale: float,
    edgecolor,
    linestyle: str,
    linewidth: float,
    label: str | None = None,
):
    position_covariance = covariance[:2, :2]
    position_covariance = 0.5 * (
        position_covariance
        + position_covariance.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(position_covariance)
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    width = 2.0 * scale * np.sqrt(eigenvalues[0])
    height = 2.0 * scale * np.sqrt(eigenvalues[1])

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0],
        )
    )

    ellipse = Ellipse(
        xy=mean[:2],
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    ax.add_patch(ellipse)


def compute_scene_limits(
    true_local_states: np.ndarray,
    posterior_local_means: np.ndarray,
    landmark_positions: np.ndarray | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [
        true_local_states[..., :2].reshape(-1, 2),
        posterior_local_means[..., :2].reshape(-1, 2),
    ]

    if landmark_positions is not None and landmark_positions.size > 0:
        points.append(landmark_positions.reshape(-1, 2))

    all_points = np.concatenate(points, axis=0)

    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)

    span = np.maximum(max_xy - min_xy, 1.0)
    margin = 0.25 * span

    return (
        (float(min_xy[0] - margin[0]), float(max_xy[0] + margin[0])),
        (float(min_xy[1] - margin[1]), float(max_xy[1] + margin[1])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Animate an estimator trace using position uncertainty and "
            "position error on the right panels."
        )
    )

    parser.add_argument(
        "--trace-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--ellipse-scale",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--position-trigger-radius",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    data = np.load(args.trace_path, allow_pickle=True)

    true_local_states = np.asarray(data["true_local_states"])
    posterior_local_means = np.asarray(data["posterior_local_means"])

    if "after_ego_local_means" in data.files:
        pre_message_local_means = np.asarray(data["after_ego_local_means"])
    else:
        pre_message_local_means = np.asarray(data["predicted_local_means"])

    if "after_ego_global_covariances" in data.files:
        pre_message_global_covariances = np.asarray(
            data["after_ego_global_covariances"]
        )
    else:
        pre_message_global_covariances = np.asarray(
            data["predicted_global_covariances"]
        )

    posterior_global_covariances = np.asarray(
        data["posterior_global_covariances"]
    )

    communication_decisions = np.asarray(
        data["communication_decisions"]
    )

    landmark_positions = None
    if "landmark_positions" in data.files:
        landmark_positions = np.asarray(data["landmark_positions"])

    ego_agent_id = int(
        read_scalar(
            data=data,
            name="ego_agent_id",
            default=0,
        )
    )

    num_agents = infer_num_agents(true_local_states)
    local_state_dim = int(
        read_scalar(
            data=data,
            name="local_state_dim",
            default=true_local_states.shape[-1],
        )
    )

    horizon = min(
        true_local_states.shape[0],
        posterior_local_means.shape[0],
        pre_message_local_means.shape[0],
        posterior_global_covariances.shape[0],
        pre_message_global_covariances.shape[0],
        communication_decisions.shape[0],
    )

    true_local_states = true_local_states[:horizon]
    posterior_local_means = posterior_local_means[:horizon]
    pre_message_local_means = pre_message_local_means[:horizon]
    posterior_global_covariances = posterior_global_covariances[:horizon]
    pre_message_global_covariances = pre_message_global_covariances[:horizon]
    communication_decisions = communication_decisions[:horizon]

    if "time_indices" in data.files:
        time_indices = np.asarray(data["time_indices"])[:horizon]
    else:
        time_indices = np.arange(1, horizon + 1)

    remote_agent_ids = [
        agent_id
        for agent_id in range(num_agents)
        if agent_id != ego_agent_id
    ]

    pre_message_local_covariances = extract_local_covariances(
        global_covariances=pre_message_global_covariances,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    posterior_local_covariances = extract_local_covariances(
        global_covariances=posterior_global_covariances,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    pre_message_position_radius = position_uncertainty_radius(
        local_covariances=pre_message_local_covariances,
        scale=args.ellipse_scale,
    )

    posterior_position_radius = position_uncertainty_radius(
        local_covariances=posterior_local_covariances,
        scale=args.ellipse_scale,
    )

    pre_message_position_error = position_error(
        estimated_local_states=pre_message_local_means,
        true_local_states=true_local_states,
    )

    posterior_position_error = position_error(
        estimated_local_states=posterior_local_means,
        true_local_states=true_local_states,
    )

    communication_by_agent = {
        agent_id: communication_for_agent(
            communication_decisions=communication_decisions,
            agent_id=agent_id,
            remote_agent_ids=remote_agent_ids,
            num_agents=num_agents,
            horizon=horizon,
        )
        for agent_id in remote_agent_ids
    }

    x_limits, y_limits = compute_scene_limits(
        true_local_states=true_local_states,
        posterior_local_means=posterior_local_means,
        landmark_positions=landmark_positions,
    )

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

    scene_ax = fig.add_subplot(grid[:, 0])
    radius_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[1, 1])

    color_map = plt.get_cmap("tab10")
    agent_colors = {
        agent_id: color_map(agent_id)
        for agent_id in range(num_agents)
    }

    max_radius = float(
        np.max(
            [
                np.max(pre_message_position_radius[:, remote_agent_ids]),
                np.max(posterior_position_radius[:, remote_agent_ids]),
                args.position_trigger_radius
                if args.position_trigger_radius is not None
                else 0.0,
            ]
        )
    )

    max_error = float(
        np.max(
            [
                np.max(pre_message_position_error[:, remote_agent_ids]),
                np.max(posterior_position_error[:, remote_agent_ids]),
            ]
        )
    )

    radius_ymax = max(max_radius * 1.2, 1.0e-3)
    error_ymax = max(max_error * 1.2, 1.0e-3)

    def draw_frame(frame_index: int):
        scene_ax.clear()
        radius_ax.clear()
        error_ax.clear()

        step = int(time_indices[frame_index])

        communicated_agents = [
            agent_id
            for agent_id in remote_agent_ids
            if communication_by_agent[agent_id][frame_index]
        ]

        communication_text = (
            "none"
            if not communicated_agents
            else ", ".join(
                f"agent {agent_id}"
                for agent_id in communicated_agents
            )
        )

        scene_ax.set_title(
            "Global scene: true states and ego belief"
        )
        scene_ax.set_xlabel("x position")
        scene_ax.set_ylabel("y position")
        scene_ax.set_xlim(*x_limits)
        scene_ax.set_ylim(*y_limits)
        scene_ax.set_aspect("equal", adjustable="box")
        scene_ax.grid(True, alpha=0.25)

        scene_ax.text(
            0.02,
            0.98,
            f"Step {step}\nCommunication: {communication_text}",
            transform=scene_ax.transAxes,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

        if landmark_positions is not None and landmark_positions.size > 0:
            scene_ax.scatter(
                landmark_positions[:, 0],
                landmark_positions[:, 1],
                marker="*",
                s=120,
                c="black",
                label="Landmark",
            )

        for agent_id in range(num_agents):
            color = agent_colors[agent_id]

            true_position = true_local_states[
                frame_index,
                agent_id,
                :2,
            ]

            posterior_position = posterior_local_means[
                frame_index,
                agent_id,
                :2,
            ]

            scene_ax.scatter(
                true_position[0],
                true_position[1],
                marker="o",
                s=45,
                color=color,
                label=(
                    "True state"
                    if agent_id == 0
                    else None
                ),
            )

            scene_ax.scatter(
                posterior_position[0],
                posterior_position[1],
                marker="X",
                s=55,
                color=color,
                label=(
                    "Posterior belief mean"
                    if agent_id == 0
                    else None
                ),
            )

            add_covariance_ellipse(
                ax=scene_ax,
                mean=posterior_local_means[frame_index, agent_id],
                covariance=posterior_local_covariances[
                    frame_index,
                    agent_id,
                ],
                scale=args.ellipse_scale,
                edgecolor=color,
                linestyle="-",
                linewidth=1.5,
                label=(
                    f"{args.ellipse_scale:.1f}σ posterior ellipse"
                    if agent_id == 0
                    else None
                ),
            )

            if agent_id in communicated_agents:
                pre_position = pre_message_local_means[
                    frame_index,
                    agent_id,
                    :2,
                ]

                scene_ax.scatter(
                    pre_position[0],
                    pre_position[1],
                    marker="x",
                    s=60,
                    color=color,
                    label=(
                        "Pre-message belief mean"
                        if agent_id == communicated_agents[0]
                        else None
                    ),
                )

                scene_ax.plot(
                    [pre_position[0], posterior_position[0]],
                    [pre_position[1], posterior_position[1]],
                    linestyle="--",
                    color=color,
                    linewidth=1.2,
                    label=(
                        "Message correction"
                        if agent_id == communicated_agents[0]
                        else None
                    ),
                )

                add_covariance_ellipse(
                    ax=scene_ax,
                    mean=pre_message_local_means[
                        frame_index,
                        agent_id,
                    ],
                    covariance=pre_message_local_covariances[
                        frame_index,
                        agent_id,
                    ],
                    scale=args.ellipse_scale,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=1.2,
                )

        scene_ax.legend(loc="lower left", fontsize=8)

        current_step = int(time_indices[frame_index])

        for agent_id in remote_agent_ids:
            color = agent_colors[agent_id]

            radius_ax.plot(
                time_indices,
                pre_message_position_radius[:, agent_id],
                color=color,
                linestyle="-",
                label=f"Agent {agent_id}: before message",
            )

            radius_ax.plot(
                time_indices,
                posterior_position_radius[:, agent_id],
                color=color,
                linestyle="--",
                label=f"Agent {agent_id}: posterior",
            )

            message_mask = communication_by_agent[agent_id]

            radius_ax.scatter(
                time_indices[message_mask],
                pre_message_position_radius[message_mask, agent_id],
                color=color,
                marker="v",
                s=35,
            )

            error_ax.plot(
                time_indices,
                pre_message_position_error[:, agent_id],
                color=color,
                linestyle="-",
                label=f"Agent {agent_id}: before message",
            )

            error_ax.plot(
                time_indices,
                posterior_position_error[:, agent_id],
                color=color,
                linestyle="--",
                label=f"Agent {agent_id}: posterior",
            )

        if args.position_trigger_radius is not None:
            radius_ax.axhline(
                args.position_trigger_radius,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Position trigger radius",
            )

        radius_ax.axvline(
            current_step,
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

        error_ax.axvline(
            current_step,
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

        radius_ax.set_title(
            "Remote position uncertainty and threshold"
        )
        radius_ax.set_xlabel("Environment step")
        radius_ax.set_ylabel(
            f"{args.ellipse_scale:.1f}σ position radius"
        )
        radius_ax.set_xlim(
            float(time_indices[0]),
            float(time_indices[-1]),
        )
        radius_ax.set_ylim(0.0, radius_ymax)
        radius_ax.grid(True, alpha=0.25)
        radius_ax.legend(loc="upper left", fontsize=8)

        error_ax.set_title(
            "Remote position-estimation error"
        )
        error_ax.set_xlabel("Environment step")
        error_ax.set_ylabel("Position error")
        error_ax.set_xlim(
            float(time_indices[0]),
            float(time_indices[-1]),
        )
        error_ax.set_ylim(0.0, error_ymax)
        error_ax.grid(True, alpha=0.25)
        error_ax.legend(loc="upper left", fontsize=8)

        fig.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=horizon,
        interval=1000 / args.fps,
        repeat=True,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = args.output_path.suffix.lower()

    if suffix == ".gif":
        writer = animation.PillowWriter(
            fps=args.fps,
        )
    else:
        writer = animation.FFMpegWriter(
            fps=args.fps,
            metadata={
                "title": args.output_path.name,
            },
        )

    ani.save(
        args.output_path,
        writer=writer,
        dpi=args.dpi,
    )

    plt.close(fig)

    print(f"Saved animation to: {args.output_path}")


if __name__ == "__main__":
    main()
