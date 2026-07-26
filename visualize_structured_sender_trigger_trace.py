#!/usr/bin/env python3
"""Visualize a structured covariance-triggered estimator trace.

Reads covariance_trigger_estimator_trace.npz and produces:
  * all_agents_three_way.png
  * all_agents_true_estimator.png
  * a synchronized MP4/GIF animation

Animation layout:
  left: true and estimator robots, headings, trails, uncertainty ellipses
  right top: prior/posterior position uncertainty and threshold
  right bottom: prior/posterior position error
  both right panels: a moving vertical time cursor
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

AGENT_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
)
START_COLOR = "#009E73"
END_COLOR = "#D55E00"
COMMUNICATION_COLOR = "#CC79A7"
THRESHOLD_COLOR = "#D55E00"
CURSOR_COLOR = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "results/figures/structured_sender_position_trigger/"
            "sender_trigger_estimator_trace.npz"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/animations/structured_sender_position_trigger"
        ),
    )
    parser.add_argument(
        "--animation-name",
        default="structured_sender_position_trigger.mp4",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--agent-radius", type=float, default=0.11)
    parser.add_argument(
        "--track-offset-scale",
        type=float,
        default=1.25,
        help=(
            "Side-track offset from the robot center, expressed as a "
            "multiple of agent-radius."
        ),
    )
    parser.add_argument(
        "--track-half-length-scale",
        type=float,
        default=1.35,
        help=(
            "Half-length of each tank track, expressed as a multiple "
            "of agent-radius."
        ),
    )
    parser.add_argument(
        "--track-linewidth",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--front-marker-scale",
        type=float,
        default=0.72,
        help=(
            "Distance of the small front marker from the center, as a "
            "multiple of agent-radius. Set to 0 to hide it."
        ),
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help=(
            "Number of recent trajectory points to retain. "
            "Use 0 (default) to keep the complete route visible."
        ),
    )
    parser.add_argument(
        "--ellipse-scale",
        type=float,
        default=-1.0,
        help="Negative means use position_trigger_scale saved in the trace.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--show-communication-markers", action="store_true")
    return parser.parse_args()



def load_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trace not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        trace = {key: np.asarray(data[key]) for key in data.files}

    required = {
        "truth",
        "model_only",
        "estimator_prior",
        "estimator_posterior",
        "posterior_covariances",
        "pre_message_position_radii",
        "post_message_position_radii",
        "pre_message_position_errors",
        "post_message_position_errors",
        "communications",
        "sender_error_threshold",
        "sender_covariance_radius_threshold",
        "sender_covariance_scale",
    }
    missing = sorted(required.difference(trace))
    if missing:
        raise KeyError(f"Trace is missing required arrays: {missing}")

    truth = trace["truth"]
    model_only = trace["model_only"]
    prior = trace["estimator_prior"]
    posterior = trace["estimator_posterior"]
    covariances = trace["posterior_covariances"]

    if truth.ndim != 3 or truth.shape[-1] != 5:
        raise ValueError("truth must have shape (H+1, N, 5).")
    if model_only.shape != truth.shape:
        raise ValueError("model_only must match truth shape.")
    if posterior.shape != truth.shape:
        raise ValueError("estimator_posterior must match truth shape.")
    if prior.shape != truth[1:].shape:
        raise ValueError("estimator_prior must have shape (H, N, 5).")

    horizon = truth.shape[0] - 1
    num_agents = truth.shape[1]
    global_dim = num_agents * 5

    if covariances.shape != (horizon, global_dim, global_dim):
        raise ValueError(
            "posterior_covariances must have shape "
            f"{(horizon, global_dim, global_dim)}, got {covariances.shape}."
        )

    matrix_shape = (horizon, num_agents)
    for key in (
        "pre_message_position_radii",
        "post_message_position_radii",
        "pre_message_position_errors",
        "post_message_position_errors",
        "communications",
    ):
        if trace[key].shape != matrix_shape:
            raise ValueError(
                f"{key} must have shape {matrix_shape}, "
                f"got {trace[key].shape}."
            )

    return trace


def agent_color(agent_id: int) -> str:
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def add_start_end(axis: plt.Axes, trajectory: np.ndarray) -> None:
    axis.scatter(
        trajectory[0, 0], trajectory[0, 1],
        s=85, marker="o", facecolor=START_COLOR,
        edgecolor="white", linewidth=1.1, zorder=8,
    )
    axis.scatter(
        trajectory[-1, 0], trajectory[-1, 1],
        s=105, marker="X", facecolor=END_COLOR,
        edgecolor="white", linewidth=1.1, zorder=8,
    )



def plot_all_agents_three_way(
    truth: np.ndarray,
    model_only: np.ndarray,
    posterior: np.ndarray,
    communications: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 8.5))

    for agent_id in range(truth.shape[1]):
        color = agent_color(agent_id)
        axis.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            color=color,
            linewidth=2.6,
            linestyle="-",
        )
        axis.plot(
            model_only[:, agent_id, 0],
            model_only[:, agent_id, 1],
            color=color,
            linewidth=1.9,
            linestyle=":",
            alpha=0.80,
        )
        axis.plot(
            posterior[:, agent_id, 0],
            posterior[:, agent_id, 1],
            color=color,
            linewidth=2.1,
            linestyle="--",
            alpha=0.95,
        )
        add_start_end(axis, truth[:, agent_id, :2])

        steps = np.flatnonzero(communications[:, agent_id]) + 1
        if steps.size:
            axis.scatter(
                posterior[steps, agent_id, 0],
                posterior[steps, agent_id, 1],
                s=45,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=9,
            )

    agent_handles = [
        Line2D(
            [0],
            [0],
            color=agent_color(i),
            linewidth=3,
            label=f"agent {i}",
        )
        for i in range(truth.shape[1])
    ]
    method_handles = [
        Line2D(
            [0], [0], color="#222222", linewidth=2.5,
            linestyle="-", label="true"
        ),
        Line2D(
            [0], [0], color="#222222", linewidth=2.0,
            linestyle=":", label="model-only"
        ),
        Line2D(
            [0], [0], color="#222222", linewidth=2.0,
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

    legend_agents = axis.legend(
        handles=agent_handles,
        loc="upper left",
        title="Agent color",
    )
    axis.add_artist(legend_agents)
    axis.legend(
        handles=method_handles,
        loc="lower right",
        title="Meaning",
    )

    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_title(
        "Sender-triggered common belief: all-agent trajectories"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.32)
    save_figure(figure, output, dpi)



def plot_all_agents_true_estimator(
    truth: np.ndarray,
    posterior: np.ndarray,
    communications: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 8.5))

    for agent_id in range(truth.shape[1]):
        color = agent_color(agent_id)
        axis.plot(
            truth[:, agent_id, 0],
            truth[:, agent_id, 1],
            color=color,
            linewidth=2.7,
            linestyle="-",
        )
        axis.plot(
            posterior[:, agent_id, 0],
            posterior[:, agent_id, 1],
            color=color,
            linewidth=2.1,
            linestyle="--",
            alpha=0.92,
        )
        add_start_end(axis, truth[:, agent_id, :2])

        steps = np.flatnonzero(communications[:, agent_id]) + 1
        if steps.size:
            axis.scatter(
                posterior[steps, agent_id, 0],
                posterior[steps, agent_id, 1],
                s=45,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.3,
                zorder=9,
            )

    agent_handles = [
        Line2D(
            [0],
            [0],
            color=agent_color(i),
            linewidth=3,
            label=f"agent {i}",
        )
        for i in range(truth.shape[1])
    ]
    method_handles = [
        Line2D(
            [0], [0], color="#222222", linewidth=2.5,
            linestyle="-", label="true"
        ),
        Line2D(
            [0], [0], color="#222222", linewidth=2.1,
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

    legend_agents = axis.legend(
        handles=agent_handles,
        loc="upper left",
        title="Agent color",
    )
    axis.add_artist(legend_agents)
    axis.legend(
        handles=method_handles,
        loc="lower right",
        title="Meaning",
    )

    axis.set_xlabel("x position")
    axis.set_ylabel("y position")
    axis.set_title(
        "Sender-triggered common belief: true and posterior"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.32)
    save_figure(figure, output, dpi)


def covariance_ellipse_parameters(
    covariance_2d: np.ndarray,
    scale: float,
) -> tuple[float, float, float]:
    covariance_2d = 0.5 * (covariance_2d + covariance_2d.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_2d)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major = eigenvectors[:, 0]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    width = 2.0 * scale * np.sqrt(eigenvalues[0])
    height = 2.0 * scale * np.sqrt(eigenvalues[1])
    return float(width), float(height), float(angle)


def tank_track_segments(
    x: float,
    y: float,
    heading: float,
    radius: float,
    track_offset_scale: float,
    track_half_length_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right tank-track line segments.

    The track direction is parallel to the robot heading. The two tracks are
    offset along the heading-normal direction, producing a top-down tank-like
    glyph rather than two forward-facing rays.
    """
    heading_vector = np.asarray([
        np.cos(heading),
        np.sin(heading),
    ])
    normal_vector = np.asarray([
        -np.sin(heading),
        np.cos(heading),
    ])

    center = np.asarray([x, y])
    track_offset = track_offset_scale * radius
    half_length = track_half_length_scale * radius

    segments = []
    for side in (-1.0, 1.0):
        track_center = center + side * track_offset * normal_vector
        start = track_center - half_length * heading_vector
        end = track_center + half_length * heading_vector
        segments.append(np.stack([start, end], axis=0))

    return segments[0], segments[1]


def front_marker_position(
    x: float,
    y: float,
    heading: float,
    radius: float,
    front_marker_scale: float,
) -> tuple[float, float]:
    """Return a small marker position indicating the forward direction."""
    distance = front_marker_scale * radius
    return (
        float(x + distance * np.cos(heading)),
        float(y + distance * np.sin(heading)),
    )


def position_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimate[..., :2] - truth[..., :2], axis=-1)


def scene_limits(
    truth: np.ndarray,
    posterior: np.ndarray,
) -> tuple[float, float, float, float]:
    positions = np.concatenate([truth[..., :2], posterior[..., :2]], axis=0)
    x_min = float(np.min(positions[..., 0]))
    x_max = float(np.max(positions[..., 0]))
    y_min = float(np.min(positions[..., 1]))
    y_max = float(np.max(positions[..., 1]))
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    return (
        x_min - 0.10 * x_range,
        x_max + 0.10 * x_range,
        y_min - 0.10 * y_range,
        y_max + 0.10 * y_range,
    )


def create_animation(
    trace: dict[str, Any],
    output_path: Path,
    fps: int,
    dpi: int,
    agent_radius: float,
    track_offset_scale: float,
    track_half_length_scale: float,
    track_linewidth: float,
    front_marker_scale: float,
    trail_length: int,
    ellipse_scale: float,
    frame_step: int,
    show_communication_markers: bool,
) -> Path:
    truth = trace["truth"]
    prior = trace["estimator_prior"]
    posterior = trace["estimator_posterior"]
    covariances = trace["posterior_covariances"]
    pre_radii = trace["pre_message_position_radii"]
    post_radii = trace["post_message_position_radii"]
    pre_error = trace["pre_message_position_errors"]
    post_error = trace["post_message_position_errors"]
    communications = trace["communications"]
    error_threshold = float(trace["sender_error_threshold"])
    covariance_threshold = float(
        trace["sender_covariance_radius_threshold"]
    )

    horizon = truth.shape[0] - 1
    num_agents = truth.shape[1]
    if track_offset_scale <= 0.0:
        raise ValueError("track-offset-scale must be positive.")
    if track_half_length_scale <= 0.0:
        raise ValueError("track-half-length-scale must be positive.")
    if track_linewidth <= 0.0:
        raise ValueError("track-linewidth must be positive.")
    if front_marker_scale < 0.0:
        raise ValueError("front-marker-scale must be non-negative.")
    if frame_step < 1:
        raise ValueError("frame-step must be at least 1.")

    frame_indices = np.arange(1, horizon + 1, frame_step, dtype=np.int32)
    if frame_indices[-1] != horizon:
        frame_indices = np.concatenate([frame_indices, np.asarray([horizon])])

    time = np.arange(1, horizon + 1)

    x_min, x_max, y_min, y_max = scene_limits(truth, posterior)
    radius_max = max(
        float(np.max(pre_radii)),
        float(np.max(post_radii)),
        covariance_threshold,
        1e-6,
    )
    error_max = max(
        float(np.max(pre_error)),
        float(np.max(post_error)),
        error_threshold,
        1e-6,
    )

    figure = plt.figure(figsize=(17.5, 8.8))
    grid = figure.add_gridspec(
        2, 2,
        width_ratios=(1.72, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.18,
        hspace=0.26,
    )
    scene_axis = figure.add_subplot(grid[:, 0])
    uncertainty_axis = figure.add_subplot(grid[0, 1])
    error_axis = figure.add_subplot(grid[1, 1])

    scene_axis.set_xlim(x_min, x_max)
    scene_axis.set_ylim(y_min, y_max)
    scene_axis.set_aspect("equal", adjustable="box")
    scene_axis.set_xlabel("x position")
    scene_axis.set_ylabel("y position")
    scene_axis.set_title("True state and sender-triggered common belief")
    scene_axis.grid(True, alpha=0.28)

    uncertainty_axis.set_xlim(1, horizon)
    uncertainty_axis.set_ylim(0.0, radius_max * 1.10)
    uncertainty_axis.set_xlabel("environment step")
    uncertainty_axis.set_ylabel(f"{ellipse_scale:g}σ position radius")
    uncertainty_axis.set_title("All-agent position uncertainty and threshold")
    uncertainty_axis.grid(True, alpha=0.28)

    error_axis.set_xlim(1, horizon)
    error_axis.set_ylim(0.0, error_max * 1.10)
    error_axis.set_xlabel("environment step")
    error_axis.set_ylabel("position error")
    error_axis.set_title("All-agent position prediction error")
    error_axis.grid(True, alpha=0.28)

    uncertainty_axis.axhline(
        covariance_threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"covariance threshold = {covariance_threshold:g}",
    )

    error_axis.axhline(
        error_threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"error threshold = {error_threshold:g}",
    )

    true_bodies = []
    estimator_bodies = []
    ellipses = []
    true_track_lines = []
    estimator_track_lines = []
    true_front_markers = []
    estimator_front_markers = []
    true_trails = []
    estimator_trails = []
    communication_markers = []

    for agent_id in range(num_agents):
        color = agent_color(agent_id)
        true_body = Circle(
            (0.0, 0.0), agent_radius,
            facecolor=color, edgecolor="#111111",
            linewidth=1.1, alpha=0.90, zorder=6,
        )
        estimator_body = Circle(
            (0.0, 0.0), agent_radius * 0.92,
            facecolor="none", edgecolor=color,
            linewidth=2.0, linestyle="--", zorder=7,
        )
        ellipse = Ellipse(
            (0.0, 0.0), 0.0, 0.0,
            facecolor=color, edgecolor=color,
            linewidth=1.5, linestyle=":", alpha=0.16, zorder=2,
        )
        scene_axis.add_patch(ellipse)
        scene_axis.add_patch(true_body)
        scene_axis.add_patch(estimator_body)

        true_track_1, = scene_axis.plot(
            [], [],
            color="#222222",
            linewidth=track_linewidth,
            linestyle="-",
            solid_capstyle="round",
            zorder=8,
        )
        true_track_2, = scene_axis.plot(
            [], [],
            color="#222222",
            linewidth=track_linewidth,
            linestyle="-",
            solid_capstyle="round",
            zorder=8,
        )
        est_track_1, = scene_axis.plot(
            [], [],
            color=color,
            linewidth=max(track_linewidth - 0.8, 1.0),
            linestyle="--",
            dash_capstyle="round",
            zorder=9,
        )
        est_track_2, = scene_axis.plot(
            [], [],
            color=color,
            linewidth=max(track_linewidth - 0.8, 1.0),
            linestyle="--",
            dash_capstyle="round",
            zorder=9,
        )
        true_front_marker, = scene_axis.plot(
            [], [],
            marker="o",
            markersize=4.2,
            markerfacecolor="#111111",
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="None",
            zorder=10,
        )
        est_front_marker, = scene_axis.plot(
            [], [],
            marker="o",
            markersize=4.2,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
            linestyle="None",
            zorder=10,
        )
        true_trail, = scene_axis.plot([], [], color=color, linewidth=2.2, alpha=0.52, zorder=3)
        est_trail, = scene_axis.plot([], [], color=color, linewidth=1.8, linestyle="--", alpha=0.70, zorder=4)
        comm_marker, = scene_axis.plot(
            [], [], marker="D", markersize=7.5,
            markerfacecolor="none", markeredgecolor=COMMUNICATION_COLOR,
            markeredgewidth=1.6, linestyle="None", zorder=10,
        )

        true_bodies.append(true_body)
        estimator_bodies.append(estimator_body)
        ellipses.append(ellipse)
        true_track_lines.append((true_track_1, true_track_2))
        estimator_track_lines.append((est_track_1, est_track_2))
        true_front_markers.append(true_front_marker)
        estimator_front_markers.append(est_front_marker)
        true_trails.append(true_trail)
        estimator_trails.append(est_trail)
        communication_markers.append(comm_marker)

    uncertainty_lines = {}
    error_lines = {}
    for agent_id in range(num_agents):
        color = agent_color(agent_id)
        line_prior, = uncertainty_axis.plot(
            time,
            pre_radii[:, agent_id],
            color=color,
            linewidth=2.1,
            label=f"agent {agent_id}: before message",
        )
        line_post, = uncertainty_axis.plot(
            time,
            post_radii[:, agent_id],
            color=color,
            linewidth=1.9,
            linestyle="--",
            label=f"agent {agent_id}: posterior",
        )
        uncertainty_lines[(agent_id, "prior")] = line_prior
        uncertainty_lines[(agent_id, "posterior")] = line_post

        error_prior, = error_axis.plot(
            time,
            pre_error[:, agent_id],
            color=color,
            linewidth=2.1,
            label=f"agent {agent_id}: before message",
        )
        error_post, = error_axis.plot(
            time,
            post_error[:, agent_id],
            color=color,
            linewidth=1.9,
            linestyle="--",
            label=f"agent {agent_id}: posterior",
        )
        error_lines[(agent_id, "prior")] = error_prior
        error_lines[(agent_id, "posterior")] = error_post

        event_mask = communications[:, agent_id]
        if np.any(event_mask):
            uncertainty_axis.scatter(
                time[event_mask],
                pre_radii[event_mask, agent_id],
                s=34,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.2,
                zorder=5,
            )
            error_axis.scatter(
                time[event_mask],
                pre_error[event_mask, agent_id],
                s=34,
                marker="D",
                facecolor="none",
                edgecolor=COMMUNICATION_COLOR,
                linewidth=1.2,
                zorder=5,
            )

    uncertainty_axis.legend(fontsize=8.5, loc="upper left")
    error_axis.legend(fontsize=8.5, loc="upper left")

    uncertainty_cursor = uncertainty_axis.axvline(1, color=CURSOR_COLOR, linestyle=":", linewidth=1.8)
    error_cursor = error_axis.axvline(1, color=CURSOR_COLOR, linestyle=":", linewidth=1.8)

    status = scene_axis.text(
        0.015, 0.985, "",
        transform=scene_axis.transAxes,
        ha="left", va="top", fontsize=10.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "#444444",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.3",
        },
        zorder=20,
    )

    role_handles = [
        Line2D(
            [0], [0], color=agent_color(i), linewidth=3,
            label=f"agent {i}",
        )
        for i in range(num_agents)
    ]
    meaning_handles = [
        Line2D(
            [0], [0],
            color="#111111",
            marker="o",
            markerfacecolor="#777777",
            linewidth=3.2,
            linestyle="-",
            label="true tank body/tracks",
        ),
        Line2D(
            [0], [0],
            color="#555555",
            marker="o",
            markerfacecolor="none",
            linewidth=2.6,
            linestyle="--",
            label="estimator tank body/tracks",
        ),
        Line2D([0], [0], color="#555555", linewidth=4, linestyle=":", alpha=0.45, label=f"{ellipse_scale:g}σ position ellipse"),
    ]
    scene_legend_handles = role_handles + meaning_handles
    scene_axis.legend(
        handles=scene_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        fontsize=8.2,
        frameon=True,
        title="Agent colors and drawing meaning",
        borderaxespad=0.0,
    )

    def update(frame_number: int):
        step = int(frame_indices[frame_number])
        transition_index = step - 1
        trail_start = (
            0
            if trail_length <= 0
            else max(0, step - trail_length)
        )

        triggered = [
            int(i)
            for i in np.flatnonzero(
                communications[transition_index]
            )
        ]

        if not triggered:
            comm_text = "none"
        else:
            descriptions = []
            for agent_id in triggered:
                error_hit = (
                    pre_error[transition_index, agent_id]
                    > error_threshold
                )
                covariance_hit = (
                    pre_radii[transition_index, agent_id]
                    > covariance_threshold
                )
                if error_hit and covariance_hit:
                    reason = "error+cov"
                elif error_hit:
                    reason = "error"
                elif covariance_hit:
                    reason = "cov"
                else:
                    reason = "broadcast"
                descriptions.append(
                    f"agent {agent_id} ({reason})"
                )
            comm_text = ", ".join(descriptions)

        status.set_text(
            f"Step {step}\nSender broadcast: {comm_text}"
        )

        for agent_id in range(num_agents):
            true_state = truth[step, agent_id]
            est_state = posterior[step, agent_id]
            true_bodies[agent_id].center = (float(true_state[0]), float(true_state[1]))
            estimator_bodies[agent_id].center = (float(est_state[0]), float(est_state[1]))

            true_segments = tank_track_segments(
                x=float(true_state[0]),
                y=float(true_state[1]),
                heading=float(true_state[2]),
                radius=agent_radius,
                track_offset_scale=track_offset_scale,
                track_half_length_scale=track_half_length_scale,
            )
            est_segments = tank_track_segments(
                x=float(est_state[0]),
                y=float(est_state[1]),
                heading=float(est_state[2]),
                radius=agent_radius * 0.92,
                track_offset_scale=track_offset_scale,
                track_half_length_scale=track_half_length_scale,
            )
            for line, segment in zip(
                true_track_lines[agent_id],
                true_segments,
            ):
                line.set_data(segment[:, 0], segment[:, 1])
            for line, segment in zip(
                estimator_track_lines[agent_id],
                est_segments,
            ):
                line.set_data(segment[:, 0], segment[:, 1])

            if front_marker_scale > 0.0:
                true_front = front_marker_position(
                    x=float(true_state[0]),
                    y=float(true_state[1]),
                    heading=float(true_state[2]),
                    radius=agent_radius,
                    front_marker_scale=front_marker_scale,
                )
                est_front = front_marker_position(
                    x=float(est_state[0]),
                    y=float(est_state[1]),
                    heading=float(est_state[2]),
                    radius=agent_radius * 0.92,
                    front_marker_scale=front_marker_scale,
                )
                true_front_markers[agent_id].set_data(
                    [true_front[0]],
                    [true_front[1]],
                )
                estimator_front_markers[agent_id].set_data(
                    [est_front[0]],
                    [est_front[1]],
                )
            else:
                true_front_markers[agent_id].set_data([], [])
                estimator_front_markers[agent_id].set_data([], [])

            true_trails[agent_id].set_data(
                truth[trail_start:step + 1, agent_id, 0],
                truth[trail_start:step + 1, agent_id, 1],
            )
            estimator_trails[agent_id].set_data(
                posterior[trail_start:step + 1, agent_id, 0],
                posterior[trail_start:step + 1, agent_id, 1],
            )

            start = agent_id * 5
            block = covariances[transition_index, start:start + 5, start:start + 5]
            width, height, angle = covariance_ellipse_parameters(block[:2, :2], ellipse_scale)
            ellipse = ellipses[agent_id]
            ellipse.center = (float(est_state[0]), float(est_state[1]))
            ellipse.width = width
            ellipse.height = height
            ellipse.angle = angle

            if show_communication_markers and communications[transition_index, agent_id]:
                communication_markers[agent_id].set_data([est_state[0]], [est_state[1]])
            else:
                communication_markers[agent_id].set_data([], [])

        # Right-side curves are fully visible from the first frame.
        # Only the synchronized vertical cursor moves.
        uncertainty_cursor.set_xdata([step, step])
        error_cursor.set_xdata([step, step])
        return []

    figure.subplots_adjust(bottom=0.17)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / float(fps),
        blit=False,
        repeat=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        actual_output = output_path
    elif animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=3500,
            metadata={"title": "Structured sender-triggered common belief"},
        )
        movie.save(output_path, writer=writer, dpi=dpi)
        actual_output = output_path
    else:
        actual_output = output_path.with_suffix(".gif")
        print("ffmpeg unavailable; writing GIF instead:", actual_output)
        movie.save(actual_output, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    plt.close(figure)
    return actual_output



def main() -> None:
    args = parse_args()
    trace = load_trace(args.trace)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    truth = trace["truth"]
    model_only = trace["model_only"]
    posterior = trace["estimator_posterior"]
    communications = trace["communications"]

    ellipse_scale = (
        float(trace["sender_covariance_scale"])
        if args.ellipse_scale < 0.0
        else args.ellipse_scale
    )
    if ellipse_scale <= 0.0:
        raise ValueError("ellipse-scale must be positive.")

    plot_all_agents_three_way(
        truth,
        model_only,
        posterior,
        communications,
        args.output_directory / "all_agents_sender_three_way.png",
        args.dpi,
    )
    plot_all_agents_true_estimator(
        truth,
        posterior,
        communications,
        args.output_directory
        / "all_agents_sender_true_estimator.png",
        args.dpi,
    )

    animation_output = create_animation(
        trace=trace,
        output_path=args.output_directory / args.animation_name,
        fps=args.fps,
        dpi=args.dpi,
        agent_radius=args.agent_radius,
        track_offset_scale=args.track_offset_scale,
        track_half_length_scale=args.track_half_length_scale,
        track_linewidth=args.track_linewidth,
        front_marker_scale=args.front_marker_scale,
        trail_length=args.trail_length,
        ellipse_scale=ellipse_scale,
        frame_step=args.frame_step,
        show_communication_markers=args.show_communication_markers,
    )

    print("Trace:", args.trace)
    print("Output directory:", args.output_directory)
    print("Static figures:")
    print(" - all_agents_sender_three_way.png")
    print(" - all_agents_sender_true_estimator.png")
    print("Animation:", animation_output)
    print(
        "Sender semantics: every agent evaluates its private position "
        "error and its common-belief covariance radius. A private "
        "observation enters the common belief only after broadcast."
    )


if __name__ == "__main__":
    main()
