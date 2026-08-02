"""Seed-40 training-process animations for the three-actor policy tuple."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from private_mb_sac.agents.snapshots import exchange_actor_snapshots
from private_mb_sac.core.rng import forced_environment_key
from private_mb_sac.envs.model_state import split_model_state
from private_mb_sac.rollout.real_collector import (
    normalize_actor_observations,
    resolve_episode_done,
)


_AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
_LANDMARK_COLOR = "#666666"


def _positions_from_model_state(model_state):
    local_states, landmarks = split_model_state(
        model_state,
        num_agents=3,
        num_landmarks=3,
    )
    return (
        np.asarray(local_states[..., :2], dtype=np.float32),
        np.asarray(landmarks, dtype=np.float32),
    )


def rollout_seed40_episode(
    *,
    adapter: Any,
    actor_apply_fn: Any,
    actor_params: Sequence[Any],
    normalizer: Any,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Run one deterministic real-environment episode from exact seed 40."""

    if horizon < 1:
        raise ValueError("horizon must be positive.")

    snapshot_bank = exchange_actor_snapshots(
        tuple(actor_params),
        synchronization_round=0,
    )
    root = forced_environment_key()
    env_state, features = adapter.reset(root)

    agents, landmarks = _positions_from_model_state(
        features.model_state
    )
    agent_positions = [agents]
    landmark_positions = [landmarks]
    actions_history = []
    rewards_history = []
    collision_history = []
    minimum_distance_history = []

    for step_index in range(horizon):
        raw_observations = adapter.build_actor_observations(
            features,
            features.local_states,
        )
        actor_observations = normalize_actor_observations(
            raw_observations,
            normalizer,
        )
        means, _ = actor_apply_fn(
            {"params": snapshot_bank.params_by_agent},
            actor_observations,
        )
        actions = jnp.tanh(means)
        step_key = jax.random.fold_in(root, step_index + 1)
        output = adapter.step(step_key, env_state, actions)
        episode_done, _, _ = resolve_episode_done(adapter, output)

        env_state = output.env_state
        features = output.features
        agents, landmarks = _positions_from_model_state(
            features.model_state
        )
        agent_positions.append(agents)
        landmark_positions.append(landmarks)
        actions_history.append(
            np.asarray(jax.device_get(actions), dtype=np.float32)
        )
        rewards_history.append(
            float(
                np.asarray(
                    jax.device_get(jnp.mean(output.rewards))
                )
            )
        )
        collision_history.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "pair_collision_rate",
                            jnp.asarray(0.0, dtype=jnp.float32),
                        )
                    )
                )
            )
        )
        minimum_distance_history.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "min_pair_distance",
                            jnp.asarray(jnp.nan, dtype=jnp.float32),
                        )
                    )
                )
            )
        )

        if bool(np.asarray(jax.device_get(episode_done))):
            break

    return {
        "agent_positions": np.asarray(
            agent_positions,
            dtype=np.float32,
        ),
        "landmark_positions": np.asarray(
            landmark_positions,
            dtype=np.float32,
        ),
        "actions": np.asarray(actions_history, dtype=np.float32),
        "rewards": np.asarray(rewards_history, dtype=np.float32),
        "collision_rates": np.asarray(
            collision_history,
            dtype=np.float32,
        ),
        "minimum_distances": np.asarray(
            minimum_distance_history,
            dtype=np.float32,
        ),
    }


def _plot_limits(trajectory: dict[str, np.ndarray]):
    all_points = np.concatenate(
        [
            trajectory["agent_positions"].reshape(-1, 2),
            trajectory["landmark_positions"].reshape(-1, 2),
        ],
        axis=0,
    )
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.0)
    half = 0.5 * span + 0.25
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
    )


def save_round_animation(
    *,
    adapter: Any,
    actor_apply_fn: Any,
    actor_params: Sequence[Any],
    normalizer: Any,
    round_index: int,
    output_directory: str | Path,
    horizon: int = 25,
    fps: int = 5,
    dpi: int = 140,
    trail_length: int = 25,
    save_gif: bool = True,
    save_mp4: bool = True,
) -> dict[str, Any]:
    """Render one seed-40 GIF and optional MP4 for a training round."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    trajectory = rollout_seed40_episode(
        adapter=adapter,
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_params,
        normalizer=normalizer,
        horizon=horizon,
    )
    episode_length = int(len(trajectory["rewards"]))
    limits = _plot_limits(trajectory)

    figure, axis = plt.subplots(figsize=(6.6, 6.2))

    def draw(frame_index):
        axis.clear()
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(
            f"Private model-based SAC | round {round_index} | env seed 40"
        )

        state_index = min(frame_index, episode_length)
        start_index = max(0, state_index - trail_length)

        landmarks = trajectory["landmark_positions"][state_index]
        for landmark_id, position in enumerate(landmarks):
            axis.add_patch(
                Circle(
                    position,
                    0.10,
                    facecolor=_LANDMARK_COLOR,
                    edgecolor="black",
                    alpha=0.35,
                )
            )
            axis.text(
                position[0],
                position[1],
                f"L{landmark_id}",
                ha="center",
                va="center",
                fontsize=8,
            )

        collision_now = (
            state_index > 0
            and trajectory["collision_rates"][state_index - 1] > 0.0
        )
        for agent_id in range(3):
            trail = trajectory["agent_positions"][
                start_index : state_index + 1,
                agent_id,
            ]
            axis.plot(
                trail[:, 0],
                trail[:, 1],
                color=_AGENT_COLORS[agent_id],
                linewidth=1.8,
                alpha=0.70,
            )
            position = trajectory["agent_positions"][
                state_index,
                agent_id,
            ]
            axis.add_patch(
                Circle(
                    position,
                    0.075,
                    facecolor=_AGENT_COLORS[agent_id],
                    edgecolor="#CC79A7" if collision_now else "black",
                    linewidth=3.0 if collision_now else 1.2,
                    alpha=0.95,
                )
            )
            axis.text(
                position[0],
                position[1],
                f"A{agent_id}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

            if frame_index < episode_length:
                action = trajectory["actions"][
                    frame_index,
                    agent_id,
                ]
                axis.arrow(
                    position[0],
                    position[1],
                    0.12 * action[0],
                    0.12 * action[1],
                    width=0.006,
                    head_width=0.035,
                    length_includes_head=True,
                    color=_AGENT_COLORS[agent_id],
                    alpha=0.85,
                )

        cumulative_return = float(
            np.sum(trajectory["rewards"][:state_index])
        )
        collision = (
            float(trajectory["collision_rates"][state_index - 1])
            if state_index > 0 else 0.0
        )
        minimum_distance = (
            float(trajectory["minimum_distances"][state_index - 1])
            if state_index > 0 else float("nan")
        )
        axis.text(
            0.02,
            0.98,
            (
                f"step {state_index:02d}/{episode_length:02d}\n"
                f"return {cumulative_return:.3f}\n"
                f"collision {collision:.3f}\n"
                f"min distance {minimum_distance:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )
        return ()

    animator = mpl_animation.FuncAnimation(
        figure,
        draw,
        frames=episode_length + 1,
        interval=1000 / max(int(fps), 1),
        repeat=True,
        blit=False,
    )

    stem = f"seed40_round_{round_index:04d}"
    outputs: dict[str, Any] = {}
    if save_gif:
        gif_path = output_directory / f"{stem}.gif"
        animator.save(
            gif_path,
            writer=mpl_animation.PillowWriter(fps=fps),
            dpi=dpi,
        )
        outputs["gif"] = str(gif_path)

    if save_mp4 and shutil.which("ffmpeg") is not None:
        mp4_path = output_directory / f"{stem}.mp4"
        animator.save(
            mp4_path,
            writer=mpl_animation.FFMpegWriter(
                fps=fps,
                bitrate=2400,
            ),
            dpi=dpi,
        )
        outputs["mp4"] = str(mp4_path)
    elif save_mp4:
        outputs["mp4_skipped"] = "ffmpeg_not_found"

    plt.close(figure)

    summary = {
        "round": int(round_index),
        "environment_seed": 40,
        "episode_length": episode_length,
        "episode_return": float(np.sum(trajectory["rewards"])),
        "mean_collision_rate": float(
            np.mean(trajectory["collision_rates"])
            if episode_length else 0.0
        ),
        "minimum_pair_distance": float(
            np.nanmin(trajectory["minimum_distances"])
            if episode_length else np.nan
        ),
        "mean_action_by_agent": (
            np.mean(trajectory["actions"], axis=0).tolist()
            if episode_length else []
        ),
        "action_saturation_rate_by_agent": (
            (np.abs(trajectory["actions"]) > 0.95)
            .mean(axis=(0, 2))
            .tolist()
            if episode_length else []
        ),
        "outputs": outputs,
    }
    summary_path = output_directory / f"{stem}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    outputs["summary"] = str(summary_path)
    return summary
