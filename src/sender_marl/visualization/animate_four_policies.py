#!/usr/bin/env python3
"""Animate PS-IPPO, PS-MAPPO, Independent IPPO, and DPO on Simple Spread."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from sender_marl.core.distributions import deterministic_squashed_action
from sender_marl.core.networks import ContinuousActor
from sender_marl.envs import make_env_adapter
from sender_marl.on_policy.independent import make_independent_actor_apply


AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
LANDMARK_COLOR = "#777777"
COLLISION_COLOR = "#CC79A7"


@dataclass
class Trajectory:
    name: str
    checkpoint: str
    checkpoint_update: int | None
    parameter_mode: str
    agent_positions: np.ndarray
    landmark_positions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    collision_rates: np.ndarray
    min_pair_distances: np.ndarray

    @property
    def episode_length(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps-ippo", type=Path, required=True)
    parser.add_argument("--ps-mappo", type=Path, required=True)
    parser.add_argument("--independent-ippo", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/policy_animations"),
    )
    parser.add_argument("--env-seed", type=int, default=271828)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--trail-length", type=int, default=25)
    parser.add_argument("--agent-radius", type=float, default=0.075)
    parser.add_argument("--landmark-radius", type=float, default=0.10)
    parser.add_argument("--axis-padding", type=float, default=0.25)
    parser.add_argument("--no-mp4", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dictionary, got {type(checkpoint).__name__}"
        )
    if "actor_params" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no actor_params. Keys: {sorted(checkpoint.keys())}"
        )
    return checkpoint


def make_policy(actor_params: Any, num_agents: int):
    actor = ContinuousActor(action_dim=2)
    independent = (
        isinstance(actor_params, (tuple, list))
        and len(actor_params) == num_agents
    )

    if independent:
        apply_fn = make_independent_actor_apply(actor.apply, num_agents)

        def policy(obs):
            mean, _ = apply_fn({"params": actor_params}, obs)
            return deterministic_squashed_action(mean)

        return jax.jit(policy), "independent"

    def policy(obs):
        mean, _ = actor.apply({"params": actor_params}, obs)
        return deterministic_squashed_action(mean)

    return jax.jit(policy), "parameter-sharing"


def extract_positions(env_state, num_agents: int, num_landmarks: int):
    positions = np.asarray(jax.device_get(env_state.p_pos), dtype=np.float32)
    total = num_agents + num_landmarks
    if positions.ndim != 2 or positions.shape[0] < total:
        raise ValueError(f"Unexpected p_pos shape: {positions.shape}")
    return (
        positions[:num_agents].copy(),
        positions[num_agents:total].copy(),
    )


def rollout_policy(
    name: str,
    checkpoint_path: Path,
    env_seed: int,
    max_steps: int,
    num_agents: int,
    num_landmarks: int,
) -> Trajectory:
    checkpoint = load_checkpoint(checkpoint_path)
    policy, parameter_mode = make_policy(
        checkpoint["actor_params"],
        num_agents,
    )

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        max_steps=max_steps,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )

    # Same reset key for every algorithm.
    env_state, features = adapter.reset(jax.random.PRNGKey(env_seed))
    agents, landmarks = extract_positions(
        env_state,
        num_agents,
        num_landmarks,
    )

    agent_positions = [agents]
    landmark_positions = [landmarks]
    actions_history = []
    rewards = []
    collision_rates = []
    min_pair_distances = []

    for step_index in range(max_steps):
        actor_obs = adapter.build_actor_observations(
            features,
            features.local_states,
        )
        actions = policy(actor_obs)

        # Same step-key schedule for every algorithm.
        step_key = jax.random.fold_in(
            jax.random.PRNGKey(env_seed),
            step_index + 1,
        )
        output = adapter.step(step_key, env_state, actions)
        env_state = output.env_state
        features = output.features

        agents, landmarks = extract_positions(
            env_state,
            num_agents,
            num_landmarks,
        )
        agent_positions.append(agents)
        landmark_positions.append(landmarks)
        actions_history.append(
            np.asarray(jax.device_get(actions), dtype=np.float32)
        )
        rewards.append(
            float(np.asarray(jax.device_get(jnp.mean(output.rewards))))
        )
        collision_rates.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "pair_collision_rate",
                            jnp.asarray(0.0),
                        )
                    )
                )
            )
        )
        min_pair_distances.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "min_pair_distance",
                            jnp.asarray(np.nan),
                        )
                    )
                )
            )
        )

        if bool(np.asarray(jax.device_get(output.episode_done))):
            break

    return Trajectory(
        name=name,
        checkpoint=str(checkpoint_path),
        checkpoint_update=checkpoint.get("update"),
        parameter_mode=parameter_mode,
        agent_positions=np.asarray(agent_positions),
        landmark_positions=np.asarray(landmark_positions),
        actions=np.asarray(actions_history),
        rewards=np.asarray(rewards),
        collision_rates=np.asarray(collision_rates),
        min_pair_distances=np.asarray(min_pair_distances),
    )


def global_limits(trajectories: list[Trajectory], padding: float):
    points = []
    for trajectory in trajectories:
        points.append(trajectory.agent_positions.reshape(-1, 2))
        points.append(trajectory.landmark_positions.reshape(-1, 2))
    points = np.concatenate(points, axis=0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    span = max(float(high[0] - low[0]), float(high[1] - low[1]), 1.0)
    half = 0.5 * span + padding
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
    )


def draw_frame(
    ax,
    trajectory: Trajectory,
    frame: int,
    limits,
    trail_length: int,
    agent_radius: float,
    landmark_radius: float,
):
    ax.clear()
    x_min, x_max, y_min, y_max = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(trajectory.name)

    frame = min(frame, trajectory.episode_length)
    trail_start = max(0, frame - trail_length)

    landmarks = trajectory.landmark_positions[frame]
    for landmark_id, position in enumerate(landmarks):
        ax.add_patch(
            Circle(
                position,
                landmark_radius,
                facecolor=LANDMARK_COLOR,
                edgecolor="black",
                alpha=0.35,
            )
        )
        ax.text(
            position[0],
            position[1],
            f"L{landmark_id}",
            ha="center",
            va="center",
            fontsize=8,
        )

    collision = False
    collision_rate = 0.0
    min_distance = np.nan
    if frame > 0:
        metric_id = min(frame - 1, trajectory.episode_length - 1)
        collision_rate = float(trajectory.collision_rates[metric_id])
        min_distance = float(trajectory.min_pair_distances[metric_id])
        collision = collision_rate > 0.0

    for agent_id in range(trajectory.agent_positions.shape[1]):
        trail = trajectory.agent_positions[
            trail_start:frame + 1,
            agent_id,
        ]
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            color=AGENT_COLORS[agent_id],
            linewidth=1.8,
            alpha=0.65,
        )
        position = trajectory.agent_positions[frame, agent_id]
        ax.add_patch(
            Circle(
                position,
                agent_radius,
                facecolor=AGENT_COLORS[agent_id],
                edgecolor=COLLISION_COLOR if collision else "black",
                linewidth=3.0 if collision else 1.2,
            )
        )
        ax.text(
            position[0],
            position[1],
            f"A{agent_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

        if frame < trajectory.actions.shape[0]:
            action = trajectory.actions[frame, agent_id]
            ax.arrow(
                position[0],
                position[1],
                0.12 * action[0],
                0.12 * action[1],
                width=0.006,
                head_width=0.035,
                color=AGENT_COLORS[agent_id],
                length_includes_head=True,
                alpha=0.8,
            )

    cumulative_return = float(trajectory.rewards[:frame].sum())
    ax.text(
        0.02,
        0.98,
        (
            f"step {frame:02d}/{trajectory.episode_length:02d}\n"
            f"return {cumulative_return:.3f}\n"
            f"collision {collision_rate:.3f}\n"
            f"min dist {min_distance:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def save_animation(fig, animator, gif_path, mp4_path, fps: int, dpi: int):
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    animator.save(
        gif_path,
        writer=animation.PillowWriter(fps=fps),
        dpi=dpi,
    )
    print("Saved:", gif_path)

    if mp4_path is not None:
        animator.save(
            mp4_path,
            writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
            dpi=dpi,
        )
        print("Saved:", mp4_path)


def animate_one(
    trajectory: Trajectory,
    limits,
    output_dir: Path,
    fps: int,
    dpi: int,
    trail_length: int,
    agent_radius: float,
    landmark_radius: float,
    save_mp4: bool,
):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))

    def update(frame):
        draw_frame(
            ax,
            trajectory,
            frame,
            limits,
            trail_length,
            agent_radius,
            landmark_radius,
        )
        return ()

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=trajectory.episode_length + 1,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    slug = trajectory.name.lower().replace(" ", "_").replace("-", "_")
    save_animation(
        fig,
        animator,
        output_dir / f"{slug}.gif",
        output_dir / f"{slug}.mp4" if save_mp4 else None,
        fps,
        dpi,
    )
    plt.close(fig)


def animate_four(
    trajectories: list[Trajectory],
    limits,
    output_dir: Path,
    fps: int,
    dpi: int,
    trail_length: int,
    agent_radius: float,
    landmark_radius: float,
    save_mp4: bool,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.reshape(-1)
    frame_count = max(t.episode_length for t in trajectories) + 1

    def update(frame):
        for ax, trajectory in zip(axes, trajectories):
            draw_frame(
                ax,
                trajectory,
                frame,
                limits,
                trail_length,
                agent_radius,
                landmark_radius,
            )
        fig.suptitle(
            "Simple Spread: same environment seed, deterministic actors",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return ()

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    save_animation(
        fig,
        animator,
        output_dir / "four_algorithm_comparison.gif",
        output_dir / "four_algorithm_comparison.mp4" if save_mp4 else None,
        fps,
        dpi,
    )
    plt.close(fig)


def save_summary(
    trajectories: list[Trajectory],
    output_dir: Path,
    env_seed: int,
):
    result = {
        "environment_seed": env_seed,
        "action_mode": "deterministic_tanh_mean",
        "trajectories": [],
    }
    for trajectory in trajectories:
        item = {
            "name": trajectory.name,
            "checkpoint": trajectory.checkpoint,
            "checkpoint_update": trajectory.checkpoint_update,
            "parameter_mode": trajectory.parameter_mode,
            "episode_length": trajectory.episode_length,
            "total_return": trajectory.total_return,
            "mean_pair_collision_rate": float(
                trajectory.collision_rates.mean()
            ),
            "minimum_pair_distance": float(
                np.nanmin(trajectory.min_pair_distances)
            ),
        }
        result["trajectories"].append(item)
        print(
            f"{trajectory.name:<18} "
            f"return={item['total_return']:.4f} "
            f"collision={item['mean_pair_collision_rate']:.4f} "
            f"min_dist={item['minimum_pair_distance']:.4f} "
            f"mode={item['parameter_mode']}"
        )

    path = output_dir / "trajectory_summary.json"
    path.write_text(json.dumps(result, indent=2))
    print("Saved:", path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        ("PS-IPPO", args.ps_ippo),
        ("PS-MAPPO", args.ps_mappo),
        ("Independent IPPO", args.independent_ippo),
        ("DPO", args.dpo),
    ]
    trajectories = [
        rollout_policy(
            name,
            path,
            args.env_seed,
            args.max_steps,
            args.num_agents,
            args.num_landmarks,
        )
        for name, path in checkpoints
    ]

    limits = global_limits(trajectories, args.axis_padding)
    save_mp4 = not args.no_mp4 and shutil.which("ffmpeg") is not None
    if not save_mp4 and not args.no_mp4:
        print("ffmpeg not found; GIF files will still be generated.")

    for trajectory in trajectories:
        animate_one(
            trajectory,
            limits,
            args.output_dir,
            args.fps,
            args.dpi,
            args.trail_length,
            args.agent_radius,
            args.landmark_radius,
            save_mp4,
        )

    animate_four(
        trajectories,
        limits,
        args.output_dir,
        args.fps,
        args.dpi,
        args.trail_length,
        args.agent_radius,
        args.landmark_radius,
        save_mp4,
    )
    save_summary(trajectories, args.output_dir, args.env_seed)


if __name__ == "__main__":
    main()
