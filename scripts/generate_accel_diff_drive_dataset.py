#!/usr/bin/env python3
"""Generate trajectory-split data for local dynamics-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

from aorpo.data.accel_diff_drive_dataset import (
    collect_piecewise_constant_trajectories,
    save_trajectory_dataset,
    trajectory_split_indices,
)
from aorpo.envs.accel_diff_drive_multi_agent_env import (
    AccelDiffDriveConfig,
    AccelDiffDriveMultiAgentEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/accel_diff_drive_deterministic.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--hold-min", type=int, default=5)
    parser.add_argument("--hold-max", type=int, default=20)
    parser.add_argument("--linear-action-scale", type=float, default=1.0)
    parser.add_argument("--angular-action-scale", type=float, default=1.0)
    parser.add_argument("--linear-noise-std", type=float, default=0.0)
    parser.add_argument("--angular-noise-std", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AccelDiffDriveConfig(
        num_agents=args.num_agents,
        dt=args.dt,
        max_steps=args.horizon,
        linear_acceleration_noise_std=args.linear_noise_std,
        angular_acceleration_noise_std=args.angular_noise_std,
        wrap_heading=False,
    )
    env = AccelDiffDriveMultiAgentEnv(config)

    dataset = collect_piecewise_constant_trajectories(
        env=env,
        rng=jax.random.PRNGKey(args.seed),
        num_trajectories=args.num_trajectories,
        horizon=args.horizon,
        hold_steps_min=args.hold_min,
        hold_steps_max=args.hold_max,
        linear_action_scale=args.linear_action_scale,
        angular_action_scale=args.angular_action_scale,
    )
    splits = trajectory_split_indices(
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )

    metadata = env.metadata()
    metadata.update(
        {
            "num_trajectories": args.num_trajectories,
            "horizon": args.horizon,
            "action_hold_steps_min": args.hold_min,
            "action_hold_steps_max": args.hold_max,
            "linear_action_scale": args.linear_action_scale,
            "angular_action_scale": args.angular_action_scale,
            "split_unit": "trajectory",
        }
    )

    output = save_trajectory_dataset(
        args.output,
        dataset,
        metadata,
        splits,
    )

    print(f"Saved dataset to {output}")
    print(f"local_states shape: {dataset.local_states.shape}")
    print(f"actions shape: {dataset.actions.shape}")
    print(
        "trajectory split sizes:",
        {name: len(indices) for name, indices in splits.items()},
    )


if __name__ == "__main__":
    main()
