#!/usr/bin/env python3
"""Run only fixed-budget periodic communication baselines.

This script reuses utilities from ``run_sender_baseline_comparison.py`` and
evaluates one or more periodic intervals, e.g. periodic_50 and periodic_100.

Default experiment:
    10 trajectories x 5 seeds x 3 noise scenarios x 2 periodic methods
    = 300 runs.

Place this file in the repository root next to:
    run_sender_baseline_comparison.py
    evaluate_structured_sender_full_state_trigger_estimator.py
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import jax
import numpy as np

import run_sender_baseline_comparison as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(
            "evaluate_structured_sender_full_state_trigger_estimator.py"
        ),
    )
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
        default=Path("results/periodic_budget_baselines"),
    )
    parser.add_argument("--horizon", type=int, default=200)

    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--trajectory-indices", type=int, nargs="+", default=None
    )
    trajectory_group.add_argument(
        "--num-trajectories", type=int, default=10
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--scenarios",
        type=float,
        nargs=2,
        action="append",
        metavar=("PROCESS_SCALE", "MEASUREMENT_SCALE"),
        default=None,
        help=(
            "Repeat once per scenario. Default: "
            "(0,0), (1,0), and (1,1)."
        ),
    )
    parser.add_argument(
        "--periodic-intervals",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Periodic intervals to evaluate, e.g. 50 100.",
    )
    parser.add_argument(
        "--base-process-noise-std",
        type=float,
        nargs=2,
        default=[0.2, 0.4],
        metavar=("LINEAR_ACCEL", "ANGULAR_ACCEL"),
    )
    parser.add_argument(
        "--base-measurement-noise-std",
        type=float,
        nargs=5,
        default=[0.02, 0.02, 0.01, 0.02, 0.02],
        metavar=("PX", "PY", "PHI", "V", "OMEGA"),
    )
    parser.add_argument(
        "--state-scales", type=float, nargs=5, default=None
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("ensemble_mean", "infoprop_ci"),
        default="ensemble_mean",
    )
    parser.add_argument("--epistemic-process-scale", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0e-6)
    parser.add_argument(
        "--measurement-variance-floor", type=float, default=1.0e-9
    )
    parser.add_argument("--model-init-seed", type=int, default=0)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 0:
        raise ValueError("horizon must be positive.")
    if args.num_trajectories is not None and args.num_trajectories <= 0:
        raise ValueError("num-trajectories must be positive.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if not args.periodic_intervals:
        raise ValueError("At least one periodic interval is required.")
    if any(interval <= 0 for interval in args.periodic_intervals):
        raise ValueError("Every periodic interval must be positive.")
    if len(set(args.periodic_intervals)) != len(args.periodic_intervals):
        raise ValueError("periodic-intervals must not contain duplicates.")
    if any(value < 0.0 for value in args.base_process_noise_std):
        raise ValueError("base-process-noise-std must be non-negative.")
    if any(value < 0.0 for value in args.base_measurement_noise_std):
        raise ValueError("base-measurement-noise-std must be non-negative.")


def method_specs(args: argparse.Namespace) -> list[base.MethodSpec]:
    return [
        base.MethodSpec(
            name=f"periodic_{interval}",
            mode="periodic",
            periodic_interval=int(interval),
        )
        for interval in args.periodic_intervals
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    raw_path = args.output_directory / base.RAW_FILENAME
    aggregate_path = args.output_directory / base.AGGREGATE_FILENAME
    failures_path = args.output_directory / base.FAILURES_FILENAME
    if args.overwrite:
        for path in (raw_path, aggregate_path, failures_path):
            if path.exists():
                path.unlink()

    evaluator = base.load_module(args.evaluator)
    dataset = evaluator.load_dataset(args.dataset)
    trajectories = base.select_trajectories(args, dataset)
    scenarios = args.scenarios or [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    scenarios = [(float(q), float(r)) for q, r in scenarios]
    methods = method_specs(args)

    planned = len(trajectories) * len(args.seeds) * len(scenarios) * len(methods)
    print("Trajectories:", trajectories)
    print("Seeds:", args.seeds)
    print("Scenarios (q, r):", scenarios)
    print("Methods:", [method.name for method in methods])
    print("Planned runs:", planned)
    print("Output:", args.output_directory)

    saved_models, standardizers, checkpoint_metadata = evaluator.load_checkpoint(
        args.checkpoint
    )
    metadata = dict(dataset["metadata"])
    metadata.update(checkpoint_metadata)
    state_scales = evaluator.resolve_state_scales(args.state_scales, metadata)

    local_states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(local_states.shape[2])
    cfg = evaluator.make_model_config(metadata)
    _, initialized_states = evaluator.init_structured_transition_models(
        rng=jax.random.PRNGKey(args.model_init_seed),
        num_agents=num_agents,
        cfg=cfg,
    )
    model_states = evaluator.restore_model_states(initialized_states, saved_models)

    completed = base.read_completed(raw_path) if args.resume else set()
    base_process = np.asarray(args.base_process_noise_std, dtype=np.float32)
    base_measurement = np.asarray(
        args.base_measurement_noise_std, dtype=np.float32
    )

    counter = 0
    executed = 0
    skipped = 0
    failed = 0

    for trajectory_index in trajectories:
        horizon = min(args.horizon, int(actions.shape[1]))
        dataset_truth = np.asarray(
            local_states[trajectory_index, : horizon + 1]
        )
        action_sequence = np.asarray(actions[trajectory_index, :horizon])

        model_only = evaluator.model_only_rollout(
            initial_states=dataset_truth[0],
            actions=action_sequence,
            model_states=model_states,
            standardizers=standardizers,
            metadata=metadata,
            prediction_mode=args.prediction_mode,
            epistemic_process_scale=args.epistemic_process_scale,
        )

        for seed in args.seeds:
            paired_seed = int(seed) + int(trajectory_index) * 100_003
            for process_scale, measurement_scale in scenarios:
                process_std = base_process * process_scale
                measurement_std = base_measurement * measurement_scale
                truth, sampled_process_noise, _ = (
                    evaluator.simulate_process_noisy_truth(
                        initial_states=dataset_truth[0],
                        actions=action_sequence,
                        metadata=metadata,
                        linear_acceleration_noise_std=float(process_std[0]),
                        angular_acceleration_noise_std=float(process_std[1]),
                        seed=paired_seed + 101,
                    )
                )
                measurements, sampled_measurement_noise = (
                    evaluator.sample_measurements(
                        truth=truth,
                        measurement_noise_std=measurement_std,
                        seed=paired_seed + 202,
                    )
                )

                for method in methods:
                    counter += 1
                    key = base.run_key(
                        trajectory_index,
                        seed,
                        process_scale,
                        measurement_scale,
                        method.name,
                    )
                    if key in completed:
                        skipped += 1
                        continue

                    print(
                        f"[{counter}/{planned}] traj={trajectory_index} "
                        f"seed={seed} q={process_scale:g} "
                        f"r={measurement_scale:g} method={method.name}"
                    )
                    try:
                        trace = base.run_policy_estimator(
                            evaluator=evaluator,
                            method=method,
                            truth=np.asarray(truth),
                            measurements=np.asarray(measurements),
                            measurement_noise_std=measurement_std,
                            actions=action_sequence,
                            model_states=model_states,
                            standardizers=standardizers,
                            metadata=metadata,
                            state_scales=state_scales,
                            prediction_mode=args.prediction_mode,
                            epistemic_process_scale=args.epistemic_process_scale,
                            initial_variance=args.initial_variance,
                            measurement_variance_floor=(
                                args.measurement_variance_floor
                            ),
                        )
                        posterior = np.concatenate(
                            [np.asarray(truth)[0:1], trace["posterior_means"]],
                            axis=0,
                        )
                        communications = trace["communications"]
                        total_broadcasts = int(communications.sum())
                        maximum_broadcasts = int(communications.size)
                        model_rmse = base.position_rmse(model_only, truth)
                        posterior_rmse = base.position_rmse(posterior, truth)
                        relative_improvement = (
                            (model_rmse - posterior_rmse)
                            / max(model_rmse, 1.0e-12)
                        )
                        row = {
                            "status": "ok",
                            "trajectory_index": int(trajectory_index),
                            "seed": int(seed),
                            "process_noise_scale": process_scale,
                            "measurement_noise_scale": measurement_scale,
                            "method": method.name,
                            "total_broadcasts": total_broadcasts,
                            "maximum_broadcasts": maximum_broadcasts,
                            "broadcast_rate": (
                                total_broadcasts / maximum_broadcasts
                            ),
                            "model_position_rmse": model_rmse,
                            "posterior_position_rmse": posterior_rmse,
                            "posterior_normalized_full_state_rmse": (
                                base.normalized_full_state_rmse(
                                    posterior, truth, state_scales
                                )
                            ),
                            "relative_position_rmse_improvement": (
                                relative_improvement
                            ),
                            "sampled_linear_process_noise_rms": float(
                                np.sqrt(
                                    np.mean(
                                        sampled_process_noise[..., 0] ** 2
                                    )
                                )
                            ),
                            "sampled_angular_process_noise_rms": float(
                                np.sqrt(
                                    np.mean(
                                        sampled_process_noise[..., 1] ** 2
                                    )
                                )
                            ),
                            "sampled_measurement_noise_rms": float(
                                np.sqrt(
                                    np.mean(sampled_measurement_noise**2)
                                )
                            ),
                        }
                        base.append_row(raw_path, row)
                        completed.add(key)
                        executed += 1
                        print(
                            f"  broadcasts={total_broadcasts}/"
                            f"{maximum_broadcasts} "
                            f"| rate={row['broadcast_rate']:.4f} "
                            f"| posterior RMSE={posterior_rmse:.5f}"
                        )
                    except Exception as error:
                        failed += 1
                        base.append_row(
                            failures_path,
                            {
                                "trajectory_index": trajectory_index,
                                "seed": seed,
                                "process_noise_scale": process_scale,
                                "measurement_noise_scale": measurement_scale,
                                "method": method.name,
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        print(
                            f"  FAILED: {type(error).__name__}: {error}"
                        )

    rows = base.read_ok_rows(raw_path)
    if not rows:
        raise RuntimeError("No successful runs were produced.")
    aggregate = base.aggregate_rows(rows)
    base.write_csv(aggregate_path, aggregate)
    if not args.no_plots:
        base.create_plots(aggregate, args.output_directory)

    config: dict[str, Any] = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["selected_trajectories"] = trajectories
    config["resolved_scenarios"] = scenarios
    config["methods"] = [method.__dict__ for method in methods]
    with (args.output_directory / "comparison_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    print("\n===== Periodic budget baselines complete =====")
    print("Successful rows:", len(rows))
    print("Executed this invocation:", executed)
    print("Skipped existing rows:", skipped)
    print("Failed this invocation:", failed)
    print("Raw CSV:", raw_path)
    print("Aggregate CSV:", aggregate_path)
    if not args.no_plots:
        print("Plots:", args.output_directory / "plots")


if __name__ == "__main__":
    main()
