#!/usr/bin/env python3
"""Compare sender-triggered communication against fixed baselines.

Place this file in the repository root next to
``evaluate_structured_sender_full_state_trigger_estimator.py``.

Default experiment size:
    10 trajectories x 5 seeds x 3 noise scenarios x 4 methods = 600 runs.

Methods:
    no_comm      : never broadcast
    periodic_20  : all agents broadcast every 20 environment steps
    error_0.10   : sender-triggered normalized full-state residual > 0.10
    full_comm    : every agent broadcasts at every step

All methods use the same process-noise and measurement-noise realization for a
fixed (trajectory, seed, scenario), which gives a paired comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_FILENAME = "raw_runs.csv"
AGGREGATE_FILENAME = "aggregate.csv"
FAILURES_FILENAME = "failures.csv"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    mode: str
    error_threshold: float | None = None
    periodic_interval: int | None = None


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
        default=Path("results/sender_baseline_comparison"),
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
            "Repeat for multiple noise scenarios. Default: "
            "(0,0), (1,0), and (1,1)."
        ),
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
    parser.add_argument("--error-threshold", type=float, default=0.10)
    parser.add_argument("--periodic-interval", type=int, default=20)
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
    if args.error_threshold < 0.0:
        raise ValueError("error-threshold must be non-negative.")
    if args.periodic_interval <= 0:
        raise ValueError("periodic-interval must be positive.")
    if any(value < 0.0 for value in args.base_process_noise_std):
        raise ValueError("base-process-noise-std must be non-negative.")
    if any(value < 0.0 for value in args.base_measurement_noise_std):
        raise ValueError("base-measurement-noise-std must be non-negative.")


def load_module(path: Path) -> Any:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    spec = importlib.util.spec_from_file_location(
        "sender_baseline_evaluator", resolved
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluator from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "load_dataset",
        "load_checkpoint",
        "make_model_config",
        "restore_model_states",
        "resolve_state_scales",
        "simulate_process_noisy_truth",
        "sample_measurements",
        "model_only_rollout",
        "init_structured_transition_models",
        "initialize_global_belief",
        "predict_structured_global_belief_oracle_actions",
        "direct_agent_observation_update",
        "diagonal_measurement_covariance",
        "reshape_belief_mean",
        "normalized_full_state_error_score",
        "dynamics_kwargs",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError("Evaluator is missing: " + ", ".join(missing))
    return module


def select_trajectories(
    args: argparse.Namespace, dataset: dict[str, Any]
) -> list[int]:
    total = int(dataset["local_states"].shape[0])
    if args.trajectory_indices is not None:
        selected = list(dict.fromkeys(int(v) for v in args.trajectory_indices))
    else:
        selected = [
            int(v)
            for v in dataset["test_indices"][: args.num_trajectories]
        ]
    if not selected:
        raise ValueError("No trajectories selected.")
    invalid = [value for value in selected if not 0 <= value < total]
    if invalid:
        raise ValueError(f"Invalid trajectory indices: {invalid}")
    return selected


def wrapped_state_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    error = np.asarray(estimate) - np.asarray(truth)
    heading = np.arctan2(np.sin(error[..., 2]), np.cos(error[..., 2]))
    error = np.array(error, copy=True)
    error[..., 2] = heading
    return error


def position_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(estimate)[..., :2] - np.asarray(truth)[..., :2]
    return float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))


def normalized_full_state_rmse(
    estimate: np.ndarray, truth: np.ndarray, state_scales: np.ndarray
) -> float:
    normalized = wrapped_state_error(estimate, truth) / state_scales[None, None, :]
    return float(np.sqrt(np.mean(normalized**2)))


def method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    return [
        MethodSpec(name="no_comm", mode="no_comm"),
        MethodSpec(
            name=f"periodic_{args.periodic_interval}",
            mode="periodic",
            periodic_interval=args.periodic_interval,
        ),
        MethodSpec(
            name=f"error_{args.error_threshold:g}",
            mode="error",
            error_threshold=args.error_threshold,
        ),
        MethodSpec(name="full_comm", mode="full_comm"),
    ]


def run_policy_estimator(
    *,
    evaluator: Any,
    method: MethodSpec,
    truth: np.ndarray,
    measurements: np.ndarray,
    measurement_noise_std: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    state_scales: np.ndarray,
    prediction_mode: str,
    epistemic_process_scale: float,
    initial_variance: float,
    measurement_variance_floor: float,
) -> dict[str, np.ndarray]:
    num_agents = int(truth.shape[1])
    local_state_dim = int(truth.shape[2])
    belief = evaluator.initialize_global_belief(
        initial_local_states=jnp.asarray(truth[0:1]),
        initial_variance=initial_variance,
    )
    measurement_covariance = evaluator.diagonal_measurement_covariance(
        measurement_noise_std=measurement_noise_std,
        variance_floor=measurement_variance_floor,
        dtype=np.asarray(truth).dtype,
    )
    state_scales_jax = jnp.asarray(state_scales)

    posterior_means: list[np.ndarray] = []
    communication_decisions: list[np.ndarray] = []
    pre_error_scores: list[np.ndarray] = []

    for step_index in range(actions.shape[0]):
        predicted_belief, _ = (
            evaluator.predict_structured_global_belief_oracle_actions(
                belief=belief,
                model_states=model_states,
                standardizers=standardizers,
                joint_actions=jnp.asarray(actions[step_index : step_index + 1]),
                num_agents=num_agents,
                prediction_mode=prediction_mode,
                epistemic_process_scale=epistemic_process_scale,
                **evaluator.dynamics_kwargs(metadata),
            )
        )
        predicted_local_means = evaluator.reshape_belief_mean(
            predicted_belief, num_agents, local_state_dim
        )
        measured_next = jnp.asarray(measurements[step_index])

        scores = np.zeros(num_agents, dtype=np.float32)
        for agent_id in range(num_agents):
            scores[agent_id] = float(
                evaluator.normalized_full_state_error_score(
                    predicted_local_mean=jnp.asarray(
                        predicted_local_means[agent_id : agent_id + 1]
                    ),
                    observed_local_state=measured_next[
                        agent_id : agent_id + 1
                    ],
                    state_scales=state_scales_jax,
                )[0]
            )

        if method.mode == "no_comm":
            decisions = np.zeros(num_agents, dtype=np.bool_)
        elif method.mode == "full_comm":
            decisions = np.ones(num_agents, dtype=np.bool_)
        elif method.mode == "periodic":
            assert method.periodic_interval is not None
            active = (step_index + 1) % method.periodic_interval == 0
            decisions = np.full(num_agents, active, dtype=np.bool_)
        elif method.mode == "error":
            assert method.error_threshold is not None
            decisions = scores > float(method.error_threshold)
        else:
            raise ValueError(f"Unknown method mode: {method.mode}")

        belief = predicted_belief
        for agent_id in range(num_agents):
            if not decisions[agent_id]:
                continue
            belief, _ = evaluator.direct_agent_observation_update(
                predicted_belief=belief,
                observed_agent_id=agent_id,
                measurement=measured_next[agent_id : agent_id + 1],
                measurement_covariance=measurement_covariance,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

        posterior_means.append(
            evaluator.reshape_belief_mean(belief, num_agents, local_state_dim)
        )
        communication_decisions.append(decisions)
        pre_error_scores.append(scores)

    return {
        "posterior_means": np.stack(posterior_means),
        "communications": np.stack(communication_decisions),
        "pre_error_scores": np.stack(pre_error_scores),
    }


def run_key(
    trajectory_index: int,
    seed: int,
    process_scale: float,
    measurement_scale: float,
    method_name: str,
) -> tuple[str, ...]:
    return (
        str(int(trajectory_index)),
        str(int(seed)),
        format(float(process_scale), ".12g"),
        format(float(measurement_scale), ".12g"),
        method_name,
    )


def read_completed(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    output: set[tuple[str, ...]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            output.add(
                run_key(
                    int(row["trajectory_index"]),
                    int(row["seed"]),
                    float(row["process_noise_scale"]),
                    float(row["measurement_noise_scale"]),
                    row["method"],
                )
            )
    return output


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_ok_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            float(row["process_noise_scale"]),
            float(row["measurement_noise_scale"]),
            row["method"],
        )
        groups[key].append(row)

    metrics = (
        "broadcast_rate",
        "posterior_position_rmse",
        "posterior_normalized_full_state_rmse",
        "relative_position_rmse_improvement",
    )
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2])):
        process_scale, measurement_scale, method = key
        members = groups[key]
        item: dict[str, Any] = {
            "process_noise_scale": process_scale,
            "measurement_noise_scale": measurement_scale,
            "method": method,
            "num_runs": len(members),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(
                np.std(values, ddof=1) if values.size > 1 else 0.0
            )
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_plots(aggregate: list[dict[str, Any]], output_directory: Path) -> None:
    scenarios = sorted(
        {
            (float(row["process_noise_scale"]), float(row["measurement_noise_scale"]))
            for row in aggregate
        }
    )
    for process_scale, measurement_scale in scenarios:
        selected = [
            row
            for row in aggregate
            if float(row["process_noise_scale"]) == process_scale
            and float(row["measurement_noise_scale"]) == measurement_scale
        ]
        selected.sort(key=lambda row: float(row["broadcast_rate_mean"]))

        figure, axis = plt.subplots(figsize=(8.2, 6.2))
        x = [float(row["broadcast_rate_mean"]) for row in selected]
        y = [float(row["posterior_position_rmse_mean"]) for row in selected]
        axis.scatter(x, y, s=75)
        for x_value, y_value, row in zip(x, y, selected):
            axis.annotate(
                row["method"],
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        axis.set_xlabel("mean broadcast rate")
        axis.set_ylabel("mean posterior position RMSE")
        axis.set_title(
            "Method comparison\n"
            f"process scale={process_scale:g}, "
            f"measurement scale={measurement_scale:g}"
        )
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        output = (
            output_directory
            / "plots"
            / f"method_comparison_q{process_scale:g}_r{measurement_scale:g}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    raw_path = args.output_directory / RAW_FILENAME
    aggregate_path = args.output_directory / AGGREGATE_FILENAME
    failures_path = args.output_directory / FAILURES_FILENAME
    if args.overwrite:
        for path in (raw_path, aggregate_path, failures_path):
            if path.exists():
                path.unlink()

    evaluator = load_module(args.evaluator)
    dataset = evaluator.load_dataset(args.dataset)
    trajectories = select_trajectories(args, dataset)
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

    completed = read_completed(raw_path) if args.resume else set()
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
                measurements, sampled_measurement_noise = evaluator.sample_measurements(
                    truth=truth,
                    measurement_noise_std=measurement_std,
                    seed=paired_seed + 202,
                )

                for method in methods:
                    counter += 1
                    key = run_key(
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
                        trace = run_policy_estimator(
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
                            measurement_variance_floor=args.measurement_variance_floor,
                        )
                        posterior = np.concatenate(
                            [np.asarray(truth)[0:1], trace["posterior_means"]],
                            axis=0,
                        )
                        communications = trace["communications"]
                        total_broadcasts = int(communications.sum())
                        maximum_broadcasts = int(communications.size)
                        model_rmse = position_rmse(model_only, truth)
                        posterior_rmse = position_rmse(posterior, truth)
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
                            "broadcast_rate": total_broadcasts / maximum_broadcasts,
                            "model_position_rmse": model_rmse,
                            "posterior_position_rmse": posterior_rmse,
                            "posterior_normalized_full_state_rmse": normalized_full_state_rmse(
                                posterior, truth, state_scales
                            ),
                            "relative_position_rmse_improvement": relative_improvement,
                            "sampled_linear_process_noise_rms": float(
                                np.sqrt(np.mean(sampled_process_noise[..., 0] ** 2))
                            ),
                            "sampled_angular_process_noise_rms": float(
                                np.sqrt(np.mean(sampled_process_noise[..., 1] ** 2))
                            ),
                            "sampled_measurement_noise_rms": float(
                                np.sqrt(np.mean(sampled_measurement_noise**2))
                            ),
                        }
                        append_row(raw_path, row)
                        completed.add(key)
                        executed += 1
                        print(
                            f"  broadcasts={total_broadcasts}/{maximum_broadcasts} "
                            f"| rate={row['broadcast_rate']:.4f} "
                            f"| posterior RMSE={posterior_rmse:.5f}"
                        )
                    except Exception as error:
                        failed += 1
                        append_row(
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
                        print(f"  FAILED: {type(error).__name__}: {error}")

    rows = read_ok_rows(raw_path)
    if not rows:
        raise RuntimeError("No successful runs were produced.")
    aggregate = aggregate_rows(rows)
    write_csv(aggregate_path, aggregate)
    if not args.no_plots:
        create_plots(aggregate, args.output_directory)

    config = vars(args).copy()
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

    print("\n===== Baseline comparison complete =====")
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
