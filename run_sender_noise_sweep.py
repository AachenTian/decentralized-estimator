#!/usr/bin/env python3
"""Run reproducible noise/threshold sweeps for the sender-triggered estimator.

Place this file in the repository root, next to
``evaluate_structured_sender_full_state_trigger_estimator.py``.

The script imports the evaluator, loads the dataset and dynamics checkpoint once,
then sweeps over:

* trajectory indices,
* random seeds,
* process-noise scales,
* measurement-noise scales, and
* sender prediction-error thresholds.

It writes one raw CSV row per run, an aggregated CSV, and heatmaps.  The
uncertainty trigger is disabled by default with a very large threshold so that
the prediction-error trigger can be studied in isolation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_FILENAME = "raw_runs.csv"
AGGREGATE_FILENAME = "aggregate.csv"
FAILURES_FILENAME = "failures.csv"
TRACE_FILENAME = "sender_full_state_trigger_estimator_trace.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep process noise, measurement noise, and sender error "
            "thresholds without regenerating a figure for every run."
        )
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(
            "evaluate_structured_sender_full_state_trigger_estimator.py"
        ),
        help="Path to the noise-enabled evaluator module.",
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
        default=Path("results/noise_sweep"),
    )
    parser.add_argument("--horizon", type=int, default=200)

    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument(
        "--trajectory-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset trajectory indices.",
    )
    trajectory_group.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Use the first N test trajectories when indices are omitted.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Noise seeds. The same standardized noise is reused across scales.",
    )
    parser.add_argument(
        "--process-noise-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
    )
    parser.add_argument(
        "--base-process-noise-std",
        type=float,
        nargs=2,
        default=[0.2, 0.4],
        metavar=("LINEAR_ACCEL", "ANGULAR_ACCEL"),
        help=(
            "Base [linear-acceleration, angular-acceleration] standard "
            "deviations multiplied by each process-noise scale."
        ),
    )
    parser.add_argument(
        "--measurement-noise-scales",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--base-measurement-noise-std",
        type=float,
        nargs=5,
        default=[0.02, 0.02, 0.01, 0.02, 0.02],
        metavar=("PX", "PY", "PHI", "V", "OMEGA"),
        help="Base 5D measurement-noise std vector multiplied by each scale.",
    )
    parser.add_argument(
        "--error-thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=1.0e6,
        help="Large default disables the uncertainty trigger.",
    )
    parser.add_argument(
        "--uncertainty-sigma-scale",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--state-scales",
        type=float,
        nargs=5,
        default=None,
        metavar=("PX", "PY", "PHI", "V", "OMEGA"),
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("ensemble_mean", "infoprop_ci"),
        default="ensemble_mean",
    )
    parser.add_argument("--epistemic-process-scale", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0e-6)
    parser.add_argument(
        "--measurement-variance-floor",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument("--position-ellipse-scale", type=float, default=2.0)
    parser.add_argument("--model-init-seed", type=int, default=0)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip configurations already present in raw_runs.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior CSV outputs before starting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected sweep size without running the estimator.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV files but skip heatmaps and trade-off plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 0:
        raise ValueError("horizon must be positive.")
    if args.num_trajectories is not None and args.num_trajectories <= 0:
        raise ValueError("num-trajectories must be positive.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if any(value < 0.0 for value in args.process_noise_scales):
        raise ValueError("process-noise-scales must be non-negative.")
    if any(value < 0.0 for value in args.measurement_noise_scales):
        raise ValueError("measurement-noise-scales must be non-negative.")
    if any(value < 0.0 for value in args.base_process_noise_std):
        raise ValueError("base-process-noise-std must be non-negative.")
    if any(value < 0.0 for value in args.base_measurement_noise_std):
        raise ValueError("base-measurement-noise-std must be non-negative.")
    if any(value < 0.0 for value in args.error_thresholds):
        raise ValueError("error-thresholds must be non-negative.")
    if args.uncertainty_threshold < 0.0:
        raise ValueError("uncertainty-threshold must be non-negative.")
    if args.uncertainty_sigma_scale <= 0.0:
        raise ValueError("uncertainty-sigma-scale must be positive.")
    if args.initial_variance < 0.0:
        raise ValueError("initial-variance must be non-negative.")
    if args.measurement_variance_floor < 0.0:
        raise ValueError("measurement-variance-floor must be non-negative.")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("max-runs must be positive when provided.")


def load_module(path: Path) -> Any:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Evaluator not found: {resolved}")
    spec = importlib.util.spec_from_file_location("sender_noise_evaluator", resolved)
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
        "run_sender_trigger_estimator",
        "init_structured_transition_models",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The evaluator is not the noise-enabled version. Missing: "
            + ", ".join(missing)
        )
    return module


def select_trajectories(
    args: argparse.Namespace,
    dataset: dict[str, Any],
) -> list[int]:
    total = int(dataset["local_states"].shape[0])
    if args.trajectory_indices is not None:
        selected = list(dict.fromkeys(args.trajectory_indices))
    else:
        test_indices = [int(value) for value in dataset["test_indices"]]
        selected = test_indices[: args.num_trajectories]

    if not selected:
        raise ValueError("No trajectory was selected.")
    invalid = [index for index in selected if not 0 <= index < total]
    if invalid:
        raise ValueError(f"Invalid trajectory indices: {invalid}")
    return selected


def canonical_float(value: float) -> str:
    return format(float(value), ".12g")


def run_key(
    trajectory_index: int,
    seed: int,
    process_scale: float,
    measurement_scale: float,
    error_threshold: float,
) -> tuple[str, ...]:
    return (
        str(int(trajectory_index)),
        str(int(seed)),
        canonical_float(process_scale),
        canonical_float(measurement_scale),
        canonical_float(error_threshold),
    )


def read_completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, ...]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            completed.add(
                (
                    row["trajectory_index"],
                    row["seed"],
                    canonical_float(float(row["process_noise_scale"])),
                    canonical_float(float(row["measurement_noise_scale"])),
                    canonical_float(float(row["error_threshold"])),
                )
            )
    return completed


def wrapped_state_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    error = np.asarray(estimate) - np.asarray(truth)
    heading = np.arctan2(np.sin(error[..., 2]), np.cos(error[..., 2]))
    error = error.copy()
    error[..., 2] = heading
    return error


def root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def position_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    difference = np.asarray(estimate)[..., :2] - np.asarray(truth)[..., :2]
    squared_distance = np.sum(np.square(difference), axis=-1)
    return float(np.sqrt(np.mean(squared_distance)))


def mean_position_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    difference = np.asarray(estimate)[..., :2] - np.asarray(truth)[..., :2]
    return float(np.mean(np.linalg.norm(difference, axis=-1)))


def normalized_full_state_rmse(
    estimate: np.ndarray,
    truth: np.ndarray,
    state_scales: np.ndarray,
) -> float:
    error = wrapped_state_error(estimate, truth)
    normalized = error / state_scales[None, None, :]
    return root_mean_square(normalized)


def build_metrics(
    *,
    trajectory_index: int,
    seed: int,
    process_scale: float,
    measurement_scale: float,
    error_threshold: float,
    linear_process_std: float,
    angular_process_std: float,
    measurement_noise_std: np.ndarray,
    truth: np.ndarray,
    model_only: np.ndarray,
    trace: dict[str, np.ndarray],
    state_scales: np.ndarray,
    sampled_process_noise: np.ndarray,
    sampled_measurement_noise: np.ndarray,
) -> dict[str, Any]:
    posterior = np.concatenate(
        [truth[0:1], np.asarray(trace["posterior_means"])],
        axis=0,
    )
    communications = np.asarray(trace["communications"], dtype=np.bool_)
    horizon, num_agents = communications.shape
    total_messages = int(communications.sum())
    maximum_messages = int(horizon * num_agents)

    row: dict[str, Any] = {
        "status": "ok",
        "trajectory_index": int(trajectory_index),
        "seed": int(seed),
        "horizon": int(horizon),
        "num_agents": int(num_agents),
        "process_noise_scale": float(process_scale),
        "measurement_noise_scale": float(measurement_scale),
        "error_threshold": float(error_threshold),
        "linear_acceleration_process_noise_std": float(linear_process_std),
        "angular_acceleration_process_noise_std": float(angular_process_std),
        "measurement_noise_std_px": float(measurement_noise_std[0]),
        "measurement_noise_std_py": float(measurement_noise_std[1]),
        "measurement_noise_std_phi": float(measurement_noise_std[2]),
        "measurement_noise_std_v": float(measurement_noise_std[3]),
        "measurement_noise_std_omega": float(measurement_noise_std[4]),
        "total_broadcasts": total_messages,
        "maximum_broadcasts": maximum_messages,
        "broadcast_rate": total_messages / maximum_messages,
        "model_position_rmse": position_rmse(model_only, truth),
        "posterior_position_rmse": position_rmse(posterior, truth),
        "model_mean_position_error": mean_position_error(model_only, truth),
        "posterior_mean_position_error": mean_position_error(posterior, truth),
        "model_normalized_full_state_rmse": normalized_full_state_rmse(
            model_only, truth, state_scales
        ),
        "posterior_normalized_full_state_rmse": normalized_full_state_rmse(
            posterior, truth, state_scales
        ),
        "mean_pre_message_trigger_residual": float(
            np.mean(trace["predicted_full_state_errors"])
        ),
        "p95_pre_message_trigger_residual": float(
            np.quantile(trace["predicted_full_state_errors"], 0.95)
        ),
        "max_pre_message_trigger_residual": float(
            np.max(trace["predicted_full_state_errors"])
        ),
        "mean_pre_message_true_error": float(
            np.mean(trace["predicted_true_full_state_errors"])
        ),
        "mean_post_message_true_error": float(
            np.mean(trace["posterior_true_full_state_errors"])
        ),
        "mean_pre_message_uncertainty": float(
            np.mean(trace["predicted_full_state_uncertainties"])
        ),
        "sampled_linear_process_noise_rms": root_mean_square(
            sampled_process_noise[..., 0]
        ),
        "sampled_angular_process_noise_rms": root_mean_square(
            sampled_process_noise[..., 1]
        ),
        "sampled_measurement_noise_rms": root_mean_square(
            sampled_measurement_noise
        ),
    }

    for agent_id in range(num_agents):
        row[f"agent_{agent_id}_broadcasts"] = int(
            communications[:, agent_id].sum()
        )
        row[f"agent_{agent_id}_posterior_position_rmse"] = position_rmse(
            posterior[:, agent_id : agent_id + 1],
            truth[:, agent_id : agent_id + 1],
        )
    return row


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
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
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "status",
        "trajectory_index",
        "seed",
        "process_noise_scale",
        "measurement_noise_scale",
        "error_threshold",
    }
    columns = []
    for key in rows[0]:
        if key in excluded:
            continue
        try:
            float(rows[0][key])
        except (TypeError, ValueError):
            continue
        columns.append(key)
    return columns


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            float(row["process_noise_scale"]),
            float(row["measurement_noise_scale"]),
            float(row["error_threshold"]),
        )
        groups[key].append(row)

    metrics = numeric_columns(rows)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        process_scale, measurement_scale, threshold = key
        members = groups[key]
        item: dict[str, Any] = {
            "process_noise_scale": process_scale,
            "measurement_noise_scale": measurement_scale,
            "error_threshold": threshold,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_axis_value(value: float) -> str:
    return f"{value:g}"


def save_heatmap(
    aggregate: list[dict[str, Any]],
    *,
    threshold: float,
    metric: str,
    title: str,
    label: str,
    output: Path,
) -> None:
    selected = [
        row
        for row in aggregate
        if math.isclose(float(row["error_threshold"]), threshold)
    ]
    if not selected:
        return
    process_values = sorted(
        {float(row["process_noise_scale"]) for row in selected}
    )
    measurement_values = sorted(
        {float(row["measurement_noise_scale"]) for row in selected}
    )
    lookup = {
        (
            float(row["measurement_noise_scale"]),
            float(row["process_noise_scale"]),
        ): float(row[metric])
        for row in selected
    }
    matrix = np.full(
        (len(measurement_values), len(process_values)),
        np.nan,
        dtype=np.float64,
    )
    for row_index, measurement_scale in enumerate(measurement_values):
        for column_index, process_scale in enumerate(process_values):
            matrix[row_index, column_index] = lookup.get(
                (measurement_scale, process_scale), np.nan
            )

    figure, axis = plt.subplots(figsize=(8.4, 6.4))
    image = axis.imshow(matrix, aspect="auto", origin="lower")
    figure.colorbar(image, ax=axis, label=label)
    axis.set_xticks(np.arange(len(process_values)))
    axis.set_xticklabels([format_axis_value(value) for value in process_values])
    axis.set_yticks(np.arange(len(measurement_values)))
    axis.set_yticklabels(
        [format_axis_value(value) for value in measurement_values]
    )
    axis.set_xlabel("process-noise scale")
    axis.set_ylabel("measurement-noise scale")
    axis.set_title(f"{title}\nerror threshold = {threshold:g}")

    finite = matrix[np.isfinite(matrix)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            axis.text(
                column_index,
                row_index,
                f"{value:.3g}",
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_tradeoff_plots(
    aggregate: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    process_values = sorted(
        {float(row["process_noise_scale"]) for row in aggregate}
    )
    measurement_values = sorted(
        {float(row["measurement_noise_scale"]) for row in aggregate}
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    for process_scale in process_values:
        for measurement_scale in measurement_values:
            rows = [
                row
                for row in aggregate
                if math.isclose(
                    float(row["process_noise_scale"]), process_scale
                )
                and math.isclose(
                    float(row["measurement_noise_scale"]), measurement_scale
                )
            ]
            rows.sort(key=lambda row: float(row["error_threshold"]))
            if not rows:
                continue
            x = np.asarray([float(row["broadcast_rate_mean"]) for row in rows])
            y = np.asarray(
                [float(row["posterior_position_rmse_mean"]) for row in rows]
            )
            thresholds = [float(row["error_threshold"]) for row in rows]

            figure, axis = plt.subplots(figsize=(7.2, 5.6))
            axis.plot(x, y, marker="o")
            for x_value, y_value, threshold in zip(x, y, thresholds):
                axis.annotate(
                    f"δ={threshold:g}",
                    (x_value, y_value),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            axis.set_xlabel("mean broadcast rate")
            axis.set_ylabel("mean posterior position RMSE")
            axis.set_title(
                "Communication–accuracy trade-off\n"
                f"process scale={process_scale:g}, "
                f"measurement scale={measurement_scale:g}"
            )
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            process_name = canonical_float(process_scale).replace(".", "p")
            measurement_name = canonical_float(measurement_scale).replace(".", "p")
            filename = (
                f"tradeoff_process_{process_name}_"
                f"measurement_{measurement_name}.png"
            )
            figure.savefig(
                output_directory / filename,
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)


def create_plots(
    aggregate: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    thresholds = sorted({float(row["error_threshold"]) for row in aggregate})
    heatmap_directory = output_directory / "heatmaps"
    for threshold in thresholds:
        threshold_name = canonical_float(threshold).replace(".", "p")
        save_heatmap(
            aggregate,
            threshold=threshold,
            metric="broadcast_rate_mean",
            title="Mean broadcast rate",
            label="broadcast rate",
            output=heatmap_directory
            / f"broadcast_rate_threshold_{threshold_name}.png",
        )
        save_heatmap(
            aggregate,
            threshold=threshold,
            metric="posterior_position_rmse_mean",
            title="Mean posterior position RMSE",
            label="position RMSE",
            output=heatmap_directory
            / f"posterior_position_rmse_threshold_{threshold_name}.png",
        )
        save_heatmap(
            aggregate,
            threshold=threshold,
            metric="posterior_normalized_full_state_rmse_mean",
            title="Mean normalized full-state RMSE",
            label="normalized RMSE",
            output=heatmap_directory
            / f"posterior_full_state_rmse_threshold_{threshold_name}.png",
        )
    save_tradeoff_plots(aggregate, output_directory / "tradeoffs")


def write_config(
    args: argparse.Namespace,
    trajectories: list[int],
    total_runs: int,
) -> None:
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["selected_trajectories"] = trajectories
    payload["planned_runs"] = total_runs
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with (args.output_directory / "sweep_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)


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

    planned = (
        len(trajectories)
        * len(args.seeds)
        * len(args.process_noise_scales)
        * len(args.measurement_noise_scales)
        * len(args.error_thresholds)
    )
    if args.max_runs is not None:
        planned = min(planned, args.max_runs)
    write_config(args, trajectories, planned)

    print("Selected trajectories:", trajectories)
    print("Seeds:", args.seeds)
    print("Process-noise scales:", args.process_noise_scales)
    print("Measurement-noise scales:", args.measurement_noise_scales)
    print("Error thresholds:", args.error_thresholds)
    print("Planned runs:", planned)
    print("Output directory:", args.output_directory)
    if args.dry_run:
        return

    saved_models, standardizers, checkpoint_metadata = (
        evaluator.load_checkpoint(args.checkpoint)
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
    model_states = evaluator.restore_model_states(
        initialized_states,
        saved_models,
    )
    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, got {len(standardizers)}."
        )

    completed = read_completed_keys(raw_path) if args.resume else set()
    base_process = np.asarray(args.base_process_noise_std, dtype=np.float32)
    base_measurement = np.asarray(
        args.base_measurement_noise_std, dtype=np.float32
    )

    total_counter = 0
    executed_counter = 0
    skipped_counter = 0
    failed_counter = 0

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

        process_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        measurement_cache: dict[
            tuple[int, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}

        for seed in args.seeds:
            paired_seed = int(seed) + int(trajectory_index) * 100_003
            for process_scale in args.process_noise_scales:
                process_key = (int(seed), float(process_scale))
                process_std = base_process * float(process_scale)
                if process_key not in process_cache:
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
                    process_cache[process_key] = (
                        np.asarray(truth),
                        np.asarray(sampled_process_noise),
                    )
                truth, sampled_process_noise = process_cache[process_key]

                for measurement_scale in args.measurement_noise_scales:
                    measurement_key = (
                        int(seed),
                        float(process_scale),
                        float(measurement_scale),
                    )
                    measurement_std = (
                        base_measurement * float(measurement_scale)
                    )
                    if measurement_key not in measurement_cache:
                        measurements, sampled_measurement_noise = (
                            evaluator.sample_measurements(
                                truth=truth,
                                measurement_noise_std=measurement_std,
                                seed=paired_seed + 202,
                            )
                        )
                        measurement_cache[measurement_key] = (
                            np.asarray(measurements),
                            np.asarray(sampled_measurement_noise),
                        )
                    measurements, sampled_measurement_noise = measurement_cache[
                        measurement_key
                    ]

                    for error_threshold in args.error_thresholds:
                        total_counter += 1
                        if args.max_runs is not None and executed_counter >= args.max_runs:
                            break

                        key = run_key(
                            trajectory_index,
                            seed,
                            process_scale,
                            measurement_scale,
                            error_threshold,
                        )
                        if key in completed:
                            skipped_counter += 1
                            continue

                        prefix = (
                            f"[{executed_counter + 1}/{planned}] "
                            f"traj={trajectory_index} seed={seed} "
                            f"q={process_scale:g} r={measurement_scale:g} "
                            f"delta={error_threshold:g}"
                        )
                        print(prefix)
                        try:
                            trace = evaluator.run_sender_trigger_estimator(
                                truth=truth,
                                measurements=measurements,
                                measurement_noise=sampled_measurement_noise,
                                measurement_noise_std=measurement_std,
                                actions=action_sequence,
                                model_states=model_states,
                                standardizers=standardizers,
                                metadata=metadata,
                                state_scales=state_scales,
                                sender_full_state_error_threshold=float(
                                    error_threshold
                                ),
                                sender_full_state_uncertainty_threshold=float(
                                    args.uncertainty_threshold
                                ),
                                sender_uncertainty_sigma_scale=float(
                                    args.uncertainty_sigma_scale
                                ),
                                position_ellipse_scale=float(
                                    args.position_ellipse_scale
                                ),
                                prediction_mode=args.prediction_mode,
                                epistemic_process_scale=float(
                                    args.epistemic_process_scale
                                ),
                                initial_variance=float(args.initial_variance),
                                measurement_variance=float(
                                    args.measurement_variance_floor
                                ),
                            )
                            row = build_metrics(
                                trajectory_index=trajectory_index,
                                seed=seed,
                                process_scale=process_scale,
                                measurement_scale=measurement_scale,
                                error_threshold=error_threshold,
                                linear_process_std=float(process_std[0]),
                                angular_process_std=float(process_std[1]),
                                measurement_noise_std=measurement_std,
                                truth=truth,
                                model_only=model_only,
                                trace=trace,
                                state_scales=state_scales,
                                sampled_process_noise=sampled_process_noise,
                                sampled_measurement_noise=sampled_measurement_noise,
                            )
                            append_csv_row(raw_path, row)
                            completed.add(key)
                            executed_counter += 1
                            print(
                                "  broadcasts="
                                f"{row['total_broadcasts']}/"
                                f"{row['maximum_broadcasts']} | "
                                f"rate={row['broadcast_rate']:.4f} | "
                                "posterior pos RMSE="
                                f"{row['posterior_position_rmse']:.5f}"
                            )
                        except Exception as error:  # keep long sweeps running
                            failed_counter += 1
                            failure = {
                                "trajectory_index": trajectory_index,
                                "seed": seed,
                                "process_noise_scale": process_scale,
                                "measurement_noise_scale": measurement_scale,
                                "error_threshold": error_threshold,
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                                "traceback": traceback.format_exc(),
                            }
                            append_csv_row(failures_path, failure)
                            print(f"  FAILED: {type(error).__name__}: {error}")

                    if args.max_runs is not None and executed_counter >= args.max_runs:
                        break
                if args.max_runs is not None and executed_counter >= args.max_runs:
                    break
            if args.max_runs is not None and executed_counter >= args.max_runs:
                break
        if args.max_runs is not None and executed_counter >= args.max_runs:
            break

    ok_rows = read_ok_rows(raw_path)
    if not ok_rows:
        raise RuntimeError("The sweep produced no successful runs.")
    aggregate = aggregate_rows(ok_rows)
    write_csv(aggregate_path, aggregate)
    if not args.no_plots:
        create_plots(aggregate, args.output_directory)

    print("\n===== Sweep complete =====")
    print("Successful rows:", len(ok_rows))
    print("Executed this invocation:", executed_counter)
    print("Skipped existing rows:", skipped_counter)
    print("Failed this invocation:", failed_counter)
    print("Raw CSV:", raw_path)
    print("Aggregate CSV:", aggregate_path)
    if not args.no_plots:
        print("Heatmaps:", args.output_directory / "heatmaps")
        print("Trade-off plots:", args.output_directory / "tradeoffs")


if __name__ == "__main__":
    main()
