#!/usr/bin/env python3
"""Evaluate dynamics checkpoints trained for different numbers of steps.

The analysis treats each independently trained model seed as the statistical
unit. Trajectory-level metrics are first averaged within each model seed and
then aggregated across model seeds, avoiding overconfident confidence intervals
from treating every trajectory as an independent trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diff_drive_estimator_sanity_utils import (
    build_models,
    load_dataset,
    model_only_rollout,
    resolve_state_scales,
    run_error_trigger_estimator,
    summarize_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_noise005.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        nargs=4,
        action="append",
        metavar=("LABEL", "GRADIENT_STEPS", "MODEL_SEED", "PATH"),
        required=True,
        help=(
            "Repeat for every checkpoint, for example: "
            "--checkpoint steps_500 500 0 checkpoints/steps_500_seed0.pkl"
        ),
    )
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--error-threshold", type=float, default=0.10)
    parser.add_argument(
        "--prediction-mode",
        choices=("ensemble_mean", "infoprop_ci"),
        default="ensemble_mean",
    )
    parser.add_argument("--epistemic-process-scale", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1.0e-6)
    parser.add_argument("--measurement-variance", type=float, default=1.0e-9)
    parser.add_argument("--model-init-seed", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/training_steps_sanity_check/evaluation"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


METRICS = [
    "broadcast_rate",
    "model_position_rmse",
    "prior_position_rmse",
    "posterior_position_rmse",
    "model_normalized_full_state_rmse",
    "posterior_normalized_full_state_rmse",
    "mean_prior_trigger_score",
    "mean_posterior_trigger_score",
    "mean_correction_ratio_on_broadcast",
    "mean_inter_broadcast_interval",
]


def make_per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["model_label", "gradient_steps", "model_seed"]
    return (
        raw.groupby(columns, as_index=False, sort=True)[METRICS]
        .mean(numeric_only=True)
        .sort_values(["gradient_steps", "model_seed"])
        .reset_index(drop=True)
    )


def make_aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_label", "gradient_steps"]
    grouped = per_seed.groupby(keys, sort=True)
    output = grouped.size().rename("num_model_seeds").reset_index()
    for metric in METRICS:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats = stats.rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        output = output.merge(stats, on=keys, how="left")
    return output.sort_values("gradient_steps").reset_index(drop=True)


def ci95(std: np.ndarray, count: np.ndarray) -> np.ndarray:
    return 1.96 * np.nan_to_num(std, nan=0.0) / np.sqrt(np.maximum(count, 1))


def plot_metric(
    aggregate: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    ordered = aggregate.sort_values("gradient_steps")
    x = ordered["gradient_steps"].to_numpy(float)
    y = ordered[mean_column].to_numpy(float)
    yerr = ci95(
        ordered[std_column].to_numpy(float),
        ordered["num_model_seeds"].to_numpy(float),
    )
    plt.figure(figsize=(7.8, 5.5))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xscale("log")
    plt.xticks(x, [str(int(value)) for value in x])
    for x_value, y_value in zip(x, y):
        plt.annotate(
            str(int(x_value)),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("gradient steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_plots(
    aggregate: pd.DataFrame,
    per_seed: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "model_position_rmse_mean",
            "model_position_rmse_std",
            "model-only position RMSE",
            "Model accuracy versus training duration",
            "gradient_steps_vs_model_rmse.png",
        ),
        (
            "broadcast_rate_mean",
            "broadcast_rate_std",
            "broadcast rate",
            "Triggered communication versus training duration",
            "gradient_steps_vs_broadcast_rate.png",
        ),
        (
            "posterior_position_rmse_mean",
            "posterior_position_rmse_std",
            "posterior position RMSE",
            "Estimator accuracy versus training duration",
            "gradient_steps_vs_posterior_rmse.png",
        ),
        (
            "mean_correction_ratio_on_broadcast_mean",
            "mean_correction_ratio_on_broadcast_std",
            "mean correction ratio",
            "Measurement-update effectiveness",
            "gradient_steps_vs_correction_ratio.png",
        ),
    ]
    for mean_col, std_col, ylabel, title, filename in specs:
        plot_metric(
            aggregate,
            mean_col,
            std_col,
            ylabel,
            title,
            output_directory / filename,
            dpi,
        )

    plt.figure(figsize=(7.8, 5.5))
    plt.scatter(
        per_seed["broadcast_rate"],
        per_seed["model_position_rmse"],
    )
    for _, row in per_seed.iterrows():
        plt.annotate(
            f"{int(row['gradient_steps'])}/s{int(row['model_seed'])}",
            (row["broadcast_rate"], row["model_position_rmse"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("mean broadcast rate per trained model")
    plt.ylabel("mean model-only position RMSE per trained model")
    plt.title("Prediction degradation and triggered communication")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_directory / "model_rmse_vs_broadcast_rate_by_seed.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    args = parse_args()
    if args.num_trajectories <= 0:
        raise ValueError("num-trajectories must be positive.")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive.")
    if args.error_threshold < 0.0:
        raise ValueError("error-threshold must be non-negative.")

    dataset = load_dataset(args.dataset)
    states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(states.shape[2])
    if args.trajectory_indices is None:
        selected = dataset["test_indices"][: args.num_trajectories]
    else:
        selected = np.asarray(args.trajectory_indices, dtype=int)
    if len(selected) == 0:
        raise ValueError("No evaluation trajectories were selected.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    print("Trajectories:", selected.tolist())
    print("Error threshold:", args.error_threshold)
    print("Checkpoint count:", len(args.checkpoint))

    for label, steps_text, seed_text, checkpoint_text in args.checkpoint:
        steps = int(steps_text)
        model_seed = int(seed_text)
        checkpoint = Path(checkpoint_text)
        model_states, standardizers, metadata = build_models(
            checkpoint,
            dataset["metadata"],
            num_agents,
            model_init_seed=args.model_init_seed,
        )
        state_scales = resolve_state_scales(metadata)
        recorded_steps = metadata.get("gradient_steps")
        recorded_seed = metadata.get("training_seed")
        if recorded_steps is not None and int(recorded_steps) != steps:
            raise ValueError(
                f"Checkpoint {checkpoint} records gradient_steps={recorded_steps}, "
                f"but CLI specifies {steps}."
            )
        if recorded_seed is not None and int(recorded_seed) != model_seed:
            raise ValueError(
                f"Checkpoint {checkpoint} records training_seed={recorded_seed}, "
                f"but CLI specifies {model_seed}."
            )

        print(
            f"\nModel {label}: steps={steps}, seed={model_seed}, "
            f"checkpoint={checkpoint}"
        )
        for run_index, trajectory_index in enumerate(selected, start=1):
            horizon = min(args.horizon, int(actions.shape[1]))
            truth = np.asarray(states[trajectory_index, : horizon + 1])
            action_sequence = np.asarray(actions[trajectory_index, :horizon])
            model_only = model_only_rollout(
                truth[0],
                action_sequence,
                model_states,
                standardizers,
                metadata,
                prediction_mode=args.prediction_mode,
                epistemic_process_scale=args.epistemic_process_scale,
            )
            trace = run_error_trigger_estimator(
                truth,
                action_sequence,
                model_states,
                standardizers,
                metadata,
                state_scales,
                error_threshold=args.error_threshold,
                prediction_mode=args.prediction_mode,
                epistemic_process_scale=args.epistemic_process_scale,
                initial_variance=args.initial_variance,
                measurement_variance=args.measurement_variance,
            )
            summary = summarize_trace(truth, model_only, trace, state_scales)
            rows.append(
                {
                    "model_label": label,
                    "gradient_steps": steps,
                    "model_seed": model_seed,
                    "trajectory_index": int(trajectory_index),
                    **summary,
                }
            )
            print(
                f"  [{run_index}/{len(selected)}] traj={trajectory_index} "
                f"model_rmse={summary['model_position_rmse']:.5f} "
                f"rate={summary['broadcast_rate']:.4f} "
                f"posterior_rmse={summary['posterior_position_rmse']:.5f}"
            )

    raw = pd.DataFrame(rows)
    per_seed = make_per_seed(raw)
    aggregate = make_aggregate(per_seed)

    raw_path = args.output_directory / "raw_trajectory_runs.csv"
    per_seed_path = args.output_directory / "per_model_seed.csv"
    aggregate_path = args.output_directory / "aggregate.csv"
    raw.to_csv(raw_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    make_plots(
        aggregate,
        per_seed,
        args.output_directory / "plots",
        args.dpi,
    )

    correlation = float("nan")
    valid = per_seed[["model_position_rmse", "broadcast_rate"]].dropna()
    if len(valid) >= 2:
        correlation = float(
            np.corrcoef(
                valid["model_position_rmse"],
                valid["broadcast_rate"],
            )[0, 1]
        )

    print("\n===== Training-steps sanity check complete =====")
    print("Raw trajectory CSV:", raw_path)
    print("Per-model-seed CSV:", per_seed_path)
    print("Aggregate CSV:", aggregate_path)
    print("Plots:", args.output_directory / "plots")
    print("Correlation(model RMSE, broadcast rate):", f"{correlation:.4f}")
    print(
        aggregate[
            [
                "gradient_steps",
                "num_model_seeds",
                "model_position_rmse_mean",
                "broadcast_rate_mean",
                "posterior_position_rmse_mean",
                "mean_correction_ratio_on_broadcast_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
