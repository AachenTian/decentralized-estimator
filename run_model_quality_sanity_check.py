#!/usr/bin/env python3
"""Evaluate whether worse learned models trigger communication more often."""

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
        nargs=3,
        action="append",
        metavar=("LABEL", "TRAIN_FRACTION", "PATH"),
        required=True,
        help=(
            "Repeat for each model, e.g. --checkpoint model_100 1.0 "
            "checkpoints/model_100.pkl"
        ),
    )
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--error-threshold", type=float, default=0.1)
    parser.add_argument(
        "--prediction-mode",
        choices=("ensemble_mean", "infoprop_ci"),
        default="ensemble_mean",
    )
    parser.add_argument("--epistemic-process-scale", type=float, default=1.0)
    parser.add_argument("--initial-variance", type=float, default=1e-6)
    parser.add_argument("--measurement-variance", type=float, default=1e-9)
    parser.add_argument("--model-init-seed", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/model_quality_sanity_check"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def aggregate_rows(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    grouped = raw.groupby(["model_label", "train_data_fraction"], sort=True)
    output = grouped.size().rename("num_trajectories").reset_index()
    for metric in metrics:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats = stats.rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        output = output.merge(
            stats,
            on=["model_label", "train_data_fraction"],
            how="left",
        )
    return output.sort_values("train_data_fraction").reset_index(drop=True)


def ci95(std: np.ndarray, count: np.ndarray) -> np.ndarray:
    return 1.96 * np.nan_to_num(std, nan=0.0) / np.sqrt(np.maximum(count, 1))


def save_metric_plot(
    aggregate: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    ordered = aggregate.sort_values("train_data_fraction")
    x = 100.0 * ordered["train_data_fraction"].to_numpy(float)
    y = ordered[mean_column].to_numpy(float)
    error = ci95(
        ordered[std_column].to_numpy(float),
        ordered["num_trajectories"].to_numpy(float),
    )
    plt.figure(figsize=(7.6, 5.4))
    plt.errorbar(x, y, yerr=error, marker="o", capsize=4)
    for x_value, y_value, label in zip(x, y, ordered["model_label"]):
        plt.annotate(
            str(label),
            (x_value, y_value),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("training-data fraction (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_plots(aggregate: pd.DataFrame, output: Path, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    save_metric_plot(
        aggregate,
        "model_position_rmse_mean",
        "model_position_rmse_std",
        "model-only position RMSE",
        "Model quality versus training-data fraction",
        output / "training_fraction_vs_model_rmse.png",
        dpi,
    )
    save_metric_plot(
        aggregate,
        "broadcast_rate_mean",
        "broadcast_rate_std",
        "broadcast rate",
        "Sender-trigger rate versus training-data fraction",
        output / "training_fraction_vs_broadcast_rate.png",
        dpi,
    )
    save_metric_plot(
        aggregate,
        "posterior_position_rmse_mean",
        "posterior_position_rmse_std",
        "posterior position RMSE",
        "Estimator error versus training-data fraction",
        output / "training_fraction_vs_posterior_rmse.png",
        dpi,
    )
    save_metric_plot(
        aggregate,
        "mean_correction_ratio_on_broadcast_mean",
        "mean_correction_ratio_on_broadcast_std",
        "mean correction ratio",
        "Measurement-update correction effectiveness",
        output / "training_fraction_vs_correction_ratio.png",
        dpi,
    )

    ordered = aggregate.sort_values("train_data_fraction")
    plt.figure(figsize=(7.6, 5.4))
    plt.errorbar(
        ordered["broadcast_rate_mean"],
        ordered["model_position_rmse_mean"],
        xerr=ci95(
            ordered["broadcast_rate_std"].to_numpy(float),
            ordered["num_trajectories"].to_numpy(float),
        ),
        yerr=ci95(
            ordered["model_position_rmse_std"].to_numpy(float),
            ordered["num_trajectories"].to_numpy(float),
        ),
        fmt="o",
        capsize=4,
    )
    for _, row in ordered.iterrows():
        plt.annotate(
            f"{100 * row['train_data_fraction']:.0f}%",
            (row["broadcast_rate_mean"], row["model_position_rmse_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    plt.xlabel("mean broadcast rate")
    plt.ylabel("mean model-only position RMSE")
    plt.title("Prediction degradation and triggered communication")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output / "model_rmse_vs_broadcast_rate.png",
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

    for label, fraction_text, checkpoint_text in args.checkpoint:
        fraction = float(fraction_text)
        checkpoint = Path(checkpoint_text)
        model_states, standardizers, metadata = build_models(
            checkpoint,
            dataset["metadata"],
            num_agents,
            model_init_seed=args.model_init_seed,
        )
        state_scales = resolve_state_scales(metadata)
        print(f"\nModel {label}: fraction={fraction:g}, checkpoint={checkpoint}")

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
            summary = summarize_trace(
                truth, model_only, trace, state_scales
            )
            row = {
                "model_label": label,
                "train_data_fraction": fraction,
                "trajectory_index": int(trajectory_index),
                **summary,
            }
            rows.append(row)
            print(
                f"  [{run_index}/{len(selected)}] traj={trajectory_index} "
                f"model_rmse={summary['model_position_rmse']:.5f} "
                f"rate={summary['broadcast_rate']:.4f} "
                f"posterior_rmse={summary['posterior_position_rmse']:.5f}"
            )

    raw = pd.DataFrame(rows)
    aggregate = aggregate_rows(raw)
    raw_path = args.output_directory / "raw_runs.csv"
    aggregate_path = args.output_directory / "aggregate.csv"
    raw.to_csv(raw_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    make_plots(aggregate, args.output_directory / "plots", args.dpi)

    print("\n===== Model-quality sanity check complete =====")
    print("Raw CSV:", raw_path)
    print("Aggregate CSV:", aggregate_path)
    print("Plots:", args.output_directory / "plots")
    print(
        aggregate[
            [
                "model_label",
                "train_data_fraction",
                "model_position_rmse_mean",
                "broadcast_rate_mean",
                "posterior_position_rmse_mean",
                "mean_correction_ratio_on_broadcast_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
