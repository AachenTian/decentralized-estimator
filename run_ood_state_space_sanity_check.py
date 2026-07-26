#!/usr/bin/env python3
"""Evaluate estimator behavior in a state region excluded from model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diff_drive_estimator_sanity_utils import (
    build_models,
    load_checkpoint,
    load_dataset,
    model_only_rollout,
    resolve_state_scales,
    run_error_trigger_estimator,
    wrapped_state_error,
)

STATE_DIMENSIONS = {"px": 0, "py": 1, "phi": 2, "v": 3, "omega": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_noise005.npz"),
    )
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-checkpoint", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=10)
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
        "--holdout-dimension",
        choices=tuple(STATE_DIMENSIONS),
        default=None,
        help="Override checkpoint metadata when provided.",
    )
    parser.add_argument("--holdout-threshold", type=float, default=None)
    parser.add_argument(
        "--holdout-side",
        choices=("upper", "lower"),
        default=None,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/ood_state_space_sanity_check"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve_holdout_definition(args: argparse.Namespace) -> tuple[str, int, float, str]:
    _, _, metadata = load_checkpoint(args.heldout_checkpoint)
    dimension = args.holdout_dimension or metadata.get("holdout_dimension")
    threshold = (
        args.holdout_threshold
        if args.holdout_threshold is not None
        else metadata.get("holdout_threshold")
    )
    side = args.holdout_side or metadata.get("holdout_side")
    if dimension not in STATE_DIMENSIONS:
        raise ValueError(
            "Held-out checkpoint metadata does not contain a valid "
            "holdout_dimension. Pass --holdout-dimension explicitly."
        )
    if threshold is None:
        raise ValueError(
            "Held-out checkpoint metadata does not contain a threshold. "
            "Pass --holdout-threshold explicitly."
        )
    if side not in ("upper", "lower"):
        raise ValueError(
            "Held-out checkpoint metadata does not contain a valid side. "
            "Pass --holdout-side explicitly."
        )
    return dimension, STATE_DIMENSIONS[dimension], float(threshold), side


def is_ood(values: np.ndarray, threshold: float, side: str) -> np.ndarray:
    if side == "upper":
        return values > threshold
    return values < threshold


def select_ood_trajectories(
    states: np.ndarray,
    test_indices: np.ndarray,
    dimension_index: int,
    threshold: float,
    side: str,
    horizon: int,
    count: int,
) -> np.ndarray:
    scored: list[tuple[float, int]] = []
    for trajectory_index in test_indices:
        values = states[
            trajectory_index,
            1 : horizon + 1,
            :,
            dimension_index,
        ]
        occupancy = float(is_ood(values, threshold, side).mean())
        scored.append((occupancy, int(trajectory_index)))
    scored.sort(reverse=True)
    selected = [index for occupancy, index in scored if occupancy > 0.0][:count]
    if not selected:
        raise RuntimeError(
            "No test trajectory enters the held-out region. Choose another "
            "dimension/quantile or generate targeted evaluation trajectories."
        )
    return np.asarray(selected, dtype=int)


def phase_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Split one agent's timeline into pre, in, and post OOD phases."""
    phases = {
        "pre_ood": np.zeros_like(mask, dtype=bool),
        "in_ood": mask.astype(bool),
        "post_ood": np.zeros_like(mask, dtype=bool),
    }
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        phases["pre_ood"][:] = True
        return phases
    phases["pre_ood"][: indices[0]] = True
    phases["post_ood"][indices[-1] + 1 :] = True
    return phases


def phase_summary(
    phase_mask: np.ndarray,
    truth_next: np.ndarray,
    model_next: np.ndarray,
    prior_next: np.ndarray,
    posterior_next: np.ndarray,
    communications: np.ndarray,
    prior_scores: np.ndarray,
    posterior_scores: np.ndarray,
    state_scales: np.ndarray,
) -> dict[str, float] | None:
    if not np.any(phase_mask):
        return None
    truth_phase = truth_next[phase_mask]
    model_phase = model_next[phase_mask]
    prior_phase = prior_next[phase_mask]
    posterior_phase = posterior_next[phase_mask]
    comm_phase = communications[phase_mask]
    prior_score_phase = prior_scores[phase_mask]
    posterior_score_phase = posterior_scores[phase_mask]

    def position_rmse(estimate: np.ndarray) -> float:
        error = estimate[..., :2] - truth_phase[..., :2]
        return float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))

    model_error = wrapped_state_error(model_phase, truth_phase) / state_scales
    posterior_error = (
        wrapped_state_error(posterior_phase, truth_phase) / state_scales
    )
    broadcast_mask = comm_phase.astype(bool)
    if np.any(broadcast_mask):
        correction = (
            prior_score_phase[broadcast_mask]
            - posterior_score_phase[broadcast_mask]
        ) / np.maximum(prior_score_phase[broadcast_mask], 1e-12)
        correction_ratio = float(np.mean(correction))
    else:
        correction_ratio = float("nan")

    return {
        "num_steps": int(np.sum(phase_mask)),
        "broadcast_rate": float(np.mean(comm_phase)),
        "model_position_rmse": position_rmse(model_phase),
        "prior_position_rmse": position_rmse(prior_phase),
        "posterior_position_rmse": position_rmse(posterior_phase),
        "model_normalized_full_state_rmse": float(
            np.sqrt(np.mean(model_error**2))
        ),
        "posterior_normalized_full_state_rmse": float(
            np.sqrt(np.mean(posterior_error**2))
        ),
        "mean_prior_trigger_score": float(np.mean(prior_score_phase)),
        "mean_posterior_trigger_score": float(
            np.mean(posterior_score_phase)
        ),
        "mean_correction_ratio_on_broadcast": correction_ratio,
    }


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
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
    ]
    grouped = raw.groupby(["model", "phase"], sort=False)
    output = grouped.size().rename("num_segments").reset_index()
    for metric in metrics:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats = stats.rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        output = output.merge(stats, on=["model", "phase"], how="left")
    return output


def plot_grouped_metric(
    aggregate_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    phases = ["pre_ood", "in_ood", "post_ood"]
    models = ["full_coverage", "heldout_region"]
    x = np.arange(len(phases), dtype=float)
    width = 0.36
    plt.figure(figsize=(8.2, 5.6))
    for model_index, model in enumerate(models):
        means = []
        errors = []
        for phase in phases:
            row = aggregate_df[
                (aggregate_df["model"] == model)
                & (aggregate_df["phase"] == phase)
            ]
            if row.empty:
                means.append(np.nan)
                errors.append(0.0)
                continue
            item = row.iloc[0]
            means.append(float(item[f"{metric}_mean"]))
            std = float(item[f"{metric}_std"])
            count = max(int(item["num_segments"]), 1)
            errors.append(1.96 * (0.0 if np.isnan(std) else std) / np.sqrt(count))
        offset = (model_index - 0.5) * width
        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=4,
            label=model,
        )
    plt.xticks(x, ["pre-OOD", "in-OOD", "post-OOD"])
    plt.xlabel("trajectory phase")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def plot_representative_trace(
    truth: np.ndarray,
    model_only: np.ndarray,
    trace: dict[str, np.ndarray],
    dimension: str,
    dimension_index: int,
    threshold: float,
    side: str,
    error_threshold: float,
    output: Path,
    dpi: int,
) -> None:
    occupancy = is_ood(truth[1:, :, dimension_index], threshold, side).mean(axis=0)
    agent_id = int(np.argmax(occupancy))
    values = truth[1:, agent_id, dimension_index]
    mask = is_ood(values, threshold, side)
    steps = np.arange(1, len(values) + 1)
    communications = trace["communications"][:, agent_id]
    prior_scores = trace["prior_scores"][:, agent_id]
    posterior_scores = trace["posterior_scores"][:, agent_id]
    model_position_error = np.linalg.norm(
        model_only[1:, agent_id, :2] - truth[1:, agent_id, :2], axis=-1
    )
    prior_position_error = np.linalg.norm(
        trace["predicted_means"][:, agent_id, :2]
        - truth[1:, agent_id, :2],
        axis=-1,
    )
    posterior_position_error = np.linalg.norm(
        trace["posterior_means"][:, agent_id, :2]
        - truth[1:, agent_id, :2],
        axis=-1,
    )

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    for start, stop in contiguous_spans(mask):
        for axis in axes:
            axis.axvspan(start + 1, stop, alpha=0.15)

    axes[0].plot(steps, values, label=f"true {dimension}")
    axes[0].axhline(threshold, linestyle="--", label="holdout boundary")
    axes[0].set_ylabel(dimension)
    axes[0].set_title(
        f"Held-out-region response, representative agent {agent_id}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, prior_scores, label="prior trigger score")
    axes[1].plot(steps, posterior_scores, label="posterior score")
    axes[1].axhline(error_threshold, linestyle="--", label="trigger threshold")
    event_steps = steps[communications]
    for event_step in event_steps:
        axes[1].axvline(event_step, alpha=0.25)
    axes[1].set_ylabel("normalized error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, model_position_error, label="model-only")
    axes[2].plot(steps, prior_position_error, label="common prior")
    axes[2].plot(steps, posterior_position_error, label="common posterior")
    axes[2].set_xlabel("time step")
    axes[2].set_ylabel("position error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dimension, dimension_index, threshold, side = resolve_holdout_definition(args)
    dataset = load_dataset(args.dataset)
    states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(states.shape[2])
    horizon = min(args.horizon, int(actions.shape[1]))
    selected = select_ood_trajectories(
        states,
        dataset["test_indices"],
        dimension_index,
        threshold,
        side,
        horizon,
        args.num_trajectories,
    )

    print(
        f"Held-out region: {dimension} {side} threshold {threshold:.6f}"
    )
    print("Selected OOD trajectories:", selected.tolist())
    args.output_directory.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("full_coverage", args.full_checkpoint),
        ("heldout_region", args.heldout_checkpoint),
    ]
    rows: list[dict[str, float | int | str]] = []
    representative = None

    for model_label, checkpoint in model_specs:
        model_states, standardizers, metadata = build_models(
            checkpoint,
            dataset["metadata"],
            num_agents,
            model_init_seed=args.model_init_seed,
        )
        state_scales = resolve_state_scales(metadata)
        for trajectory_index in selected:
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
            if model_label == "heldout_region" and representative is None:
                representative = (truth, model_only, trace)

            for agent_id in range(num_agents):
                values = truth[1:, agent_id, dimension_index]
                ood_mask = is_ood(values, threshold, side)
                for phase, mask in phase_masks(ood_mask).items():
                    summary = phase_summary(
                        mask,
                        truth[1:, agent_id],
                        model_only[1:, agent_id],
                        trace["predicted_means"][:, agent_id],
                        trace["posterior_means"][:, agent_id],
                        trace["communications"][:, agent_id],
                        trace["prior_scores"][:, agent_id],
                        trace["posterior_scores"][:, agent_id],
                        state_scales,
                    )
                    if summary is None:
                        continue
                    rows.append(
                        {
                            "model": model_label,
                            "trajectory_index": int(trajectory_index),
                            "agent_id": agent_id,
                            "phase": phase,
                            **summary,
                        }
                    )

    raw = pd.DataFrame(rows)
    aggregate_df = aggregate(raw)
    raw.to_csv(args.output_directory / "raw_phase_metrics.csv", index=False)
    aggregate_df.to_csv(
        args.output_directory / "aggregate_phase_metrics.csv", index=False
    )

    plots = args.output_directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_grouped_metric(
        aggregate_df,
        "broadcast_rate",
        "broadcast rate",
        "Communication response before, inside, and after the held-out region",
        plots / "phase_broadcast_rate.png",
        args.dpi,
    )
    plot_grouped_metric(
        aggregate_df,
        "posterior_position_rmse",
        "posterior position RMSE",
        "Estimator accuracy before, inside, and after the held-out region",
        plots / "phase_posterior_position_rmse.png",
        args.dpi,
    )
    plot_grouped_metric(
        aggregate_df,
        "model_position_rmse",
        "model-only position RMSE",
        "Open-loop model degradation in the held-out region",
        plots / "phase_model_position_rmse.png",
        args.dpi,
    )
    plot_grouped_metric(
        aggregate_df,
        "mean_correction_ratio_on_broadcast",
        "mean correction ratio",
        "Measurement-update effectiveness in the held-out region",
        plots / "phase_correction_ratio.png",
        args.dpi,
    )

    if representative is not None:
        truth, model_only, trace = representative
        plot_representative_trace(
            truth,
            model_only,
            trace,
            dimension,
            dimension_index,
            threshold,
            side,
            args.error_threshold,
            plots / "representative_ood_trace.png",
            args.dpi,
        )

    print("\n===== OOD state-space sanity check complete =====")
    print("Raw metrics:", args.output_directory / "raw_phase_metrics.csv")
    print(
        "Aggregate metrics:",
        args.output_directory / "aggregate_phase_metrics.csv",
    )
    print("Plots:", plots)
    print(
        aggregate_df[
            [
                "model",
                "phase",
                "broadcast_rate_mean",
                "model_position_rmse_mean",
                "posterior_position_rmse_mean",
                "mean_correction_ratio_on_broadcast_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
