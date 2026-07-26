#!/usr/bin/env python3
"""Evaluate full, random-matched, and held-out action-region models.

The held-out region is read from checkpoint metadata. Results are first
averaged within each independently trained model seed and then aggregated
across model seeds for confidence intervals.
"""

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


MODEL_ORDER = ["full_coverage", "random_matched", "heldout_region"]
PHASE_ORDER = ["pre_ood", "in_ood", "post_ood"]


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
        metavar=("MODEL", "MODEL_SEED", "PATH"),
        required=True,
    )
    parser.add_argument("--num-trajectories", type=int, default=10)
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
        default=Path("results/action_ood_sanity_check/evaluation"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_region_definition(checkpoints: list[list[str]]) -> dict:
    definitions: list[dict] = []
    for model, _, path_text in checkpoints:
        if model != "heldout_region":
            continue
        _, _, metadata = load_checkpoint(Path(path_text))
        definition = metadata.get("ood_region_definition")
        if not isinstance(definition, dict):
            raise ValueError(
                f"Held-out checkpoint {path_text} has no ood_region_definition."
            )
        definitions.append(definition)
    if not definitions:
        raise ValueError("At least one heldout_region checkpoint is required.")
    reference = definitions[0]
    for definition in definitions[1:]:
        if definition != reference:
            raise ValueError(
                "Held-out checkpoints do not share the same region definition."
            )
    return reference


def region_mask(
    current: np.ndarray,
    actions: np.ndarray,
    definition: dict,
) -> np.ndarray:
    region_type = str(definition["region_type"])
    if region_type == "linear_upper_saturation":
        return (
            (current[..., 3] > float(definition["v_threshold"]))
            & (actions[..., 0] > float(definition["a_threshold"]))
        )
    if region_type == "angular_saturation":
        aligned_alpha = np.sign(current[..., 4]) * actions[..., 1]
        return (
            (
                np.abs(current[..., 4])
                > float(definition["abs_omega_threshold"])
            )
            & (
                aligned_alpha
                > float(definition["aligned_alpha_threshold"])
            )
        )
    if region_type == "high_control_norm":
        max_a = float(definition["max_linear_acceleration"])
        max_alpha = float(definition["max_angular_acceleration"])
        norm = np.sqrt(
            (actions[..., 0] / max_a) ** 2
            + (actions[..., 1] / max_alpha) ** 2
        )
        return norm > float(definition["control_norm_threshold"])
    raise ValueError(f"Unsupported region type: {region_type}")


def select_ood_trajectories(
    states: np.ndarray,
    actions: np.ndarray,
    test_indices: np.ndarray,
    definition: dict,
    horizon: int,
    count: int,
) -> np.ndarray:
    scored: list[tuple[float, int]] = []
    for trajectory_index in test_indices:
        current = states[trajectory_index, :horizon]
        act = actions[trajectory_index, :horizon]
        occupancy = float(region_mask(current, act, definition).mean())
        scored.append((occupancy, int(trajectory_index)))
    scored.sort(reverse=True)
    selected = [index for occupancy, index in scored if occupancy > 0.0][:count]
    if not selected:
        raise RuntimeError(
            "No test trajectory enters the held-out action region. "
            "Use lower quantiles or generate targeted trajectories."
        )
    return np.asarray(selected, dtype=int)


def phase_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
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
    posterior_error = wrapped_state_error(posterior_phase, truth_phase) / state_scales
    broadcast_mask = comm_phase.astype(bool)
    if np.any(broadcast_mask):
        correction = (
            prior_score_phase[broadcast_mask]
            - posterior_score_phase[broadcast_mask]
        ) / np.maximum(prior_score_phase[broadcast_mask], 1.0e-12)
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
        "mean_posterior_trigger_score": float(np.mean(posterior_score_phase)),
        "mean_correction_ratio_on_broadcast": correction_ratio,
    }


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
]


def make_per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "model_seed", "phase"]
    return (
        raw.groupby(keys, as_index=False, sort=False)[METRICS]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )


def make_aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "phase"]
    grouped = per_seed.groupby(keys, sort=False)
    output = grouped.size().rename("num_model_seeds").reset_index()
    for metric in METRICS:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats = stats.rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        output = output.merge(stats, on=keys, how="left")
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    phase_rank = {name: index for index, name in enumerate(PHASE_ORDER)}
    output["_model_rank"] = output["model"].map(model_rank)
    output["_phase_rank"] = output["phase"].map(phase_rank)
    return (
        output.sort_values(["_model_rank", "_phase_rank"])
        .drop(columns=["_model_rank", "_phase_rank"])
        .reset_index(drop=True)
    )


def ci95(std: float, count: int) -> float:
    if not np.isfinite(std):
        return 0.0
    return 1.96 * std / np.sqrt(max(count, 1))


def plot_grouped_metric(
    aggregate: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    x = np.arange(len(PHASE_ORDER), dtype=float)
    width = 0.24
    plt.figure(figsize=(9.0, 5.8))
    for model_index, model in enumerate(MODEL_ORDER):
        means = []
        errors = []
        for phase in PHASE_ORDER:
            row = aggregate[
                (aggregate["model"] == model)
                & (aggregate["phase"] == phase)
            ]
            if row.empty:
                means.append(np.nan)
                errors.append(0.0)
                continue
            item = row.iloc[0]
            means.append(float(item[f"{metric}_mean"]))
            errors.append(
                ci95(
                    float(item[f"{metric}_std"]),
                    int(item["num_model_seeds"]),
                )
            )
        offset = (model_index - 1.0) * width
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
    actions: np.ndarray,
    model_only: np.ndarray,
    trace: dict[str, np.ndarray],
    definition: dict,
    error_threshold: float,
    output: Path,
    dpi: int,
) -> None:
    masks = region_mask(truth[:-1], actions, definition)
    occupancy = masks.mean(axis=0)
    agent_id = int(np.argmax(occupancy))
    mask = masks[:, agent_id]
    steps = np.arange(1, len(mask) + 1)
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

    figure, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), sharex=True)
    for start, stop in contiguous_spans(mask):
        for axis in axes:
            axis.axvspan(start + 1, stop, alpha=0.15)

    region_type = str(definition["region_type"])
    if region_type == "linear_upper_saturation":
        axes[0].plot(steps, truth[:-1, agent_id, 3], label="linear velocity v")
        axes[0].axhline(
            float(definition["v_threshold"]), linestyle="--", label="v boundary"
        )
        axes[1].plot(steps, actions[:, agent_id, 0], label="linear acceleration a")
        axes[1].axhline(
            float(definition["a_threshold"]), linestyle="--", label="a boundary"
        )
    elif region_type == "angular_saturation":
        axes[0].plot(
            steps,
            np.abs(truth[:-1, agent_id, 4]),
            label="absolute angular velocity",
        )
        axes[0].axhline(
            float(definition["abs_omega_threshold"]),
            linestyle="--",
            label="omega boundary",
        )
        aligned = np.sign(truth[:-1, agent_id, 4]) * actions[:, agent_id, 1]
        axes[1].plot(steps, aligned, label="aligned angular acceleration")
        axes[1].axhline(
            float(definition["aligned_alpha_threshold"]),
            linestyle="--",
            label="alpha boundary",
        )
    else:
        max_a = float(definition["max_linear_acceleration"])
        max_alpha = float(definition["max_angular_acceleration"])
        norm = np.sqrt(
            (actions[:, agent_id, 0] / max_a) ** 2
            + (actions[:, agent_id, 1] / max_alpha) ** 2
        )
        axes[0].plot(steps, norm, label="normalized control norm")
        axes[0].axhline(
            float(definition["control_norm_threshold"]),
            linestyle="--",
            label="region boundary",
        )
        axes[1].step(steps, mask.astype(int), where="mid", label="OOD indicator")

    axes[0].set_title(
        f"Held-out action-region response, representative agent {agent_id}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, prior_scores, label="prior trigger score")
    axes[2].plot(steps, posterior_scores, label="posterior score")
    axes[2].axhline(error_threshold, linestyle="--", label="trigger threshold")
    for event_step in steps[communications]:
        axes[2].axvline(event_step, alpha=0.25)
    axes[2].set_ylabel("normalized error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps, model_position_error, label="model-only")
    axes[3].plot(steps, prior_position_error, label="common prior")
    axes[3].plot(steps, posterior_position_error, label="common posterior")
    axes[3].set_xlabel("time step")
    axes[3].set_ylabel("position error")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def make_in_ood_comparison(per_seed: pd.DataFrame) -> pd.DataFrame:
    in_ood = per_seed[per_seed["phase"] == "in_ood"].copy()
    pivot = in_ood.pivot(index="model_seed", columns="model", values=METRICS)
    rows = []
    for seed in sorted(in_ood["model_seed"].unique()):
        row: dict[str, float | int] = {"model_seed": int(seed)}
        for baseline in ("full_coverage", "random_matched"):
            for metric in (
                "model_position_rmse",
                "broadcast_rate",
                "posterior_position_rmse",
            ):
                heldout_value = float(pivot.loc[seed, (metric, "heldout_region")])
                baseline_value = float(pivot.loc[seed, (metric, baseline)])
                row[f"heldout_minus_{baseline}_{metric}"] = (
                    heldout_value - baseline_value
                )
                row[f"heldout_over_{baseline}_{metric}"] = (
                    heldout_value / max(abs(baseline_value), 1.0e-12)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    definition = read_region_definition(args.checkpoint)
    dataset = load_dataset(args.dataset)
    states = dataset["local_states"]
    actions = dataset["actions"]
    num_agents = int(states.shape[2])
    horizon = min(args.horizon, int(actions.shape[1]))
    selected = select_ood_trajectories(
        states,
        actions,
        dataset["test_indices"],
        definition,
        horizon,
        args.num_trajectories,
    )

    print("Held-out region:", definition)
    print("Selected OOD trajectories:", selected.tolist())
    args.output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    representative = None

    for model_label, seed_text, checkpoint_text in args.checkpoint:
        if model_label not in MODEL_ORDER:
            raise ValueError(f"Unknown model label: {model_label}")
        model_seed = int(seed_text)
        checkpoint = Path(checkpoint_text)
        model_states, standardizers, metadata = build_models(
            checkpoint,
            dataset["metadata"],
            num_agents,
            model_init_seed=args.model_init_seed,
        )
        state_scales = resolve_state_scales(metadata)
        print(
            f"\nEvaluating model={model_label}, seed={model_seed}, "
            f"checkpoint={checkpoint}"
        )

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
            if (
                model_label == "heldout_region"
                and model_seed == min(
                    int(item[1])
                    for item in args.checkpoint
                    if item[0] == "heldout_region"
                )
                and representative is None
            ):
                representative = (truth, action_sequence, model_only, trace)

            masks = region_mask(truth[:-1], action_sequence, definition)
            for agent_id in range(num_agents):
                for phase, mask in phase_masks(masks[:, agent_id]).items():
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
                            "model_seed": model_seed,
                            "trajectory_index": int(trajectory_index),
                            "agent_id": agent_id,
                            "phase": phase,
                            **summary,
                        }
                    )

    raw = pd.DataFrame(rows)
    per_seed = make_per_seed(raw)
    aggregate = make_aggregate(per_seed)
    comparison = make_in_ood_comparison(per_seed)

    raw_path = args.output_directory / "raw_phase_metrics.csv"
    per_seed_path = args.output_directory / "per_model_seed_phase_metrics.csv"
    aggregate_path = args.output_directory / "aggregate_phase_metrics.csv"
    comparison_path = args.output_directory / "in_ood_paired_comparison.csv"
    raw.to_csv(raw_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    plots = args.output_directory / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "broadcast_rate",
            "broadcast rate",
            "Communication response to the held-out action region",
            "phase_broadcast_rate.png",
        ),
        (
            "model_position_rmse",
            "model-only position RMSE",
            "Model degradation in the held-out action region",
            "phase_model_position_rmse.png",
        ),
        (
            "posterior_position_rmse",
            "posterior position RMSE",
            "Estimator accuracy in the held-out action region",
            "phase_posterior_position_rmse.png",
        ),
        (
            "mean_correction_ratio_on_broadcast",
            "mean correction ratio",
            "Measurement-update effectiveness",
            "phase_correction_ratio.png",
        ),
    ]
    for metric, ylabel, title, filename in specs:
        plot_grouped_metric(
            aggregate,
            metric,
            ylabel,
            title,
            plots / filename,
            args.dpi,
        )

    if representative is not None:
        plot_representative_trace(
            *representative,
            definition,
            args.error_threshold,
            plots / "representative_action_ood_trace.png",
            args.dpi,
        )

    print("\n===== Action-region OOD sanity check complete =====")
    print("Raw metrics:", raw_path)
    print("Per-model-seed metrics:", per_seed_path)
    print("Aggregate metrics:", aggregate_path)
    print("Paired in-OOD comparison:", comparison_path)
    print("Plots:", plots)
    print(
        aggregate[
            [
                "model",
                "phase",
                "num_model_seeds",
                "broadcast_rate_mean",
                "model_position_rmse_mean",
                "posterior_position_rmse_mean",
                "mean_correction_ratio_on_broadcast_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
