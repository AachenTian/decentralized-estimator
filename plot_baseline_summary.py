#!/usr/bin/env python3
"""Create publication-oriented summary plots for baseline comparison results.

Usage:
    python plot_baseline_summary.py \
        --aggregate results/sender_baseline_comparison_5workers/merged/aggregate.csv \
        --output-directory results/sender_baseline_comparison_5workers/merged/summary_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["no_comm", "error_0.1", "periodic_20", "full_comm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        required=True,
        help="Path to aggregate.csv produced by the baseline comparison.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory in which plots and summary tables are saved.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def scenario_label(q: float, r: float) -> str:
    return f"q={q:g}, r={r:g}"


def ordered_methods(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    ordered = [name for name in METHOD_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def annotated_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    output: Path,
    dpi: int,
    value_format: str,
) -> None:
    figure, axis = plt.subplots(figsize=(2.2 * len(column_labels) + 3.0,
                                         0.75 * len(row_labels) + 2.6))
    image = axis.imshow(matrix, aspect="auto")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)

    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("noise scenario")
    axis.set_ylabel("communication method")
    axis.set_title(title)

    finite_values = matrix[np.isfinite(matrix)]
    midpoint = (
        0.5 * (float(np.min(finite_values)) + float(np.max(finite_values)))
        if finite_values.size
        else 0.0
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                text = "NA"
            else:
                text = format(value, value_format)
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value > midpoint else "black",
                fontsize=9,
            )

    save_figure(output, dpi)


def grouped_bar_with_ci(
    df: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    methods = ordered_methods(df["method"])
    scenarios = (
        df[["process_noise_scale", "measurement_noise_scale"]]
        .drop_duplicates()
        .sort_values(["process_noise_scale", "measurement_noise_scale"])
    )
    scenario_pairs = list(scenarios.itertuples(index=False, name=None))
    labels = [scenario_label(float(q), float(r)) for q, r in scenario_pairs]

    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(methods), 1)

    figure, axis = plt.subplots(figsize=(max(9.0, 2.2 * len(labels) + 4.0), 6.2))

    for method_index, method in enumerate(methods):
        means = []
        cis = []
        for q, r in scenario_pairs:
            selected = df[
                (df["method"] == method)
                & np.isclose(df["process_noise_scale"], float(q))
                & np.isclose(df["measurement_noise_scale"], float(r))
            ]
            if selected.empty:
                means.append(np.nan)
                cis.append(0.0)
                continue
            row = selected.iloc[0]
            mean = float(row[metric_mean])
            std = float(row[metric_std])
            count = max(int(row["num_runs"]), 1)
            ci95 = 1.96 * std / np.sqrt(count)
            means.append(mean)
            cis.append(ci95)

        offset = (method_index - 0.5 * (len(methods) - 1)) * width
        axis.bar(
            x + offset,
            means,
            width=width,
            yerr=cis,
            capsize=3,
            label=method,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel("noise scenario")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    save_figure(output, dpi)


def zoomed_pareto_plots(df: pd.DataFrame, output_directory: Path, dpi: int) -> None:
    scenarios = (
        df[["process_noise_scale", "measurement_noise_scale"]]
        .drop_duplicates()
        .sort_values(["process_noise_scale", "measurement_noise_scale"])
    )

    for q, r in scenarios.itertuples(index=False, name=None):
        selected = df[
            np.isclose(df["process_noise_scale"], float(q))
            & np.isclose(df["measurement_noise_scale"], float(r))
        ].copy()
        selected = selected.sort_values("broadcast_rate_mean")

        figure, axis = plt.subplots(figsize=(8.2, 6.2))
        axis.errorbar(
            selected["broadcast_rate_mean"],
            selected["posterior_position_rmse_mean"],
            xerr=1.96
            * selected["broadcast_rate_std"]
            / np.sqrt(selected["num_runs"].clip(lower=1)),
            yerr=1.96
            * selected["posterior_position_rmse_std"]
            / np.sqrt(selected["num_runs"].clip(lower=1)),
            fmt="o",
            capsize=3,
        )

        for _, row in selected.iterrows():
            axis.annotate(
                str(row["method"]),
                (
                    float(row["broadcast_rate_mean"]),
                    float(row["posterior_position_rmse_mean"]),
                ),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
            )

        non_full = selected[selected["method"] != "full_comm"]
        if not non_full.empty:
            xmax = max(0.06, 1.25 * float(non_full["broadcast_rate_mean"].max()))
            axis.set_xlim(left=-0.002, right=xmax)

        axis.set_xlabel("mean broadcast rate")
        axis.set_ylabel("mean posterior position RMSE")
        axis.set_title(
            "Communication–accuracy trade-off (zoomed)\n"
            + scenario_label(float(q), float(r))
        )
        axis.grid(True, alpha=0.3)

        filename = f"pareto_zoom_q{float(q):g}_r{float(r):g}.png"
        save_figure(output_directory / filename, dpi)


def main() -> None:
    args = parse_args()
    if not args.aggregate.exists():
        raise FileNotFoundError(args.aggregate)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.aggregate)

    required = {
        "process_noise_scale",
        "measurement_noise_scale",
        "method",
        "num_runs",
        "broadcast_rate_mean",
        "broadcast_rate_std",
        "posterior_position_rmse_mean",
        "posterior_position_rmse_std",
        "posterior_normalized_full_state_rmse_mean",
        "posterior_normalized_full_state_rmse_std",
        "relative_position_rmse_improvement_mean",
        "relative_position_rmse_improvement_std",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"aggregate.csv is missing columns: {missing}")

    methods = ordered_methods(df["method"])
    scenario_df = (
        df[["process_noise_scale", "measurement_noise_scale"]]
        .drop_duplicates()
        .sort_values(["process_noise_scale", "measurement_noise_scale"])
    )
    scenario_pairs = list(scenario_df.itertuples(index=False, name=None))
    scenario_labels = [
        scenario_label(float(q), float(r)) for q, r in scenario_pairs
    ]

    metric_specs = [
        (
            "broadcast_rate_mean",
            "Mean broadcast rate",
            "broadcast rate",
            "heatmap_broadcast_rate.png",
            ".4f",
        ),
        (
            "posterior_position_rmse_mean",
            "Mean posterior position RMSE",
            "position RMSE",
            "heatmap_position_rmse.png",
            ".4f",
        ),
        (
            "posterior_normalized_full_state_rmse_mean",
            "Mean normalized full-state RMSE",
            "normalized full-state RMSE",
            "heatmap_full_state_rmse.png",
            ".4f",
        ),
        (
            "relative_position_rmse_improvement_mean",
            "Mean relative position-RMSE improvement",
            "relative improvement",
            "heatmap_relative_improvement.png",
            ".3f",
        ),
    ]

    for metric, title, colorbar_label, filename, value_format in metric_specs:
        pivot = (
            df.pivot_table(
                index="method",
                columns=["process_noise_scale", "measurement_noise_scale"],
                values=metric,
                aggfunc="first",
            )
            .reindex(methods)
            .reindex(columns=pd.MultiIndex.from_tuples(scenario_pairs))
        )
        annotated_heatmap(
            matrix=pivot.to_numpy(dtype=float),
            row_labels=methods,
            column_labels=scenario_labels,
            title=title,
            colorbar_label=colorbar_label,
            output=args.output_directory / filename,
            dpi=args.dpi,
            value_format=value_format,
        )

    grouped_bar_with_ci(
        df=df,
        metric_mean="posterior_position_rmse_mean",
        metric_std="posterior_position_rmse_std",
        ylabel="posterior position RMSE",
        title="Posterior position RMSE with 95% confidence intervals",
        output=args.output_directory / "bar_position_rmse_ci95.png",
        dpi=args.dpi,
    )
    grouped_bar_with_ci(
        df=df,
        metric_mean="broadcast_rate_mean",
        metric_std="broadcast_rate_std",
        ylabel="broadcast rate",
        title="Broadcast rate with 95% confidence intervals",
        output=args.output_directory / "bar_broadcast_rate_ci95.png",
        dpi=args.dpi,
    )
    grouped_bar_with_ci(
        df=df,
        metric_mean="posterior_normalized_full_state_rmse_mean",
        metric_std="posterior_normalized_full_state_rmse_std",
        ylabel="normalized full-state RMSE",
        title="Normalized full-state RMSE with 95% confidence intervals",
        output=args.output_directory / "bar_full_state_rmse_ci95.png",
        dpi=args.dpi,
    )

    zoomed_pareto_plots(df, args.output_directory, args.dpi)

    clean_columns = [
        "process_noise_scale",
        "measurement_noise_scale",
        "method",
        "num_runs",
        "broadcast_rate_mean",
        "broadcast_rate_std",
        "posterior_position_rmse_mean",
        "posterior_position_rmse_std",
        "posterior_normalized_full_state_rmse_mean",
        "posterior_normalized_full_state_rmse_std",
        "relative_position_rmse_improvement_mean",
        "relative_position_rmse_improvement_std",
    ]
    df[clean_columns].to_csv(
        args.output_directory / "baseline_summary_table.csv",
        index=False,
    )

    print("Created summary plots in:", args.output_directory)
    for path in sorted(args.output_directory.glob("*.png")):
        print(" -", path.name)
    print(" - baseline_summary_table.csv")


if __name__ == "__main__":
    main()
