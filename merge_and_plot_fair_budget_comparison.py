#!/usr/bin/env python3
"""Merge original and fixed-budget baseline results, then create final plots.

Expected inputs:
  1. Original methods:
       no_comm, periodic_20, error_0.1, full_comm
  2. New methods:
       periodic_50, periodic_100

The merged design contains:
    10 trajectories x 5 seeds x 3 scenarios x 6 methods = 900 unique runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_sender_baseline_comparison as base


METHOD_ORDER = [
    "no_comm",
    "periodic_100",
    "error_0.1",
    "periodic_50",
    "periodic_20",
    "full_comm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-raw",
        type=Path,
        default=Path(
            "results/sender_baseline_comparison_5workers/"
            "merged/raw_runs.csv"
        ),
    )
    parser.add_argument(
        "--periodic-raw",
        type=Path,
        default=Path(
            "results/periodic_budget_baselines_5workers/"
            "merged/raw_runs.csv"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/fair_budget_method_comparison"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def ci95(std: pd.Series, count: pd.Series) -> pd.Series:
    return 1.96 * std / np.sqrt(count.clip(lower=1))


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_pareto_plots(aggregate: pd.DataFrame, output: Path, dpi: int) -> None:
    scenarios = (
        aggregate[
            ["process_noise_scale", "measurement_noise_scale"]
        ]
        .drop_duplicates()
        .sort_values(
            ["process_noise_scale", "measurement_noise_scale"]
        )
    )

    for q, r in scenarios.itertuples(index=False, name=None):
        selected = aggregate[
            np.isclose(aggregate["process_noise_scale"], float(q))
            & np.isclose(
                aggregate["measurement_noise_scale"], float(r)
            )
        ].copy()

        order = {
            name: index for index, name in enumerate(METHOD_ORDER)
        }
        selected["method_order"] = selected["method"].map(order)
        selected = selected.sort_values(
            ["broadcast_rate_mean", "method_order"]
        )

        figure, axis = plt.subplots(figsize=(8.8, 6.4))
        axis.errorbar(
            selected["broadcast_rate_mean"],
            selected["posterior_position_rmse_mean"],
            xerr=ci95(
                selected["broadcast_rate_std"],
                selected["num_runs"],
            ),
            yerr=ci95(
                selected["posterior_position_rmse_std"],
                selected["num_runs"],
            ),
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

        axis.set_xlabel("mean broadcast rate")
        axis.set_ylabel("mean posterior position RMSE")
        axis.set_title(
            "Communication–accuracy trade-off\n"
            f"process scale={float(q):g}, "
            f"measurement scale={float(r):g}"
        )
        axis.grid(True, alpha=0.3)
        save_figure(
            output / f"pareto_all_q{float(q):g}_r{float(r):g}.png",
            dpi,
        )

        zoom = selected[selected["method"] != "full_comm"].copy()
        figure, axis = plt.subplots(figsize=(8.8, 6.4))
        axis.errorbar(
            zoom["broadcast_rate_mean"],
            zoom["posterior_position_rmse_mean"],
            xerr=ci95(
                zoom["broadcast_rate_std"],
                zoom["num_runs"],
            ),
            yerr=ci95(
                zoom["posterior_position_rmse_std"],
                zoom["num_runs"],
            ),
            fmt="o",
            capsize=3,
        )
        for _, row in zoom.iterrows():
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
        xmax = max(
            0.06,
            1.18 * float(zoom["broadcast_rate_mean"].max()),
        )
        axis.set_xlim(-0.002, xmax)
        axis.set_xlabel("mean broadcast rate")
        axis.set_ylabel("mean posterior position RMSE")
        axis.set_title(
            "Fair-budget communication–accuracy comparison\n"
            f"process scale={float(q):g}, "
            f"measurement scale={float(r):g}"
        )
        axis.grid(True, alpha=0.3)
        save_figure(
            output / f"pareto_zoom_q{float(q):g}_r{float(r):g}.png",
            dpi,
        )


def annotated_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    output: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(2.3 * len(column_labels) + 3.2,
                 0.72 * len(row_labels) + 2.8)
    )
    image = axis.imshow(matrix, aspect="auto")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xlabel("noise scenario")
    axis.set_ylabel("method")
    axis.set_title(title)

    finite = matrix[np.isfinite(matrix)]
    midpoint = (
        0.5 * (float(np.min(finite)) + float(np.max(finite)))
        if finite.size
        else 0.0
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:.4f}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if np.isfinite(value) and value > midpoint
                    else "black"
                ),
                fontsize=8.5,
            )
    save_figure(output, dpi)


def make_heatmaps(
    aggregate: pd.DataFrame, output: Path, dpi: int
) -> None:
    methods_present = set(aggregate["method"])
    methods = [
        method for method in METHOD_ORDER if method in methods_present
    ]
    scenarios = (
        aggregate[
            ["process_noise_scale", "measurement_noise_scale"]
        ]
        .drop_duplicates()
        .sort_values(
            ["process_noise_scale", "measurement_noise_scale"]
        )
    )
    scenario_pairs = list(
        scenarios.itertuples(index=False, name=None)
    )
    scenario_labels = [
        f"q={float(q):g}, r={float(r):g}"
        for q, r in scenario_pairs
    ]

    specs = [
        (
            "broadcast_rate_mean",
            "Mean broadcast rate",
            "broadcast rate",
            "heatmap_broadcast_rate.png",
        ),
        (
            "posterior_position_rmse_mean",
            "Mean posterior position RMSE",
            "position RMSE",
            "heatmap_position_rmse.png",
        ),
        (
            "posterior_normalized_full_state_rmse_mean",
            "Mean normalized full-state RMSE",
            "normalized full-state RMSE",
            "heatmap_full_state_rmse.png",
        ),
    ]

    columns = pd.MultiIndex.from_tuples(scenario_pairs)
    for metric, title, label, filename in specs:
        pivot = (
            aggregate.pivot_table(
                index="method",
                columns=[
                    "process_noise_scale",
                    "measurement_noise_scale",
                ],
                values=metric,
                aggfunc="first",
            )
            .reindex(methods)
            .reindex(columns=columns)
        )
        annotated_heatmap(
            matrix=pivot.to_numpy(dtype=float),
            row_labels=methods,
            column_labels=scenario_labels,
            title=title,
            colorbar_label=label,
            output=output / filename,
            dpi=dpi,
        )


def make_fair_budget_table(
    aggregate: pd.DataFrame, output: Path
) -> None:
    selected = aggregate[
        aggregate["method"].isin(
            ["periodic_100", "error_0.1", "periodic_50"]
        )
    ].copy()
    selected = selected[
        [
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
        ]
    ]
    selected.to_csv(output / "fair_budget_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.original_raw.exists():
        raise FileNotFoundError(args.original_raw)
    if not args.periodic_raw.exists():
        raise FileNotFoundError(args.periodic_raw)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(args.original_raw)
    periodic = pd.read_csv(args.periodic_raw)
    raw = pd.concat([original, periodic], ignore_index=True)

    keys = [
        "trajectory_index",
        "seed",
        "process_noise_scale",
        "measurement_noise_scale",
        "method",
    ]
    raw = (
        raw.drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    raw.to_csv(args.output_directory / "raw_runs.csv", index=False)

    rows = raw.to_dict(orient="records")
    aggregate_rows = base.aggregate_rows(rows)
    base.write_csv(
        args.output_directory / "aggregate.csv",
        aggregate_rows,
    )
    aggregate = pd.DataFrame(aggregate_rows)

    print("Unique merged rows:", len(raw))
    print("Expected unique rows: 900")
    print("Methods:", sorted(raw["method"].unique()))

    plots = args.output_directory / "plots"
    make_pareto_plots(aggregate, plots, args.dpi)
    make_heatmaps(aggregate, plots, args.dpi)
    make_fair_budget_table(aggregate, args.output_directory)

    print("Merged raw CSV:", args.output_directory / "raw_runs.csv")
    print("Aggregate CSV:", args.output_directory / "aggregate.csv")
    print("Fair-budget table:", args.output_directory / "fair_budget_summary.csv")
    print("Plots:", plots)


if __name__ == "__main__":
    main()
