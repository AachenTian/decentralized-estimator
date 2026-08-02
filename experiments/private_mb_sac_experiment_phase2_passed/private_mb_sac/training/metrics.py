"""Canonical metric names for the private three-owner experiment.

The coordinator should gather owner-local metrics first, then call the helper
functions in this module. Keeping the names centralized prevents W&B dashboards
from fragmenting because of spelling differences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


OWNER_KEYS = {
    # Real collection
    "real/env_steps_round",
    "real/env_steps_total",
    "real/episodes_round",
    "real/terminal_transitions_round",
    "real/terminal_transitions_total",
    "real/return_mean",
    "real/collision_rate",
    "real/replay_size",
    # Dynamics
    "dynamics/loss",
    "dynamics/nll",
    "dynamics/normalized_mse",
    "dynamics/mean_logvar",
    "dynamics/updates_total",
    "dynamics/normalizer_samples",
    "dynamics/skipped",
    "dynamics/state_rmse",
    "dynamics/position_rmse",
    "dynamics/velocity_rmse",
    "dynamics/noncollision_velocity_rmse",
    "dynamics/collision_velocity_rmse",
    "dynamics/epistemic_mean",
    "dynamics/aleatoric_mean",
    "dynamics/coverage95",
    "dynamics/grad_norm",
    "dynamics/nonterminal_fraction",
    # Model rollout
    "model/generated_transitions",
    "model/replay_size",
    "model/mean_horizon",
    "model/max_horizon",
    "model/termination_horizon_fraction",
    "model/termination_done_fraction",
    "model/termination_collision_fraction",
    "model/termination_uncertainty_fraction",
    "model/predicted_collision_rate",
    "model/epistemic_mean",
    # Critic
    "critic/loss",
    "critic/q1_loss",
    "critic/q2_loss",
    "critic/q1_mean",
    "critic/q2_mean",
    "critic/target_q_mean",
    "critic/td_abs",
    "critic/fixed_td_abs",
    "critic/grad_norm",
    "critic/updates_total",
    # Actor / temperature
    "actor/loss",
    "actor/policy_q",
    "actor/entropy",
    "actor/log_prob_mean",
    "actor/action_saturation_rate",
    "actor/grad_norm",
    "actor/updates_total",
    "alpha/value",
    "alpha/loss",
    # Timing
    "timing/collection_seconds",
    "timing/dynamics_seconds",
    "timing/model_rollout_seconds",
    "timing/sac_seconds",
}


def validate_owner_metrics(metrics: Mapping[str, Any]) -> None:
    unknown = sorted(set(metrics) - OWNER_KEYS)
    if unknown:
        raise KeyError(
            "Unknown owner metric names. Add them to OWNER_KEYS first: "
            + ", ".join(unknown)
        )


def aggregate_owner_metrics(
    metrics_by_owner: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute system means/min/max for scalar owner metrics."""
    if not metrics_by_owner:
        return {}

    common_keys = set.intersection(
        *(set(metrics.keys()) for metrics in metrics_by_owner)
    )
    output: dict[str, float] = {}

    for key in sorted(common_keys):
        values = []
        for metrics in metrics_by_owner:
            try:
                array = np.asarray(metrics[key], dtype=np.float64)
            except Exception:
                values = []
                break
            if array.ndim != 0:
                values = []
                break
            values.append(float(array))

        if not values:
            continue

        safe_key = key.replace("/", "_")
        output[f"owners_mean/{safe_key}"] = float(np.mean(values))
        output[f"owners_min/{safe_key}"] = float(np.min(values))
        output[f"owners_max/{safe_key}"] = float(np.max(values))

    return output
