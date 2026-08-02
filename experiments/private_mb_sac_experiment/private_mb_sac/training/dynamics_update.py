"""Independent owner-local training for five-member dynamics ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import numpy as np

from private_mb_sac.core.types import DynamicsNormalizer
from private_mb_sac.dynamics.ensemble import (
    initialize_independent_dynamics_states,
)
from private_mb_sac.dynamics.loss import (
    dynamics_train_step,
    evaluate_dynamics_batch,
)
from private_mb_sac.dynamics.normalization import (
    fit_private_dynamics_normalizer,
)


@dataclass
class OwnerDynamicsRuntime:
    owner_id: int
    train_state: Any
    normalizer: DynamicsNormalizer | None = None
    updates_total: int = 0


def initialize_owner_dynamics_runtimes(
    config: Any,
) -> tuple[Any, list[OwnerDynamicsRuntime]]:
    model, states = initialize_independent_dynamics_states(
        seed=int(config.experiment.seed) + 40_000,
        num_owners=int(config.experiment.num_agents),
        ensemble_size=int(config.dynamics.ensemble_size),
        input_dim=int(config.dynamics.input_dim),
        output_dim=int(config.dynamics.output_dim),
        hidden_dims=config.dynamics.hidden_dims,
        learning_rate=float(config.dynamics.learning_rate),
        maximum_gradient_norm=float(config.dynamics.maximum_gradient_norm),
        minimum_logvar=float(config.dynamics.min_logvar),
        maximum_logvar=float(config.dynamics.max_logvar),
    )
    runtimes = [
        OwnerDynamicsRuntime(owner_id=index, train_state=state)
        for index, state in enumerate(states)
    ]
    return model, runtimes


def train_private_owner_dynamics(
    *,
    runtime: OwnerDynamicsRuntime,
    replay: Any,
    config: Any,
    updates: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Fit/freeze private stats, update one ensemble, and evaluate one-step error."""
    valid_count = len(replay.valid_dynamics_indices)
    minimum_samples = int(config.dynamics.normalization_min_samples)
    if valid_count < minimum_samples:
        return {
            "dynamics/skipped": 1.0,
            "dynamics/nonterminal_fraction": float(
                valid_count / max(len(replay), 1)
            ),
            "dynamics/updates_total": float(runtime.updates_total),
            "dynamics/normalizer_samples": 0.0,
        }

    if runtime.normalizer is None:
        runtime.normalizer = fit_private_dynamics_normalizer(
            replay,
            minimum_std=float(config.dynamics.normalization_min_std),
            clip=float(config.dynamics.normalization_clip),
            agent_state_dim=int(config.dynamics.output_dim),
        )

    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)
    train_metrics: list[dict[str, float]] = []
    start = perf_counter()
    ensemble_size = int(config.dynamics.ensemble_size)

    for _ in range(int(updates)):
        batch = replay.sample(
            int(batch_size),
            rng,
            nonterminal_only=True,
            replace=True,
        )
        jax_key, bootstrap_key = jax.random.split(jax_key)
        bootstrap_indices = jax.random.randint(
            bootstrap_key,
            shape=(ensemble_size, int(batch_size)),
            minval=0,
            maxval=int(batch_size),
        )
        runtime.train_state, metrics = dynamics_train_step(
            runtime.train_state,
            runtime.normalizer,
            batch,
            bootstrap_indices,
        )
        jax.block_until_ready(metrics["loss"])
        train_metrics.append(
            {key: float(np.asarray(value)) for key, value in metrics.items()}
        )
        runtime.updates_total += 1

    evaluation_size = min(
        int(config.dynamics.evaluation_batch_size),
        valid_count,
    )
    evaluation_batch = replay.sample(
        evaluation_size,
        rng,
        nonterminal_only=True,
        replace=False,
    )
    evaluation = evaluate_dynamics_batch(
        runtime.train_state,
        runtime.normalizer,
        evaluation_batch,
        ensemble_size=ensemble_size,
    )
    evaluation = {
        key: float(np.asarray(value)) for key, value in evaluation.items()
    }

    def average(name: str) -> float:
        return float(np.mean([metrics[name] for metrics in train_metrics]))

    return {
        "dynamics/skipped": 0.0,
        "dynamics/loss": average("loss"),
        "dynamics/nll": float(evaluation["nll"]),
        "dynamics/normalized_mse": average("normalized_mse"),
        "dynamics/state_rmse": float(evaluation["state_rmse"]),
        "dynamics/position_rmse": float(evaluation["position_rmse"]),
        "dynamics/velocity_rmse": float(evaluation["velocity_rmse"]),
        "dynamics/noncollision_velocity_rmse": float(
            evaluation["noncollision_velocity_rmse"]
        ),
        "dynamics/collision_velocity_rmse": float(
            evaluation["collision_velocity_rmse"]
        ),
        "dynamics/epistemic_mean": float(evaluation["epistemic_mean"]),
        "dynamics/aleatoric_mean": float(evaluation["aleatoric_mean"]),
        "dynamics/coverage95": float(evaluation["coverage95"]),
        "dynamics/grad_norm": average("grad_norm"),
        "dynamics/mean_logvar": average("mean_logvar"),
        "dynamics/nonterminal_fraction": float(
            valid_count / max(len(replay), 1)
        ),
        "dynamics/updates_total": float(runtime.updates_total),
        "dynamics/normalizer_samples": float(
            np.asarray(runtime.normalizer.sample_count)
        ),
        "timing/dynamics_seconds": float(perf_counter() - start),
    }


def pairwise_parameter_distance(states: list[OwnerDynamicsRuntime]) -> list[float]:
    """Return L2 distances between owner parameter trees."""
    flattened = []
    for runtime in states:
        leaves = [
            np.asarray(leaf).reshape(-1)
            for leaf in jax.tree_util.tree_leaves(runtime.train_state.params)
        ]
        flattened.append(np.concatenate(leaves))
    distances = []
    for first in range(len(flattened)):
        for second in range(first + 1, len(flattened)):
            distances.append(
                float(np.linalg.norm(flattened[first] - flattened[second]))
            )
    return distances
