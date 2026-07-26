#!/usr/bin/env python3
"""Shared utilities for diff-drive estimator sanity-check experiments."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aorpo.agents.structured_diff_drive_dynamics import (
    init_structured_transition_models,
    predict_structured_next,
)
from aorpo.estimator.global_belief import (
    direct_agent_observation_update,
    extract_agent_covariance,
    initialize_global_belief,
    make_isotropic_covariance,
)
from aorpo.estimator.structured_global_belief import (
    predict_structured_global_belief_oracle_actions,
)
from aorpo.estimator.normalized_full_state_trigger import (
    normalized_full_state_error_score,
)


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        result = {
            "local_states": np.asarray(data["local_states"]),
            "actions": np.asarray(data["actions"]),
            "train_indices": np.asarray(data["train_trajectory_indices"]),
            "validation_indices": np.asarray(
                data["validation_trajectory_indices"]
            ),
            "test_indices": np.asarray(data["test_trajectory_indices"]),
            "metadata": json.loads(str(data["metadata_json"])),
        }
    states = result["local_states"]
    actions = result["actions"]
    if states.ndim != 4 or states.shape[-1] != 5:
        raise ValueError("Expected local_states shape (K, H+1, N, 5).")
    if actions.ndim != 4 or actions.shape[-1] != 2:
        raise ValueError("Expected actions shape (K, H, N, 2).")
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError("There must be one more state than action.")
    return result


def parse_checkpoint_object(
    loaded: Any,
) -> tuple[Any, list[Any], dict[str, Any]]:
    if isinstance(loaded, dict):
        saved_models = None
        for key in (
            "model_params",
            "saved_model_params",
            "params",
            "model_states",
        ):
            if key in loaded:
                saved_models = loaded[key]
                break
        standardizers = None
        for key in ("standardizers", "local_standardizers"):
            if key in loaded:
                standardizers = loaded[key]
                break
        metadata = loaded.get("metadata", {})
        if saved_models is None or standardizers is None:
            raise KeyError(
                "Could not locate model parameters and standardizers. "
                f"Checkpoint keys: {sorted(loaded.keys())}"
            )
        return saved_models, list(standardizers), dict(metadata)
    if isinstance(loaded, (tuple, list)) and len(loaded) == 3:
        saved_models, standardizers, metadata = loaded
        return saved_models, list(standardizers), dict(metadata)
    raise TypeError("Unsupported checkpoint layout.")


def load_checkpoint(
    path: Path,
) -> tuple[Any, list[Any], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    utility_error = None
    try:
        from aorpo.utils.checkpoints import (
            load_independent_dynamics_checkpoint,
        )

        loaded = load_independent_dynamics_checkpoint(str(path))
        return parse_checkpoint_object(loaded)
    except Exception as error:  # pragma: no cover - repository dependent
        utility_error = error
    try:
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        return parse_checkpoint_object(loaded)
    except Exception as pickle_error:
        raise RuntimeError(
            "Unable to load checkpoint.\n"
            f"Repository utility error: {utility_error}\n"
            f"Direct pickle error: {pickle_error}"
        ) from pickle_error


def make_model_config(metadata: dict[str, Any]) -> Any:
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=int(metadata.get("ensemble_size", 5)),
            hidden_dims=tuple(metadata.get("hidden_dims", [256, 256])),
            min_logvar=float(metadata.get("min_logvar", -6.0)),
            max_logvar=float(metadata.get("max_logvar", 0.5)),
            lr=float(metadata.get("learning_rate", 1.0e-3)),
        )
    )


def restore_model_states(initialized_states: list[Any], saved: Any) -> list[Any]:
    try:
        from aorpo.utils.checkpoints import restore_independent_model_states

        return list(
            restore_independent_model_states(initialized_states, saved)
        )
    except Exception:
        saved_items = list(saved)
        if len(saved_items) != len(initialized_states):
            raise ValueError(
                "Saved model count does not match initialized model count: "
                f"{len(saved_items)} versus {len(initialized_states)}."
            )
        restored = []
        for initialized, saved_item in zip(initialized_states, saved_items):
            params = (
                saved_item.params
                if hasattr(saved_item, "params")
                else saved_item
            )
            restored.append(initialized.replace(params=params))
        return restored


def build_models(
    checkpoint_path: Path,
    dataset_metadata: dict[str, Any],
    num_agents: int,
    model_init_seed: int = 0,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    saved, standardizers, checkpoint_metadata = load_checkpoint(
        checkpoint_path
    )
    metadata = dict(dataset_metadata)
    metadata.update(checkpoint_metadata)
    cfg = make_model_config(metadata)
    _, initialized = init_structured_transition_models(
        rng=jax.random.PRNGKey(model_init_seed),
        num_agents=num_agents,
        cfg=cfg,
    )
    model_states = restore_model_states(initialized, saved)
    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, got {len(standardizers)}."
        )
    return model_states, standardizers, metadata


def dynamics_kwargs(metadata: dict[str, Any]) -> dict[str, float]:
    return {
        "dt": float(metadata["dt"]),
        "min_v": float(metadata["min_v"]),
        "max_v": float(metadata["max_v"]),
        "max_omega": float(metadata["max_omega"]),
    }


def resolve_state_scales(metadata: dict[str, Any]) -> np.ndarray:
    max_abs_v = max(
        abs(float(metadata["min_v"])),
        abs(float(metadata["max_v"])),
    )
    max_omega = abs(float(metadata["max_omega"]))
    return np.asarray(
        [1.0, 1.0, np.pi, max(max_abs_v, 1e-6), max(max_omega, 1e-6)],
        dtype=np.float32,
    )


def model_only_rollout(
    initial_states: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    prediction_mode: str = "ensemble_mean",
    epistemic_process_scale: float = 1.0,
) -> np.ndarray:
    current = jnp.asarray(initial_states)
    trajectory = [np.asarray(current)]
    for action_t in actions:
        next_states = []
        for agent_id in range(current.shape[0]):
            next_state, _ = predict_structured_next(
                train_state=model_states[agent_id],
                standardizer=standardizers[agent_id],
                local_state=current[agent_id : agent_id + 1],
                local_action=jnp.asarray(
                    action_t[agent_id : agent_id + 1]
                ),
                prediction_mode=prediction_mode,
                epistemic_process_scale=epistemic_process_scale,
                **dynamics_kwargs(metadata),
            )
            next_states.append(next_state[0])
        current = jnp.stack(next_states, axis=0)
        trajectory.append(np.asarray(current))
    return np.stack(trajectory, axis=0)


def _reshape_belief_mean(
    belief: Any, num_agents: int, local_state_dim: int
) -> np.ndarray:
    return np.asarray(
        belief.mean.reshape(1, num_agents, local_state_dim)[0]
    )


def run_error_trigger_estimator(
    truth: np.ndarray,
    actions: np.ndarray,
    model_states: list[Any],
    standardizers: list[Any],
    metadata: dict[str, Any],
    state_scales: np.ndarray,
    error_threshold: float = 0.1,
    prediction_mode: str = "ensemble_mean",
    epistemic_process_scale: float = 1.0,
    initial_variance: float = 1e-6,
    measurement_variance: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Run the prediction-error-only sender trigger.

    This intentionally disables covariance triggering so the sanity check
    isolates whether worse model predictions cause more broadcasts.
    """
    num_agents = truth.shape[1]
    local_state_dim = truth.shape[2]
    belief = initialize_global_belief(
        initial_local_states=jnp.asarray(truth[0:1]),
        initial_variance=initial_variance,
    )
    measurement_covariance = make_isotropic_covariance(
        batch_size=1,
        dimension=local_state_dim,
        variance=measurement_variance,
        dtype=jnp.asarray(truth).dtype,
    )
    scales = jnp.asarray(state_scales)

    predicted_means: list[np.ndarray] = []
    posterior_means: list[np.ndarray] = []
    prior_scores: list[np.ndarray] = []
    posterior_scores: list[np.ndarray] = []
    communications: list[np.ndarray] = []

    for step_index in range(actions.shape[0]):
        prior, _ = predict_structured_global_belief_oracle_actions(
            belief=belief,
            model_states=model_states,
            standardizers=standardizers,
            joint_actions=jnp.asarray(actions[step_index : step_index + 1]),
            num_agents=num_agents,
            prediction_mode=prediction_mode,
            epistemic_process_scale=epistemic_process_scale,
            **dynamics_kwargs(metadata),
        )
        true_next = jnp.asarray(truth[step_index + 1])
        prior_local = _reshape_belief_mean(
            prior, num_agents, local_state_dim
        )
        decisions = np.zeros(num_agents, dtype=bool)
        scores = np.zeros(num_agents, dtype=np.float32)

        for agent_id in range(num_agents):
            score = normalized_full_state_error_score(
                predicted_local_mean=jnp.asarray(
                    prior_local[agent_id : agent_id + 1]
                ),
                observed_local_state=true_next[agent_id : agent_id + 1],
                state_scales=scales,
            )[0]
            scores[agent_id] = float(score)
            decisions[agent_id] = bool(score > error_threshold)

        belief = prior
        for agent_id in range(num_agents):
            if not decisions[agent_id]:
                continue
            # Sequential updates all start from the latest common posterior.
            belief, _ = direct_agent_observation_update(
                predicted_belief=belief,
                observed_agent_id=agent_id,
                measurement=true_next[agent_id : agent_id + 1],
                measurement_covariance=measurement_covariance,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

        posterior_local = _reshape_belief_mean(
            belief, num_agents, local_state_dim
        )
        post_scores = np.zeros(num_agents, dtype=np.float32)
        for agent_id in range(num_agents):
            post_scores[agent_id] = float(
                normalized_full_state_error_score(
                    predicted_local_mean=jnp.asarray(
                        posterior_local[agent_id : agent_id + 1]
                    ),
                    observed_local_state=true_next[
                        agent_id : agent_id + 1
                    ],
                    state_scales=scales,
                )[0]
            )

        predicted_means.append(prior_local)
        posterior_means.append(posterior_local)
        prior_scores.append(scores)
        posterior_scores.append(post_scores)
        communications.append(decisions)

    return {
        "predicted_means": np.stack(predicted_means),
        "posterior_means": np.stack(posterior_means),
        "prior_scores": np.stack(prior_scores),
        "posterior_scores": np.stack(posterior_scores),
        "communications": np.stack(communications),
    }


def wrapped_state_error(
    estimate: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    error = np.asarray(estimate) - np.asarray(truth)
    error[..., 2] = np.arctan2(
        np.sin(error[..., 2]), np.cos(error[..., 2])
    )
    return error


def position_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(estimate)[..., :2] - np.asarray(truth)[..., :2]
    return float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))


def normalized_full_state_rmse(
    estimate: np.ndarray,
    truth: np.ndarray,
    state_scales: np.ndarray,
) -> float:
    error = wrapped_state_error(estimate, truth) / state_scales
    return float(np.sqrt(np.mean(error**2)))


def mean_inter_broadcast_interval(communications: np.ndarray) -> float:
    intervals: list[int] = []
    for agent_id in range(communications.shape[1]):
        event_steps = np.flatnonzero(communications[:, agent_id])
        if len(event_steps) >= 2:
            intervals.extend(np.diff(event_steps).tolist())
    return float(np.mean(intervals)) if intervals else float("nan")


def summarize_trace(
    truth: np.ndarray,
    model_only: np.ndarray,
    trace: dict[str, np.ndarray],
    state_scales: np.ndarray,
) -> dict[str, float]:
    posterior = np.concatenate(
        [truth[0:1], trace["posterior_means"]], axis=0
    )
    prior_truth = truth[1:]
    communications = trace["communications"]
    broadcast_mask = communications.astype(bool)
    prior_scores = trace["prior_scores"]
    posterior_scores = trace["posterior_scores"]
    if np.any(broadcast_mask):
        reductions = (
            prior_scores[broadcast_mask] - posterior_scores[broadcast_mask]
        ) / np.maximum(prior_scores[broadcast_mask], 1e-12)
        correction_ratio = float(np.mean(reductions))
    else:
        correction_ratio = float("nan")

    return {
        "broadcasts": int(communications.sum()),
        "maximum_broadcasts": int(communications.size),
        "broadcast_rate": float(communications.mean()),
        "model_position_rmse": position_rmse(model_only, truth),
        "prior_position_rmse": position_rmse(
            trace["predicted_means"], prior_truth
        ),
        "posterior_position_rmse": position_rmse(posterior, truth),
        "model_normalized_full_state_rmse": normalized_full_state_rmse(
            model_only, truth, state_scales
        ),
        "posterior_normalized_full_state_rmse": (
            normalized_full_state_rmse(posterior, truth, state_scales)
        ),
        "mean_prior_trigger_score": float(np.mean(prior_scores)),
        "mean_posterior_trigger_score": float(
            np.mean(posterior_scores)
        ),
        "mean_correction_ratio_on_broadcast": correction_ratio,
        "mean_inter_broadcast_interval": mean_inter_broadcast_interval(
            communications
        ),
    }
