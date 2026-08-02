"""Best/latest checkpoints and exact local-training resume support."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np


def _host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: np.asarray(jax.device_get(leaf)),
        tree,
    )


def _export_train_state(state) -> dict:
    return {
        "step": int(np.asarray(state.step)),
        "params": _host_tree(state.params),
        "opt_state": _host_tree(state.opt_state),
    }


def _restore_train_state(state, payload: Mapping[str, Any]):
    return state.replace(
        step=int(payload["step"]),
        params=payload["params"],
        opt_state=payload["opt_state"],
    )


def _export_sac_batch(batch):
    if batch is None:
        return None
    return type(batch)(
        *[_host_tree(value) for value in batch]
    )


def save_latest_checkpoint(
    path: str | Path,
    *,
    round_index: int,
    best_return: float,
    best_round: int,
    actor_normalizer: Any,
    owners: Sequence[Any],
    dynamics_runtimes: Sequence[Any],
    model_runtimes: Sequence[Any],
    sac_runtimes: Sequence[Any],
    config: Mapping[str, Any],
    include_replays: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "private_mb_sac_full_resume_v1",
        "round_index": int(round_index),
        "best_return": float(best_return),
        "best_round": int(best_round),
        "actor_normalizer": _host_tree(actor_normalizer),
        "config": dict(config),
        "owners": [
            {
                "owner_id": int(runtime.owner_id),
                "env_state": _host_tree(runtime.env_state),
                "features": _host_tree(runtime.features),
                "env_steps_total": int(runtime.env_steps_total),
                "terminal_transitions_total": int(
                    runtime.terminal_transitions_total
                ),
                "real_replay": (
                    runtime.real_replay.state_dict()
                    if include_replays else None
                ),
            }
            for runtime in owners
        ],
        "dynamics": [
            {
                "owner_id": int(runtime.owner_id),
                "train_state": _export_train_state(
                    runtime.train_state
                ),
                "normalizer": (
                    None
                    if runtime.normalizer is None
                    else _host_tree(runtime.normalizer)
                ),
                "updates_total": int(runtime.updates_total),
            }
            for runtime in dynamics_runtimes
        ],
        "models": [
            {
                "owner_id": int(runtime.owner_id),
                "generated_total": int(runtime.generated_total),
                "model_replay": (
                    runtime.model_replay.state_dict()
                    if include_replays else None
                ),
            }
            for runtime in model_runtimes
        ],
        "sac": [
            {
                "owner_id": int(runtime.owner_id),
                "actor_state": _export_train_state(
                    runtime.learner_state.actor_state
                ),
                "critic_state": _export_train_state(
                    runtime.learner_state.critic_state
                ),
                "alpha_state": _export_train_state(
                    runtime.learner_state.alpha_state
                ),
                "target_critic_params": _host_tree(
                    runtime.learner_state.target_critic_params
                ),
                "critic_updates_total": int(
                    runtime.critic_updates_total
                ),
                "actor_updates_total": int(
                    runtime.actor_updates_total
                ),
                "fixed_batch": _export_sac_batch(
                    runtime.fixed_batch
                ),
            }
            for runtime in sac_runtimes
        ],
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def save_actor_checkpoint(
    path: str | Path,
    *,
    round_index: int,
    evaluation_metrics: Mapping[str, Any],
    actor_params: Sequence[Any],
    actor_normalizer: Any,
    config: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "private_mb_sac_actor_tuple_v1",
        "algorithm": "private_model_based_sac_snapshot_sync",
        "round": int(round_index),
        "evaluation_metrics": dict(evaluation_metrics),
        "actor_params": tuple(
            _host_tree(params) for params in actor_params
        ),
        "actor_normalizer": _host_tree(actor_normalizer),
        "config": dict(config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "private_mb_sac_full_resume_v1":
        raise ValueError(
            f"Unsupported checkpoint format: {payload.get('format')}"
        )
    return payload


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    owners: Sequence[Any],
    dynamics_runtimes: Sequence[Any],
    model_runtimes: Sequence[Any],
    sac_runtimes: Sequence[Any],
) -> tuple[int, float, int, Any]:
    if not (
        len(owners)
        == len(dynamics_runtimes)
        == len(model_runtimes)
        == len(sac_runtimes)
        == 3
    ):
        raise ValueError("Exactly three owner runtimes are required.")

    for runtime, saved in zip(owners, payload["owners"]):
        runtime.env_state = saved["env_state"]
        runtime.features = saved["features"]
        runtime.env_steps_total = int(saved["env_steps_total"])
        runtime.terminal_transitions_total = int(
            saved["terminal_transitions_total"]
        )
        if saved["real_replay"] is None:
            raise ValueError(
                "Resume requires a checkpoint containing real replays."
            )
        runtime.real_replay.load_state_dict(saved["real_replay"])

    for runtime, saved in zip(
        dynamics_runtimes,
        payload["dynamics"],
    ):
        runtime.train_state = _restore_train_state(
            runtime.train_state,
            saved["train_state"],
        )
        runtime.normalizer = saved["normalizer"]
        runtime.updates_total = int(saved["updates_total"])

    for runtime, saved in zip(model_runtimes, payload["models"]):
        runtime.generated_total = int(saved["generated_total"])
        if saved["model_replay"] is None:
            raise ValueError(
                "Resume requires a checkpoint containing model replays."
            )
        runtime.model_replay.load_state_dict(
            saved["model_replay"]
        )

    for runtime, saved in zip(sac_runtimes, payload["sac"]):
        learner = runtime.learner_state
        runtime.learner_state = learner.replace(
            actor_state=_restore_train_state(
                learner.actor_state,
                saved["actor_state"],
            ),
            critic_state=_restore_train_state(
                learner.critic_state,
                saved["critic_state"],
            ),
            alpha_state=_restore_train_state(
                learner.alpha_state,
                saved["alpha_state"],
            ),
            target_critic_params=saved["target_critic_params"],
        )
        runtime.critic_updates_total = int(
            saved["critic_updates_total"]
        )
        runtime.actor_updates_total = int(
            saved["actor_updates_total"]
        )
        runtime.fixed_batch = saved["fixed_batch"]

    return (
        int(payload["round_index"]),
        float(payload["best_return"]),
        int(payload["best_round"]),
        payload["actor_normalizer"],
    )
