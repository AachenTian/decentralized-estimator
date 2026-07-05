from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def save_independent_dynamics_checkpoint(
    checkpoint_path: str | Path,
    model_states: Sequence[TrainState],
    standardizers: Sequence[Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Save independent local dynamics parameters and standardizers.

    The checkpoint intentionally stores model parameters instead of full
    TrainState objects. Optimizer states are not needed for online inference.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_params": [
            jax.device_get(model_state.params)
            for model_state in model_states
        ],
        "standardizers": jax.device_get(standardizers),
        "metadata": metadata,
    }

    with checkpoint_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved independent dynamics checkpoint to: {checkpoint_path}")


def load_independent_dynamics_checkpoint(
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """
    Load a previously saved independent local dynamics checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    with checkpoint_path.open("rb") as file:
        payload = pickle.load(file)

    required_keys = {
        "model_params",
        "standardizers",
        "metadata",
    }

    missing_keys = required_keys.difference(payload.keys())

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    payload["model_params"] = jax.tree_util.tree_map(
        jnp.asarray,
        payload["model_params"],
    )

    payload["standardizers"] = jax.tree_util.tree_map(
        jnp.asarray,
        payload["standardizers"],
    )

    return payload


def restore_independent_model_states(
    initialized_model_states: Sequence[TrainState],
    saved_model_params: Sequence[Any],
) -> List[TrainState]:
    """
    Restore model parameters into newly initialized TrainState objects.

    The model architecture must match the architecture used during training.
    """
    if len(initialized_model_states) != len(saved_model_params):
        raise ValueError(
            "Model count mismatch between initialized states and checkpoint. "
            f"Expected {len(initialized_model_states)}, "
            f"received {len(saved_model_params)}."
        )

    return [
        model_state.replace(params=model_params)
        for model_state, model_params in zip(
            initialized_model_states,
            saved_model_params,
        )
    ]