# evaluate_event_triggered_estimator.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import hydra
import jax
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from flax.training.train_state import TrainState

from aorpo.agents.independent_dynamics import (
    init_independent_transition_models,
)
from aorpo.estimator.online_oracle import (
    run_online_oracle_global_belief_evaluation,
)
from aorpo.utils.checkpoints import (
    load_independent_dynamics_checkpoint,
    restore_independent_model_states,
)


def validate_parameter_tree_shapes(
    initialized_params: Any,
    saved_params: Any,
    agent_id: int,
) -> None:
    """
    Validate that a checkpoint parameter tree matches the initialized model.

    This catches architecture or configuration mismatches before online
    evaluation starts.
    """
    initialized_leaves, initialized_tree = jax.tree_util.tree_flatten(
        initialized_params
    )

    saved_leaves, saved_tree = jax.tree_util.tree_flatten(
        saved_params
    )

    if initialized_tree != saved_tree:
        raise ValueError(
            "Checkpoint parameter tree structure does not match the "
            f"initialized model for agent {agent_id}. "
            "Ensure that the current model architecture configuration "
            "matches the configuration used during training."
        )

    if len(initialized_leaves) != len(saved_leaves):
        raise ValueError(
            "Checkpoint parameter leaf count does not match the "
            f"initialized model for agent {agent_id}."
        )

    for leaf_index, (initialized_leaf, saved_leaf) in enumerate(
        zip(initialized_leaves, saved_leaves)
    ):
        initialized_shape = getattr(initialized_leaf, "shape", None)
        saved_shape = getattr(saved_leaf, "shape", None)

        if initialized_shape != saved_shape:
            raise ValueError(
                "Checkpoint parameter shape mismatch for "
                f"agent {agent_id}, leaf {leaf_index}: "
                f"expected {initialized_shape}, got {saved_shape}. "
                "Ensure that the ensemble architecture and dimensions "
                "match the training configuration."
            )


def validate_checkpoint_metadata(
    metadata: dict[str, Any],
    cfg: DictConfig,
) -> None:
    """
    Validate basic checkpoint metadata against the current configuration.
    """
    expected_num_agents = cfg.train.num_opponents + 1
    expected_action_dim = cfg.env.act_dim
    expected_local_state_dim = 4

    saved_num_agents = metadata.get("num_agents")
    saved_action_dim = metadata.get("action_dim")
    saved_local_state_dim = metadata.get("local_state_dim")

    if saved_num_agents != expected_num_agents:
        raise ValueError(
            "Checkpoint num_agents does not match the current config. "
            f"Checkpoint={saved_num_agents}, "
            f"config={expected_num_agents}."
        )

    if saved_action_dim != expected_action_dim:
        raise ValueError(
            "Checkpoint action_dim does not match the current config. "
            f"Checkpoint={saved_action_dim}, "
            f"config={expected_action_dim}."
        )

    if saved_local_state_dim != expected_local_state_dim:
        raise ValueError(
            "Checkpoint local_state_dim does not match the evaluator. "
            f"Checkpoint={saved_local_state_dim}, "
            f"evaluator={expected_local_state_dim}."
        )


def restore_dynamics_models(
    cfg: DictConfig,
    checkpoint_path: Path,
) -> tuple[Sequence[TrainState], Sequence[Any], dict[str, Any]]:
    """
    Rebuild local dynamics model states and restore a saved checkpoint.
    """
    checkpoint = load_independent_dynamics_checkpoint(
        checkpoint_path
    )

    metadata = checkpoint["metadata"]
    validate_checkpoint_metadata(
        metadata=metadata,
        cfg=cfg,
    )

    num_agents = cfg.train.num_opponents + 1
    action_dim = cfg.env.act_dim

    initialization_key = jax.random.PRNGKey(
        int(cfg.seed) + 98765
    )

    _, initialized_model_states = (
        init_independent_transition_models(
            rng=initialization_key,
            num_agents=num_agents,
            act_dim=action_dim,
            cfg=cfg,
        )
    )

    saved_model_params = checkpoint["model_params"]

    if len(saved_model_params) != num_agents:
        raise ValueError(
            "Checkpoint model count does not match the current config. "
            f"Checkpoint={len(saved_model_params)}, "
            f"config={num_agents}."
        )

    for agent_id, (model_state, saved_params) in enumerate(
        zip(initialized_model_states, saved_model_params)
    ):
        validate_parameter_tree_shapes(
            initialized_params=model_state.params,
            saved_params=saved_params,
            agent_id=agent_id,
        )

    restored_model_states = restore_independent_model_states(
        initialized_model_states=initialized_model_states,
        saved_model_params=saved_model_params,
    )

    standardizers = checkpoint["standardizers"]

    if len(standardizers) != num_agents:
        raise ValueError(
            "Checkpoint standardizer count does not match the current "
            f"config. Checkpoint={len(standardizers)}, "
            f"config={num_agents}."
        )

    return restored_model_states, standardizers, metadata


@hydra.main(
    config_path="aorpo/configs",
    config_name="train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Load trained local dynamics models and evaluate online communication modes.
    """
    checkpoint_path = Path(
        to_absolute_path(
            str(cfg.independent_dynamics.checkpoint_path)
        )
    )

    print(
        "\n===== Standalone Event-Triggered Estimator Evaluation ====="
    )

    print(f"Checkpoint path: {checkpoint_path}")
    print(
        "Communication mode: "
        f"{cfg.online_estimator.communication_mode}"
    )

    print(
        "Number of online episodes: "
        f"{cfg.online_estimator.num_episodes}"
    )

    model_states, standardizers, metadata = restore_dynamics_models(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
    )

    print("Checkpoint restored successfully.")
    print(
        f"Checkpoint agents: {metadata['num_agents']} | "
        f"action_dim: {metadata['action_dim']} | "
        f"local_state_dim: {metadata['local_state_dim']}"
    )

    run_online_oracle_global_belief_evaluation(
        model_states=model_states,
        standardizers=standardizers,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()