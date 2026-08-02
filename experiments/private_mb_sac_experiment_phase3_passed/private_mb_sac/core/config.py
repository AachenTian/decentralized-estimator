"""Configuration loading and validation for the isolated experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access.

    The class remains a normal mapping, so it can be serialized directly to
    YAML/JSON and passed to W&B without custom encoders.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _convert(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({str(k): _convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_convert(v) for v in value)
    return value


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError("The top-level YAML value must be a mapping.")
    config = _convert(raw)
    validate_config(config)
    return config


def clone_config(config: Mapping[str, Any]) -> ConfigNode:
    return _convert(deepcopy(dict(config)))


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(config), sort_keys=False))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _require(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    required = (
        "experiment.num_agents",
        "experiment.num_landmarks",
        "experiment.max_steps",
        "synchronization.rounds",
        "collection.num_envs_per_owner",
        "collection.rollout_length",
        "actor.observation_dim",
        "actor.action_dim",
        "critic.state_dim",
        "dynamics.ensemble_size",
        "dynamics.maximum_gradient_norm",
        "dynamics.normalization_min_samples",
        "actor.normalization_calibration.rounds",
        "model_rollout.batch_size",
        "model_rollout.uncertainty_min_samples",
        "tracking.project",
        "tracking.mode",
    )
    for key in required:
        _require(config, key)

    positive_keys = (
        "experiment.num_agents",
        "experiment.num_landmarks",
        "experiment.max_steps",
        "synchronization.rounds",
        "collection.num_envs_per_owner",
        "collection.rollout_length",
        "replay.real_capacity",
        "replay.model_capacity",
        "actor.observation_dim",
        "actor.action_dim",
        "critic.state_dim",
        "dynamics.ensemble_size",
        "dynamics.maximum_gradient_norm",
        "dynamics.normalization_min_samples",
        "dynamics.evaluation_batch_size",
        "actor.normalization_calibration.num_envs",
        "actor.normalization_calibration.rollout_length",
        "actor.normalization_calibration.rounds",
        "model_rollout.batch_size",
        "model_rollout.uncertainty_min_samples",
        "model_rollout.reward_validation_batch_size",
    )
    invalid = {
        key: _require(config, key)
        for key in positive_keys
        if int(_require(config, key)) < 1
    }
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")

    num_agents = int(_require(config, "experiment.num_agents"))
    action_dim = int(_require(config, "actor.action_dim"))
    declared_joint_action_dim = int(_require(config, "critic.joint_action_dim"))
    if declared_joint_action_dim != num_agents * action_dim:
        raise ValueError(
            "critic.joint_action_dim must equal num_agents * action_dim: "
            f"{declared_joint_action_dim} != {num_agents} * {action_dim}."
        )

    declared_per_round = int(
        _require(config, "collection.real_transitions_per_owner_per_round")
    )
    computed_per_round = int(_require(config, "collection.num_envs_per_owner")) * int(
        _require(config, "collection.rollout_length")
    )
    if declared_per_round != computed_per_round:
        raise ValueError(
            "collection.real_transitions_per_owner_per_round is inconsistent: "
            f"declared {declared_per_round}, computed {computed_per_round}."
        )

    if bool(_require(config, "synchronization.average_actor_parameters")):
        raise ValueError("Actor parameter averaging must remain disabled.")
    if not bool(_require(config, "synchronization.exchange_actor_snapshots_only")):
        raise ValueError("The algorithm requires actor snapshot exchange.")
    if bool(_require(config, "model_rollout.communication_enabled")):
        raise ValueError("Communication must remain disabled in this experiment.")
    if bool(_require(config, "model_rollout.opponent_model_enabled")):
        raise ValueError("Opponent models must remain disabled.")
    if int(_require(config, "dynamics.ensemble_size")) != 5:
        raise ValueError("This experiment is defined with exactly five dynamics members.")
    expected_input_dim = (
        int(_require(config, "critic.state_dim"))
        + int(_require(config, "critic.joint_action_dim"))
    )
    if int(_require(config, "dynamics.input_dim")) != expected_input_dim:
        raise ValueError(
            "dynamics.input_dim must equal state_dim + joint_action_dim: "
            f"{_require(config, 'dynamics.input_dim')} != {expected_input_dim}."
        )
    expected_output_dim = 4 * num_agents
    if int(_require(config, "dynamics.output_dim")) != expected_output_dim:
        raise ValueError(
            "dynamics.output_dim must contain all three local-state deltas: "
            f"{_require(config, 'dynamics.output_dim')} != {expected_output_dim}."
        )
    if str(_require(config, "dynamics.normalization_mode")) != "frozen_after_warmup":
        raise ValueError(
            "Phase two supports only frozen_after_warmup dynamics normalization."
        )


    uncertainty_metric = str(_require(config, "model_rollout.uncertainty_metric"))
    if uncertainty_metric not in {"max_epistemic_variance", "mean_epistemic_variance"}:
        raise ValueError("Unsupported model-rollout uncertainty metric.")
    quantile = float(_require(config, "model_rollout.uncertainty_quantile"))
    if not 0.0 < quantile <= 1.0:
        raise ValueError("model_rollout.uncertainty_quantile must lie in (0, 1].")
    if float(_require(config, "model_rollout.uncertainty_multiplier")) <= 0.0:
        raise ValueError("model_rollout.uncertainty_multiplier must be positive.")
    if float(_require(config, "model_rollout.reward_validation_tolerance")) < 0.0:
        raise ValueError("Reward validation tolerance must be non-negative.")

    project = str(_require(config, "tracking.project"))
    if project.strip() == "AORPO-dynamics model":
        raise ValueError("The old W&B project name must not be reused.")
    mode = str(_require(config, "tracking.mode"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("tracking.mode must be online, offline, or disabled.")


def resolved_run_name(config: Mapping[str, Any]) -> str:
    template = str(_require(config, "tracking.run_name_template"))
    return template.format(
        seed=int(_require(config, "experiment.seed")),
        rounds=int(_require(config, "synchronization.rounds")),
        ensemble_size=int(_require(config, "dynamics.ensemble_size")),
    )


def expected_interaction_counts(
    *,
    rounds: int,
    num_owners: int,
    num_envs_per_owner: int,
    rollout_length: int,
) -> tuple[int, int]:
    """Return `(per_owner_steps, system_steps)` for real collection."""
    values = (rounds, num_owners, num_envs_per_owner, rollout_length)
    if any(int(value) < 1 for value in values):
        raise ValueError("All interaction-count arguments must be positive.")
    per_owner = int(rounds) * int(num_envs_per_owner) * int(rollout_length)
    return per_owner, int(num_owners) * per_owner
