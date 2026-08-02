"""Executable smoke tests for private collection and private dynamics training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from private_mb_sac.agents.networks import (
    initialize_independent_actor_params,
    make_independent_actor_apply,
)
from private_mb_sac.core.config import (
    clone_config,
    dump_config,
    expected_interaction_counts,
    load_config,
    resolved_run_name,
)
from private_mb_sac.dynamics.normalization import (
    save_actor_observation_normalizer,
)
from private_mb_sac.envs.adapter import (
    identity_actor_normalizer,
    make_private_mb_sac_adapter,
)
from private_mb_sac.replay.buffer import create_private_replays
from private_mb_sac.rollout.real_collector import make_private_real_collector
from private_mb_sac.tracking.wandb_logger import WandbLogger
from private_mb_sac.training.calibration import (
    calibrate_common_actor_normalizer,
)
from private_mb_sac.training.coordinator import (
    collect_private_round,
    initialize_owner_runtimes,
)
from private_mb_sac.training.dynamics_update import (
    initialize_owner_dynamics_runtimes,
    pairwise_parameter_distance,
    train_private_owner_dynamics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dynamics-smoke-test", action="store_true")
    parser.add_argument("--smoke-rounds", type=int, default=1)
    parser.add_argument("--smoke-num-envs", type=int, default=2)
    parser.add_argument("--dynamics-smoke-updates", type=int, default=10)
    parser.add_argument("--dynamics-smoke-batch-size", type=int, default=32)
    parser.add_argument("--calibration-smoke-rounds", type=int, default=1)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")


def _finite_dynamics_metrics(metrics: dict[str, float]) -> bool:
    required = (
        "dynamics/loss",
        "dynamics/nll",
        "dynamics/state_rmse",
        "dynamics/position_rmse",
        "dynamics/velocity_rmse",
        "dynamics/epistemic_mean",
        "dynamics/aleatoric_mean",
        "dynamics/coverage95",
    )
    return all(math.isfinite(float(metrics[name])) for name in required)


def main() -> None:
    args = parse_args()
    selected_modes = int(args.smoke_test) + int(args.dynamics_smoke_test)
    if selected_modes != 1:
        raise SystemExit(
            "Select exactly one mode: --smoke-test or --dynamics-smoke-test."
        )

    phase2 = bool(args.dynamics_smoke_test)
    config = clone_config(load_config(args.config))
    config.synchronization.rounds = int(args.smoke_rounds)
    config.collection.num_envs_per_owner = int(args.smoke_num_envs)
    config.collection.real_transitions_per_owner_per_round = (
        int(args.smoke_num_envs) * int(config.collection.rollout_length)
    )
    if phase2:
        config.dynamics.normalization_min_samples = min(
            int(config.dynamics.normalization_min_samples),
            int(args.dynamics_smoke_batch_size),
        )
        config.actor.normalization_calibration.num_envs = int(
            args.smoke_num_envs
        )
        config.actor.normalization_calibration.rounds = int(
            args.calibration_smoke_rounds
        )
    if args.disable_wandb:
        config.tracking.enabled = False
        config.tracking.mode = "disabled"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_dir / "resolved_config.yaml")

    adapter = make_private_mb_sac_adapter(config)
    actor, live_actor_params = initialize_independent_actor_params(
        seed=int(config.experiment.seed),
        num_agents=int(config.experiment.num_agents),
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        hidden_dims=config.actor.hidden_dims,
    )
    actor_apply_fn = make_independent_actor_apply(
        actor,
        num_agents=int(config.experiment.num_agents),
    )

    calibration_metrics: dict[str, Any] = {
        "sample_count": 0,
        "environment_steps": 0,
    }
    if phase2 and bool(config.actor.normalization_calibration.enabled):
        calibration_collector = make_private_real_collector(
            adapter,
            actor_apply_fn,
            num_envs=int(config.actor.normalization_calibration.num_envs),
            rollout_length=int(
                config.actor.normalization_calibration.rollout_length
            ),
            jit=not args.no_jit,
        )
        normalizer, calibration_metrics = calibrate_common_actor_normalizer(
            adapter=adapter,
            collector=calibration_collector,
            actor_params=live_actor_params,
            environment_seed=int(config.experiment.seed) + 5_000,
            policy_seed=int(config.experiment.seed) + 6_000,
            num_envs=int(config.actor.normalization_calibration.num_envs),
            rollout_length=int(
                config.actor.normalization_calibration.rollout_length
            ),
            rounds=int(config.actor.normalization_calibration.rounds),
            clip=float(config.actor.normalization_calibration.clip),
            minimum_std=float(
                config.actor.normalization_calibration.minimum_std
            ),
        )
        save_actor_observation_normalizer(
            normalizer,
            args.output_dir / "actor_normalization.json",
            sample_count=int(calibration_metrics["sample_count"]),
        )
    else:
        normalizer = identity_actor_normalizer(adapter)

    collector = make_private_real_collector(
        adapter,
        actor_apply_fn,
        num_envs=int(config.collection.num_envs_per_owner),
        rollout_length=int(config.collection.rollout_length),
        jit=not args.no_jit,
    )
    replays = create_private_replays(
        num_owners=int(config.experiment.num_agents),
        capacity=int(config.replay.real_capacity),
        num_agents=int(config.experiment.num_agents),
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        model_state_dim=int(config.critic.state_dim),
    )
    owners = initialize_owner_runtimes(
        adapter=adapter,
        replays=replays,
        environment_seed=int(config.experiment.seed) + 10_000,
        num_envs=int(config.collection.num_envs_per_owner),
    )

    dynamics_runtimes = None
    if phase2:
        _dynamics_model, dynamics_runtimes = (
            initialize_owner_dynamics_runtimes(config)
        )

    phase_name = "phase2_dynamics_smoke" if phase2 else "phase1_smoke"
    logger = WandbLogger(
        enabled=bool(config.tracking.enabled),
        project=str(config.tracking.project),
        entity=config.tracking.entity,
        group=str(config.tracking.group),
        job_type=str(config.tracking.job_type),
        mode=str(config.tracking.mode),
        name=resolved_run_name(config) + f"_{phase_name}",
        tags=list(config.tracking.tags) + [phase_name.replace("_", "-")],
        config=config,
        output_dir=args.output_dir,
        save_code=bool(config.tracking.save_code),
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    last_owner_metrics: list[dict[str, float]] = []
    try:
        if phase2:
            logger.log_system(
                {
                    "calibration_env_steps": float(
                        calibration_metrics["environment_steps"]
                    ),
                    "calibration_samples": float(
                        calibration_metrics["sample_count"]
                    ),
                    "calibration_std_min": float(
                        calibration_metrics["std_min"]
                    ),
                    "calibration_std_max": float(
                        calibration_metrics["std_max"]
                    ),
                },
                round_index=0,
                commit=True,
            )

        for round_index in range(1, int(config.synchronization.rounds) + 1):
            owner_metrics, system_metrics = collect_private_round(
                round_index=round_index,
                live_actor_params=live_actor_params,
                owner_runtimes=owners,
                collector=collector,
                normalizer=normalizer,
                environment_seed=int(config.experiment.seed) + 20_000,
                policy_seed=int(config.experiment.seed) + 30_000,
                num_envs=int(config.collection.num_envs_per_owner),
                rollout_length=int(config.collection.rollout_length),
                use_random_actions=(bool(args.random_actions) or phase2),
            )

            if phase2:
                assert dynamics_runtimes is not None
                for owner_id, dynamics_runtime in enumerate(dynamics_runtimes):
                    dynamics_metrics = train_private_owner_dynamics(
                        runtime=dynamics_runtime,
                        replay=owners[owner_id].real_replay,
                        config=config,
                        updates=int(args.dynamics_smoke_updates),
                        batch_size=int(args.dynamics_smoke_batch_size),
                        seed=(
                            int(config.experiment.seed)
                            + 50_000
                            + round_index * 100
                            + owner_id
                        ),
                    )
                    owner_metrics[owner_id].update(dynamics_metrics)
                system_metrics["dynamics_updates_total"] = float(
                    sum(runtime.updates_total for runtime in dynamics_runtimes)
                )
                system_metrics["dynamics_seconds_total"] = float(
                    sum(
                        metrics.get("timing/dynamics_seconds", 0.0)
                        for metrics in owner_metrics
                    )
                )

            for owner_id, metrics in enumerate(owner_metrics):
                logger.log_owner(
                    owner_id,
                    metrics,
                    round_index=round_index,
                    commit=False,
                )
            logger.log_system(
                system_metrics,
                round_index=round_index,
                commit=True,
            )
            append_jsonl(
                metrics_path,
                {
                    "round": round_index,
                    "owners": owner_metrics,
                    "system": system_metrics,
                },
            )
            last_owner_metrics = owner_metrics

            print(f"\nRound {round_index}")
            for owner_id, owner in enumerate(owners):
                line = (
                    f"  owner {owner.owner_id}: "
                    f"env_steps={owner.env_steps_total}, "
                    f"replay={len(owner.real_replay)}, "
                    f"terminal={owner.real_replay.stats.terminal_count}"
                )
                if phase2:
                    line += (
                        f", dyn_updates="
                        f"{int(owner_metrics[owner_id]['dynamics/updates_total'])}, "
                        f"state_rmse="
                        f"{owner_metrics[owner_id]['dynamics/state_rmse']:.6f}"
                    )
                print(line)
            print(
                "  system: "
                f"env_steps={int(system_metrics['real_env_steps_total'])}"
            )
    finally:
        logger.finish()

    expected_per_owner, expected_system = expected_interaction_counts(
        rounds=int(config.synchronization.rounds),
        num_owners=int(config.experiment.num_agents),
        num_envs_per_owner=int(config.collection.num_envs_per_owner),
        rollout_length=int(config.collection.rollout_length),
    )
    actual_per_owner = tuple(owner.env_steps_total for owner in owners)
    actual_system = sum(actual_per_owner)
    if any(value != expected_per_owner for value in actual_per_owner):
        raise RuntimeError(
            f"Per-owner interaction mismatch: actual={actual_per_owner}, "
            f"expected={expected_per_owner}."
        )
    if actual_system != expected_system:
        raise RuntimeError(
            f"System interaction mismatch: {actual_system} != {expected_system}."
        )

    terminal_counts = [
        owner.real_replay.stats.terminal_count for owner in owners
    ]
    expected_terminal_count = (
        int(config.synchronization.rounds)
        * int(config.collection.num_envs_per_owner)
        if int(config.collection.rollout_length)
        == int(config.experiment.max_steps)
        else None
    )
    if expected_terminal_count is not None and any(
        count != expected_terminal_count for count in terminal_counts
    ):
        raise RuntimeError(
            "Terminal accounting mismatch. For an episode-aligned rollout, "
            f"expected {expected_terminal_count} terminal transitions per "
            f"owner, got {terminal_counts}."
        )

    summary: dict[str, Any] = {
        "status": f"{phase_name}_passed",
        "rounds": int(config.synchronization.rounds),
        "num_envs_per_owner": int(config.collection.num_envs_per_owner),
        "rollout_length": int(config.collection.rollout_length),
        "per_owner_env_steps": list(actual_per_owner),
        "system_env_steps": actual_system,
        "calibration_env_steps": int(calibration_metrics["environment_steps"]),
        "replay_sizes": [len(owner.real_replay) for owner in owners],
        "terminal_counts": terminal_counts,
        "expected_terminal_count_per_owner": expected_terminal_count,
        "replay_object_ids_unique": len({id(owner.real_replay) for owner in owners})
        == len(owners),
    }

    if phase2:
        assert dynamics_runtimes is not None
        if any(metrics.get("dynamics/skipped", 1.0) != 0.0 for metrics in last_owner_metrics):
            raise RuntimeError("At least one private dynamics update was skipped.")
        if not all(_finite_dynamics_metrics(metrics) for metrics in last_owner_metrics):
            raise RuntimeError(
                "At least one owner produced non-finite core dynamics metrics."
            )
        parameter_distances = pairwise_parameter_distance(dynamics_runtimes)
        if any(distance <= 0.0 for distance in parameter_distances):
            raise RuntimeError(
                "Independent dynamics owners unexpectedly share identical parameters."
            )
        summary.update(
            {
                "ensemble_size_per_owner": int(config.dynamics.ensemble_size),
                "total_ensemble_members": int(config.dynamics.ensemble_size)
                * int(config.experiment.num_agents),
                "dynamics_updates_per_owner": [
                    runtime.updates_total for runtime in dynamics_runtimes
                ],
                "dynamics_parameter_pairwise_l2": parameter_distances,
                "owner_dynamics_metrics": last_owner_metrics,
                "actor_normalization_file": "actor_normalization.json",
            }
        )

    summary_path = args.output_dir / f"{phase_name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"\n{phase_name} passed.")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
