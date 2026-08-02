"""Formal 200-round private model-based SAC training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from private_mb_sac.agents.learner import (
    initialize_owner_sac_runtimes,
    live_actor_params,
)
from private_mb_sac.agents.networks import (
    initialize_independent_actor_params,
    make_independent_actor_apply,
)
from private_mb_sac.agents.snapshots import (
    exchange_actor_snapshots,
)
from private_mb_sac.core.config import (
    clone_config,
    dump_config,
    load_config,
    resolved_run_name,
)
from private_mb_sac.dynamics.normalization import (
    save_actor_observation_normalizer,
)
from private_mb_sac.envs.adapter import (
    make_private_mb_sac_adapter,
)
from private_mb_sac.evaluation.animation import save_round_animation
from private_mb_sac.evaluation.evaluator import (
    evaluate_live_actors,
    make_deterministic_evaluator,
)
from private_mb_sac.replay.buffer import create_private_replays
from private_mb_sac.rollout.model_rollout import (
    model_rollout_horizon,
)
from private_mb_sac.rollout.real_collector import (
    make_private_real_collector,
)
from private_mb_sac.tracking.wandb_logger import WandbLogger
from private_mb_sac.training.calibration import (
    calibrate_common_actor_normalizer,
)
from private_mb_sac.training.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_actor_checkpoint,
    save_latest_checkpoint,
)
from private_mb_sac.training.coordinator import (
    collect_private_round,
    initialize_owner_runtimes,
)
from private_mb_sac.training.dynamics_update import (
    initialize_owner_dynamics_runtimes,
    train_private_owner_dynamics,
)
from private_mb_sac.training.model_rollout import (
    generate_private_owner_model_rollout,
    initialize_owner_model_runtimes,
    validate_real_reward_reconstruction,
)
from private_mb_sac.training.sac_training import (
    pairwise_actor_parameter_distance,
    train_private_owner_sac,
    tree_l2_distance,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/private_mb_sac_seed0"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, allow_nan=True)
            + "\n"
        )


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))

    if args.seed is not None:
        config.experiment.seed = int(args.seed)
    if args.rounds is not None:
        config.synchronization.rounds = int(args.rounds)
    if args.eval_interval is not None:
        config.evaluation.interval_rounds = int(
            args.eval_interval
        )
    if args.checkpoint_interval is not None:
        config.checkpoint.interval_rounds = int(
            args.checkpoint_interval
        )
    if args.disable_wandb:
        config.tracking.enabled = False
        config.tracking.mode = "disabled"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_dir / "resolved_config.yaml")

    seed = int(config.experiment.seed)
    num_agents = int(config.experiment.num_agents)
    num_envs = int(config.collection.num_envs_per_owner)
    rollout_length = int(config.collection.rollout_length)

    adapter = make_private_mb_sac_adapter(config)
    actor, initial_actor_params = initialize_independent_actor_params(
        seed=seed,
        num_agents=num_agents,
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        hidden_dims=config.actor.hidden_dims,
    )
    actor_apply = make_independent_actor_apply(
        actor,
        num_agents=num_agents,
    )

    calibration_collector = make_private_real_collector(
        adapter,
        actor_apply,
        num_envs=int(
            config.actor.normalization_calibration.num_envs
        ),
        rollout_length=int(
            config.actor.normalization_calibration.rollout_length
        ),
        jit=not args.no_jit,
    )
    actor_normalizer, calibration = (
        calibrate_common_actor_normalizer(
            adapter=adapter,
            collector=calibration_collector,
            actor_params=initial_actor_params,
            environment_seed=int(config.environment_rng.forced_seed),
            policy_seed=seed + 6_000,
            num_envs=int(
                config.actor.normalization_calibration.num_envs
            ),
            rollout_length=int(
                config.actor.normalization_calibration.rollout_length
            ),
            rounds=int(
                config.actor.normalization_calibration.rounds
            ),
            clip=float(
                config.actor.normalization_calibration.clip
            ),
            minimum_std=float(
                config.actor.normalization_calibration.minimum_std
            ),
        )
    )

    collector = make_private_real_collector(
        adapter,
        actor_apply,
        num_envs=num_envs,
        rollout_length=rollout_length,
        jit=not args.no_jit,
    )
    evaluator = make_deterministic_evaluator(
        adapter,
        actor_apply,
        num_envs=int(config.evaluation.num_envs),
        horizon=int(config.evaluation.horizon),
        jit=not args.no_jit,
    )

    real_replays = create_private_replays(
        num_owners=num_agents,
        capacity=int(config.replay.real_capacity),
        num_agents=num_agents,
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        model_state_dim=int(config.critic.state_dim),
    )
    owners = initialize_owner_runtimes(
        adapter=adapter,
        replays=real_replays,
        environment_seed=int(config.environment_rng.forced_seed),
        num_envs=num_envs,
    )
    _, dynamics_runtimes = initialize_owner_dynamics_runtimes(
        config
    )
    model_runtimes = initialize_owner_model_runtimes(config)
    _, sac_runtimes = initialize_owner_sac_runtimes(
        config,
        actor=actor,
        initial_actor_params=initial_actor_params,
    )

    start_round = 1
    best_return = float("-inf")
    best_round = 0

    if args.resume is not None:
        payload = load_checkpoint(args.resume)
        (
            completed_round,
            best_return,
            best_round,
            actor_normalizer,
        ) = restore_training_state(
            payload,
            owners=owners,
            dynamics_runtimes=dynamics_runtimes,
            model_runtimes=model_runtimes,
            sac_runtimes=sac_runtimes,
        )
        start_round = completed_round + 1
        print(
            f"Resumed from round {completed_round}; "
            f"continuing at round {start_round}."
        )

    save_actor_observation_normalizer(
        actor_normalizer,
        args.output_dir / "actor_normalization.json",
        sample_count=int(calibration["sample_count"]),
    )

    logger = WandbLogger(
        enabled=bool(config.tracking.enabled),
        project=str(config.tracking.project),
        entity=config.tracking.entity,
        group=str(config.tracking.group),
        job_type=str(config.tracking.job_type),
        mode=str(config.tracking.mode),
        name=resolved_run_name(config),
        tags=list(config.tracking.tags) + ["formal-training"],
        config=config,
        output_dir=args.output_dir,
        save_code=bool(config.tracking.save_code),
    )

    metrics_path = args.output_dir / str(
        config.formal_training.metrics_jsonl
    )
    evaluation_path = args.output_dir / str(
        config.formal_training.evaluation_jsonl
    )

    animation_rounds = {
        int(value) for value in config.animation.rounds
    }
    animation_directory = (
        args.output_dir
        / str(config.animation.output_subdirectory)
    )

    def render_requested_animation(
        round_index: int,
        actor_params,
    ):
        if (
            not bool(config.animation.enabled)
            or round_index not in animation_rounds
        ):
            return None

        summary = save_round_animation(
            adapter=adapter,
            actor_apply_fn=actor_apply,
            actor_params=actor_params,
            normalizer=actor_normalizer,
            round_index=round_index,
            output_directory=animation_directory,
            horizon=int(config.animation.horizon),
            fps=int(config.animation.fps),
            dpi=int(config.animation.dpi),
            trail_length=int(config.animation.trail_length),
            save_gif=bool(config.animation.save_gif),
            save_mp4=bool(config.animation.save_mp4),
        )
        print(
            f"Saved seed-40 animation for round {round_index}: "
            f"{summary['outputs']}"
        )
        return summary

    try:
        logger.log_system(
            {
                "calibration_env_steps": float(
                    calibration["environment_steps"]
                ),
                "calibration_samples": float(
                    calibration["sample_count"]
                ),
                "calibration_std_min": float(
                    calibration["std_min"]
                ),
                "calibration_std_max": float(
                    calibration["std_max"]
                ),
            },
            round_index=max(0, start_round - 1),
            commit=True,
        )

        if start_round == 1 and 0 in animation_rounds:
            initial_live_params = live_actor_params(sac_runtimes)
            save_actor_checkpoint(
                checkpoint_dir / "actors_round_0000.pkl",
                round_index=0,
                evaluation_metrics={},
                actor_params=initial_live_params,
                actor_normalizer=actor_normalizer,
                config=config,
            )
            render_requested_animation(0, initial_live_params)

        total_rounds = int(config.synchronization.rounds)
        for round_index in range(
            start_round,
            total_rounds + 1,
        ):
            round_start = perf_counter()
            current_actor_params = live_actor_params(sac_runtimes)
            snapshot_bank = exchange_actor_snapshots(
                current_actor_params,
                synchronization_round=round_index,
            )

            owner_metrics, system_metrics = collect_private_round(
                round_index=round_index,
                live_actor_params=current_actor_params,
                owner_runtimes=owners,
                collector=collector,
                normalizer=actor_normalizer,
                environment_seed=int(config.environment_rng.forced_seed),
                policy_seed=seed + 30_000,
                num_envs=num_envs,
                rollout_length=rollout_length,
                use_random_actions=False,
                snapshot_bank=snapshot_bank,
            )

            for owner_id, dynamics_runtime in enumerate(
                dynamics_runtimes
            ):
                dynamics_metrics = train_private_owner_dynamics(
                    runtime=dynamics_runtime,
                    replay=owners[owner_id].real_replay,
                    config=config,
                    updates=int(config.dynamics.updates_per_round),
                    batch_size=int(config.replay.batch_size),
                    seed=seed + 50_000 + round_index * 100 + owner_id,
                )
                owner_metrics[owner_id].update(dynamics_metrics)

            horizon = model_rollout_horizon(
                round_index,
                horizon_min=int(config.model_rollout.horizon_min),
                horizon_max=int(config.model_rollout.horizon_max),
                schedule_start_round=int(
                    config.model_rollout.schedule_start_round
                ),
                schedule_end_round=int(
                    config.model_rollout.schedule_end_round
                ),
            )
            for owner_id, model_runtime in enumerate(
                model_runtimes
            ):
                if bool(
                    config.formal_training
                    .validate_reward_reconstruction_every_round
                ):
                    reward_metrics = (
                        validate_real_reward_reconstruction(
                            owners[owner_id].real_replay,
                            config,
                            batch_size=int(
                                config.model_rollout
                                .reward_validation_batch_size
                            ),
                            seed=(
                                seed
                                + 70_000
                                + round_index * 100
                                + owner_id
                            ),
                        )
                    )
                    if (
                        reward_metrics[
                            "model/reward_reconstruction_max_abs"
                        ]
                        > float(
                            config.model_rollout
                            .reward_validation_tolerance
                        )
                    ):
                        raise RuntimeError(
                            "Reward reconstruction validation failed "
                            f"for owner {owner_id}: {reward_metrics}"
                        )
                    owner_metrics[owner_id].update(
                        reward_metrics
                    )

                model_metrics, _ = (
                    generate_private_owner_model_rollout(
                        runtime=model_runtime,
                        dynamics_runtime=dynamics_runtimes[
                            owner_id
                        ],
                        real_replay=owners[
                            owner_id
                        ].real_replay,
                        snapshot_bank=snapshot_bank,
                        actor_apply_fn=actor_apply,
                        actor_normalizer=actor_normalizer,
                        config=config,
                        round_index=round_index,
                        batch_size=int(
                            config.model_rollout.batch_size
                        ),
                        horizon=horizon,
                        seed=(
                            seed
                            + 80_000
                            + round_index * 100
                            + owner_id
                        ),
                        jit=not args.no_jit,
                    )
                )
                owner_metrics[owner_id].update(model_metrics)

            for owner_id, sac_runtime in enumerate(sac_runtimes):
                sac_metrics = train_private_owner_sac(
                    runtime=sac_runtime,
                    real_replay=owners[owner_id].real_replay,
                    model_replay=model_runtimes[
                        owner_id
                    ].model_replay,
                    snapshot_bank=snapshot_bank,
                    actor_normalizer=actor_normalizer,
                    dynamics_normalizer=dynamics_runtimes[
                        owner_id
                    ].normalizer,
                    config=config,
                    updates=int(config.critic.updates_per_round),
                    batch_size=int(config.replay.batch_size),
                    seed=seed + 90_000 + round_index * 100 + owner_id,
                    jit=not args.no_jit,
                )
                owner_metrics[owner_id].update(sac_metrics)

            updated_actor_params = live_actor_params(sac_runtimes)
            actor_snapshot_drift = [
                tree_l2_distance(
                    updated_actor_params[owner_id],
                    snapshot_bank.params_by_agent[owner_id],
                )
                for owner_id in range(num_agents)
            ]

            system_metrics.update(
                {
                    "model_rollout_horizon": float(horizon),
                    "synthetic_transitions_round": float(
                        sum(
                            metrics[
                                "model/generated_transitions"
                            ]
                            for metrics in owner_metrics
                        )
                    ),
                    "synthetic_transitions_total": float(
                        sum(
                            runtime.generated_total
                            for runtime in model_runtimes
                        )
                    ),
                    "real_replay_size_total": float(
                        sum(
                            len(runtime.real_replay)
                            for runtime in owners
                        )
                    ),
                    "model_replay_size_total": float(
                        sum(
                            len(runtime.model_replay)
                            for runtime in model_runtimes
                        )
                    ),
                    "dynamics_updates_total": float(
                        sum(
                            runtime.updates_total
                            for runtime in dynamics_runtimes
                        )
                    ),
                    "critic_updates_total": float(
                        sum(
                            runtime.critic_updates_total
                            for runtime in sac_runtimes
                        )
                    ),
                    "actor_updates_total": float(
                        sum(
                            runtime.actor_updates_total
                            for runtime in sac_runtimes
                        )
                    ),
                    "actor_snapshot_drift_mean": float(
                        np.mean(actor_snapshot_drift)
                    ),
                    "actor_snapshot_drift_min": float(
                        np.min(actor_snapshot_drift)
                    ),
                    "actor_pairwise_parameter_distance_mean": float(
                        np.mean(
                            pairwise_actor_parameter_distance(
                                sac_runtimes
                            )
                        )
                    ),
                    "round_seconds_total": float(
                        perf_counter() - round_start
                    ),
                }
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

            evaluation_metrics = None
            if (
                round_index == 1
                or round_index
                % int(config.evaluation.interval_rounds)
                == 0
                or round_index == total_rounds
            ):
                evaluation_metrics = evaluate_live_actors(
                    evaluator,
                    actor_params=updated_actor_params,
                    normalizer=actor_normalizer,
                    environment_seed=int(
                        config.environment_rng.forced_seed
                    ),
                    num_envs=int(config.evaluation.num_envs),
                    horizon=int(config.evaluation.horizon),
                )
                improved = (
                    evaluation_metrics["return_mean"]
                    > best_return
                )
                if improved:
                    best_return = evaluation_metrics["return_mean"]
                    best_round = round_index

                evaluation_metrics.update(
                    {
                        "best_return_so_far": float(best_return),
                        "best_round": float(best_round),
                    }
                )
                logger.log_evaluation(
                    evaluation_metrics,
                    round_index=round_index,
                    commit=True,
                )
                append_jsonl(
                    evaluation_path,
                    {
                        "round": round_index,
                        **evaluation_metrics,
                    },
                )

                if (
                    improved
                    and bool(config.checkpoint.save_best)
                ):
                    save_actor_checkpoint(
                        checkpoint_dir / "best.pkl",
                        round_index=round_index,
                        evaluation_metrics=evaluation_metrics,
                        actor_params=updated_actor_params,
                        actor_normalizer=actor_normalizer,
                        config=config,
                    )

            animation_due = (
                bool(config.animation.enabled)
                and round_index in animation_rounds
            )
            if animation_due:
                save_actor_checkpoint(
                    checkpoint_dir
                    / f"actors_round_{round_index:04d}.pkl",
                    round_index=round_index,
                    evaluation_metrics=(
                        evaluation_metrics or {}
                    ),
                    actor_params=updated_actor_params,
                    actor_normalizer=actor_normalizer,
                    config=config,
                )
                render_requested_animation(
                    round_index,
                    updated_actor_params,
                )

            checkpoint_due = (
                round_index
                % int(config.checkpoint.interval_rounds)
                == 0
                or round_index == total_rounds
            )
            if checkpoint_due:
                if bool(config.checkpoint.save_latest):
                    save_latest_checkpoint(
                        checkpoint_dir / "latest.pkl",
                        round_index=round_index,
                        best_return=best_return,
                        best_round=best_round,
                        actor_normalizer=actor_normalizer,
                        owners=owners,
                        dynamics_runtimes=dynamics_runtimes,
                        model_runtimes=model_runtimes,
                        sac_runtimes=sac_runtimes,
                        config=config,
                        include_replays=bool(
                            config.checkpoint
                            .include_replays_in_latest
                        ),
                    )
                if (
                    bool(
                        config.checkpoint
                        .save_numbered_actor_checkpoints
                    )
                    and not animation_due
                ):
                    save_actor_checkpoint(
                        checkpoint_dir
                        / f"actors_round_{round_index:04d}.pkl",
                        round_index=round_index,
                        evaluation_metrics=(
                            evaluation_metrics or {}
                        ),
                        actor_params=updated_actor_params,
                        actor_normalizer=actor_normalizer,
                        config=config,
                    )

            evaluation_text = (
                ""
                if evaluation_metrics is None
                else (
                    f", eval_return="
                    f"{evaluation_metrics['return_mean']:.4f}, "
                    f"best={best_return:.4f}@{best_round}"
                )
            )
            print(
                f"Round {round_index}/{total_rounds}: "
                f"system_steps="
                f"{int(system_metrics['real_env_steps_total'])}, "
                f"horizon={horizon}, "
                f"model_generated="
                f"{int(system_metrics['synthetic_transitions_round'])}, "
                f"critic_updates="
                f"{int(system_metrics['critic_updates_total'])}, "
                f"actor_updates="
                f"{int(system_metrics['actor_updates_total'])}"
                f"{evaluation_text}"
            )

    finally:
        logger.finish()

    final_summary = {
        "status": "formal_training_completed",
        "rounds": int(config.synchronization.rounds),
        "best_return": float(best_return),
        "best_round": int(best_round),
        "per_owner_env_steps": [
            runtime.env_steps_total for runtime in owners
        ],
        "system_training_env_steps": int(
            sum(runtime.env_steps_total for runtime in owners)
        ),
        "calibration_env_steps": int(
            calibration["environment_steps"]
        ),
        "dynamics_updates_per_owner": [
            runtime.updates_total
            for runtime in dynamics_runtimes
        ],
        "critic_updates_per_owner": [
            runtime.critic_updates_total
            for runtime in sac_runtimes
        ],
        "actor_updates_per_owner": [
            runtime.actor_updates_total
            for runtime in sac_runtimes
        ],
        "synthetic_transitions_per_owner": [
            runtime.generated_total
            for runtime in model_runtimes
        ],
    }
    (
        args.output_dir / "formal_training_summary.json"
    ).write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
