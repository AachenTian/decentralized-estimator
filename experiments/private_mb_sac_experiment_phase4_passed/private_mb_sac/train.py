"""Smoke-test entry point through three independent private SAC learners."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from private_mb_sac.agents.learner import (
    initialize_owner_sac_runtimes,
    live_actor_params as current_live_actor_params,
)
from private_mb_sac.agents.networks import (
    initialize_independent_actor_params,
    make_independent_actor_apply,
)
from private_mb_sac.agents.snapshots import (
    exchange_actor_snapshots,
    trees_allclose,
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
from private_mb_sac.rollout.real_collector import (
    make_private_real_collector,
)
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
        default=Path("outputs/smoke"),
    )

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dynamics-smoke-test", action="store_true")
    parser.add_argument("--model-rollout-smoke-test", action="store_true")
    parser.add_argument("--sac-smoke-test", action="store_true")

    parser.add_argument("--smoke-rounds", type=int, default=1)
    parser.add_argument("--smoke-num-envs", type=int, default=2)
    parser.add_argument("--dynamics-smoke-updates", type=int, default=10)
    parser.add_argument("--dynamics-smoke-batch-size", type=int, default=32)
    parser.add_argument("--model-smoke-batch-size", type=int, default=32)
    parser.add_argument("--model-smoke-horizon", type=int, default=2)
    parser.add_argument("--sac-smoke-updates", type=int, default=10)
    parser.add_argument("--sac-smoke-batch-size", type=int, default=32)
    parser.add_argument("--calibration-smoke-rounds", type=int, default=1)

    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, allow_nan=True)
            + "\n"
        )


def finite_metrics(metrics, keys) -> bool:
    return all(
        key in metrics and math.isfinite(float(metrics[key]))
        for key in keys
    )


def main() -> None:
    args = parse_args()
    mode_flags = (
        args.smoke_test,
        args.dynamics_smoke_test,
        args.model_rollout_smoke_test,
        args.sac_smoke_test,
    )
    if sum(map(int, mode_flags)) != 1:
        raise SystemExit("Select exactly one smoke-test mode.")

    phase2 = (
        args.dynamics_smoke_test
        or args.model_rollout_smoke_test
        or args.sac_smoke_test
    )
    phase3 = args.model_rollout_smoke_test or args.sac_smoke_test
    phase4 = args.sac_smoke_test

    config = clone_config(load_config(args.config))
    config.synchronization.rounds = args.smoke_rounds
    config.collection.num_envs_per_owner = args.smoke_num_envs
    config.collection.real_transitions_per_owner_per_round = (
        args.smoke_num_envs
        * int(config.collection.rollout_length)
    )

    if phase2:
        config.dynamics.normalization_min_samples = min(
            int(config.dynamics.normalization_min_samples),
            args.dynamics_smoke_batch_size,
        )
        config.actor.normalization_calibration.num_envs = (
            args.smoke_num_envs
        )
        config.actor.normalization_calibration.rounds = (
            args.calibration_smoke_rounds
        )

    if phase3:
        config.model_rollout.batch_size = args.model_smoke_batch_size
        config.model_rollout.horizon_min = args.model_smoke_horizon
        config.model_rollout.horizon_max = args.model_smoke_horizon
        config.model_rollout.uncertainty_min_samples = min(
            int(config.model_rollout.uncertainty_min_samples),
            args.dynamics_smoke_batch_size,
        )

    if phase4:
        config.sac.minimum_replay_size = min(
            int(config.sac.minimum_replay_size),
            args.sac_smoke_batch_size,
        )
        config.sac.fixed_batch_size = args.sac_smoke_batch_size

    if args.disable_wandb:
        config.tracking.enabled = False
        config.tracking.mode = "disabled"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_dir / "resolved_config.yaml")

    adapter = make_private_mb_sac_adapter(config)
    actor, initial_actor_params = initialize_independent_actor_params(
        seed=int(config.experiment.seed),
        num_agents=int(config.experiment.num_agents),
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        hidden_dims=config.actor.hidden_dims,
    )
    actor_apply = make_independent_actor_apply(
        actor,
        num_agents=int(config.experiment.num_agents),
    )

    calibration = {
        "sample_count": 0,
        "environment_steps": 0,
        "std_min": 1.0,
        "std_max": 1.0,
    }
    if phase2 and bool(config.actor.normalization_calibration.enabled):
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
                environment_seed=int(config.experiment.seed) + 5_000,
                policy_seed=int(config.experiment.seed) + 6_000,
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
        save_actor_observation_normalizer(
            actor_normalizer,
            args.output_dir / "actor_normalization.json",
            sample_count=int(calibration["sample_count"]),
        )
    else:
        actor_normalizer = identity_actor_normalizer(adapter)

    collector = make_private_real_collector(
        adapter,
        actor_apply,
        num_envs=int(config.collection.num_envs_per_owner),
        rollout_length=int(config.collection.rollout_length),
        jit=not args.no_jit,
    )
    real_replays = create_private_replays(
        num_owners=int(config.experiment.num_agents),
        capacity=int(config.replay.real_capacity),
        num_agents=int(config.experiment.num_agents),
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        model_state_dim=int(config.critic.state_dim),
    )
    owners = initialize_owner_runtimes(
        adapter=adapter,
        replays=real_replays,
        environment_seed=int(config.experiment.seed) + 10_000,
        num_envs=int(config.collection.num_envs_per_owner),
    )

    dynamics_runtimes = None
    model_runtimes = None
    sac_runtimes = None

    if phase2:
        _, dynamics_runtimes = initialize_owner_dynamics_runtimes(
            config
        )
    if phase3:
        model_runtimes = initialize_owner_model_runtimes(config)
    if phase4:
        _, sac_runtimes = initialize_owner_sac_runtimes(
            config,
            actor=actor,
            initial_actor_params=initial_actor_params,
        )

    phase_name = (
        "phase4_independent_sac_smoke"
        if phase4
        else (
            "phase3_model_rollout_smoke"
            if phase3
            else (
                "phase2_dynamics_smoke"
                if phase2
                else "phase1_smoke"
            )
        )
    )

    logger = WandbLogger(
        enabled=bool(config.tracking.enabled),
        project=str(config.tracking.project),
        entity=config.tracking.entity,
        group=str(config.tracking.group),
        job_type=str(config.tracking.job_type),
        mode=str(config.tracking.mode),
        name=resolved_run_name(config) + "_" + phase_name,
        tags=list(config.tracking.tags)
        + [phase_name.replace("_", "-")],
        config=config,
        output_dir=args.output_dir,
        save_code=bool(config.tracking.save_code),
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    last_owner_metrics = []
    last_snapshot_bank = None
    last_snapshot_copies = None
    actor_drift_from_round_snapshot = [0.0, 0.0, 0.0]

    try:
        if phase2:
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
                round_index=0,
                commit=True,
            )

        for round_index in range(
            1,
            int(config.synchronization.rounds) + 1,
        ):
            live_params = (
                current_live_actor_params(sac_runtimes)
                if phase4
                else initial_actor_params
            )
            snapshot_bank = exchange_actor_snapshots(
                live_params,
                synchronization_round=round_index,
            )
            snapshot_copies = tuple(
                snapshot_bank.params_by_agent
            )

            owner_metrics, system_metrics = collect_private_round(
                round_index=round_index,
                live_actor_params=live_params,
                owner_runtimes=owners,
                collector=collector,
                normalizer=actor_normalizer,
                environment_seed=int(config.experiment.seed) + 20_000,
                policy_seed=int(config.experiment.seed) + 30_000,
                num_envs=int(config.collection.num_envs_per_owner),
                rollout_length=int(config.collection.rollout_length),
                use_random_actions=(
                    args.random_actions
                    or (phase2 and not phase4)
                ),
                snapshot_bank=snapshot_bank,
            )

            if phase2:
                for owner_id, dynamics_runtime in enumerate(
                    dynamics_runtimes
                ):
                    dynamics_metrics = train_private_owner_dynamics(
                        runtime=dynamics_runtime,
                        replay=owners[owner_id].real_replay,
                        config=config,
                        updates=args.dynamics_smoke_updates,
                        batch_size=args.dynamics_smoke_batch_size,
                        seed=(
                            int(config.experiment.seed)
                            + 50_000
                            + round_index * 100
                            + owner_id
                        ),
                    )
                    owner_metrics[owner_id].update(dynamics_metrics)

                system_metrics["dynamics_updates_total"] = float(
                    sum(
                        runtime.updates_total
                        for runtime in dynamics_runtimes
                    )
                )

            if phase3:
                horizon = int(config.model_rollout.horizon_max)
                for owner_id, model_runtime in enumerate(
                    model_runtimes
                ):
                    reward_metrics = validate_real_reward_reconstruction(
                        owners[owner_id].real_replay,
                        config,
                        batch_size=int(
                            config.model_rollout.reward_validation_batch_size
                        ),
                        seed=(
                            70_000
                            + round_index * 100
                            + owner_id
                        ),
                    )
                    if (
                        reward_metrics[
                            "model/reward_reconstruction_max_abs"
                        ]
                        > float(
                            config.model_rollout.reward_validation_tolerance
                        )
                    ):
                        raise RuntimeError(
                            f"Owner {owner_id} reward reconstruction "
                            f"mismatch: {reward_metrics}"
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
                            batch_size=args.model_smoke_batch_size,
                            horizon=horizon,
                            seed=(
                                80_000
                                + round_index * 100
                                + owner_id
                            ),
                            jit=not args.no_jit,
                        )
                    )
                    owner_metrics[owner_id].update(reward_metrics)
                    owner_metrics[owner_id].update(model_metrics)

                system_metrics[
                    "synthetic_transitions_round"
                ] = float(
                    sum(
                        metrics["model/generated_transitions"]
                        for metrics in owner_metrics
                    )
                )
                system_metrics[
                    "synthetic_transitions_total"
                ] = float(
                    sum(
                        runtime.generated_total
                        for runtime in model_runtimes
                    )
                )
                system_metrics["model_replay_size_total"] = float(
                    sum(
                        len(runtime.model_replay)
                        for runtime in model_runtimes
                    )
                )

            if phase4:
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
                        updates=args.sac_smoke_updates,
                        batch_size=args.sac_smoke_batch_size,
                        seed=(
                            90_000
                            + round_index * 100
                            + owner_id
                        ),
                        jit=not args.no_jit,
                    )
                    owner_metrics[owner_id].update(sac_metrics)

                new_live_params = current_live_actor_params(
                    sac_runtimes
                )
                actor_drift_from_round_snapshot = [
                    tree_l2_distance(
                        new_live_params[owner_id],
                        snapshot_bank.params_by_agent[owner_id],
                    )
                    for owner_id in range(
                        int(config.experiment.num_agents)
                    )
                ]
                system_metrics[
                    "critic_updates_total"
                ] = float(
                    sum(
                        runtime.critic_updates_total
                        for runtime in sac_runtimes
                    )
                )
                system_metrics[
                    "actor_updates_total"
                ] = float(
                    sum(
                        runtime.actor_updates_total
                        for runtime in sac_runtimes
                    )
                )
                system_metrics[
                    "actor_snapshot_drift_mean"
                ] = float(
                    np.mean(actor_drift_from_round_snapshot)
                )
                system_metrics[
                    "actor_snapshot_drift_min"
                ] = float(
                    np.min(actor_drift_from_round_snapshot)
                )
                system_metrics[
                    "actor_pairwise_parameter_distance_mean"
                ] = float(
                    np.mean(
                        pairwise_actor_parameter_distance(
                            sac_runtimes
                        )
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

            print(f"\nRound {round_index}")
            for owner_id, owner in enumerate(owners):
                line = (
                    f"  owner {owner_id}: "
                    f"env_steps={owner.env_steps_total}, "
                    f"real_replay={len(owner.real_replay)}, "
                    f"terminal={owner.real_replay.stats.terminal_count}"
                )
                if phase2:
                    line += (
                        f", dyn_updates="
                        f"{dynamics_runtimes[owner_id].updates_total}"
                    )
                if phase3:
                    line += (
                        f", model_replay="
                        f"{len(model_runtimes[owner_id].model_replay)}"
                    )
                if phase4:
                    line += (
                        f", critic_updates="
                        f"{sac_runtimes[owner_id].critic_updates_total}, "
                        f"actor_updates="
                        f"{sac_runtimes[owner_id].actor_updates_total}"
                    )
                print(line)
            print(
                "  system: env_steps="
                f"{int(system_metrics['real_env_steps_total'])}"
            )

            last_owner_metrics = owner_metrics
            last_snapshot_bank = snapshot_bank
            last_snapshot_copies = snapshot_copies
    finally:
        logger.finish()

    rounds = int(config.synchronization.rounds)
    per_owner_expected, system_expected = expected_interaction_counts(
        rounds=rounds,
        num_owners=int(config.experiment.num_agents),
        num_envs_per_owner=int(config.collection.num_envs_per_owner),
        rollout_length=int(config.collection.rollout_length),
    )
    per_owner_actual = [
        owner.env_steps_total for owner in owners
    ]
    terminal_counts = [
        owner.real_replay.stats.terminal_count for owner in owners
    ]

    if per_owner_actual != [per_owner_expected] * 3:
        raise RuntimeError(
            "Per-owner environment-step mismatch: "
            f"{per_owner_actual} versus {per_owner_expected}."
        )
    if sum(per_owner_actual) != system_expected:
        raise RuntimeError("System environment-step mismatch.")
    if len({id(owner.real_replay) for owner in owners}) != 3:
        raise RuntimeError("Private real replay sharing was detected.")

    summary = {
        "status": phase_name + "_passed",
        "rounds": rounds,
        "num_envs_per_owner": int(
            config.collection.num_envs_per_owner
        ),
        "rollout_length": int(config.collection.rollout_length),
        "per_owner_env_steps": per_owner_actual,
        "system_env_steps": sum(per_owner_actual),
        "calibration_env_steps": int(
            calibration["environment_steps"]
        ),
        "real_replay_sizes": [
            len(owner.real_replay) for owner in owners
        ],
        "terminal_counts": terminal_counts,
        "real_replay_object_ids_unique": (
            len({id(owner.real_replay) for owner in owners}) == 3
        ),
    }

    if phase2:
        required_dynamics = (
            "dynamics/loss",
            "dynamics/nll",
            "dynamics/state_rmse",
            "dynamics/position_rmse",
            "dynamics/velocity_rmse",
            "dynamics/epistemic_mean",
            "dynamics/aleatoric_mean",
            "dynamics/coverage95",
        )
        if any(
            metrics.get("dynamics/skipped", 1.0) != 0.0
            for metrics in last_owner_metrics
        ):
            raise RuntimeError("A dynamics update was skipped.")
        if not all(
            finite_metrics(metrics, required_dynamics)
            for metrics in last_owner_metrics
        ):
            raise RuntimeError("Non-finite dynamics metrics detected.")

        summary.update(
            {
                "ensemble_size_per_owner": int(
                    config.dynamics.ensemble_size
                ),
                "total_ensemble_members": (
                    int(config.experiment.num_agents)
                    * int(config.dynamics.ensemble_size)
                ),
                "dynamics_updates_per_owner": [
                    runtime.updates_total
                    for runtime in dynamics_runtimes
                ],
                "dynamics_parameter_pairwise_l2": (
                    pairwise_parameter_distance(
                        dynamics_runtimes
                    )
                ),
                "actor_normalization_file": (
                    "actor_normalization.json"
                ),
            }
        )

    if phase3:
        generated = [
            runtime.generated_total
            for runtime in model_runtimes
        ]
        maximum = (
            rounds
            * args.model_smoke_batch_size
            * args.model_smoke_horizon
        )
        if any(
            count < 1 or count > maximum
            for count in generated
        ):
            raise RuntimeError(
                "Invalid synthetic-transition counts: "
                f"{generated}, maximum={maximum}."
            )
        if len(
            {id(runtime.model_replay) for runtime in model_runtimes}
        ) != 3:
            raise RuntimeError("Private model replay sharing detected.")
        if any(
            float(metrics["model/landmark_drift_max"]) != 0.0
            for metrics in last_owner_metrics
        ):
            raise RuntimeError(
                "Landmark movement occurred in model rollout."
            )

        summary.update(
            {
                "model_rollout_batch_size": (
                    args.model_smoke_batch_size
                ),
                "model_rollout_horizon": args.model_smoke_horizon,
                "maximum_synthetic_transitions_per_owner": (
                    maximum
                ),
                "synthetic_transitions_per_owner": generated,
                "synthetic_transitions_system": sum(generated),
                "model_replay_sizes": [
                    len(runtime.model_replay)
                    for runtime in model_runtimes
                ],
                "model_replay_object_ids_unique": (
                    len(
                        {
                            id(runtime.model_replay)
                            for runtime in model_runtimes
                        }
                    )
                    == 3
                ),
            }
        )

    if phase4:
        required_sac = (
            "critic/loss",
            "critic/q1_loss",
            "critic/q2_loss",
            "critic/td_abs",
            "critic/fixed_td_abs",
            "actor/loss",
            "actor/policy_q",
            "actor/entropy",
            "actor/grad_norm",
            "alpha/value",
            "alpha/loss",
        )
        if any(
            metrics.get("sac/skipped", 1.0) != 0.0
            for metrics in last_owner_metrics
        ):
            raise RuntimeError("A private SAC update was skipped.")
        if not all(
            finite_metrics(metrics, required_sac)
            for metrics in last_owner_metrics
        ):
            raise RuntimeError("Non-finite private SAC metrics detected.")

        expected_critic_updates = (
            rounds * args.sac_smoke_updates
        )
        expected_actor_updates = sum(
            1
            for update_index in range(
                1,
                expected_critic_updates + 1,
            )
            if (
                update_index
                > int(config.sac.critic_warmup_updates)
                and (
                    update_index
                    - int(config.sac.critic_warmup_updates)
                )
                % int(config.sac.policy_delay)
                == 0
            )
        )
        actual_critic_updates = [
            runtime.critic_updates_total
            for runtime in sac_runtimes
        ]
        actual_actor_updates = [
            runtime.actor_updates_total
            for runtime in sac_runtimes
        ]
        if actual_critic_updates != [
            expected_critic_updates
        ] * 3:
            raise RuntimeError(
                "Critic update-count mismatch: "
                f"{actual_critic_updates}."
            )
        if actual_actor_updates != [
            expected_actor_updates
        ] * 3:
            raise RuntimeError(
                "Actor update-count mismatch: "
                f"{actual_actor_updates}."
            )
        if not all(
            distance > 0.0
            for distance in actor_drift_from_round_snapshot
        ):
            raise RuntimeError(
                "At least one live actor did not move away from its "
                "round-frozen snapshot."
            )

        snapshot_unchanged = all(
            trees_allclose(
                last_snapshot_bank.params_by_agent[owner_id],
                last_snapshot_copies[owner_id],
            )
            for owner_id in range(3)
        )
        if not snapshot_unchanged:
            raise RuntimeError(
                "A round-frozen actor snapshot was mutated."
            )

        summary.update(
            {
                "sac_updates_requested_per_owner_per_round": (
                    args.sac_smoke_updates
                ),
                "sac_batch_size": args.sac_smoke_batch_size,
                "critic_updates_per_owner": actual_critic_updates,
                "actor_updates_per_owner": actual_actor_updates,
                "actor_drift_from_round_snapshot_l2": (
                    actor_drift_from_round_snapshot
                ),
                "actor_pairwise_parameter_l2": (
                    pairwise_actor_parameter_distance(
                        sac_runtimes
                    )
                ),
                "snapshot_bank_unchanged": snapshot_unchanged,
                "sac_runtime_object_ids_unique": (
                    len({id(runtime) for runtime in sac_runtimes})
                    == 3
                ),
                "final_alpha_per_owner": [
                    float(
                        np.exp(
                            np.asarray(
                                runtime.learner_state.alpha_state.params[
                                    "log_alpha"
                                ]
                            )
                        )
                    )
                    for runtime in sac_runtimes
                ],
                "owner_metrics": last_owner_metrics,
            }
        )

    summary_path = (
        args.output_dir / f"{phase_name}_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"\n{phase_name} passed.")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
