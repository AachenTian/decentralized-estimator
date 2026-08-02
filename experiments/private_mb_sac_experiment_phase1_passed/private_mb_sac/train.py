"""Phase-one executable: three private real collectors plus offline W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import yaml

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
from private_mb_sac.envs.adapter import (
    identity_actor_normalizer,
    make_private_mb_sac_adapter,
)
from private_mb_sac.replay.buffer import create_private_replays
from private_mb_sac.rollout.real_collector import make_private_real_collector
from private_mb_sac.tracking.wandb_logger import WandbLogger
from private_mb_sac.training.coordinator import (
    collect_private_round,
    initialize_owner_runtimes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_smoke"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-rounds", type=int, default=1)
    parser.add_argument("--smoke-num-envs", type=int, default=2)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    config = clone_config(load_config(args.config))

    if not args.smoke_test:
        raise SystemExit(
            "Phase one currently implements collection/replay/snapshot validation only. "
            "Run with --smoke-test. Dynamics and SAC updates are the next phases."
        )

    config.synchronization.rounds = int(args.smoke_rounds)
    config.collection.num_envs_per_owner = int(args.smoke_num_envs)
    config.collection.real_transitions_per_owner_per_round = (
        int(args.smoke_num_envs) * int(config.collection.rollout_length)
    )
    if args.disable_wandb:
        config.tracking.enabled = False
        config.tracking.mode = "disabled"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_dir / "resolved_config.yaml")

    adapter = make_private_mb_sac_adapter(config)
    normalizer = identity_actor_normalizer(adapter)
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

    logger = WandbLogger(
        enabled=bool(config.tracking.enabled),
        project=str(config.tracking.project),
        entity=config.tracking.entity,
        group=str(config.tracking.group),
        job_type=str(config.tracking.job_type),
        mode=str(config.tracking.mode),
        name=resolved_run_name(config) + "_phase1_smoke",
        tags=list(config.tracking.tags) + ["phase1-smoke"],
        config=config,
        output_dir=args.output_dir,
        save_code=bool(config.tracking.save_code),
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    try:
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
                use_random_actions=bool(args.random_actions),
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
            for owner in owners:
                print(
                    f"  owner {owner.owner_id}: "
                    f"env_steps={owner.env_steps_total}, "
                    f"replay={len(owner.real_replay)}, "
                    f"terminal={owner.real_replay.stats.terminal_count}"
                )
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
    if expected_terminal_count is not None:
        if any(
            count != expected_terminal_count
            for count in terminal_counts
        ):
            raise RuntimeError(
                "Terminal accounting mismatch. For an episode-aligned "
                f"rollout, expected {expected_terminal_count} terminal "
                f"transitions per owner, got {terminal_counts}."
            )

    summary = {
        "status": "phase1_smoke_passed",
        "rounds": int(config.synchronization.rounds),
        "num_envs_per_owner": int(config.collection.num_envs_per_owner),
        "rollout_length": int(config.collection.rollout_length),
        "per_owner_env_steps": list(actual_per_owner),
        "system_env_steps": actual_system,
        "replay_sizes": [len(owner.real_replay) for owner in owners],
        "terminal_counts": terminal_counts,
        "expected_terminal_count_per_owner": expected_terminal_count,
        "replay_object_ids_unique": len({id(owner.real_replay) for owner in owners})
        == len(owners),
    }
    (args.output_dir / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\nPhase-one smoke test passed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
