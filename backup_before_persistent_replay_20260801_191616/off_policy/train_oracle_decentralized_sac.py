from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from sender_marl.envs import make_env_adapter
from sender_marl.off_policy.evaluation import make_independent_sac_evaluator
from sender_marl.off_policy.independent import (
    create_independent_sac_states,
    independent_actor_params,
    make_independent_sac_actor_apply,
)
from sender_marl.off_policy.networks import SACActor, TwinQNetwork
from sender_marl.off_policy.normalization import (
    NormalizationStats,
    RunningMoments,
)
from sender_marl.off_policy.oracle_decentralized_collector import (
    make_fixed_environment_key_schedule,
    make_fixed_initial_reset_keys,
    make_fixed_rng_oracle_rollout_collector,
    make_owner_policy_key_schedule,
)
from sender_marl.off_policy.replay_buffer import JointReplayBuffer
from sender_marl.off_policy.sac_state import SACConfig
from sender_marl.off_policy.sac_update import (
    make_fixed_batch_critic_evaluator,
    make_sac_update,
)


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _device_float(value: Any) -> float:
    return float(jax.device_get(value))


def _tree_copy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: leaf, tree)


def _replace_focal_snapshot(
    synchronized_actor_snapshots: Sequence[Any],
    *,
    focal_agent_id: int,
    current_focal_actor_params: Any,
) -> tuple[Any, ...]:
    """Keep opponents frozen while allowing the focal actor to evolve."""

    snapshots = list(synchronized_actor_snapshots)
    snapshots[focal_agent_id] = current_focal_actor_params
    return tuple(snapshots)


def _new_private_buffer(args, spec) -> JointReplayBuffer:
    return JointReplayBuffer(
        capacity=args.buffer_capacity,
        num_agents=spec.num_agents,
        observation_dim=spec.actor_obs_dim,
        action_dim=spec.policy_action_dim,
        model_state_dim=spec.model_state_dim,
    )


def _append_real_rollout(rollout, replay_buffer: JointReplayBuffer) -> None:
    batch = jax.device_get(rollout.batch)
    time_steps, num_envs, _num_agents = batch.rewards.shape
    flat_count = time_steps * num_envs
    replay_buffer.add_batch(
        observations=np.asarray(batch.observations).reshape(
            flat_count,
            batch.observations.shape[-2],
            batch.observations.shape[-1],
        ),
        actions=np.asarray(batch.actions).reshape(
            flat_count,
            batch.actions.shape[-2],
            batch.actions.shape[-1],
        ),
        rewards=np.asarray(batch.rewards[..., 0]).reshape(flat_count),
        next_observations=np.asarray(batch.next_observations).reshape(
            flat_count,
            batch.next_observations.shape[-2],
            batch.next_observations.shape[-1],
        ),
        dones=np.asarray(batch.episode_dones).reshape(flat_count),
        model_states=np.asarray(batch.model_states).reshape(flat_count, -1),
        next_model_states=np.asarray(batch.next_model_states).reshape(
            flat_count, -1
        ),
    )


def _calibrate_normalizers(
    *,
    adapter,
    raw_actor_apply,
    actor_params,
    initial_env_state,
    initial_features,
    environment_seed: int,
    policy_seed: int,
    num_owners: int,
    num_envs: int,
    rollout_length: int,
    normalization_rounds: int,
    normalization_clip: float,
    normalization_epsilon: float,
    normalization_min_std: float,
) -> tuple[NormalizationStats, NormalizationStats, NormalizationStats]:
    """Calibrate frozen obs/state/action statistics with fixed random rollouts.

    Calibration uses a dedicated deterministic RNG namespace and separate
    environment carries, so it never changes the training initial states or the
    training environment RNG schedule. Actor observations are pooled across
    homogeneous agents and rollout owners. Joint-state statistics remain
    feature-wise. Action statistics are pooled across homogeneous agents and
    retained per action coordinate.
    """

    if normalization_rounds < 1:
        raise ValueError("normalization-rounds must be positive.")

    obs_moments = RunningMoments((adapter.spec.actor_obs_dim,))
    state_moments = RunningMoments((adapter.spec.model_state_dim,))
    action_moments = RunningMoments((adapter.spec.policy_action_dim,))

    calibration_collector = make_fixed_rng_oracle_rollout_collector(
        adapter,
        raw_actor_apply,
        num_envs=num_envs,
        rollout_length=rollout_length,
        jit=True,
    )
    owner_env_states = tuple(
        _tree_copy(initial_env_state) for _ in range(num_owners)
    )
    owner_features = tuple(
        _tree_copy(initial_features) for _ in range(num_owners)
    )

    calibration_round_offset = 1_000_000
    for calibration_round in range(normalization_rounds):
        schedule_round = calibration_round_offset + calibration_round
        environment_schedule = make_fixed_environment_key_schedule(
            environment_seed,
            synchronization_round=schedule_round,
            rollout_length=rollout_length,
            num_envs=num_envs,
        )
        next_env_states = []
        next_features = []
        for owner_agent_id in range(num_owners):
            policy_schedule = make_owner_policy_key_schedule(
                policy_seed,
                synchronization_round=schedule_round,
                owner_agent_id=owner_agent_id,
                rollout_length=rollout_length,
            )
            rollout = calibration_collector(
                actor_params,
                owner_env_states[owner_agent_id],
                owner_features[owner_agent_id],
                environment_schedule,
                policy_schedule,
                jnp.asarray(True),
            )
            _block_until_ready(rollout.batch)
            batch = jax.device_get(rollout.batch)

            observations = np.asarray(batch.observations, dtype=np.float32)
            next_observations = np.asarray(
                batch.next_observations, dtype=np.float32
            )
            model_states = np.asarray(batch.model_states, dtype=np.float32)
            next_model_states = np.asarray(
                batch.next_model_states, dtype=np.float32
            )
            actions = np.asarray(batch.actions, dtype=np.float32)

            obs_moments.update(observations.reshape(-1, observations.shape[-1]))
            obs_moments.update(
                next_observations.reshape(-1, next_observations.shape[-1])
            )
            state_moments.update(model_states.reshape(-1, model_states.shape[-1]))
            state_moments.update(
                next_model_states.reshape(-1, next_model_states.shape[-1])
            )
            action_moments.update(actions.reshape(-1, actions.shape[-1]))

            next_env_states.append(rollout.env_state)
            next_features.append(rollout.features)
        owner_env_states = tuple(next_env_states)
        owner_features = tuple(next_features)

    finalize_kwargs = {
        "clip": normalization_clip,
        "epsilon": normalization_epsilon,
        "minimum_std": normalization_min_std,
    }
    return (
        obs_moments.finalize(**finalize_kwargs),
        state_moments.finalize(**finalize_kwargs),
        action_moments.finalize(**finalize_kwargs),
    )


def _should_update_actor(
    critic_gradient_step: int,
    *,
    critic_warmup_steps: int,
    policy_delay: int,
) -> bool:
    if critic_gradient_step <= critic_warmup_steps:
        return False
    return (
        critic_gradient_step - critic_warmup_steps
    ) % policy_delay == 0


def _save_checkpoint(
    output_dir: Path,
    synchronization_round: int,
    learner_states,
    args: argparse.Namespace,
    sac_config: SACConfig,
    observation_stats: NormalizationStats,
    state_stats: NormalizationStats,
    action_stats: NormalizationStats,
    *,
    filename: str,
    critic_gradient_steps_total: int,
    actor_update_cycles_total: int,
    evaluation_metrics: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "oracle_environment_decentralized_rollout_sac",
        "parameter_sharing": False,
        "trajectory_sharing": False,
        "private_replay_per_agent": True,
        "actor_snapshot_exchange": True,
        "snapshot_frequency": "once_per_synchronization_round",
        "rollout_transition_source": "real_environment_oracle_dynamics",
        "environment_rng_protocol": (
            "fixed_common_random_numbers_across_rollout_owners"
        ),
        "environment_seed": args.environment_seed,
        "policy_seed": args.policy_seed,
        "evaluation_seed": args.evaluation_seed,
        "synchronization_round": synchronization_round,
        "critic_gradient_steps_total": critic_gradient_steps_total,
        "actor_update_cycles_total": actor_update_cycles_total,
        "actor_params": jax.device_get(
            independent_actor_params(learner_states)
        ),
        "critic_params": jax.device_get(
            tuple(state.critic_state.params for state in learner_states)
        ),
        "target_critic_params": jax.device_get(
            tuple(state.target_critic_params for state in learner_states)
        ),
        "log_alpha_params": jax.device_get(
            tuple(state.alpha_state.params for state in learner_states)
        ),
        "args": vars(args),
        "sac_config": asdict(sac_config),
        "normalization": {
            "observation": observation_stats.as_serializable(),
            "state": state_stats.as_serializable(),
            "action": action_stats.as_serializable(),
            "statistics_frozen": True,
        },
        "evaluation_metrics": evaluation_metrics,
    }
    with (output_dir / filename).open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate decentralized actor/critic training by replacing learned "
            "dynamics with private real-environment rollouts. Each agent owns "
            "its replay and learner, exchanges frozen actor snapshots once per "
            "round, and uses a fixed common environment RNG schedule."
        )
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--total-sync-rounds", type=int, default=320)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=20260729,
        help=(
            "Fixed stateless environment RNG seed. The same per-round step and "
            "reset keys are used by all three private rollout owners."
        ),
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=314159,
        help="Fixed owner-specific policy/action sampling seed.",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=271828,
        help="The same deterministic evaluation environments are reused.",
    )

    parser.add_argument("--buffer-capacity", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-batch-size", type=int, default=256)
    parser.add_argument(
        "--random-collection-rounds",
        type=int,
        default=10,
        help="Initial rounds using deterministic owner-specific random actions.",
    )
    parser.add_argument(
        "--normalization-rounds",
        type=int,
        default=10,
        help=(
            "Dedicated fixed-random calibration rounds used to estimate and "
            "freeze observation, state, and action normalization statistics."
        ),
    )
    parser.add_argument(
        "--normalization-clip", type=float, default=10.0
    )
    parser.add_argument(
        "--normalization-epsilon", type=float, default=1e-6
    )
    parser.add_argument(
        "--normalization-min-std", type=float, default=1e-3
    )

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256]
    )
    parser.add_argument("--policy-delay", type=int, default=4)
    parser.add_argument("--critic-warmup-steps", type=int, default=1000)

    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle_decentralized_sac_seed0"),
    )
    args = parser.parse_args()

    positive = {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "total_sync_rounds": args.total_sync_rounds,
        "gradient_steps": args.gradient_steps,
        "buffer_capacity": args.buffer_capacity,
        "batch_size": args.batch_size,
        "fixed_batch_size": args.fixed_batch_size,
        "policy_delay": args.policy_delay,
        "eval_envs": args.eval_envs,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}.")
    if args.random_collection_rounds < 0:
        raise ValueError("random-collection-rounds must be non-negative.")
    if args.normalization_rounds < 1:
        raise ValueError("normalization-rounds must be positive.")
    if args.normalization_clip <= 0.0:
        raise ValueError("normalization-clip must be positive.")
    if args.normalization_epsilon <= 0.0:
        raise ValueError("normalization-epsilon must be positive.")
    if args.normalization_min_std <= 0.0:
        raise ValueError("normalization-min-std must be positive.")
    if args.critic_warmup_steps < 0:
        raise ValueError("critic-warmup-steps must be non-negative.")

    private_transitions_per_round = args.num_envs * args.rollout_length
    if args.buffer_capacity < private_transitions_per_round:
        raise ValueError(
            "buffer-capacity must hold one full private rollout round: "
            f"need at least {private_transitions_per_round}."
        )
    if private_transitions_per_round < max(
        args.batch_size, args.fixed_batch_size
    ):
        raise ValueError(
            "One private rollout round must contain at least batch-size and "
            "fixed-batch-size transitions."
        )

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        max_steps=args.max_steps,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )
    spec = adapter.spec
    target_entropy = (
        -float(spec.policy_action_dim)
        if args.target_entropy is None
        else float(args.target_entropy)
    )
    sac_config = SACConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        alpha_learning_rate=args.alpha_lr,
        initial_alpha=args.initial_alpha,
        target_entropy=target_entropy,
        max_gradient_norm=args.max_grad_norm,
    )

    critic_utd_ratio = args.gradient_steps / float(
        private_transitions_per_round
    )
    actor_utd_ratio = critic_utd_ratio / float(args.policy_delay)
    total_simulator_steps_per_round = (
        spec.num_agents * private_transitions_per_round
    )
    normalization_simulator_steps = (
        args.normalization_rounds * total_simulator_steps_per_round
    )

    print("EnvSpec:", spec)
    print("Training mode: oracle-environment decentralized rollout SAC")
    print("Actor/Critic/Temperature parameter sharing: disabled")
    print("Trajectory and replay sharing: disabled")
    print("Private rollout owner count:", spec.num_agents)
    print("Actor snapshot synchronization: once before every rollout round")
    print("Opponent snapshots: frozen for collection and round updates")
    print("Actor input: normalized local oracle-state observation")
    print("Critic input: normalized joint model state + normalized joint action")
    print("Normalization: fixed zero-mean/unit-std statistics")
    print("Normalization statistics sharing: common and frozen, not trainable")
    print(f"Normalization calibration rounds: {args.normalization_rounds}")
    print(f"Normalization clip: ±{args.normalization_clip}")
    print("Private replay lifetime: one synchronization round")
    print(
        "Environment RNG protocol: fixed common-random-number schedule "
        "shared by all rollout owners"
    )
    print(f"Environment seed: {args.environment_seed}")
    print(f"Policy sampling seed: {args.policy_seed}")
    print(f"Fixed evaluation seed: {args.evaluation_seed}")
    print(f"Private transitions per round: {private_transitions_per_round}")
    print(f"Total simulator steps per round: {total_simulator_steps_per_round}")
    print(
        "Normalization calibration simulator steps: "
        f"{normalization_simulator_steps}"
    )
    print(f"Critic update-to-data ratio per owner: {critic_utd_ratio:.6f}")
    print(f"Post-warmup actor update-to-data ratio: {actor_utd_ratio:.6f}")

    hidden_dims = tuple(args.hidden_dims)
    actor = SACActor(
        action_dim=spec.policy_action_dim,
        hidden_dims=hidden_dims,
    )
    critic = TwinQNetwork(hidden_dims=hidden_dims)
    raw_independent_actor_apply = make_independent_sac_actor_apply(
        actor.apply,
        spec.num_agents,
        observation_stats=None,
    )

    initialization_key = jax.random.PRNGKey(args.seed)
    initialization_key = jax.random.fold_in(initialization_key, 0xA11CE)
    learner_states = create_independent_sac_states(
        initialization_key,
        actor,
        critic,
        num_agents=spec.num_agents,
        actor_obs_dim=spec.actor_obs_dim,
        critic_state_dim=spec.model_state_dim,
        action_dim=spec.policy_action_dim,
        config=sac_config,
    )

    common_initial_reset_keys = make_fixed_initial_reset_keys(
        args.environment_seed,
        num_envs=args.num_envs,
    )
    initial_env_state, initial_features = jax.vmap(adapter.reset)(
        common_initial_reset_keys
    )
    owner_env_states = tuple(
        _tree_copy(initial_env_state) for _ in range(spec.num_agents)
    )
    owner_features = tuple(
        _tree_copy(initial_features) for _ in range(spec.num_agents)
    )

    print("Calibrating frozen observation/state/action normalizers...")
    observation_stats, state_stats, action_stats = _calibrate_normalizers(
        adapter=adapter,
        raw_actor_apply=raw_independent_actor_apply,
        actor_params=independent_actor_params(learner_states),
        initial_env_state=initial_env_state,
        initial_features=initial_features,
        environment_seed=args.environment_seed,
        policy_seed=args.policy_seed,
        num_owners=spec.num_agents,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        normalization_rounds=args.normalization_rounds,
        normalization_clip=args.normalization_clip,
        normalization_epsilon=args.normalization_epsilon,
        normalization_min_std=args.normalization_min_std,
    )
    print(
        "Observation normalizer std range: "
        f"[{float(jnp.min(observation_stats.std)):.6f}, "
        f"{float(jnp.max(observation_stats.std)):.6f}]"
    )
    print(
        "State normalizer std range: "
        f"[{float(jnp.min(state_stats.std)):.6f}, "
        f"{float(jnp.max(state_stats.std)):.6f}]"
    )
    print(
        "Action normalizer mean/std: "
        f"mean={np.asarray(action_stats.mean)}, "
        f"std={np.asarray(action_stats.std)}"
    )

    independent_actor_apply = make_independent_sac_actor_apply(
        actor.apply,
        spec.num_agents,
        observation_stats=observation_stats,
    )
    collector = make_fixed_rng_oracle_rollout_collector(
        adapter,
        independent_actor_apply,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        jit=True,
    )
    agent_updates = tuple(
        make_sac_update(
            sac_config,
            agent_id=agent_id,
            num_agents=spec.num_agents,
            observation_stats=observation_stats,
            state_stats=state_stats,
            action_stats=action_stats,
            jit=True,
        )
        for agent_id in range(spec.num_agents)
    )
    fixed_batch_evaluators = tuple(
        make_fixed_batch_critic_evaluator(
            sac_config,
            agent_id=agent_id,
            num_agents=spec.num_agents,
            observation_stats=observation_stats,
            state_stats=state_stats,
            action_stats=action_stats,
            jit=True,
        )
        for agent_id in range(spec.num_agents)
    )
    evaluator = make_independent_sac_evaluator(
        adapter,
        actor,
        observation_stats=observation_stats,
        num_envs=args.eval_envs,
        horizon=args.max_steps + 1,
        jit=True,
    )

    replay_rngs = tuple(
        np.random.default_rng(args.seed + 10_000 + agent_id)
        for agent_id in range(spec.num_agents)
    )
    fixed_batch_rngs = tuple(
        np.random.default_rng(args.seed + 20_000 + agent_id)
        for agent_id in range(spec.num_agents)
    )
    update_key = jax.random.fold_in(
        jax.random.PRNGKey(args.seed),
        0xBADC0DE,
    )
    fixed_evaluation_key = jax.random.PRNGKey(args.evaluation_seed)
    fixed_batches = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "algorithm": "oracle_environment_decentralized_rollout_sac",
                "parameter_sharing": False,
                "trajectory_sharing": False,
                "private_replay_per_agent": True,
                "private_replay_lifetime": "one_synchronization_round",
                "actor_snapshot_exchange": True,
                "normalization": {
                    "enabled": True,
                    "statistics_frozen": True,
                    "statistics_scope": "common_across_agents_and_owners",
                    "calibration_rounds": args.normalization_rounds,
                    "calibration_simulator_steps": normalization_simulator_steps,
                    "clip": args.normalization_clip,
                    "epsilon": args.normalization_epsilon,
                    "minimum_std": args.normalization_min_std,
                    "observation": observation_stats.as_serializable(),
                    "state": state_stats.as_serializable(),
                    "action": action_stats.as_serializable(),
                    "reward_normalized": False,
                },
                "environment_rng": {
                    "protocol": "fixed_common_random_numbers",
                    "same_across_rollout_owners": True,
                    "stateless_by_round": True,
                    "environment_seed": args.environment_seed,
                    "policy_seed": args.policy_seed,
                    "evaluation_seed": args.evaluation_seed,
                },
                "arguments": vars(args)
                | {"output_dir": str(args.output_dir)},
                "sac": asdict(sac_config),
                "env_spec": asdict(spec),
                "private_transitions_per_round": private_transitions_per_round,
                "total_simulator_steps_per_round": (
                    total_simulator_steps_per_round
                ),
                "critic_utd_ratio_per_owner": critic_utd_ratio,
                "actor_utd_ratio_per_owner": actor_utd_ratio,
            },
            handle,
            indent=2,
        )
    with (args.output_dir / "normalization.json").open("w") as handle:
        json.dump(
            {
                "observation": observation_stats.as_serializable(),
                "state": state_stats.as_serializable(),
                "action": action_stats.as_serializable(),
                "statistics_frozen": True,
                "reward_normalized": False,
                "calibration_rounds": args.normalization_rounds,
                "calibration_simulator_steps": normalization_simulator_steps,
            },
            handle,
            indent=2,
        )

    history_path = args.output_dir / "metrics.jsonl"
    history_path.write_text("")
    best_evaluation_return = float("-inf")
    best_round = -1
    critic_gradient_steps_total = 0
    actor_update_cycles_total = 0

    for round_index in range(1, args.total_sync_rounds + 1):
        zero_based_round = round_index - 1

        # This is the only parameter exchange in the round. The tuple is never
        # averaged and is frozen for all three private environment rollouts.
        synchronized_actor_snapshots = independent_actor_params(
            learner_states
        )
        common_environment_schedule = make_fixed_environment_key_schedule(
            args.environment_seed,
            synchronization_round=zero_based_round,
            rollout_length=args.rollout_length,
            num_envs=args.num_envs,
        )
        use_random_actions = round_index <= args.random_collection_rounds

        # Strict private data ownership: buffers are refreshed after every
        # synchronization and never shared between agents.
        private_buffers = tuple(
            _new_private_buffer(args, spec)
            for _ in range(spec.num_agents)
        )
        round_rollouts = []
        next_owner_env_states = []
        next_owner_features = []
        for owner_agent_id in range(spec.num_agents):
            owner_policy_schedule = make_owner_policy_key_schedule(
                args.policy_seed,
                synchronization_round=zero_based_round,
                owner_agent_id=owner_agent_id,
                rollout_length=args.rollout_length,
            )
            rollout = collector(
                synchronized_actor_snapshots,
                owner_env_states[owner_agent_id],
                owner_features[owner_agent_id],
                common_environment_schedule,
                owner_policy_schedule,
                jnp.asarray(use_random_actions),
            )
            _block_until_ready(rollout.batch)
            _append_real_rollout(
                rollout,
                private_buffers[owner_agent_id],
            )
            round_rollouts.append(rollout)
            next_owner_env_states.append(rollout.env_state)
            next_owner_features.append(rollout.features)
        owner_env_states = tuple(next_owner_env_states)
        owner_features = tuple(next_owner_features)

        if fixed_batches is None:
            fixed_batches = tuple(
                private_buffers[agent_id].sample(
                    args.fixed_batch_size,
                    fixed_batch_rngs[agent_id],
                )
                for agent_id in range(spec.num_agents)
            )

        metrics: dict[str, float | int | bool] = {
            "synchronization_round": round_index,
            "private_environment_steps_per_owner": (
                round_index * private_transitions_per_round
            ),
            "training_simulator_steps": (
                round_index * total_simulator_steps_per_round
            ),
            "normalization_calibration_simulator_steps": (
                normalization_simulator_steps
            ),
            "total_simulator_steps": (
                normalization_simulator_steps
                + round_index * total_simulator_steps_per_round
            ),
            "agent_transitions_generated": (
                round_index
                * total_simulator_steps_per_round
                * spec.num_agents
            ),
            "random_action_collection": use_random_actions,
            "actor_snapshot_sync_count": round_index,
            "environment_seed": args.environment_seed,
            "policy_seed": args.policy_seed,
            "common_environment_rng_across_owners": True,
            "private_replay_refreshed_this_round": True,
            "critic_utd_ratio_per_owner": critic_utd_ratio,
            "actor_utd_ratio_per_owner": actor_utd_ratio,
            "critic_gradient_steps_total": critic_gradient_steps_total,
            "actor_update_cycles_total": actor_update_cycles_total,
            "normalization_enabled": True,
            "normalization_statistics_frozen": True,
            "observation_normalizer_std_min": float(
                jnp.min(observation_stats.std)
            ),
            "observation_normalizer_std_max": float(
                jnp.max(observation_stats.std)
            ),
            "state_normalizer_std_min": float(jnp.min(state_stats.std)),
            "state_normalizer_std_max": float(jnp.max(state_stats.std)),
            "action_normalizer_std_min": float(jnp.min(action_stats.std)),
            "action_normalizer_std_max": float(jnp.max(action_stats.std)),
        }
        for agent_id in range(spec.num_agents):
            rollout_i = round_rollouts[agent_id]
            metrics[f"agent{agent_id}_private_replay_size"] = len(
                private_buffers[agent_id]
            )
            metrics[f"agent{agent_id}_rollout_mean_reward"] = (
                _device_float(jnp.mean(rollout_i.batch.rewards))
            )
            metrics[f"agent{agent_id}_rollout_pair_collision_rate"] = (
                _device_float(
                    jnp.mean(rollout_i.batch.pair_collision_rates)
                )
            )
            metrics[f"agent{agent_id}_completed_episode_transitions"] = int(
                jax.device_get(
                    jnp.sum(rollout_i.batch.episode_dones)
                )
            )
        metrics["rollout_mean_reward"] = float(
            np.mean(
                [
                    metrics[f"agent{i}_rollout_mean_reward"]
                    for i in range(spec.num_agents)
                ]
            )
        )
        metrics["rollout_pair_collision_rate"] = float(
            np.mean(
                [
                    metrics[f"agent{i}_rollout_pair_collision_rate"]
                    for i in range(spec.num_agents)
                ]
            )
        )

        per_agent_metric_lists = [[] for _ in range(spec.num_agents)]
        actor_updates_this_round = 0
        for _gradient_step in range(args.gradient_steps):
            next_critic_gradient_step = critic_gradient_steps_total + 1
            do_actor_update = _should_update_actor(
                next_critic_gradient_step,
                critic_warmup_steps=args.critic_warmup_steps,
                policy_delay=args.policy_delay,
            )
            update_key, gradient_key = jax.random.split(update_key)
            agent_keys = jax.random.split(gradient_key, spec.num_agents)

            next_learner_states = []
            for agent_id in range(spec.num_agents):
                replay_batch = private_buffers[agent_id].sample(
                    args.batch_size,
                    replay_rngs[agent_id],
                )
                focal_actor_tuple = _replace_focal_snapshot(
                    synchronized_actor_snapshots,
                    focal_agent_id=agent_id,
                    current_focal_actor_params=(
                        learner_states[agent_id].actor_state.params
                    ),
                )
                output = agent_updates[agent_id](
                    agent_keys[agent_id],
                    learner_states[agent_id],
                    replay_batch,
                    focal_actor_tuple,
                    jnp.asarray(do_actor_update),
                )
                _block_until_ready(output.metrics)
                next_learner_states.append(output.learner_state)
                per_agent_metric_lists[agent_id].append(
                    {
                        name: _device_float(value)
                        for name, value in output.metrics.items()
                    }
                )
            learner_states = tuple(next_learner_states)
            critic_gradient_steps_total = next_critic_gradient_step
            if do_actor_update:
                actor_update_cycles_total += 1
                actor_updates_this_round += 1

        averaged_agent_metrics = []
        for agent_id, metric_list in enumerate(per_agent_metric_lists):
            agent_average = {
                name: float(np.mean([entry[name] for entry in metric_list]))
                for name in metric_list[0]
            }
            averaged_agent_metrics.append(agent_average)
            for name, value in agent_average.items():
                metrics[f"agent{agent_id}_{name}"] = value
        for name in averaged_agent_metrics[0]:
            metrics[name] = float(
                np.mean(
                    [entry[name] for entry in averaged_agent_metrics]
                )
            )

        metrics["critic_gradient_steps_total"] = critic_gradient_steps_total
        metrics["actor_update_cycles_total"] = actor_update_cycles_total
        metrics["actor_updates_this_round"] = actor_updates_this_round
        metrics["critic_warmup_complete"] = (
            critic_gradient_steps_total > args.critic_warmup_steps
        )

        do_evaluation = (
            round_index == 1
            or round_index % args.eval_interval == 0
            or round_index == args.total_sync_rounds
        )
        if do_evaluation:
            # Deliberately reuse the exact same evaluation key every time.
            evaluation = evaluator(
                fixed_evaluation_key,
                independent_actor_params(learner_states),
            )
            _block_until_ready(evaluation)
            metrics["evaluation_return_mean"] = _device_float(
                jnp.mean(evaluation.episode_returns)
            )
            metrics["evaluation_return_std"] = _device_float(
                jnp.std(evaluation.episode_returns)
            )
            metrics["evaluation_length_mean"] = _device_float(
                jnp.mean(evaluation.episode_lengths)
            )
            metrics["evaluation_completion_rate"] = _device_float(
                jnp.mean(evaluation.completed.astype(jnp.float32))
            )
            metrics[
                "evaluation_pair_collision_rate_mean"
            ] = _device_float(
                jnp.mean(evaluation.episode_pair_collision_rates)
            )
            metrics[
                "evaluation_pair_collision_rate_std"
            ] = _device_float(
                jnp.std(evaluation.episode_pair_collision_rates)
            )

            current_actor_params = independent_actor_params(learner_states)
            fixed_agent_metrics = []
            for agent_id in range(spec.num_agents):
                focal_actor_tuple = _replace_focal_snapshot(
                    synchronized_actor_snapshots,
                    focal_agent_id=agent_id,
                    current_focal_actor_params=current_actor_params[agent_id],
                )
                fixed_metrics_i = fixed_batch_evaluators[agent_id](
                    learner_states[agent_id],
                    fixed_batches[agent_id],
                    focal_actor_tuple,
                )
                _block_until_ready(fixed_metrics_i)
                fixed_metrics_i = {
                    name: _device_float(value)
                    for name, value in fixed_metrics_i.items()
                }
                fixed_agent_metrics.append(fixed_metrics_i)
                for name, value in fixed_metrics_i.items():
                    metrics[f"agent{agent_id}_{name}"] = value
            for name in fixed_agent_metrics[0]:
                metrics[name] = float(
                    np.mean([entry[name] for entry in fixed_agent_metrics])
                )

            if metrics["evaluation_return_mean"] > best_evaluation_return:
                best_evaluation_return = float(
                    metrics["evaluation_return_mean"]
                )
                best_round = round_index
                _save_checkpoint(
                    args.output_dir,
                    round_index,
                    learner_states,
                    args,
                    sac_config,
                    observation_stats,
                    state_stats,
                    action_stats,
                    filename="best.pkl",
                    critic_gradient_steps_total=critic_gradient_steps_total,
                    actor_update_cycles_total=actor_update_cycles_total,
                    evaluation_metrics={
                        "return_mean": metrics["evaluation_return_mean"],
                        "return_std": metrics["evaluation_return_std"],
                        "pair_collision_rate_mean": metrics[
                            "evaluation_pair_collision_rate_mean"
                        ],
                        "fixed_td_abs_mean": metrics[
                            "fixed_td_abs_mean"
                        ],
                    },
                )

        with history_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if do_evaluation:
            print(
                f"round={round_index:04d} "
                f"owner_env_steps={metrics['private_environment_steps_per_owner']} "
                f"total_sim_steps={metrics['total_simulator_steps']} "
                f"eval_return={metrics['evaluation_return_mean']:.4f} "
                f"collision={metrics['evaluation_pair_collision_rate_mean']:.4f} "
                f"critic_steps={critic_gradient_steps_total} "
                f"actor_updates={actor_update_cycles_total} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"td_abs={metrics['td_abs_mean']:.4f} "
                f"fixed_td={metrics['fixed_td_abs_mean']:.4f} "
                f"alpha={metrics['alpha']:.4f} "
                f"entropy={metrics['policy_entropy']:.4f}"
            )

        if (
            round_index % args.checkpoint_interval == 0
            or round_index == args.total_sync_rounds
        ):
            _save_checkpoint(
                args.output_dir,
                round_index,
                learner_states,
                args,
                sac_config,
                observation_stats,
                state_stats,
                action_stats,
                filename="latest.pkl",
                critic_gradient_steps_total=critic_gradient_steps_total,
                actor_update_cycles_total=actor_update_cycles_total,
            )

    print("Training complete.")
    print("Results:", args.output_dir)
    print("Metrics:", history_path)
    print("Latest checkpoint:", args.output_dir / "latest.pkl")
    print("Best checkpoint:", args.output_dir / "best.pkl")
    print("Critic gradient steps:", critic_gradient_steps_total)
    print("Actor update cycles:", actor_update_cycles_total)
    print(
        "Best evaluation return:",
        f"{best_evaluation_return:.4f}",
        "at synchronization round",
        best_round,
    )


if __name__ == "__main__":
    main()
