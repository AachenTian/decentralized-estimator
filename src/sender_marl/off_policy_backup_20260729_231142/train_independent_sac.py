from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sender_marl.envs import make_env_adapter
from sender_marl.off_policy.collector import make_real_rollout_collector
from sender_marl.off_policy.evaluation import make_independent_sac_evaluator
from sender_marl.off_policy.independent import (
    create_independent_sac_states,
    independent_actor_params,
    make_independent_sac_actor_apply,
)
from sender_marl.off_policy.networks import SACActor, TwinQNetwork
from sender_marl.off_policy.replay_buffer import JointReplayBuffer
from sender_marl.off_policy.sac_state import SACConfig
from sender_marl.off_policy.sac_update import make_sac_update


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _device_float(value: Any) -> float:
    return float(jax.device_get(value))


def _save_checkpoint(
    output_dir: Path,
    update_index: int,
    learner_states,
    args: argparse.Namespace,
    sac_config: SACConfig,
    *,
    filename: str,
    evaluation_metrics: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "independent_sac_joint_critic",
        "parameter_sharing": False,
        "actor_input": "local_actor_observation",
        "critic_input": "joint_model_state_and_joint_action",
        "actor_snapshot_exchange_ready": True,
        "model_rollout_implemented": False,
        "update": update_index,
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
        "evaluation_metrics": evaluation_metrics,
    }
    with (output_dir / filename).open("wb") as handle:
        pickle.dump(payload, handle)


def _append_real_rollout(
    rollout,
    replay_buffer: JointReplayBuffer,
) -> None:
    batch = jax.device_get(rollout.batch)
    time_steps, num_envs, _num_agents = batch.rewards.shape
    flat_count = time_steps * num_envs

    # Collector broadcasts the shared team reward over the agent axis. Store it
    # once with the complete joint context.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent decentralized SAC actors with one private "
            "centralized-input twin critic per agent. Every critic consumes "
            "the full model state and full joint action. Actor, critic, "
            "target, entropy temperature, and optimizer parameters are never "
            "shared or averaged."
        )
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--total-updates", type=int, default=5000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--buffer-capacity", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
        help="Real joint replay size before gradient updates begin.",
    )

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-alpha", type=float, default=0.2)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="Defaults to -policy_action_dim for each focal actor.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256]
    )

    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/independent_sac_joint_critic_oracle"),
    )
    args = parser.parse_args()

    positive_integer_arguments = {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "total_updates": args.total_updates,
        "gradient_steps": args.gradient_steps,
        "buffer_capacity": args.buffer_capacity,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "eval_envs": args.eval_envs,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
    }
    invalid = {
        name: value
        for name, value in positive_integer_arguments.items()
        if value < 1
    }
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}.")
    if args.buffer_capacity < args.batch_size:
        raise ValueError("buffer-capacity must be at least batch-size.")

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

    env_steps_per_update = args.num_envs * args.rollout_length
    utd_ratio = args.gradient_steps / float(env_steps_per_update)
    print("EnvSpec:", spec)
    print("Training mode: independent SAC with private joint critics")
    print("Actor input: local actor observation")
    print(
        "Critic input: joint model state + joint action "
        f"({spec.model_state_dim} + "
        f"{spec.num_agents * spec.policy_action_dim} dimensions)"
    )
    print("Actor/Critic/Temperature parameter sharing: disabled")
    print("Remote-state source: oracle (optimizer-validation stage)")
    print(f"Update-to-data ratio: {utd_ratio:.6f}")
    print(
        "Opponent policy use: frozen actor snapshot tuple per gradient step"
    )

    hidden_dims = tuple(args.hidden_dims)
    actor = SACActor(
        action_dim=spec.policy_action_dim,
        hidden_dims=hidden_dims,
    )
    critic = TwinQNetwork(hidden_dims=hidden_dims)
    independent_actor_apply = make_independent_sac_actor_apply(
        actor.apply, spec.num_agents
    )

    key = jax.random.PRNGKey(args.seed)
    (
        _key,
        init_key,
        reset_key,
        collect_key,
        update_key,
        eval_key,
    ) = jax.random.split(key, 6)
    learner_states = create_independent_sac_states(
        init_key,
        actor,
        critic,
        num_agents=spec.num_agents,
        actor_obs_dim=spec.actor_obs_dim,
        critic_state_dim=spec.model_state_dim,
        action_dim=spec.policy_action_dim,
        config=sac_config,
    )

    reset_keys = jax.random.split(reset_key, args.num_envs)
    env_state, features = jax.vmap(adapter.reset)(reset_keys)
    collector = make_real_rollout_collector(
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
            jit=True,
        )
        for agent_id in range(spec.num_agents)
    )
    evaluator = make_independent_sac_evaluator(
        adapter,
        actor,
        num_envs=args.eval_envs,
        horizon=args.max_steps + 1,
        jit=True,
    )

    replay_buffer = JointReplayBuffer(
        capacity=args.buffer_capacity,
        num_agents=spec.num_agents,
        observation_dim=spec.actor_obs_dim,
        action_dim=spec.policy_action_dim,
        model_state_dim=spec.model_state_dim,
    )
    # Agents sample independently from the same real joint data pool.
    replay_rngs = tuple(
        np.random.default_rng(args.seed + 10_000 + agent_id)
        for agent_id in range(spec.num_agents)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "algorithm": "independent_sac_joint_critic",
                "parameter_sharing": False,
                "actor_input": "local_actor_observation",
                "critic_input": "joint_model_state_and_joint_action",
                "arguments": vars(args)
                | {"output_dir": str(args.output_dir)},
                "sac": asdict(sac_config),
                "env_spec": asdict(spec),
                "utd_ratio": utd_ratio,
            },
            handle,
            indent=2,
        )

    history_path = args.output_dir / "metrics.jsonl"
    history_path.write_text("")
    best_evaluation_return = float("-inf")
    best_update = -1

    for update_index in range(1, args.total_updates + 1):
        collect_key, this_collect_key = jax.random.split(collect_key)
        use_random_actions = len(replay_buffer) < args.learning_starts
        rollout = collector(
            this_collect_key,
            independent_actor_params(learner_states),
            env_state,
            features,
            jnp.asarray(use_random_actions),
        )
        _block_until_ready(rollout.batch)
        env_state = rollout.env_state
        features = rollout.features
        _append_real_rollout(rollout, replay_buffer)

        metrics: dict[str, float | int | bool] = {
            "update": update_index,
            "environment_steps": update_index * env_steps_per_update,
            "agent_transitions": (
                update_index * env_steps_per_update * spec.num_agents
            ),
            "replay_size": len(replay_buffer),
            "random_action_collection": use_random_actions,
            "utd_ratio": utd_ratio,
            "rollout_mean_reward": _device_float(
                jnp.mean(rollout.batch.rewards)
            ),
            "rollout_pair_collision_rate": _device_float(
                jnp.mean(rollout.batch.pair_collision_rates)
            ),
            "completed_episode_transitions": int(
                jax.device_get(jnp.sum(rollout.batch.episode_dones))
            ),
        }

        ready_to_learn = len(replay_buffer) >= max(
            args.learning_starts, args.batch_size
        )
        if ready_to_learn:
            per_agent_metric_lists = [
                [] for _ in range(spec.num_agents)
            ]
            for _gradient_step in range(args.gradient_steps):
                update_key, gradient_key = jax.random.split(update_key)
                agent_keys = jax.random.split(
                    gradient_key, spec.num_agents
                )

                # Synchronize actor snapshots for inference only. All agents in
                # this gradient step see the same frozen pre-update tuple, so
                # Python loop order cannot change another learner's target.
                actor_snapshots = independent_actor_params(learner_states)
                next_learner_states = []
                for agent_id in range(spec.num_agents):
                    replay_batch = replay_buffer.sample(
                        args.batch_size, replay_rngs[agent_id]
                    )
                    output = agent_updates[agent_id](
                        agent_keys[agent_id],
                        learner_states[agent_id],
                        replay_batch,
                        actor_snapshots,
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

            averaged_agent_metrics = []
            for agent_id, metric_list in enumerate(
                per_agent_metric_lists
            ):
                agent_average = {
                    name: float(
                        np.mean([entry[name] for entry in metric_list])
                    )
                    for name in metric_list[0]
                }
                averaged_agent_metrics.append(agent_average)
                for name, value in agent_average.items():
                    metrics[f"agent{agent_id}_{name}"] = value

            for name in averaged_agent_metrics[0]:
                metrics[name] = float(
                    np.mean(
                        [
                            agent_metrics[name]
                            for agent_metrics in averaged_agent_metrics
                        ]
                    )
                )

        do_evaluation = (
            update_index == 1
            or update_index % args.eval_interval == 0
            or update_index == args.total_updates
        )
        if do_evaluation:
            eval_key, this_eval_key = jax.random.split(eval_key)
            evaluation = evaluator(
                this_eval_key,
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

            if metrics["evaluation_return_mean"] > best_evaluation_return:
                best_evaluation_return = float(
                    metrics["evaluation_return_mean"]
                )
                best_update = update_index
                _save_checkpoint(
                    args.output_dir,
                    update_index,
                    learner_states,
                    args,
                    sac_config,
                    filename="best.pkl",
                    evaluation_metrics={
                        "return_mean": float(
                            metrics["evaluation_return_mean"]
                        ),
                        "return_std": float(
                            metrics["evaluation_return_std"]
                        ),
                        "length_mean": float(
                            metrics["evaluation_length_mean"]
                        ),
                        "completion_rate": float(
                            metrics["evaluation_completion_rate"]
                        ),
                        "pair_collision_rate_mean": float(
                            metrics[
                                "evaluation_pair_collision_rate_mean"
                            ]
                        ),
                        "pair_collision_rate_std": float(
                            metrics[
                                "evaluation_pair_collision_rate_std"
                            ]
                        ),
                    },
                )

        with history_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if do_evaluation:
            optimization_text = (
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"td_abs={metrics['td_abs_mean']:.4f} "
                f"q={metrics['policy_q_mean']:.4f} "
                f"alpha={metrics['alpha']:.4f} "
                f"entropy={metrics['policy_entropy']:.4f}"
                if ready_to_learn
                else "learning=not_started"
            )
            print(
                f"update={update_index:05d} "
                f"env_steps={metrics['environment_steps']} "
                f"replay={metrics['replay_size']} "
                f"eval_return={metrics['evaluation_return_mean']:.4f} "
                f"completion={metrics['evaluation_completion_rate']:.3f} "
                f"collision={metrics['evaluation_pair_collision_rate_mean']:.4f} "
                f"{optimization_text}"
            )

        if (
            update_index % args.checkpoint_interval == 0
            or update_index == args.total_updates
        ):
            _save_checkpoint(
                args.output_dir,
                update_index,
                learner_states,
                args,
                sac_config,
                filename="latest.pkl",
            )

    print("Training complete.")
    print("Results:", args.output_dir)
    print("Metrics:", history_path)
    print("Latest checkpoint:", args.output_dir / "latest.pkl")
    print("Best checkpoint:", args.output_dir / "best.pkl")
    print(
        "Best evaluation return:",
        f"{best_evaluation_return:.4f}",
        "at update",
        best_update,
    )


if __name__ == "__main__":
    main()
