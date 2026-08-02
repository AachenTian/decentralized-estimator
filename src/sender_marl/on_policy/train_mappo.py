from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from sender_marl.core.networks import (
    CentralizedValueNetwork,
    ContinuousActor,
)
from sender_marl.envs import make_env_adapter
from sender_marl.on_policy.evaluation import make_policy_evaluator
from sender_marl.on_policy.rollout import make_rollout_collector
from sender_marl.on_policy.train_state import (
    PPOConfig,
    create_ppo_train_states,
)
from sender_marl.on_policy.update import make_ppo_update


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _to_float_dict(metrics) -> dict[str, float]:
    return {
        name: float(jax.device_get(value))
        for name, value in metrics.items()
    }


def _save_checkpoint(
    output_dir: Path,
    update_index: int,
    actor_state,
    value_state,
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    *,
    filename: str = "latest.pkl",
    evaluation_metrics: dict[str, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update_index,
        "actor_params": jax.device_get(actor_state.params),
        "value_params": jax.device_get(value_state.params),
        "args": vars(args),
        "ppo_config": asdict(ppo_config),
        "evaluation_metrics": evaluation_metrics,
    }
    with (output_dir / filename).open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train parameter-sharing MAPPO on an environment adapter. "
            "This first version uses oracle remote states to validate the "
            "centralized-critic optimizer independently of the estimator."
        )
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mappo_simple_spread"),
    )
    args = parser.parse_args()

    if args.total_updates < 1:
        raise ValueError("total-updates must be positive.")
    if args.eval_interval < 1:
        raise ValueError("eval-interval must be positive.")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive.")

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        max_steps=args.max_steps,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )
    spec = adapter.spec
    print("EnvSpec:", spec)
    print("Training mode: parameter-sharing MAPPO with centralized critic")
    print("Remote-state source: oracle (centralized-critic validation stage)")

    ppo_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coef,
        value_loss_coefficient=args.value_coef,
        actor_learning_rate=args.actor_lr,
        value_learning_rate=args.value_lr,
        max_gradient_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )

    actor = ContinuousActor(action_dim=spec.policy_action_dim)
    value_network = CentralizedValueNetwork()
    dummy_actor_obs = jnp.zeros(
        (spec.num_agents, spec.actor_obs_dim),
        dtype=jnp.float32,
    )
    dummy_critic_obs = jnp.zeros(
        (spec.num_agents, spec.critic_obs_dim),
        dtype=jnp.float32,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key, collect_key, update_key, eval_key = jax.random.split(key, 5)
    actor_state, value_state = create_ppo_train_states(
        init_key,
        actor,
        value_network,
        dummy_actor_obs,
        dummy_critic_obs,
        ppo_config,
    )

    collector = make_rollout_collector(
        adapter,
        actor.apply,
        value_network.apply,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        critic_mode="centralized",
        jit=True,
    )
    ppo_update = make_ppo_update(ppo_config, jit=True)
    evaluator = make_policy_evaluator(
        adapter,
        actor.apply,
        num_envs=args.eval_envs,
        horizon=args.max_steps + 1,
        jit=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "arguments": vars(args) | {"output_dir": str(args.output_dir)},
                "ppo": asdict(ppo_config),
                "env_spec": asdict(spec),
            },
            handle,
            indent=2,
        )

    history_path = args.output_dir / "metrics.jsonl"
    history_path.write_text("")

    best_evaluation_return = float("-inf")
    best_update = -1

    environment_steps_per_update = args.num_envs * args.rollout_length
    agent_transitions_per_update = (
        environment_steps_per_update * spec.num_agents
    )

    for update_index in range(1, args.total_updates + 1):
        collect_key, this_collect_key = jax.random.split(collect_key)
        rollout = collector(
            this_collect_key,
            actor_state.params,
            value_state.params,
        )

        update_key, this_update_key = jax.random.split(update_key)
        update_output = ppo_update(
            this_update_key,
            actor_state,
            value_state,
            rollout,
        )
        actor_state = update_output.actor_state
        value_state = update_output.value_state
        _block_until_ready(update_output.metrics)

        metrics = _to_float_dict(update_output.metrics)
        metrics["update"] = update_index
        metrics["environment_steps"] = (
            update_index * environment_steps_per_update
        )
        metrics["agent_transitions"] = (
            update_index * agent_transitions_per_update
        )
        metrics["rollout_mean_reward"] = float(
            jnp.mean(rollout.batch.rewards)
        )
        metrics["completed_episode_transitions"] = int(
            jnp.sum(rollout.batch.episode_dones)
        )
        if "pair_collision_rate" in rollout.batch.metrics:
            metrics["rollout_pair_collision_rate"] = float(
                jnp.mean(
                    rollout.batch.metrics["pair_collision_rate"]
                )
            )

        if update_index == 1 or update_index % args.eval_interval == 0:
            eval_key, this_eval_key = jax.random.split(eval_key)
            evaluation = evaluator(this_eval_key, actor_state.params)
            _block_until_ready(evaluation)
            metrics["evaluation_return_mean"] = float(
                jnp.mean(evaluation.episode_returns)
            )
            metrics["evaluation_return_std"] = float(
                jnp.std(evaluation.episode_returns)
            )
            metrics["evaluation_length_mean"] = float(
                jnp.mean(evaluation.episode_lengths)
            )
            metrics["evaluation_completion_rate"] = float(
                jnp.mean(evaluation.completed.astype(jnp.float32))
            )
            metrics["evaluation_pair_collision_rate_mean"] = float(
                jnp.mean(evaluation.episode_pair_collision_rates)
            )
            metrics["evaluation_pair_collision_rate_std"] = float(
                jnp.std(evaluation.episode_pair_collision_rates)
            )

            if (
                metrics["evaluation_return_mean"]
                > best_evaluation_return
            ):
                best_evaluation_return = metrics[
                    "evaluation_return_mean"
                ]
                best_update = update_index
                _save_checkpoint(
                    args.output_dir,
                    update_index,
                    actor_state,
                    value_state,
                    args,
                    ppo_config,
                    filename="best.pkl",
                    evaluation_metrics={
                        "return_mean": metrics[
                            "evaluation_return_mean"
                        ],
                        "return_std": metrics[
                            "evaluation_return_std"
                        ],
                        "length_mean": metrics[
                            "evaluation_length_mean"
                        ],
                        "completion_rate": metrics[
                            "evaluation_completion_rate"
                        ],
                        "pair_collision_rate_mean": metrics[
                            "evaluation_pair_collision_rate_mean"
                        ],
                        "pair_collision_rate_std": metrics[
                            "evaluation_pair_collision_rate_std"
                        ],
                    },
                )

        with history_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if update_index == 1 or update_index % args.eval_interval == 0:
            print(
                f"update={update_index:04d} "
                f"env_steps={metrics['environment_steps']} "
                f"eval_return={metrics.get('evaluation_return_mean', float('nan')):.4f} "
                f"completion={metrics.get('evaluation_completion_rate', float('nan')):.3f} "
                f"collision={metrics.get('evaluation_pair_collision_rate_mean', float('nan')):.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['unscaled_value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f} "
                f"kl={metrics['approx_kl']:.6f} "
                f"clip={metrics['clip_fraction']:.3f} "
                f"ev={metrics['explained_variance']:.3f}"
            )

        if (
            update_index % args.checkpoint_interval == 0
            or update_index == args.total_updates
        ):
            _save_checkpoint(
                args.output_dir,
                update_index,
                actor_state,
                value_state,
                args,
                ppo_config,
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
