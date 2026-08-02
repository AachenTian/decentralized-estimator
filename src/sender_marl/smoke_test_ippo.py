from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from sender_marl.core.networks import ContinuousActor, LocalValueNetwork
from sender_marl.envs import make_env_adapter
from sender_marl.on_policy.evaluation import make_policy_evaluator
from sender_marl.on_policy.rollout import make_rollout_collector
from sender_marl.on_policy.train_state import (
    PPOConfig,
    create_ppo_train_states,
)
from sender_marl.on_policy.update import make_ppo_update


def _assert_finite_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and not bool(jnp.all(jnp.isfinite(leaf))):
            raise AssertionError("A JAX array contains NaN or Inf.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=3,
        num_landmarks=3,
        max_steps=25,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )
    spec = adapter.spec
    actor = ContinuousActor(action_dim=spec.policy_action_dim)
    value_network = LocalValueNetwork()
    config = PPOConfig(update_epochs=2, num_minibatches=2)

    key = jax.random.PRNGKey(args.seed)
    key, init_key, collect_key, update_key, eval_key = jax.random.split(key, 5)
    dummy_obs = jnp.zeros(
        (spec.num_agents, spec.actor_obs_dim),
        dtype=jnp.float32,
    )
    actor_state, value_state = create_ppo_train_states(
        init_key,
        actor,
        value_network,
        dummy_obs,
        dummy_obs,
        config,
    )
    initial_actor_params = actor_state.params

    collector = make_rollout_collector(
        adapter,
        actor.apply,
        value_network.apply,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        critic_mode="local",
        jit=True,
    )
    update_fn = make_ppo_update(config, jit=True)

    for update_index in range(args.updates):
        collect_key, step_collect_key = jax.random.split(collect_key)
        rollout = collector(
            step_collect_key,
            actor_state.params,
            value_state.params,
        )
        update_key, step_update_key = jax.random.split(update_key)
        output = update_fn(
            step_update_key,
            actor_state,
            value_state,
            rollout,
        )
        actor_state = output.actor_state
        value_state = output.value_state
        _assert_finite_tree(output.metrics)
        print(
            f"update={update_index + 1} "
            f"policy_loss={float(output.metrics['policy_loss']):.4f} "
            f"value_loss={float(output.metrics['unscaled_value_loss']):.4f} "
            f"entropy={float(output.metrics['entropy']):.4f} "
            f"kl={float(output.metrics['approx_kl']):.6f}"
        )

    changed = any(
        not bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(initial_actor_params),
            jax.tree_util.tree_leaves(actor_state.params),
        )
    )
    if not changed:
        raise AssertionError("Actor parameters did not change.")

    evaluator = make_policy_evaluator(
        adapter,
        actor.apply,
        num_envs=4,
        horizon=26,
        jit=True,
    )
    evaluation = evaluator(eval_key, actor_state.params)
    _assert_finite_tree(evaluation)
    print("Evaluation returns:", evaluation.episode_returns)
    print("Completion rate:", float(jnp.mean(evaluation.completed)))
    print("All IPPO smoke tests passed.")


if __name__ == "__main__":
    main()
