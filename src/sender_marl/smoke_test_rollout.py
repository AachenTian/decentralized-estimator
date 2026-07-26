from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from sender_marl.core.networks import (
    CentralizedValueNetwork,
    ContinuousActor,
    LocalValueNetwork,
)
from sender_marl.envs import make_env_adapter
from sender_marl.on_policy.rollout import make_rollout_collector


def _assert_finite(name: str, value) -> None:
    if not bool(jnp.all(jnp.isfinite(value))):
        raise AssertionError(f"{name} contains NaN or Inf.")


def _block_until_ready(tree) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-length", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

    actor = ContinuousActor(action_dim=spec.policy_action_dim)
    local_value = LocalValueNetwork()
    centralized_value = CentralizedValueNetwork()

    key = jax.random.PRNGKey(args.seed)
    key, actor_key, local_value_key, centralized_value_key, collect_key = (
        jax.random.split(key, 5)
    )
    dummy_actor_obs = jnp.zeros(
        (spec.num_agents, spec.actor_obs_dim),
        dtype=jnp.float32,
    )
    dummy_centralized_obs = jnp.zeros(
        (spec.num_agents, spec.critic_obs_dim),
        dtype=jnp.float32,
    )
    actor_params = actor.init(actor_key, dummy_actor_obs)["params"]
    local_value_params = local_value.init(
        local_value_key,
        dummy_actor_obs,
    )["params"]
    centralized_value_params = centralized_value.init(
        centralized_value_key,
        dummy_centralized_obs,
    )["params"]

    local_collector = make_rollout_collector(
        adapter,
        actor.apply,
        local_value.apply,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        critic_mode="local",
        jit=True,
    )
    local_output = local_collector(
        collect_key,
        actor_params,
        local_value_params,
    )
    _block_until_ready(local_output)
    batch = local_output.batch

    expected = {
        "actor_obs": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
            spec.actor_obs_dim,
        ),
        "critic_obs": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
            spec.actor_obs_dim,
        ),
        "actions": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
            spec.policy_action_dim,
        ),
        "log_probs": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
        ),
        "values": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
        ),
        "rewards": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
        ),
        "dones": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
        ),
        "local_states": (
            args.rollout_length,
            args.num_envs,
            spec.num_agents,
            spec.local_state_dim,
        ),
        "pair_collision_rate": (
            args.rollout_length,
            args.num_envs,
        ),
    }
    actual = {
        "actor_obs": batch.actor_obs.shape,
        "critic_obs": batch.critic_obs.shape,
        "actions": batch.actions.shape,
        "log_probs": batch.log_probs.shape,
        "values": batch.values.shape,
        "rewards": batch.rewards.shape,
        "dones": batch.dones.shape,
        "local_states": batch.local_states.shape,
        "pair_collision_rate": batch.metrics["pair_collision_rate"].shape,
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise AssertionError(
                f"{name}: expected {expected_shape}, got {actual[name]}."
            )

    for name, value in {
        "actor_obs": batch.actor_obs,
        "actions": batch.actions,
        "pre_tanh_actions": batch.pre_tanh_actions,
        "log_probs": batch.log_probs,
        "values": batch.values,
        "rewards": batch.rewards,
        "local_states": batch.local_states,
        "next_local_states": batch.next_local_states,
    }.items():
        _assert_finite(name, value)

    if not bool(jnp.all(batch.actions <= 1.0 + 1e-6)):
        raise AssertionError("Actions exceed +1.")
    if not bool(jnp.all(batch.actions >= -1.0 - 1e-6)):
        raise AssertionError("Actions are below -1.")

    centralized_collector = make_rollout_collector(
        adapter,
        actor.apply,
        centralized_value.apply,
        num_envs=max(1, min(args.num_envs, 2)),
        rollout_length=2,
        critic_mode="centralized",
        jit=True,
    )
    centralized_output = centralized_collector(
        jax.random.fold_in(collect_key, 1),
        actor_params,
        centralized_value_params,
    )
    _block_until_ready(centralized_output)
    central_shape = centralized_output.batch.critic_obs.shape
    expected_central_shape = (
        2,
        max(1, min(args.num_envs, 2)),
        spec.num_agents,
        spec.critic_obs_dim,
    )
    if central_shape != expected_central_shape:
        raise AssertionError(
            "Centralized critic observation shape mismatch: "
            f"expected {expected_central_shape}, got {central_shape}."
        )

    print("Local rollout shapes:")
    for name, shape in actual.items():
        print(f"  {name}: {shape}")
    print("Centralized critic shape:", central_shape)
    print(
        "Mean reward:",
        float(jnp.mean(batch.rewards)),
    )
    print(
        "Mean pair-step collision rate:",
        float(jnp.mean(batch.metrics["pair_collision_rate"])),
    )
    print(
        "Completed episode transitions:",
        int(jnp.sum(batch.episode_dones)),
    )
    print("All rollout collector smoke tests passed.")


if __name__ == "__main__":
    main()
