from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from sender_marl.envs import make_env_adapter, registered_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the environment-independent JAX MARL adapter."
    )
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--remote-feature-mode",
        choices=("relative_position", "relative_state"),
        default="relative_state",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Registered adapters:", registered_envs())

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        max_steps=args.max_steps,
        action_mode="force_2d",
        remote_feature_mode=args.remote_feature_mode,
    )
    print("EnvSpec:", adapter.spec)

    key = jax.random.PRNGKey(0)
    key, reset_key, action_key, step_key = jax.random.split(key, 4)

    # Single-environment reset and observation construction.
    env_state, features = adapter.reset(reset_key)
    common_local_states = features.local_states
    actor_obs = adapter.build_actor_observations(features, common_local_states)
    critic_obs = adapter.build_critic_observations(features)

    assert actor_obs.shape == (
        adapter.spec.num_agents,
        adapter.spec.actor_obs_dim,
    )
    assert critic_obs.shape == (adapter.spec.critic_obs_dim,)

    policy_actions = jax.random.uniform(
        action_key,
        shape=(adapter.spec.num_agents, adapter.spec.policy_action_dim),
        minval=-1.0,
        maxval=1.0,
    )
    step_output = adapter.step(step_key, env_state, policy_actions)

    assert step_output.rewards.shape == (adapter.spec.num_agents,)
    assert step_output.dones.shape == (adapter.spec.num_agents,)
    assert step_output.features.local_states.shape == (
        adapter.spec.num_agents,
        adapter.spec.local_state_dim,
    )
    assert jnp.all(jnp.isfinite(step_output.rewards))

    # Privacy check: change the true remote states while keeping the common state
    # and agent 0's private/public information fixed. Agent 0's actor input must
    # not change.
    modified_local_states = features.local_states.at[1:].add(123.0)
    modified_features = features.replace(local_states=modified_local_states)
    actor_obs_modified = adapter.build_actor_observations(
        modified_features,
        common_local_states,
    )
    privacy_error = jnp.max(jnp.abs(actor_obs[0] - actor_obs_modified[0]))
    assert float(privacy_error) == 0.0, (
        "True remote states leaked into agent 0 actor observation: "
        f"max difference={float(privacy_error)}"
    )

    # JIT reset/step.
    jitted_reset = jax.jit(adapter.reset)
    jitted_step = jax.jit(adapter.step)
    jit_state, jit_features = jitted_reset(reset_key)
    _ = jitted_step(step_key, jit_state, policy_actions)

    # VMAP reset and observation construction across independent environments.
    reset_keys = jax.random.split(key, args.num_envs)
    batched_states, batched_features = jax.vmap(adapter.reset)(reset_keys)
    batched_actor_obs = jax.vmap(adapter.build_actor_observations)(
        batched_features,
        batched_features.local_states,
    )
    assert batched_actor_obs.shape == (
        args.num_envs,
        adapter.spec.num_agents,
        adapter.spec.actor_obs_dim,
    )

    print("Single actor_obs shape:", actor_obs.shape)
    print("Single critic_obs shape:", critic_obs.shape)
    print("Batched actor_obs shape:", batched_actor_obs.shape)
    print("Rewards:", step_output.rewards)
    print("Privacy check max difference:", float(privacy_error))
    print("JIT reset local-state shape:", jit_features.local_states.shape)
    print("All environment adapter smoke tests passed.")


if __name__ == "__main__":
    main()
