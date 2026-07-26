from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from sender_marl.envs import make_env_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate random-policy collision rates in Simple Spread."
    )
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--collision-distance", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        max_steps=args.max_steps,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
        collision_distance=args.collision_distance,
    )

    def run_episode(key: jax.Array):
        reset_key, rollout_key = jax.random.split(key)
        env_state, features = adapter.reset(reset_key)
        initial_metrics = adapter.compute_collision_metrics(
            features.local_states
        )

        def step_fn(carry, _unused):
            state, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            actions = jax.random.uniform(
                action_key,
                shape=(
                    adapter.spec.num_agents,
                    adapter.spec.policy_action_dim,
                ),
                minval=-1.0,
                maxval=1.0,
            )
            output = adapter.step(step_key, state, actions)
            metrics = output.metrics
            record = (
                metrics["collision_count"],
                metrics["pair_collision_rate"],
                metrics["any_collision"],
                metrics["min_pair_distance"],
            )
            return (output.env_state, key), record

        (_final_state, _final_key), records = jax.lax.scan(
            step_fn,
            (env_state, rollout_key),
            xs=None,
            length=args.max_steps,
        )
        return (
            records,
            initial_metrics["pair_collision_rate"],
            initial_metrics["any_collision"],
            initial_metrics["min_pair_distance"],
        )

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_episodes)
    (
        records,
        initial_pair_rate,
        initial_any_collision,
        initial_min_distance,
    ) = jax.jit(jax.vmap(run_episode))(keys)
    collision_count, pair_rate, any_collision, min_distance = records

    print(f"Episodes: {args.num_episodes}")
    print(f"Steps per episode: {args.max_steps}")
    print(f"Collision distance: {args.collision_distance:.3f}")
    print(
        "Initial pair collision rate:",
        f"{float(jnp.mean(initial_pair_rate)):.6f}",
    )
    print(
        "Initial states with any collision:",
        f"{float(jnp.mean(initial_any_collision)):.6f}",
    )
    print(
        "Mean colliding pairs per transition step:",
        f"{float(jnp.mean(collision_count)):.6f}",
    )
    print(
        "Mean pair-step collision rate:",
        f"{float(jnp.mean(pair_rate)):.6f}",
    )
    episode_collision_rate = jnp.mean(
        initial_any_collision | jnp.any(any_collision, axis=1)
    )
    minimum_observed_distance = jnp.minimum(
        jnp.min(initial_min_distance),
        jnp.min(min_distance),
    )

    print(
        "Episodes with at least one collision:",
        f"{float(episode_collision_rate):.6f}",
    )
    print(
        "Minimum observed pair distance:",
        f"{float(minimum_observed_distance):.6f}",
    )


if __name__ == "__main__":
    main()
