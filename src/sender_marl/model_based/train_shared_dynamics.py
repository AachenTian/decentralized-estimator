from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sender_marl.envs import make_env_adapter
from sender_marl.model_based.collector import make_random_dynamics_collector
from sender_marl.model_based.dynamics_state import (
    DynamicsStandardizer,
    SharedDynamicsConfig,
    create_shared_dynamics_train_state,
)
from sender_marl.model_based.dynamics_update import (
    evaluate_shared_dynamics_batch,
    make_shared_dynamics_train_step,
)
from sender_marl.model_based.replay_buffer import PooledLocalDynamicsBuffer


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _append_rollout(
    rollout,
    buffer: PooledLocalDynamicsBuffer,
    adapter,
) -> None:
    """Append valid transitions with per-agent collision labels.

    Episode-boundary transitions are removed because JaxMARL may auto-reset
    and return the next episode's initial state. Collision involvement is
    computed from the current local state x_t and is used only for validation.
    """

    batch = jax.device_get(rollout.batch)

    local_states = np.asarray(batch.local_states, dtype=np.float32)
    actions = np.asarray(batch.actions, dtype=np.float32)
    next_local_states = np.asarray(
        batch.next_local_states, dtype=np.float32
    )
    episode_dones = np.asarray(batch.episode_dones, dtype=bool)

    time_steps, num_envs, num_agents, state_dim = local_states.shape
    action_dim = actions.shape[-1]
    expected_collision_shape = (time_steps, num_envs, num_agents)

    collision_metrics = adapter.compute_collision_metrics(
        jnp.asarray(local_states)
    )
    agent_collision_mask = np.asarray(
        jax.device_get(collision_metrics["agent_collision_mask"]),
        dtype=np.bool_,
    )
    if agent_collision_mask.shape != expected_collision_shape:
        raise ValueError(
            "Unexpected agent collision mask shape: "
            f"{agent_collision_mask.shape}; expected "
            f"{expected_collision_shape}."
        )

    valid_joint_transitions = np.logical_not(episode_dones)
    valid_local_transitions = np.broadcast_to(
        valid_joint_transitions[..., None],
        expected_collision_shape,
    ).reshape(-1)

    flat_local_states = local_states.reshape(-1, state_dim)
    flat_actions = actions.reshape(-1, action_dim)
    flat_next_local_states = next_local_states.reshape(-1, state_dim)
    flat_collision_flags = agent_collision_mask.reshape(-1)
    agent_ids = np.broadcast_to(
        np.arange(num_agents, dtype=np.int32)[None, None, :],
        expected_collision_shape,
    ).reshape(-1)

    if not np.any(valid_local_transitions):
        return

    buffer.add_batch(
        local_states=flat_local_states[valid_local_transitions],
        actions=flat_actions[valid_local_transitions],
        next_local_states=flat_next_local_states[valid_local_transitions],
        agent_ids=agent_ids[valid_local_transitions],
        collision_flags=flat_collision_flags[valid_local_transitions],
    )


def _collect_dataset(
    *,
    adapter,
    key,
    num_envs: int,
    rollout_length: int,
    requested_environment_steps: int,
    capacity: int,
):
    buffer = PooledLocalDynamicsBuffer(
        capacity=capacity,
        local_state_dim=adapter.spec.local_state_dim,
        action_dim=adapter.spec.policy_action_dim,
    )
    key, reset_key, collect_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, num_envs)
    env_state, features = jax.vmap(adapter.reset)(reset_keys)
    collector = make_random_dynamics_collector(
        adapter,
        num_envs=num_envs,
        rollout_length=rollout_length,
        jit=True,
    )
    joint_steps_per_call = num_envs * rollout_length
    calls = math.ceil(requested_environment_steps / joint_steps_per_call)
    for _ in range(calls):
        collect_key, this_key = jax.random.split(collect_key)
        rollout = collector(this_key, env_state, features)
        _block_until_ready(rollout.batch)
        env_state = rollout.env_state
        features = rollout.features
        _append_rollout(rollout, buffer, adapter)
    return buffer, key


def _rmse_from_sse(sse: float, count: int, dimensions: int) -> float:
    if count <= 0:
        return float("nan")
    return float(np.sqrt(sse / (count * dimensions)))


def _evaluate_dataset(
    train_state,
    standardizer,
    data,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Evaluate aggregate, per-agent, and collision-conditioned one-step error."""

    from sender_marl.model_based.dynamics_update import predict_shared_dynamics

    size = int(data.local_states.shape[0])
    sums: dict[str, float] = {}
    per_dimension_squared_sum = None
    per_agent_squared_sum: dict[int, float] = {}
    per_agent_count: dict[int, int] = {}

    collision_count = 0
    noncollision_count = 0
    collision_state_sse = 0.0
    noncollision_state_sse = 0.0
    collision_position_sse = 0.0
    noncollision_position_sse = 0.0
    collision_velocity_sse = 0.0
    noncollision_velocity_sse = 0.0

    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        chunk = data.replace(
            local_states=data.local_states[start:end],
            actions=data.actions[start:end],
            next_local_states=data.next_local_states[start:end],
            agent_ids=data.agent_ids[start:end],
            collision_flags=data.collision_flags[start:end],
        )
        metrics = jax.device_get(
            evaluate_shared_dynamics_batch(
                train_state,
                standardizer,
                chunk,
            )
        )
        weight = end - start
        for key in [
            "state_mae",
            "mixture_nll",
            "mean_aleatoric_std",
            "mean_epistemic_std",
            "interval_coverage_95",
        ]:
            sums[key] = sums.get(key, 0.0) + float(metrics[key]) * weight

        prediction = jax.device_get(
            predict_shared_dynamics(
                train_state,
                standardizer,
                chunk.local_states,
                chunk.actions,
            ).next_mean
        )
        targets = np.asarray(jax.device_get(chunk.next_local_states))
        errors = np.asarray(prediction, dtype=np.float64) - np.asarray(
            targets, dtype=np.float64
        )
        squared_errors = np.square(errors)

        dimension_sse = np.sum(squared_errors, axis=0)
        if per_dimension_squared_sum is None:
            per_dimension_squared_sum = dimension_sse
        else:
            per_dimension_squared_sum += dimension_sse

        agent_ids = np.asarray(jax.device_get(chunk.agent_ids))
        for agent_id in np.unique(agent_ids):
            mask = agent_ids == agent_id
            per_agent_squared_sum[int(agent_id)] = (
                per_agent_squared_sum.get(int(agent_id), 0.0)
                + float(np.sum(squared_errors[mask]))
            )
            per_agent_count[int(agent_id)] = (
                per_agent_count.get(int(agent_id), 0)
                + int(np.sum(mask)) * errors.shape[1]
            )

        collision = np.asarray(
            jax.device_get(chunk.collision_flags), dtype=bool
        )
        noncollision = np.logical_not(collision)
        local_collision_count = int(np.sum(collision))
        local_noncollision_count = int(np.sum(noncollision))
        collision_count += local_collision_count
        noncollision_count += local_noncollision_count

        if local_collision_count:
            collision_state_sse += float(np.sum(squared_errors[collision]))
            collision_position_sse += float(
                np.sum(squared_errors[collision, :2])
            )
            collision_velocity_sse += float(
                np.sum(squared_errors[collision, 2:4])
            )
        if local_noncollision_count:
            noncollision_state_sse += float(
                np.sum(squared_errors[noncollision])
            )
            noncollision_position_sse += float(
                np.sum(squared_errors[noncollision, :2])
            )
            noncollision_velocity_sse += float(
                np.sum(squared_errors[noncollision, 2:4])
            )

    assert per_dimension_squared_sum is not None
    state_dim = int(len(per_dimension_squared_sum))
    per_dimension_rmse = np.sqrt(per_dimension_squared_sum / size)
    result: dict[str, Any] = {
        key: value / size for key, value in sums.items()
    }
    result["state_rmse"] = float(
        np.sqrt(np.sum(per_dimension_squared_sum) / (size * state_dim))
    )
    result["per_dimension_rmse"] = per_dimension_rmse.tolist()
    if state_dim == 4:
        result["position_rmse"] = float(
            np.sqrt(np.mean(np.square(per_dimension_rmse[:2])))
        )
        result["velocity_rmse"] = float(
            np.sqrt(np.mean(np.square(per_dimension_rmse[2:])))
        )

    for agent_id in sorted(per_agent_squared_sum):
        result[f"agent{agent_id}_state_rmse"] = float(
            np.sqrt(
                per_agent_squared_sum[agent_id]
                / per_agent_count[agent_id]
            )
        )

    result.update(
        {
            "collision_local_rate": float(collision_count / size),
            "collision_count": collision_count,
            "noncollision_count": noncollision_count,
            "collision_state_rmse": _rmse_from_sse(
                collision_state_sse, collision_count, state_dim
            ),
            "noncollision_state_rmse": _rmse_from_sse(
                noncollision_state_sse, noncollision_count, state_dim
            ),
            "collision_position_rmse": _rmse_from_sse(
                collision_position_sse, collision_count, 2
            ),
            "noncollision_position_rmse": _rmse_from_sse(
                noncollision_position_sse, noncollision_count, 2
            ),
            "collision_velocity_rmse": _rmse_from_sse(
                collision_velocity_sse, collision_count, 2
            ),
            "noncollision_velocity_rmse": _rmse_from_sse(
                noncollision_velocity_sse, noncollision_count, 2
            ),
        }
    )
    return result


def _save_checkpoint(
    path: Path,
    *,
    train_state,
    standardizer,
    config,
    env_spec,
    update: int,
    validation_metrics: dict[str, Any],
) -> None:
    payload = {
        "algorithm": "shared_factorized_probabilistic_dynamics_ensemble",
        "factorization": "p(x_next|x,a) ~= product_i p(x_i_next|x_i,a_i)",
        "parameter_sharing": True,
        "pooled_homogeneous_agent_data": True,
        "update": update,
        "params": jax.device_get(train_state.params),
        "standardizer": standardizer.to_serializable(),
        "dynamics_config": asdict(config),
        "env_spec": asdict(env_spec),
        "validation_metrics": validation_metrics,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared probabilistic local dynamics ensemble by pooling "
            "factorized transitions from all homogeneous agents."
        )
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=8)
    parser.add_argument("--train-env-steps", type=int, default=20_000)
    parser.add_argument("--validation-env-steps", type=int, default=5_000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--min-logvar", type=float, default=-10.0)
    parser.add_argument("--max-logvar", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/shared_factorized_dynamics_seed0"),
    )
    args = parser.parse_args()

    integer_args = {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "train_env_steps": args.train_env_steps,
        "validation_env_steps": args.validation_env_steps,
        "model_updates": args.model_updates,
        "batch_size": args.batch_size,
        "ensemble_size": args.ensemble_size,
        "eval_interval": args.eval_interval,
        "eval_chunk_size": args.eval_chunk_size,
    }
    invalid = {name: value for name, value in integer_args.items() if value < 1}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}.")

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
    print("Dynamics input: local state + local action")
    print("Dynamics target: next local state - local state")
    print("Dynamics parameter sharing: enabled")
    print("Agent identity input: disabled")
    print("Collection policy: uniform random actions")
    print("Terminal/reset transitions: excluded")
    print("Collision labels: per-agent involvement at current state x_t")
    print("Collision labels used as model input: disabled")

    train_capacity = (
        math.ceil(args.train_env_steps / (args.num_envs * args.rollout_length))
        * args.num_envs
        * args.rollout_length
        * spec.num_agents
    )
    validation_capacity = (
        math.ceil(args.validation_env_steps / (args.num_envs * args.rollout_length))
        * args.num_envs
        * args.rollout_length
        * spec.num_agents
    )

    key = jax.random.PRNGKey(args.seed)
    key, train_collect_key, validation_collect_key, init_key, update_key = (
        jax.random.split(key, 5)
    )
    print("Collecting training transitions...")
    train_buffer, _ = _collect_dataset(
        adapter=adapter,
        key=train_collect_key,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        requested_environment_steps=args.train_env_steps,
        capacity=train_capacity,
    )
    print("Collecting independent validation transitions...")
    validation_buffer, _ = _collect_dataset(
        adapter=adapter,
        key=validation_collect_key,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        requested_environment_steps=args.validation_env_steps,
        capacity=validation_capacity,
    )
    train_data = train_buffer.all_data()
    validation_data = validation_buffer.all_data()
    standardizer = DynamicsStandardizer.fit(
        jax.device_get(train_data.local_states),
        jax.device_get(train_data.actions),
        jax.device_get(train_data.next_local_states),
    )
    config = SharedDynamicsConfig(
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        max_gradient_norm=args.max_grad_norm,
        min_logvar=args.min_logvar,
        max_logvar=args.max_logvar,
    )
    _, train_state = create_shared_dynamics_train_state(
        init_key,
        local_state_dim=spec.local_state_dim,
        action_dim=spec.policy_action_dim,
        config=config,
    )
    train_step = make_shared_dynamics_train_step(
        ensemble_size=args.ensemble_size,
        jit=True,
    )
    replay_rng = np.random.default_rng(args.seed + 30_000)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "algorithm": "shared_factorized_probabilistic_dynamics_ensemble",
                "arguments": vars(args) | {"output_dir": str(args.output_dir)},
                "dynamics": asdict(config),
                "env_spec": asdict(spec),
                "training_local_transitions": len(train_buffer),
                "validation_local_transitions": len(validation_buffer),
                "terminal_reset_transitions_excluded": True,
                "collision_label_semantics": (
                    "per-agent collision involvement at current state x_t"
                ),
                "collision_label_used_as_model_input": False,
            },
            handle,
            indent=2,
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    best_validation_rmse = float("inf")
    best_update = -1

    for update in range(1, args.model_updates + 1):
        batch = train_buffer.sample(args.batch_size, replay_rng)
        update_key, this_key = jax.random.split(update_key)
        train_state, train_metrics = train_step(
            this_key,
            train_state,
            standardizer,
            batch,
        )
        _block_until_ready(train_metrics)
        do_evaluate = (
            update == 1
            or update % args.eval_interval == 0
            or update == args.model_updates
        )
        if not do_evaluate:
            continue

        validation_metrics = _evaluate_dataset(
            train_state,
            standardizer,
            validation_data,
            chunk_size=args.eval_chunk_size,
        )
        metrics = {
            "update": update,
            **{
                name: float(jax.device_get(value))
                for name, value in train_metrics.items()
            },
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            },
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(
            f"update={update:05d} "
            f"nll={metrics['dynamics_nll']:.4f} "
            f"val_rmse={metrics['validation_state_rmse']:.6f} "
            f"pos_rmse={metrics.get('validation_position_rmse', float('nan')):.6f} "
            f"vel_rmse={metrics.get('validation_velocity_rmse', float('nan')):.6f} "
            f"epi_std={metrics['validation_mean_epistemic_std']:.6f} "
            f"coverage95={metrics['validation_interval_coverage_95']:.3f} "
            f"collision_rate={metrics['validation_collision_local_rate']:.4f} "
            f"noncol_vel={metrics['validation_noncollision_velocity_rmse']:.6f} "
            f"col_vel={metrics['validation_collision_velocity_rmse']:.6f}"
        )
        if validation_metrics["state_rmse"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["state_rmse"]
            best_update = update
            _save_checkpoint(
                args.output_dir / "best.pkl",
                train_state=train_state,
                standardizer=standardizer,
                config=config,
                env_spec=spec,
                update=update,
                validation_metrics=validation_metrics,
            )
        _save_checkpoint(
            args.output_dir / "latest.pkl",
            train_state=train_state,
            standardizer=standardizer,
            config=config,
            env_spec=spec,
            update=update,
            validation_metrics=validation_metrics,
        )

    print("Training complete.")
    print("Results:", args.output_dir)
    print("Metrics:", metrics_path)
    print("Latest checkpoint:", args.output_dir / "latest.pkl")
    print("Best checkpoint:", args.output_dir / "best.pkl")
    print(
        "Best validation state RMSE:",
        f"{best_validation_rmse:.6f}",
        "at update",
        best_update,
    )


if __name__ == "__main__":
    main()
