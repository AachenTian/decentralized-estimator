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
from sender_marl.model_based.joint_collector import (
    make_random_joint_dynamics_collector,
)
from sender_marl.model_based.joint_dynamics_state import (
    IndependentJointDynamicsConfig,
    JointDynamicsStandardizer,
    create_independent_joint_dynamics_train_states,
)
from sender_marl.model_based.joint_dynamics_update import (
    evaluate_independent_joint_dynamics_batch,
    make_independent_joint_dynamics_train_step,
    predict_independent_joint_dynamics,
)
from sender_marl.model_based.joint_replay_buffer import (
    JointDynamicsReplayBuffer,
)


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _append_rollout(
    rollout,
    buffer: JointDynamicsReplayBuffer,
    adapter,
) -> None:
    """Append valid joint transitions with per-agent collision labels."""

    batch = jax.device_get(rollout.batch)
    model_states = np.asarray(batch.model_states, dtype=np.float32)
    joint_actions = np.asarray(batch.joint_actions, dtype=np.float32)
    local_states = np.asarray(batch.local_states, dtype=np.float32)
    next_local_states = np.asarray(
        batch.next_local_states, dtype=np.float32
    )
    episode_dones = np.asarray(batch.episode_dones, dtype=bool)

    time_steps, num_envs, num_agents, local_state_dim = local_states.shape
    model_state_dim = model_states.shape[-1]
    action_dim = joint_actions.shape[-1]

    collision_metrics = adapter.compute_collision_metrics(
        jnp.asarray(local_states)
    )
    collision_flags = np.asarray(
        jax.device_get(collision_metrics["agent_collision_mask"]),
        dtype=bool,
    )
    expected_collision_shape = (time_steps, num_envs, num_agents)
    if collision_flags.shape != expected_collision_shape:
        raise ValueError(
            "Unexpected collision mask shape "
            f"{collision_flags.shape}; expected {expected_collision_shape}."
        )

    # Exclude terminal rows because the collector resets the environment before
    # continuing. The returned next state at a boundary is not a physical
    # transition target for the dynamics model.
    valid = np.logical_not(episode_dones).reshape(-1)
    if not np.any(valid):
        return

    buffer.add_batch(
        model_states=model_states.reshape(-1, model_state_dim)[valid],
        joint_actions=joint_actions.reshape(
            -1, num_agents, action_dim
        )[valid],
        local_states=local_states.reshape(
            -1, num_agents, local_state_dim
        )[valid],
        next_local_states=next_local_states.reshape(
            -1, num_agents, local_state_dim
        )[valid],
        agent_collision_flags=collision_flags.reshape(
            -1, num_agents
        )[valid],
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
    buffer = JointDynamicsReplayBuffer(
        capacity=capacity,
        model_state_dim=adapter.spec.model_state_dim,
        num_agents=adapter.spec.num_agents,
        action_dim=adapter.spec.policy_action_dim,
        local_state_dim=adapter.spec.local_state_dim,
    )
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_envs)
    env_state, features = jax.vmap(adapter.reset)(reset_keys)
    collector = make_random_joint_dynamics_collector(
        adapter,
        num_envs=num_envs,
        rollout_length=rollout_length,
        jit=True,
    )
    collected = 0
    while collected < requested_environment_steps:
        key, collect_key = jax.random.split(key)
        rollout = collector(collect_key, env_state, features)
        _block_until_ready(rollout.batch)
        env_state = rollout.env_state
        features = rollout.features
        _append_rollout(rollout, buffer, adapter)
        collected += num_envs * rollout_length
    return buffer, key


def _rmse_from_sse(sse: float, count: int, dimensions: int) -> float:
    if count <= 0:
        return float("nan")
    return float(np.sqrt(sse / (count * dimensions)))


def _evaluate_dataset(
    train_states,
    standardizers,
    data,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Evaluate all independently owned joint-conditioned models."""

    num_agents = int(data.local_states.shape[1])
    state_dim = int(data.local_states.shape[2])
    size = int(data.model_states.shape[0])

    global_state_sse = 0.0
    global_dimension_sse = np.zeros((state_dim,), dtype=np.float64)
    global_collision_state_sse = 0.0
    global_noncollision_state_sse = 0.0
    global_collision_position_sse = 0.0
    global_noncollision_position_sse = 0.0
    global_collision_velocity_sse = 0.0
    global_noncollision_velocity_sse = 0.0
    global_collision_count = 0
    global_noncollision_count = 0

    weighted_metrics = {
        "state_mae": 0.0,
        "mixture_nll": 0.0,
        "mean_aleatoric_std": 0.0,
        "mean_epistemic_std": 0.0,
        "interval_coverage_95": 0.0,
    }
    per_agent: dict[int, dict[str, float]] = {}

    for agent_id in range(num_agents):
        agent_state_sse = 0.0
        agent_dimension_sse = np.zeros((state_dim,), dtype=np.float64)
        agent_collision_state_sse = 0.0
        agent_noncollision_state_sse = 0.0
        agent_collision_position_sse = 0.0
        agent_noncollision_position_sse = 0.0
        agent_collision_velocity_sse = 0.0
        agent_noncollision_velocity_sse = 0.0
        agent_collision_count = 0
        agent_noncollision_count = 0
        agent_weighted = {key: 0.0 for key in weighted_metrics}

        for start in range(0, size, chunk_size):
            end = min(start + chunk_size, size)
            chunk = data.replace(
                model_states=data.model_states[start:end],
                joint_actions=data.joint_actions[start:end],
                local_states=data.local_states[start:end],
                next_local_states=data.next_local_states[start:end],
                agent_collision_flags=data.agent_collision_flags[start:end],
            )
            metrics = jax.device_get(
                evaluate_independent_joint_dynamics_batch(
                    train_states[agent_id],
                    standardizers[agent_id],
                    chunk,
                    agent_id=agent_id,
                )
            )
            weight = end - start
            for key in agent_weighted:
                agent_weighted[key] += float(metrics[key]) * weight

            prediction = jax.device_get(
                predict_independent_joint_dynamics(
                    train_states[agent_id],
                    standardizers[agent_id],
                    chunk.model_states,
                    chunk.joint_actions,
                    chunk.local_states[:, agent_id, :],
                ).next_mean
            )
            targets = np.asarray(
                jax.device_get(chunk.next_local_states[:, agent_id, :])
            )
            errors = np.asarray(prediction, dtype=np.float64) - np.asarray(
                targets, dtype=np.float64
            )
            squared = np.square(errors)
            agent_state_sse += float(np.sum(squared))
            agent_dimension_sse += np.sum(squared, axis=0)

            collision = np.asarray(
                jax.device_get(
                    chunk.agent_collision_flags[:, agent_id]
                ),
                dtype=bool,
            )
            noncollision = ~collision
            collision_count = int(np.sum(collision))
            noncollision_count = int(np.sum(noncollision))
            agent_collision_count += collision_count
            agent_noncollision_count += noncollision_count

            if collision_count:
                agent_collision_state_sse += float(
                    np.sum(squared[collision])
                )
                agent_collision_position_sse += float(
                    np.sum(squared[collision, :2])
                )
                agent_collision_velocity_sse += float(
                    np.sum(squared[collision, 2:4])
                )
            if noncollision_count:
                agent_noncollision_state_sse += float(
                    np.sum(squared[noncollision])
                )
                agent_noncollision_position_sse += float(
                    np.sum(squared[noncollision, :2])
                )
                agent_noncollision_velocity_sse += float(
                    np.sum(squared[noncollision, 2:4])
                )

        dimension_rmse = np.sqrt(agent_dimension_sse / size)
        agent_result = {
            "state_rmse": _rmse_from_sse(
                agent_state_sse, size, state_dim
            ),
            "position_rmse": _rmse_from_sse(
                float(np.sum(agent_dimension_sse[:2])), size, 2
            ),
            "velocity_rmse": _rmse_from_sse(
                float(np.sum(agent_dimension_sse[2:4])), size, 2
            ),
            "collision_local_rate": float(agent_collision_count / size),
            "collision_state_rmse": _rmse_from_sse(
                agent_collision_state_sse,
                agent_collision_count,
                state_dim,
            ),
            "noncollision_state_rmse": _rmse_from_sse(
                agent_noncollision_state_sse,
                agent_noncollision_count,
                state_dim,
            ),
            "collision_position_rmse": _rmse_from_sse(
                agent_collision_position_sse,
                agent_collision_count,
                2,
            ),
            "noncollision_position_rmse": _rmse_from_sse(
                agent_noncollision_position_sse,
                agent_noncollision_count,
                2,
            ),
            "collision_velocity_rmse": _rmse_from_sse(
                agent_collision_velocity_sse,
                agent_collision_count,
                2,
            ),
            "noncollision_velocity_rmse": _rmse_from_sse(
                agent_noncollision_velocity_sse,
                agent_noncollision_count,
                2,
            ),
            "per_dimension_rmse": dimension_rmse.tolist(),
            **{key: value / size for key, value in agent_weighted.items()},
        }
        per_agent[agent_id] = agent_result

        global_state_sse += agent_state_sse
        global_dimension_sse += agent_dimension_sse
        global_collision_state_sse += agent_collision_state_sse
        global_noncollision_state_sse += agent_noncollision_state_sse
        global_collision_position_sse += agent_collision_position_sse
        global_noncollision_position_sse += agent_noncollision_position_sse
        global_collision_velocity_sse += agent_collision_velocity_sse
        global_noncollision_velocity_sse += agent_noncollision_velocity_sse
        global_collision_count += agent_collision_count
        global_noncollision_count += agent_noncollision_count
        for key in weighted_metrics:
            weighted_metrics[key] += agent_weighted[key]

    local_count = size * num_agents
    result: dict[str, Any] = {
        "state_rmse": _rmse_from_sse(
            global_state_sse, local_count, state_dim
        ),
        "position_rmse": _rmse_from_sse(
            float(np.sum(global_dimension_sse[:2])), local_count, 2
        ),
        "velocity_rmse": _rmse_from_sse(
            float(np.sum(global_dimension_sse[2:4])), local_count, 2
        ),
        "per_dimension_rmse": np.sqrt(
            global_dimension_sse / local_count
        ).tolist(),
        "collision_local_rate": float(global_collision_count / local_count),
        "collision_count": global_collision_count,
        "noncollision_count": global_noncollision_count,
        "collision_state_rmse": _rmse_from_sse(
            global_collision_state_sse,
            global_collision_count,
            state_dim,
        ),
        "noncollision_state_rmse": _rmse_from_sse(
            global_noncollision_state_sse,
            global_noncollision_count,
            state_dim,
        ),
        "collision_position_rmse": _rmse_from_sse(
            global_collision_position_sse,
            global_collision_count,
            2,
        ),
        "noncollision_position_rmse": _rmse_from_sse(
            global_noncollision_position_sse,
            global_noncollision_count,
            2,
        ),
        "collision_velocity_rmse": _rmse_from_sse(
            global_collision_velocity_sse,
            global_collision_count,
            2,
        ),
        "noncollision_velocity_rmse": _rmse_from_sse(
            global_noncollision_velocity_sse,
            global_noncollision_count,
            2,
        ),
    }
    for key, value in weighted_metrics.items():
        result[key] = value / local_count
    for agent_id, metrics in per_agent.items():
        for key, value in metrics.items():
            result[f"agent{agent_id}_{key}"] = value
    return result


def _save_checkpoint(
    path: Path,
    *,
    train_states,
    standardizers,
    config,
    env_spec,
    update: int,
    validation_metrics: dict[str, Any],
    collision_batch_fraction: float | None,
) -> None:
    payload = {
        "algorithm": "independent_joint_conditioned_probabilistic_dynamics",
        "model_ownership": "one independent ensemble per agent",
        "input": "full model state + joint policy action",
        "output": "owned agent local delta state",
        "parameter_sharing": False,
        "rollout_snapshot_protocol": (
            "before local model rollout, each agent receives frozen actor and "
            "dynamics snapshots of the other agents"
        ),
        "update": update,
        "params": [jax.device_get(state.params) for state in train_states],
        "standardizers": [
            standardizer.to_serializable()
            for standardizer in standardizers
        ],
        "dynamics_config": asdict(config),
        "env_spec": asdict(env_spec),
        "collision_batch_fraction": collision_batch_fraction,
        "validation_metrics": validation_metrics,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one independent probabilistic dynamics ensemble per agent. "
            "Every model receives the full state and joint action and predicts "
            "only its owned agent's next local state."
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
    parser.add_argument("--model-updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--min-logvar", type=float, default=-10.0)
    parser.add_argument("--max-logvar", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--collision-batch-fraction",
        type=float,
        default=-1.0,
        help=(
            "Negative means uniform replay. A value in [0,1] requests that "
            "fraction of every agent-specific minibatch from transitions where "
            "that agent participates in a collision. Use uniform replay first "
            "for the direct local-input versus joint-input comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/independent_joint_dynamics_seed0"),
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
    if args.collision_batch_fraction > 1.0:
        raise ValueError("collision-batch-fraction must be <= 1.0.")
    collision_batch_fraction = (
        None
        if args.collision_batch_fraction < 0.0
        else float(args.collision_batch_fraction)
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
    print("EnvSpec:", spec)
    print("Dynamics ownership: one independent model per agent")
    print("Dynamics input: full model state + joint action")
    print("Dynamics output: owned agent local delta state")
    print("Dynamics parameter sharing: disabled")
    print("Collection policy: uniform random actions")
    print("Terminal/reset transitions: excluded")
    print("Collision labels: per-agent involvement at current state x_t")
    print("Collision labels used as model input: disabled")
    print(
        "Training replay sampling:",
        "uniform"
        if collision_batch_fraction is None
        else f"collision-balanced fraction={collision_batch_fraction:.3f}",
    )
    print(
        "Future rollout protocol: exchange frozen actor and dynamics snapshots "
        "before each agent independently simulates a joint trajectory"
    )

    train_capacity = (
        math.ceil(
            args.train_env_steps / (args.num_envs * args.rollout_length)
        )
        * args.num_envs
        * args.rollout_length
    )
    validation_capacity = (
        math.ceil(
            args.validation_env_steps
            / (args.num_envs * args.rollout_length)
        )
        * args.num_envs
        * args.rollout_length
    )

    key = jax.random.PRNGKey(args.seed)
    (
        key,
        train_collect_key,
        validation_collect_key,
        init_key,
        update_key,
    ) = jax.random.split(key, 5)
    print("Collecting training joint transitions...")
    train_buffer, _ = _collect_dataset(
        adapter=adapter,
        key=train_collect_key,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        requested_environment_steps=args.train_env_steps,
        capacity=train_capacity,
    )
    print("Collecting independent validation joint transitions...")
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

    standardizers = []
    for agent_id in range(spec.num_agents):
        standardizers.append(
            JointDynamicsStandardizer.fit(
                jax.device_get(train_data.model_states),
                jax.device_get(train_data.joint_actions),
                jax.device_get(train_data.local_states[:, agent_id, :]),
                jax.device_get(
                    train_data.next_local_states[:, agent_id, :]
                ),
            )
        )

    config = IndependentJointDynamicsConfig(
        ensemble_size=args.ensemble_size,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        max_gradient_norm=args.max_grad_norm,
        min_logvar=args.min_logvar,
        max_logvar=args.max_logvar,
    )
    _, train_states = create_independent_joint_dynamics_train_states(
        init_key,
        num_agents=spec.num_agents,
        model_state_dim=spec.model_state_dim,
        action_dim=spec.policy_action_dim,
        local_state_dim=spec.local_state_dim,
        config=config,
    )
    train_steps = [
        make_independent_joint_dynamics_train_step(
            ensemble_size=args.ensemble_size,
            agent_id=agent_id,
            jit=True,
        )
        for agent_id in range(spec.num_agents)
    ]
    replay_rngs = [
        np.random.default_rng(args.seed + 30_000 + agent_id)
        for agent_id in range(spec.num_agents)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            {
                "algorithm": (
                    "independent_joint_conditioned_probabilistic_dynamics"
                ),
                "arguments": vars(args)
                | {"output_dir": str(args.output_dir)},
                "dynamics": asdict(config),
                "env_spec": asdict(spec),
                "training_joint_transitions": len(train_buffer),
                "validation_joint_transitions": len(validation_buffer),
                "terminal_reset_transitions_excluded": True,
                "parameter_sharing": False,
                "input": "full model state + joint action",
                "output": "owned agent local delta state",
                "collision_label_semantics": (
                    "per-agent collision involvement at current state x_t"
                ),
                "collision_label_used_as_model_input": False,
                "collision_batch_fraction": collision_batch_fraction,
                "rollout_snapshot_protocol": (
                    "exchange frozen actor and dynamics snapshots before each "
                    "agent independently simulates a joint trajectory"
                ),
            },
            handle,
            indent=2,
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    best_validation_rmse = float("inf")
    best_update = -1

    for update in range(1, args.model_updates + 1):
        next_states = []
        per_agent_train_metrics = []
        update_key, *agent_keys = jax.random.split(
            update_key, spec.num_agents + 1
        )
        for agent_id in range(spec.num_agents):
            batch = train_buffer.sample(
                args.batch_size,
                replay_rngs[agent_id],
                agent_id=agent_id,
                collision_fraction=collision_batch_fraction,
            )
            new_state, metrics = train_steps[agent_id](
                agent_keys[agent_id],
                train_states[agent_id],
                standardizers[agent_id],
                batch,
            )
            next_states.append(new_state)
            per_agent_train_metrics.append(metrics)
        train_states = next_states
        _block_until_ready(per_agent_train_metrics)

        do_evaluate = (
            update == 1
            or update % args.eval_interval == 0
            or update == args.model_updates
        )
        if not do_evaluate:
            continue

        validation_metrics = _evaluate_dataset(
            train_states,
            standardizers,
            validation_data,
            chunk_size=args.eval_chunk_size,
        )
        metrics: dict[str, Any] = {"update": update}
        for metric_name in per_agent_train_metrics[0]:
            values = [
                float(jax.device_get(agent_metrics[metric_name]))
                for agent_metrics in per_agent_train_metrics
            ]
            metrics[metric_name] = float(np.mean(values))
            for agent_id, value in enumerate(values):
                metrics[f"agent{agent_id}_{metric_name}"] = value
        metrics.update(
            {
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            }
        )
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        print(
            f"update={update:05d} "
            f"nll={metrics['dynamics_nll']:.4f} "
            f"val_rmse={metrics['validation_state_rmse']:.6f} "
            f"pos_rmse={metrics['validation_position_rmse']:.6f} "
            f"vel_rmse={metrics['validation_velocity_rmse']:.6f} "
            f"epi_std={metrics['validation_mean_epistemic_std']:.6f} "
            f"coverage95={metrics['validation_interval_coverage_95']:.3f} "
            f"collision_rate={metrics['validation_collision_local_rate']:.4f} "
            f"noncol_vel={metrics['validation_noncollision_velocity_rmse']:.6f} "
            f"col_vel={metrics['validation_collision_velocity_rmse']:.6f}"
        )
        print(
            "  per-agent collision velocity RMSE: "
            + " ".join(
                f"a{agent_id}="
                f"{metrics[f'validation_agent{agent_id}_collision_velocity_rmse']:.6f}"
                for agent_id in range(spec.num_agents)
            )
        )

        if validation_metrics["state_rmse"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["state_rmse"]
            best_update = update
            _save_checkpoint(
                args.output_dir / "best.pkl",
                train_states=train_states,
                standardizers=standardizers,
                config=config,
                env_spec=spec,
                update=update,
                validation_metrics=validation_metrics,
                collision_batch_fraction=collision_batch_fraction,
            )
        _save_checkpoint(
            args.output_dir / "latest.pkl",
            train_states=train_states,
            standardizers=standardizers,
            config=config,
            env_spec=spec,
            update=update,
            validation_metrics=validation_metrics,
            collision_batch_fraction=collision_batch_fraction,
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
