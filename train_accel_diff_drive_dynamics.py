
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from aorpo.agents.independent_dynamics import (
    init_independent_transition_models,
    init_local_standardizers,
    make_direct_local_transition_batch,
    predict_local_ensemble_next_gaussians,
    predict_local_next,
    train_independent_transition_step,
)
from aorpo.evaluation.dynamics_metrics import (
    gaussian_nll,
    marginal_coverage,
    one_step_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one probabilistic local ensemble per agent on the "
            "acceleration-controlled differential-drive dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_debug.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--max-eval-transitions", type=int, default=10000)

    parser.add_argument("--num-members", type=int, default=5)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-logvar", type=float, default=-10.0)
    parser.add_argument("--max-logvar", type=float, default=0.5)

    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50],
    )
    parser.add_argument("--num-rollout-trajectories", type=int, default=10)

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "checkpoints/independent_dynamics_accel_diff_drive.pkl"
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save using the repository's checkpoint utility.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        arrays = {
            "local_states": np.asarray(data["local_states"]),
            "actions": np.asarray(data["actions"]),
        }
        splits = {
            "train": np.asarray(data["train_trajectory_indices"]),
            "validation": np.asarray(
                data["validation_trajectory_indices"]
            ),
            "test": np.asarray(data["test_trajectory_indices"]),
        }
        metadata = json.loads(str(data["metadata_json"]))

    local_states = arrays["local_states"]
    actions = arrays["actions"]

    if local_states.ndim != 4:
        raise ValueError(
            "local_states must have shape (K, H+1, N, D), "
            f"got {local_states.shape}."
        )
    if actions.ndim != 4:
        raise ValueError(
            "actions must have shape (K, H, N, A), "
            f"got {actions.shape}."
        )
    if local_states.shape[0] != actions.shape[0]:
        raise ValueError("State/action trajectory counts do not match.")
    if local_states.shape[1] != actions.shape[1] + 1:
        raise ValueError(
            "Each trajectory must contain one more state than action."
        )
    if local_states.shape[2] != actions.shape[2]:
        raise ValueError("State/action agent counts do not match.")

    expected_state_dim = int(metadata["local_state_dim"])
    expected_action_dim = int(metadata["action_dim"])

    if local_states.shape[-1] != expected_state_dim:
        raise ValueError(
            f"Metadata says state dim {expected_state_dim}, "
            f"array has {local_states.shape[-1]}."
        )
    if actions.shape[-1] != expected_action_dim:
        raise ValueError(
            f"Metadata says action dim {expected_action_dim}, "
            f"array has {actions.shape[-1]}."
        )

    if metadata.get("state_order") != [
        "p_x",
        "p_y",
        "phi",
        "v",
        "omega",
    ]:
        raise ValueError(
            "Unexpected state order. Expected "
            "['p_x', 'p_y', 'phi', 'v', 'omega']."
        )
    if metadata.get("action_order") != ["a_t", "alpha_z"]:
        raise ValueError(
            "Unexpected action order. Expected ['a_t', 'alpha_z']."
        )
    if bool(metadata.get("wrap_heading", True)):
        raise ValueError(
            "This delta-model training script expects unwrapped headings."
        )

    return {
        "local_states": jnp.asarray(local_states),
        "actions": jnp.asarray(actions),
        "metadata": metadata,
    }, splits


def agent_transitions(
    local_states: jax.Array,
    actions: jax.Array,
    trajectory_indices: np.ndarray,
    agent_id: int,
) -> dict[str, jax.Array]:
    """Flatten selected trajectories for exactly one agent."""
    idx = jnp.asarray(trajectory_indices, dtype=jnp.int32)
    states = local_states[idx, :, agent_id, :]
    agent_actions = actions[idx, :, agent_id, :]

    state_t = states[:, :-1, :].reshape(-1, states.shape[-1])
    next_state = states[:, 1:, :].reshape(-1, states.shape[-1])
    action_t = agent_actions.reshape(-1, agent_actions.shape[-1])

    return make_direct_local_transition_batch(
        local_state=state_t,
        local_action=action_t,
        next_local_state=next_state,
    )


def make_model_config(args: argparse.Namespace) -> Any:
    """Create the small cfg object expected by the existing model code."""
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=int(args.num_members),
            hidden_dims=tuple(int(x) for x in args.hidden_dims),
            min_logvar=float(args.min_logvar),
            max_logvar=float(args.max_logvar),
            lr=float(args.learning_rate),
        )
    )


def subset_batch(
    batch: dict[str, jax.Array],
    indices: jax.Array,
) -> dict[str, jax.Array]:
    return {name: value[indices] for name, value in batch.items()}


def aggregate_physical_prediction(
    train_state: Any,
    standardizer: Any,
    local_state: jax.Array,
    local_action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Moment-match ensemble Gaussians in original state coordinates."""
    raw = predict_local_ensemble_next_gaussians(
        train_state=train_state,
        standardizer=standardizer,
        local_state=local_state,
        local_action=local_action,
    )
    member_means = raw.ensemble_next_means
    member_covariances = raw.ensemble_next_covariances

    mean = jnp.mean(member_means, axis=1)
    aleatoric = jnp.mean(member_covariances, axis=1)

    centered = member_means - mean[:, None, :]
    ensemble_size = int(member_means.shape[1])
    denominator = max(ensemble_size - 1, 1)
    epistemic = (
        jnp.einsum("bed,bef->bdf", centered, centered)
        / denominator
    )
    return mean, aleatoric + epistemic


def evaluate_one_step(
    model_states: list[Any],
    standardizers: list[Any],
    eval_batches: list[dict[str, jax.Array]],
    max_eval_transitions: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    all_state_rmse = []

    for agent_id, (model_state, standardizer, full_batch) in enumerate(
        zip(model_states, standardizers, eval_batches)
    ):
        num_examples = min(
            int(full_batch["local_state"].shape[0]),
            max_eval_transitions,
        )
        batch = {
            name: value[:num_examples]
            for name, value in full_batch.items()
        }

        predicted_next, prediction_info = predict_local_next(
            train_state=model_state,
            standardizer=standardizer,
            local_state=batch["local_state"],
            local_action=batch["local_action"],
            deterministic=True,
        )
        target_next = batch["next_local_state"]

        physical_metrics = one_step_metrics(
            predicted_next,
            target_next,
        )

        gaussian_mean, total_covariance = (
            aggregate_physical_prediction(
                train_state=model_state,
                standardizer=standardizer,
                local_state=batch["local_state"],
                local_action=batch["local_action"],
            )
        )

        nll = gaussian_nll(
            predicted_mean=gaussian_mean,
            predicted_covariance=total_covariance,
            target=target_next,
        )
        coverage_1 = marginal_coverage(
            predicted_mean=gaussian_mean,
            predicted_covariance=total_covariance,
            target=target_next,
            sigma_scale=1.0,
        )
        coverage_2 = marginal_coverage(
            predicted_mean=gaussian_mean,
            predicted_covariance=total_covariance,
            target=target_next,
            sigma_scale=2.0,
        )

        baseline_metrics = one_step_metrics(
            batch["local_state"],
            target_next,
        )

        prefix = f"agent_{agent_id}"
        metrics[f"{prefix}/state_rmse"] = float(
            physical_metrics["state_rmse"]
        )
        metrics[f"{prefix}/position_rmse"] = float(
            physical_metrics["position_rmse"]
        )
        metrics[f"{prefix}/heading_rmse"] = float(
            physical_metrics["heading_rmse"]
        )
        metrics[f"{prefix}/linear_velocity_rmse"] = float(
            physical_metrics["linear_velocity_rmse"]
        )
        metrics[f"{prefix}/angular_velocity_rmse"] = float(
            physical_metrics["angular_velocity_rmse"]
        )
        metrics[f"{prefix}/gaussian_nll"] = float(nll)
        metrics[f"{prefix}/coverage_1sigma_mean"] = float(
            jnp.mean(coverage_1)
        )
        metrics[f"{prefix}/coverage_2sigma_mean"] = float(
            jnp.mean(coverage_2)
        )
        metrics[f"{prefix}/zero_delta_state_rmse"] = float(
            baseline_metrics["state_rmse"]
        )
        metrics[f"{prefix}/mean_total_var_norm"] = float(
            jnp.mean(prediction_info["total_var_norm"])
        )

        all_state_rmse.append(physical_metrics["state_rmse"])

    metrics["mean_state_rmse"] = float(
        jnp.mean(jnp.stack(all_state_rmse))
    )
    return metrics


def wrapped_state_error(
    predicted: jax.Array,
    target: jax.Array,
) -> jax.Array:
    error = predicted - target
    heading_error = jnp.arctan2(
        jnp.sin(error[..., 2]),
        jnp.cos(error[..., 2]),
    )
    return error.at[..., 2].set(heading_error)


def evaluate_open_loop(
    model_states: list[Any],
    standardizers: list[Any],
    local_states: jax.Array,
    actions: jax.Array,
    test_indices: np.ndarray,
    horizons: list[int],
    num_trajectories: int,
) -> dict[str, dict[int, dict[str, float]]]:
    max_available_horizon = int(actions.shape[1])
    valid_horizons = sorted(
        set(h for h in horizons if 1 <= h <= max_available_horizon)
    )
    if not valid_horizons:
        raise ValueError("No requested rollout horizon fits the dataset.")

    selected = list(test_indices[:num_trajectories])
    if not selected:
        raise ValueError("The test split contains no trajectories.")

    max_horizon = max(valid_horizons)
    results: dict[str, dict[int, dict[str, float]]] = {}

    for agent_id, (model_state, standardizer) in enumerate(
        zip(model_states, standardizers)
    ):
        errors_by_horizon = {h: [] for h in valid_horizons}

        for trajectory_id in selected:
            true_states = local_states[trajectory_id, :, agent_id, :]
            true_actions = actions[trajectory_id, :, agent_id, :]

            predicted = true_states[0:1]
            for step in range(max_horizon):
                predicted, _ = predict_local_next(
                    train_state=model_state,
                    standardizer=standardizer,
                    local_state=predicted,
                    local_action=true_actions[step:step + 1],
                    deterministic=True,
                )

                horizon = step + 1
                if horizon in errors_by_horizon:
                    error = wrapped_state_error(
                        predicted[0],
                        true_states[horizon],
                    )
                    errors_by_horizon[horizon].append(error)

        agent_result: dict[int, dict[str, float]] = {}
        for horizon, errors in errors_by_horizon.items():
            stacked = jnp.stack(errors, axis=0)
            agent_result[horizon] = {
                "state_rmse": float(jnp.sqrt(jnp.mean(stacked**2))),
                "position_rmse": float(
                    jnp.sqrt(
                        jnp.mean(
                            jnp.sum(stacked[:, 0:2] ** 2, axis=-1)
                        )
                    )
                ),
                "heading_rmse": float(
                    jnp.sqrt(jnp.mean(stacked[:, 2] ** 2))
                ),
                "linear_velocity_rmse": float(
                    jnp.sqrt(jnp.mean(stacked[:, 3] ** 2))
                ),
                "angular_velocity_rmse": float(
                    jnp.sqrt(jnp.mean(stacked[:, 4] ** 2))
                ),
            }

        results[f"agent_{agent_id}"] = agent_result

    return results


def print_one_step_metrics(metrics: dict[str, float], num_agents: int) -> None:
    print(
        f"mean test state RMSE: {metrics['mean_state_rmse']:.6e}"
    )
    for agent_id in range(num_agents):
        prefix = f"agent_{agent_id}"
        print(
            f"  {prefix}: "
            f"state={metrics[f'{prefix}/state_rmse']:.6e} | "
            f"pos={metrics[f'{prefix}/position_rmse']:.6e} | "
            f"phi={metrics[f'{prefix}/heading_rmse']:.6e} | "
            f"v={metrics[f'{prefix}/linear_velocity_rmse']:.6e} | "
            f"omega={metrics[f'{prefix}/angular_velocity_rmse']:.6e} | "
            f"NLL={metrics[f'{prefix}/gaussian_nll']:.5f} | "
            f"cov1={metrics[f'{prefix}/coverage_1sigma_mean']:.3f} | "
            f"cov2={metrics[f'{prefix}/coverage_2sigma_mean']:.3f}"
        )


def main() -> None:
    args = parse_args()
    dataset, splits = load_dataset(args.dataset)

    local_states = dataset["local_states"]
    actions = dataset["actions"]
    metadata = dataset["metadata"]

    num_agents = int(local_states.shape[2])
    local_state_dim = int(local_states.shape[3])
    action_dim = int(actions.shape[3])

    print("===== Acceleration-controlled diff-drive dynamics =====")
    print(f"dataset: {args.dataset}")
    print(f"local_states: {tuple(local_states.shape)}")
    print(f"actions: {tuple(actions.shape)}")
    print(
        f"num_agents={num_agents}, "
        f"local_state_dim={local_state_dim}, "
        f"action_dim={action_dim}"
    )
    print(f"train trajectories: {splits['train'].tolist()}")
    print(f"validation trajectories: {splits['validation'].tolist()}")
    print(f"test trajectories: {splits['test'].tolist()}")

    train_batches = [
        agent_transitions(
            local_states,
            actions,
            splits["train"],
            agent_id,
        )
        for agent_id in range(num_agents)
    ]
    test_batches = [
        agent_transitions(
            local_states,
            actions,
            splits["test"],
            agent_id,
        )
        for agent_id in range(num_agents)
    ]

    standardizers = init_local_standardizers(
        num_agents=num_agents,
        act_dim=action_dim,
        local_state_dim=local_state_dim,
    )
    standardizers = [
        standardizer.update(
            local_state=batch["local_state"],
            local_action=batch["local_action"],
            next_local_state=batch["next_local_state"],
        )
        for standardizer, batch in zip(standardizers, train_batches)
    ]

    cfg = make_model_config(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, model_key = jax.random.split(rng)

    _, model_states = init_independent_transition_models(
        rng=model_key,
        num_agents=num_agents,
        act_dim=action_dim,
        cfg=cfg,
        local_state_dim=local_state_dim,
    )

    train_step = jax.jit(train_independent_transition_step)

    for step in trange(
        1,
        args.gradient_steps + 1,
        desc="Training local dynamics ensembles",
    ):
        metrics_per_agent = []

        for agent_id in range(num_agents):
            rng, sample_key = jax.random.split(rng)
            train_size = int(
                train_batches[agent_id]["local_state"].shape[0]
            )
            sample_indices = jax.random.randint(
                sample_key,
                shape=(args.batch_size,),
                minval=0,
                maxval=train_size,
            )
            batch = subset_batch(
                train_batches[agent_id],
                sample_indices,
            )

            model_states[agent_id], train_metrics = train_step(
                train_state=model_states[agent_id],
                local_batch=batch,
                standardizer=standardizers[agent_id],
            )
            metrics_per_agent.append(train_metrics)

        should_evaluate = (
            step == 1
            or step % args.eval_interval == 0
            or step == args.gradient_steps
        )
        if should_evaluate:
            mean_train_nll = float(
                jnp.mean(
                    jnp.stack(
                        [
                            item["transition_nll"]
                            for item in metrics_per_agent
                        ]
                    )
                )
            )
            print(
                f"\nstep {step}/{args.gradient_steps} | "
                f"mean minibatch NLL={mean_train_nll:.6f}"
            )
            metrics = evaluate_one_step(
                model_states=model_states,
                standardizers=standardizers,
                eval_batches=test_batches,
                max_eval_transitions=args.max_eval_transitions,
            )
            print_one_step_metrics(metrics, num_agents)

    print("\n===== Open-loop model rollout =====")
    rollout_results = evaluate_open_loop(
        model_states=model_states,
        standardizers=standardizers,
        local_states=local_states,
        actions=actions,
        test_indices=splits["test"],
        horizons=list(args.rollout_horizons),
        num_trajectories=args.num_rollout_trajectories,
    )
    for agent_name, horizon_results in rollout_results.items():
        print(agent_name)
        for horizon, values in horizon_results.items():
            print(
                f"  h={horizon:3d}: "
                f"state={values['state_rmse']:.6e} | "
                f"pos={values['position_rmse']:.6e} | "
                f"phi={values['heading_rmse']:.6e} | "
                f"v={values['linear_velocity_rmse']:.6e} | "
                f"omega={values['angular_velocity_rmse']:.6e}"
            )

    if args.save_checkpoint:
        from aorpo.utils.checkpoints import (
            save_independent_dynamics_checkpoint,
        )

        args.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata.update(
            {
                "dataset_path": str(args.dataset),
                "num_agents": num_agents,
                "local_state_dim": local_state_dim,
                "action_dim": action_dim,
                "ensemble_size": int(args.num_members),
                "hidden_dims": list(args.hidden_dims),
                "gradient_steps": int(args.gradient_steps),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "training_seed": int(args.seed),
                "train_trajectory_indices": splits[
                    "train"
                ].tolist(),
                "validation_trajectory_indices": splits[
                    "validation"
                ].tolist(),
                "test_trajectory_indices": splits[
                    "test"
                ].tolist(),
            }
        )
        save_independent_dynamics_checkpoint(
            checkpoint_path=str(args.checkpoint_path),
            model_states=model_states,
            standardizers=standardizers,
            metadata=checkpoint_metadata,
        )
        print(f"\nSaved checkpoint: {args.checkpoint_path}")

    print("\nDynamics-model training finished.")


if __name__ == "__main__":
    main()
