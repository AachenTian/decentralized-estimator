#!/usr/bin/env python3
"""Train physics-structured probabilistic diff-drive dynamics ensembles."""

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

from aorpo.agents.structured_diff_drive_dynamics import (
    init_structured_standardizers,
    init_structured_transition_models,
    make_structured_transition_batch,
    moment_match_structured_prediction,
    predict_structured_ensemble_next_gaussians,
    predict_structured_infoprop,
    predict_structured_next,
    train_structured_transition_step,
)
from aorpo.evaluation.dynamics_metrics import (
    gaussian_nll,
    marginal_coverage,
    one_step_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_500traj.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--max-eval-transitions", type=int, default=20000)

    parser.add_argument("--num-members", type=int, default=5)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--min-logvar", type=float, default=-6.0)
    parser.add_argument("--max-logvar", type=float, default=0.5)
    parser.add_argument("--mean-mse-weight", type=float, default=0.1)

    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50, 100],
    )
    parser.add_argument("--num-rollout-trajectories", type=int, default=20)
    parser.add_argument(
        "--prediction-modes",
        nargs="+",
        default=["ensemble_mean", "infoprop_ci"],
        choices=["ensemble_mean", "infoprop_ci"],
    )
    parser.add_argument(
        "--epistemic-process-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "checkpoints/"
            "structured_diff_drive_dynamics_accel_control.pkl"
        ),
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        local_states = np.asarray(data["local_states"])
        actions = np.asarray(data["actions"])
        splits = {
            "train": np.asarray(data["train_trajectory_indices"]),
            "validation": np.asarray(
                data["validation_trajectory_indices"]
            ),
            "test": np.asarray(data["test_trajectory_indices"]),
        }
        metadata = json.loads(str(data["metadata_json"]))

    if local_states.shape[-1] != 5:
        raise ValueError("Expected state [px, py, phi, v, omega].")
    if actions.shape[-1] != 2:
        raise ValueError("Expected action [a_t, alpha_z].")
    if local_states.shape[1] != actions.shape[1] + 1:
        raise ValueError("There must be one more state than action.")
    return (
        jnp.asarray(local_states),
        jnp.asarray(actions),
        splits,
        metadata,
    )


def agent_transitions(
    local_states,
    actions,
    trajectory_indices,
    agent_id,
):
    idx = jnp.asarray(trajectory_indices, dtype=jnp.int32)
    states = local_states[idx, :, agent_id, :]
    agent_actions = actions[idx, :, agent_id, :]
    current = states[:, :-1, :].reshape(-1, 5)
    nxt = states[:, 1:, :].reshape(-1, 5)
    act = agent_actions.reshape(-1, 2)
    return make_structured_transition_batch(current, act, nxt)


def make_cfg(args):
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=args.num_members,
            hidden_dims=tuple(args.hidden_dims),
            min_logvar=args.min_logvar,
            max_logvar=args.max_logvar,
            lr=args.learning_rate,
        )
    )


def subset(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def wrapped_error(predicted, target):
    error = predicted - target
    heading = jnp.arctan2(
        jnp.sin(error[..., 2]),
        jnp.cos(error[..., 2]),
    )
    return error.at[..., 2].set(heading)


def aggregation_predictions(
    model_state,
    standardizer,
    local_state,
    local_action,
    metadata,
):
    kwargs = dict(
        train_state=model_state,
        standardizer=standardizer,
        local_state=local_state,
        local_action=local_action,
        dt=float(metadata["dt"]),
        min_v=float(metadata["min_v"]),
        max_v=float(metadata["max_v"]),
        max_omega=float(metadata["max_omega"]),
    )

    raw = predict_structured_ensemble_next_gaussians(**kwargs)
    moment_mean, _, _, moment_total = (
        moment_match_structured_prediction(raw)
    )
    infoprop = predict_structured_infoprop(**kwargs)
    return {
        "ensemble_mean": (moment_mean, moment_total),
        "infoprop_ci": (
            infoprop.next_mean,
            infoprop.total_predictive_covariance,
        ),
    }


def evaluate_one_step(
    model_states,
    standardizers,
    eval_batches,
    metadata,
    max_examples,
):
    output = {}
    for agent_id, (model_state, standardizer, batch) in enumerate(
        zip(model_states, standardizers, eval_batches)
    ):
        count = min(max_examples, int(batch["local_state"].shape[0]))
        small = {key: value[:count] for key, value in batch.items()}
        predictions = aggregation_predictions(
            model_state,
            standardizer,
            small["local_state"],
            small["local_action"],
            metadata,
        )

        for mode, (mean, covariance) in predictions.items():
            physical = one_step_metrics(
                mean,
                small["next_local_state"],
            )
            nll = gaussian_nll(
                mean,
                covariance,
                small["next_local_state"],
            )
            cov1 = marginal_coverage(
                mean,
                covariance,
                small["next_local_state"],
                sigma_scale=1.0,
            )
            cov2 = marginal_coverage(
                mean,
                covariance,
                small["next_local_state"],
                sigma_scale=2.0,
            )
            prefix = f"agent_{agent_id}/{mode}"
            output[f"{prefix}/state_rmse"] = float(
                physical["state_rmse"]
            )
            output[f"{prefix}/position_rmse"] = float(
                physical["position_rmse"]
            )
            output[f"{prefix}/heading_rmse"] = float(
                physical["heading_rmse"]
            )
            output[f"{prefix}/v_rmse"] = float(
                physical["linear_velocity_rmse"]
            )
            output[f"{prefix}/omega_rmse"] = float(
                physical["angular_velocity_rmse"]
            )
            output[f"{prefix}/nll"] = float(nll)
            output[f"{prefix}/cov1"] = float(jnp.mean(cov1))
            output[f"{prefix}/cov2"] = float(jnp.mean(cov2))
    return output


def evaluate_open_loop(
    model_states,
    standardizers,
    local_states,
    actions,
    test_indices,
    metadata,
    modes,
    horizons,
    max_trajectories,
    epistemic_process_scale,
):
    selected = list(test_indices[:max_trajectories])
    max_horizon = min(max(horizons), int(actions.shape[1]))
    valid_horizons = [h for h in horizons if h <= max_horizon]
    results = {}

    for mode in modes:
        mode_results = {}
        for agent_id, (model_state, standardizer) in enumerate(
            zip(model_states, standardizers)
        ):
            errors = {h: [] for h in valid_horizons}
            for trajectory_id in selected:
                truth = local_states[trajectory_id, :, agent_id, :]
                act = actions[trajectory_id, :, agent_id, :]
                predicted = truth[0:1]
                for step in range(max_horizon):
                    predicted, _ = predict_structured_next(
                        train_state=model_state,
                        standardizer=standardizer,
                        local_state=predicted,
                        local_action=act[step:step + 1],
                        dt=float(metadata["dt"]),
                        min_v=float(metadata["min_v"]),
                        max_v=float(metadata["max_v"]),
                        max_omega=float(metadata["max_omega"]),
                        prediction_mode=mode,
                        epistemic_process_scale=epistemic_process_scale,
                    )
                    horizon = step + 1
                    if horizon in errors:
                        errors[horizon].append(
                            wrapped_error(
                                predicted[0],
                                truth[horizon],
                            )
                        )

            agent_result = {}
            for horizon, values in errors.items():
                stacked = jnp.stack(values)
                agent_result[horizon] = {
                    "state_rmse": float(
                        jnp.sqrt(jnp.mean(stacked**2))
                    ),
                    "position_rmse": float(
                        jnp.sqrt(
                            jnp.mean(
                                jnp.sum(stacked[:, :2] ** 2, axis=-1)
                            )
                        )
                    ),
                    "heading_rmse": float(
                        jnp.sqrt(jnp.mean(stacked[:, 2] ** 2))
                    ),
                    "v_rmse": float(
                        jnp.sqrt(jnp.mean(stacked[:, 3] ** 2))
                    ),
                    "omega_rmse": float(
                        jnp.sqrt(jnp.mean(stacked[:, 4] ** 2))
                    ),
                }
            mode_results[f"agent_{agent_id}"] = agent_result
        results[mode] = mode_results
    return results


def print_one_step(metrics, num_agents, modes):
    for mode in modes:
        print(f"\nOne-step aggregation: {mode}")
        for agent_id in range(num_agents):
            prefix = f"agent_{agent_id}/{mode}"
            print(
                f"  agent_{agent_id}: "
                f"state={metrics[f'{prefix}/state_rmse']:.4e} | "
                f"pos={metrics[f'{prefix}/position_rmse']:.4e} | "
                f"phi={metrics[f'{prefix}/heading_rmse']:.4e} | "
                f"v={metrics[f'{prefix}/v_rmse']:.4e} | "
                f"omega={metrics[f'{prefix}/omega_rmse']:.4e} | "
                f"NLL={metrics[f'{prefix}/nll']:.3f} | "
                f"cov1={metrics[f'{prefix}/cov1']:.3f} | "
                f"cov2={metrics[f'{prefix}/cov2']:.3f}"
            )


def main():
    args = parse_args()
    local_states, actions, splits, metadata = load_dataset(args.dataset)
    num_agents = int(local_states.shape[2])

    print("===== Structured probabilistic diff-drive dynamics =====")
    print("dataset:", args.dataset)
    print("local_states:", tuple(local_states.shape))
    print("actions:", tuple(actions.shape))
    print("input features: [sin(phi), cos(phi), v, omega, a_t, alpha_z]")
    print("stochastic target: [delta_v, delta_omega]")
    print("full next state reconstructed analytically")
    print("train trajectories:", len(splits["train"]))
    print("validation trajectories:", len(splits["validation"]))
    print("test trajectories:", len(splits["test"]))

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

    standardizers = init_structured_standardizers(num_agents)
    standardizers = [
        standardizer.update(
            batch["local_state"],
            batch["local_action"],
            batch["next_local_state"],
        )
        for standardizer, batch in zip(
            standardizers,
            train_batches,
        )
    ]

    cfg = make_cfg(args)
    rng = jax.random.PRNGKey(args.seed)
    rng, model_key = jax.random.split(rng)
    _, model_states = init_structured_transition_models(
        model_key,
        num_agents,
        cfg,
    )

    train_step = jax.jit(
        train_structured_transition_step,
        static_argnames=("mean_mse_weight",),
    )

    for step in trange(
        1,
        args.gradient_steps + 1,
        desc="Training structured ensembles",
    ):
        step_metrics = []
        for agent_id in range(num_agents):
            rng, sample_key = jax.random.split(rng)
            size = int(train_batches[agent_id]["local_state"].shape[0])
            indices = jax.random.randint(
                sample_key,
                shape=(args.batch_size,),
                minval=0,
                maxval=size,
            )
            batch = subset(train_batches[agent_id], indices)
            model_states[agent_id], metrics = train_step(
                model_states[agent_id],
                batch,
                standardizers[agent_id],
                mean_mse_weight=args.mean_mse_weight,
            )
            step_metrics.append(metrics)

        if (
            step == 1
            or step % args.eval_interval == 0
            or step == args.gradient_steps
        ):
            mean_nll = float(
                jnp.mean(
                    jnp.stack(
                        [item["transition_nll"] for item in step_metrics]
                    )
                )
            )
            print(
                f"\nstep {step}/{args.gradient_steps} | "
                f"mean minibatch dynamic-target NLL={mean_nll:.6f}"
            )
            metrics = evaluate_one_step(
                model_states,
                standardizers,
                test_batches,
                metadata,
                args.max_eval_transitions,
            )
            print_one_step(
                metrics,
                num_agents,
                args.prediction_modes,
            )

    print("\n===== Open-loop rollouts =====")
    rollouts = evaluate_open_loop(
        model_states,
        standardizers,
        local_states,
        actions,
        splits["test"],
        metadata,
        args.prediction_modes,
        list(args.rollout_horizons),
        args.num_rollout_trajectories,
        args.epistemic_process_scale,
    )
    for mode, mode_results in rollouts.items():
        print(f"\nPrediction mode: {mode}")
        for agent_name, horizon_results in mode_results.items():
            print(agent_name)
            for horizon, values in horizon_results.items():
                print(
                    f"  h={horizon:3d} | "
                    f"state={values['state_rmse']:.4e} | "
                    f"pos={values['position_rmse']:.4e} | "
                    f"phi={values['heading_rmse']:.4e} | "
                    f"v={values['v_rmse']:.4e} | "
                    f"omega={values['omega_rmse']:.4e}"
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
                "model_kind": (
                    "structured_diff_drive_delta_velocity_gaussian"
                ),
                "input_feature_order": [
                    "sin_phi",
                    "cos_phi",
                    "v",
                    "omega",
                    "a_t",
                    "alpha_z",
                ],
                "target_order": ["delta_v", "delta_omega"],
                "ensemble_size": args.num_members,
                "hidden_dims": list(args.hidden_dims),
                "min_logvar": args.min_logvar,
                "max_logvar": args.max_logvar,
                "learning_rate": args.learning_rate,
                "gradient_steps": args.gradient_steps,
                "batch_size": args.batch_size,
                "training_seed": args.seed,
                "epistemic_process_scale": (
                    args.epistemic_process_scale
                ),
                "train_trajectory_indices": (
                    splits["train"].tolist()
                ),
                "validation_trajectory_indices": (
                    splits["validation"].tolist()
                ),
                "test_trajectory_indices": (
                    splits["test"].tolist()
                ),
            }
        )
        save_independent_dynamics_checkpoint(
            checkpoint_path=str(args.checkpoint_path),
            model_states=model_states,
            standardizers=standardizers,
            metadata=checkpoint_metadata,
        )
        print("\nSaved checkpoint:", args.checkpoint_path)

    print("\nStructured dynamics training finished.")


if __name__ == "__main__":
    main()
