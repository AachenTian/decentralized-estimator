#!/usr/bin/env python3
"""Train structured diff-drive dynamics with a controlled action-region filter.

Three training-data modes are supported:

* full_coverage: use every transition;
* heldout_region: remove a joint state-action region;
* random_matched: remove the same number of transitions uniformly at random.

The random-matched model distinguishes a true distribution-shift effect from a
simple reduction in the number of available training transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from aorpo.agents.structured_diff_drive_dynamics import (
    init_structured_standardizers,
    init_structured_transition_models,
    make_structured_transition_batch,
    train_structured_transition_step,
)
from aorpo.utils.checkpoints import save_independent_dynamics_checkpoint

import train_structured_diff_drive_dynamics_variant as base


REGION_TYPES = (
    "linear_upper_saturation",
    "angular_saturation",
    "high_control_norm",
)
TRAINING_MODES = ("full_coverage", "heldout_region", "random_matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/accel_diff_drive_noise005.npz"),
    )
    parser.add_argument("--training-mode", choices=TRAINING_MODES, required=True)
    parser.add_argument(
        "--region-type",
        choices=REGION_TYPES,
        default="linear_upper_saturation",
    )
    parser.add_argument("--state-quantile", type=float, default=0.70)
    parser.add_argument("--action-quantile", type=float, default=0.70)
    parser.add_argument("--random-match-seed", type=int, default=90210)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--max-eval-transitions", type=int, default=20000)
    parser.add_argument("--num-members", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--min-logvar", type=float, default=-6.0)
    parser.add_argument("--max-logvar", type=float, default=0.5)
    parser.add_argument("--mean-mse-weight", type=float, default=0.1)
    parser.add_argument(
        "--prediction-modes",
        nargs="+",
        default=["ensemble_mean"],
        choices=["ensemble_mean", "infoprop_ci"],
    )
    parser.add_argument("--epistemic-process-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser.parse_args()


def validate_quantile(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")


def flatten_train_data(
    local_states: np.ndarray,
    actions: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = local_states[train_indices, :-1].reshape(-1, 5)
    act = actions[train_indices].reshape(-1, 2)
    return current, act


def make_region_definition(
    local_states: np.ndarray,
    actions: np.ndarray,
    train_indices: np.ndarray,
    metadata: dict,
    region_type: str,
    state_quantile: float,
    action_quantile: float,
) -> dict[str, float | str]:
    validate_quantile(state_quantile, "state-quantile")
    validate_quantile(action_quantile, "action-quantile")
    current, act = flatten_train_data(local_states, actions, train_indices)

    if region_type == "linear_upper_saturation":
        return {
            "region_type": region_type,
            "state_quantile": state_quantile,
            "action_quantile": action_quantile,
            "v_threshold": float(np.quantile(current[:, 3], state_quantile)),
            "a_threshold": float(np.quantile(act[:, 0], action_quantile)),
        }
    if region_type == "angular_saturation":
        aligned_alpha = np.sign(current[:, 4]) * act[:, 1]
        return {
            "region_type": region_type,
            "state_quantile": state_quantile,
            "action_quantile": action_quantile,
            "abs_omega_threshold": float(
                np.quantile(np.abs(current[:, 4]), state_quantile)
            ),
            "aligned_alpha_threshold": float(
                np.quantile(aligned_alpha, action_quantile)
            ),
        }
    if region_type == "high_control_norm":
        max_a = max(float(metadata["max_linear_acceleration"]), 1.0e-8)
        max_alpha = max(float(metadata["max_angular_acceleration"]), 1.0e-8)
        control_norm = np.sqrt((act[:, 0] / max_a) ** 2 + (act[:, 1] / max_alpha) ** 2)
        return {
            "region_type": region_type,
            "state_quantile": state_quantile,
            "action_quantile": action_quantile,
            "control_norm_threshold": float(
                np.quantile(control_norm, action_quantile)
            ),
            "max_linear_acceleration": max_a,
            "max_angular_acceleration": max_alpha,
        }
    raise ValueError(region_type)


def region_mask(
    current: np.ndarray,
    act: np.ndarray,
    definition: dict[str, float | str],
) -> np.ndarray:
    region_type = str(definition["region_type"])
    if region_type == "linear_upper_saturation":
        return (
            (current[:, 3] > float(definition["v_threshold"]))
            & (act[:, 0] > float(definition["a_threshold"]))
        )
    if region_type == "angular_saturation":
        aligned_alpha = np.sign(current[:, 4]) * act[:, 1]
        return (
            (np.abs(current[:, 4]) > float(definition["abs_omega_threshold"]))
            & (aligned_alpha > float(definition["aligned_alpha_threshold"]))
        )
    if region_type == "high_control_norm":
        max_a = float(definition["max_linear_acceleration"])
        max_alpha = float(definition["max_angular_acceleration"])
        norm = np.sqrt((act[:, 0] / max_a) ** 2 + (act[:, 1] / max_alpha) ** 2)
        return norm > float(definition["control_norm_threshold"])
    raise ValueError(region_type)


def agent_batch(
    local_states: np.ndarray,
    actions: np.ndarray,
    trajectory_indices: np.ndarray,
    agent_id: int,
    training_mode: str,
    definition: dict[str, float | str],
    random_match_seed: int,
) -> tuple[dict[str, jax.Array], dict[str, int]]:
    states = local_states[trajectory_indices, :, agent_id]
    current = states[:, :-1].reshape(-1, 5)
    nxt = states[:, 1:].reshape(-1, 5)
    act = actions[trajectory_indices, :, agent_id].reshape(-1, 2)
    ood = region_mask(current, act, definition)
    heldout_keep = ~ood

    if training_mode == "full_coverage":
        keep_indices = np.arange(len(current))
    elif training_mode == "heldout_region":
        keep_indices = np.flatnonzero(heldout_keep)
    elif training_mode == "random_matched":
        keep_count = int(np.sum(heldout_keep))
        rng = np.random.default_rng(random_match_seed + 1009 * agent_id)
        keep_indices = np.sort(
            rng.choice(len(current), size=keep_count, replace=False)
        )
    else:
        raise ValueError(training_mode)

    if len(keep_indices) == 0:
        raise RuntimeError(f"No transitions retained for agent {agent_id}.")

    batch = make_structured_transition_batch(
        jnp.asarray(current[keep_indices]),
        jnp.asarray(act[keep_indices]),
        jnp.asarray(nxt[keep_indices]),
    )
    stats = {
        "total_transitions": int(len(current)),
        "region_transitions": int(np.sum(ood)),
        "retained_transitions": int(len(keep_indices)),
    }
    return batch, stats


def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_dynamics=SimpleNamespace(
            num_members=args.num_members,
            hidden_dims=tuple(args.hidden_dims),
            min_logvar=args.min_logvar,
            max_logvar=args.max_logvar,
            lr=args.learning_rate,
        )
    )


def main() -> None:
    args = parse_args()
    if args.gradient_steps <= 0:
        raise ValueError("gradient-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    local_states_jax, actions_jax, splits, metadata = base.load_dataset(args.dataset)
    local_states = np.asarray(local_states_jax)
    actions = np.asarray(actions_jax)
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    num_agents = int(local_states.shape[2])

    definition = make_region_definition(
        local_states,
        actions,
        train_indices,
        metadata,
        args.region_type,
        args.state_quantile,
        args.action_quantile,
    )
    train_items = [
        agent_batch(
            local_states,
            actions,
            train_indices,
            agent_id,
            args.training_mode,
            definition,
            args.random_match_seed,
        )
        for agent_id in range(num_agents)
    ]
    train_batches = [item[0] for item in train_items]
    train_stats = [item[1] for item in train_items]
    test_batches = [
        base.agent_transitions(
            local_states_jax,
            actions_jax,
            splits["test"],
            agent_id,
        )
        for agent_id in range(num_agents)
    ]

    print("===== Controlled action-region dynamics training =====")
    print("dataset:", args.dataset)
    print("training mode:", args.training_mode)
    print("region definition:", json.dumps(definition, indent=2))
    for agent_id, stats in enumerate(train_stats):
        print(f"agent_{agent_id} training stats:", stats)

    standardizers = init_structured_standardizers(num_agents)
    standardizers = [
        standardizer.update(
            batch["local_state"],
            batch["local_action"],
            batch["next_local_state"],
        )
        for standardizer, batch in zip(standardizers, train_batches)
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

    metrics = {}
    for step in trange(
        1,
        args.gradient_steps + 1,
        desc=f"Training {args.training_mode}",
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
            batch = base.subset(train_batches[agent_id], indices)
            model_states[agent_id], item = train_step(
                model_states[agent_id],
                batch,
                standardizers[agent_id],
                mean_mse_weight=args.mean_mse_weight,
            )
            step_metrics.append(item)

        if (
            step == 1
            or step % args.eval_interval == 0
            or step == args.gradient_steps
        ):
            mean_nll = float(
                jnp.mean(
                    jnp.stack([item["transition_nll"] for item in step_metrics])
                )
            )
            print(
                f"\nstep {step}/{args.gradient_steps} | "
                f"mean minibatch NLL={mean_nll:.6f}"
            )
            metrics = base.evaluate_one_step(
                model_states,
                standardizers,
                test_batches,
                metadata,
                args.max_eval_transitions,
            )
            base.print_one_step(metrics, num_agents, args.prediction_modes)

    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "model_kind": "structured_diff_drive_delta_velocity_gaussian",
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
            "training_mode": args.training_mode,
            "random_match_seed": args.random_match_seed,
            "ood_region_definition": definition,
            "num_training_transitions_per_agent": [
                int(batch["local_state"].shape[0]) for batch in train_batches
            ],
            "training_filter_stats": train_stats,
            "train_trajectory_indices": train_indices.tolist(),
            "validation_trajectory_indices": splits["validation"].tolist(),
            "test_trajectory_indices": splits["test"].tolist(),
        }
    )
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_independent_dynamics_checkpoint(
        checkpoint_path=str(args.checkpoint_path),
        model_states=model_states,
        standardizers=standardizers,
        metadata=checkpoint_metadata,
    )
    print("Saved checkpoint:", args.checkpoint_path)

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": str(args.dataset),
                    "checkpoint": str(args.checkpoint_path),
                    "training_mode": args.training_mode,
                    "training_seed": args.seed,
                    "gradient_steps": args.gradient_steps,
                    "region_definition": definition,
                    "training_filter_stats": train_stats,
                    "one_step": metrics,
                },
                handle,
                indent=2,
            )
        print("Saved metrics:", args.metrics_json)


if __name__ == "__main__":
    main()
