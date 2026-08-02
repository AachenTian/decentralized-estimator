#!/usr/bin/env python3
"""Animate one trained oracle decentralized SAC policy on Simple Spread.

This evaluator is intentionally independent from the training loop:
  * loads the three independent actor parameter trees from a pickle checkpoint;
  * loads frozen actor-observation normalization from the checkpoint or from
    normalization.json in the checkpoint directory;
  * runs one deterministic episode with action = tanh(actor mean);
  * saves GIF, optional MP4, trajectory NPZ, and JSON summary.

The parameter parser supports the common checkpoint layouts used during the
oracle decentralized SAC experiments.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from sender_marl.envs import make_env_adapter


AGENT_COLORS = ("#0072B2", "#D55E00", "#009E73")
LANDMARK_COLOR = "#666666"
COLLISION_COLOR = "#CC79A7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help="Defaults to checkpoint_dir/normalization.json.",
    )
    parser.add_argument("--env-seed", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-landmarks", type=int, default=3)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--trail-length", type=int, default=25)
    parser.add_argument("--normalization-clip", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/policy_animations/oracle_sac_env_seed30"),
    )
    parser.add_argument("--no-mp4", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def tree_has_kernel(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "kernel" in value and is_array_like(value["kernel"]):
            return True
        return any(tree_has_kernel(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(tree_has_kernel(child) for child in value)
    return False


def looks_like_param_tree(value: Any) -> bool:
    return isinstance(value, Mapping) and tree_has_kernel(value)


def unwrap_params(value: Any) -> Any:
    while isinstance(value, Mapping) and set(value.keys()) == {"params"}:
        value = value["params"]
    return value


def find_three_actor_params(checkpoint: Any, num_agents: int) -> tuple[Any, ...]:
    """Locate one actor parameter tree per agent."""

    preferred_keys = (
        "actor_params",
        "actors",
        "actor_parameters",
        "actor_params_by_agent",
        "policy_params",
        "policies",
        "snapshot_actor_params",
    )

    def candidate_to_tuple(value: Any) -> tuple[Any, ...] | None:
        value = unwrap_params(value)

        if isinstance(value, (tuple, list)) and len(value) == num_agents:
            items = tuple(unwrap_params(item) for item in value)
            if all(looks_like_param_tree(item) for item in items):
                return items

        if isinstance(value, Mapping):
            agent_keys = [
                key for key in value
                if re.fullmatch(r"(agent[_-]?)?\d+", str(key))
            ]
            if len(agent_keys) == num_agents:
                def agent_index(key: Any) -> int:
                    match = re.search(r"\d+", str(key))
                    return int(match.group()) if match else -1
                ordered = tuple(
                    unwrap_params(value[key])
                    for key in sorted(agent_keys, key=agent_index)
                )
                if all(looks_like_param_tree(item) for item in ordered):
                    return ordered

            # Sometimes each learner entry contains actor_params.
            learner_keys = [
                key for key in value
                if re.fullmatch(r"(agent|learner|owner)[_-]?\d+", str(key))
            ]
            if len(learner_keys) == num_agents:
                ordered_entries = sorted(
                    learner_keys,
                    key=lambda key: int(re.search(r"\d+", str(key)).group()),
                )
                items = []
                for key in ordered_entries:
                    entry = value[key]
                    if not isinstance(entry, Mapping):
                        return None
                    found = None
                    for actor_key in preferred_keys:
                        if actor_key in entry:
                            found = unwrap_params(entry[actor_key])
                            break
                    if found is None or not looks_like_param_tree(found):
                        return None
                    items.append(found)
                return tuple(items)

        return None

    if isinstance(checkpoint, Mapping):
        for key in preferred_keys:
            if key in checkpoint:
                result = candidate_to_tuple(checkpoint[key])
                if result is not None:
                    return result

    direct = candidate_to_tuple(checkpoint)
    if direct is not None:
        return direct

    # Recursive fallback.
    def walk(value: Any, path: tuple[str, ...] = ()):
        if isinstance(value, Mapping):
            for key, child in value.items():
                result = candidate_to_tuple(child)
                if result is not None and any(
                    token in str(key).lower()
                    for token in ("actor", "policy", "learner", "agent")
                ):
                    return result
                nested = walk(child, path + (str(key),))
                if nested is not None:
                    return nested
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                nested = walk(child, path + (str(index),))
                if nested is not None:
                    return nested
        return None

    result = walk(checkpoint)
    if result is None:
        keys = sorted(checkpoint.keys()) if isinstance(checkpoint, Mapping) else []
        raise KeyError(
            "Could not locate three independent actor parameter trees. "
            f"Top-level checkpoint keys: {keys}"
        )
    return result


def natural_key(path: str):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path)
    ]


def flatten_mapping(
    value: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, Any]]:
    leaves = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            leaves.extend(flatten_mapping(child, prefix + (str(key),)))
    else:
        leaves.append(("/".join(prefix), value))
    return leaves


def dense_layers(params: Any):
    params = unwrap_params(params)
    leaves = flatten_mapping(params)
    grouped: dict[str, dict[str, Any]] = {}

    for path, value in leaves:
        if "/" not in path:
            continue
        parent, leaf_name = path.rsplit("/", 1)
        if leaf_name in {"kernel", "bias"}:
            grouped.setdefault(parent, {})[leaf_name] = value

    layers = []
    for path, entry in grouped.items():
        if "kernel" not in entry or "bias" not in entry:
            continue
        kernel = jnp.asarray(entry["kernel"], dtype=jnp.float32)
        bias = jnp.asarray(entry["bias"], dtype=jnp.float32)
        if kernel.ndim == 2 and bias.ndim == 1:
            layers.append((path, kernel, bias))

    if not layers:
        raise ValueError("No Dense kernel/bias pairs found in actor parameters.")
    return sorted(layers, key=lambda item: natural_key(item[0]))


def infer_actor_kind(params: Any) -> str:
    paths = [path.lower() for path, _ in flatten_mapping(unwrap_params(params))]
    has_scalar_log_std_parameter = any(
        path.endswith("log_std") and not path.endswith("log_std/kernel")
        for path in paths
    )
    return "tanh_hidden" if has_scalar_log_std_parameter else "relu_hidden"


def select_forward_chain(params: Any, obs_dim: int, action_dim: int):
    layers = dense_layers(params)
    remaining = list(layers)
    chain = []
    current_dim = obs_dim

    while remaining:
        compatible = [
            layer for layer in remaining
            if int(layer[1].shape[0]) == current_dim
        ]
        if not compatible:
            break

        output_candidates = [
            layer for layer in compatible
            if int(layer[1].shape[1]) == action_dim
        ]

        if output_candidates:
            named_mean = [
                layer for layer in output_candidates
                if any(
                    token in layer[0].lower()
                    for token in ("mean", "mu", "actor_mean", "policy_mean")
                )
            ]
            selected = (
                sorted(named_mean, key=lambda item: natural_key(item[0]))[0]
                if named_mean
                else sorted(
                    output_candidates,
                    key=lambda item: natural_key(item[0]),
                )[0]
            )
            chain.append(selected)
            break

        hidden_candidates = [
            layer for layer in compatible
            if int(layer[1].shape[1]) != action_dim
        ]
        if not hidden_candidates:
            break
        selected = sorted(
            hidden_candidates,
            key=lambda item: natural_key(item[0]),
        )[0]
        chain.append(selected)
        remaining.remove(selected)
        current_dim = int(selected[1].shape[1])

    if not chain or int(chain[-1][1].shape[1]) != action_dim:
        description = [
            (path, tuple(kernel.shape))
            for path, kernel, _ in layers
        ]
        raise ValueError(
            "Unable to infer actor mean forward chain. "
            f"Dense layers: {description}"
        )
    return chain


def make_generic_actor_apply(params: Any, obs_dim: int, action_dim: int):
    kind = infer_actor_kind(params)
    chain = select_forward_chain(params, obs_dim, action_dim)

    def apply(observation: jax.Array) -> jax.Array:
        x = jnp.asarray(observation, dtype=jnp.float32)
        for index, (_path, kernel, bias) in enumerate(chain):
            x = x @ kernel + bias
            if index < len(chain) - 1:
                x = jnp.tanh(x) if kind == "tanh_hidden" else jax.nn.relu(x)
        return jnp.tanh(x)

    return jax.jit(apply), kind, [path for path, _, _ in chain]


def recursive_find_numeric_array(
    value: Any,
    key_predicate,
) -> np.ndarray | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key_predicate(str(key).lower()):
                try:
                    array = np.asarray(child, dtype=np.float32)
                    if array.ndim == 1:
                        return array
                except Exception:
                    pass
        for child in value.values():
            result = recursive_find_numeric_array(child, key_predicate)
            if result is not None:
                return result
    elif isinstance(value, (tuple, list)):
        for child in value:
            result = recursive_find_numeric_array(child, key_predicate)
            if result is not None:
                return result
    return None


def load_normalization(
    checkpoint: Any,
    path: Path | None,
    obs_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    sources: list[tuple[str, Any]] = [("checkpoint", checkpoint)]

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Normalization file not found: {path}")
        sources.insert(0, (str(path), json.loads(path.read_text())))

    mean_tokens = (
        "obs_mean",
        "observation_mean",
        "actor_obs_mean",
        "actor_observation_mean",
    )
    std_tokens = (
        "obs_std",
        "observation_std",
        "actor_obs_std",
        "actor_observation_std",
    )

    for source_name, source in sources:
        mean = recursive_find_numeric_array(
            source,
            lambda key: key in mean_tokens,
        )
        std = recursive_find_numeric_array(
            source,
            lambda key: key in std_tokens,
        )
        if mean is not None and std is not None:
            if mean.shape != (obs_dim,) or std.shape != (obs_dim,):
                raise ValueError(
                    f"Observation normalization in {source_name} has "
                    f"mean/std shapes {mean.shape}/{std.shape}, expected "
                    f"({obs_dim},)."
                )
            return mean, std, source_name

    # Broader fallback for nested {"observation": {"mean": ..., "std": ...}}.
    def search_observation_block(value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                if "obs" in str(key).lower() or "observation" in str(key).lower():
                    if isinstance(child, Mapping):
                        mean = child.get("mean")
                        std = child.get("std")
                        if mean is not None and std is not None:
                            return (
                                np.asarray(mean, dtype=np.float32),
                                np.asarray(std, dtype=np.float32),
                            )
                result = search_observation_block(child)
                if result is not None:
                    return result
        elif isinstance(value, (tuple, list)):
            for child in value:
                result = search_observation_block(child)
                if result is not None:
                    return result
        return None

    for source_name, source in sources:
        result = search_observation_block(source)
        if result is not None:
            mean, std = result
            if mean.shape == (obs_dim,) and std.shape == (obs_dim,):
                return mean, std, source_name

    raise KeyError(
        "Actor-observation normalization was not found. Pass the exact "
        "--normalization path, normally checkpoint_dir/normalization.json."
    )


def extract_positions(env_state: Any, num_agents: int, num_landmarks: int):
    positions = np.asarray(jax.device_get(env_state.p_pos), dtype=np.float32)
    total = num_agents + num_landmarks
    return positions[:num_agents].copy(), positions[num_agents:total].copy()


def run_episode(args: argparse.Namespace):
    checkpoint = load_pickle(args.checkpoint)
    actor_params = find_three_actor_params(checkpoint, args.num_agents)

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        max_steps=args.max_steps,
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )
    obs_dim = int(adapter.spec.actor_obs_dim)
    action_dim = int(adapter.spec.policy_action_dim)

    normalization_path = args.normalization
    if normalization_path is None:
        normalization_path = args.checkpoint.parent / "normalization.json"

    obs_mean, obs_std, norm_source = load_normalization(
        checkpoint,
        normalization_path if normalization_path.exists() else None,
        obs_dim,
    )
    obs_std = np.maximum(obs_std, 1.0e-3)

    actor_functions = []
    actor_metadata = []
    for agent_id, params in enumerate(actor_params):
        apply_fn, kind, chain = make_generic_actor_apply(
            params,
            obs_dim,
            action_dim,
        )
        actor_functions.append(apply_fn)
        actor_metadata.append(
            {
                "agent": agent_id,
                "hidden_activation": kind,
                "forward_chain": chain,
            }
        )

    reset_key = jax.random.PRNGKey(args.env_seed)
    env_state, features = adapter.reset(reset_key)

    agents, landmarks = extract_positions(
        env_state,
        args.num_agents,
        args.num_landmarks,
    )
    agent_positions = [agents]
    landmark_positions = [landmarks]
    actions_history = []
    reward_history = []
    collision_history = []
    minimum_distance_history = []

    for step_index in range(args.max_steps):
        raw_obs = adapter.build_actor_observations(
            features,
            features.local_states,
        )
        normalized_obs = jnp.clip(
            (
                raw_obs
                - jnp.asarray(obs_mean, dtype=jnp.float32)
            )
            / jnp.asarray(obs_std, dtype=jnp.float32),
            -args.normalization_clip,
            args.normalization_clip,
        )

        actions = jnp.stack(
            [
                actor_functions[agent_id](normalized_obs[agent_id])
                for agent_id in range(args.num_agents)
            ],
            axis=0,
        )

        step_key = jax.random.fold_in(
            jax.random.PRNGKey(args.env_seed),
            step_index + 1,
        )
        output = adapter.step(step_key, env_state, actions)

        env_state = output.env_state
        features = output.features
        agents, landmarks = extract_positions(
            env_state,
            args.num_agents,
            args.num_landmarks,
        )

        agent_positions.append(agents)
        landmark_positions.append(landmarks)
        actions_history.append(np.asarray(jax.device_get(actions)))
        reward_history.append(
            float(np.asarray(jax.device_get(jnp.mean(output.rewards))))
        )
        collision_history.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "pair_collision_rate",
                            jnp.asarray(0.0),
                        )
                    )
                )
            )
        )
        minimum_distance_history.append(
            float(
                np.asarray(
                    jax.device_get(
                        output.metrics.get(
                            "min_pair_distance",
                            jnp.asarray(np.nan),
                        )
                    )
                )
            )
        )

        if bool(np.asarray(jax.device_get(output.episode_done))):
            break

    result = {
        "agent_positions": np.asarray(agent_positions, dtype=np.float32),
        "landmark_positions": np.asarray(landmark_positions, dtype=np.float32),
        "actions": np.asarray(actions_history, dtype=np.float32),
        "rewards": np.asarray(reward_history, dtype=np.float32),
        "collision_rates": np.asarray(collision_history, dtype=np.float32),
        "minimum_distances": np.asarray(
            minimum_distance_history,
            dtype=np.float32,
        ),
        "actor_metadata": actor_metadata,
        "normalization_source": norm_source,
        "checkpoint": checkpoint,
    }
    return result


def limits_from_result(result, padding=0.25):
    points = np.concatenate(
        [
            result["agent_positions"].reshape(-1, 2),
            result["landmark_positions"].reshape(-1, 2),
        ],
        axis=0,
    )
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    span = max(float((high - low).max()), 1.0)
    half = 0.5 * span + padding
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
    )


def create_animation(args, result):
    limits = limits_from_result(result)
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    episode_length = len(result["rewards"])

    def update(frame):
        ax.clear()
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Oracle decentralized SAC — environment seed {args.env_seed}"
        )

        state_index = min(frame, episode_length)
        trail_start = max(0, state_index - args.trail_length)

        landmarks = result["landmark_positions"][state_index]
        for landmark_id, position in enumerate(landmarks):
            ax.add_patch(
                Circle(
                    position,
                    0.10,
                    facecolor=LANDMARK_COLOR,
                    edgecolor="black",
                    alpha=0.35,
                )
            )
            ax.text(
                position[0],
                position[1],
                f"L{landmark_id}",
                ha="center",
                va="center",
                fontsize=8,
            )

        collision_now = (
            frame > 0
            and result["collision_rates"][min(frame - 1, episode_length - 1)]
            > 0.0
        )

        for agent_id in range(args.num_agents):
            trail = result["agent_positions"][
                trail_start:state_index + 1,
                agent_id,
            ]
            ax.plot(
                trail[:, 0],
                trail[:, 1],
                color=AGENT_COLORS[agent_id],
                linewidth=1.8,
                alpha=0.65,
            )

            position = result["agent_positions"][state_index, agent_id]
            ax.add_patch(
                Circle(
                    position,
                    0.075,
                    facecolor=AGENT_COLORS[agent_id],
                    edgecolor=COLLISION_COLOR if collision_now else "black",
                    linewidth=3.0 if collision_now else 1.2,
                    alpha=0.95,
                )
            )
            ax.text(
                position[0],
                position[1],
                f"A{agent_id}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

            if frame < episode_length:
                action = result["actions"][frame, agent_id]
                ax.arrow(
                    position[0],
                    position[1],
                    0.12 * action[0],
                    0.12 * action[1],
                    width=0.006,
                    head_width=0.035,
                    length_includes_head=True,
                    color=AGENT_COLORS[agent_id],
                    alpha=0.8,
                )

        cumulative_return = float(
            result["rewards"][:state_index].sum()
        )
        collision = (
            float(result["collision_rates"][state_index - 1])
            if state_index > 0 else 0.0
        )
        minimum_distance = (
            float(result["minimum_distances"][state_index - 1])
            if state_index > 0 else np.nan
        )
        ax.text(
            0.02,
            0.98,
            (
                f"step {state_index:02d}/{episode_length:02d}\n"
                f"return {cumulative_return:.3f}\n"
                f"collision {collision:.3f}\n"
                f"min distance {minimum_distance:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        return ()

    animator = animation.FuncAnimation(
        fig,
        update,
        frames=episode_length + 1,
        interval=1000 / args.fps,
        repeat=True,
        blit=False,
    )
    return fig, animator


def checkpoint_round(checkpoint: Any):
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("round", "sync_round", "update", "step"):
        if key in checkpoint:
            value = checkpoint[key]
            try:
                return int(np.asarray(value))
            except Exception:
                return str(value)
    return None


def save_outputs(args, result):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.output_dir / "oracle_sac_seed30_trajectory.npz"
    np.savez_compressed(
        npz_path,
        agent_positions=result["agent_positions"],
        landmark_positions=result["landmark_positions"],
        actions=result["actions"],
        rewards=result["rewards"],
        collision_rates=result["collision_rates"],
        minimum_distances=result["minimum_distances"],
    )

    checkpoint = result["checkpoint"]
    summary = {
        "algorithm": "oracle_decentralized_sac",
        "checkpoint": str(args.checkpoint),
        "checkpoint_round": checkpoint_round(checkpoint),
        "environment_seed": args.env_seed,
        "deterministic_action": "tanh(actor_mean)",
        "normalization_source": result["normalization_source"],
        "episode_length": int(len(result["rewards"])),
        "episode_return": float(result["rewards"].sum()),
        "mean_pair_collision_rate": float(
            result["collision_rates"].mean()
            if len(result["collision_rates"]) else 0.0
        ),
        "minimum_pair_distance": float(
            np.nanmin(result["minimum_distances"])
            if len(result["minimum_distances"]) else np.nan
        ),
        "mean_actions_by_agent": (
            result["actions"].mean(axis=0).tolist()
            if len(result["actions"])
            else []
        ),
        "action_saturation_rate_by_agent": (
            (np.abs(result["actions"]) > 0.95)
            .mean(axis=(0, 2))
            .tolist()
            if len(result["actions"])
            else []
        ),
        "actor_parser": result["actor_metadata"],
    }
    summary_path = args.output_dir / "oracle_sac_seed30_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    fig, animator = create_animation(args, result)
    gif_path = args.output_dir / "oracle_sac_seed30.gif"
    animator.save(
        gif_path,
        writer=animation.PillowWriter(fps=args.fps),
        dpi=args.dpi,
    )
    print("Saved:", gif_path)

    if not args.no_mp4 and shutil.which("ffmpeg") is not None:
        mp4_path = args.output_dir / "oracle_sac_seed30.mp4"
        animator.save(
            mp4_path,
            writer=animation.FFMpegWriter(
                fps=args.fps,
                bitrate=2400,
            ),
            dpi=args.dpi,
        )
        print("Saved:", mp4_path)
    elif not args.no_mp4:
        print("ffmpeg was not found; MP4 output was skipped.")

    plt.close(fig)
    print("Saved:", npz_path)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    result = run_episode(args)
    save_outputs(args, result)


if __name__ == "__main__":
    main()
