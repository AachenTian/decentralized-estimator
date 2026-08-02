"""Owner-private real/model replay mixing and repeated SAC updates."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from private_mb_sac.agents.learner import attach_owner_sac_functions
from private_mb_sac.core.types import SACBatch


def _merge_replay_batches(batches):
    names = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "dones",
        "model_states",
        "next_model_states",
    )
    return {
        name: np.concatenate(
            [np.asarray(getattr(batch, name)) for batch in batches],
            axis=0,
        )
        for name in names
    }


def sample_private_sac_batch(
    *,
    real_replay,
    model_replay,
    owner_id: int,
    batch_size: int,
    real_fraction: float,
    rng: np.random.Generator,
):
    """Sample exclusively from one owner's real and model replay objects."""

    if len(real_replay) < 1:
        raise ValueError("The owner's private real replay is empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= real_fraction <= 1.0:
        raise ValueError("real_fraction must lie in [0, 1].")

    if len(model_replay) < 1:
        real_count = batch_size
        model_count = 0
    else:
        real_count = int(round(batch_size * real_fraction))
        real_count = min(max(real_count, 1), batch_size)
        model_count = batch_size - real_count

    sampled = [
        real_replay.sample(
            real_count,
            rng,
            replace=True,
        )
    ]
    if model_count > 0:
        sampled.append(
            model_replay.sample(
                model_count,
                rng,
                replace=True,
            )
        )

    arrays = _merge_replay_batches(sampled)
    permutation = rng.permutation(batch_size)

    return (
        SACBatch(
            observations=jnp.asarray(
                arrays["observations"][permutation]
            ),
            actions=jnp.asarray(arrays["actions"][permutation]),
            rewards=jnp.asarray(
                arrays["rewards"][permutation, owner_id]
            ),
            next_observations=jnp.asarray(
                arrays["next_observations"][permutation]
            ),
            dones=jnp.asarray(
                arrays["dones"][permutation, owner_id]
            ),
            model_states=jnp.asarray(
                arrays["model_states"][permutation]
            ),
            next_model_states=jnp.asarray(
                arrays["next_model_states"][permutation]
            ),
        ),
        {
            "sac/real_batch_fraction": float(real_count / batch_size),
            "sac/model_batch_fraction": float(model_count / batch_size),
        },
    )


def sample_fixed_real_batch(
    replay,
    *,
    owner_id: int,
    batch_size: int,
    seed: int,
):
    raw = replay.sample(
        max(1, int(batch_size)),
        np.random.default_rng(seed),
        replace=True,
    )
    return SACBatch(
        observations=raw.observations,
        actions=raw.actions,
        rewards=raw.rewards[:, owner_id],
        next_observations=raw.next_observations,
        dones=raw.dones[:, owner_id],
        model_states=raw.model_states,
        next_model_states=raw.next_model_states,
    )


def train_private_owner_sac(
    *,
    runtime,
    real_replay,
    model_replay,
    snapshot_bank,
    actor_normalizer,
    dynamics_normalizer,
    config,
    updates: int,
    batch_size: int,
    seed: int,
    jit: bool,
):
    minimum_replay_size = int(config.sac.minimum_replay_size)
    if (
        len(real_replay) < minimum_replay_size
        or dynamics_normalizer is None
    ):
        return {
            "sac/skipped": 1.0,
            "critic/updates_total": float(runtime.critic_updates_total),
            "actor/updates_total": float(runtime.actor_updates_total),
        }

    attach_owner_sac_functions(
        runtime,
        config=config,
        actor_normalizer=actor_normalizer,
        dynamics_normalizer=dynamics_normalizer,
        jit=jit,
    )

    if runtime.fixed_batch is None:
        runtime.fixed_batch = sample_fixed_real_batch(
            real_replay,
            owner_id=runtime.owner_id,
            batch_size=min(
                int(config.sac.fixed_batch_size),
                max(len(real_replay), 1),
            ),
            seed=seed + 1,
        )

    host_rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed + 2)
    update_metrics = []
    mixture_metrics = []
    start = perf_counter()

    for _ in range(int(updates)):
        batch, mixture = sample_private_sac_batch(
            real_replay=real_replay,
            model_replay=model_replay,
            owner_id=runtime.owner_id,
            batch_size=int(batch_size),
            real_fraction=float(
                config.replay.real_fraction_in_sac_batch
            ),
            rng=host_rng,
        )

        next_critic_update = runtime.critic_updates_total + 1
        warmup = int(config.sac.critic_warmup_updates)
        policy_delay = int(config.sac.policy_delay)
        update_actor = (
            next_critic_update > warmup
            and (next_critic_update - warmup) % policy_delay == 0
        )

        jax_key, update_key = jax.random.split(jax_key)
        new_state, metrics = runtime.update_fn(
            update_key,
            runtime.learner_state,
            batch,
            snapshot_bank,
            jnp.asarray(update_actor),
        )
        jax.block_until_ready(metrics["critic_loss"])

        runtime.learner_state = new_state
        runtime.critic_updates_total += 1
        if update_actor:
            runtime.actor_updates_total += 1

        update_metrics.append(
            {
                key: float(np.asarray(value))
                for key, value in metrics.items()
            }
        )
        mixture_metrics.append(mixture)

    fixed_metrics = runtime.fixed_evaluator_fn(
        runtime.learner_state,
        runtime.fixed_batch,
        snapshot_bank,
    )
    fixed_metrics = {
        key: float(np.asarray(value))
        for key, value in fixed_metrics.items()
    }

    def average(name: str) -> float:
        return float(
            np.mean([metrics[name] for metrics in update_metrics])
        )

    return {
        "sac/skipped": 0.0,
        "sac/real_batch_fraction": float(
            np.mean(
                [
                    metrics["sac/real_batch_fraction"]
                    for metrics in mixture_metrics
                ]
            )
        ),
        "sac/model_batch_fraction": float(
            np.mean(
                [
                    metrics["sac/model_batch_fraction"]
                    for metrics in mixture_metrics
                ]
            )
        ),
        "critic/loss": average("critic_loss"),
        "critic/q1_loss": average("q1_loss"),
        "critic/q2_loss": average("q2_loss"),
        "critic/q1_mean": average("q1_mean"),
        "critic/q2_mean": average("q2_mean"),
        "critic/target_q_mean": average("target_q_mean"),
        "critic/td_abs": average("td_abs"),
        "critic/fixed_td_abs": float(
            fixed_metrics["fixed_td_abs"]
        ),
        "critic/grad_norm": average("critic_grad_norm"),
        "critic/updates_total": float(
            runtime.critic_updates_total
        ),
        "actor/loss": average("actor_loss"),
        "actor/policy_q": average("policy_q"),
        "actor/entropy": average("entropy"),
        "actor/log_prob_mean": average("log_prob_mean"),
        "actor/action_saturation_rate": average(
            "action_saturation_rate"
        ),
        "actor/grad_norm": average("actor_grad_norm"),
        "actor/updates_total": float(runtime.actor_updates_total),
        "alpha/value": average("alpha_value"),
        "alpha/loss": average("alpha_loss"),
        "timing/sac_seconds": float(perf_counter() - start),
    }


def tree_l2_distance(first: Any, second: Any) -> float:
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    if len(first_leaves) != len(second_leaves):
        raise ValueError("Parameter trees have different structures.")

    squared_distance = 0.0
    for first_leaf, second_leaf in zip(
        first_leaves,
        second_leaves,
    ):
        difference = (
            np.asarray(first_leaf) - np.asarray(second_leaf)
        )
        squared_distance += float(np.sum(np.square(difference)))
    return float(np.sqrt(squared_distance))


def pairwise_actor_parameter_distance(runtimes) -> list[float]:
    params = [
        runtime.learner_state.actor_state.params
        for runtime in runtimes
    ]
    distances = []
    for first in range(len(params)):
        for second in range(first + 1, len(params)):
            distances.append(
                tree_l2_distance(params[first], params[second])
            )
    return distances
