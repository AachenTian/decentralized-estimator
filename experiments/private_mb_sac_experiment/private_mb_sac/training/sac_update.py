"""Focal-agent SAC updates with round-frozen opponent actor snapshots."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from private_mb_sac.core.types import ActorSnapshotBank, SACBatch
from private_mb_sac.rollout.real_collector import (
    normalize_actor_observations,
)


_LOG_2PI = float(jnp.log(2.0 * jnp.pi))
_LOG_2 = float(jnp.log(2.0))


def _sample_squashed_gaussian(key, mean, log_std):
    bounded_log_std = jnp.clip(log_std, -5.0, 2.0)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    pre_tanh = mean + jnp.exp(bounded_log_std) * noise
    action = jnp.tanh(pre_tanh)

    normalized = (pre_tanh - mean) * jnp.exp(-bounded_log_std)
    gaussian_log_prob = jnp.sum(
        -0.5 * (
            jnp.square(normalized)
            + 2.0 * bounded_log_std
            + _LOG_2PI
        ),
        axis=-1,
    )
    log_det = jnp.sum(
        2.0 * (_LOG_2 - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)),
        axis=-1,
    )
    return action, gaussian_log_prob - log_det


def _deterministic_action_and_log_prob(mean, log_std):
    bounded_log_std = jnp.clip(log_std, -5.0, 2.0)
    pre_tanh = mean
    action = jnp.tanh(pre_tanh)
    gaussian_log_prob = jnp.sum(
        -0.5 * (2.0 * bounded_log_std + _LOG_2PI),
        axis=-1,
    )
    log_det = jnp.sum(
        2.0 * (_LOG_2 - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)),
        axis=-1,
    )
    return action, gaussian_log_prob - log_det


def normalize_critic_states(states, normalizer):
    values = jnp.asarray(states, dtype=jnp.float32)
    return jnp.clip(
        (values - normalizer.state_mean)
        / jnp.maximum(normalizer.state_std, 1.0e-3),
        -normalizer.clip,
        normalizer.clip,
    )


def normalize_critic_actions(actions, normalizer):
    values = jnp.asarray(actions, dtype=jnp.float32)
    flat = values.reshape(values.shape[:-2] + (-1,))
    normalized = jnp.clip(
        (flat - normalizer.action_mean)
        / jnp.maximum(normalizer.action_std, 1.0e-3),
        -normalizer.clip,
        normalizer.clip,
    )
    return normalized.reshape(values.shape)


def compose_focal_joint_actions(
    focal_actions,
    opponent_actions,
    *,
    owner_id: int,
):
    """Insert differentiable focal actions into stop-gradient opponents."""

    frozen = jax.lax.stop_gradient(
        jnp.asarray(opponent_actions, dtype=jnp.float32)
    )
    return frozen.at[:, owner_id, :].set(
        jnp.asarray(focal_actions, dtype=jnp.float32)
    )


def _polyak_update(online_params, target_params, tau: float):
    return jax.tree_util.tree_map(
        lambda online, target: (
            tau * online + (1.0 - tau) * target
        ),
        online_params,
        target_params,
    )


def _joint_policy_actions(
    key,
    *,
    actor_apply_fn,
    focal_owner_id: int,
    focal_live_params: Any,
    snapshot_bank: ActorSnapshotBank,
    normalized_observations,
    deterministic: bool,
):
    """Generate a complete joint action while keeping every result constant."""

    num_agents = len(snapshot_bank.params_by_agent)
    keys = jax.random.split(key, num_agents)
    actions = []
    log_probs = []

    for agent_id in range(num_agents):
        params = (
            focal_live_params
            if agent_id == focal_owner_id
            else snapshot_bank.params_by_agent[agent_id]
        )
        mean, log_std = actor_apply_fn(
            {"params": params},
            normalized_observations[:, agent_id, :],
        )
        if deterministic:
            action, log_prob = _deterministic_action_and_log_prob(
                mean,
                log_std,
            )
        else:
            action, log_prob = _sample_squashed_gaussian(
                keys[agent_id],
                mean,
                log_std,
            )
        actions.append(jax.lax.stop_gradient(action))
        log_probs.append(jax.lax.stop_gradient(log_prob))

    return (
        jnp.stack(actions, axis=1),
        jnp.stack(log_probs, axis=1),
    )


def _frozen_snapshot_actions(
    key,
    *,
    actor_apply_fn,
    snapshot_bank: ActorSnapshotBank,
    normalized_observations,
):
    num_agents = len(snapshot_bank.params_by_agent)
    keys = jax.random.split(key, num_agents)
    actions = []

    for agent_id in range(num_agents):
        mean, log_std = actor_apply_fn(
            {"params": snapshot_bank.params_by_agent[agent_id]},
            normalized_observations[:, agent_id, :],
        )
        action, _ = _sample_squashed_gaussian(
            keys[agent_id],
            mean,
            log_std,
        )
        actions.append(action)

    return jax.lax.stop_gradient(jnp.stack(actions, axis=1))


def make_private_sac_update(
    config,
    *,
    owner_id: int,
    actor_normalizer,
    dynamics_normalizer,
    jit: bool = True,
):
    gamma = float(config.critic.gamma)
    tau = float(config.critic.tau)
    target_entropy = float(config.temperature.target_entropy)

    def update(
        key,
        learner_state,
        batch: SACBatch,
        snapshot_bank: ActorSnapshotBank,
        update_actor,
    ):
        target_key, opponent_key, actor_key = jax.random.split(key, 3)

        observations = normalize_actor_observations(
            batch.observations,
            actor_normalizer,
        )
        next_observations = normalize_actor_observations(
            batch.next_observations,
            actor_normalizer,
        )
        states = normalize_critic_states(
            batch.model_states,
            dynamics_normalizer,
        )
        next_states = normalize_critic_states(
            batch.next_model_states,
            dynamics_normalizer,
        )
        replay_actions = normalize_critic_actions(
            batch.actions,
            dynamics_normalizer,
        )

        alpha_before = jnp.exp(
            learner_state.alpha_state.params["log_alpha"]
        )

        next_actions, next_log_probs = _joint_policy_actions(
            target_key,
            actor_apply_fn=learner_state.actor_state.apply_fn,
            focal_owner_id=owner_id,
            focal_live_params=learner_state.actor_state.params,
            snapshot_bank=snapshot_bank,
            normalized_observations=next_observations,
            deterministic=False,
        )
        normalized_next_actions = normalize_critic_actions(
            next_actions,
            dynamics_normalizer,
        )
        target_q1, target_q2 = learner_state.critic_state.apply_fn(
            {"params": learner_state.target_critic_params},
            next_states,
            normalized_next_actions,
        )
        target_value = (
            jnp.minimum(target_q1, target_q2)
            - jax.lax.stop_gradient(alpha_before)
            * next_log_probs[:, owner_id]
        )
        td_target = jax.lax.stop_gradient(
            batch.rewards
            + gamma * (1.0 - batch.dones) * target_value
        )

        def critic_loss_fn(params):
            q1, q2 = learner_state.critic_state.apply_fn(
                {"params": params},
                states,
                replay_actions,
            )
            q1_error = q1 - td_target
            q2_error = q2 - td_target
            q1_loss = jnp.mean(jnp.square(q1_error))
            q2_loss = jnp.mean(jnp.square(q2_error))
            return q1_loss + q2_loss, {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1_mean": jnp.mean(q1),
                "q2_mean": jnp.mean(q2),
                "target_q_mean": jnp.mean(td_target),
                "td_abs": 0.5 * (
                    jnp.mean(jnp.abs(q1_error))
                    + jnp.mean(jnp.abs(q2_error))
                ),
            }

        (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
            critic_loss_fn,
            has_aux=True,
        )(learner_state.critic_state.params)
        critic_grad_norm = optax.global_norm(critic_grads)
        new_critic_state = learner_state.critic_state.apply_gradients(
            grads=critic_grads
        )

        frozen_opponent_actions = _frozen_snapshot_actions(
            opponent_key,
            actor_apply_fn=learner_state.actor_state.apply_fn,
            snapshot_bank=snapshot_bank,
            normalized_observations=observations,
        )

        def actor_loss_fn(actor_params):
            mean, log_std = learner_state.actor_state.apply_fn(
                {"params": actor_params},
                observations[:, owner_id, :],
            )
            focal_action, focal_log_prob = _sample_squashed_gaussian(
                actor_key,
                mean,
                log_std,
            )
            joint_actions = compose_focal_joint_actions(
                focal_action,
                frozen_opponent_actions,
                owner_id=owner_id,
            )
            normalized_joint_actions = normalize_critic_actions(
                joint_actions,
                dynamics_normalizer,
            )
            q1, q2 = new_critic_state.apply_fn(
                {"params": new_critic_state.params},
                states,
                normalized_joint_actions,
            )
            minimum_q = jnp.minimum(q1, q2)
            loss = jnp.mean(
                jax.lax.stop_gradient(alpha_before)
                * focal_log_prob
                - minimum_q
            )
            return loss, {
                "policy_q": jnp.mean(minimum_q),
                "entropy": -jnp.mean(focal_log_prob),
                "log_prob_mean": jnp.mean(focal_log_prob),
                "action_saturation_rate": jnp.mean(
                    (jnp.abs(focal_action) > 0.95).astype(jnp.float32)
                ),
                "sample_log_prob": focal_log_prob,
            }

        (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
            actor_loss_fn,
            has_aux=True,
        )(learner_state.actor_state.params)
        actor_grad_norm = optax.global_norm(actor_grads)

        do_actor_update = jnp.asarray(update_actor, dtype=jnp.bool_)
        new_actor_state = jax.lax.cond(
            do_actor_update,
            lambda state: state.apply_gradients(grads=actor_grads),
            lambda state: state,
            learner_state.actor_state,
        )

        entropy_residual = jax.lax.stop_gradient(
            actor_aux["sample_log_prob"] + target_entropy
        )

        def alpha_loss_fn(alpha_params):
            return -jnp.mean(
                alpha_params["log_alpha"] * entropy_residual
            )

        alpha_loss, alpha_grads = jax.value_and_grad(
            alpha_loss_fn
        )(learner_state.alpha_state.params)
        new_alpha_state = jax.lax.cond(
            do_actor_update,
            lambda state: state.apply_gradients(grads=alpha_grads),
            lambda state: state,
            learner_state.alpha_state,
        )
        alpha_value = jnp.exp(
            new_alpha_state.params["log_alpha"]
        )

        new_target_params = _polyak_update(
            new_critic_state.params,
            learner_state.target_critic_params,
            tau,
        )
        new_learner_state = learner_state.replace(
            actor_state=new_actor_state,
            critic_state=new_critic_state,
            alpha_state=new_alpha_state,
            target_critic_params=new_target_params,
        )
        metrics = {
            "critic_loss": critic_loss,
            "critic_grad_norm": critic_grad_norm,
            "actor_loss": actor_loss,
            "actor_grad_norm": actor_grad_norm,
            "alpha_loss": alpha_loss,
            "alpha_value": alpha_value,
            "actor_update_applied": do_actor_update.astype(jnp.float32),
            **{
                key: value
                for key, value in actor_aux.items()
                if key != "sample_log_prob"
            },
            **critic_aux,
        }
        return new_learner_state, metrics

    return jax.jit(update) if jit else update


def make_fixed_batch_critic_evaluator(
    config,
    *,
    owner_id: int,
    actor_normalizer,
    dynamics_normalizer,
    jit: bool = True,
):
    gamma = float(config.critic.gamma)

    def evaluate(
        learner_state,
        batch: SACBatch,
        snapshot_bank: ActorSnapshotBank,
    ):
        observations = normalize_actor_observations(
            batch.observations,
            actor_normalizer,
        )
        next_observations = normalize_actor_observations(
            batch.next_observations,
            actor_normalizer,
        )
        states = normalize_critic_states(
            batch.model_states,
            dynamics_normalizer,
        )
        next_states = normalize_critic_states(
            batch.next_model_states,
            dynamics_normalizer,
        )
        actions = normalize_critic_actions(
            batch.actions,
            dynamics_normalizer,
        )

        next_actions, next_log_probs = _joint_policy_actions(
            jax.random.PRNGKey(0),
            actor_apply_fn=learner_state.actor_state.apply_fn,
            focal_owner_id=owner_id,
            focal_live_params=learner_state.actor_state.params,
            snapshot_bank=snapshot_bank,
            normalized_observations=next_observations,
            deterministic=True,
        )
        normalized_next_actions = normalize_critic_actions(
            next_actions,
            dynamics_normalizer,
        )
        target_q1, target_q2 = learner_state.critic_state.apply_fn(
            {"params": learner_state.target_critic_params},
            next_states,
            normalized_next_actions,
        )
        alpha = jnp.exp(
            learner_state.alpha_state.params["log_alpha"]
        )
        target = batch.rewards + gamma * (1.0 - batch.dones) * (
            jnp.minimum(target_q1, target_q2)
            - alpha * next_log_probs[:, owner_id]
        )

        q1, q2 = learner_state.critic_state.apply_fn(
            {"params": learner_state.critic_state.params},
            states,
            actions,
        )
        return {
            "fixed_td_abs": 0.5 * (
                jnp.mean(jnp.abs(q1 - target))
                + jnp.mean(jnp.abs(q2 - target))
            ),
            "fixed_q1_mean": jnp.mean(q1),
            "fixed_q2_mean": jnp.mean(q2),
            "fixed_target_q_mean": jnp.mean(target),
        }

    return jax.jit(evaluate) if jit else evaluate
