from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import optax
from flax import struct

from sender_marl.core.distributions import (
    diagonal_gaussian_entropy,
    squashed_gaussian_log_prob_from_pre_tanh,
)
from sender_marl.on_policy.ppo_loss import masked_mean, ppo_value_loss
from sender_marl.on_policy.train_state import PPOConfig
from sender_marl.on_policy.update import (
    explained_variance,
    prepare_ppo_training_batch,
)


Array = jax.Array


@dataclass(frozen=True)
class DPOConfig:
    """Hyperparameters for the practical DPO KL-penalty surrogate."""

    initial_beta_1: float = 0.01
    initial_beta_2: float = 0.01
    target_kl: float = 0.01
    kl_tolerance: float = 1.5
    beta_multiplier: float = 2.0
    minimum_beta: float = 1e-8
    maximum_beta: float = 1e6
    sqrt_epsilon: float = 1e-8

    def validate(self) -> None:
        if self.initial_beta_1 <= 0.0:
            raise ValueError("initial_beta_1 must be positive.")
        if self.initial_beta_2 <= 0.0:
            raise ValueError("initial_beta_2 must be positive.")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive.")
        if self.kl_tolerance <= 1.0:
            raise ValueError("kl_tolerance must be greater than 1.")
        if self.beta_multiplier <= 1.0:
            raise ValueError("beta_multiplier must be greater than 1.")
        if self.minimum_beta <= 0.0:
            raise ValueError("minimum_beta must be positive.")
        if self.maximum_beta < self.minimum_beta:
            raise ValueError(
                "maximum_beta must be at least minimum_beta."
            )
        if self.sqrt_epsilon <= 0.0:
            raise ValueError("sqrt_epsilon must be positive.")


def diagonal_gaussian_kl(
    old_mean: Array,
    old_log_std: Array,
    new_mean: Array,
    new_log_std: Array,
) -> Array:
    """KL(old || new) for diagonal Gaussians, summed over actions.

    Both policies use the same invertible tanh transform, so the KL of the
    squashed policies equals the KL of their underlying Gaussian policies.
    """

    old_mean = jnp.asarray(old_mean, dtype=jnp.float32)
    old_log_std = jnp.asarray(old_log_std, dtype=jnp.float32)
    new_mean = jnp.asarray(new_mean, dtype=jnp.float32)
    new_log_std = jnp.asarray(new_log_std, dtype=jnp.float32)

    old_variance = jnp.exp(2.0 * old_log_std)
    new_variance = jnp.exp(2.0 * new_log_std)
    mean_difference = old_mean - new_mean

    elementwise = (
        2.0 * (new_log_std - old_log_std)
        + (old_variance + jnp.square(mean_difference)) / new_variance
        - 1.0
    )
    return jnp.maximum(0.5 * jnp.sum(elementwise, axis=-1), 0.0)


def adapt_beta(
    beta: Array,
    observed_kl: Array,
    config: DPOConfig,
) -> Array:
    """Apply the adaptive DPO coefficient rule after one policy update."""

    increase = observed_kl > config.target_kl * config.kl_tolerance
    decrease = observed_kl < config.target_kl / config.kl_tolerance
    scale = jnp.where(
        increase,
        config.beta_multiplier,
        jnp.where(decrease, 1.0 / config.beta_multiplier, 1.0),
    )
    return jnp.clip(
        beta * scale,
        config.minimum_beta,
        config.maximum_beta,
    )


def select_agent_rollout(rollout: Any, agent_id: int) -> Any:
    """Extract one agent while preserving the singleton agent dimension."""

    batch = rollout.batch.replace(
        actor_obs=rollout.batch.actor_obs[..., agent_id : agent_id + 1, :],
        critic_obs=rollout.batch.critic_obs[
            ..., agent_id : agent_id + 1, :
        ],
        pre_tanh_actions=rollout.batch.pre_tanh_actions[
            ..., agent_id : agent_id + 1, :
        ],
        log_probs=rollout.batch.log_probs[..., agent_id : agent_id + 1],
        values=rollout.batch.values[..., agent_id : agent_id + 1],
        rewards=rollout.batch.rewards[..., agent_id : agent_id + 1],
        dones=rollout.batch.dones[..., agent_id : agent_id + 1],
        alive=rollout.batch.alive[..., agent_id : agent_id + 1],
    )
    return rollout.replace(
        batch=batch,
        final_values=rollout.final_values[..., agent_id : agent_id + 1],
    )


@struct.dataclass
class DPOSingleAgentUpdateOutput:
    actor_state: Any
    value_state: Any
    beta_1: Array
    beta_2: Array
    metrics: Mapping[str, Array]
    key: Array


@dataclass(frozen=True)
class IndependentDPOUpdateOutput:
    actor_states: tuple[Any, ...]
    value_states: tuple[Any, ...]
    beta_1: Array
    beta_2: Array
    metrics: dict[str, Array]


def make_single_agent_dpo_update(
    ppo_config: PPOConfig,
    dpo_config: DPOConfig,
    *,
    num_agents: int,
    jit: bool = True,
):
    """Build one agent's practical DPO actor/local-critic update."""

    ppo_config.validate()
    dpo_config.validate()
    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    def update(
        key: Array,
        actor_state: Any,
        value_state: Any,
        beta_1: Array,
        beta_2: Array,
        rollout: Any,
    ) -> DPOSingleAgentUpdateOutput:
        training_batch = prepare_ppo_training_batch(rollout, ppo_config)
        num_samples = training_batch.actor_obs.shape[0]
        if num_samples % ppo_config.num_minibatches != 0:
            raise ValueError(
                "The flattened rollout size must be divisible by "
                f"num_minibatches; got {num_samples} samples and "
                f"{ppo_config.num_minibatches} minibatches."
            )
        minibatch_size = num_samples // ppo_config.num_minibatches

        # DPO constrains every optimization epoch relative to the behavior
        # policy that generated this rollout, not relative to the latest
        # minibatch parameters.
        old_actor_params = actor_state.params

        def actor_statistics(params: Any, batch: Any):
            new_mean, new_log_std = actor_state.apply_fn(
                {"params": params},
                batch.actor_obs,
            )
            old_mean, old_log_std = actor_state.apply_fn(
                {"params": old_actor_params},
                batch.actor_obs,
            )
            old_mean = jax.lax.stop_gradient(old_mean)
            old_log_std = jax.lax.stop_gradient(old_log_std)

            new_log_probs = squashed_gaussian_log_prob_from_pre_tanh(
                batch.pre_tanh_actions,
                new_mean,
                new_log_std,
            )
            ratio = jnp.exp(new_log_probs - batch.old_log_probs)
            surrogate = masked_mean(
                ratio * batch.advantages,
                batch.alive,
            ) / float(num_agents)

            average_kl = masked_mean(
                diagonal_gaussian_kl(
                    old_mean,
                    old_log_std,
                    new_mean,
                    new_log_std,
                ),
                batch.alive,
            )
            entropy = masked_mean(
                diagonal_gaussian_entropy(new_log_std),
                batch.alive,
            )
            return surrogate, average_kl, entropy, ratio

        def run_epoch(carry, _):
            actor_state, value_state, epoch_key = carry
            next_epoch_key, permutation_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(
                permutation_key,
                num_samples,
            )
            minibatch_indices = permutation.reshape(
                ppo_config.num_minibatches,
                minibatch_size,
            )

            def run_minibatch(states, indices):
                actor_state, value_state = states
                minibatch = jax.tree_util.tree_map(
                    lambda value: value[indices],
                    training_batch,
                )

                def actor_objective(params):
                    surrogate, average_kl, entropy, ratio = (
                        actor_statistics(params, minibatch)
                    )
                    sqrt_kl = jnp.sqrt(
                        average_kl + dpo_config.sqrt_epsilon
                    )
                    kl_penalty = (
                        beta_1 * sqrt_kl + beta_2 * average_kl
                    )
                    policy_loss = -surrogate
                    total_loss = (
                        policy_loss
                        + kl_penalty
                        - ppo_config.entropy_coefficient * entropy
                    )
                    metrics = {
                        "actor_loss": total_loss,
                        "policy_loss": policy_loss,
                        "decentralized_surrogate": surrogate,
                        "entropy": entropy,
                        "average_kl": average_kl,
                        "sqrt_kl": sqrt_kl,
                        "kl_penalty": kl_penalty,
                        "mean_ratio": masked_mean(
                            ratio,
                            minibatch.alive,
                        ),
                    }
                    return total_loss, metrics

                (actor_loss, actor_metrics), actor_grads = (
                    jax.value_and_grad(actor_objective, has_aux=True)(
                        actor_state.params
                    )
                )
                actor_gradient_norm = optax.global_norm(actor_grads)
                actor_state = actor_state.apply_gradients(
                    grads=actor_grads
                )

                def value_objective(params):
                    return ppo_value_loss(
                        params,
                        value_state.apply_fn,
                        minibatch,
                        clip_epsilon=ppo_config.clip_epsilon,
                        value_loss_coefficient=(
                            ppo_config.value_loss_coefficient
                        ),
                    )

                (value_loss, value_metrics), value_grads = (
                    jax.value_and_grad(value_objective, has_aux=True)(
                        value_state.params
                    )
                )
                value_gradient_norm = optax.global_norm(value_grads)
                value_state = value_state.apply_gradients(
                    grads=value_grads
                )

                metrics = {
                    **actor_metrics,
                    **value_metrics,
                    "actor_gradient_norm": actor_gradient_norm,
                    "value_gradient_norm": value_gradient_norm,
                    "combined_loss": actor_loss + value_loss,
                }
                return (actor_state, value_state), metrics

            (actor_state, value_state), minibatch_metrics = jax.lax.scan(
                run_minibatch,
                (actor_state, value_state),
                minibatch_indices,
            )
            epoch_metrics = jax.tree_util.tree_map(
                lambda value: jnp.mean(value, axis=0),
                minibatch_metrics,
            )
            return (
                actor_state,
                value_state,
                next_epoch_key,
            ), epoch_metrics

        key, epoch_key = jax.random.split(key)
        (actor_state, value_state, _), epoch_metrics = jax.lax.scan(
            run_epoch,
            (actor_state, value_state, epoch_key),
            xs=None,
            length=ppo_config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0),
            epoch_metrics,
        )

        # Measure the final policy against the behavior policy on the full
        # rollout, then adapt both DPO coefficients once per policy update.
        _, final_kl, final_entropy, final_ratio = actor_statistics(
            actor_state.params,
            training_batch,
        )
        next_beta_1 = adapt_beta(beta_1, final_kl, dpo_config)
        next_beta_2 = adapt_beta(beta_2, final_kl, dpo_config)

        updated_values = value_state.apply_fn(
            {"params": value_state.params},
            training_batch.critic_obs,
        )
        metrics = {
            **metrics,
            "average_kl": final_kl,
            "approx_kl": final_kl,
            "entropy": final_entropy,
            "mean_ratio": masked_mean(
                final_ratio,
                training_batch.alive,
            ),
            "beta_1_before": beta_1,
            "beta_2_before": beta_2,
            "beta_1": next_beta_1,
            "beta_2": next_beta_2,
            "explained_variance": explained_variance(
                updated_values,
                training_batch.returns,
                training_batch.alive,
            ),
            "advantage_mean": masked_mean(
                training_batch.advantages,
                training_batch.alive,
            ),
            "advantage_std": jnp.sqrt(
                masked_mean(
                    jnp.square(
                        training_batch.advantages
                        - masked_mean(
                            training_batch.advantages,
                            training_batch.alive,
                        )
                    ),
                    training_batch.alive,
                )
                + 1e-8
            ),
        }

        return DPOSingleAgentUpdateOutput(
            actor_state=actor_state,
            value_state=value_state,
            beta_1=next_beta_1,
            beta_2=next_beta_2,
            metrics=metrics,
            key=key,
        )

    return jax.jit(update) if jit else update


def make_independent_dpo_update(
    ppo_config: PPOConfig,
    dpo_config: DPOConfig,
    *,
    num_agents: int,
    jit_single_agent_update: bool = True,
):
    """Build fully independent DPO updates for all agents."""

    single_agent_update = make_single_agent_dpo_update(
        ppo_config,
        dpo_config,
        num_agents=num_agents,
        jit=jit_single_agent_update,
    )

    def update(
        key: Array,
        actor_states: tuple[Any, ...],
        value_states: tuple[Any, ...],
        beta_1: Array,
        beta_2: Array,
        rollout: Any,
    ) -> IndependentDPOUpdateOutput:
        if len(actor_states) != num_agents:
            raise ValueError("Incorrect number of actor states.")
        if len(value_states) != num_agents:
            raise ValueError("Incorrect number of value states.")
        if beta_1.shape != (num_agents,):
            raise ValueError(
                f"beta_1 must have shape {(num_agents,)}, got {beta_1.shape}."
            )
        if beta_2.shape != (num_agents,):
            raise ValueError(
                f"beta_2 must have shape {(num_agents,)}, got {beta_2.shape}."
            )

        agent_keys = jax.random.split(key, num_agents)
        outputs = []
        for agent_id in range(num_agents):
            outputs.append(
                single_agent_update(
                    agent_keys[agent_id],
                    actor_states[agent_id],
                    value_states[agent_id],
                    beta_1[agent_id],
                    beta_2[agent_id],
                    select_agent_rollout(rollout, agent_id),
                )
            )

        new_actor_states = tuple(
            output.actor_state for output in outputs
        )
        new_value_states = tuple(
            output.value_state for output in outputs
        )
        new_beta_1 = jnp.stack(
            [output.beta_1 for output in outputs],
            axis=0,
        )
        new_beta_2 = jnp.stack(
            [output.beta_2 for output in outputs],
            axis=0,
        )

        metrics = jax.tree_util.tree_map(
            lambda *values: jnp.mean(jnp.stack(values, axis=0), axis=0),
            *(output.metrics for output in outputs),
        )
        for agent_id, output in enumerate(outputs):
            metrics[f"agent_{agent_id}_kl"] = output.metrics[
                "average_kl"
            ]
            metrics[f"agent_{agent_id}_beta_1"] = output.beta_1
            metrics[f"agent_{agent_id}_beta_2"] = output.beta_2

        return IndependentDPOUpdateOutput(
            actor_states=new_actor_states,
            value_states=new_value_states,
            beta_1=new_beta_1,
            beta_2=new_beta_2,
            metrics=metrics,
        )

    return update
