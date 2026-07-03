# aorpo/agents/independent_dynamics.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Any

import jax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState


LOCAL_STATE_DIM = 4  # [p_x, p_y, v_x, v_y]


# ============================================================
# 1. Extract local transitions from the existing global replay batch
# ============================================================

def extract_local_kinematic_state(
    flat_state: jnp.ndarray,
    agent_id: int,
    num_agents: int,
    num_landmarks: int,
) -> jnp.ndarray:
    """
    Extract agent i's physical state from a flattened MPE state.

    Expected flattened layout:
        [all p_pos, all p_vel, communication, dones, step]

    Args:
        flat_state:    (B, full_state_dim)
        agent_id:      integer in [0, num_agents - 1]
        num_agents:    number of agents
        num_landmarks: number of landmarks

    Returns:
        local_state: (B, 4) = [p_x, p_y, v_x, v_y]
    """
    flat_state = jnp.asarray(flat_state)

    if flat_state.ndim != 2:
        raise ValueError(
            "flat_state must have shape (batch_size, full_state_dim), "
            f"but got {flat_state.shape}."
        )

    if not 0 <= agent_id < num_agents:
        raise ValueError(
            f"agent_id must be in [0, {num_agents - 1}], got {agent_id}."
        )

    num_objects = num_agents + num_landmarks
    batch_size = flat_state.shape[0]

    # p_pos occupies the first 2 * num_objects dimensions.
    pos_dim = 2 * num_objects
    p_pos = flat_state[:, :pos_dim].reshape(batch_size, num_objects, 2)

    # p_vel immediately follows p_pos.
    vel_start = pos_dim
    vel_end = vel_start + pos_dim
    p_vel = flat_state[:, vel_start:vel_end].reshape(batch_size, num_objects, 2)

    agent_pos = p_pos[:, agent_id, :]
    agent_vel = p_vel[:, agent_id, :]

    return jnp.concatenate([agent_pos, agent_vel], axis=-1)


def extract_agent_action(
    batch: Dict[str, jnp.ndarray],
    agent_id: int,
    num_agents: int,
    act_dim: int,
) -> jnp.ndarray:
    """
    Extract action a_i from the existing replay batch.

    Current replay convention:
        agent_0 -> batch["a_ego"]
        agent_1 ... agent_(N-1) -> slices of batch["a_opp"]

    Returns:
        action_i: (B, act_dim)
    """
    if not 0 <= agent_id < num_agents:
        raise ValueError(
            f"agent_id must be in [0, {num_agents - 1}], got {agent_id}."
        )

    if agent_id == 0:
        return jnp.asarray(batch["a_ego"])

    a_opp = jnp.asarray(batch["a_opp"])

    start = (agent_id - 1) * act_dim
    end = start + act_dim

    if a_opp.shape[-1] < end:
        raise ValueError(
            f"a_opp has final dimension {a_opp.shape[-1]}, but agent {agent_id} "
            f"requires slice [{start}:{end}]."
        )

    return a_opp[:, start:end]


def make_local_transition_batch(
    batch: Dict[str, Any],
    agent_id: int,
    num_agents: int,
    num_landmarks: int,
    act_dim: int,
) -> Dict[str, jnp.ndarray]:
    """
    Convert one global replay batch into one local transition batch.

    Returns:
        {
            "local_state":      z_t^i,      shape (B, 4)
            "local_action":     a_t^i,      shape (B, act_dim)
            "next_local_state": z_{t+1}^i,  shape (B, 4)
            "delta":            delta z_t^i, shape (B, 4)
        }
    """
    local_state = extract_local_kinematic_state(
        batch["state"],
        agent_id=agent_id,
        num_agents=num_agents,
        num_landmarks=num_landmarks,
    )

    next_local_state = extract_local_kinematic_state(
        batch["next_state"],
        agent_id=agent_id,
        num_agents=num_agents,
        num_landmarks=num_landmarks,
    )

    local_action = extract_agent_action(
        batch,
        agent_id=agent_id,
        num_agents=num_agents,
        act_dim=act_dim,
    )

    return {
        "local_state": local_state,
        "local_action": local_action,
        "next_local_state": next_local_state,
        "delta": next_local_state - local_state,
    }


# ============================================================
# 2. Per-agent running normalizer
# ============================================================

@struct.dataclass
class LocalStandardizerRS:
    """
    Running statistics for one local dynamics model.

    State:  [p_x, p_y, v_x, v_y]
    Action: a_i
    Target: delta state
    """

    local_state_stats: running_statistics.RunningStatisticsState
    action_stats: running_statistics.RunningStatisticsState
    delta_stats: running_statistics.RunningStatisticsState

    @classmethod
    def create(
        cls,
        local_state_dim: int,
        act_dim: int,
    ) -> "LocalStandardizerRS":
        return cls(
            local_state_stats=running_statistics.init_state(
                jnp.zeros((local_state_dim,), dtype=jnp.float32)
            ),
            action_stats=running_statistics.init_state(
                jnp.zeros((act_dim,), dtype=jnp.float32)
            ),
            delta_stats=running_statistics.init_state(
                jnp.zeros((local_state_dim,), dtype=jnp.float32)
            ),
        )

    def update(
        self,
        local_state: jnp.ndarray,
        local_action: jnp.ndarray,
        next_local_state: jnp.ndarray,
    ) -> "LocalStandardizerRS":
        delta = next_local_state - local_state

        return LocalStandardizerRS(
            local_state_stats=running_statistics.update(
                self.local_state_stats,
                local_state,
            ),
            action_stats=running_statistics.update(
                self.action_stats,
                local_action,
            ),
            delta_stats=running_statistics.update(
                self.delta_stats,
                delta,
            ),
        )

    def norm_local_state(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.local_state_stats)

    def norm_action(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.action_stats)

    def norm_delta(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.delta_stats)

    def denorm_delta(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.denormalize(x, self.delta_stats)


def init_local_standardizers(
    num_agents: int,
    act_dim: int,
) -> List[LocalStandardizerRS]:
    """Create one local normalizer per agent."""
    return [
        LocalStandardizerRS.create(
            local_state_dim=LOCAL_STATE_DIM,
            act_dim=act_dim,
        )
        for _ in range(num_agents)
    ]


def update_local_standardizers(
    standardizers: Sequence[LocalStandardizerRS],
    batch: Dict[str, Any],
    num_agents: int,
    num_landmarks: int,
    act_dim: int,
) -> List[LocalStandardizerRS]:
    """Update every agent's local normalizer using a sampled global batch."""
    updated = []

    for agent_id, std in enumerate(standardizers):
        local_batch = make_local_transition_batch(
            batch=batch,
            agent_id=agent_id,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            act_dim=act_dim,
        )

        updated.append(
            std.update(
                local_state=local_batch["local_state"],
                local_action=local_batch["local_action"],
                next_local_state=local_batch["next_local_state"],
            )
        )

    return updated


# ============================================================
# 3. Ensemble local dynamics model
# ============================================================

class SingleIndependentDynamics(nn.Module):
    """One Gaussian local delta-dynamics network."""

    hidden_dims: Sequence[int]
    out_dim: int = LOCAL_STATE_DIM
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = x

        for width in self.hidden_dims:
            h = nn.relu(nn.Dense(width)(h))

        mu = nn.Dense(self.out_dim, name="delta_mean")(h)
        logvar = nn.Dense(self.out_dim, name="delta_logvar")(h)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)

        return mu, logvar


class EnsembleIndependentDynamics(nn.Module):
    """
    Ensemble of local Gaussian dynamics models.

    Input:
        [p_x, p_y, v_x, v_y, a_i]

    Output per member:
        mean and log-variance of normalized delta z_i
    """

    num_members: int
    hidden_dims: Sequence[int]
    out_dim: int = LOCAL_STATE_DIM
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def setup(self):
        self.members = nn.vmap(
            SingleIndependentDynamics,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_members,
        )(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: (B, local_state_dim + act_dim)

        Returns:
            mu:     (E, B, 4)
            logvar: (E, B, 4)
        """
        return self.members(x)


# ============================================================
# 4. Initialization and training
# ============================================================

def init_independent_transition_model(
    rng: jax.Array,
    act_dim: int,
    cfg: Any,
) -> Tuple[EnsembleIndependentDynamics, TrainState]:
    """
    Initialize one agent's independent ensemble dynamics model.
    """
    model = EnsembleIndependentDynamics(
        num_members=cfg.model_dynamics.num_members,
        hidden_dims=tuple(cfg.model_dynamics.hidden_dims),
        out_dim=LOCAL_STATE_DIM,
        min_logvar=cfg.model_dynamics.min_logvar,
        max_logvar=cfg.model_dynamics.max_logvar,
    )

    dummy_input = jnp.zeros(
        (1, LOCAL_STATE_DIM + act_dim),
        dtype=jnp.float32,
    )

    params = model.init(rng, dummy_input)["params"]

    optimizer = optax.adam(cfg.model_dynamics.lr)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    return model, train_state


def init_independent_transition_models(
    rng: jax.Array,
    num_agents: int,
    act_dim: int,
    cfg: Any,
) -> Tuple[EnsembleIndependentDynamics, List[TrainState]]:
    """
    Initialize one independent ensemble model per agent.

    All models share architecture, but each has separate parameters and optimizer state.
    """
    keys = jax.random.split(rng, num_agents)

    model = None
    train_states = []

    for key in keys:
        model_i, state_i = init_independent_transition_model(
            rng=key,
            act_dim=act_dim,
            cfg=cfg,
        )
        model = model_i
        train_states.append(state_i)

    assert model is not None
    return model, train_states


def gaussian_nll(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """
    Negative log likelihood for a diagonal Gaussian.

    mu/logvar: (E, B, D)
    target:    (E, B, D)
    """
    inv_var = jnp.exp(-logvar)
    inv_var = jnp.clip(inv_var, 1e-6, 1e6)

    squared_error = (mu - target) ** 2
    nll_per_dim = 0.5 * (
        squared_error * inv_var
        + logvar
        + jnp.log(2.0 * jnp.pi)
    )

    return jnp.mean(jnp.sum(nll_per_dim, axis=-1))


def train_independent_transition_step(
    train_state: TrainState,
    local_batch: Dict[str, jnp.ndarray],
    standardizer: LocalStandardizerRS,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    One gradient step for one agent's local ensemble model.
    """

    def loss_fn(params):
        local_state = local_batch["local_state"]
        local_action = local_batch["local_action"]
        delta = local_batch["delta"]

        state_norm = standardizer.norm_local_state(local_state)
        action_norm = standardizer.norm_action(local_action)
        delta_norm = standardizer.norm_delta(delta)

        model_input = jnp.concatenate(
            [state_norm, action_norm],
            axis=-1,
        )

        mu, logvar = train_state.apply_fn(
            {"params": params},
            model_input,
        )

        target = jnp.broadcast_to(delta_norm, mu.shape)

        nll = gaussian_nll(mu, logvar, target)
        mse = jnp.mean((mu - target) ** 2)

        metrics = {
            "transition_nll": nll,
            "transition_mse": mse,
            "mean_logvar": jnp.mean(logvar),
        }

        return nll, metrics

    (_, metrics), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(train_state.params)

    updates, new_opt_state = train_state.tx.update(
        grads,
        train_state.opt_state,
        train_state.params,
    )

    new_params = optax.apply_updates(
        train_state.params,
        updates,
    )

    new_train_state = train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )

    return new_train_state, metrics


def train_all_independent_models(
    train_states: Sequence[TrainState],
    standardizers: Sequence[LocalStandardizerRS],
    batch: Dict[str, Any],
    num_agents: int,
    num_landmarks: int,
    act_dim: int,
) -> Tuple[List[TrainState], List[Dict[str, jnp.ndarray]]]:
    """
    Train every agent's local model once using the same sampled global replay batch.
    """
    updated_states = []
    metrics_per_agent = []

    for agent_id, (train_state, std) in enumerate(
        zip(train_states, standardizers)
    ):
        local_batch = make_local_transition_batch(
            batch=batch,
            agent_id=agent_id,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            act_dim=act_dim,
        )

        new_state, metrics = train_independent_transition_step(
            train_state=train_state,
            local_batch=local_batch,
            standardizer=std,
        )

        updated_states.append(new_state)
        metrics_per_agent.append(metrics)

    return updated_states, metrics_per_agent


# ============================================================
# 5. Prediction and ensemble uncertainty
# ============================================================

def ensemble_moments(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    Moment matching for an ensemble of diagonal Gaussian predictions.

    Args:
        mu:     (E, B, 4)
        logvar: (E, B, 4)

    Returns:
        mean:           (B, 4)
        aleatoric_var:  (B, 4)
        epistemic_var:  (B, 4)
        total_var:      (B, 4)

    All returned quantities are in normalized delta-state coordinates.
    """
    member_var = jnp.exp(logvar)

    mean = jnp.mean(mu, axis=0)
    aleatoric_var = jnp.mean(member_var, axis=0)
    epistemic_var = jnp.mean(
        (mu - mean[None, ...]) ** 2,
        axis=0,
    )

    total_var = aleatoric_var + epistemic_var

    return {
        "mean": mean,
        "aleatoric_var": aleatoric_var,
        "epistemic_var": epistemic_var,
        "total_var": total_var,
    }


def predict_local_next(
    train_state: TrainState,
    standardizer: LocalStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    deterministic: bool = True,
    rng: Optional[jax.Array] = None,
    member_idx: Optional[int] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Predict one agent's next local physical state.

    Args:
        local_state:  (B, 4)
        local_action: (B, act_dim)

    Returns:
        next_local_state: (B, 4)
        prediction_info: dictionary containing ensemble means and variances.

    Important:
        The uncertainty values returned here remain in normalized delta-state
        coordinates. This is intentional, so you can connect them directly
        to your own uncertainty-propagation derivation.
    """
    local_state = jnp.asarray(local_state)
    local_action = jnp.asarray(local_action)

    if local_state.ndim != 2:
        raise ValueError(
            "local_state must have shape (B, 4), "
            f"but got {local_state.shape}."
        )

    if local_action.ndim != 2:
        raise ValueError(
            "local_action must have shape (B, act_dim), "
            f"but got {local_action.shape}."
        )

    state_norm = standardizer.norm_local_state(local_state)
    action_norm = standardizer.norm_action(local_action)

    model_input = jnp.concatenate(
        [state_norm, action_norm],
        axis=-1,
    )

    ensemble_mu, ensemble_logvar = train_state.apply_fn(
        {"params": train_state.params},
        model_input,
    )

    moments = ensemble_moments(
        ensemble_mu,
        ensemble_logvar,
    )

    if deterministic:
        delta_norm = moments["mean"]

    else:
        if rng is None:
            raise ValueError(
                "rng is required when deterministic=False."
            )

        rng, member_key, noise_key = jax.random.split(rng, 3)

        if member_idx is None:
            member_idx = jax.random.randint(
                member_key,
                shape=(),
                minval=0,
                maxval=ensemble_mu.shape[0],
            )

        mu_member = ensemble_mu[member_idx]
        std_member = jnp.exp(0.5 * ensemble_logvar[member_idx])

        delta_norm = mu_member + std_member * jax.random.normal(
            noise_key,
            shape=mu_member.shape,
        )

    delta = standardizer.denorm_delta(delta_norm)
    next_local_state = local_state + delta

    prediction_info = {
        "ensemble_delta_mu_norm": ensemble_mu,
        "ensemble_delta_logvar_norm": ensemble_logvar,
        "delta_mean_norm": moments["mean"],
        "aleatoric_var_norm": moments["aleatoric_var"],
        "epistemic_var_norm": moments["epistemic_var"],
        "total_var_norm": moments["total_var"],
        "delta_mean": standardizer.denorm_delta(moments["mean"]),
    }

    return next_local_state, prediction_info