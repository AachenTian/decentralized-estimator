"""Physics-structured probabilistic dynamics for acceleration-controlled diff drive.

Local state
-----------
    x = [p_x, p_y, phi, v, omega]

Local action
------------
    u = [a_t, alpha_z]

Neural input features
---------------------
    [sin(phi), cos(phi), v, omega, a_t, alpha_z]

Neural stochastic target
------------------------
    [delta_v, delta_omega]

The network does not relearn translation invariance or trigonometric
kinematics. It predicts only the uncertain velocity increments. Full
next-state means and 5x5 process covariances are reconstructed through the
known midpoint integration map.

The model architecture remains an ensemble of MLP Gaussian predictors.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

from aorpo.estimator.infoprop import infoprop_ensemble_prediction


LOCAL_STATE_DIM = 5
ACTION_DIM = 2
STATE_FEATURE_DIM = 4
MODEL_INPUT_DIM = STATE_FEATURE_DIM + ACTION_DIM
DYNAMIC_TARGET_DIM = 2


class StructuredEnsembleGaussianPrediction(NamedTuple):
    """Per-member full next-state Gaussians in physical coordinates."""

    ensemble_next_means: jnp.ndarray       # (B, E, 5)
    ensemble_next_covariances: jnp.ndarray # (B, E, 5, 5)
    ensemble_dynamic_delta_means: jnp.ndarray # (B, E, 2)
    ensemble_dynamic_delta_covariances: jnp.ndarray # (B, E, 2, 2)


class StructuredInfopropPrediction(NamedTuple):
    """Infoprop-style post-processing in full state coordinates."""

    next_mean: jnp.ndarray
    process_covariance: jnp.ndarray
    fused_covariance: jnp.ndarray
    epistemic_covariance: jnp.ndarray
    total_predictive_covariance: jnp.ndarray
    posterior_mean: jnp.ndarray
    posterior_covariance: jnp.ndarray


def state_features(local_state: jnp.ndarray) -> jnp.ndarray:
    """Return [sin(phi), cos(phi), v, omega]."""
    local_state = jnp.asarray(local_state)
    if local_state.shape[-1] != LOCAL_STATE_DIM:
        raise ValueError(
            f"Expected local state dimension {LOCAL_STATE_DIM}, "
            f"got {local_state.shape[-1]}."
        )
    phi = local_state[..., 2]
    v = local_state[..., 3]
    omega = local_state[..., 4]
    return jnp.stack(
        [jnp.sin(phi), jnp.cos(phi), v, omega],
        axis=-1,
    )


def dynamic_delta_target(
    local_state: jnp.ndarray,
    next_local_state: jnp.ndarray,
) -> jnp.ndarray:
    """Return [delta_v, delta_omega]."""
    local_state = jnp.asarray(local_state)
    next_local_state = jnp.asarray(next_local_state)
    return next_local_state[..., 3:5] - local_state[..., 3:5]


def make_structured_transition_batch(
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    next_local_state: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Build a direct per-agent transition batch."""
    local_state = jnp.asarray(local_state)
    local_action = jnp.asarray(local_action)
    next_local_state = jnp.asarray(next_local_state)

    if local_state.ndim != 2 or next_local_state.ndim != 2:
        raise ValueError("States must have shape (B, 5).")
    if local_action.ndim != 2:
        raise ValueError("Actions must have shape (B, 2).")
    if local_state.shape != next_local_state.shape:
        raise ValueError("Current and next state shapes must match.")
    if local_state.shape[0] != local_action.shape[0]:
        raise ValueError("State/action batch sizes must match.")
    if local_state.shape[-1] != LOCAL_STATE_DIM:
        raise ValueError("Expected five-dimensional local state.")
    if local_action.shape[-1] != ACTION_DIM:
        raise ValueError("Expected two-dimensional local action.")

    return {
        "local_state": local_state,
        "local_action": local_action,
        "next_local_state": next_local_state,
        "target": dynamic_delta_target(local_state, next_local_state),
    }


@struct.dataclass
class StructuredDiffDriveStandardizerRS:
    """Running statistics for structured diff-drive dynamics."""

    feature_stats: running_statistics.RunningStatisticsState
    action_stats: running_statistics.RunningStatisticsState
    target_stats: running_statistics.RunningStatisticsState

    @classmethod
    def create(cls) -> "StructuredDiffDriveStandardizerRS":
        return cls(
            feature_stats=running_statistics.init_state(
                jnp.zeros((STATE_FEATURE_DIM,), dtype=jnp.float32)
            ),
            action_stats=running_statistics.init_state(
                jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
            ),
            target_stats=running_statistics.init_state(
                jnp.zeros((DYNAMIC_TARGET_DIM,), dtype=jnp.float32)
            ),
        )

    def update(
        self,
        local_state: jnp.ndarray,
        local_action: jnp.ndarray,
        next_local_state: jnp.ndarray,
    ) -> "StructuredDiffDriveStandardizerRS":
        features = state_features(local_state)
        target = dynamic_delta_target(local_state, next_local_state)
        return StructuredDiffDriveStandardizerRS(
            feature_stats=running_statistics.update(
                self.feature_stats,
                features,
            ),
            action_stats=running_statistics.update(
                self.action_stats,
                local_action,
            ),
            target_stats=running_statistics.update(
                self.target_stats,
                target,
            ),
        )

    def norm_features(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.feature_stats)

    def norm_action(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.action_stats)

    def norm_target(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.normalize(x, self.target_stats)

    def denorm_target(self, x: jnp.ndarray) -> jnp.ndarray:
        return running_statistics.denormalize(x, self.target_stats)


def init_structured_standardizers(
    num_agents: int,
) -> List[StructuredDiffDriveStandardizerRS]:
    return [
        StructuredDiffDriveStandardizerRS.create()
        for _ in range(num_agents)
    ]


class SingleStructuredDiffDriveDynamics(nn.Module):
    """One Gaussian MLP for normalized [delta_v, delta_omega]."""

    hidden_dims: Sequence[int]
    min_logvar: float = -6.0
    max_logvar: float = 0.5

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = x
        for width in self.hidden_dims:
            h = nn.relu(nn.Dense(width)(h))

        mu = nn.Dense(
            DYNAMIC_TARGET_DIM,
            name="dynamic_delta_mean",
        )(h)
        logvar = nn.Dense(
            DYNAMIC_TARGET_DIM,
            name="dynamic_delta_logvar",
        )(h)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar


class EnsembleStructuredDiffDriveDynamics(nn.Module):
    """Ensemble of Gaussian MLPs with unchanged hidden architecture."""

    num_members: int
    hidden_dims: Sequence[int]
    min_logvar: float = -6.0
    max_logvar: float = 0.5

    def setup(self):
        self.members = nn.vmap(
            SingleStructuredDiffDriveDynamics,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_members,
        )(
            hidden_dims=self.hidden_dims,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.members(x)


def init_structured_transition_model(
    rng: jax.Array,
    cfg: Any,
) -> tuple[EnsembleStructuredDiffDriveDynamics, TrainState]:
    model = EnsembleStructuredDiffDriveDynamics(
        num_members=int(cfg.model_dynamics.num_members),
        hidden_dims=tuple(cfg.model_dynamics.hidden_dims),
        min_logvar=float(cfg.model_dynamics.min_logvar),
        max_logvar=float(cfg.model_dynamics.max_logvar),
    )
    dummy_input = jnp.zeros((1, MODEL_INPUT_DIM), dtype=jnp.float32)
    params = model.init(rng, dummy_input)["params"]
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(float(cfg.model_dynamics.lr)),
    )
    return model, train_state


def init_structured_transition_models(
    rng: jax.Array,
    num_agents: int,
    cfg: Any,
) -> tuple[
    EnsembleStructuredDiffDriveDynamics,
    List[TrainState],
]:
    keys = jax.random.split(rng, num_agents)
    model = None
    states: List[TrainState] = []
    for key in keys:
        model_i, state_i = init_structured_transition_model(key, cfg)
        model = model_i
        states.append(state_i)
    assert model is not None
    return model, states


def gaussian_nll(
    mu: jnp.ndarray,
    logvar: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Diagonal Gaussian NLL for tensors shaped (E, B, 2)."""
    inv_var = jnp.clip(jnp.exp(-logvar), 1.0e-6, 1.0e6)
    squared_error = (mu - target) ** 2
    nll_per_dim = 0.5 * (
        squared_error * inv_var
        + logvar
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.mean(jnp.sum(nll_per_dim, axis=-1))


def train_structured_transition_step(
    train_state: TrainState,
    batch: dict[str, jnp.ndarray],
    standardizer: StructuredDiffDriveStandardizerRS,
    mean_mse_weight: float = 0.1,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """One gradient update for one agent ensemble."""

    def loss_fn(params):
        features = standardizer.norm_features(
            state_features(batch["local_state"])
        )
        actions = standardizer.norm_action(batch["local_action"])
        target = standardizer.norm_target(batch["target"])
        model_input = jnp.concatenate([features, actions], axis=-1)

        mu, logvar = train_state.apply_fn(
            {"params": params},
            model_input,
        )
        target_ensemble = jnp.broadcast_to(target, mu.shape)
        nll = gaussian_nll(mu, logvar, target_ensemble)
        mean_prediction = jnp.mean(mu, axis=0)
        mean_mse = jnp.mean((mean_prediction - target) ** 2)
        loss = nll + mean_mse_weight * mean_mse
        metrics = {
            "loss": loss,
            "transition_nll": nll,
            "ensemble_mean_mse": mean_mse,
            "mean_logvar": jnp.mean(logvar),
        }
        return loss, metrics

    (_, metrics), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(train_state.params)

    updates, new_opt_state = train_state.tx.update(
        grads,
        train_state.opt_state,
        train_state.params,
    )
    new_state = train_state.replace(
        params=optax.apply_updates(train_state.params, updates),
        opt_state=new_opt_state,
    )
    return new_state, metrics


def structured_next_state_from_dynamic_delta(
    local_state: jnp.ndarray,
    dynamic_delta: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
) -> jnp.ndarray:
    """Map [delta_v, delta_omega] to full next state by midpoint integration."""
    px = local_state[..., 0]
    py = local_state[..., 1]
    phi = local_state[..., 2]
    v = local_state[..., 3]
    omega = local_state[..., 4]

    delta_v = dynamic_delta[..., 0]
    delta_omega = dynamic_delta[..., 1]

    v_next = jnp.clip(v + delta_v, min_v, max_v)
    omega_next = jnp.clip(
        omega + delta_omega,
        -max_omega,
        max_omega,
    )

    v_mid = 0.5 * (v + v_next)
    omega_mid = 0.5 * (omega + omega_next)
    phi_mid = phi + 0.5 * omega_mid * dt

    px_next = px + v_mid * jnp.cos(phi_mid) * dt
    py_next = py + v_mid * jnp.sin(phi_mid) * dt
    phi_next = phi + omega_mid * dt

    return jnp.stack(
        [px_next, py_next, phi_next, v_next, omega_next],
        axis=-1,
    )


def _target_scale(
    standardizer: StructuredDiffDriveStandardizerRS,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    zeros = jnp.zeros((1, DYNAMIC_TARGET_DIM), dtype=dtype)
    ones = jnp.ones((1, DYNAMIC_TARGET_DIM), dtype=dtype)
    return (
        standardizer.denorm_target(ones)
        - standardizer.denorm_target(zeros)
    )[0]


def _full_state_jacobian(
    local_state: jnp.ndarray,
    dynamic_delta: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
) -> jnp.ndarray:
    """Jacobian d x_next / d [delta_v, delta_omega]."""

    def single(state, delta):
        return structured_next_state_from_dynamic_delta(
            state,
            delta,
            dt=dt,
            min_v=min_v,
            max_v=max_v,
            max_omega=max_omega,
        )

    flat_state = local_state.reshape((-1, LOCAL_STATE_DIM))
    flat_delta = dynamic_delta.reshape((-1, DYNAMIC_TARGET_DIM))
    flat_jacobian = jax.vmap(
        jax.jacfwd(single, argnums=1)
    )(flat_state, flat_delta)
    return flat_jacobian.reshape(
        local_state.shape[:-1]
        + (LOCAL_STATE_DIM, DYNAMIC_TARGET_DIM)
    )


def predict_structured_ensemble_next_gaussians(
    train_state: TrainState,
    standardizer: StructuredDiffDriveStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
    full_state_covariance_floor: float = 1.0e-9,
) -> StructuredEnsembleGaussianPrediction:
    """Return full 5D member means/covariances after analytic reconstruction."""
    local_state = jnp.asarray(local_state)
    local_action = jnp.asarray(local_action)

    features = standardizer.norm_features(state_features(local_state))
    actions = standardizer.norm_action(local_action)
    model_input = jnp.concatenate([features, actions], axis=-1)

    mu_norm_ebd, logvar_norm_ebd = train_state.apply_fn(
        {"params": train_state.params},
        model_input,
    )

    mu_norm_bed = jnp.swapaxes(mu_norm_ebd, 0, 1)
    logvar_norm_bed = jnp.swapaxes(logvar_norm_ebd, 0, 1)

    delta_means = standardizer.denorm_target(mu_norm_bed)
    target_scale = _target_scale(standardizer, local_state.dtype)
    delta_variances = (
        jnp.exp(logvar_norm_bed)
        * (target_scale ** 2)
    )
    delta_covariances = (
        jnp.eye(DYNAMIC_TARGET_DIM, dtype=local_state.dtype)[
            None, None, :, :
        ]
        * delta_variances[..., :, None]
    )

    expanded_state = jnp.broadcast_to(
        local_state[:, None, :],
        delta_means.shape[:-1] + (LOCAL_STATE_DIM,),
    )
    next_means = structured_next_state_from_dynamic_delta(
        expanded_state,
        delta_means,
        dt=dt,
        min_v=min_v,
        max_v=max_v,
        max_omega=max_omega,
    )

    jacobian = _full_state_jacobian(
        expanded_state,
        delta_means,
        dt=dt,
        min_v=min_v,
        max_v=max_v,
        max_omega=max_omega,
    )
    full_covariances = jnp.einsum(
        "...ik,...kl,...jl->...ij",
        jacobian,
        delta_covariances,
        jacobian,
    )
    full_covariances = (
        full_covariances
        + full_state_covariance_floor
        * jnp.eye(LOCAL_STATE_DIM, dtype=local_state.dtype)
    )

    return StructuredEnsembleGaussianPrediction(
        ensemble_next_means=next_means,
        ensemble_next_covariances=full_covariances,
        ensemble_dynamic_delta_means=delta_means,
        ensemble_dynamic_delta_covariances=delta_covariances,
    )


def moment_match_structured_prediction(
    raw: StructuredEnsembleGaussianPrediction,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Arithmetic ensemble moment matching."""
    member_means = raw.ensemble_next_means
    member_covariances = raw.ensemble_next_covariances
    mean = jnp.mean(member_means, axis=1)
    aleatoric = jnp.mean(member_covariances, axis=1)
    centered = member_means - mean[:, None, :]
    epistemic = jnp.einsum(
        "bed,bef->bdf",
        centered,
        centered,
    ) / float(member_means.shape[1])
    total = aleatoric + epistemic
    return mean, aleatoric, epistemic, total


def predict_structured_infoprop(
    train_state: TrainState,
    standardizer: StructuredDiffDriveStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
    epistemic_process_scale: float = 1.0,
    model_sample: Optional[jnp.ndarray] = None,
    jitter: float = 1.0e-6,
) -> StructuredInfopropPrediction:
    """Apply CI fusion and separate aleatoric/epistemic uncertainty.

    Important:
        If model_sample is None, the Infoprop posterior mean equals the fused
        mean because the innovation is zero. The posterior is therefore a
        covariance diagnostic, not a magical mean correction.
    """
    raw = predict_structured_ensemble_next_gaussians(
        train_state=train_state,
        standardizer=standardizer,
        local_state=local_state,
        local_action=local_action,
        dt=dt,
        min_v=min_v,
        max_v=max_v,
        max_omega=max_omega,
    )
    result = infoprop_ensemble_prediction(
        ensemble_means=raw.ensemble_next_means,
        ensemble_covariances=raw.ensemble_next_covariances,
        model_sample=model_sample,
        jitter=jitter,
    )
    total = result.fused_covariance + result.epistemic_covariance
    process = (
        result.fused_covariance
        + epistemic_process_scale * result.epistemic_covariance
    )
    return StructuredInfopropPrediction(
        next_mean=result.fused_mean,
        process_covariance=process,
        fused_covariance=result.fused_covariance,
        epistemic_covariance=result.epistemic_covariance,
        total_predictive_covariance=total,
        posterior_mean=result.posterior_mean,
        posterior_covariance=result.posterior_covariance,
    )


def predict_structured_next(
    train_state: TrainState,
    standardizer: StructuredDiffDriveStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
    prediction_mode: str = "ensemble_mean",
    epistemic_process_scale: float = 1.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Deterministic next-state prediction for rollouts."""
    raw = predict_structured_ensemble_next_gaussians(
        train_state=train_state,
        standardizer=standardizer,
        local_state=local_state,
        local_action=local_action,
        dt=dt,
        min_v=min_v,
        max_v=max_v,
        max_omega=max_omega,
    )

    if prediction_mode == "ensemble_mean":
        mean, aleatoric, epistemic, total = (
            moment_match_structured_prediction(raw)
        )
        return mean, {
            "aleatoric_covariance": aleatoric,
            "epistemic_covariance": epistemic,
            "total_predictive_covariance": total,
            "process_covariance": (
                aleatoric
                + epistemic_process_scale * epistemic
            ),
        }

    if prediction_mode == "infoprop_ci":
        result = infoprop_ensemble_prediction(
            ensemble_means=raw.ensemble_next_means,
            ensemble_covariances=raw.ensemble_next_covariances,
        )
        total = result.fused_covariance + result.epistemic_covariance
        return result.fused_mean, {
            "aleatoric_covariance": result.fused_covariance,
            "epistemic_covariance": result.epistemic_covariance,
            "total_predictive_covariance": total,
            "process_covariance": (
                result.fused_covariance
                + epistemic_process_scale
                * result.epistemic_covariance
            ),
            "posterior_mean": result.posterior_mean,
            "posterior_covariance": result.posterior_covariance,
        }

    raise ValueError(
        "prediction_mode must be 'ensemble_mean' or 'infoprop_ci'."
    )
