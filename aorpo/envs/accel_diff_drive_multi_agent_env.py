"""Independent multi-agent acceleration-controlled differential-drive dynamics.

This module intentionally contains no reward, policy, collision response, LiDAR,
or MARL API. Its only purpose is to generate clean local transitions

    (x_t^i, u_t^i, x_{t+1}^i)

for probabilistic dynamics-model training and evaluation.

Local state:
    x = [p_x, p_y, phi, v, omega]

Local action:
    u = [a_t, alpha_z]

Continuous-time model:
    p_x_dot = v cos(phi)
    p_y_dot = v sin(phi)
    phi_dot = omega
    v_dot = a_t
    omega_dot = alpha_z

Each agent evolves independently. This preserves the factorized local-dynamics
assumption used by the estimator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class AccelDiffDriveConfig(NamedTuple):
    num_agents: int = 3
    dt: float = 0.1
    max_steps: int = 500

    position_min: float = -5.0
    position_max: float = 5.0

    min_v: float = 0.0
    max_v: float = 1.0
    max_omega: float = 1.5

    max_linear_acceleration: float = 0.8
    max_angular_acceleration: float = 2.0

    initial_v_max: float = 0.4
    initial_omega_abs_max: float = 0.4

    # Noise is applied to the commanded accelerations before integration.
    linear_acceleration_noise_std: float = 0.0
    angular_acceleration_noise_std: float = 0.0

    # Keep False for delta-learning. Wrapping at +/-pi creates artificial
    # jumps in the target delta_phi.
    wrap_heading: bool = False


class AccelDiffDriveState(NamedTuple):
    positions: Array          # (N, 2)
    headings: Array           # (N,)
    linear_velocities: Array  # (N,)
    angular_velocities: Array # (N,)
    step_count: Array         # scalar int32


def wrap_angle(angle: Array) -> Array:
    """Map angles to [-pi, pi)."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


class AccelDiffDriveMultiAgentEnv:
    """Small functional dynamics environment for system identification."""

    local_state_dim: int = 5
    action_dim: int = 2
    state_order = ("p_x", "p_y", "phi", "v", "omega")
    action_order = ("a_t", "alpha_z")

    def __init__(self, config: AccelDiffDriveConfig):
        if config.num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if config.position_min >= config.position_max:
            raise ValueError("position_min must be smaller than position_max.")
        if config.min_v >= config.max_v:
            raise ValueError("min_v must be smaller than max_v.")
        self.config = config

    @property
    def num_agents(self) -> int:
        return self.config.num_agents

    def reset(self, rng: Array) -> AccelDiffDriveState:
        """Sample an independent initial state for every agent."""
        k_pos, k_phi, k_v, k_w = jax.random.split(rng, 4)
        c = self.config

        positions = jax.random.uniform(
            k_pos,
            shape=(c.num_agents, 2),
            minval=c.position_min,
            maxval=c.position_max,
            dtype=jnp.float32,
        )
        headings = jax.random.uniform(
            k_phi,
            shape=(c.num_agents,),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=jnp.float32,
        )

        initial_v_upper = min(c.initial_v_max, c.max_v)
        linear_velocities = jax.random.uniform(
            k_v,
            shape=(c.num_agents,),
            minval=c.min_v,
            maxval=initial_v_upper,
            dtype=jnp.float32,
        )

        initial_w_upper = min(c.initial_omega_abs_max, c.max_omega)
        angular_velocities = jax.random.uniform(
            k_w,
            shape=(c.num_agents,),
            minval=-initial_w_upper,
            maxval=initial_w_upper,
            dtype=jnp.float32,
        )

        return AccelDiffDriveState(
            positions=positions,
            headings=headings,
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def local_states(self, state: AccelDiffDriveState) -> Array:
        """Return local states with shape (N, 5)."""
        return jnp.concatenate(
            [
                state.positions,
                state.headings[:, None],
                state.linear_velocities[:, None],
                state.angular_velocities[:, None],
            ],
            axis=-1,
        )

    def state_from_local_states(
        self,
        local_states: Array,
        step_count: int | Array = 0,
    ) -> AccelDiffDriveState:
        """Construct an environment state from an (N, 5) local-state array."""
        self._validate_local_states(local_states)
        return AccelDiffDriveState(
            positions=local_states[:, 0:2],
            headings=local_states[:, 2],
            linear_velocities=local_states[:, 3],
            angular_velocities=local_states[:, 4],
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
        )

    def sample_actions(
        self,
        rng: Array,
        linear_scale: float = 1.0,
        angular_scale: float = 1.0,
    ) -> Array:
        """Sample independent uniformly distributed actions, shape (N, 2)."""
        if not 0.0 < linear_scale <= 1.0:
            raise ValueError("linear_scale must lie in (0, 1].")
        if not 0.0 < angular_scale <= 1.0:
            raise ValueError("angular_scale must lie in (0, 1].")

        c = self.config
        k_a, k_alpha = jax.random.split(rng)
        a_t = jax.random.uniform(
            k_a,
            shape=(c.num_agents,),
            minval=-linear_scale * c.max_linear_acceleration,
            maxval=linear_scale * c.max_linear_acceleration,
            dtype=jnp.float32,
        )
        alpha_z = jax.random.uniform(
            k_alpha,
            shape=(c.num_agents,),
            minval=-angular_scale * c.max_angular_acceleration,
            maxval=angular_scale * c.max_angular_acceleration,
            dtype=jnp.float32,
        )
        return jnp.stack([a_t, alpha_z], axis=-1)

    def analytic_next_local_states(
        self,
        local_states: Array,
        actions: Array,
        acceleration_noise: Array | None = None,
    ) -> Array:
        """Apply one midpoint-integration step directly to local arrays.

        Args:
            local_states: (..., 5).
            actions: (..., 2), broadcast-compatible with local_states.
            acceleration_noise: optional (..., 2) noise added to actions.

        Returns:
            next local states with shape (..., 5).
        """
        if local_states.shape[-1] != self.local_state_dim:
            raise ValueError(
                f"Expected local_states[..., 5], got {local_states.shape}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected actions[..., 2], got {actions.shape}.")

        c = self.config
        effective_actions = actions
        if acceleration_noise is not None:
            if acceleration_noise.shape[-1] != self.action_dim:
                raise ValueError("acceleration_noise must end in dimension 2.")
            effective_actions = effective_actions + acceleration_noise

        a_t = jnp.clip(
            effective_actions[..., 0],
            -c.max_linear_acceleration,
            c.max_linear_acceleration,
        )
        alpha_z = jnp.clip(
            effective_actions[..., 1],
            -c.max_angular_acceleration,
            c.max_angular_acceleration,
        )

        px = local_states[..., 0]
        py = local_states[..., 1]
        phi = local_states[..., 2]
        v = local_states[..., 3]
        omega = local_states[..., 4]

        v_next = jnp.clip(v + a_t * c.dt, c.min_v, c.max_v)
        omega_next = jnp.clip(
            omega + alpha_z * c.dt,
            -c.max_omega,
            c.max_omega,
        )

        v_mid = 0.5 * (v + v_next)
        omega_mid = 0.5 * (omega + omega_next)
        phi_mid = phi + 0.5 * omega_mid * c.dt

        px_next = px + v_mid * jnp.cos(phi_mid) * c.dt
        py_next = py + v_mid * jnp.sin(phi_mid) * c.dt
        phi_next = phi + omega_mid * c.dt

        if c.wrap_heading:
            phi_next = wrap_angle(phi_next)

        return jnp.stack(
            [px_next, py_next, phi_next, v_next, omega_next],
            axis=-1,
        )

    def step(
        self,
        rng: Array,
        state: AccelDiffDriveState,
        actions: Array,
    ) -> tuple[AccelDiffDriveState, Array, Array, dict[str, Array]]:
        """Advance all independent agents by one step.

        Returns:
            next_state
            next_local_states: (N, 5)
            done: scalar bool
            info: commanded/effective actions and sampled acceleration noise
        """
        self._validate_actions(actions)
        c = self.config

        k_a, k_alpha = jax.random.split(rng)
        noise = jnp.stack(
            [
                jax.random.normal(
                    k_a, shape=(c.num_agents,), dtype=jnp.float32
                )
                * c.linear_acceleration_noise_std,
                jax.random.normal(
                    k_alpha, shape=(c.num_agents,), dtype=jnp.float32
                )
                * c.angular_acceleration_noise_std,
            ],
            axis=-1,
        )

        local_t = self.local_states(state)
        local_tp1 = self.analytic_next_local_states(
            local_t,
            actions,
            acceleration_noise=noise,
        )

        next_state = self.state_from_local_states(
            local_tp1,
            step_count=state.step_count + 1,
        )
        done = next_state.step_count >= c.max_steps

        info = {
            "commanded_actions": actions,
            "acceleration_noise": noise,
            "effective_actions": actions + noise,
        }
        return next_state, local_tp1, done, info

    def metadata(self) -> dict[str, object]:
        """Metadata suitable for a dynamics-model checkpoint."""
        c = self.config
        return {
            "env_name": "accel_diff_drive",
            "num_agents": c.num_agents,
            "local_state_dim": self.local_state_dim,
            "action_dim": self.action_dim,
            "dt": c.dt,
            "state_order": list(self.state_order),
            "action_order": list(self.action_order),
            "min_v": c.min_v,
            "max_v": c.max_v,
            "max_omega": c.max_omega,
            "max_linear_acceleration": c.max_linear_acceleration,
            "max_angular_acceleration": c.max_angular_acceleration,
            "linear_acceleration_noise_std":
                c.linear_acceleration_noise_std,
            "angular_acceleration_noise_std":
                c.angular_acceleration_noise_std,
            "wrap_heading": c.wrap_heading,
            "independent_local_dynamics": True,
        }

    def _validate_actions(self, actions: Array) -> None:
        expected = (self.config.num_agents, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"Expected actions shape {expected}, got {actions.shape}."
            )

    def _validate_local_states(self, local_states: Array) -> None:
        expected = (self.config.num_agents, self.local_state_dim)
        if tuple(local_states.shape) != expected:
            raise ValueError(
                f"Expected local_states shape {expected}, "
                f"got {local_states.shape}."
            )
