#!/usr/bin/env python3
"""Verify the clean environment and analytic open-loop prediction."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aorpo.data.accel_diff_drive_dataset import (
    collect_piecewise_constant_trajectories,
    flatten_local_transitions,
)
from aorpo.envs.accel_diff_drive_multi_agent_env import (
    AccelDiffDriveConfig,
    AccelDiffDriveMultiAgentEnv,
)
from aorpo.evaluation.dynamics_metrics import (
    one_step_metrics,
    rollout_metrics,
    rollout_predictions,
)


def main() -> None:
    env = AccelDiffDriveMultiAgentEnv(
        AccelDiffDriveConfig(
            num_agents=3,
            dt=0.1,
            max_steps=60,
            linear_acceleration_noise_std=0.0,
            angular_acceleration_noise_std=0.0,
            wrap_heading=False,
        )
    )

    dataset = collect_piecewise_constant_trajectories(
        env=env,
        rng=jax.random.PRNGKey(0),
        num_trajectories=4,
        horizon=60,
        hold_steps_min=5,
        hold_steps_max=12,
    )

    state_t, action_t, state_tp1, delta_t = flatten_local_transitions(
        dataset
    )
    analytic_tp1 = env.analytic_next_local_states(state_t, action_t)
    metrics = one_step_metrics(analytic_tp1, state_tp1)

    assert state_t.shape[-1] == 5
    assert action_t.shape[-1] == 2
    assert delta_t.shape[-1] == 5
    assert float(metrics["state_rmse"]) < 1e-6

    true_trajectory = dataset.local_states[0]
    actions = dataset.actions[0]

    predicted_trajectory = rollout_predictions(
        env.analytic_next_local_states,
        true_trajectory[0],
        actions,
    )
    rollout = rollout_metrics(
        predicted_trajectory,
        true_trajectory,
        horizons=(1, 5, 10, 25, 50),
    )

    for horizon_metrics in rollout.values():
        assert float(horizon_metrics["state_rmse"]) < 2e-6

    print("Smoke test passed.")
    print("dataset local_states:", dataset.local_states.shape)
    print("dataset actions:", dataset.actions.shape)
    print("one-step state RMSE:", float(metrics["state_rmse"]))
    print(
        "50-step position RMSE:",
        float(rollout["50"]["position_rmse"]),
    )


if __name__ == "__main__":
    main()
