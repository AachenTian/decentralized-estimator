from __future__ import annotations

from typing import Any, Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from aorpo.agents.independent_dynamics import (
    extract_agent_action,
    extract_local_kinematic_state,
    init_independent_transition_models,
    init_local_standardizers,
    make_local_transition_batch,
    predict_local_next,
    train_all_independent_models,
    update_local_standardizers,
)

from aorpo.estimator.uncertainty import (
    compute_local_action_jacobian,
    compute_local_state_jacobian,
    covariance_trace_per_dim,
    predict_local_mean_and_process_covariance,
    propagate_local_covariance,
)

from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import (
    env_step,
    make_mpe_env,
)
from aorpo.utils.replay import ReplayBuffer, manual_flatten_state


def stack_pytrees(items: List[Any]) -> Any:
    """Stack a list of JAX pytrees along the first axis."""
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *items,
    )


def random_action(
    key: jax.Array,
    shape: Tuple[int, ...],
) -> jnp.ndarray:
    """
    Initial behavior policy.

    First dynamics sanity check does not require an RL policy.
    Sample bounded random actions to obtain state-action coverage.
    """
    return jax.random.uniform(
        key,
        shape=shape,
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )


def collect_random_transitions(
    env: Any,
    key: jax.Array,
    cfg: DictConfig,
    num_steps: int,
) -> Tuple[Dict[str, Any], jax.Array]:
    """
    Collect real transitions with random actions.

    This intentionally does not use the old policy/Q/model rollout stack.
    It only produces a batch compatible with the existing ReplayBuffer.
    """
    num_opponents = cfg.train.num_opponents
    act_dim = cfg.env.act_dim
    max_steps = cfg.train.max_steps

    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    states = []
    observations = []
    ego_actions = []
    opponent_actions = []
    next_states = []
    next_observations = []
    rewards = []
    dones_list = []

    for _ in range(num_steps):
        key, ego_key, opp_key, step_key = jax.random.split(key, 4)

        a_ego = random_action(
            ego_key,
            shape=(act_dim,),
        )

        a_opp = random_action(
            opp_key,
            shape=(num_opponents * act_dim,),
        )

        next_state, next_obs, rew, dones, _ = env_step(
            env=env,
            state=state,
            a_ego=a_ego,
            a_opps=a_opp,
            key=step_key,
        )

        states.append(state)
        observations.append(obs)
        ego_actions.append(a_ego)
        opponent_actions.append(a_opp)
        next_states.append(next_state)
        next_observations.append(next_obs)
        rewards.append(rew)
        dones_list.append(dones)

        # Keep the terminal transition, then reset for the next sample.
        if bool(next_state.step >= max_steps):
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
        else:
            state, obs = next_state, next_obs

    batch = {
        "state": stack_pytrees(states),
        "obs": stack_pytrees(observations),
        "a_ego": jnp.stack(ego_actions, axis=0),
        "a_opp": jnp.stack(opponent_actions, axis=0),
        "next_state": stack_pytrees(next_states),
        "next_obs": stack_pytrees(next_observations),
        "rew": stack_pytrees(rewards),
        "dones": stack_pytrees(dones_list),
    }

    return batch, key


def add_collected_batch(
    replay: ReplayBuffer,
    batch: Dict[str, Any],
    cfg: DictConfig,
) -> ReplayBuffer:
    return replay.add_batch(batch, cfg)


def evaluate_one_step(
    model_states: List[Any],
    standardizers: List[Any],
    eval_replay: ReplayBuffer,
    key: jax.Array,
    cfg: DictConfig,
) -> Tuple[Dict[str, float], jax.Array]:
    """
    Evaluate deterministic one-step local prediction on a separate dataset.
    """
    num_agents = cfg.train.num_opponents + 1
    num_landmarks = cfg.train.num_landmark
    act_dim = cfg.env.act_dim

    key, sample_key = jax.random.split(key)

    eval_batch_size = min(
        int(cfg.independent_dynamics.batch_size),
        len(eval_replay),
    )

    batch = eval_replay.sample(
        sample_key,
        batch_size=eval_batch_size,
        opp_num=cfg.train.num_opponents,
    )

    metrics: Dict[str, float] = {}
    all_mse = []
    all_zero_delta_mse = []

    for agent_id in range(num_agents):
        local_batch = make_local_transition_batch(
            batch=batch,
            agent_id=agent_id,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            act_dim=act_dim,
        )

        pred_next, pred_info = predict_local_next(
            train_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            local_state=local_batch["local_state"],
            local_action=local_batch["local_action"],
            deterministic=True,
        )

        target_next = local_batch["next_local_state"]

        # Zero-delta / keep-state baseline:
        # predict z_{t+1} = z_t
        baseline_next = local_batch["local_state"]

        zero_delta_mse = jnp.mean(
            (baseline_next - target_next) ** 2
        )

        zero_delta_pos_mse = jnp.mean(
            (baseline_next[:, :2] - target_next[:, :2]) ** 2
        )

        zero_delta_vel_mse = jnp.mean(
            (baseline_next[:, 2:] - target_next[:, 2:]) ** 2
        )

        mse = jnp.mean((pred_next - target_next) ** 2)
        pos_mse = jnp.mean(
            (pred_next[:, :2] - target_next[:, :2]) ** 2
        )
        vel_mse = jnp.mean(
            (pred_next[:, 2:] - target_next[:, 2:]) ** 2
        )

        mean_total_var = jnp.mean(pred_info["total_var_norm"])

        metrics[f"agent_{agent_id}/one_step_mse"] = float(mse)
        metrics[f"agent_{agent_id}/position_mse"] = float(pos_mse)
        metrics[f"agent_{agent_id}/velocity_mse"] = float(vel_mse)
        metrics[f"agent_{agent_id}/mean_total_var_norm"] = float(
            mean_total_var
        )

        metrics[f"agent_{agent_id}/zero_delta_mse"] = float(
            zero_delta_mse
        )
        metrics[f"agent_{agent_id}/zero_delta_position_mse"] = float(
            zero_delta_pos_mse
        )
        metrics[f"agent_{agent_id}/zero_delta_velocity_mse"] = float(
            zero_delta_vel_mse
        )

        all_mse.append(mse)
        all_zero_delta_mse.append(zero_delta_mse)

    metrics["mean_one_step_mse"] = float(jnp.mean(jnp.stack(all_mse)))
    metrics["mean_zero_delta_mse"] = float(
        jnp.mean(jnp.stack(all_zero_delta_mse))
    )

    return metrics, key

def flatten_trajectory_states(
    trajectory_batch: Dict[str, Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a collected trajectory into flattened state arrays.

    The collector stores JAXMarl State pytrees, while the local dynamics
    utilities expect arrays with shape (T, full_state_dim).
    """
    raw_state = trajectory_batch.get("state")
    raw_next_state = trajectory_batch.get("next_state")

    if raw_state is None:
        raise ValueError(
            "trajectory_batch['state'] is None. "
            "Check where eval_batch is assigned before multi-step evaluation."
        )

    if raw_next_state is None:
        raise ValueError(
            "trajectory_batch['next_state'] is None. "
            "Check where eval_batch is assigned before multi-step evaluation."
        )

    if hasattr(raw_state, "p_pos"):
        flat_state = jax.vmap(manual_flatten_state)(raw_state)
        flat_next_state = jax.vmap(manual_flatten_state)(raw_next_state)
    else:
        flat_state = jnp.asarray(raw_state)
        flat_next_state = jnp.asarray(raw_next_state)

    if flat_state.ndim != 2:
        raise ValueError(
            f"Expected flat state trajectory with shape (T, D), got {flat_state.shape}."
        )

    if flat_next_state.ndim != 2:
        raise ValueError(
            "Expected flat next-state trajectory with shape (T, D), "
            f"got {flat_next_state.shape}."
        )

    return flat_state, flat_next_state


def find_valid_rollout_starts(
    flat_states: jnp.ndarray,
    flat_next_states: jnp.ndarray,
    horizon: int,
) -> List[int]:
    """
    Find rollout starts whose horizon does not cross an environment reset.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}.")

    state_steps = flat_states[:, -1]
    next_steps = flat_next_states[:, -1]

    total_steps = int(flat_states.shape[0])
    valid_starts = []

    for start in range(total_steps - horizon + 1):
        if horizon == 1:
            valid_starts.append(start)
            continue

        is_continuous = jnp.all(
            state_steps[start + 1:start + horizon]
            == next_steps[start:start + horizon - 1]
        )

        if bool(is_continuous):
            valid_starts.append(start)

    return valid_starts

def make_isotropic_action_covariance(
    batch_size: int,
    action_dim: int,
    action_variance: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Create a manually specified isotropic action covariance.

    This helper is only a placeholder for future experiments.
    It should later be replaced by a covariance derived from
    the synchronized stochastic policy.
    """
    if action_variance < 0.0:
        raise ValueError(
            f"action_variance must be non-negative, got {action_variance}."
        )

    covariance = action_variance * jnp.eye(
        action_dim,
        dtype=dtype,
    )

    return jnp.broadcast_to(
        covariance,
        (batch_size, action_dim, action_dim),
    )


def evaluate_multistep_oracle_actions(
    model_states: List[Any],
    standardizers: List[Any],
    trajectory_batch: Dict[str, Any],
    cfg: DictConfig,
    horizon: int,
    num_rollouts: int,
    include_action_uncertainty: bool,
    manual_action_variance: float,
    initial_measurement_variance: float,
) -> Dict[str, jnp.ndarray]:
    """
    Evaluate multi-step local dynamics rollouts with covariance propagation.

    The rollout starts from the real local state. This represents a state
    estimate immediately after direct communication or direct observation.

    Actions are taken from the recorded real trajectory. Therefore,
    include_action_uncertainty should normally be False in this evaluation.

    When include_action_uncertainty is enabled, a manually specified
    isotropic action covariance is used as a temporary placeholder.
    """
    num_agents = cfg.train.num_opponents + 1
    num_landmarks = cfg.train.num_landmark
    act_dim = cfg.env.act_dim

    flat_states, flat_next_states = flatten_trajectory_states(
        trajectory_batch
    )

    valid_starts = find_valid_rollout_starts(
        flat_states=flat_states,
        flat_next_states=flat_next_states,
        horizon=horizon,
    )

    if not valid_starts:
        raise RuntimeError(
            "No valid contiguous rollout segments were found. "
            "Try reducing rollout_horizon."
        )

    stride = max(1, len(valid_starts) // num_rollouts)
    selected_starts = valid_starts[::stride][:num_rollouts]

    if not selected_starts:
        raise RuntimeError("No rollout start was selected.")

    metrics: Dict[str, jnp.ndarray] = {}

    for agent_id in range(num_agents):
        true_states = extract_local_kinematic_state(
            flat_state=flat_states,
            agent_id=agent_id,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
        )

        true_next_states = extract_local_kinematic_state(
            flat_state=flat_next_states,
            agent_id=agent_id,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
        )

        true_actions = extract_agent_action(
            batch=trajectory_batch,
            agent_id=agent_id,
            num_agents=num_agents,
            act_dim=act_dim,
        )

        total_mse_by_horizon = [[] for _ in range(horizon)]
        position_mse_by_horizon = [[] for _ in range(horizon)]
        velocity_mse_by_horizon = [[] for _ in range(horizon)]

        estimated_mse_by_horizon = [[] for _ in range(horizon)]
        covariance_trace_by_horizon = [[] for _ in range(horizon)]

        for start in selected_starts:
            predicted_state = true_states[start:start + 1]

            state_dim = predicted_state.shape[-1]

            predicted_covariance = (
                initial_measurement_variance
                * jnp.eye(
                    state_dim,
                    dtype=predicted_state.dtype,
                )
            )[None, ...]

            for h in range(horizon):
                action_t = true_actions[
                    start + h:start + h + 1
                ]

                current_predicted_state = predicted_state

                predicted_state, process_covariance, _ = (
                    predict_local_mean_and_process_covariance(
                        train_state=model_states[agent_id],
                        standardizer=standardizers[agent_id],
                        local_state=current_predicted_state,
                        local_action=action_t,
                    )
                )

                state_jacobian = compute_local_state_jacobian(
                    train_state=model_states[agent_id],
                    standardizer=standardizers[agent_id],
                    local_state=current_predicted_state,
                    local_action=action_t,
                )

                if include_action_uncertainty:
                    action_jacobian = compute_local_action_jacobian(
                        train_state=model_states[agent_id],
                        standardizer=standardizers[agent_id],
                        local_state=current_predicted_state,
                        local_action=action_t,
                    )

                    action_covariance = make_isotropic_action_covariance(
                        batch_size=1,
                        action_dim=act_dim,
                        action_variance=manual_action_variance,
                        dtype=predicted_state.dtype,
                    )

                    predicted_covariance = propagate_local_covariance(
                        state_covariance=predicted_covariance,
                        state_jacobian=state_jacobian,
                        process_covariance=process_covariance,
                        action_jacobian=action_jacobian,
                        action_covariance=action_covariance,
                        include_action_uncertainty=True,
                    )
                else:
                    predicted_covariance = propagate_local_covariance(
                        state_covariance=predicted_covariance,
                        state_jacobian=state_jacobian,
                        process_covariance=process_covariance,
                        include_action_uncertainty=False,
                    )

                target_state = true_next_states[
                    start + h:start + h + 1
                ]

                error = predicted_state - target_state

                total_mse_by_horizon[h].append(
                    jnp.mean(error ** 2)
                )

                position_mse_by_horizon[h].append(
                    jnp.mean(error[:, :2] ** 2)
                )

                velocity_mse_by_horizon[h].append(
                    jnp.mean(error[:, 2:] ** 2)
                )

                estimated_mse = covariance_trace_per_dim(
                    predicted_covariance
                )[0]

                covariance_trace = jnp.trace(
                    predicted_covariance[0]
                )

                estimated_mse_by_horizon[h].append(
                    estimated_mse
                )

                covariance_trace_by_horizon[h].append(
                    covariance_trace
                )

        metrics[f"agent_{agent_id}/multistep_mse"] = jnp.asarray(
            [
                jnp.mean(jnp.asarray(values))
                for values in total_mse_by_horizon
            ]
        )

        metrics[f"agent_{agent_id}/multistep_position_mse"] = jnp.asarray(
            [
                jnp.mean(jnp.asarray(values))
                for values in position_mse_by_horizon
            ]
        )

        metrics[f"agent_{agent_id}/multistep_velocity_mse"] = jnp.asarray(
            [
                jnp.mean(jnp.asarray(values))
                for values in velocity_mse_by_horizon
            ]
        )

        metrics[f"agent_{agent_id}/estimated_mse_from_covariance"] = (
            jnp.asarray(
                [
                    jnp.mean(jnp.asarray(values))
                    for values in estimated_mse_by_horizon
                ]
            )
        )

        metrics[f"agent_{agent_id}/covariance_trace"] = jnp.asarray(
            [
                jnp.mean(jnp.asarray(values))
                for values in covariance_trace_by_horizon
            ]
        )

    return metrics

@hydra.main(
    config_path="aorpo/configs",
    config_name="train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print("\n===== Independent Dynamics Configuration =====")
    print(OmegaConf.to_yaml(cfg.independent_dynamics))

    rng = jax.random.PRNGKey(cfg.seed)

    num_agents = cfg.train.num_opponents + 1
    num_landmarks = cfg.train.num_landmark
    num_opponents = cfg.train.num_opponents
    obs_dim = cfg.env.obs_dim
    act_dim = cfg.env.act_dim
    state_dim = cfg.env.state_dim

    train_data_steps = int(cfg.independent_dynamics.train_data_steps)
    eval_data_steps = int(cfg.independent_dynamics.eval_data_steps)
    gradient_steps = int(cfg.independent_dynamics.gradient_steps)
    batch_size = int(cfg.independent_dynamics.batch_size)
    eval_interval = int(cfg.independent_dynamics.eval_interval)

    rollout_horizon = int(
        cfg.independent_dynamics.rollout_horizon
    )

    rollout_num_starts = int(
        cfg.independent_dynamics.rollout_num_starts
    )

    include_action_uncertainty = bool(
        cfg.independent_dynamics.include_action_uncertainty
    )

    manual_action_variance = float(
        cfg.independent_dynamics.manual_action_variance
    )

    initial_measurement_variance = float(
        cfg.independent_dynamics.initial_measurement_variance
    )

    print(
        f"\nAgents={num_agents} | "
        f"landmarks={num_landmarks} | "
        f"action_dim={act_dim}"
    )
    print(
        "Local model: "
        "[p_x, p_y, v_x, v_y, own_action] -> "
        "[Δp_x, Δp_y, Δv_x, Δv_y]"
    )

    # ------------------------------------------------------------
    # 1. Build two separate real-data datasets.
    #    Training and evaluation data use different random keys.
    # ------------------------------------------------------------
    env = make_mpe_env(cfg)

    rng, train_collect_key = jax.random.split(rng)
    train_batch, rng = collect_random_transitions(
        env=env,
        key=train_collect_key,
        cfg=cfg,
        num_steps=train_data_steps,
    )

    rng, eval_collect_key = jax.random.split(rng)
    eval_batch, rng = collect_random_transitions(
        env=env,
        key=eval_collect_key,
        cfg=cfg,
        num_steps=eval_data_steps,
    )

    replay_train = ReplayBuffer.create(
        max_size=train_data_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        opp_num=num_opponents,
        state_dim=state_dim,
    )

    replay_eval = ReplayBuffer.create(
        max_size=eval_data_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        opp_num=num_opponents,
        state_dim=state_dim,
    )

    replay_train = add_collected_batch(
        replay_train,
        train_batch,
        cfg,
    )

    replay_eval = add_collected_batch(
        replay_eval,
        eval_batch,
        cfg,
    )

    print(
        f"Collected {len(replay_train)} training transitions and "
        f"{len(replay_eval)} evaluation transitions."
    )

    # ------------------------------------------------------------
    # 2. Fit one normalizer per agent from training data only.
    # ------------------------------------------------------------
    local_standardizers = init_local_standardizers(
        num_agents=num_agents,
        act_dim=act_dim,
    )

    rng, stats_key = jax.random.split(rng)
    stats_batch = replay_train.sample(
        stats_key,
        batch_size=len(replay_train),
        opp_num=num_opponents,
    )

    local_standardizers = update_local_standardizers(
        standardizers=local_standardizers,
        batch=stats_batch,
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        act_dim=act_dim,
    )

    # ------------------------------------------------------------
    # 3. Initialize one ensemble local model per agent.
    # ------------------------------------------------------------
    rng, model_key = jax.random.split(rng)

    _, model_states = init_independent_transition_models(
        rng=model_key,
        num_agents=num_agents,
        act_dim=act_dim,
        cfg=cfg,
    )

    print("Independent local dynamics models initialized.")

    # ------------------------------------------------------------
    # 4. Train.
    # ------------------------------------------------------------
    for step in trange(
        1,
        gradient_steps + 1,
        desc="Training independent dynamics",
    ):
        rng, sample_key = jax.random.split(rng)

        batch = replay_train.sample(
            sample_key,
            batch_size=batch_size,
            opp_num=num_opponents,
        )

        model_states, metrics_per_agent = train_all_independent_models(
            train_states=model_states,
            standardizers=local_standardizers,
            batch=batch,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            act_dim=act_dim,
        )

        if step == 1 or step % eval_interval == 0:
            print(f"\n--- Gradient step {step}/{gradient_steps} ---")

            for agent_id, metrics in enumerate(metrics_per_agent):
                print(
                    f"agent_{agent_id}: "
                    f"NLL={float(metrics['transition_nll']):.5f}, "
                    f"MSE(norm)={float(metrics['transition_mse']):.5f}, "
                    f"logvar={float(metrics['mean_logvar']):.5f}"
                )

            eval_metrics, rng = evaluate_one_step(
                model_states=model_states,
                standardizers=local_standardizers,
                eval_replay=replay_eval,
                key=rng,
                cfg=cfg,
            )

            print(
                f"mean one-step MSE: "
                f"{eval_metrics['mean_one_step_mse']:.7f} | "
                f"zero-delta baseline MSE: "
                f"{eval_metrics['mean_zero_delta_mse']:.7f}"
            )

            for agent_id in range(num_agents):
                print(
                    f"agent_{agent_id} | "
                    f"model pos MSE: "
                    f"{eval_metrics[f'agent_{agent_id}/position_mse']:.7f} | "
                    f"model vel MSE: "
                    f"{eval_metrics[f'agent_{agent_id}/velocity_mse']:.7f} | "
                    f"zero-delta MSE: "
                    f"{eval_metrics[f'agent_{agent_id}/zero_delta_mse']:.7f} | "
                    f"mean variance(norm): "
                    f"{eval_metrics[f'agent_{agent_id}/mean_total_var_norm']:.7f}"
                )
    print("\n===== Multi-step rollout evaluation: oracle actions =====")

    rollout_horizon = int(
        cfg.independent_dynamics.rollout_horizon
    )

    rollout_num_starts = int(
        cfg.independent_dynamics.rollout_num_starts
    )

    if eval_batch is None:
        raise RuntimeError("eval_batch is None before multi-step evaluation.")

    if eval_batch.get("state") is None:
        raise RuntimeError(
            "eval_batch['state'] is None before multi-step evaluation."
        )

    if eval_batch.get("next_state") is None:
        raise RuntimeError(
            "eval_batch['next_state'] is None before multi-step evaluation."
        )

    print(
        "\n===== Multi-step rollout evaluation with "
        "EKF-style covariance propagation ====="
    )

    print(
        f"Action uncertainty enabled: "
        f"{include_action_uncertainty}"
    )

    if include_action_uncertainty:
        print(
            f"Manual action variance: "
            f"{manual_action_variance:.6e}"
        )

    multistep_metrics = evaluate_multistep_oracle_actions(
        model_states=model_states,
        standardizers=local_standardizers,
        trajectory_batch=eval_batch,
        cfg=cfg,
        horizon=rollout_horizon,
        num_rollouts=rollout_num_starts,
        include_action_uncertainty=include_action_uncertainty,
        manual_action_variance=manual_action_variance,
        initial_measurement_variance=initial_measurement_variance,
    )

    report_horizons = [
        h for h in [1, 5, 10, rollout_horizon]
        if h <= rollout_horizon
    ]

    for agent_id in range(num_agents):
        mse_curve = multistep_metrics[
            f"agent_{agent_id}/multistep_mse"
        ]

        pos_curve = multistep_metrics[
            f"agent_{agent_id}/multistep_position_mse"
        ]

        vel_curve = multistep_metrics[
            f"agent_{agent_id}/multistep_velocity_mse"
        ]

        estimated_mse_curve = multistep_metrics[
            f"agent_{agent_id}/estimated_mse_from_covariance"
        ]

        covariance_trace_curve = multistep_metrics[
            f"agent_{agent_id}/covariance_trace"
        ]

        print(f"\nagent_{agent_id}:")

        for horizon_step in report_horizons:
            index = horizon_step - 1

            print(
                f"  horizon={horizon_step:2d} | "
                f"MSE={float(mse_curve[index]):.6f} | "
                f"position MSE={float(pos_curve[index]):.6f} | "
                f"velocity MSE={float(vel_curve[index]):.6f} | "
                f"estimated MSE from P={float(estimated_mse_curve[index]):.6f} | "
                f"trace(P)={float(covariance_trace_curve[index]):.6f}"
            )
    print("\nIndependent dynamics training finished.")


if __name__ == "__main__":
    main()