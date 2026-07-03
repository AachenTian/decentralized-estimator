from __future__ import annotations

from typing import Any, Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from aorpo.agents.independent_dynamics import (
    init_independent_transition_models,
    init_local_standardizers,
    make_local_transition_batch,
    predict_local_next,
    train_all_independent_models,
    update_local_standardizers,
)
from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import (
    env_step,
    make_mpe_env,
)
from aorpo.utils.replay import ReplayBuffer


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

        all_mse.append(mse)

    metrics["mean_one_step_mse"] = float(jnp.mean(jnp.stack(all_mse)))

    return metrics, key


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
                f"{eval_metrics['mean_one_step_mse']:.7f}"
            )

            for agent_id in range(num_agents):
                print(
                    f"agent_{agent_id} | "
                    f"pos MSE: "
                    f"{eval_metrics[f'agent_{agent_id}/position_mse']:.7f} | "
                    f"vel MSE: "
                    f"{eval_metrics[f'agent_{agent_id}/velocity_mse']:.7f} | "
                    f"mean variance(norm): "
                    f"{eval_metrics[f'agent_{agent_id}/mean_total_var_norm']:.7f}"
                )

    print("\nIndependent dynamics training finished.")


if __name__ == "__main__":
    main()