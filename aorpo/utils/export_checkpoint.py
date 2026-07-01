from __future__ import annotations

import os
import pickle

import jax
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


def _tree_shapes(tree):
    return jax.tree_util.tree_map(
        lambda x: tuple(x.shape) if hasattr(x, "shape") else None,
        tree,
    )


def _to_int(x):
    return int(jax.device_get(x))


def save_execution_checkpoint(
    transition_state,
    reward_state,
    policy_state,
    real_opponent_states,
    std,
    cfg,
    num_agents,
    num_opponents,
    obs_dim,
    state_dim,
    act_dim,
    opp_dim,
    final_epoch,
    save_name="final_execution_ckpt.pkl",
):
    """
    Save execution/prototype checkpoint.

    Saves:
    - dynamics model params: transition_state.params
    - reward model params: reward_state.params
    - ego policy params: policy_state.params
    - real opponent policy params: real_opponent_states[i].params
    - standardizer: std
    - config and dimensions

    Does NOT save:
    - opponent_states
    - q1/q2/target_q states
    - optimizer states
    """

    save_dir = os.path.join(get_original_cwd(), "exported_ckpts")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_name)

    opponent_policy_params = {
        f"agent_{i + 1}": jax.device_get(state.params)
        for i, state in enumerate(real_opponent_states)
    }

    opponent_policy_steps = {
        f"agent_{i + 1}": _to_int(state.step)
        for i, state in enumerate(real_opponent_states)
    }

    ckpt = {
        "transition_params": jax.device_get(transition_state.params),
        "reward_params": jax.device_get(reward_state.params),

        "ego_policy_params": jax.device_get(policy_state.params),
        "opponent_policy_params": opponent_policy_params,

        "std": jax.device_get(std),

        "steps": {
            "final_epoch": int(final_epoch),
            "transition_step": _to_int(transition_state.step),
            "reward_step": _to_int(reward_state.step),
            "ego_policy_step": _to_int(policy_state.step),
            "opponent_policy_steps": opponent_policy_steps,
        },

        "param_shapes": {
            "transition_params": _tree_shapes(transition_state.params),
            "reward_params": _tree_shapes(reward_state.params),
            "ego_policy_params": _tree_shapes(policy_state.params),
            "opponent_policy_params": {
                f"agent_{i + 1}": _tree_shapes(state.params)
                for i, state in enumerate(real_opponent_states)
            },
        },

        "meta": {
            "num_agents": int(num_agents),
            "num_opponents": int(num_opponents),
            "obs_dim": int(obs_dim),
            "state_dim": int(state_dim),
            "act_dim": int(act_dim),
            "opp_dim": int(opp_dim),
            "core_state_dim": int(cfg.train.core_state_dim),

            "saved_ego_policy_variable": "policy_state",
            "saved_opponent_policy_variable": "real_opponent_states",
            "not_saved": [
                "opponent_states",
                "q1_state",
                "q2_state",
                "target_q1_state",
                "target_q2_state",
            ],
        },

        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)

    print(f"✅ Execution checkpoint saved to: {save_path}")

    return save_path