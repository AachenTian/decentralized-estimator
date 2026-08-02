import numpy as np

from private_mb_sac.dynamics.normalization import (
    fit_actor_observation_normalizer,
    fit_private_dynamics_normalizer,
)
from private_mb_sac.replay.buffer import PrivateJointReplayBuffer


def _make_replay():
    replay = PrivateJointReplayBuffer(
        capacity=32,
        num_agents=3,
        observation_dim=18,
        action_dim=2,
        model_state_dim=18,
    )
    batch_size = 8
    state = np.arange(batch_size * 18, dtype=np.float32).reshape(batch_size, 18) / 10
    next_state = state.copy()
    next_state[:, :12] += 0.25
    replay.add_batch(
        observations=np.zeros((batch_size, 3, 18), np.float32),
        actions=np.linspace(-1, 1, batch_size * 6, dtype=np.float32).reshape(batch_size, 3, 2),
        rewards=np.zeros((batch_size, 3), np.float32),
        next_observations=np.zeros((batch_size, 3, 18), np.float32),
        dones=np.zeros((batch_size, 3), np.float32),
        model_states=state,
        next_model_states=next_state,
        episode_dones=np.asarray([False] * 7 + [True]),
        pair_collision_rates=np.zeros((batch_size,), np.float32),
        min_pair_distances=np.ones((batch_size,), np.float32),
    )
    return replay


def test_actor_normalizer_has_minimum_std():
    observations = np.ones((20, 18), np.float32)
    normalizer = fit_actor_observation_normalizer(
        observations,
        minimum_std=1.0e-3,
    )
    assert np.asarray(normalizer.mean).shape == (18,)
    assert np.all(np.asarray(normalizer.std) >= 1.0e-3)


def test_private_dynamics_normalizer_uses_nonterminal_data_only():
    replay = _make_replay()
    normalizer = fit_private_dynamics_normalizer(replay)
    assert int(np.asarray(normalizer.sample_count)) == 7
    assert np.asarray(normalizer.state_mean).shape == (18,)
    assert np.asarray(normalizer.action_mean).shape == (6,)
    assert np.asarray(normalizer.delta_mean).shape == (12,)
    assert np.all(np.asarray(normalizer.delta_std) >= 1.0e-3)
