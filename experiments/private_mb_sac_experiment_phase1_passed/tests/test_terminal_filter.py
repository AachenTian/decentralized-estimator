import numpy as np

from private_mb_sac.replay.buffer import PrivateJointReplayBuffer


def test_terminal_transitions_are_excluded_from_dynamics_candidates():
    replay = PrivateJointReplayBuffer(
        capacity=16,
        num_agents=3,
        observation_dim=18,
        action_dim=2,
        model_state_dim=18,
    )
    batch_size = 4
    replay.add_batch(
        observations=np.zeros((batch_size, 3, 18), np.float32),
        actions=np.zeros((batch_size, 3, 2), np.float32),
        rewards=np.zeros((batch_size, 3), np.float32),
        next_observations=np.zeros((batch_size, 3, 18), np.float32),
        dones=np.zeros((batch_size, 3), np.float32),
        model_states=np.zeros((batch_size, 18), np.float32),
        next_model_states=np.zeros((batch_size, 18), np.float32),
        episode_dones=np.asarray([False, True, False, True]),
        pair_collision_rates=np.zeros((batch_size,), np.float32),
        min_pair_distances=np.ones((batch_size,), np.float32),
    )
    assert replay.valid_dynamics_indices.tolist() == [0, 2]
    sampled = replay.sample(
        2,
        np.random.default_rng(0),
        nonterminal_only=True,
    )
    assert sampled.model_states.shape == (2, 18)
