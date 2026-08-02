import numpy as np

from private_mb_sac.replay.buffer import PrivateJointReplayBuffer


def make_buffer():
    return PrivateJointReplayBuffer(
        capacity=16,
        num_agents=3,
        observation_dim=18,
        action_dim=2,
        model_state_dim=18,
    )


def test_replay_state_dict_roundtrip():
    first = make_buffer()
    count = 7
    first.add_batch(
        observations=np.ones((count, 3, 18), np.float32),
        actions=np.ones((count, 3, 2), np.float32) * 2,
        rewards=np.ones((count, 3), np.float32) * 3,
        next_observations=np.ones((count, 3, 18), np.float32) * 4,
        dones=np.zeros((count, 3), np.float32),
        model_states=np.ones((count, 18), np.float32) * 5,
        next_model_states=np.ones((count, 18), np.float32) * 6,
        episode_dones=np.asarray(
            [False] * 6 + [True],
            dtype=np.bool_,
        ),
        pair_collision_rates=np.zeros((count,), np.float32),
        min_pair_distances=np.ones((count,), np.float32),
    )

    second = make_buffer()
    second.load_state_dict(first.state_dict())

    assert len(second) == count
    assert second.stats.terminal_count == 1
    assert np.allclose(
        second.next_model_states[:count],
        first.next_model_states[:count],
    )
