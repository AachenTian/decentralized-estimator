import numpy as np

from private_mb_sac.replay.buffer import create_private_replays


def _batch(batch_size=2):
    return dict(
        observations=np.zeros((batch_size, 3, 18), np.float32),
        actions=np.zeros((batch_size, 3, 2), np.float32),
        rewards=np.zeros((batch_size, 3), np.float32),
        next_observations=np.ones((batch_size, 3, 18), np.float32),
        dones=np.zeros((batch_size, 3), np.float32),
        model_states=np.zeros((batch_size, 18), np.float32),
        next_model_states=np.ones((batch_size, 18), np.float32),
        episode_dones=np.zeros((batch_size,), np.bool_),
        pair_collision_rates=np.zeros((batch_size,), np.float32),
        min_pair_distances=np.ones((batch_size,), np.float32),
    )


def test_private_replays_are_distinct_and_isolated():
    replays = create_private_replays(
        num_owners=3,
        capacity=32,
        num_agents=3,
        observation_dim=18,
        action_dim=2,
        model_state_dim=18,
    )
    assert len({id(replay) for replay in replays}) == 3
    replays[0].add_batch(**_batch())
    assert tuple(map(len, replays)) == (2, 0, 0)
