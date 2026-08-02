import numpy as np

from private_mb_sac.replay.buffer import PrivateJointReplayBuffer
from private_mb_sac.training.sac_training import (
    sample_private_sac_batch,
)


def make_replay(marker: float):
    replay = PrivateJointReplayBuffer(
        capacity=32,
        num_agents=3,
        observation_dim=18,
        action_dim=2,
        model_state_dim=18,
    )
    count = 8
    replay.add_batch(
        observations=np.full((count, 3, 18), marker, np.float32),
        actions=np.full((count, 3, 2), marker, np.float32),
        rewards=np.full((count, 3), marker, np.float32),
        next_observations=np.full(
            (count, 3, 18),
            marker + 0.1,
            np.float32,
        ),
        dones=np.zeros((count, 3), np.float32),
        model_states=np.full((count, 18), marker, np.float32),
        next_model_states=np.full(
            (count, 18),
            marker + 0.1,
            np.float32,
        ),
        episode_dones=np.zeros((count,), np.bool_),
        pair_collision_rates=np.zeros((count,), np.float32),
        min_pair_distances=np.ones((count,), np.float32),
    )
    return replay


def test_private_real_model_mix_contains_only_supplied_owner_data():
    real = make_replay(1.0)
    model = make_replay(2.0)

    batch, metrics = sample_private_sac_batch(
        real_replay=real,
        model_replay=model,
        owner_id=2,
        batch_size=20,
        real_fraction=0.5,
        rng=np.random.default_rng(0),
    )

    values = set(np.unique(np.asarray(batch.model_states)).tolist())
    assert values == {1.0, 2.0}
    assert metrics["sac/real_batch_fraction"] == 0.5
    assert metrics["sac/model_batch_fraction"] == 0.5
    assert batch.rewards.shape == (20,)
