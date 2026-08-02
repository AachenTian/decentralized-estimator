import jax
import numpy as np

from private_mb_sac.core.rng import (
    FORCED_ENVIRONMENT_SEED,
    make_environment_key_schedule,
    make_initial_reset_keys,
)


def test_all_reset_keys_are_exactly_seed40():
    expected = np.asarray(
        jax.random.PRNGKey(FORCED_ENVIRONMENT_SEED)
    )
    keys = np.asarray(
        make_initial_reset_keys(999, num_envs=12)
    )

    assert keys.shape == (12, 2)
    assert np.all(keys == expected[None, :])


def test_environment_schedule_ignores_owner_round_and_input_seed():
    first = make_environment_key_schedule(
        1,
        synchronization_round=1,
        rollout_length=25,
        num_envs=12,
    )
    second = make_environment_key_schedule(
        999,
        synchronization_round=200,
        rollout_length=25,
        num_envs=12,
    )

    assert np.array_equal(
        np.asarray(first.step_keys),
        np.asarray(second.step_keys),
    )
    assert np.array_equal(
        np.asarray(first.reset_keys),
        np.asarray(second.reset_keys),
    )
    assert np.all(
        np.asarray(first.reset_keys)
        == np.asarray(jax.random.PRNGKey(40))[None, None, :]
    )
