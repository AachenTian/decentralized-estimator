from types import SimpleNamespace

import jax.numpy as jnp

from private_mb_sac.rollout.real_collector import resolve_episode_done


def test_time_limit_is_terminal_even_when_native_done_is_false():
    adapter = SimpleNamespace(
        config=SimpleNamespace(max_steps=25)
    )
    output = SimpleNamespace(
        episode_done=jnp.asarray([False, False]),
        env_state=SimpleNamespace(
            step=jnp.asarray([25, 24])
        ),
    )

    episode_done, native_done, timeout_done = resolve_episode_done(
        adapter,
        output,
    )

    assert native_done.tolist() == [False, False]
    assert timeout_done.tolist() == [True, False]
    assert episode_done.tolist() == [True, False]


def test_native_done_is_preserved():
    adapter = SimpleNamespace(
        config=SimpleNamespace(max_steps=25)
    )
    output = SimpleNamespace(
        episode_done=jnp.asarray([True, False]),
        env_state=SimpleNamespace(
            step=jnp.asarray([10, 10])
        ),
    )

    episode_done, _, timeout_done = resolve_episode_done(adapter, output)

    assert timeout_done.tolist() == [False, False]
    assert episode_done.tolist() == [True, False]
