import jax
import jax.numpy as jnp

from private_mb_sac.agents.snapshots import (
    exchange_actor_snapshots,
    trees_allclose,
)


def test_snapshot_exchange_does_not_follow_live_updates():
    live = (
        {"w": jnp.asarray([1.0, 2.0])},
        {"w": jnp.asarray([3.0, 4.0])},
        {"w": jnp.asarray([5.0, 6.0])},
    )
    bank = exchange_actor_snapshots(live, synchronization_round=7)
    changed_live = tuple(
        jax.tree_util.tree_map(lambda value: value + 100.0, tree)
        for tree in live
    )

    assert bank.synchronization_round == 7
    assert all(
        trees_allclose(snapshot, original)
        for snapshot, original in zip(bank.params_by_agent, live)
    )
    assert all(
        not trees_allclose(snapshot, changed)
        for snapshot, changed in zip(bank.params_by_agent, changed_live)
    )
