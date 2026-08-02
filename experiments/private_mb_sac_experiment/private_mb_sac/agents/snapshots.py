"""Immutable actor snapshot exchange without averaging or live overwrite."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from private_mb_sac.core.types import ActorSnapshotBank


def _copy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.copy(jnp.asarray(leaf)), tree)


def exchange_actor_snapshots(
    live_params_by_agent: Sequence[Any],
    *,
    synchronization_round: int,
) -> ActorSnapshotBank:
    """Copy all live actors into one round-frozen snapshot bank.

    This function never mutates or replaces the live parameter trees and never
    averages parameters.
    """
    if synchronization_round < 0:
        raise ValueError("synchronization_round must be non-negative.")
    if not live_params_by_agent:
        raise ValueError("At least one live actor is required.")
    snapshots = tuple(_copy_tree(params) for params in live_params_by_agent)
    return ActorSnapshotBank(
        params_by_agent=snapshots,
        synchronization_round=int(synchronization_round),
    )


def trees_allclose(first: Any, second: Any, *, atol: float = 0.0) -> bool:
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    if len(first_leaves) != len(second_leaves):
        return False
    return all(
        np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=0.0)
        for a, b in zip(first_leaves, second_leaves)
    )
