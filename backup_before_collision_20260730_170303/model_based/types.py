from __future__ import annotations

from typing import Any, NamedTuple

import jax
from flax import struct


Array = jax.Array


@struct.dataclass
class DynamicsRolloutBatch:
    """Real local dynamics transitions collected from a joint environment.

    Shapes:
        local_states:       (T, E, N, D)
        actions:            (T, E, N, A)
        next_local_states:  (T, E, N, D)
        episode_dones:      (T, E)
    """

    local_states: Array
    actions: Array
    next_local_states: Array
    episode_dones: Array


@struct.dataclass
class DynamicsRolloutOutput:
    env_state: Any
    features: Any
    batch: DynamicsRolloutBatch


@struct.dataclass
class LocalDynamicsBatch:
    """Pooled per-agent transition batch.

    All homogeneous agents contribute samples to the same shared model.

    Shapes:
        local_states:       (B, D)
        actions:            (B, A)
        next_local_states:  (B, D)
        agent_ids:          (B,)
    """

    local_states: Array
    actions: Array
    next_local_states: Array
    agent_ids: Array


class EnsemblePrediction(NamedTuple):
    """Shared ensemble prediction in original physical coordinates.

    Ensemble arrays are batch-first for convenient downstream use.
    """

    ensemble_next_means: Array       # (B, K, D)
    ensemble_next_variances: Array   # (B, K, D)
    next_mean: Array                 # (B, D)
    aleatoric_variance: Array        # (B, D)
    epistemic_variance: Array        # (B, D)
    total_variance: Array            # (B, D)
    ensemble_delta_means_norm: Array # (B, K, D)
    ensemble_delta_logvars_norm: Array # (B, K, D)
