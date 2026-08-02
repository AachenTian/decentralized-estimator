from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


Array = jax.Array


@struct.dataclass
class NormalizationStats:
    """Frozen affine normalization statistics.

    `mean` and `std` follow the semantic feature shape. All leading batch axes
    are broadcast automatically. Statistics are calibrated once from a fixed
    random-policy dataset and then frozen for the complete training run.
    """

    mean: Array
    std: Array
    clip: float = struct.field(pytree_node=False, default=10.0)
    epsilon: float = struct.field(pytree_node=False, default=1e-6)

    def as_serializable(self) -> dict[str, Any]:
        return {
            "mean": np.asarray(jax.device_get(self.mean)).tolist(),
            "std": np.asarray(jax.device_get(self.std)).tolist(),
            "clip": float(self.clip),
            "epsilon": float(self.epsilon),
        }


def normalize(values: Array, stats: NormalizationStats) -> Array:
    """Apply frozen zero-mean/unit-scale normalization and symmetric clipping."""

    values = jnp.asarray(values, dtype=jnp.float32)
    normalized = (values - stats.mean) / (stats.std + stats.epsilon)
    return jnp.clip(normalized, -stats.clip, stats.clip)


def maybe_normalize(
    values: Array, stats: NormalizationStats | None
) -> Array:
    """Normalize when statistics are supplied, otherwise return float32 input."""

    if stats is None:
        return jnp.asarray(values, dtype=jnp.float32)
    return normalize(values, stats)


def denormalize(values: Array, stats: NormalizationStats) -> Array:
    """Invert the affine part of normalization (clipping is not invertible)."""

    values = jnp.asarray(values, dtype=jnp.float32)
    return values * (stats.std + stats.epsilon) + stats.mean


@dataclass
class RunningMoments:
    """Numerically stable NumPy moments over arbitrary leading batch axes."""

    feature_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.feature_shape or any(dim < 1 for dim in self.feature_shape):
            raise ValueError("feature_shape must contain positive dimensions.")
        self.count = 0
        self.mean = np.zeros(self.feature_shape, dtype=np.float64)
        self.m2 = np.zeros(self.feature_shape, dtype=np.float64)

    def update(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64)
        if tuple(array.shape[-len(self.feature_shape) :]) != self.feature_shape:
            raise ValueError(
                "Values do not match feature shape: "
                f"got {array.shape}, expected tail {self.feature_shape}."
            )
        samples = array.reshape((-1,) + self.feature_shape)
        if samples.shape[0] == 0:
            return

        batch_count = int(samples.shape[0])
        batch_mean = np.mean(samples, axis=0)
        centered = samples - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total_count)
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / total_count)
        )
        self.count = total_count

    def finalize(
        self,
        *,
        clip: float,
        epsilon: float = 1e-6,
        minimum_std: float = 1e-3,
    ) -> NormalizationStats:
        if self.count < 2:
            raise ValueError(
                "At least two calibration samples are required; "
                f"received {self.count}."
            )
        if clip <= 0.0:
            raise ValueError("clip must be positive.")
        if epsilon <= 0.0 or minimum_std <= 0.0:
            raise ValueError("epsilon and minimum_std must be positive.")

        variance = self.m2 / float(self.count)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.maximum(std, minimum_std)
        return NormalizationStats(
            mean=jnp.asarray(self.mean, dtype=jnp.float32),
            std=jnp.asarray(std, dtype=jnp.float32),
            clip=float(clip),
            epsilon=float(epsilon),
        )
