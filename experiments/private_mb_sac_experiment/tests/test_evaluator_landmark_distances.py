import jax.numpy as jnp
import numpy as np

from private_mb_sac.evaluation.evaluator import (
    _final_landmark_distances,
)


def test_final_landmark_distances_shape_and_values():
    agents = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    ).reshape(-1)
    landmarks = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=jnp.float32,
    ).reshape(-1)
    state = jnp.concatenate([agents, landmarks])[None, :]

    distances = _final_landmark_distances(
        state,
        num_agents=3,
    )

    assert distances.shape == (1, 3)
    assert np.allclose(np.asarray(distances), 0.0)
