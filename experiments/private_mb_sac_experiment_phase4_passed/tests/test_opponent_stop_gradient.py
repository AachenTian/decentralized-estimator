import jax
import jax.numpy as jnp
import numpy as np

from private_mb_sac.training.sac_update import (
    compose_focal_joint_actions,
)


def test_only_focal_action_receives_gradient():
    focal = jnp.asarray([[0.3, -0.4]], dtype=jnp.float32)
    opponents = jnp.ones((1, 3, 2), dtype=jnp.float32)

    def loss(focal_actions, opponent_actions):
        joint = compose_focal_joint_actions(
            focal_actions,
            opponent_actions,
            owner_id=1,
        )
        return jnp.sum(jnp.square(joint))

    focal_grad, opponent_grad = jax.grad(
        loss,
        argnums=(0, 1),
    )(focal, opponents)

    assert np.linalg.norm(np.asarray(focal_grad)) > 0.0
    assert np.allclose(np.asarray(opponent_grad), 0.0)
