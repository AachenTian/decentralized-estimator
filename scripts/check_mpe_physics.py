# scripts/check_mpe_physics.py
import jax.numpy as jnp
from jaxmarl import make

env = make(
    "MPE_simple_spread_v3",
    action_type="Continuous",
    u_noise=jnp.array([5.0, 5.0, 5.0]),
)

print("num_agents:", env.num_agents)
print("num_landmarks:", env.num_landmarks)
print("collide:", env.collide)
print("moveable:", env.moveable)
print("rad:", env.rad)
print("contact_force:", env.contact_force)
print("contact_margin:", env.contact_margin)