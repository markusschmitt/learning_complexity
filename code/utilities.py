import physics
from jax import numpy as jnp

def compute_inverse_KL(samples, logProbabilities, L,T,F,bc="obc"):
<<<<<<< HEAD
    return -jnp.sum(-physics.energies(samples,L,bc)/T+F) + jnp.sum(logProbabilities)
=======
    return (-jnp.sum(-physics.energies(samples,L,bc)/T-F) + jnp.sum(logProbabilities)) / 0.6931471806
>>>>>>> 717d163d578189a441758877e0845dc5c77092ea
