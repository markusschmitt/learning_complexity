import physics
from jax import numpy as jnp

def compute_inverse_KL(samples, logProbabilities, L,T,F,bc="obc"):
    return (-jnp.sum(-physics.energies(samples,L,bc)/T-F) + jnp.sum(logProbabilities)) / 0.6931471806
