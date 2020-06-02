import physics
from jax import numpy as jnp

def compute_inverse_KL(samples, logProbabilities, L,T,F):
    return -jnp.sum(-physics.energies(samples,L)/T-F) + jnp.sum(logProbabilities)
