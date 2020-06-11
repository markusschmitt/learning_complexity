import physics
import numpy as np

def compute_inverse_KL(samples, logProbabilities, L,T,F,bc="obc"):
    data = ( physics.energies(samples,L,bc)/T-F + logProbabilities ) / 0.6931471806
    return np.mean(data), np.std(data) / np.sqrt(samples.shape[0])
