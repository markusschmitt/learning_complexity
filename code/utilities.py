import physics
import numpy as np

def compute_inverse_KL(logProbabilitiesBoltzmann, logProbabilities):
    data = ( -logProbabilitiesBoltzmann + logProbabilities ) / 0.6931471806
    return np.mean(data), np.std(data) / np.sqrt(logProbabilities.shape[0])

