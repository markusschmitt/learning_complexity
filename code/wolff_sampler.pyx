cimport numpy as np
import numpy as np
cimport cython
import ctypes

cdef extern from "cpp/wolff.hpp":
    void wolff_obc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int seed)
    void wolff_pbc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int seed)


def sample_obc(int numSamples, int L, double T, int seed=0):
    cdef np.ndarray[int, ndim=3, mode="c"] samples = np.empty((numSamples,L,L), dtype=ctypes.c_int) 
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] energies = np.zeros((numSamples), dtype=np.double)

    wolff_obc_generate_samples(&samples[0,0,0], &energies[0], numSamples, L, T, seed)

    return (samples, energies) 

def sample_pbc(int numSamples, int L, double T, int seed=0):
    cdef np.ndarray[int, ndim=3, mode="c"] samples = np.empty((numSamples,L,L), dtype=ctypes.c_int) 
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] energies = np.zeros((numSamples), dtype=np.double)

    wolff_pbc_generate_samples(&samples[0,0,0], &energies[0], numSamples, L, T, seed)

    return (samples, energies) 
