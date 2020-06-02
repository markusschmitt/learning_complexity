import numpy as np

def energies(s,L):
    #return -1. * (np.sum(s[:,:-1,:] * s[:,1:,:], axis=(1,2)) + np.sum(s[:,:,:-1] * s[:,:,1:], axis=(1,2)) \
    #              + np.sum(s[:,0,:]*s[:,-1,:], axis=(1)) + np.sum(s[:,:,0]*s[:,:,-1], axis=(1)))
    return -1. * (np.sum(s[:,:-1,:] * s[:,1:,:], axis=(1,2)) + np.sum(s[:,:,:-1] * s[:,:,1:], axis=(1,2)))

def compute_entropy(L,T):
    states=np.zeros((2**(L*L),L*L))

    for k in range(2**(L*L)):
        for j in range(L*L):
            if (k >> j) & 1:
                states[k,j] = 1

    states[states==0]=-1

    states=np.reshape(states,(2**(L*L),L,L))

    E=energies(states,L)
    E=E+np.min(E)
    nrm=np.sum(np.exp(-E/T))
    P=np.exp(-E/T-np.log(nrm))
    P=P[np.where(P>1e-8)]
    return -np.sum(P * np.log(P)) / np.log(2.)

def compute_free_energy(L,T):
    states=np.zeros((2**(L*L),L*L))

    for k in range(2**(L*L)):
        for j in range(L*L):
            if (k >> j) & 1:
                states[k,j] = 1

    states[states==0]=-1

    states=np.reshape(np.array(states),(2**(L*L),L,L))

    E=energies(states,L)
    minE=np.min(E)
    E=E+minE
    nrm=np.sum(np.exp(-E/T))
    return np.log(nrm) + minE/T

def compute_energy(L,T):
    states=np.zeros((2**(L*L),L*L))

    for k in range(2**(L*L)):
        for j in range(L*L):
            if (k >> j) & 1:
                states[k,j] = 1

    states[states==0]=-1

    states=np.reshape(np.array(states),(2**(L*L),L,L))

    E=energies(states,L)
    minE=np.min(E)
    nrm=np.sum(np.exp(-(E+minE)/T))
    return np.sum(E*np.exp(-(E+minE)/T))/nrm
