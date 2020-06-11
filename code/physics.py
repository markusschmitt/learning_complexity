import numpy as np
import partition_sum

def energies(s,L,bc="obc"):
    if bc == "pbc":
        return -1. * (np.sum(s[:,:-1,:] * s[:,1:,:], axis=(1,2)) + np.sum(s[:,:,:-1] * s[:,:,1:], axis=(1,2)) \
                      + np.sum(s[:,0,:]*s[:,-1,:], axis=(1)) + np.sum(s[:,:,0]*s[:,:,-1], axis=(1)))
    else:
        return -1. * (np.sum(s[:,:-1,:] * s[:,1:,:], axis=(1,2)) + np.sum(s[:,:,:-1] * s[:,:,1:], axis=(1,2)))

def compute_entropy(L,T,bc="obc"):
    states=np.zeros((2**(L*L),L*L))

    for k in range(2**(L*L)):
        for j in range(L*L):
            if (k >> j) & 1:
                states[k,j] = 1

    states[states==0]=-1

    states=np.reshape(states,(2**(L*L),L,L))

    E=energies(states,L,bc)
    E=E+np.min(E)
    nrm=np.sum(np.exp(-E/T))
    P=np.exp(-E/T-np.log(nrm))
    P=P[np.where(P>1e-8)]
    return -np.sum(P * np.log(P)) / np.log(2.)

def compute_free_energy(L,T,bc="obc"):
    if L < 5:
        states=np.zeros((2**(L*L),L*L))

        for k in range(2**(L*L)):
            for j in range(L*L):
                if (k >> j) & 1:
                    states[k,j] = 1

        states[states==0]=-1

        states=np.reshape(np.array(states),(2**(L*L),L,L))

        E=energies(states,L,bc)
        minE=np.min(E)
        E=E+minE
        nrm=np.sum(np.exp(-E/T))
        return - (np.log(nrm) + minE/T)
    else:
        return partition_sum.free_energy(L,T) * L**2

def compute_energy(L,T,bc="obc"):
    states=np.zeros((2**(L*L),L*L))

    for k in range(2**(L*L)):
        for j in range(L*L):
            if (k >> j) & 1:
                states[k,j] = 1

    states[states==0]=-1

    states=np.reshape(np.array(states),(2**(L*L),L,L))

    E=energies(states,L,bc)
    minE=np.min(E)
    nrm=np.sum(np.exp(-(E+minE)/T))
    return np.sum(E*np.exp(-(E+minE)/T))/nrm



class OnsagerSolution():

    def __init__(self,T,J=1,nInt=1000,N=None):
        self.J = J
        self.T = T
        self.beta = 1 / T
        self.nInt = nInt

        if N is None:
            self.Delta = 0.
            self.K = 2 * self.beta * self.J
            self.k = 2 * np.sinh(self.K) / (np.cosh(self.K)**2)
        else:
            self.Delta=5. / (4.*np.sqrt(N))
            self.K = 2 * self.beta * self.J / (1. + self.Delta)
            self.k = 2 * np.sinh(self.K) / ((1+np.pi**2/N) * np.cosh(self.K)**2)


    def ent(self):
        return -( self.T*self.f() - self.energy() ) / self.T

    def f(self):
        return -0.5*np.log(2) - np.log(np.cosh(self.K)) - self._f_integral() / (2 * np.pi)

    def _f_integral(self):
        phi = [n * np.pi / self.nInt for n in range(self.nInt)]
        Phi = np.log(1+self._Delta(phi))
        return np.sum(Phi) * np.pi / self.nInt

    def energy(self):
        return -2 * self.J * np.tanh(self.K) + self.k * 4 * ((1-2*np.tanh(self.K)**2) / np.cosh(self.K)) * self._e_integral() / (2*np.pi) 

    def _e_integral(self):
        phi = [n * np.pi / self.nInt for n in range(self.nInt)]
        Phi = np.cos(phi)**2 / (self._Delta(phi) * (1+self._Delta(phi)))
        return np.sum(Phi) * np.pi / self.nInt

    def _Delta(self, phi):
        return np.sqrt(1-self.k**2 * np.cos(phi)**2)


def onsager_entropy(T,N=None):
    sol=OnsagerSolution(T,N=N)
    return sol.ent() / np.log(2.)

def onsager_free_energy(T,N=None):
    sol=OnsagerSolution(T,N=N)
    return sol.f()

def onsager_energy(T,N=None):
    sol=OnsagerSolution(T,N=N)
    return sol.energy()
