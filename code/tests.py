import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import flax
from flax import nn

import time

import json
import sys
import os

import rnn
from rnn import RNN2D
import physics
import utilities
import wolff_sampler
from generate_samples import generate_samples 

@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        return -jnp.mean(model(batch))
    grad = jax.jit(jax.grad(jax.jit(loss_fn)))(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer

# Physics parameters
L = 3
T = 3

# Model parameters
rnnUnits=[22]

# Optimizer parameters
learningRate=1e-3
beta1=0.9
beta2=0.99

# Training parameters
batchSize=128
numEpochs=200

# Training data
numSamples=4096

# Set numpy seed
np.random.seed(0)

# Model setup
rnnNet = RNN2D.partial(L=L,units=rnnUnits)
_,params = rnnNet.init_by_shape(random.PRNGKey(0),[(1,L,L)])
rnnModel = nn.Model(rnnNet,params)

# Optimizer setup
optimizer = flax.optim.Adam(learning_rate=learningRate, beta1=beta1, beta2=beta2).create(rnnModel)

# Generate data
print("*** Generating samples")
numTestSamples=500000
if numTestSamples < numSamples:
    numTestSamples = numSamples
trainData, trainEnergies, testData, testEnergies =\
    generate_samples(numTestSamples,T,L,1234,3412) 
trainData = trainData[:numSamples]
trainEnergies = trainEnergies[:numSamples]
print("*** done.")

# RNN works with spin up/down = 1/0
trainData[trainData==-1]=0
testData[testData==-1]=0

# Reshape data into 2D configurations and training data into batches
trainData = np.reshape(trainData,(int(numSamples/batchSize),batchSize,L,L))
testData = np.reshape(testData,(testData.shape[0],L,L))

# Compute physical properties of the ensemble
S = physics.compute_entropy(L,T)
F = physics.compute_free_energy(L,T)
E = physics.compute_energy(L,T)
print("*** Physical properties")
print(" > Entropy = ", S)
print(" > Free energy = ", F)
print(" > Energy (exact/Onsager) = ", E)
Etest=np.sum(testEnergies)/testEnergies.shape[0]
print(" > Energy (test data) = ", Etest)

# Training
#print("*** Starting training.")
#sample_fun=jax.jit(optimizer.target.sample)
#for ep in range(numEpochs+1):
#    for batch in trainData[np.random.permutation(len(trainData))]:
#        optimizer = train_step(optimizer, batch)

sampleNum=2000
s=jnp.zeros((sampleNum,L,L),dtype=np.int32)
key=jax.random.PRNGKey(123)
s,probs=jax.jit(optimizer.target.sample,static_argnums=[0,1])(sampleNum,key)
probs=optimizer.target(s)
s=s.at[jnp.where(s==0)].set(-1)
sampledEntropy = jnp.sum(probs)/sampleNum
sampledEnergy = jnp.sum(physics.energies(s,L))/sampleNum

print("*** Generate full enumeration")
se=rnn.get_states(L)
print("*** done")
probs=optimizer.target(se)
se=np.array(se)
se[np.where(se==0)]=-1
se=jnp.array(se)
exactEnergy = jnp.sum(jnp.exp(probs)*physics.energies(se,L))
exactEntropy = jnp.sum(jnp.exp(probs)*probs)
dNrm = np.abs(1.-jnp.sum(jnp.exp(probs)))
dE=np.abs(sampledEnergy-exactEnergy)
dS=np.abs(sampledEntropy-exactEntropy)

print("+++ Differences:")
print("  Entropy:",dS)
print("  Energy:",dE)
print("  Norm:",dNrm)
