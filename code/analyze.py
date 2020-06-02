import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from rnn import RNN2D
import flax
from flax import nn

import time

import json
import sys
import os
import glob

import physics
import utilities

@jax.jit
def train_step(optimizer, batch):
  def loss_fn(model):
    return -jnp.mean(model(batch))
  grad = jax.jit(jax.grad(loss_fn))(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer

@jax.jit
def eval(model,data,S):
    return -jnp.mean(model(data))/0.6931471806 - S

@jax.jit
def eval_log_prob(model,data,energies,F,T):
    #logPM=model(data)
    #logPE=-energies/T-F
    #return jnp.linalg.norm(logPM-logPE)/data.shape[0]
    logPM=model(data)+energies/T
    #logPE=-energies/T-F
    return jnp.std(logPM)

inputFile = sys.argv[1] 
with open(inputFile) as jsonFile:
    inParameters=json.load(jsonFile)

    # Physics parameters
    L = inParameters['Physics']['L']
    T = inParameters['Physics']['T']

    # Model parameters
    rnnUnits=inParameters['Model']['RNN_size']

    # Optimizer parameters
    learningRate=inParameters['Optimizer']['learning_rate']
    beta1=inParameters['Optimizer']['beta1']
    beta2=inParameters['Optimizer']['beta2']

    # Training parameters
    batchSize=inParameters['Training']['batch_size']
    numEpochs=inParameters['Training']['num_epochs']

    # Training data
    numSamples=inParameters['Training data']['number_of_samples']
    trainDataFolder=inParameters['Training data']['input_folder']
    
    # Training data
    outDir=inParameters['Output']['output_folder']

# Model setup
rnn = RNN2D.partial(L=L,units=rnnUnits)
_,params = rnn.init_by_shape(random.PRNGKey(0),[(1,L,L)])
rnnModel = nn.Model(rnn,params)

# Optimizer setup
optimizer = flax.optim.Adam(learning_rate=learningRate, beta1=beta1, beta2=beta2).create(rnnModel)

# Get list of saved network files
netFiles=glob.glob(outDir+"/net_checkpoints/net*")

# Get evolution of error during training
trainingEvol=np.loadtxt(outDir+"/loss_evolution.txt")

# Get step number with minimal generalization error
minGenErrorIdx=int(np.argmin(trainingEvol[:,2]))
print("Minimal generalization error of {0:.6f} found at epoch {1}.".format(trainingEvol[minGenErrorIdx,2],int(trainingEvol[minGenErrorIdx,0])))

fn=outDir+"/net_checkpoints/net_"+str(int(trainingEvol[minGenErrorIdx,0]))+".msgpack"
with open(fn, 'rb') as f:
    netBytes=f.read()
    optimizer=flax.serialization.from_bytes(optimizer,netBytes)

# Load data
trainData = np.loadtxt(trainDataFolder+"training_configs.txt")[:numSamples]
trainData[trainData==-1]=0
trainData = np.reshape(trainData,(int(numSamples/batchSize),batchSize,L,L))
testData = np.loadtxt(trainDataFolder+"test_configs.txt")
testData[testData==-1]=0
testData = np.reshape(testData,(testData.shape[0],L,L))

trainEnergies = np.loadtxt(trainDataFolder+"training_energies.txt")[:numSamples]
testEnergies = np.loadtxt(trainDataFolder+"test_energies.txt")

if L<5:
    S = physics.compute_entropy(L,T)
    F = physics.compute_free_energy(L,T)
    E = physics.compute_energy(L,T)
else:
    S=np.loadtxt(trainDataFolder+"temp_ent.txt")[1] * L*L
    F=np.loadtxt(trainDataFolder+"temp_ent.txt")[2] * L*L
    E=np.sum(testEnergies)/testEnergies.shape[0]
print("Entropy = ", S)
print("Free energy = ", F)
print("Energy = ", E)

trainErr=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
testErr=eval(optimizer.target,testData,S)
trainErrEn=eval_log_prob(optimizer.target,np.reshape(trainData,(numSamples,L,L)),trainEnergies,F,T)
testErrEn=eval_log_prob(optimizer.target,testData,testEnergies,F,T)
print("Training loss is ", trainErr)
print("Generalization loss is ", testErr)

print("Std(log(P) - E) [train] = ",trainErrEn)
print("Std(log(P) - E) [test] = ",testErrEn)

sample_fun=jax.jit(optimizer.target.sample)
rngKey=jax.random.PRNGKey(123)
KL=0.
energy=0.
sampleNum=0
s=jnp.zeros((1000,L,L),dtype=np.int8)
for k in range(1000):
    key,rngKey=jax.random.split(rngKey)
    s,prob=sample_fun(s,key)
    s=s.at[jnp.where(s==0)].set(-1)
    KL = KL + utilities.compute_inverse_KL(s,prob,L,T,F)
    energy = energy + jnp.sum(physics.energies(s,L))
    sampleNum = sampleNum + len(s)

print("Inverse KL = ", KL/sampleNum)
print("Sampled Energy = ", energy/sampleNum)
