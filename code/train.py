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

import physics
import utilities

import wolff_sampler

def create_dir(dn):
    if not os.path.isdir(dn):
        try:
            os.mkdir(dn)
        except OSError:
            print ("Creation of the directory {} failed".format(dn))

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

# Create subdirectory for network checkpoints
create_dir(outDir+"/net_checkpoints/")

# Model setup
rnn = RNN2D.partial(L=L,units=rnnUnits)
_,params = rnn.init_by_shape(random.PRNGKey(0),[(1,L,L)])
rnnModel = nn.Model(rnn,params)
#states=get_states()
#print("sum=",jnp.sum(jnp.exp(rnnModel(states))))
#exit()

# Optimizer setup
optimizer = flax.optim.Adam(learning_rate=learningRate, beta1=beta1, beta2=beta2).create(rnnModel)

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

print("*** Starting training.")
trainErr=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
testErr=eval(optimizer.target,testData,S)
#trainErrEn=eval_log_prob(optimizer.target,np.reshape(trainData,(numSamples,L,L)),trainEnergies,F,T)
#testErrEn=eval_log_prob(optimizer.target,testData,testEnergies,F,T)
print("  -> current loss is ", trainErr, testErr)
with open(outDir+"loss_evolution.txt", 'w') as outFile:
    outFile.write("# Training step   train error   test error\n")
    outFile.write('{0} {1:.6f} {2:.6f}\n'.format(0,trainErr,testErr))
for ep in range(numEpochs+1):
    t0 = time.perf_counter()
    if (ep+1) % 100 == 0:
        print("Epoch ", ep+1)
    loss = 0.
    for batch in trainData[np.random.permutation(len(trainData))]:
        optimizer = train_step(optimizer, batch)
        t1 = time.perf_counter()
    if (ep+1) % 100 == 0:
        print("  -> epoch took {0:.6f} seconds.".format(t1-t0))
        trainErr=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
        testErr=eval(optimizer.target,testData,S)
        #trainErrEn=eval_log_prob(optimizer.target,np.reshape(trainData,(numSamples,L,L)),trainEnergies,F,T)
        #testErrEn=eval_log_prob(optimizer.target,testData,testEnergies,F,T)
        print("  -> current loss is ", trainErr, testErr)
        with open(outDir+"loss_evolution.txt", 'a') as outFile:
            outFile.write("{0} {1:.6f} {2:.6f}\n".format(ep+1,trainErr,testErr))
        with open(outDir+"/net_checkpoints/"+"net_"+str(ep+1)+".msgpack", 'wb') as outFile:
            outFile.write(flax.serialization.to_bytes(optimizer))



sample_fun=jax.jit(optimizer.target.sample)
rngKey=jax.random.PRNGKey(123)
KL=0.
energy=0.
sampleNum=0
s=jnp.zeros((1000,L,L),dtype=np.int8)
#s=np.zeros((1000,L,L),dtype=np.int8)
for k in range(100):
    key,rngKey=jax.random.split(rngKey)
    s,prob=sample_fun(s,key)
    s=s.at[jnp.where(s==0)].set(-1)
    KL = KL + utilities.compute_inverse_KL(s,prob,L,T,F)
    energy = energy + jnp.sum(physics.energies(s,L))
    sampleNum = sampleNum + len(s)

print(KL/sampleNum)
print(energy/sampleNum)
