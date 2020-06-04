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
from generate_samples import generate_samples 

class Timer:
    def __init__(self,name=None):
        self.t0=time.perf_counter()
        self.name=name

    def stop(self,name=None):
        t1=time.perf_counter()
        if name is not None:
            self.name=name
        if self.name is not None:
            print("{0} {1:.4f} sec".format(self.name,t1-self.t0))
        return t1

    def reset(self):
        self.t0=time.perf_counter()

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
    grad = jax.jit(jax.grad(jax.jit(loss_fn)))(optimizer.target)
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

def eval_sample_f(optimizer, L, T, F, num=1000000):
    rngKey=jax.random.PRNGKey(123)
    s,prob=jax.jit(optimizer.target.sample)(jnp.zeros((num,L,L),dtype=np.int32),rngKey)
    KL = utilities.compute_inverse_KL(s,prob,L,T,F) / num
    energy = jnp.sum(physics.energies(s,L)) / num

    return KL, energy
eval_sample=jax.jit(eval_sample_f, static_argnums=(1,2,3))

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
    numSamples=inParameters['Training data']['number_of_training_samples']
    trainDataFileName=inParameters['Training data']['training_data']
    
    # Training data
    outDir=inParameters['Output']['output_folder']

# Set numpy seed
np.random.seed(0)

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
if inParameters['Training data']['training_data']=="generate":
    print("*** Generating samples")
    numTestSamples=inParameters['Training data']['number_of_test_samples']
    if numTestSamples < numSamples:
        numTestSamples = numSamples
    trainData, trainEnergies, testData, testEnergies =\
        generate_samples(numTestSamples,T,L,
                        inParameters['Training data']['seed_training'],
                        inParameters['Training data']['seed_test'],
                        outDir=outDir)
    trainData = trainData[:numSamples]
    trainEnergies = trainEnergies[:numSamples]
    print("*** done.")
else:
    with open(trainDataFileName, 'rb') as dataFile:
        data = np.load(dataFile)
        trainData = data['trainSample'][:numSamples]
        testData = data['testSample']
        trainEnergies = data['trainEnergies'][:numSamples]
        testEnergies = data['testEnergies']

# RNN works with spin up/down = 1/0
trainData[trainData==-1]=0
testData[testData==-1]=0

# Reshape data into 2D configurations and training data into batches
trainData = np.reshape(trainData,(int(numSamples/batchSize),batchSize,L,L))
testData = np.reshape(testData,(testData.shape[0],L,L))

# Compute physical properties of the ensemble
if L<5:
    S = physics.compute_entropy(L,T)
    F = physics.compute_free_energy(L,T)
    E = physics.compute_energy(L,T)
else:
    S=physics.onsager_entropy(T) * L*L
    F=physics.onsager_free_energy(T) * L*L
    E=physics.onsager_energy(T) * L*L
print("*** Physical properties")
print(" > Entropy = ", S)
print(" > Free energy = ", F)
print(" > Energy (exact/Onsager) = ", E)
Etest=np.sum(testEnergies)/testEnergies.shape[0]
print(" > Energy (test data) = ", Etest)

# Training
print("*** Starting training.")
trainErr=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
testErr=eval(optimizer.target,testData,S)
sampleTime=Timer(" -> Time to sample RNN:")
invKL,E = eval_sample(optimizer, L, T, F)
sampleTime.stop()
print("  -> current loss is ", trainErr, testErr,invKL,np.abs(E-Etest))
with open(outDir+"loss_evolution.txt", 'w') as outFile:
    outFile.write("# Training step   train error   test error   inv. KL   energy diff.\n")
    outFile.write('{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n'.format(0,trainErr,testErr,invKL,np.abs(E-Etest)))
epochTimer = Timer(" -> Time for 100 epochs:")
sample_fun=jax.jit(optimizer.target.sample)
for ep in range(numEpochs+1):
    if (ep+1) % 100 == 0:
        print("Epoch ", ep+1)
    loss = 0.
    for batch in trainData[np.random.permutation(len(trainData))]:
        optimizer = train_step(optimizer, batch)
    if (ep+1) % 100 == 0:
        epochTimer.stop()
        trainErr=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
        testErr=eval(optimizer.target,testData,S)

        tmpT=Timer(" -> Time to sample RNN:")
        s=jnp.zeros((1000000,L,L),dtype=np.int32)
        invKL=0.
        energy=0.
        sampleNum=0
        rngKey=jax.random.PRNGKey(123)
        for k in range(1):
            key,rngKey=jax.random.split(rngKey)
            #s,prob=sample_fun(s,key)
            s,prob=jax.jit(optimizer.target.sample,static_argnums=1)(s,key)
            s=s.at[jnp.where(s==0)].set(-1)
            invKL = invKL + utilities.compute_inverse_KL(s,prob,L,T,F)
            energy = energy + jnp.sum(physics.energies(s,L))
            sampleNum = sampleNum + len(s)

        tmpT.stop()
        #sampleTime=Timer(" -> Time to sample RNN:")
        #invKL,energy = eval_sample(optimizer, L, T, F)
        print("  -> current loss is ", trainErr, testErr, invKL/sampleNum, np.abs(energy/sampleNum-Etest))
        #sampleTime.stop()
        with open(outDir+"loss_evolution.txt", 'a') as outFile:
            outFile.write("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n".format(ep+1,trainErr,testErr,invKL/sampleNum,np.abs(energy/sampleNum-Etest)))
        with open(outDir+"/net_checkpoints/"+"net_"+str(ep+1)+".msgpack", 'wb') as outFile:
            outFile.write(flax.serialization.to_bytes(optimizer))
        epochTimer.reset()
