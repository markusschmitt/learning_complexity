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
import partition_sum
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
    return -jnp.mean(model(data))/0.6931471806 - S, jnp.std(model(data))/0.6931471806/jnp.sqrt(data.shape[0])

def sample_from_model(model, sampleNum, key=jax.random.PRNGKey(123)):
    return model.sample(sampleNum, key)

inputFile = sys.argv[1] 
with open(inputFile) as jsonFile:
    inParameters=json.load(jsonFile)

    # Physics parameters
    L = inParameters['Physics']['L']
    T = inParameters['Physics']['T']
    bc = inParameters['Physics']['boundary_condition']

    # Model parameters
    rnnUnits=inParameters['Model']['RNN_size']
    netInitSeed=inParameters['Model']['init_seed']
    netInitScale=inParameters['Model']['init_scale']

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
rnnNet = RNN2D.partial(L=L,units=rnnUnits,initScale=netInitScale)
_,params = rnnNet.init_by_shape(random.PRNGKey(netInitSeed),[(1,L,L)])
rnnModel = nn.Model(rnnNet,params)

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
                        outDir=None, bc=bc)
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
F,E,S = partition_sum.get_thermodynamics(L,T)
Etest = ( np.mean(testEnergies), np.std(testEnergies)/np.sqrt(testEnergies.shape[0]) )

print("*** Physical properties")
print(" > Temperature = ", T)
print(" > Entropy density = ", S)
print(" > Free energy density = ", F)
print(" > Energy density (exact/Onsager) = ", E)
print(" > Energy density (test data) = {} +/- {}".format(Etest[0]/L**2, Etest[1]/L**2))

# Work with extensive quantities below
F*=L**2
E*=L**2
S*=L**2

# Training
print("*** Starting training.")
trainErr, _ = eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
testErr = eval(optimizer.target,testData,S)

# Compute figures of merit from RNN samples
tmpT=Timer(" -> Time to sample RNN:")
sampleNum=250000
s,prob = jax.jit(sample_from_model,static_argnums=(1))(optimizer.target, sampleNum)
s=s.at[jnp.where(s==0)].set(-1)
Es = physics.energies(s,L,bc=bc)
invKL = utilities.compute_inverse_KL(-Es/T+F,prob)
energy = ( jnp.mean(Es), jnp.std(Es) / jnp.sqrt(sampleNum) ) 

print("  -> current loss is ", trainErr/L**2, testErr[0]/L**2,invKL[0]/L**2,np.abs(energy[0]-Etest[0])/L**2)
with open(outDir+"loss_evolution.txt", 'w') as outFile:
    outFile.write("# Training step   train error   test error   inv. KL   energy density diff.  |  (MC errors:)  test error  inv. KL   energy density diff.\n")
    outFile.write("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} | {5:.6f} {6:.6f} {7:.6f}\n".format(1e-1,trainErr/L**2,
                                                                      testErr[0]/L**2,
                                                                      invKL[0]/L**2,
                                                                      np.abs(energy[0]-Etest[0])/L**2,
                                                                      testErr[1]/L**2, invKL[1]/L**2, (energy[1]+Etest[1])/L**2))

# Timer for epoch compute time
epochTimer = Timer(" -> Time for 100 epochs:")
nextOutputStep=1.
for ep in range(numEpochs+1):
    # Perform training steps
    for batch in trainData[np.random.permutation(len(trainData))]:
        optimizer = train_step(optimizer, batch)

    # Write output
    if (ep+1) >= int(nextOutputStep) or ep==numEpochs:
        print("Epoch ", ep+1)
        epochTimer.stop(" -> Time for {} epochs:".format((1+nextOutputStep)//2))
        #nextOutputStep*=2
        nextOutputStep = nextOutputStep * np.sqrt(2.)
        trainErr,_=eval(optimizer.target,np.reshape(trainData,(numSamples,L,L)),S)
        testErr=eval(optimizer.target,testData,S)

        # Compute figures of merit from RNN samples
        tmpT=Timer(" -> Time to sample RNN:")
        s,prob = jax.jit(sample_from_model, static_argnums=1)(optimizer.target, sampleNum)
        s=s.at[jnp.where(s==0)].set(-1)
        Es = physics.energies(s,L,bc=bc)
        invKL = utilities.compute_inverse_KL(-Es/T+F,prob)
        energy = ( jnp.mean(Es), jnp.std(Es) / jnp.sqrt(sampleNum) ) 

        tmpT.stop()
        print("  -> current loss is ", trainErr/L**2, testErr[0]/L**2, invKL[0]/L**2, jnp.abs(energy[0]-Etest[0])/L**2)
        print("  -> current energy density is {0:.6f} +/- {1:.6f}".format(energy[0]/L**2,energy[1]/L**2))
        with open(outDir+"loss_evolution.txt", 'a') as outFile:
            outFile.write("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} | {5:.6f} {6:.6f} {7:.6f}\n".format(ep+1,trainErr/L**2,
                                                                              testErr[0]/L**2,
                                                                              invKL[0]/L**2,
                                                                              jnp.abs(energy[0]-Etest[0])/L**2,
                                                                              testErr[1]/L**2, invKL[1]/L**2, (energy[1]+Etest[1])/L**2))
        with open(outDir+"/net_checkpoints/"+"net_"+str(ep+1)+".msgpack", 'wb') as outFile:
            outFile.write(flax.serialization.to_bytes(optimizer))
        epochTimer.reset()

