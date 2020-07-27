import samplers
import numpy as np
import json
import sys

def generate_samples(numSamples,T,L,trainSeed=1234,testSeed=3412,outDir=None,bc="obc",numSweeps=1,samplerType="wolff"):

    # Choose sampler
    if samplerType == "wolff":
        if bc == "obc":
            sampler=samplers.wolff_sample_obc
        else:
            sampler=samplers.wolff_sample_pbc
        numSweeps = -1 # Automatic choice of number of updates
    if samplerType == "mcmc":
        if bc == "obc":
            sampler=samplers.mcmc_sample_obc
        else:
            sampler=samplers.mcmc_sample_pbc

    # Generate data
    trainSample,trainEnergies=sampler(numSamples, L=L, T=T,seed=trainSeed, numSweeps=numSweeps)
    testSample,testEnergies=sampler(numSamples, L=L, T=T,seed=testSeed, numSweeps=numSweeps)

    # Save data
    if outDir is not None:
        with open(outDir+"/training_data.npz", 'wb') as outFile:
            print("Saving training data to {}".format(outDir+"/training_data.npz"))
            np.savez(outFile,trainSample=trainSample,trainEnergies=trainEnergies,testSample=testSample,testEnergies=testEnergies)

    return (trainSample, trainEnergies, testSample, testEnergies)


if __name__ == '__main__':
    # Read input
    inputFile=sys.argv[1]
    with open(inputFile) as jsonFile:
        inParameters=json.load(jsonFile)
        
        numSamples=inParameters['number_of_samples']
        T=inParameters['T']
        L=inParameters['L']
        trainSeed=inParameters['seed_training']
        testSeed=inParameters['seed_test']

        outDir=inParameters['output_folder']

    generate_samples(numSamples,T,L,trainSeed=trainSeed,testSeed=testSeed,outDir=outDir)

