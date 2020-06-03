import wolff_sampler
import numpy as np
import json
import sys

def generate_samples(numSamples,T,L,trainSeed=1234,testSeed=3412,outDir=None):
    # Read input
    # Generate data
    trainSample,trainEnergies=wolff_sampler.sample_obc(numSamples, L=4, T=T,seed=trainSeed)
    testSample,testEnergies=wolff_sampler.sample_obc(numSamples, L=4, T=T,seed=testSeed)

    if outDir is not None:
        with open(outDir+"/training_data.npz", 'wb') as outFile:
            print("Saving training data to {}".format(outDir+"/training_data.npz"))
            np.savez(outFile,trainSample=trainSample,trainEnergies=trainEnergies,testSample=testSample,testEnergies=testEnergies)

    return (trainSample, trainEnergies, testSample, testEnergies)


if __name__ == '__main__':
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

