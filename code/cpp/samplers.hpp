#pragma once

#include <cstring>
#include <random>
#include <string>
#include <vector>

void wolff_obc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int numSweeps, int seed);
void wolff_pbc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int numSweeps, int seed);

void mcmc_obc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int numSweeps, int seed);
void mcmc_pbc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int numSweeps, int seed);
