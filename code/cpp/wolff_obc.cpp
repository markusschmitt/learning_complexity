// Wolff cluster algorithm for the 2-D Ising Model
// https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=11&ved=2ahUKEwjlm53htLbnAhW8CTQIHThkBHUQFjAKegQIBBAB&url=https%3A%2F%2Fwww.uio.no%2Fstudier%2Femner%2Fmatnat%2Ffys%2Fnedlagte-emner%2FFYS4410%2Fv07%2Fundervisningsmateriale%2FMaterial%2520for%2520Part%2520I%2FPrograms%2FPrograms%2520for%2520Project%25201%2Fwolff.cpp&usg=AOvVaw2jgfda-M8jjdfMdrbnaVxs

#include "wolff_obc.hpp"

using namespace std;

double J = 1.;                  // ferromagnetic coupling
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
vector<int> s;                  // spin configuration
double T;                       // temperature

// For random numbers
std::mt19937 rng(0);
std::uniform_real_distribution<double> distr(0.,1.);
double qadran() {return distr(rng);}

void initialize ( ) {
    s = vector<int>(N);
    for (int i = 0; i < Lx; i++)
        for (int j = 0; j < Ly; j++)
            s[Ly*i+j] = qadran() < 0.5 ? +1 : -1;   // hot start
}

vector<bool> cluster;               // boolean variables identify which spins belong to a cluster
double addProbability;              // 1 - e^(-2J/kT)

void initialize_cluster() {

    // allocate array for spin cluster labels
    cluster = vector<bool>(N);

    // compute the probability to add a like spin to the cluster
    addProbability = 1 - exp(-2*J/T);
}

// declare functions to implement Wolff algorithm
void grow_cluster(int i, int j, int clusterSpin);
void try_add(int i, int j, int clusterSpin);

void do_mc_step() {
    // no cluster defined so clear the cluster array
    for (int i = 0; i < Lx; i++)
    for (int j = 0; j < Lx; j++)
        cluster[Ly*i+j] = false;

    // choose a random spin and grow a cluster
    int i = int(qadran() * Lx);
    int j = int(qadran() * Ly);
    grow_cluster(i, j, s[Ly*i+j]);
}

void grow_cluster(int i, int j, int clusterSpin) {

    // mark the spin as belonging to the cluster and flip it
    cluster[Ly*i+j] = true;
    s[Ly*i+j] = -s[Ly*i+j];

    // find the indices of the 4 neighbors
    // assuming open boundary conditions
    int iPrev = i-1;
    int iNext = i+1;
    int jPrev = j-1;
    int jNext = j+1;

    // if the neighbor spin does not belong to the
    // cluster, then try to add it to the cluster
    if (iPrev >= 0) {
    if (!cluster[Ly*iPrev+j])
        try_add(iPrev, j, clusterSpin);}
    if (iNext < Lx) {
    if (!cluster[Ly*iNext+j])
        try_add(iNext, j, clusterSpin);}
    if (jPrev >= 0) {
    if (!cluster[Ly*i+jPrev])
        try_add(i, jPrev, clusterSpin);}
    if (jNext < Ly) {
    if (!cluster[Ly*i+jNext])
        try_add(i, jNext, clusterSpin);}
}

void try_add(int i, int j, int clusterSpin) {
    if (s[Ly*i+j] == clusterSpin)
        if (qadran() < addProbability)
            grow_cluster(i, j, clusterSpin);
}

double measure_energy() {

    double E=0;
    for(int i=0; i<Lx; i++) {
        for(int j=0; j<Ly; j++) {
            if(i+1<Lx) E -= J * s[Ly*i+j] * s[Ly*(i+1)+j];
            if(j+1<Ly) E -= J * s[Ly*i+j] * s[Ly*i+j+1];
        }
    }

    return E;
}

void wolff_obc_generate_samples(int* samples, double* energies, int numSamples, int L, double temperature, int seed) {
    int MCSteps;
    Lx = L;
    Ly = L;
    T = temperature;
    MCSteps = numSamples;
    rng.seed(seed);
    N = Lx * Ly;

    initialize();
    initialize_cluster();
    int thermSteps = 1000;
    for (int i = 0; i < thermSteps; i++)
        do_mc_step();

    for (int i = 0; i < MCSteps; i++) {
        //for(int k=0; k<5; k++) do_mc_step();
        do_mc_step();
        memcpy(samples+N*i, &s[0], N*sizeof(int));
        energies[i] = measure_energy();
    }
}

