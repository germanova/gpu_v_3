#ifndef PARAMS
#define PARAMS

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <vector_types.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <chrono>

// number of monte carlo iterations
const int N_iter = 1000000000;
// 1/temperature for strength reduction
const float T_inv = 1.0f / 2.0f;
// lattice size
const int mat_size = 32;
// parameters for matrix initialization
const int matrix_init_spins = mat_size * mat_size;
const int seed = 99;
// the fraction of spins to try to flip in parallel
const double frac_elements = 0.5;
// how many threads will be needed
const int parallel_spins = matrix_init_spins * frac_elements;
// number of threads per block
const int num_threads = 1024;
// number of blocks
const int num_blocks = (parallel_spins + num_threads - 1) / num_threads;
// iterarions per thread
const int iter_per_thread = N_iter / parallel_spins;
// seeds
const int row_seed = 10;
const int col_seed = 29;
const int threshold_seed = 44;

__global__ void init(int *matrix, int mat_size, int seed);
__host__ void print_state(int *mat, int mat_size);
__host__ void mean_energy(int *mat, int mat_size);
// end the if stated at the start
#endif