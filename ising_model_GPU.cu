#include "params_GPU.cuh"

// Parallelized Ising Model with CUDA

// nvcc -g ising_model_GPU.cu params_GPU.cu -o ising_model
// ./ising_model

// Declare the device functions

// a function to get the energy difference using nearest neighbours sum
__device__ int diff(int *mat, int i, int j)
{
    int current_position = mat[i * mat_size + j];
    int nearest_neighbors = 0;

    // here its assumed that when the spin is on the borders it will have only a subset
    // of neighbours according to its position. A spin on the upper right corner (0,0)
    // will only have two neighbors 0,1 ; 1,0
    if (i == 0)
    {
        nearest_neighbors += mat[(i + 1) * mat_size + j];
    }
    else if (i == mat_size - 1)
    {
        nearest_neighbors += mat[(i - 1) * mat_size + j];
    }
    else
    {
        nearest_neighbors += mat[(i + 1) * mat_size + j];
        nearest_neighbors += mat[(i - 1) * mat_size + j];
    }

    if (j == 0)
    {
        nearest_neighbors += mat[i * mat_size + (j + 1)];
    }
    else if (j == mat_size - 1)
    {
        nearest_neighbors += mat[i * mat_size + (j - 1)];
    }
    else
    {
        nearest_neighbors += mat[i * mat_size + (j + 1)];
        nearest_neighbors += mat[i * mat_size + (j - 1)];
    }

    int nn_operation = current_position * nearest_neighbors;

    return nn_operation;
}

__global__ void update(int iter_per_thread, int *matrix, float *LookUpTable, int mat_size, int parallel_spins, int row_seed, int col_seed, int threshold_seed)
{

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id <= parallel_spins)
    {
        curandState row_state;
        curand_init(row_seed, thread_id, 0, &row_state);

        curandState col_state;
        curand_init(col_seed, thread_id, 0, &col_state);

        curandState threshold_state;
        curand_init(threshold_seed, thread_id, 0, &threshold_state);

        for (int i = 0; i <= iter_per_thread; i++)
        {
            double row_value = curand_uniform_double(&row_state);
            // Scale and round the random float to an integer in [0, mat_size]
            int spin_row = static_cast<int>(row_value * mat_size);

            double col_value = curand_uniform_double(&col_state);
            // Scale and round the random float to an integer in [0, mat_size]
            int spin_col = static_cast<int>(col_value * mat_size);

            float threshold_value = curand_uniform(&threshold_state);

            // energy difference
            // common subexpression
            int dE = diff(matrix, spin_row, spin_col);

            // spin criteria

            if (dE <= 0 || threshold_value < LookUpTable[dE + 4])
            {
                matrix[spin_row * mat_size + spin_col] *= -1;
            }
        }
    }

    __syncthreads();
}

int main()
{

    auto start = std::chrono::high_resolution_clock::now();
    int *device_mat;
    cudaMalloc((void **)&device_mat, matrix_init_spins * sizeof(int));

    // block and grid dimensions for kernel launch
    dim3 block_dim(32, 32);
    dim3 grid_dim((mat_size + block_dim.x - 1) / block_dim.x, (mat_size + block_dim.y - 1) / block_dim.y);
    init<<<grid_dim, block_dim>>>(device_mat, mat_size, seed);
    cudaDeviceSynchronize();

    int *mat = new int[matrix_init_spins];
    cudaMemcpy(mat, device_mat, matrix_init_spins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_mat);

    if (mat_size <= 64)
    {
        //  a function to print the final state of the matrix
        print_state(mat, mat_size);
    }

    mean_energy(mat, mat_size);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxBlocks = prop.maxGridSize[0]; // Maximum number of blocks in the x-dimension.
    std::cout << "max blocks: " << maxBlocks << "\n"
              << std::endl;

    std::cout << "parallel_spins: " << parallel_spins << "\n"
              << std::endl;
    std::cout << "num_threads: " << num_threads << "\n"
              << std::endl;
    std::cout << "num_blocks: " << num_blocks << "\n"
              << std::endl;
    std::cout << "Interations: " << N_iter << "\n"
              << std::endl;
    std::cout << "Interations per Thread: " << iter_per_thread << "\n"
              << std::endl;

    // create a lookup table with the 9 possible values of the change in energy
    float LookUpTable[9] = {
        std::exp(8.0f * T_inv), // -8
        std::exp(6.0f * T_inv), // -6
        std::exp(4.0f * T_inv), // -4
        std::exp(2.0f * T_inv), // -2
        1.0f,                   // 0
        std::exp(-2.0f * T_inv),
        std::exp(-4.0f * T_inv),
        std::exp(-6.0f * T_inv),
        std::exp(-8.0f * T_inv)};

    float *device_LookUpTable;
    cudaMalloc((void **)&device_LookUpTable, 9 * sizeof(float));
    cudaMemcpy(device_LookUpTable, LookUpTable, 9 * sizeof(float), cudaMemcpyHostToDevice);

    int *device_result;
    cudaMalloc((void **)&device_result, matrix_init_spins * sizeof(int));
    cudaMemcpy(device_result, mat, matrix_init_spins * sizeof(int), cudaMemcpyHostToDevice);

    // a function that runs the monte carlo simulation based on Metropolis Algorithm
    update<<<num_blocks, num_threads>>>(iter_per_thread, device_result, device_LookUpTable, mat_size, parallel_spins, row_seed, col_seed, threshold_seed);
    cudaDeviceSynchronize();

    cudaError_t copyError = cudaGetLastError();
    if (copyError != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(copyError));
    }

    int *host_result = new int[matrix_init_spins];
    cudaMemcpy(host_result, device_result, matrix_init_spins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_LookUpTable);
    cudaFree(device_result);

    //  a function to print the final state of the matrix
    if (mat_size <= 64)
    {
        //  a function to print the final state of the matrix
        print_state(host_result, mat_size);
        // print the mean of energy in the lattice
        mean_energy(host_result, mat_size);
    }
    else
    {
        mean_energy(host_result, mat_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() / 1000000 << " seconds" << std::endl;
}
