#include "params_GPU.cuh"

__global__ void init(int *matrix, int mat_size, int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < mat_size && j < mat_size)
    {
        int idx = i * mat_size + j;
        curandState state;
        curand_init(seed, idx, 0, &state);
        float value = curand_uniform(&state);
        if (value > 0.5)
        {
            matrix[idx] = 1;
        }
        else
        {
            matrix[idx] = -1;
        }
    }
}

// a function to print the final state of the matrix
__host__ void print_state(int *mat, int mat_size)
{
    // print matrix
    for (int i = 0; i < mat_size; ++i)
    {
        for (int j = 0; j < mat_size; ++j)
        {
            if (mat[i * mat_size + j] == 1)
            {
                std::cout << "+ ";
            }
            else
            {
                std::cout << "- ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
// a function to get the mean energy on the lattice
__host__ void mean_energy(int *mat, int mat_size)
{
    double sum = 0.0;
    for (int i = 0; i < mat_size; i++)
    {
        for (int j = 0; j < mat_size; j++)
        {
            sum += mat[i * mat_size + j];
        }
    }

    int total_elements = mat_size * mat_size;
    double mean = sum / (total_elements);
    std::cout << "Mean of " << total_elements << " samples: " << mean << std::endl;
}
