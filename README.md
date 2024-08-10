# Parallel Implementation of the Ising Model in C++ using CUDA

## Overview

Parallel implementation of the Ising Model using C++ and CUDA. The Ising Model is a mathematical model of ferromagnetism in statistical mechanics, and this implementation leverages CUDA to perform large-scale simulations efficiently on NVIDIA GPUs.

## Prerequisites

- **NVIDIA GPU**: A compatible NVIDIA GPU for CUDA execution.
- **CUDA Toolkit**: The CUDA Toolkit installed on your system.
- **NVCC Compiler**: NVIDIA's CUDA Compiler (`nvcc`), which comes with the CUDA Toolkit.

#### run using:
nvcc -g ising_model_GPU.cu params_GPU.cu -o ising_model
./ising_model