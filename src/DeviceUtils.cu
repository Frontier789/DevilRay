#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <optional>

#include "DeviceUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/SampleScene.hpp"

void cudaCheckLAstError(const char *file, int line, bool abort)
{
    const auto code = cudaPeekAtLastError();
    
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort) exit(code);
    }
}


CudaRandomStates::CudaRandomStates(Size2i resolution)
    : size(resolution)
    , rand_states(nullptr)
{
    cudaMalloc(&rand_states, resolution.width * resolution.height * sizeof(*rand_states));

    init();
}

CudaRandomStates::~CudaRandomStates()
{
    cudaFree(rand_states);
    rand_states = nullptr;
}


void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return;
    }

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "CUDA Device Info:" << std::endl;
    std::cout << "Name: " << deviceProp.name << std::endl;
    std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
}

__global__ void initRand(curandState *randStates, int width, int height, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &randStates[idx]);
}

void CudaRandomStates::init()
{
    dim3 dimBlock(32, 32);
    dim3 dimGrid;
    dimGrid.x = (size.width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (size.height + dimBlock.y - 1) / dimBlock.y;

    initRand<<<dimGrid, dimBlock>>>(rand_states, size.width, size.height, 42);
    CUDA_ERROR_CHECK();
}


#include "DeviceVectorImpl.hpp"
