#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <optional>

#include "device/DevUtils.hpp"

#include "tracing/Camera.hpp"
#include "tracing/TriangleMesh.hpp"
#include "device/Random.hpp"

void cudaCheckLastError(const char *file, int line, bool abort)
{
    const auto code = cudaPeekAtLastError();

    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort) exit(code);
    }
}


CudaRandom::CudaRandom(int state_count)
{
    rand_states = DeviceBuffer<curandState>::allocate(state_count);
    init();
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

__global__ void initRand(curandState *randStates, int elemCount, unsigned long seed) {
    int idx = KERNEL_IDX(elemCount);

    curand_init(seed, idx, 0, randStates + idx);
}

void CudaRandom::init()
{
    const int elemCount = rand_states.elementCount;

    dim3 dimBlock(128, 1);
    dim3 dimGrid;
    dimGrid.x = (elemCount + dimBlock.x - 1) / dimBlock.x;

    initRand<<<dimGrid, dimBlock>>>(rand_states.devicePtr(), elemCount, 42);
    CUDA_ERROR_CHECK();
}
