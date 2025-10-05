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
