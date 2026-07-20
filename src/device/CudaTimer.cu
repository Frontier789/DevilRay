// AI-assisted (Claude): CudaTimer extracted from RendererImpl.hpp into its own TU.
#include "device/CudaTimer.hpp"

#include "device/DevUtils.hpp"

#include <cuda_runtime.h>

CudaTimer::CudaTimer()
{
    cudaEventCreate(&start);
    CUDA_ERROR_CHECK();

    cudaEventCreate(&stop);
    CUDA_ERROR_CHECK();

    cudaEventRecord(start, 0);
    CUDA_ERROR_CHECK();
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaTimer::reset()
{
    cudaEventRecord(start, 0);
    CUDA_ERROR_CHECK();
}

float CudaTimer::elapsedMs() const
{
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    return elapsed_ms;
}
