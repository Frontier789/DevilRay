#pragma once

#include <cuda_runtime.h>

struct CudaTimer
{
    CudaTimer();
    ~CudaTimer();

    void reset();
    float elapsedMs() const;

    cudaEvent_t start;
    cudaEvent_t stop;
};
