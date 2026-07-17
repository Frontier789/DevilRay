#pragma once

#include <curand_kernel.h>

struct CudaRng
{
    curandState *state;

    __device__ float rnd()
    {
        return curand_uniform(state);
    }
};
