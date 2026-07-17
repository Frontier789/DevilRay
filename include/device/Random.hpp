#pragma once

#include "device/Buffer.hpp"
#include "Utils.hpp"

#include <curand_kernel.h>

using curandState = curandStateXORWOW;

struct CudaRandom
{
    CudaRandom(int state_count);

    curandState *devicePtr() {return rand_states.devicePtr();}

private:
    DeviceBuffer<curandState> rand_states;

    void init();
};
