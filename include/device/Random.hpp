#pragma once

#include "device/Vector.hpp"
#include "Utils.hpp"

#include <curand_kernel.h>

using curandState = curandStateXORWOW;

struct CudaRandomStates
{
    CudaRandomStates(Size2i resolution);

    curandState *devicePtr() const {return rand_states.devicePtr();}

private:
    Size2i size;
    mutable DeviceVector<curandState> rand_states;

    void init();
};
