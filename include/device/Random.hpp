#pragma once

#include "Utils.hpp"

struct curandStateXORWOW;
using curandState = curandStateXORWOW;

struct CudaRandomStates
{
    CudaRandomStates(Size2i resolution);
    ~CudaRandomStates();

    inline curandState *ptr() const {return rand_states;}

private:
    Size2i size;
    curandState *rand_states;

    void init();
};
