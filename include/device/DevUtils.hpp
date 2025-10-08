#pragma once

#include "Utils.hpp"
#include "device/Random.hpp"
#include "device/Array.hpp"

void printCudaDeviceInfo();

#define CUDA_ERROR_CHECK() { cudaCheckLAstError(__FILE__, __LINE__); }
void cudaCheckLAstError(const char *file, int line, bool abort=true);
