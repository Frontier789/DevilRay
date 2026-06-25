#pragma once

#include "Utils.hpp"
#include "device/Random.hpp"
#include "device/Array.hpp"

void printCudaDeviceInfo();

#define CUDA_ERROR_CHECK() { cudaCheckLastError(__FILE__, __LINE__, true); }
void cudaCheckLastError(const char *file, int line, bool abort=true);
