#pragma once

void printCudaDeviceInfo();

#define CUDA_ERROR_CHECK() { cudaCheckLastError(__FILE__, __LINE__, true); }
void cudaCheckLastError(const char *file, int line, bool abort=true);

#define KERNEL_IDX(limit) blockIdx.x * blockDim.x + threadIdx.x; if ((blockIdx.x * blockDim.x + threadIdx.x) >= (limit)) return;