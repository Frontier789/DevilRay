#include "device/Array.hpp"

#include <cuda_runtime.h>

namespace device_array
{
    void *deviceAlloc(int byteCount)
    {
        void *ptr;
        cudaMalloc(&ptr, byteCount);
        return ptr;
    }

    void deviceFree(void *ptr)
    {
        cudaFree(ptr);
    }

    void copyToDevice(void *device_target, const void *host_source, int byteCount)
    {
        cudaMemcpy(device_target, host_source, byteCount, cudaMemcpyHostToDevice);
    }
    
    void copyToHost(void *host_target, const void *device_source, int byteCount)
    {
        cudaMemcpy(host_target, device_source, byteCount, cudaMemcpyDeviceToHost);
    }
}
