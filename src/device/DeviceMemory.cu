#include "device/DeviceMemory.hpp"
#include "device/DevUtils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

DeviceMemoryManager::DeviceMemoryManager()
    : m_devicePtr{nullptr}
{}

DeviceMemoryManager::DeviceMemoryManager(DeviceMemoryManager &&other)
    : m_devicePtr{other.m_devicePtr}
{
    other.m_devicePtr = nullptr;
}

DeviceMemoryManager &DeviceMemoryManager::operator=(DeviceMemoryManager &&other)
{
    free();

    m_devicePtr = other.m_devicePtr;
    other.m_devicePtr = nullptr;

    return *this;
}

DeviceMemoryManager::~DeviceMemoryManager()
{
    free();
}

void DeviceMemoryManager::allocate(size_t bytes)
{
    free();

    cudaMalloc(&m_devicePtr, bytes);
    CUDA_ERROR_CHECK();
}

void DeviceMemoryManager::free()
{
    if (m_devicePtr) {
        cudaFree(m_devicePtr);
        m_devicePtr = nullptr;
    }
}

void DeviceMemoryManager::copyFromHost(const void *src, size_t bytes)
{
    if (empty()) {
        throw std::runtime_error("Copying into an unallocated device buffer.");
    }

    cudaMemcpy(m_devicePtr, src, bytes, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();
}

void DeviceMemoryManager::copyToHost(void *dst, size_t bytes) const
{
    if (m_devicePtr == nullptr) return;

    cudaMemcpy(dst, m_devicePtr, bytes, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();
}

bool DeviceMemoryManager::empty() const
{
    return m_devicePtr == nullptr;
}
