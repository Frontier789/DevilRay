#include "device/Vector.hpp"

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
}

void DeviceMemoryManager::free()
{
    cudaFree(m_devicePtr);
    m_devicePtr = nullptr;
}

void DeviceMemoryManager::memcpy(const void *src, size_t bytes)
{
    if (empty()) {
        throw std::runtime_error("Memcpying into a nullptr device memory.");
    }

    cudaMemcpy(m_devicePtr, src, bytes, cudaMemcpyHostToDevice);
}

bool DeviceMemoryManager::empty() const
{
    return m_devicePtr == nullptr;
}
