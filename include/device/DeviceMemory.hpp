#pragma once

#include <cstddef>

// Owns a single device allocation and isolates the raw CUDA transfer calls
// behind a minimal RAII interface. Both DeviceArray and DeviceVector build on
// this, so every cudaMalloc/cudaMemcpy lives in one translation unit.
struct DeviceMemoryManager
{
    DeviceMemoryManager();
    DeviceMemoryManager(const DeviceMemoryManager &) = delete;
    DeviceMemoryManager(DeviceMemoryManager &&);
    ~DeviceMemoryManager();

    DeviceMemoryManager &operator=(const DeviceMemoryManager &) = delete;
    DeviceMemoryManager &operator=(DeviceMemoryManager &&);

    template<typename T>
    T *getPtr() { return static_cast<T*>(m_devicePtr); }

    void allocate(size_t bytes);
    void free();
    void copyFromHost(const void *src, size_t bytes);
    void copyToHost(void *dst, size_t bytes) const;
    bool empty() const;

private:
    void *m_devicePtr;
};
