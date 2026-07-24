#pragma once

#include "device/DeviceMemory.hpp"

#include <span>
#include <vector>

template<typename T>
struct DeviceBuffer
{
    DeviceMemoryManager deviceData;
    int elementCount;

    T *devicePtr() { return deviceData.getPtr<T>(); }
    std::span<T> deviceSpan() { return std::span{devicePtr(), static_cast<size_t>(elementCount)}; }

    std::vector<T> toHost() const
    {
        std::vector<T> host(static_cast<size_t>(elementCount));
        deviceData.copyToHost(host.data(), sizeof(T) * static_cast<size_t>(elementCount));
        return host;
    }

    static DeviceBuffer<T> allocate(int numberOfElements)
    {
        DeviceMemoryManager data;
        data.allocate(numberOfElements * sizeof(T));

        return DeviceBuffer {
            .deviceData = std::move(data),
            .elementCount = numberOfElements,
        };
    }
};

template<typename U>
DeviceBuffer<U> deviceBufferFrom(const std::vector<U> &hostData)
{
    auto buffer = DeviceBuffer<U>::allocate(static_cast<int>(hostData.size()));
    buffer.deviceData.copyFromHost(hostData.data(), hostData.size() * sizeof(U));

    return buffer;
}
