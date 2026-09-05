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

    void copyFromHost(const T *ptr, size_t count)
    {
        deviceData.copyFromHost(ptr, sizeof(T) * count);
    }

    void copyFromHost(const std::vector<T> &hostData)
    {
        deviceData.copyFromHost(hostData.data(), sizeof(T) * hostData.size());
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

    static DeviceBuffer<T> fromHost(const std::vector<T> &hostData)
    {
        const auto numel = static_cast<int>(hostData.size());

        DeviceMemoryManager data;
        data.allocate(numel * sizeof(T));
        data.copyFromHost(hostData.data(), numel * sizeof(T));

        return DeviceBuffer {
            .deviceData = std::move(data),
            .elementCount = numel,
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
