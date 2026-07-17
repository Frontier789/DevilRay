#pragma once

#include "device/DeviceMemory.hpp"

#include <span>

template<typename T>
struct DeviceBuffer
{
    DeviceMemoryManager deviceData;
    int elementCount;

    T *devicePtr() { return deviceData.getPtr<T>(); }
    std::span<T> deviceSpan() { return std::span{devicePtr(), static_cast<size_t>(elementCount)}; }

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
