#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

#include <cstdint>

struct Buffers
{
    Buffers(Size2i resolution);

    void reset();
    void ensureDeviceAllocation();

    uint64_t totalCasts() const;

    static constexpr int maxPathLength = 10;

    DeviceArray<Vec4> color;
    mutable DeviceArray<uint32_t> casts;
};
