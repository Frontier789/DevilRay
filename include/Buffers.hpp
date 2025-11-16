#pragma once

#include "device/DevUtils.hpp"

#include "tracing/Intersection.hpp"

#include <array>

struct Buffers
{
    Buffers(Size2i resolution);
    
    void reset();

    uint64_t totalCasts() const;
    
    static constexpr int maxPathLength = 10;

    DeviceArray<Vec4> color;
    mutable DeviceArray<uint32_t> casts;
};
