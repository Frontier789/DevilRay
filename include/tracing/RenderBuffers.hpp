#pragma once

#include "Utils.hpp"
#include "device/Buffer.hpp"

struct RenderBuffers
{
    DeviceBuffer<Vec4> colors;
};

struct RenderBuffersDevice
{
    Vec4 *colors;

    Size2i resolution;
};