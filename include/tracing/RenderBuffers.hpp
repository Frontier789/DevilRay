#pragma once

#include "Utils.hpp"
#include "device/Buffer.hpp"

struct RenderBuffersView
{
    Vec4 *colors;

    Size2i resolution;
};

struct RenderBuffers
{
    DeviceBuffer<Vec4> colors;

    Size2i resolution;

    RenderBuffersView view()
    {
        return RenderBuffersView{
            .colors = colors.devicePtr(),
            .resolution = resolution,
        };
    }
};