#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

#include <span>

struct WavefrontData
{
    DeviceBuffer<Ray> rays;
};

struct WavefrontDataDevice
{
    Ray *rays;
};
