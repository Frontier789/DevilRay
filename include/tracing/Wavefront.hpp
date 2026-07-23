#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

#include <span>

struct WavefrontData
{
    DeviceBuffer<Ray> rays;
    DeviceBuffer<int> current_mat;
};

struct WavefrontDataDevice
{
    Ray *rays;
    int *current_mat;
};
