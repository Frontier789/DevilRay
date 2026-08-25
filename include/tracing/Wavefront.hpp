#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

#include <span>

struct WavefrontData
{
    DeviceBuffer<Ray> rays;
    DeviceBuffer<int> current_mat;
    DeviceBuffer<int> sort_index;
};

struct WavefrontDataDevice
{
    Ray *rays;
    int *current_mat;
    int *sort_index;
};
