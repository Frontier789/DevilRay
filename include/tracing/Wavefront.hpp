#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"

#include <span>

struct WavefrontDataView
{
    Ray *rays;
    int *current_mat;
    int *sort_index;
};

struct WavefrontData
{
    DeviceBuffer<Ray> rays;
    DeviceBuffer<int> current_mat;
    DeviceBuffer<int> sort_index;

    WavefrontDataView view()
    {
        return WavefrontDataView{
            .rays = rays.devicePtr(),
            .current_mat = current_mat.devicePtr(),
            .sort_index = sort_index.devicePtr(),
        };
    }
};
