#pragma once

#include "device/Buffer.hpp"

#include <cstddef>

struct DeviceBinning
{
    DeviceBuffer<std::byte> temp_storage;
    DeviceBuffer<int> identity_keys;
    DeviceBuffer<int> num_selected;

    static DeviceBinning create(int key_count);

    void execute(int *flag, int *output_indices);
};
