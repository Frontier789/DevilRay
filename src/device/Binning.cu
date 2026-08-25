#include "device/Binning.hpp"

#include <numeric>

#include <cub/device/device_select.cuh>

DeviceBinning DeviceBinning::create(int key_count)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, 
        temp_storage_bytes, 
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        static_cast<int*>(nullptr),
        key_count
    );

    auto identity_index_vec = std::vector<int>(key_count, 0);
    std::iota(identity_index_vec.begin(), identity_index_vec.end(), 0);

    return DeviceBinning{
        .temp_storage = DeviceBuffer<std::byte>::allocate(temp_storage_bytes),
        .identity_keys = DeviceBuffer<int>::fromHost(identity_index_vec),
        .num_selected = DeviceBuffer<int>::fromHost({key_count}),
    };
}

void DeviceBinning::execute(int *flag, int *output_indices)
{
    auto temp_byte_count = static_cast<size_t>(temp_storage.elementCount);

    cub::DeviceSelect::Flagged(
        temp_storage.devicePtr(),
        temp_byte_count,
        identity_keys.devicePtr(),
        flag,
        output_indices,
        num_selected.devicePtr(),
        identity_keys.elementCount
    );
}
