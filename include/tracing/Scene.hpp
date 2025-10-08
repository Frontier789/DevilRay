#pragma once

#include "tracing/Objects.hpp"
#include "device/Vector.hpp"

struct Scene
{
    void deleteDeviceMemory();
    void ensureDeviceAllocation();

    DeviceVector<Object> objects;
    DeviceVector<Material> materials;
};

