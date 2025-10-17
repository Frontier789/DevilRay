#pragma once

#include "tracing/Material.hpp"
#include "tracing/Objects.hpp"
#include "device/Vector.hpp"

struct Scene
{
    void deleteDeviceMemory();
    void ensureDeviceAllocation();

    DeviceVector<Object> objects;
    DeviceVector<Material> materials;
};

