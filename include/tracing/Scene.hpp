#pragma once

#include "tracing/Material.hpp"
#include "tracing/TriangleMesh.hpp"
#include "device/Vector.hpp"

#include <list>

struct ObjectsInfo
{
    float total_radiant_power;
};

struct Scene
{
    void deleteDeviceMemory();
    void ensureDeviceAllocation();

    ObjectsInfo info;

    DeviceVector<TriangleMeshView> objects{{}};
    DeviceVector<Material> materials{{}};

    std::list<TriangleMesh> mesh_storage;
};

