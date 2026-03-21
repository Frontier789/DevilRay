#pragma once

#include "tracing/Material.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/GpuTris.hpp"
#include "device/Vector.hpp"

#include <list>

struct ObjectsInfo
{
    float total_surface_area;
};

struct Scene
{
    void deleteDeviceMemory();
    void ensureDeviceAllocation();

    ObjectsInfo info;

    DeviceVector<TriangleMesh> objects;
    DeviceVector<Material> materials;

    std::list<GpuTris> mesh_storage;
};

