#pragma once

#include "tracing/DistributionSamplers.hpp"
#include "tracing/Material.hpp"
#include "tracing/Objects.hpp"
#include "device/Vector.hpp"
#include "models/Mesh.hpp"

#include "Utils.hpp"

struct GpuTris
{
    DeviceVector<Vec3> points;
    DeviceVector<Vec3> normals;
    DeviceVector<Triangle> triangles;

    AliasTable triangleSampler;
};

GpuTris convertMeshToTris(const Mesh &mesh);
TrisCollection viewGpuTris(GpuTris &tris);
