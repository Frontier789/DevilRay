#pragma once

#include "Utils.hpp"
#include "tracing/Material.hpp"
#include "tracing/Objects.hpp"
#include "device/Vector.hpp"
#include "models/Mesh.hpp"

struct GpuTris
{
    DeviceVector<Vec3> points;
    DeviceVector<Vec3> normals;
    DeviceVector<Triangle> triangles;
};

GpuTris convertMeshToTris(const Mesh &mesh);
TrisCollection viewGpuTris(GpuTris &tris);
