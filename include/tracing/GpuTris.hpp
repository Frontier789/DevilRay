#pragma once

#include "tracing/DistributionSamplers.hpp"
#include "tracing/Material.hpp"
#include "tracing/TriangleMesh.hpp"
#include "device/Vector.hpp"
#include "models/Mesh.hpp"

#include "Utils.hpp"

struct GpuTris
{
    DeviceVector<Vec3> points;
    DeviceVector<Vec3> normals;
    DeviceVector<Triangle> triangles;

    AliasTable triangleSampler;

    BBH bbh;
};

GpuTris convertMeshToTris(Mesh &mesh, bool generateTriangleSampler = true);
GpuTris createQuadMesh(Vec3 center, Vec3 normal, Vec3 right, float size);
TriangleMesh viewGpuTris(GpuTris &tris);
