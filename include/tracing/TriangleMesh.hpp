#pragma once

#include "Utils.hpp"
#include "models/Mesh.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "models/BBH.hpp"
#include "device/Vector.hpp"
#include "Transform.hpp"

struct TriangleMeshView
{
    Vec3 *points;
    Vec3 *normals;
    Triangle *triangles;
    int triangle_count;

    Transform model_to_world;
    int material;

    AliasTableView triangle_sampler;
    float surface_area;
    float base_surface_area;

    BBHView bbh;

    void setPosition(const Vec3 &pos);
    void setScale(const Vec3 &scale);
};

struct TriangleMesh
{
    DeviceVector<Vec3> points;
    DeviceVector<Vec3> normals;
    DeviceVector<Triangle> triangles;

    AliasTable triangle_sampler;

    BBH bbh;

    TriangleMeshView view();
};

TriangleMesh convertMeshToTris(Mesh &mesh, bool generateTriangleSampler = true);
TriangleMesh createQuadMesh(Vec3 center, Vec3 normal, Vec3 right, float size);
