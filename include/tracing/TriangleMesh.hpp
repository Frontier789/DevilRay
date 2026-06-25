#pragma once

#include "Utils.hpp"
#include "models/Mesh.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "models/BBH.hpp"
#include "Transform.hpp"

struct TriangleMesh
{
    Vec3 *points;
    Vec3 *normals;
    Triangle *triangles;
    int triangle_count;

    Transform model_to_world;
    int material;

    AliasEntry *triangle_sampler;
    float surface_area;
    float base_surface_area;

    BBHGpuView bbh;

    void setPosition(const Vec3 &pos);
    void setScale(const Vec3 &scale);
};
