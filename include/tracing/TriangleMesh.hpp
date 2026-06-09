#pragma once

#include "Utils.hpp"
#include "models/Mesh.hpp"
#include "tracing/DistributionSamplers.hpp"

struct TriangleMesh
{
    Vec3 *points;
    Vec3 *normals;
    Triangle *triangles;
    int tris_count;

    Vec3 s;
    Vec3 p;
    int material;
    
    AliasEntry *tris_sampler;
    float surface_area;
    float base_surface_area;

    void setPosition(const Vec3 &pos);
    void setScale(const Vec3 &scale);
};
