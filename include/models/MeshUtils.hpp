#pragma once

#include "Utils.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/TriangleMesh.hpp"

inline HD Vec3 triangleNormal(const TriangleVertices &triangle)
{
    return (triangle.a - triangle.b).cross(triangle.a - triangle.c).normalized();
}

inline HD Vec3 surfaceNormal(const TriangleMesh &object, int triangleID, const Vec3 &ray_dir)
{
    const auto &triangle = object.triangles[triangleID];

    const auto triangle_vertices = TriangleVertices{
        .a = object.points[triangle.a.pi],
        .b = object.points[triangle.b.pi],
        .c = object.points[triangle.c.pi],
    };

    auto n = triangleNormal(triangle_vertices);
    const auto &inv_s = object.model_to_world.s.inv();
    n = (n * inv_s).normalized();

    if (n.dot(ray_dir) > 0) n = n * -1;

    return n;
}
