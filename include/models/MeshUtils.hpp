#pragma once

#include "Utils.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/TriangleMesh.hpp"

inline HD Vec3 triangleNormal(const TriangleVertices &triangle)
{
    return (triangle.a - triangle.b).cross(triangle.a - triangle.c).normalized();
}

inline HD Vec3 triangleBarycentric(const TriangleVertices &triangle, const Vec3 &point)
{
    const auto n_f = (triangle.a - triangle.b).cross(triangle.a - triangle.c);
    const auto sgn_area2 = n_f.dot(n_f);

    const auto n_1 = (point - triangle.a).cross(triangle.c - triangle.a);
    const auto n_2 = (triangle.b - triangle.a).cross(point - triangle.a);

    const auto w_b = n_f.dot(n_1) / sgn_area2;
    const auto w_c = n_f.dot(n_2) / sgn_area2;
    const auto w_a = 1 - w_b - w_c;

    return Vec3{w_a, w_b, w_c};
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
