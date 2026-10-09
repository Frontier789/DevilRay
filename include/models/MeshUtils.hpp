#pragma once

#include "Utils.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/TriangleMesh.hpp"

inline HD Vec3 triangleFaceNormal(const TriangleVertices &triangle)
{
    return (triangle.a - triangle.b).cross(triangle.a - triangle.c);
}

inline HD Vec3 barycentricCoordinates(const TriangleVertices &triangle, const Vec3 &normal, const Vec3 &hit)
{
    const auto sgn_area2 = normal.dot(normal);

    auto n = normal.normalized();

    const auto n_1 = (hit - triangle.a).cross(triangle.c - triangle.a);
    const auto n_2 = (triangle.b - triangle.a).cross(hit - triangle.a);

    const auto w_B = normal.dot(n_1) / sgn_area2;
    const auto w_C = normal.dot(n_2) / sgn_area2;
    const auto w_A = 1 - w_B - w_C;

    return Vec3{w_A, w_B, w_C};
}

// inline HD Vec3 barycentricCoordinates(const TriangleVertices &triangle, const Vec3 &hit)
// {
//     const auto n_f = triangleFaceNormal(triangle);
//     const auto sgn_area2 = n_f.dot(n_f);

//     auto n = n_f.normalized();

//     const auto n_1 = (hit - triangle.a).cross(triangle.c - triangle.a);
//     const auto n_2 = (triangle.b - triangle.a).cross(hit - triangle.a);

//     const auto w_B = n_f.dot(n_1) / sgn_area2;
//     const auto w_C = n_f.dot(n_2) / sgn_area2;
//     const auto w_A = 1 - w_B - w_C;

//     return Vec3{w_A, w_B, w_C};
// }

inline HD Vec3 surfaceNormal(const TriangleMeshView &object, int triangleID, const Vec3 &ray_dir, const Vec3 &hit)
{
    const auto &triangle = object.triangles[triangleID];
    const auto &M = object.model_to_world;

    const auto triangle_vertices = TriangleVertices{
        .a = M.applyToPoint(object.points[triangle.a.pi]),
        .b = M.applyToPoint(object.points[triangle.b.pi]),
        .c = M.applyToPoint(object.points[triangle.c.pi]),
    };

    const auto flat_normal = triangleFaceNormal(triangle_vertices);
    const auto bary = barycentricCoordinates(triangle_vertices, flat_normal, hit);
    const auto &inv_s = object.model_to_world.s.inv();
    auto n = ((object.normals[triangle.a.ni] * bary.x +
               object.normals[triangle.b.ni] * bary.y +
               object.normals[triangle.c.ni] * bary.z) * inv_s).normalized();

    if (flat_normal.dot(ray_dir) > 0) n = n * -1;

    return n;
}
