#pragma once

#include "tracing/TriangleMesh.hpp"
#include "tracing/Intersection.hpp"

HD Vec3 triangleNormal(const TriangleVertices &triangle)
{
    return (triangle.a - triangle.b).cross(triangle.a - triangle.c).normalized();
}

HD std::optional<Intersection> getIntersection(const Ray &ray, const TriangleMesh &tris)
{
    std::optional<Intersection> best = std::nullopt;

    for (int i=0;i<tris.tris_count;++i)
    {
        const auto &indices = tris.triangles[i];

        const auto triangle = TriangleVertices{
            .a = tris.points[indices.a.pi] * tris.s + tris.p,
            .b = tris.points[indices.b.pi] * tris.s + tris.p,
            .c = tris.points[indices.c.pi] * tris.s + tris.p,
        };

        const auto intersection = testTriangleIntersection(ray, triangle);
        
        if (!intersection.has_value()) continue;

        if (!best.has_value() || best->t > intersection->t)
        {
            const auto norm = (triangle.a - triangle.b).cross(triangle.a - triangle.c);
            const bool is_ccw = norm.dot(ray.v) > 0;

            auto n = triangleNormal(triangle);
            if (dot(n, ray.v) > 0) n = n * -1;

            best = Intersection{
                .t = intersection->t,
                .p = ray.p + ray.v * intersection->t,
                .uv = Vec2f{0,0},
                .n = n,
                .mat = tris.material,
                .triangle = TriangleHitData{
                    .bari = intersection->bari,
                    .area = triangleArea(triangle.a, triangle.b, triangle.c),
                    .ccw = is_ccw,
                },
            };
        }
    }

    return best;
}