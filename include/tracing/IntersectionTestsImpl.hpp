#pragma once

#include "tracing/TriangleMesh.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Benchmark.hpp"

#include "Scene.hpp"

HD Vec3 triangleNormal(const TriangleVertices &triangle)
{
    return (triangle.a - triangle.b).cross(triangle.a - triangle.c).normalized();
}

template<Benchmark B>
HD std::optional<Intersection> getIntersectionImpl(const Ray &ray, const TriangleMesh &tris, B &benchmark)
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

        benchmark.registerTriangleTest();
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

HD std::optional<Intersection> getIntersection(const Ray &ray, const TriangleMesh &tris)
{
    benchmark::Skip skip_benchmarks;
    return getIntersectionImpl(ray, tris, skip_benchmarks);
}

HD std::optional<Intersection> getIntersectionBenchmark(const Ray &ray, const TriangleMesh &tris, benchmark::HitTests &benchmark)
{
    return getIntersectionImpl(ray, tris, benchmark);
}


HD std::optional<Intersection> cast(const Ray &ray, const std::span<const TriangleMesh> objects, const ObjectsInfo &info)
{
    std::optional<Intersection> best = std::nullopt;

    for (const auto &mesh : objects)
    {
        auto intersection = getIntersection(ray, mesh);

        if (!intersection.has_value()) continue;

        if (!best.has_value() || best->t > intersection->t)
        {
            intersection->object = &mesh;
            best = intersection;
        }
    }

    return best;
}

HD std::optional<TriangleIntersection> testTriangleIntersection(const Ray &ray, const TriangleVertices &triangle)
{
    if (ray.p.anyNan() || ray.v.anyNan()) return std::nullopt;

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);
    const auto sgn_area2 = n_f.dot(n_f);

    if (sgn_area2 < 1e-14f) return std::nullopt;

    auto n = n_f.normalized();

    auto dp = dot(ray.p - A, n);

    if (dp < 0) {
        n = n * -1;
        dp = -dp;
    }

    // (ray.p - o) . n + ray.v . n * t = 0

    const auto d = -dot(ray.v, n);
    if (d < 1e-7f) return std::nullopt;

    const float t = dp / d;

    const Vec3 p = ray.p + ray.v * t;

    const auto n_1 = (p - A).cross(C - A);
    const auto n_2 = (B - A).cross(p - A);

    const auto w_B = n_f.dot(n_1) / sgn_area2;
    const auto w_C = n_f.dot(n_2) / sgn_area2;
    const auto w_A = 1 - w_B - w_C;

    if (w_A < 0 || w_B < 0 || w_C < 0) return std::nullopt;

    return TriangleIntersection{
        .t = t,
        .bari = Vec3{w_A, w_B, w_C},
    };
}
