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
HD std::optional<Intersection> getIntersectionTris(
    const Ray &ray, const TriangleMesh &tris,
    int tris_begin, int tris_end,
    std::optional<Intersection> &best,
    B &benchmark
){
    for (int i=tris_begin;i<tris_end;++i)
    {
        const auto &indices = tris.triangles[i];

        const auto triangle = TriangleVertices{
            .a = tris.points[indices.a.pi],
            .b = tris.points[indices.b.pi],
            .c = tris.points[indices.c.pi],
        };

        benchmark.registerTriangleTest();
        const auto intersection = testTriangleIntersection(ray, triangle);

        if (!intersection.has_value()) continue;

        if (!best.has_value() || best->t > intersection->t)
        {
            const auto world_triangle = TriangleVertices{
                .a = tris.modelToWorld.applyToPoint(triangle.a),
                .b = tris.modelToWorld.applyToPoint(triangle.b),
                .c = tris.modelToWorld.applyToPoint(triangle.c),
            };

            const auto norm = (triangle.a - triangle.b).cross(triangle.a - triangle.c);
            const bool is_ccw = norm.dot(ray.v) > 0;

            auto n = triangleNormal(triangle);
            if (dot(n, ray.v) > 0) n = n * -1;

            const auto inv_s = tris.modelToWorld.s.inv();
            n = (n * inv_s).normalized();

            const auto p_in_model = ray.p + ray.v * intersection->t;

            best = Intersection{
                .t = intersection->t,
                .p = tris.modelToWorld.applyToPoint(p_in_model),
                .uv = Vec2f{0,0},
                .n = n,
                .mat = tris.material,
                .object = &tris,
                .triangle = TriangleHitData{
                    .bari = intersection->bari,
                    .area = triangleArea(world_triangle.a, world_triangle.b, world_triangle.c),
                    .ccw = is_ccw,
                },
            };
        }
    }

    return best;
}

struct BoxInterval
{
    float enter_time;
    float exit_time;
};

HD BoxInterval findBoxInterval(const float min, const float max, const float p, const float v)
{
    const auto t_min = (min - p) / v;
    const auto t_max = (max - p) / v;

    return BoxInterval{
        .enter_time = std::min(t_min, t_max),
        .exit_time = std::max(t_min, t_max),
    };
}

HD std::optional<float> testBoxIntersection(const AABB &box, const Ray &ray)
{
    const auto interval_x = findBoxInterval(box.min.x, box.max.x, ray.p.x, ray.v.x);
    const auto interval_y = findBoxInterval(box.min.y, box.max.y, ray.p.y, ray.v.y);
    const auto interval_z = findBoxInterval(box.min.z, box.max.z, ray.p.z, ray.v.z);

    const auto enter_time = std::max(std::max(interval_x.enter_time, interval_y.enter_time), interval_z.enter_time);
    const auto exit_time = std::min(std::min(interval_x.exit_time, interval_y.exit_time), interval_z.exit_time);

    if (exit_time < 0 || enter_time > exit_time) {
        return std::nullopt;
    }

    return std::max(enter_time, 0.0f);
}

template<Benchmark B>
HD std::optional<Intersection> getIntersectionImpl(
    const Ray &ray_in_world,
    const TriangleMesh &tris,
    B &benchmark
){
    std::optional<Intersection> best = std::nullopt;
    
    const auto &bbh = tris.bbh;
    const auto ray = tris.modelToWorld.applyInverse(ray_in_world);

    // getIntersectionTris(ray, tris, 0, tris.tris_count, best, benchmark);
    // return best;
    
    int bbox_index = 0;
    while (bbox_index < bbh.nodes.size())
    {
        const auto &node = bbh.nodes[bbox_index];
        benchmark.registerBBoxTest();
        const auto bboxHit = testBoxIntersection(node.box, ray);

        if (bboxHit.has_value())
        {
            if (node.isLeaf())
            {
                getIntersectionTris(ray, tris, node.tris_begin, node.tris_end, best, benchmark);
            }

            ++bbox_index;
        }
        else
        {
            bbox_index = node.skip_index;
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
