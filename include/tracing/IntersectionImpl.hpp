#pragma once

#include "tracing/TriangleMesh.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/Benchmark.hpp"

#include "Scene.hpp"

#include <optional>
#include <algorithm>


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

    if (exit_time < 0 || enter_time > exit_time || std::isnan(enter_time) || std::isnan(exit_time)) {
        return std::nullopt;
    }

    return std::max(enter_time, 0.0f);
}


HD TriangleHit intersectTriangle(const Ray &ray_model, const TriangleVertices &triangle)
{
    const auto &ray = ray_model;

    if (ray.p.anyNan() || ray.v.anyNan()) return TriangleHit::missed();

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);
    const auto sgn_area2 = n_f.dot(n_f);

    if (sgn_area2 < 1e-14f) return TriangleHit::missed();

    auto n = n_f.normalized();

    // Face the normal toward the ray origin so the test is two-sided.
    auto dp = (ray.p - A).dot(n);
    if (dp < 0) {
        n = n * -1;
        dp = -dp;
    }

    const auto d = -ray.v.dot(n);
    if (d < 1e-7f) return TriangleHit::missed();

    const float t = dp / d;

    // Barycentric containment: reject hits outside the triangle.
    const Vec3 p = ray.p + ray.v * t;

    const auto n_1 = (p - A).cross(C - A);
    const auto n_2 = (B - A).cross(p - A);

    const auto w_B = n_f.dot(n_1) / sgn_area2;
    const auto w_C = n_f.dot(n_2) / sgn_area2;
    const auto w_A = 1 - w_B - w_C;

    if (w_A < 0 || w_B < 0 || w_C < 0) return TriangleHit::missed();

    return TriangleHit{
        .t = t
    };
}

constexpr bool rayPointsToRight(const Ray& ray, uint32_t depth)
{
    uint32_t axis = depth % 3;

    if (axis == 0) return ray.v.x > 0.0f;
    if (axis == 1) return ray.v.y > 0.0f;
    return ray.v.z > 0.0f;
}

constexpr int nearChild(const Ray& ray, uint32_t depth, int current_node, const BBHNode &node)
{
    if (rayPointsToRight(ray, depth))
    {
        const auto left_child = current_node + 1;
        return left_child;
    }

    return node.right_child;
}

constexpr int farChild(const Ray& ray, uint32_t depth, int current_node, const BBHNode &node)
{
    if (rayPointsToRight(ray, depth))
    {
        return node.right_child;
    }

    const auto left_child = current_node + 1;
    return left_child;
}

template<Benchmark B>
HD void intersectTris(
    const Ray &ray_model, const TriangleMeshView &mesh,
    int tris_begin, int tris_end,
    MeshHit &best,
    B &benchmark
){
    for (int i=tris_begin;i<tris_end;++i)
    {
        const auto &indices = mesh.triangles[i];

        const auto triangle = TriangleVertices{
            .a = mesh.points[indices.a.pi],
            .b = mesh.points[indices.b.pi],
            .c = mesh.points[indices.c.pi],
        };

        benchmark.registerTriangleTest();
        const auto intersection = intersectTriangle(ray_model, triangle);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = MeshHit{
            .t = intersection.t,
            .triangleID = i,
        };
    }
}

template<Benchmark B>
HD MeshHit intersectMeshImpl(const Ray &ray_world, const TriangleMeshView &mesh, B &benchmark)
{
    auto best = MeshHit::missed();
    
    const auto &bbh = mesh.bbh;
    const auto ray = mesh.model_to_world.applyInverse(ray_world);

    // intersectTris(ray, tris, 0, tris.triangle_count, best, benchmark);
    // return best;

    uint32_t bit_trail = 0;
    uint32_t depth = 0;
    int current_index = 0;

    while (current_index >= 0 && current_index < bbh.nodes.size())
    {
        const auto &node = bbh.nodes[current_index];
        benchmark.registerBBoxTest();
        const auto bboxHit = testBoxIntersection(node.box, ray);

        if (bboxHit.has_value())
        {
            const auto t = *bboxHit;

            if (!best.valid() || t < best.t)
            {
                if (node.isLeaf())
                {
                    intersectTris(ray, mesh, node.tris_begin, node.tris_end, best, benchmark);
                }
                else
                {
                    const auto near_child = nearChild(ray, depth, current_index, node);
                    
                    bit_trail &= ~(1u << depth);

                    current_index = near_child;
                    depth++;

                    continue;
                }
            }
        }

        while (true)
        {
            if (current_index == 0) {
                current_index = -1;
                break;
            }

            const auto parent_index = bbh.nodes[current_index].parent_index;
            const auto &parent = bbh.nodes[parent_index];

            depth--;
            const auto far_child_visited = bit_trail & (1u << depth);

            if (!far_child_visited)
            {
                const auto far_child = farChild(ray, depth, parent_index, parent);

                bit_trail |= (1u << depth);

                current_index = far_child;
                depth++;
                break;
            }
            else
            {
                current_index = parent_index;
            }
        }
    }

    return best;
}

HD MeshHit intersectMesh(const Ray &ray_world, const TriangleMeshView &mesh)
{
    benchmark::Skip skip_benchmarks;
    return intersectMeshImpl(ray_world, mesh, skip_benchmarks);
}

template<Benchmark B>
HD SceneHit intersectSceneImpl(const Ray &ray_world, const std::span<const TriangleMeshView> &meshes, B &benchmark)
{
    auto best = SceneHit::missed();

    for (int i=0; i<meshes.size(); ++i)
    {
        auto intersection = intersectMeshImpl(ray_world, meshes[i], benchmark);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = SceneHit {
            .t = intersection.t,
            .triangleID = intersection.triangleID,
            .meshID = i,
        };
    }

    return best;
}

HD SceneHit intersectScene(const Ray &ray_world, const std::span<const TriangleMeshView> &meshes)
{
    benchmark::Skip skip_benchmarks;
    return intersectSceneImpl(ray_world, meshes, skip_benchmarks);
}

HD SceneHit intersectSceneBenchmark(const Ray &ray_world, const std::span<const TriangleMeshView> &meshes, benchmark::HitTests &benchmark)
{
    return intersectSceneImpl(ray_world, meshes, benchmark);
}

template<Benchmark B>
HD bool occludedSceneImpl(Vec3 p0, Vec3 p1, std::span<const TriangleMeshView> objects, B &benchmark)
{
    const auto distance = (p1 - p0).length();
    const auto v = (p1 - p0) / distance;

    Ray ray{.p = p0 + v * 1e-5, .v = v};

    const auto hit = intersectSceneImpl(ray, objects, benchmark);
    if (!hit.valid()) return false;

    return hit.t < distance - 1e-5 * 2;
}

HD bool occludedScene(Vec3 p0, Vec3 p1, std::span<const TriangleMeshView> objects)
{
    benchmark::Skip skip_benchmarks;
    return occludedSceneImpl(p0, p1, objects, skip_benchmarks);
}

HD bool occludedSceneBenchmark(Vec3 p0, Vec3 p1, std::span<const TriangleMeshView> objects, benchmark::HitTests &benchmark)
{
    return occludedSceneImpl(p0, p1, objects, benchmark);
}
