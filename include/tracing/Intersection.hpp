#pragma once

#include "Utils.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/Benchmark.hpp"

#include <span>

struct TriangleHit
{
    float t;

    HD static constexpr TriangleHit missed() { return TriangleHit{.t = -1}; }
    HD constexpr bool valid() const { return t >= 0; }
};

struct MeshHit
{
    float t;
    int triangleID;

    HD static constexpr MeshHit missed() { return MeshHit{.t = -1, .triangleID = -1}; }
    HD constexpr bool valid() const { return t >= 0; }
};

struct SceneHit
{
    float t;
    int triangleID;
    int meshID;

    HD static constexpr SceneHit missed() { return SceneHit{.t = -1, .triangleID = -1, .meshID = -1}; }
    HD constexpr bool valid() const { return t >= 0; }
};

struct TriangleVertices
{
    Vec3 a;
    Vec3 b;
    Vec3 c;
};

HD TriangleHit intersectTriangle(const Ray &ray_model, const TriangleVertices &triangle);
HD MeshHit intersectMesh(const Ray &ray_world, const TriangleMesh &mesh);
HD SceneHit intersectScene(const Ray &ray_world, const std::span<const TriangleMesh> &meshes);
HD SceneHit intersectSceneBenchmark(const Ray &ray_world, const std::span<const TriangleMesh> &meshes, benchmark::HitTests &benchmark);

HD bool occludedScene(Vec3 p0, Vec3 p1, std::span<const TriangleMesh> objects);
HD bool occludedSceneBenchmark(Vec3 p0, Vec3 p1, std::span<const TriangleMesh> objects, benchmark::HitTests &benchmark);

