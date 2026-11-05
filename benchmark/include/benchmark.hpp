#pragma once

#include "Utils.hpp"
#include "device/Array.hpp"
#include "device/Random.hpp"
#include "models/Mesh.hpp"
#include "tracing/Benchmark.hpp"
#include "tracing/TriangleMesh.hpp"

void benchmarkRayCast(
    CudaRandom &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMeshView &tris, Vec3 center, float radius
);

struct BenchmarkGenerator
{
    static BenchmarkGenerator create(int ray_count, Mesh &mesh);

    void step();
    benchmark::HitTests aggregateResults() const;

    CudaRandom randStates;
    mutable DeviceArray<benchmark::HitTests> stats;

    TriangleMesh tris;
    TriangleMeshView gpu_tris;

    Vec3 center;
    float radius;
    int ray_count;
};
