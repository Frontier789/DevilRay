#pragma once

#include "tracing/Intersection.hpp"
#include "tracing/DistributionSamplers.hpp"
#include "tracing/TriangleMesh.hpp"
#include "tracing/GpuTris.hpp"
#include "device/Random.hpp"
#include "models/BBH.hpp"

#include <curand.h>
#include <curand_kernel.h>

void benchmarkRayCast(
    CudaRandomStates &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
);

struct BenchmarkGenerator
{
    static BenchmarkGenerator create(int ray_count, Mesh &mesh);

    void step();
    benchmark::HitTests aggregateResults() const;

    CudaRandomStates randStates;
    mutable DeviceArray<benchmark::HitTests> stats;

    GpuTris tris;
    TriangleMesh gpu_tris;

    Vec3 center;
    float radius;
    int ray_count;
};
