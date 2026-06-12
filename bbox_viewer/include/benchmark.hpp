#pragma once

#include <tracing/Intersection.hpp>
#include <tracing/DistributionSamplers.hpp>
#include <tracing/TriangleMesh.hpp>
#include <tracing/GpuTris.hpp>
#include <device/Random.hpp>

#include <curand.h>
#include <curand_kernel.h>

void benchmarkRayCast(
    CudaRandomStates &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
);

struct BenchmarkGenerator
{
    static BenchmarkGenerator create(int ray_count, const Mesh &mesh);

    BenchmarkGenerator() = default;

    BenchmarkGenerator(const BenchmarkGenerator &) = delete;
    BenchmarkGenerator &operator=(const BenchmarkGenerator &) = delete;

    BenchmarkGenerator(BenchmarkGenerator &&) = default;
    BenchmarkGenerator &operator=(BenchmarkGenerator &&) = default;

    void step();
    benchmark::HitTests aggregateResults() const;

    std::unique_ptr<CudaRandomStates> randStates;
    benchmark::HitTests *stats;

    GpuTris tris;
    Vec3 center;
    float radius;
    int ray_count;
};
