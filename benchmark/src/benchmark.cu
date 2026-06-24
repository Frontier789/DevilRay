#include "benchmark.hpp"

#include <tracing/IntersectionTestsImpl.hpp>
#include <device/DevUtils.hpp>

struct CudaRandom
{
    curandState *state;

    __device__ float rnd()
    {
        return curand_uniform(state);
    }
};

__global__ void runRaycasts(
    curandState *randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh tris,
    Vec3 center, float radius
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRandom{randStates + idx};

    const auto dir0 = uniformSphereSample(rng);
    const auto dir1 = uniformSphereSample(rng);

    const auto r = radius * 1.1f;
    const auto p0 = dir0 * r + center;
    const auto p1 = (dir1 - dir0) * r + center;

    const auto ray = Ray{.p = p0, .v = (p1 - p0).normalized()};

    const auto intersection = getIntersectionBenchmark(ray, tris, stats[idx]);

    if (intersection.has_value())
    {
        stats[idx].registerTriangleHit();
    }
}

void benchmarkRayCast(
    CudaRandomStates &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
)
{
    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(randStates.devicePtr(), stats, ray_count, tris, center, radius);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();
}