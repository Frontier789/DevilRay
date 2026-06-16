#include "benchmark.hpp"

#include <tracing/IntersectionTestsImpl.hpp>

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

    const auto pn = uniformSphereSample(rng);
    const auto p = pn * radius + center;
    const auto v = uniformHemisphereSample(pn * -1, rng);

    const auto ray = Ray{.p = p, .v = v};

    getIntersectionBenchmark(ray, tris, stats[idx]);
}

void benchmarkRayCast(
    CudaRandomStates &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
)
{
    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(randStates.ptr(), stats, ray_count, tris, center, radius);
}