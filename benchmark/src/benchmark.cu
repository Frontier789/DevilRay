#include "benchmark.hpp"

#include "tracing/IntersectionImpl.hpp"
#include "device/CudaRng.hpp"
#include "device/DevUtils.hpp"

HD Ray generateRay(Vec3 center, float radius, CudaRng &rng)
{
    const auto dir0 = uniformSphereSample(rng);
    const auto dir1 = uniformSphereSample(rng);

    const auto r = radius * 1.1f;
    const auto p0 = dir0 * r + center;
    const auto p1 = (dir1 - dir0) * r + center;

    const auto ray = Ray{.p = p0, .v = (p1 - p0).normalized()};

    return ray;
}

__global__ void runRaycasts(
    curandState *rand_states, benchmark::HitTests *stats, int ray_count,
    const TriangleMeshView tris,
    Vec3 center, float radius
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRng{rand_states + idx};

    const auto ray = generateRay(center, radius, rng);

    const std::span<const TriangleMeshView> meshes{&tris, 1};
    const auto intersection = intersectSceneBenchmark(ray, meshes, stats[idx]);

    if (intersection.valid())
    {
        stats[idx].registerTriangleHit();
    }
}

void benchmarkRayCast(
    CudaRandom &rand_states, benchmark::HitTests *stats, int ray_count,
    const TriangleMeshView &tris, Vec3 center, float radius
)
{
    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(rand_states.devicePtr(), stats, ray_count, tris, center, radius);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();
}
