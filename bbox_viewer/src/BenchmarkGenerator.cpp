#include "benchmark.hpp"

BenchmarkGenerator BenchmarkGenerator::create(int ray_count, const Mesh &mesh)
{
    BenchmarkGenerator gen;

    gen.randStates = std::make_unique<CudaRandomStates>(Size2i{.width = ray_count, .height = 1});
    gen.tris = GpuTris{convertMeshToTris(mesh, false)};
    gen.ray_count = ray_count;
    gen.bbh = generateSimpleBBH(mesh);

    cudaMalloc(&gen.stats, sizeof(*gen.stats) * ray_count);
    cudaMemset(gen.stats, 0, sizeof(*gen.stats) * ray_count);

    const auto bounds = calculateMeshBounds(mesh);

    gen.center = bounds.center;
    gen.radius = bounds.extent * 0.8f;

    return gen;
}

void BenchmarkGenerator::step()
{
    benchmarkRayCast(*randStates, stats, ray_count,
        viewGpuTris(tris), bbh, center, radius
    );
}

benchmark::HitTests BenchmarkGenerator::aggregateResults() const
{
    std::vector<benchmark::HitTests> benches(ray_count);

    cudaMemcpy(benches.data(), stats, sizeof(*stats) * ray_count, cudaMemcpyDeviceToHost);

    benchmark::HitTests summed{};
    for (const auto &test : benches)
    {
        summed.bbox_tests += test.bbox_tests;
        summed.triangle_tests += test.triangle_tests;
    }

    return summed;
}
