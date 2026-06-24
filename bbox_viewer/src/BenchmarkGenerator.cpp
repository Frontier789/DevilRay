#include "benchmark.hpp"

BenchmarkGenerator BenchmarkGenerator::create(int ray_count, Mesh &mesh)
{
    auto randStates = CudaRandomStates(Size2i{.width = ray_count, .height = 1});
    auto tris = GpuTris{convertMeshToTris(mesh, false)};

    auto stats = DeviceArray<benchmark::HitTests>(ray_count, benchmark::HitTests{});
    auto gpu_tris = viewGpuTris(tris);
    const auto bounds = calculateMeshBounds(mesh);

    stats.ensureDeviceAllocation();

    return BenchmarkGenerator{
        .randStates = std::move(randStates),
        .stats = std::move(stats),
        .tris = std::move(tris),
        .gpu_tris = std::move(gpu_tris),
        .center = bounds.center,
        .radius = bounds.extent * 1.1f,
        .ray_count = ray_count,
    };
}

void BenchmarkGenerator::step()
{
    benchmarkRayCast(
        randStates, stats.devicePtr(), ray_count,
        gpu_tris, center, radius
    );
}

benchmark::HitTests BenchmarkGenerator::aggregateResults() const
{
    stats.updateHostData();
    
    benchmark::HitTests summed{};
    for (const auto &test : stats.hostSpan())
    {
        summed.triangle_tests += test.triangle_tests;
        summed.triangle_hits += test.triangle_hits;
        summed.bbox_tests += test.bbox_tests;
    }

    return summed;
}
