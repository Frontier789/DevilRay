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



template<Benchmark B>
HD std::optional<float> getIntersectionTEST(
    const Ray &ray,
    const TriangleMesh &tris,
    const BBHGpuView bbh, B &benchmark
){
    std::optional<float> best_t = std::nullopt;
    int bbox_index = 0;

    while (bbox_index < bbh.nodes.size())
    {
        const auto &node = bbh.nodes[bbox_index];
        benchmark.registerBBoxTest();
        const auto bboxHit = testBoxIntersection(node.box, ray);

        if (bboxHit.has_value())
        {
            if (node.isLeaf())
            {
                for (int i=node.tris_begin;i<node.tris_end;++i)
                {
                    const auto &indices = tris.triangles[i];

                    const auto triangle = TriangleVertices{
                        .a = tris.points[indices.a.pi] * tris.s + tris.p,
                        .b = tris.points[indices.b.pi] * tris.s + tris.p,
                        .c = tris.points[indices.c.pi] * tris.s + tris.p,
                    };

                    benchmark.registerTriangleTest();
                    const auto intersection = testTriangleIntersection(ray, triangle);

                    if (!intersection.has_value()) continue;

                    if (!best_t.has_value() || best_t > intersection->t)
                    {
                        best_t = intersection->t;
                    }
                }
            }

            ++bbox_index;
        }
        else
        {
            bbox_index = node.skip_index;
        }
    }

    return best_t;
}

__global__ void runRaycasts(
    curandState *randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh tris, const BBHGpuView bbh,
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

    getIntersectionTEST(ray, tris, std::move(bbh), stats[idx]);
}

void benchmarkRayCast(
    CudaRandomStates &randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, BBH &bbh, Vec3 center, float radius
)
{
    auto bbh_gpu = createBBHGpuView(bbh);

    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(randStates.ptr(), stats, ray_count, tris, bbh_gpu, center, radius);
}