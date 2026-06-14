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

struct BoxInterval
{
    float enter_time;
    float exit_time;
};

HD BoxInterval findBoxInterval(const float min, const float max, const float p, const float v)
{
    const auto t_min = (min - p) / v;
    const auto t_max = (max - p) / v;

    return BoxInterval{
        .enter_time = std::min(t_min, t_max),
        .exit_time = std::max(t_min, t_max),
    };
}

HD std::optional<float> testBoxIntersection(const AABB &box, const Ray &ray)
{
    const auto interval_x = findBoxInterval(box.min.x, box.max.x, ray.p.x, ray.v.x);
    const auto interval_y = findBoxInterval(box.min.y, box.max.y, ray.p.y, ray.v.y);
    const auto interval_z = findBoxInterval(box.min.z, box.max.z, ray.p.z, ray.v.z);

    const auto enter_time = std::max(std::max(interval_x.enter_time, interval_y.enter_time), interval_z.enter_time);
    const auto exit_time = std::min(std::min(interval_x.exit_time, interval_y.exit_time), interval_z.exit_time);

    if (exit_time < 0 || enter_time > exit_time) {
        return std::nullopt;
    }

    return std::max(enter_time, 0.0f);
}

template<Benchmark B>
HD std::optional<float> getIntersectionTEST(
    const Ray &ray,
    const TriangleMesh &tris,
    const BBHGpuView bbh, B &benchmark
){
    benchmark.registerBBoxTest();
    const auto bboxHit = testBoxIntersection(bbh.nodes[0].box, ray);

    if (!bboxHit.has_value()) return std::nullopt;

    std::optional<float> best_t = std::nullopt;

    for (int i=0;i<tris.tris_count;++i)
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