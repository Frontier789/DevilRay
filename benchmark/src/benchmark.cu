#include "benchmark.hpp"

#include "tracing/IntersectionTestsImpl.hpp"
#include "device/DevUtils.hpp"
#include <iostream>

struct CudaRandom
{
    curandState *state;

    __device__ float rnd()
    {
        return curand_uniform(state);
    }
};

HD Ray generateRay(Vec3 center, float radius, CudaRandom &rng)
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
    curandState *randStates, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh tris,
    Vec3 center, float radius
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRandom{randStates + idx};

    const auto ray = generateRay(center, radius, rng);

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct IntersectionTime
{
    float t;

    HD static constexpr IntersectionTime missed() { return IntersectionTime{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};

struct IntersectionTriangle
{
    float t;
    int triangleID;

    HD static constexpr IntersectionTriangle missed() { return IntersectionTriangle{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};

struct IntersectionMesh
{
    float t;
    int triangleID;
    int meshID;

    HD static constexpr IntersectionMesh missed() { return IntersectionMesh{.t = -1}; }

    HD constexpr bool valid() const { return t >= 0; }
};


HD IntersectionTime testTriangleIntersection_dev(const Ray &ray, const TriangleVertices &triangle)
{
    if (ray.p.anyNan() || ray.v.anyNan()) return IntersectionTime::missed();

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);

    auto dp = (A - ray.p).dot(n_f);
    const auto d = ray.v.dot(n_f);

    if (d < 1e-7f) return IntersectionTime::missed();

    const float t = dp / d;

    return IntersectionTime{
        .t = t
    };
}





template<Benchmark B>
HD void getIntersectionTris_dev(
    const Ray &ray, const TriangleMesh &tris,
    int tris_begin, int tris_end,
    IntersectionTriangle &best,
    B &benchmark
){
    for (int i=tris_begin;i<tris_end;++i)
    {
        const auto &indices = tris.triangles[i];

        const auto triangle = TriangleVertices{
            .a = tris.points[indices.a.pi],
            .b = tris.points[indices.b.pi],
            .c = tris.points[indices.c.pi],
        };

        benchmark.registerTriangleTest();
        const auto intersection = testTriangleIntersection_dev(ray, triangle);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = IntersectionTriangle{
            .t = intersection.t,
            .triangleID = i,
        };
    }
}








template<Benchmark B>
HD IntersectionTriangle getIntersectionImpl_dev(
    const Ray &ray_in_world,
    const TriangleMesh &tris,
    B &benchmark
){
    auto best = IntersectionTriangle::missed();
    
    const auto &bbh = tris.bbh;
    const auto ray = tris.model_to_world.applyInverse(ray_in_world);

    // getIntersectionTris(ray, tris, 0, tris.triangle_count, best, benchmark);
    // return best;

    uint32_t bit_trail = 0;
    uint32_t depth = 0;
    int current_index = 0;

    while (current_index >= 0 && current_index < bbh.nodes.size())
    {
        const auto &node = bbh.nodes[current_index];
        benchmark.registerBBoxTest();
        const auto bboxHit = testBoxIntersection(node.box, ray);

        if (bboxHit.has_value())
        {
            const auto t = *bboxHit;

            if (!best.valid() || t < best.t)
            {
                if (node.isLeaf())
                {
                    getIntersectionTris_dev(ray, tris, node.tris_begin, node.tris_end, best, benchmark);
                }
                else
                {
                    const auto near_child = nearChild(ray, depth, current_index, node);
                    
                    bit_trail &= ~(1u << depth);

                    current_index = near_child;
                    depth++;

                    continue;
                }
            }
        }

        while (true)
        {
            if (current_index == 0) {
                current_index = -1;
                break;
            }

            const auto parent_index = bbh.nodes[current_index].parent_index;
            const auto &parent = bbh.nodes[parent_index];

            depth--;
            const auto far_child_visited = bit_trail & (1u << depth);

            if (!far_child_visited)
            {
                const auto far_child = farChild(ray, depth, parent_index, parent);

                bit_trail |= (1u << depth);

                current_index = far_child;
                depth++;
                break;
            }
            else
            {
                current_index = parent_index;
            }
        }
    }

    return best;
}


HD auto getIntersection_dev(const Ray &ray, const TriangleMesh &tris)
{
    benchmark::Skip skip_benchmarks;
    return getIntersectionImpl_dev(ray, tris, skip_benchmarks);
}



HD IntersectionMesh cast_dev(const Ray &ray, const std::span<const TriangleMesh> objects, const ObjectsInfo &info)
{
    auto best = IntersectionMesh::missed();

    for (int i=0; i<objects.size(); ++i)
    {
        auto intersection = getIntersection_dev(ray, objects[i]);

        if (!intersection.valid()) continue;
        if (best.valid() && best.t <= intersection.t) continue;

        best = IntersectionMesh {
            .t = intersection.t,
            .triangleID = intersection.triangleID,
            .meshID = i,
        };
    }

    return best;
}






struct WavefrontData
{
    DeviceBuffer<Ray> rays;
};

struct WavefrontDataDevice
{
    std::span<Ray> rays;
};

struct VertexData
{
    DeviceBuffer<Vec4> pos;
};

struct VertexDataDevice
{
    std::span<Vec4> pos;
};

__global__ void initWavefront(
    WavefrontDataDevice wavefront, curandState *randStates,
    Vec3 center, float radius, int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRandom{randStates + idx};

    const auto ray = generateRay(center, radius, rng);

    wavefront.rays[idx] = ray;
}

__global__ void findNextVertex(
    WavefrontDataDevice wavefront,
    VertexDataDevice nextVertex,
    TriangleMesh gpuTris,
    int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ray = wavefront.rays[idx];
    const auto intersection = cast_dev(ray, std::span<TriangleMesh>{&gpuTris, 1}, ObjectsInfo{.total_radiant_power = 1});

    if (intersection.valid()) {
        const auto p = ray.p + ray.v * intersection.t;
        nextVertex.pos[idx] = Vec4::from(p, 1);
    } else {
        nextVertex.pos[idx] = Vec4{0,0,0, -1};
    }
}

void testWavefront(Mesh &mesh)
{
    constexpr int RAY_COUNT = 1'000'000;

    auto tris = GpuTris{convertMeshToTris(mesh, false)};
    auto gpuTris = viewGpuTris(tris);
    const auto bounds = calculateMeshBounds(mesh);


    CudaRandomStates states(Size2i{.width = RAY_COUNT, .height = 1});
    WavefrontData wavefront{.rays = DeviceBuffer<Ray>::allocate(RAY_COUNT)};
    VertexData vertexData{.pos = DeviceBuffer<Vec4>::allocate(RAY_COUNT)};

    WavefrontDataDevice wavefrontView{.rays = wavefront.rays.deviceSpan()};
    VertexDataDevice vertexDataView{.pos = vertexData.pos.deviceSpan()};

    dim3 dimBlock(128, 1);
    dim3 dimGrid((RAY_COUNT + dimBlock.x - 1) / dimBlock.x, 1);

    const auto runCudaCalls = [&]{
        initWavefront<<<dimGrid, dimBlock>>>(wavefrontView, states.devicePtr(), bounds.center, bounds.extent * 0.5f, RAY_COUNT);
        findNextVertex<<<dimGrid, dimBlock>>>(wavefrontView, vertexDataView, gpuTris, RAY_COUNT);
        cudaDeviceSynchronize();
        CUDA_ERROR_CHECK();
    };

    runCudaCalls();

    Timer t;
    runCudaCalls();

    std::cout << "Wavefront init: " << t.elapsedSeconds()*1000 << "ms" << std::endl;
}
