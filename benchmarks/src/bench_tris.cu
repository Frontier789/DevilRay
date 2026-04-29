#include "models/Mesh.hpp"
#include "tracing/GpuTris.hpp"
#include "tracing/Intersection.hpp"
#include "device/DevUtils.hpp"
#include "Utils.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <filesystem>
#include <tracing/DistributionSamplers.hpp>

namespace
{

HD std::optional<TriangleIntersection> testTriangleIntersectionImpl(const Ray &ray, const TriangleVertices &triangle)
{
    if (ray.p.anyNan() || ray.v.anyNan()) return std::nullopt;

    const auto A = triangle.a;
    const auto B = triangle.b;
    const auto C = triangle.c;

    const auto n_f = (A - B).cross(A - C);
    const auto sgn_area2 = n_f.dot(n_f);

    if (sgn_area2 < 1e-14f) return std::nullopt;

    auto n = n_f.normalized();

    auto dp = dot(ray.p - A, n);

    if (dp < 0) {
        n = n * -1;
        dp = -dp;
    }

    // (ray.p - o) . n + ray.v . n * t = 0

    const auto d = -dot(ray.v, n);
    if (d < 1e-7f) return std::nullopt;

    const float t = dp / d;

    const Vec3 p = ray.p + ray.v * t;

    const auto n_1 = (p - A).cross(C - A);
    const auto n_2 = (B - A).cross(p - A);

    const auto w_B = n_f.dot(n_1) / sgn_area2;
    const auto w_C = n_f.dot(n_2) / sgn_area2;
    const auto w_A = 1 - w_B - w_C;

    if (w_A < 0 || w_B < 0 || w_C < 0) return std::nullopt;

    return TriangleIntersection{
        .t = t,
        .bari = Vec3{w_A, w_B, w_C},
    };
}

__global__ void traceKernel(
    const Ray *rays,
    int rayCount,
    const Vec3 *points,
    const Triangle *triangles,
    int triCount,
    Vec3 scale,
    Vec3 position,
    float *outT,
    int *outHitFlag)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rayCount) return;

    const Ray ray = rays[idx];

    float bestT = INFINITY;
    int hit = 0;

    for (int i = 0; i < triCount; ++i)
    {
        const Triangle indices = triangles[i];

        const TriangleVertices tri{
            .a = points[indices.a.pi] * scale + position,
            .b = points[indices.b.pi] * scale + position,
            .c = points[indices.c.pi] * scale + position,
        };

        const auto isect = testTriangleIntersectionImpl(ray, tri);
        const auto t = isect->t;
        if (isect.has_value() && t < bestT)
        {
            bestT = isect->t;
            hit = 1;
        }
    }

    outT[idx] = bestT;
    outHitFlag[idx] = hit;
}

struct Box
{
    Vec3 min;
    Vec3 max;

    constexpr Box extend(const Vec3 &p) const {
        return Box {
            .min = Vec3{std::min(min.x, p.x), std::min(min.y, p.y), std::min(min.z, p.z)},
            .max = Vec3{std::max(max.x, p.x), std::max(max.y, p.y), std::max(max.z, p.z)},
        };
    }

    constexpr Vec3 center() const {
        return (min + max) * 0.5f;
    }

    static constexpr Box empty() {
        constexpr auto inf = std::numeric_limits<float>::infinity();
        return Box{
            .min = Vec3{inf, inf, inf},
            .max = Vec3{-inf, -inf, -inf},
        };
    }

    constexpr float diagonal() const {
        return (max - min).length();
    }
};

#pragma nv_exec_check_disable
template<typename Rng>
HD Vec3 uniformBoxSample(const Box &box, Rng &r)
{
    const auto x = r.rnd();
    const auto y = r.rnd();
    const auto z = r.rnd();

    return Vec3{
        box.min.x * (1 - x) + box.max.x * x,
        box.min.y * (1 - y) + box.max.y * y,
        box.min.z * (1 - z) + box.max.z * z,
    };
}

} // namespace


class Benchmark
{
    GpuTris gpuTris;
    TrisCollection tris;
    Box entire_bbox;

    void calculateBBox(const Mesh &mesh)
    {
        entire_bbox = Box::empty();

        for (const Vec3 &p : mesh.points)
        {
            entire_bbox = entire_bbox.extend(p);
        }
    }

public:
    Benchmark(const std::filesystem::path &objectPath)
    {
        if (!std::filesystem::exists(objectPath)) {
            throw std::runtime_error("Object file not found: " + objectPath.string());
        }

        std::cout << "Loading mesh: " << objectPath << std::endl;

        const Mesh mesh = loadMesh(objectPath);
        gpuTris = convertMeshToTris(mesh);
        tris = viewGpuTris(gpuTris);

        std::cout << "Triangles: " << tris.tris_count << std::endl;

        calculateBBox(mesh);
    }

    const TrisCollection &getTris() const { return tris; }
    const Box &getBBox() const { return entire_bbox; }
};

struct Rng
{
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> u01{0, 1};

    float rnd()
    {
        return u01(rng);
    }
};

std::vector<Ray> generateRays(const Box &bounds, int rayCount)
{
    std::cout << "Generating " << rayCount << " random rays..." << std::endl;

    const Vec3 center = bounds.center();
    const float radius = bounds.diagonal() * 1.5f + 1.0f;

    std::vector<Ray> rays;
    rays.reserve(rayCount);

    Rng rng;

    for (int i = 0; i < rayCount; ++i)
    {
        const Vec3 dir = uniformSphereSample(rng);
        const Vec3 origin = center + dir * radius;

        const Vec3 target = uniformBoxSample(bounds, rng);

        const Vec3 v = (target - origin).normalized();
        rays.push_back(Ray{origin, v});
    }

    return rays;
}

int main(int argc, char **argv)
{
    const std::string meshPath = (argc > 1) ? argv[1] : "models/suzanne.obj";
    constexpr int rayCount = 10'000'000;

    printCudaDeviceInfo();

    Benchmark benchmark(meshPath);

    const auto &tris = benchmark.getTris();
    const auto hostRays = generateRays(benchmark.getBBox(), rayCount);

    Ray *dRays = nullptr;
    float *dT = nullptr;
    int *dHit = nullptr;

    cudaMalloc(&dRays, sizeof(Ray) * rayCount);
    cudaMalloc(&dT,    sizeof(float) * rayCount);
    cudaMalloc(&dHit,  sizeof(int) * rayCount);
    CUDA_ERROR_CHECK();

    cudaMemcpy(dRays, hostRays.data(), sizeof(Ray) * rayCount, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK();

    constexpr int blockSize = 128;
    const int gridSize = (rayCount + blockSize - 1) / blockSize;

    // Warm up
    traceKernel<<<gridSize, blockSize>>>(
        dRays, rayCount,
        tris.points, tris.triangles, tris.tris_count,
        tris.s, tris.p,
        dT, dHit);
    CUDA_ERROR_CHECK();
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();

    // Benchmark
    std::cout << "Tracing on GPU..." << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    traceKernel<<<gridSize, blockSize>>>(
        dRays, rayCount,
        tris.points, tris.triangles, tris.tris_count,
        tris.s, tris.p,
        dT, dHit);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_ERROR_CHECK();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<int> hostHit(rayCount);
    std::vector<float> hostT(rayCount);
    cudaMemcpy(hostHit.data(), dHit, sizeof(int) * rayCount,   cudaMemcpyDeviceToHost);
    cudaMemcpy(hostT.data(),   dT,   sizeof(float) * rayCount, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();

    long long hitCount = 0;
    for (int i = 0; i < rayCount; ++i)
    {
        if (hostHit[i])
        {
            ++hitCount;
        }
    }

    const double seconds = ms / 1000.0;
    const long long isectTests = static_cast<long long>(rayCount)
                               * static_cast<long long>(tris.tris_count);

    std::cout << "----------------------------------------\n";
    std::cout << "Rays:                 " << rayCount << "\n";
    std::cout << "Triangles:            " << tris.tris_count << "\n";
    std::cout << "Intersection tests:   " << isectTests << "\n";
    std::cout << "Hits:                 " << hitCount
              << "  (" << (100.0 * hitCount / rayCount) << "%)\n";
    std::cout << "GPU kernel time:      " << ms << " ms\n";
    std::cout << "Rays / second:        " << (rayCount / seconds) << "\n";
    std::cout << "Tri tests / second:   " << (isectTests / seconds) << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dRays);
    cudaFree(dT);
    cudaFree(dHit);
}
