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

} // namespace

int main(int argc, char **argv)
{
    const std::string meshPath = (argc > 1) ? argv[1] : "models/suzanne.obj";
    constexpr int rayCount = 10'000'000;

    printCudaDeviceInfo();

    std::cout << "Loading mesh: " << meshPath << std::endl;
    const Mesh mesh = loadMesh(meshPath);
    std::cout << "Triangles: " << mesh.triangles.size()
              << ", points: "  << mesh.points.size() << std::endl;

    GpuTris gpuTris = convertMeshToTris(mesh);
    TrisCollection tris = viewGpuTris(gpuTris);

    // Bounding box (host-side) for ray generation
    Vec3 bbMin{ std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};
    Vec3 bbMax{-std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity()};
    for (const Vec3 &p : mesh.points)
    {
        bbMin = Vec3{std::min(bbMin.x, p.x), std::min(bbMin.y, p.y), std::min(bbMin.z, p.z)};
        bbMax = Vec3{std::max(bbMax.x, p.x), std::max(bbMax.y, p.y), std::max(bbMax.z, p.z)};
    }
    const Vec3 center = (bbMin + bbMax) * 0.5f;
    const float radius = (bbMax - bbMin).length() * 1.5f + 1.0f;

    // Generate rays on host: origins on a sphere around the mesh, aimed at a
    // random point inside the bbox. This yields a healthy mix of hits/misses.
    std::cout << "Generating " << rayCount << " random rays..." << std::endl;

    std::vector<Ray> hostRays;
    hostRays.reserve(rayCount);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (int i = 0; i < rayCount; ++i)
    {
        const float z = 1.0f - 2.0f * u01(rng);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = 2.0f * pi * u01(rng);
        const Vec3 dir{r * std::cos(phi), r * std::sin(phi), z};

        const Vec3 origin = center + dir * radius;

        const Vec3 target = Vec3{
            bbMin.x + (bbMax.x - bbMin.x) * u01(rng),
            bbMin.y + (bbMax.y - bbMin.y) * u01(rng),
            bbMin.z + (bbMax.z - bbMin.z) * u01(rng),
        } * 1.0f;

        const Vec3 v = (target - origin).normalized();
        hostRays.push_back(Ray{origin, v});
    }

    // Upload rays + allocate result buffers
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

    // Pull back results to compute hit stats
    std::vector<int> hostHit(rayCount);
    std::vector<float> hostT(rayCount);
    cudaMemcpy(hostHit.data(), dHit, sizeof(int) * rayCount,   cudaMemcpyDeviceToHost);
    cudaMemcpy(hostT.data(),   dT,   sizeof(float) * rayCount, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK();

    long long hitCount = 0;
    double accumT = 0.0;
    for (int i = 0; i < rayCount; ++i)
    {
        if (hostHit[i])
        {
            ++hitCount;
            accumT += hostT[i];
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
    std::cout << "Avg hit t:            "
              << (hitCount > 0 ? accumT / hitCount : 0.0) << "\n";
    std::cout << "GPU kernel time:      " << ms << " ms\n";
    std::cout << "Rays / second:        " << (rayCount / seconds) << "\n";
    std::cout << "Tri tests / second:   " << (isectTests / seconds) << "\n";
    std::cout << "ns / ray:             " << (seconds * 1e9 / rayCount) << "\n";
    std::cout << "ns / tri test:        " << (seconds * 1e9 / isectTests) << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dRays);
    cudaFree(dT);
    cudaFree(dHit);
}
