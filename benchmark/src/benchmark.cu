#include "benchmark.hpp"

#include "tracing/IntersectionImpl.hpp"
#include "tracing/LightSampling.hpp"
#include "tracing/ShadingUtils.hpp"
#include "tracing/Wavefront.hpp"
#include "tracing/Path.hpp"
#include "tracing/RenderBuffers.hpp"
#include "models/MeshUtils.hpp"
#include "device/CudaRng.hpp"
#include "device/DevUtils.hpp"

#include <iostream>

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
    const TriangleMesh tris,
    Vec3 center, float radius
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRng{rand_states + idx};

    const auto ray = generateRay(center, radius, rng);

    const std::span<const TriangleMesh> meshes{&tris, 1};
    const auto intersection = intersectSceneBenchmark(ray, meshes, stats[idx]);

    if (intersection.valid())
    {
        stats[idx].registerTriangleHit();
    }
}

void benchmarkRayCast(
    CudaRandom &rand_states, benchmark::HitTests *stats, int ray_count,
    const TriangleMesh &tris, Vec3 center, float radius
)
{
    dim3 dimBlock(32, 1);
    dim3 dimGrid((ray_count + dimBlock.x - 1) / dimBlock.x, 1);

    runRaycasts<<<dimGrid, dimBlock>>>(rand_states.devicePtr(), stats, ray_count, tris, center, radius);
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initWavefront(
    WavefrontDataDevice wavefront,
    PathVertexDataDevice vertex,
    curandState *rand_states,
    Vec3 center, float radius, int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    auto rng = CudaRng{rand_states + idx};

    const auto ray = generateRay(center, radius, rng);

    wavefront.rays[idx] = ray;

    vertex.throughput[idx] = Vec4{1, 1, 1, 0};
    vertex.prevSpecular[idx] = 1;
    vertex.bsdfPdfPrev[idx] = 0;
}

__global__ void findNextVertex(
    WavefrontDataDevice wavefront,
    PathVertexDataDevice nextVertex,
    std::span<const TriangleMesh> objects,
    int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ray = wavefront.rays[idx];
    const auto intersection = intersectScene(ray, objects);

    if (intersection.valid()) {
        nextVertex.t[idx] = intersection.t;
        nextVertex.ids[idx] = TriangleIdentifier{
            .meshID = intersection.meshID,
            .triangleID = intersection.triangleID,
        };
    } else {
        nextVertex.ids[idx] = TriangleIdentifier::invalid();
    }
}

__global__ void sampleBrdfDirection(
    PathVertexDataDevice vertex,
    WavefrontDataDevice wavefront,
    curandState *randStates,
    std::span<const TriangleMesh> objects,
    const std::span<const Material> materials,
    int ray_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ids = vertex.ids[idx];
    if (!ids.valid()) return;

    const auto &object = objects[ids.meshID];
    auto rng = CudaRng{randStates + idx};

    const auto ray = wavefront.rays[idx];
    const auto pos = ray.p + ray.v * vertex.t[idx];
    const auto n = surfaceNormal(object, ids.triangleID, ray.v);

    const auto w_out = cosineWeightedHemisphereSample(n, rng);
    const float bsdf_pdf = cosineWeightedHemisphereDirPdf(w_out, n);

    wavefront.rays[idx] = Ray{
        .p = pos + n * 1e-5f,
        .v = w_out,
    };

    const auto &material = *std::get_if<DiffuseMaterial>(&materials[object.material]);

    vertex.bsdfPdfPrev[idx] = bsdf_pdf;
    vertex.prevSpecular[idx] = 0;
    vertex.throughput[idx] *= material.diffuse_reflectance;
}

__global__ void shade(
    PathVertexDataDevice vertex,
    WavefrontDataDevice wavefront,
    std::span<const Material> materials,
    std::span<const TriangleMesh> objects,
    curandState *rand_states,
    std::span<const AliasEntry> light_table,
    int ray_count,
    ObjectsInfo info,
    RenderBuffersDevice output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ray_count) return;

    const auto ids = vertex.ids[idx];
    if (!ids.valid()) return;

    const auto &object = objects[ids.meshID];
    const auto &material = materials[object.material];
    auto rng = CudaRng{rand_states + idx};

    Vec4 color{0,0,0,0};

    if (const auto *diffuse_material = std::get_if<DiffuseMaterial>(&material))
    {
        const auto &ray = wavefront.rays[idx];
        const auto prevPos = ray.p;
        const auto pos = ray.p + ray.v * vertex.t[idx];
        const auto n = surfaceNormal(object, ids.triangleID, ray.v);

        const auto nee_pdf = computeNeePdf(prevPos, pos, n, material, info);
        color = color + misWeightedEmission(
            diffuse_material->emission, vertex.throughput[idx],
            vertex.prevSpecular[idx], MisPdfs{
                .bsdf_pdf = vertex.bsdfPdfPrev[idx],
                .nee_pdf = nee_pdf,
            }
        );

        LightSample light_sample = samplePointOnLights(objects, light_table, rng);
        const auto *light_material = std::get_if<DiffuseMaterial>(&materials[light_sample.mat]);

        const auto nee_pdf_bsdf = cosineWeightedHemispherePdf(pos, light_sample.p, n);
        const auto nee_pdf_nee = light_sample.pdf * areaToSolidAngle(pos, light_sample.p, light_sample.n);

        const auto Ld_nee = evaluateDirectLighting(
            pos, n, diffuse_material->diffuse_reflectance,
            light_sample, light_material->emission, objects, info);

        if (nee_pdf_nee + nee_pdf_bsdf > 0)
            color = color + Ld_nee * vertex.throughput[idx] * powerHeuristic(nee_pdf_nee, nee_pdf_bsdf);
    }
    else if (const auto *transparent_material = std::get_if<TransparentMaterial>(&material)) {
        // PASS
    }

    output.colors[idx] = output.colors[idx] + color;
}

void testWavefront(Mesh &mesh)
{
    constexpr int RAY_COUNT = 1'000'000;

    auto tris = GpuTris{convertMeshToTris(mesh, true)};
    auto gpuTris = viewGpuTris(tris);
    const auto bounds = calculateMeshBounds(mesh);

    gpuTris.material = 0;

    DiffuseMaterial light_material{};
    light_material.emission = Vec4{1, 1, 1, 0};
    light_material.diffuse_reflectance = Vec4{0.7, 0.7, 0.7, 0};
    DeviceVector<Material> materials{std::vector<Material>{Material{light_material}}};
    materials.ensureDeviceAllocation();

    const auto total_radiant_power =
        gpuTris.surface_area * luminance(radiantExitance(materials.hostPtr()[0]));
    const auto info = ObjectsInfo{.total_radiant_power = total_radiant_power};

    auto light_sampler = generateAliasTable(std::vector<float>{1.0f});
    light_sampler.entries.ensureDeviceAllocation();
    const std::span<const AliasEntry> light_table = light_sampler.entries.deviceSpan();

    // Upload the single object into device memory
    auto objects = DeviceBuffer<TriangleMesh>::allocate(1);
    objects.deviceData.copyFromHost(&gpuTris, sizeof(TriangleMesh));
    const std::span<const TriangleMesh> objectsView = objects.deviceSpan();

    CudaRandom rng(RAY_COUNT);
    WavefrontData wavefront{.rays = DeviceBuffer<Ray>::allocate(RAY_COUNT)};
    PathVertexData pathVertexData{
        .t = DeviceBuffer<float>::allocate(RAY_COUNT),
        .bsdfPdfPrev = DeviceBuffer<float>::allocate(RAY_COUNT),
        .throughput = DeviceBuffer<Vec4>::allocate(RAY_COUNT),
        .prevSpecular = DeviceBuffer<int>::allocate(RAY_COUNT),
        .ids = DeviceBuffer<TriangleIdentifier>::allocate(RAY_COUNT),
    };
    RenderBuffers output{.colors = DeviceBuffer<Vec4>::allocate(RAY_COUNT)};

    WavefrontDataDevice wavefrontView{.rays = wavefront.rays.devicePtr()};
    PathVertexDataDevice pathVertexDataView{
        .t = pathVertexData.t.devicePtr(),
        .bsdfPdfPrev = pathVertexData.bsdfPdfPrev.devicePtr(),
        .throughput = pathVertexData.throughput.devicePtr(),
        .prevSpecular = pathVertexData.prevSpecular.devicePtr(),
        .ids = pathVertexData.ids.devicePtr(),
    };
    RenderBuffersDevice outputView{
        .colors = output.colors.devicePtr(),
        .resolution = Size2i{.width = RAY_COUNT, .height = 1},
    };

    dim3 dimBlock(128, 1);
    dim3 dimGrid((RAY_COUNT + dimBlock.x - 1) / dimBlock.x, 1);

    const auto runCudaCalls = [&]{
        initWavefront<<<dimGrid, dimBlock>>>(wavefrontView, pathVertexDataView, rng.devicePtr(), bounds.center, bounds.extent * 0.5f, RAY_COUNT);
        for (int depth=0;depth<8;++depth) {
            findNextVertex<<<dimGrid, dimBlock>>>(wavefrontView, pathVertexDataView, objectsView, RAY_COUNT);
            shade<<<dimGrid, dimBlock>>>(pathVertexDataView, wavefrontView, materials.deviceSpan(), objectsView, rng.devicePtr(), light_table, RAY_COUNT, info, outputView);
            sampleBrdfDirection<<<dimGrid, dimBlock>>>(pathVertexDataView, wavefrontView, rng.devicePtr(), objectsView, materials.deviceSpan(), RAY_COUNT);
        }
        cudaDeviceSynchronize();
        CUDA_ERROR_CHECK();
    };

    runCudaCalls();

    Timer t;
    runCudaCalls();

    std::cout << "Wavefront init: " << t.elapsedSeconds()*1000 << "ms" << std::endl;
}
