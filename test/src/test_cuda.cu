// AI-generated tests (Claude), reviewed by hand before committing.
//
// Compiled as CUDA (.cu) so it can both (a) call the HD BSDF helpers on the CPU
// and (b) launch the real __global__ kernels on the GPU. The header-defined
// intersection routines are pulled in directly (like the benchmark) so the test
// TU is self-contained and never drags in the renderer translation unit.

#include "TracingTestHelpers.hpp"

#include "tracing/IntersectionImpl.hpp"
#include "tracing/GpuTris.hpp"
#include "tracing/Camera.hpp"
#include "tracing/CameraRay.hpp"
#include "device/Array.hpp"
#include "device/Vector.hpp"
#include "device/Random.hpp"
#include "device/DevUtils.hpp"

#include <cuda_runtime.h>
#include <utility>

// checkerPattern is declared in PathGeneration.hpp and defined in the library's
// implementation.cu. Provide a local definition so debugShade links here without
// pulling in that translation unit (which would duplicate the kernel symbols).
HD Vec4 checkerPattern(const Vec2f &uv, int checker_count, Vec4 dark, Vec4 bright)
{
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;
    const float checker = checker_x ^ checker_y;
    return bright * checker + dark * (1 - checker);
}

#include "tracing/PathGeneration.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <vector>

using test::expectVec3Near;
using test::expectVec4Near;

namespace
{
    // Minimal HD-callable RNG: returns a fixed value. Used to instantiate the HD
    // BSDF/camera helpers on the host without tripping nvcc's host/device
    // execution-space check (the std-based test RNGs are __host__ only).
    struct FixedRng
    {
        float value;
        HD float rnd() { return value; }
    };
}

// ==========================================================================
// Host-side tests for the HD BSDF helpers (run on the CPU).
// ==========================================================================

TEST(ReflectTest, MirrorsRayAcrossNormal)
{
    // Straight-on ray (dotp = i.n = -1) flips to the opposite direction.
    expectVec3Near(reflect(Vec3{0, 0, -1}, Vec3{0, 0, 1}, -1.0f), {0, 0, 1});
}

TEST(ReflectTest, PreservesTangentAndLength)
{
    const Vec3 i = Vec3{1, 0, -1}.normalized();
    const Vec3 n{0, 0, 1};
    const Vec3 r = reflect(i, n, i.dot(n));
    expectVec3Near(r, Vec3{1, 0, 1}.normalized());
    EXPECT_NEAR(r.length(), 1.0f, 1e-5f);
}

TEST(SchlickTest, NormalIncidenceEqualsR0)
{
    const float R0 = ((1.0f - 1.5f) / (1.0f + 1.5f)) * ((1.0f - 1.5f) / (1.0f + 1.5f));
    EXPECT_NEAR(schlick(1.0f, 1.0f, 1.5f), R0, 1e-6f); // cosθ = 1 -> R0
}

TEST(SchlickTest, GrazingIncidenceApproachesOne)
{
    EXPECT_NEAR(schlick(0.0f, 1.0f, 1.5f), 1.0f, 1e-6f); // cosθ = 0 -> 1
}

TEST(SchlickTest, SymmetricInIndices)
{
    EXPECT_NEAR(schlick(0.3f, 1.0f, 1.5f), schlick(0.3f, 1.5f, 1.0f), 1e-6f);
}

TEST(ReflectOrRefractRayTest, RefractsStraightThroughAtNormalIncidence)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    int inside = VACUUM_MAT;
    FixedRng rng{0.9f}; // above Fresnel reflectance -> transmit
    const Vec3 v = reflectOrRefractRay(Vec3{0, 0, -1}, glass, inside, /*hit_mat=*/7,
                                       /*enter=*/true, Vec3{0, 0, 1}, rng);

    expectVec3Near(v, {0, 0, -1}, 1e-5f); // no bending at normal incidence
    EXPECT_EQ(inside, 7);                 // now travelling inside the medium
}

TEST(ReflectOrRefractRayTest, FresnelReflectsWhenRngBelowReflectance)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    int inside = VACUUM_MAT;
    FixedRng rng{0.0f}; // below reflectance -> reflect
    const Vec3 v = reflectOrRefractRay(Vec3{0, 0, -1}, glass, inside, 7,
                                       /*enter=*/true, Vec3{0, 0, 1}, rng);

    expectVec3Near(v, {0, 0, 1}, 1e-5f); // mirror reflection
    EXPECT_EQ(inside, VACUUM_MAT);       // stays outside on reflection
}

TEST(ReflectOrRefractRayTest, TotalInternalReflectionAtGrazingExit)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    int inside = 7; // exiting the medium
    FixedRng rng{0.9f}; // ignored: TIR does not consult the rng
    const Vec3 v = reflectOrRefractRay(Vec3{0.8f, 0, -0.6f}, glass, inside, 7,
                                       /*enter=*/false, Vec3{0, 0, 1}, rng);

    expectVec3Near(v, {0.8f, 0, 0.6f}, 1e-5f); // fully reflected
    EXPECT_EQ(inside, 7);                       // stays inside
}

// Device-resident scene: owns the GPU triangle storage and exposes it as a span.
struct DeviceScene
{
    GpuTris tris;
    TriangleMesh host_view;
    DeviceBuffer<TriangleMesh> objects;

    std::span<const TriangleMesh> span() { return objects.deviceSpan(); }
};

namespace
{
    PathVertexDataDevice deviceView(PathVertexData &p)
    {
        return PathVertexDataDevice{
            .t = p.t.devicePtr(), .bsdfPdfPrev = p.bsdfPdfPrev.devicePtr(),
            .throughput = p.throughput.devicePtr(), .prevSpecular = p.prevSpecular.devicePtr(),
            .ids = p.ids.devicePtr()};
    }

    WavefrontDataDevice deviceView(WavefrontData &w)
    {
        return WavefrontDataDevice{.rays = w.rays.devicePtr(), .current_mat = w.current_mat.devicePtr()};
    }

    RenderBuffersDevice deviceView(RenderBuffers &o, Size2i resolution)
    {
        return RenderBuffersDevice{.colors = o.colors.devicePtr(), .resolution = resolution};
    }

    // Uploads a mesh (with per-triangle light sampler) to the device and keeps
    // the backing storage alive for as long as the returned object lives.
    DeviceScene uploadScene(Mesh mesh, int material)
    {
        GpuTris tris = convertMeshToTris(mesh, /*generateTriangleSampler=*/true);
        TriangleMesh view = viewGpuTris(tris);
        view.material = material;
        view.model_to_world = Transform{.s = {1, 1, 1}, .p = {0, 0, 0}};

        auto objects = DeviceBuffer<TriangleMesh>::allocate(1);
        objects.deviceData.copyFromHost(&view, sizeof(TriangleMesh));

        return DeviceScene{.tris = std::move(tris), .host_view = view, .objects = std::move(objects)};
    }

    // Device-resident, uniformly weighted alias table over `objectCount` lights.
    DeviceBuffer<AliasEntry> uploadLightTable(int objectCount)
    {
        AliasTable table = generateAliasTable(std::vector<float>(objectCount, 1.0f));
        const std::vector<AliasEntry> entries(table.entries.hostPtr(),
                                              table.entries.hostPtr() + table.entries.size());
        return deviceBufferFrom(entries);
    }

    void syncDevice()
    {
        CUDA_ERROR_CHECK();
        cudaDeviceSynchronize();
        CUDA_ERROR_CHECK();
    }

    DiffuseMaterial diffuse(Vec4 reflectance, Vec4 emission, Vec4 debug = {0, 0, 0, 0})
    {
        DiffuseMaterial m{};
        m.debug_color = debug;
        m.emission = emission;
        m.diffuse_reflectance = reflectance;
        return m;
    }

    PathVertexData makePaths(const std::vector<TriangleIdentifier> &ids,
                             const std::vector<float> &t,
                             const std::vector<Vec4> &throughput,
                             const std::vector<int> &prevSpecular,
                             const std::vector<float> &bsdfPdfPrev)
    {
        return PathVertexData{
            .t = deviceBufferFrom(t),
            .bsdfPdfPrev = deviceBufferFrom(bsdfPdfPrev),
            .throughput = deviceBufferFrom(throughput),
            .prevSpecular = deviceBufferFrom(prevSpecular),
            .ids = deviceBufferFrom(ids)};
    }
}

// Wrapper kernel so shadeDiffuseMaterial can be exercised directly on the GPU.
__global__ void callShadeDiffuseMaterial(
    PathVertexDataDevice vertex, WavefrontDataDevice wavefront, curandState *rand_states,
    std::span<const TriangleMesh> objects, std::span<const Material> materials,
    std::span<const AliasEntry> light_table, int path_count, ObjectsInfo info, RenderBuffersDevice output)
{
    int idx = KERNEL_IDX(path_count);
    const auto ids = vertex.ids[idx];
    if (!ids.valid()) return;

    const auto &object = objects[ids.meshID];
    const auto &material = materials[object.material];
    auto rng = CudaRng{rand_states + idx};

    shadeDiffuseMaterial(vertex, wavefront, ids, material, idx, object, rng,
                         objects, materials, light_table, info, output);
}

// ==========================================================================
// CUDA kernel tests.
//
// Registered with CTest as a single binary (see add_shared_gpu_test_dr), so
// every case runs in one process and the CUDA runtime is initialized only once
// instead of once per test.
// ==========================================================================

TEST(Kernels, InitPathsSetsInitialState)
{
    const int N = 8;
    // Seed with sentinels to prove the kernel overwrites every field.
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>(N, TriangleIdentifier::invalid()),
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{9, 9, 9, 9}),
        std::vector<int>(N, -1), std::vector<float>(N, -1.f));

    initPaths<<<1, N>>>(deviceView(paths), N);
    syncDevice();

    const auto throughput = paths.throughput.toHost();
    const auto prevSpecular = paths.prevSpecular.toHost();
    const auto bsdfPdf = paths.bsdfPdfPrev.toHost();
    for (int i = 0; i < N; ++i)
    {
        expectVec4Near(throughput[i], {1, 1, 1, 0});
        EXPECT_EQ(prevSpecular[i], 1);
        EXPECT_FLOAT_EQ(bsdfPdf[i], 0.0f);
    }
}

TEST(Kernels, InitCameraRaysMatchesHostCameraRay)
{
    const Camera cam{
        .transform = Matrix4x4f::identity(),
        .intrinsics = Intrinsics{.focal_length = 1.0f, .center = Vec2f{1.0f, 1.0f}},
        .resolution = Size2i{2, 2},
        .physical_pixel_size = Size2f{1.0f, 1.0f},
    };
    const int N = cam.resolution.area();

    WavefrontData wf{.rays = DeviceBuffer<Ray>::allocate(N), .current_mat = DeviceBuffer<int>::allocate(N)};
    CudaRandom rng(N); // Center sampling ignores the rng, so this stays deterministic.

    initCameraRays<<<1, N>>>(deviceView(wf), PixelSampling::Center, cam, rng.devicePtr());
    syncDevice();

    const auto rays = wf.rays.toHost();
    const auto current_mat = wf.current_mat.toHost();
    FixedRng hostRng{0.5f};
    for (int idx = 0; idx < N; ++idx)
    {
        const auto pixel = Vec2f{static_cast<float>(idx % cam.resolution.width),
                                 static_cast<float>(idx / cam.resolution.width)};
        const Ray expected = cameraRay(cam, pixel, PixelSampling::Center, 0, hostRng);
        expectVec3Near(rays[idx].p, expected.p, 1e-5f);
        expectVec3Near(rays[idx].v, expected.v, 1e-5f);
        EXPECT_EQ(current_mat[idx], VACUUM_MAT);
    }
}

TEST(Kernels, ExtendPathsRecordsHitAndMiss)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 2;
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}},  // hits
                                                  Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}}),        // misses
        .current_mat = deviceBufferFrom(std::vector<int>(N, VACUUM_MAT))};
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>(N, TriangleIdentifier::invalid()),
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(), N, casts.devicePtr(), /*depth=*/0);
    syncDevice();

    const auto ids = paths.ids.toHost();
    const auto t = paths.t.toHost();
    const auto castCounts = casts.toHost();

    EXPECT_TRUE(ids[0].valid());
    EXPECT_EQ(ids[0].meshID, 0);
    EXPECT_EQ(ids[0].triangleID, 0);
    EXPECT_NEAR(t[0], 5.0f, 1e-4f);
    EXPECT_FALSE(ids[1].valid());

    // depth 0: every path casts a primary ray.
    EXPECT_EQ(castCounts[0], 1u);
    EXPECT_EQ(castCounts[1], 1u);
}

TEST(Kernels, ExtendPathsCountsOnlyLivePaths)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 2;
    const Ray hit{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{hit, hit}),
        .current_mat = deviceBufferFrom(std::vector<int>(N, VACUUM_MAT))};
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}, // alive last depth
                                        TriangleIdentifier::invalid()},                    // terminated last depth
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(), N, casts.devicePtr(), /*depth=*/1);
    syncDevice();

    const auto castCounts = casts.toHost();
    EXPECT_EQ(castCounts[0], 1u); // live path counted
    EXPECT_EQ(castCounts[1], 0u); // dead path not counted
}

TEST(Kernels, SampleBsdfDiffuseScattersIntoHemisphere)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(
        std::vector<Material>{Material{diffuse(/*reflectance=*/{0.5f, 0.25f, 0.125f, 0}, /*emission=*/{0, 0, 0, 0})}});

    const int N = 1;
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}}),
        .current_mat = deviceBufferFrom(std::vector<int>{VACUUM_MAT})};
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{0}, std::vector<float>{0.f});

    CudaRandom rng(N);
    sampleBsdfDirection<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(), materials.deviceSpan(), N);
    syncDevice();

    // Throughput picks up the reflectance (independent of the random direction).
    expectVec4Near(paths.throughput.toHost()[0], {0.5f, 0.25f, 0.125f, 0});
    EXPECT_EQ(paths.prevSpecular.toHost()[0], 0);
    EXPECT_EQ(wf.current_mat.toHost()[0], VACUUM_MAT);

    // Surface normal faces the incoming ray (+z here); the bounce stays in that hemisphere.
    const Vec3 n{0, 0, 1};
    const Ray bounce = wf.rays.toHost()[0];
    EXPECT_NEAR(bounce.v.length(), 1.0f, 1e-4f);
    EXPECT_GE(bounce.v.dot(n), -1e-5f);
    expectVec3Near(bounce.p, {0.2f, 0.2f, 1e-5f}, 1e-4f);
    // The recorded pdf is consistent with the direction that was actually drawn.
    EXPECT_NEAR(paths.bsdfPdfPrev.toHost()[0], cosineWeightedHemisphereDirPdf(bounce.v, n), 1e-4f);
}

TEST(Kernels, SampleBsdfTransparentMarksSpecular)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(std::vector<Material>{Material{glass}});

    const int N = 1;
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}}),
        .current_mat = deviceBufferFrom(std::vector<int>{VACUUM_MAT})}; // entering
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{0}, std::vector<float>{0.f});

    CudaRandom rng(N);
    sampleBsdfDirection<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(), materials.deviceSpan(), N);
    syncDevice();

    // Whether the draw reflects or refracts, the transparent branch marks the
    // bounce specular with a unit pdf and produces a unit-length axis-aligned ray.
    EXPECT_EQ(paths.prevSpecular.toHost()[0], 1);
    EXPECT_FLOAT_EQ(paths.bsdfPdfPrev.toHost()[0], 1.0f);
    const Ray bounce = wf.rays.toHost()[0];
    EXPECT_NEAR(bounce.v.length(), 1.0f, 1e-4f);
    EXPECT_NEAR(std::abs(bounce.v.z), 1.0f, 1e-4f);
    const int medium = wf.current_mat.toHost()[0];
    EXPECT_TRUE(medium == VACUUM_MAT || medium == 0);
}

TEST(Kernels, ShadeSkipsInvalidAndTransparentHits)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(std::vector<Material>{Material{glass}});
    DeviceBuffer<AliasEntry> lightTable = uploadLightTable(1);

    const int N = 2;
    const Ray ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{ray, ray}),
        .current_mat = deviceBufferFrom(std::vector<int>(N, VACUUM_MAT))};
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier::invalid(),                     // no hit
                                        TriangleIdentifier{.meshID = 0, .triangleID = 0}}, // transparent hit
        std::vector<float>{0.f, 5.0f}, std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};

    CudaRandom rng(N);
    const ObjectsInfo info{.total_radiant_power = 1.0f};
    shade<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                    materials.deviceSpan(), lightTable.deviceSpan(), N, info, deviceView(out, Size2i{N, 1}));
    syncDevice();

    const auto colors = out.colors.toHost();
    expectVec4Near(colors[0], {0, 0, 0, 0});
    expectVec4Near(colors[1], {0, 0, 0, 0});
}

TEST(Kernels, ShadeDiffuseAccumulatesEmission)
{
    const Vec4 emission{2, 3, 4, 0};
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(
        std::vector<Material>{Material{diffuse(/*reflectance=*/{0.5f, 0.5f, 0.5f, 0}, emission)}});
    DeviceBuffer<AliasEntry> lightTable = uploadLightTable(1);

    const int N = 1;
    WavefrontData wf{
        .rays = deviceBufferFrom(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}}),
        .current_mat = deviceBufferFrom(std::vector<int>{VACUUM_MAT})};
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{1},     // prevSpecular: count emission directly (no MIS weight)
        std::vector<float>{0.f});
    RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};

    CudaRandom rng(N);
    const ObjectsInfo info{.total_radiant_power = 1.0f};
    callShadeDiffuseMaterial<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                                       materials.deviceSpan(), lightTable.deviceSpan(), N, info, deviceView(out, Size2i{N, 1}));
    syncDevice();

    // prevSpecular contributes raw emission; NEE can only add non-negative light.
    const Vec4 c = out.colors.toHost()[0];
    EXPECT_GE(c.x, emission.x - 1e-4f);
    EXPECT_GE(c.y, emission.y - 1e-4f);
    EXPECT_GE(c.z, emission.z - 1e-4f);
    EXPECT_TRUE(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z));
}

TEST(Kernels, DebugShadeColorsFirstHit)
{
    const Vec4 debugColor{0.2f, 0.4f, 0.6f, 0};
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(
        std::vector<Material>{Material{diffuse({0, 0, 0, 0}, {0, 0, 0, 0}, debugColor)}});

    auto runMode = [&](DebugOptions mode, bool hit) {
        const int N = 1;
        WavefrontData wf{
            .rays = deviceBufferFrom(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}}),
            .current_mat = deviceBufferFrom(std::vector<int>{VACUUM_MAT})};
        PathVertexData paths = makePaths(
            std::vector<TriangleIdentifier>{hit ? TriangleIdentifier{.meshID = 0, .triangleID = 0}
                                                : TriangleIdentifier::invalid()},
            std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{0, 0, 0, 0}},
            std::vector<int>{0}, std::vector<float>{0.f});
        RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};

        debugShade<<<1, N>>>(deviceView(paths), deviceView(wf), scene.span(), materials.deviceSpan(),
                             mode, N, deviceView(out, Size2i{N, 1}));
        syncDevice();
        return out.colors.toHost()[0];
    };

    // Miss -> black regardless of mode.
    expectVec4Near(runMode(DebugOptions::Off, /*hit=*/false), {0, 0, 0, 0});

    // Off -> the material's debug color.
    expectVec4Near(runMode(DebugOptions::Off, true), debugColor);

    // Barycentric coordinates of (0.2, 0.2) inside triangle A(0,0) B(1,0) C(0,1).
    expectVec4Near(runMode(DebugOptions::BariCoords, true), {0.6f, 0.2f, 0.2f, 0}, 1e-4f);

    // Ray hits the back face here, so the winding reads clockwise.
    expectVec4Near(runMode(DebugOptions::WindingOrder, true), {0.53f, 0.82f, 1.0f, 0}, 1e-4f);

    // uv is fixed at (0,0) so the checker samples the dark tile: 0.5 * debug color.
    expectVec4Near(runMode(DebugOptions::UVChecker, true),
                   {0.5f * debugColor.x, 0.5f * debugColor.y, 0.5f * debugColor.z, 0}, 1e-4f);
}

// ==========================================================================
// DeviceArray / DeviceVector host<->device transfer.
//
// Only the cases that actually allocate or copy device memory live here, so
// they share this binary's single CUDA context. The pure host-side cases
// (fill, size tracking, move of an unallocated buffer) stay in test_device.cpp
// where they run per-case without touching the GPU.
// ==========================================================================

TEST(DeviceArrayTest, HostDeviceRoundTripRestoresData)
{
    DeviceArray<int> array(3, 5);
    array.reset();
    array.ensureDeviceAllocation();

    // Clobber the host copy, then pull it back from the device.
    array.hostPtr()[0] = 999;
    array.hostPtr()[1] = -1;
    array.updateHostData();

    EXPECT_EQ(array.hostPtr()[0], 5);
    EXPECT_EQ(array.hostPtr()[1], 5);
    EXPECT_EQ(array.hostPtr()[2], 5);
}

TEST(DeviceVectorTest, LazyAllocationProvidesDevicePointer)
{
    DeviceVector<float> vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    vector.ensureDeviceAllocation();

    EXPECT_NE(vector.devicePtr(), nullptr);
    EXPECT_EQ(vector.deviceSpan().size(), 4u);
}

TEST(DeviceVectorTest, MoveKeepsDeviceAllocation)
{
    DeviceVector<int> source(std::vector<int>{10, 20, 30});
    source.ensureDeviceAllocation();

    DeviceVector<int> moved = std::move(source);
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved.hostPtr()[0], 10);
    EXPECT_NE(moved.devicePtr(), nullptr);
}

// ==========================================================================
// GpuTris device view: viewGpuTris allocates the mesh on the device, so it
// belongs with the GPU tests rather than the host-only scene tests.
// ==========================================================================

TEST(ViewGpuTrisTest, WiresUpDevicePointersAndSurfaceArea)
{
    GpuTris quad = createQuadMesh(Vec3{0, 0, 0}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 2.0f);
    const TriangleMesh view = viewGpuTris(quad);

    EXPECT_EQ(view.triangle_count, 2);
    EXPECT_NE(view.points, nullptr);
    EXPECT_NE(view.normals, nullptr);
    EXPECT_NE(view.triangles, nullptr);
    EXPECT_NE(view.triangle_sampler, nullptr);
    EXPECT_FALSE(view.bbh.isEmpty());

    EXPECT_NEAR(view.base_surface_area, 4.0f, 1e-5f);
    EXPECT_FLOAT_EQ(view.surface_area, view.base_surface_area);
}
