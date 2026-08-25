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
#include "device/Binning.hpp"
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
#include <numeric>
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
            .ids = p.ids.devicePtr(), .alive = p.alive.devicePtr()};
    }

    WavefrontDataDevice deviceView(WavefrontData &w)
    {
        return WavefrontDataDevice{.rays = w.rays.devicePtr(), .current_mat = w.current_mat.devicePtr(),
                                   .sort_index = w.sort_index.devicePtr()};
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

    // TriangleIdentifier no longer carries a validity sentinel; path liveness
    // lives in PathVertexData::alive, so a lane that never hit anything just
    // holds filler ids that the kernels must not read.
    constexpr TriangleIdentifier NO_HIT{.meshID = -1, .triangleID = -1};

    PathVertexData makePaths(const std::vector<TriangleIdentifier> &ids,
                             const std::vector<int> &alive,
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
            .ids = deviceBufferFrom(ids),
            .alive = deviceBufferFrom(alive)};
    }

    // Wavefront whose sort_index is the identity permutation: the state
    // initCameraRays leaves behind, and what the binning pass reproduces while
    // every path is still alive.
    WavefrontData makeWavefront(const std::vector<Ray> &rays, const std::vector<int> &current_mat)
    {
        std::vector<int> identity(rays.size());
        std::iota(identity.begin(), identity.end(), 0);

        return WavefrontData{
            .rays = deviceBufferFrom(rays),
            .current_mat = deviceBufferFrom(current_mat),
            .sort_index = deviceBufferFrom(identity)};
    }

    // Wavefront with an explicit, already-compacted sort_index, as the binning
    // pass writes it once some paths have terminated.
    WavefrontData makeWavefront(const std::vector<Ray> &rays, const std::vector<int> &current_mat,
                                const std::vector<int> &sort_index)
    {
        return WavefrontData{
            .rays = deviceBufferFrom(rays),
            .current_mat = deviceBufferFrom(current_mat),
            .sort_index = deviceBufferFrom(sort_index)};
    }

    // The wavefront kernels read their launch bound from device memory (cub's
    // DeviceSelect::Flagged writes the live-path count there during binning),
    // so tests must hand them a device-resident count rather than a plain int.
    DeviceBuffer<int> makeAliveCount(int count)
    {
        return deviceBufferFrom(std::vector<int>{count});
    }
}

// Wrapper kernel so shadeDiffuseMaterial can be exercised directly on the GPU.
// The prologue mirrors shade's: the launch bound is read from device memory and
// the lane is reached through the binning pass' compacted sort_index.
__global__ void callShadeDiffuseMaterial(
    PathVertexDataDevice vertex, WavefrontDataDevice wavefront, curandState *rand_states,
    std::span<const TriangleMesh> objects, std::span<const Material> materials,
    std::span<const AliasEntry> light_table, int *alive_count, ObjectsInfo info, RenderBuffersDevice output)
{
    const int thread_idx = KERNEL_IDX(*alive_count);
    const int idx = wavefront.sort_index[thread_idx];
    if (!vertex.alive[idx]) return;

    const auto ids = vertex.ids[idx];
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
        std::vector<TriangleIdentifier>(N, NO_HIT),
        std::vector<int>(N, 0), // alive sentinel: the kernel must raise every lane
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{9, 9, 9, 9}),
        std::vector<int>(N, -1), std::vector<float>(N, -1.f));

    initPaths<<<1, N>>>(deviceView(paths), N);
    syncDevice();

    const auto throughput = paths.throughput.toHost();
    const auto prevSpecular = paths.prevSpecular.toHost();
    const auto bsdfPdf = paths.bsdfPdfPrev.toHost();
    const auto alive = paths.alive.toHost();
    for (int i = 0; i < N; ++i)
    {
        expectVec4Near(throughput[i], {1, 1, 1, 0});
        EXPECT_EQ(prevSpecular[i], 1);
        EXPECT_FLOAT_EQ(bsdfPdf[i], 0.0f);
        EXPECT_EQ(alive[i], 1); // every path starts alive
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

    WavefrontData wf{.rays = DeviceBuffer<Ray>::allocate(N),
                     .current_mat = DeviceBuffer<int>::allocate(N),
                     // Sentinel: the kernel must seed the identity permutation.
                     .sort_index = deviceBufferFrom(std::vector<int>(N, -1))};
    CudaRandom rng(N); // Center sampling ignores the rng, so this stays deterministic.

    initCameraRays<<<1, N>>>(deviceView(wf), PixelSampling::Center, cam, rng.devicePtr());
    syncDevice();

    const auto rays = wf.rays.toHost();
    const auto current_mat = wf.current_mat.toHost();
    const auto sort_index = wf.sort_index.toHost();
    FixedRng hostRng{0.5f};
    for (int idx = 0; idx < N; ++idx)
    {
        const auto pixel = Vec2f{static_cast<float>(idx % cam.resolution.width),
                                 static_cast<float>(idx / cam.resolution.width)};
        const Ray expected = cameraRay(cam, pixel, PixelSampling::Center, 0, hostRng);
        expectVec3Near(rays[idx].p, expected.p, 1e-5f);
        expectVec3Near(rays[idx].v, expected.v, 1e-5f);
        EXPECT_EQ(current_mat[idx], VACUUM_MAT);
        EXPECT_EQ(sort_index[idx], idx); // unbinned wavefront is the identity order
    }
}

TEST(Kernels, ExtendPathsRecordsHitAndMiss)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 2;
    WavefrontData wf = makeWavefront(
        std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}},   // hits
                         Ray{.p = {5, 5, 5}, .v = {0, 0, -1}}},        // misses
        std::vector<int>(N, VACUUM_MAT));
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>(N, NO_HIT),
        std::vector<int>(N, 1), // both paths enter the bounce alive
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(),
                          aliveCount.devicePtr(), casts.devicePtr(), /*depth=*/0);
    syncDevice();

    const auto ids = paths.ids.toHost();
    const auto t = paths.t.toHost();
    const auto alive = paths.alive.toHost();
    const auto castCounts = casts.toHost();

    // The hit records its ids and distance and stays alive.
    EXPECT_EQ(alive[0], 1);
    EXPECT_EQ(ids[0].meshID, 0);
    EXPECT_EQ(ids[0].triangleID, 0);
    EXPECT_NEAR(t[0], 5.0f, 1e-4f);

    // The miss is retired by clearing its alive flag.
    EXPECT_EQ(alive[1], 0);

    // depth 0: every path casts a primary ray.
    EXPECT_EQ(castCounts[0], 1u);
    EXPECT_EQ(castCounts[1], 1u);
}

// The alive flag guards the kernel body even when a stale sort_index still
// points at a terminated lane.
TEST(Kernels, ExtendPathsCountsOnlyLivePaths)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 2;
    const Ray hit{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    WavefrontData wf = makeWavefront(std::vector<Ray>{hit, hit}, std::vector<int>(N, VACUUM_MAT));
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}, NO_HIT},
        std::vector<int>{1, 0}, // lane 0 alive, lane 1 terminated at the last depth
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(),
                          aliveCount.devicePtr(), casts.devicePtr(), /*depth=*/1);
    syncDevice();

    const auto castCounts = casts.toHost();
    EXPECT_EQ(castCounts[0], 1u); // live path counted
    EXPECT_EQ(castCounts[1], 0u); // dead path not counted
}

// With a compacted sort_index and a live count below the buffer size, the
// kernel must touch exactly the selected lanes and leave the rest alone.
TEST(Kernels, ExtendPathsFollowsCompactedSortIndex)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 3;
    const Ray hit{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    // Lane 0 is dead, so binning compacts the live lanes 1 and 2 to the front.
    WavefrontData wf = makeWavefront(std::vector<Ray>{hit, hit, hit},
                                     std::vector<int>(N, VACUUM_MAT),
                                     std::vector<int>{1, 2, 0});
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>(N, NO_HIT),
        std::vector<int>{0, 1, 1},
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));
    DeviceBuffer<int> aliveCount = makeAliveCount(2); // only two live paths

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(),
                          aliveCount.devicePtr(), casts.devicePtr(), /*depth=*/1);
    syncDevice();

    const auto castCounts = casts.toHost();
    const auto ids = paths.ids.toHost();
    EXPECT_EQ(castCounts[0], 0u); // dead lane never reached by the compacted index
    EXPECT_EQ(castCounts[1], 1u);
    EXPECT_EQ(castCounts[2], 1u);

    // The two live lanes recorded their hit; the dead one keeps its filler ids.
    EXPECT_EQ(ids[0].meshID, NO_HIT.meshID);
    EXPECT_EQ(ids[1].meshID, 0);
    EXPECT_EQ(ids[2].meshID, 0);
}

TEST(Kernels, SampleBsdfDiffuseScattersIntoHemisphere)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(
        std::vector<Material>{Material{diffuse(/*reflectance=*/{0.5f, 0.25f, 0.125f, 0}, /*emission=*/{0, 0, 0, 0})}});

    const int N = 1;
    WavefrontData wf = makeWavefront(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}},
                                     std::vector<int>{VACUUM_MAT});
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<int>{1},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{0}, std::vector<float>{0.f});
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    CudaRandom rng(N);
    sampleBsdfDirection<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                                  materials.deviceSpan(), aliveCount.devicePtr());
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
    WavefrontData wf = makeWavefront(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}},
                                     std::vector<int>{VACUUM_MAT}); // entering
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<int>{1},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{0}, std::vector<float>{0.f});
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    CudaRandom rng(N);
    sampleBsdfDirection<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                                  materials.deviceSpan(), aliveCount.devicePtr());
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

TEST(Kernels, ShadeSkipsDeadAndTransparentHits)
{
    TransparentMaterial glass{};
    glass.inside_medium.ior = 1.5f;

    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);
    DeviceBuffer<Material> materials = deviceBufferFrom(std::vector<Material>{Material{glass}});
    DeviceBuffer<AliasEntry> lightTable = uploadLightTable(1);

    const int N = 2;
    const Ray ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    WavefrontData wf = makeWavefront(std::vector<Ray>{ray, ray}, std::vector<int>(N, VACUUM_MAT));
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{NO_HIT,                                            // no hit
                                        TriangleIdentifier{.meshID = 0, .triangleID = 0}}, // transparent hit
        std::vector<int>{0, 1}, // lane 0 died on the miss, lane 1 is still alive
        std::vector<float>{0.f, 5.0f}, std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    CudaRandom rng(N);
    const ObjectsInfo info{.total_radiant_power = 1.0f};
    shade<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                    materials.deviceSpan(), lightTable.deviceSpan(), aliveCount.devicePtr(),
                    info, deviceView(out, Size2i{N, 1}));
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
    WavefrontData wf = makeWavefront(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}},
                                     std::vector<int>{VACUUM_MAT});
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>{TriangleIdentifier{.meshID = 0, .triangleID = 0}},
        std::vector<int>{1},
        std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{1, 1, 1, 0}},
        std::vector<int>{1},     // prevSpecular: count emission directly (no MIS weight)
        std::vector<float>{0.f});
    RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};
    DeviceBuffer<int> aliveCount = makeAliveCount(N);

    CudaRandom rng(N);
    const ObjectsInfo info{.total_radiant_power = 1.0f};
    callShadeDiffuseMaterial<<<1, N>>>(deviceView(paths), deviceView(wf), rng.devicePtr(), scene.span(),
                                       materials.deviceSpan(), lightTable.deviceSpan(),
                                       aliveCount.devicePtr(), info, deviceView(out, Size2i{N, 1}));
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
        // The debug path never runs the binning pass, so sort_index stays the
        // identity permutation initCameraRays wrote.
        WavefrontData wf = makeWavefront(std::vector<Ray>{Ray{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}}},
                                         std::vector<int>{VACUUM_MAT});
        PathVertexData paths = makePaths(
            std::vector<TriangleIdentifier>{hit ? TriangleIdentifier{.meshID = 0, .triangleID = 0} : NO_HIT},
            std::vector<int>{hit ? 1 : 0},
            std::vector<float>{5.0f}, std::vector<Vec4>{Vec4{0, 0, 0, 0}},
            std::vector<int>{0}, std::vector<float>{0.f});
        RenderBuffers out{.colors = deviceBufferFrom(std::vector<Vec4>(N, Vec4{0, 0, 0, 0}))};
        DeviceBuffer<int> aliveCount = makeAliveCount(N);

        debugShade<<<1, N>>>(deviceView(paths), deviceView(wf), scene.span(), materials.deviceSpan(),
                             mode, aliveCount.devicePtr(), deviceView(out, Size2i{N, 1}));
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
// DeviceBuffer upload and the DeviceBinning stream-compaction pass that turns
// the per-path alive flags into the compacted sort_index the wavefront kernels
// are launched over.
// ==========================================================================

TEST(DeviceBufferTest, FromHostUploadsAndRoundTrips)
{
    const std::vector<int> host{5, -3, 17, 0};
    DeviceBuffer<int> buffer = DeviceBuffer<int>::fromHost(host);

    EXPECT_EQ(buffer.elementCount, 4);
    EXPECT_EQ(buffer.toHost(), host);
}

TEST(DeviceBinningTest, StartsWithEveryLaneSelected)
{
    // The debug render path never calls execute(), so create() must leave the
    // count at the full path count for the kernels' launch bound to be right.
    const int N = 6;
    DeviceBinning binning = DeviceBinning::create(N);

    EXPECT_EQ(binning.num_selected.toHost()[0], N);
    EXPECT_EQ(binning.identity_keys.elementCount, N);
}

TEST(DeviceBinningTest, CompactsIndicesOfFlaggedLanes)
{
    const int N = 8;
    DeviceBinning binning = DeviceBinning::create(N);

    // Lanes 1, 2, 5 and 7 are still alive.
    DeviceBuffer<int> flags = deviceBufferFrom(std::vector<int>{0, 1, 1, 0, 0, 1, 0, 1});
    DeviceBuffer<int> out = deviceBufferFrom(std::vector<int>(N, -1));

    binning.execute(flags.devicePtr(), out.devicePtr());
    syncDevice();

    EXPECT_EQ(binning.num_selected.toHost()[0], 4);

    // The live lane indices are packed to the front, in ascending order.
    const auto indices = out.toHost();
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);
    EXPECT_EQ(indices[2], 5);
    EXPECT_EQ(indices[3], 7);
    EXPECT_EQ(indices[4], -1); // tail past num_selected is left untouched
}

TEST(DeviceBinningTest, AllAliveYieldsIdentityPermutation)
{
    const int N = 5;
    DeviceBinning binning = DeviceBinning::create(N);

    DeviceBuffer<int> flags = deviceBufferFrom(std::vector<int>(N, 1));
    DeviceBuffer<int> out = deviceBufferFrom(std::vector<int>(N, -1));

    binning.execute(flags.devicePtr(), out.devicePtr());
    syncDevice();

    EXPECT_EQ(binning.num_selected.toHost()[0], N);
    const auto indices = out.toHost();
    for (int i = 0; i < N; ++i) EXPECT_EQ(indices[i], i);
}

TEST(DeviceBinningTest, AllDeadSelectsNothing)
{
    const int N = 4;
    DeviceBinning binning = DeviceBinning::create(N);

    DeviceBuffer<int> flags = deviceBufferFrom(std::vector<int>(N, 0));
    DeviceBuffer<int> out = deviceBufferFrom(std::vector<int>(N, -1));

    binning.execute(flags.devicePtr(), out.devicePtr());
    syncDevice();

    // Every subsequent kernel launch then has a zero bound and does no work.
    EXPECT_EQ(binning.num_selected.toHost()[0], 0);
    EXPECT_EQ(out.toHost()[0], -1);
}

// The binning pass feeding extendPaths: only the lanes cub selected get cast.
TEST(DeviceBinningTest, DrivesExtendPathsOverLiveLanesOnly)
{
    DeviceScene scene = uploadScene(test::unitTriangleMesh(), 0);

    const int N = 4;
    const Ray hit{.p = {0.2f, 0.2f, 5}, .v = {0, 0, -1}};
    WavefrontData wf = makeWavefront(std::vector<Ray>(N, hit), std::vector<int>(N, VACUUM_MAT));
    PathVertexData paths = makePaths(
        std::vector<TriangleIdentifier>(N, NO_HIT),
        std::vector<int>{0, 1, 0, 1}, // lanes 1 and 3 survived the last bounce
        std::vector<float>(N, 0.f), std::vector<Vec4>(N, Vec4{1, 1, 1, 0}),
        std::vector<int>(N, 0), std::vector<float>(N, 0.f));
    DeviceBuffer<uint32_t> casts = deviceBufferFrom(std::vector<uint32_t>(N, 0u));

    DeviceBinning binning = DeviceBinning::create(N);
    binning.execute(paths.alive.devicePtr(), wf.sort_index.devicePtr());

    extendPaths<<<1, N>>>(deviceView(wf), deviceView(paths), scene.span(),
                          binning.num_selected.devicePtr(), casts.devicePtr(), /*depth=*/2);
    syncDevice();

    const auto castCounts = casts.toHost();
    EXPECT_EQ(castCounts[0], 0u);
    EXPECT_EQ(castCounts[1], 1u);
    EXPECT_EQ(castCounts[2], 0u);
    EXPECT_EQ(castCounts[3], 1u);

    // Both live lanes hit the triangle, so they stay alive for the next bounce.
    const auto alive = paths.alive.toHost();
    EXPECT_EQ(alive[1], 1);
    EXPECT_EQ(alive[3], 1);
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
