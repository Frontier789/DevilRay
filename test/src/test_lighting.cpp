// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/IntersectionImpl.hpp"
#include "tracing/Scene.hpp"
#include "tracing/LightSampling.hpp"
#include "tracing/ShadingUtils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

using test::DeterministicRng;
using test::HostObject;
using test::makeHostObject;

namespace
{
    DiffuseMaterial emissiveMaterial(float emission)
    {
        DiffuseMaterial material{};
        material.debug_color = {0, 0, 0, 0};
        material.emission = {emission, emission, emission, 0};
        material.diffuse_reflectance = {0, 0, 0, 0};
        return material;
    }
}

// --- misWeightedEmission ---

TEST(MisWeightedEmissionTest, SpecularBounceReturnsRawEmission)
{
    const Vec4 emission{2, 4, 6, 0};
    const Vec4 throughput{1, 0.5f, 0.25f, 0};

    const auto result = misWeightedEmission(emission, throughput, true, MisPdfs{.bsdf_pdf = 5, .nee_pdf = 3});
    test::expectVec4Near(result, {2, 2, 1.5f, 0});
}

TEST(MisWeightedEmissionTest, NonSpecularWeightsByPowerHeuristic)
{
    const Vec4 emission{2, 4, 6, 0};
    const Vec4 throughput{1, 1, 1, 0};
    const MisPdfs pdfs{.bsdf_pdf = 3, .nee_pdf = 1};

    const auto result = misWeightedEmission(emission, throughput, false, pdfs);
    const float weight = powerHeuristic(pdfs.bsdf_pdf, pdfs.nee_pdf);
    test::expectVec4Near(result, emission * weight);
}

TEST(MisWeightedEmissionTest, ZeroPdfNonSpecularContributesNothing)
{
    const auto result = misWeightedEmission({2, 4, 6, 0}, {1, 1, 1, 0}, false, MisPdfs{.bsdf_pdf = 0, .nee_pdf = 0});
    test::expectVec4Near(result, {0, 0, 0, 0});
}

// --- computeNeePdf ---

TEST(ComputeNeePdfTest, EmitterPowerShareTimesSolidAngleConversion)
{
    constexpr float emission = 0.5f;
    constexpr float total_radiant_power = 10.0f;
    constexpr float distance_to_emitter = 2.0f;
    constexpr float cosine_at_emitter = 1.0f;

    const Material emitter = emissiveMaterial(emission);
    const Vec3 vertexPos{0, 0, 0};
    const Vec3 emitterPos{0, 0, distance_to_emitter};
    const Vec3 emitterNormalFacingVertex{0, 0, -1};
    const ObjectsInfo info{.total_radiant_power = total_radiant_power};

    const float nee_pdf = computeNeePdf(vertexPos, emitterPos, emitterNormalFacingVertex, emitter, info);

    const float radiant_exitance_of_lambertian_emitter = emission * pi;
    const float emitter_power_share = radiant_exitance_of_lambertian_emitter / total_radiant_power;
    const float area_to_solid_angle = distance_to_emitter * distance_to_emitter / cosine_at_emitter;
    EXPECT_NEAR(nee_pdf, emitter_power_share * area_to_solid_angle, 1e-6f);
}

// --- evaluateDirectLighting ---

TEST(DirectLightingTest, UnoccludedMatchesRenderingEquation)
{
    const Vec3 surfacePos{0, 0, 0};
    const Vec3 surfaceNormal{0, 0, 1};
    const Vec4 reflectance{0.8f, 0.8f, 0.8f, 0};
    const Vec4 emission{1, 1, 1, 0};
    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};

    const auto result = evaluateDirectLighting(
        surfacePos, surfaceNormal, reflectance, light, emission,
        std::span<const TriangleMeshView>{});

    const float brdf = 0.8f / pi;
    const float geometric = 1.0f * 1.0f / 4.0f;
    const float expected = 1.0f * brdf * geometric / light.pdf;
    test::expectVec4Near(result, {expected, expected, expected, 0}, 1e-6f);
}

TEST(DirectLightingTest, BackFacingSurfaceContributesNothing)
{
    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};

    const auto result = evaluateDirectLighting(
        {0, 0, 0}, {0, 0, -1}, {0.8f, 0.8f, 0.8f, 0}, light, {1, 1, 1, 0},
        std::span<const TriangleMeshView>{});

    test::expectVec4Near(result, {0, 0, 0, 0});
}

TEST(DirectLightingTest, OccluderBlocksContribution)
{
    HostObject occluder = makeHostObject(test::flatTriangleAtZ(1.0f));
    const std::array<TriangleMeshView, 1> objects{occluder.hostView()};

    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};

    const auto result = evaluateDirectLighting(
        {0, 0, 0}, {0, 0, 1}, {0.8f, 0.8f, 0.8f, 0}, light, {1, 1, 1, 0},
        objects);

    test::expectVec4Near(result, {0, 0, 0, 0});
}

// --- samplePointOnLights ---

TEST(SamplePointOnLightsTest, SamplesLieOnEmitterWithCorrectPdf)
{
    constexpr float half = 1.0f;
    constexpr float totalArea = (2 * half) * (2 * half);

    HostObject light = makeHostObject(test::squareMeshXY(half), /*withTriangleSampler=*/true);
    const std::array<TriangleMeshView, 1> objects{light.hostView()};

    const AliasTable objectTable = generateAliasTable(std::vector<float>{1.0f});
    const AliasTableView lightTable = objectTable.hostView();

    DeterministicRng rng;
    for (int i = 0; i < 2000; ++i)
    {
        const auto sample = samplePointOnLights(std::span<const TriangleMeshView>{objects}, lightTable, rng);

        EXPECT_NEAR(sample.p.z, 0.0f, 1e-5f);
        EXPECT_LE(std::abs(sample.p.x), half + 1e-5f);
        EXPECT_LE(std::abs(sample.p.y), half + 1e-5f);
        EXPECT_NEAR(std::abs(sample.n.z), 1.0f, 1e-5f);
        EXPECT_NEAR(sample.pdf, 1.0f / totalArea, 1e-5f);
        EXPECT_EQ(sample.mat, light.material);
    }
}

TEST(SamplePointOnLightsTest, ScaledAndTranslatedEmitterIsSampledInWorldSpace)
{
    constexpr float model_half = 1.0f;
    constexpr float uniform_scale = 3.0f;
    const Vec3 translation{10, 0, 0};

    constexpr float world_half = model_half * uniform_scale;
    constexpr float world_area = (2 * world_half) * (2 * world_half);

    HostObject light = makeHostObject(test::squareMeshXY(model_half), /*withTriangleSampler=*/true);
    light.transform = Transform{.s = {uniform_scale, uniform_scale, uniform_scale}, .p = translation};
    const std::array<TriangleMeshView, 1> objects{light.hostView()};

    const AliasTable objectTable = generateAliasTable(std::vector<float>{1.0f});
    const AliasTableView lightTable = objectTable.hostView();

    DeterministicRng rng;
    for (int i = 0; i < 2000; ++i)
    {
        const auto drawn = samplePointOnLights(std::span<const TriangleMeshView>{objects}, lightTable, rng);

        EXPECT_LE(std::abs(drawn.p.x - translation.x), world_half + 1e-4f);
        EXPECT_LE(std::abs(drawn.p.y - translation.y), world_half + 1e-4f);
        EXPECT_NEAR(drawn.p.z, translation.z, 1e-4f);
        EXPECT_NEAR(drawn.pdf, 1.0f / world_area, 1e-5f);
    }
}
