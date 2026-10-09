// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/Intersection.hpp"
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

// --- NEE / BSDF pdf components used for emission MIS ---
// The wavefront path splits the megakernel's computeNextBounceMisPdfs into a
// stored bsdf pdf (cosineWeightedHemispherePdf, recorded at the sampling vertex)
// and computeNeePdf, evaluated at the emitter. This exercises both.

TEST(NextBounceMisPdfsTest, MatchesAnalyticFormula)
{
    const Material emitter = emissiveMaterial(0.5f);

    const Vec3 vertexPos{0, 0, 0};
    const Vec3 vertexNormal{0, 0, 1};
    const Vec3 nextPos{0, 0, 2};
    const Vec3 nextNormal{0, 0, -1};
    const ObjectsInfo info{.total_radiant_power = 10.0f};

    const float nee_pdf = computeNeePdf(vertexPos, nextPos, nextNormal, emitter, info);
    const float bsdf_pdf = cosineWeightedHemispherePdf(vertexPos, nextPos, vertexNormal);

    const float radiantExitanceLuminance = luminance(radiantExitance(emitter));
    const float expectedNee =
        radiantExitanceLuminance / info.total_radiant_power * areaToSolidAngle(vertexPos, nextPos, nextNormal);

    EXPECT_NEAR(bsdf_pdf, cosineWeightedHemispherePdf(vertexPos, nextPos, vertexNormal), 1e-6f);
    EXPECT_NEAR(nee_pdf, expectedNee, 1e-6f);
    EXPECT_NEAR(bsdf_pdf, 1.0f / pi, 1e-6f);
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
