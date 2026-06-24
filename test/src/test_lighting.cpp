// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/Camera.hpp"
#include "tracing/PathGeneration.hpp"
#include "tracing/LightSampling.hpp"

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

// --- computeNextBounceMisPdfs ---

TEST(NextBounceMisPdfsTest, MatchesAnalyticFormula)
{
    std::vector<Material> materials;
    materials.push_back(emissiveMaterial(0.5f));

    const PathEntry vertex{.p = {0, 0, 0}, .uv = {0, 0}, .n = {0, 0, 1}, .mat = 0, .total_throughput = {1, 1, 1, 0}, .triangle_area = 1};
    const PathEntry next{.p = {0, 0, 2}, .uv = {0, 0}, .n = {0, 0, -1}, .mat = 0, .total_throughput = {1, 1, 1, 0}, .triangle_area = 1};
    const ObjectsInfo info{.total_radiant_power = 10.0f};

    const auto pdfs = computeNextBounceMisPdfs(vertex, next, std::span<const Material>{materials}, info);

    const float expectedBsdf = cosineWeightedHemispherePdf(vertex.p, next.p, vertex.n);
    const float radiantExitanceLuminance = luminance(radiantExitance(materials[0]));
    const float expectedNee =
        radiantExitanceLuminance / info.total_radiant_power * areaToSolidAngle(vertex.p, next.p, next.n);

    EXPECT_NEAR(pdfs.bsdf_pdf, expectedBsdf, 1e-6f);
    EXPECT_NEAR(pdfs.nee_pdf, expectedNee, 1e-6f);
    EXPECT_NEAR(pdfs.bsdf_pdf, 1.0f / pi, 1e-6f);
}

// --- evaluateDirectLighting ---

TEST(DirectLightingTest, UnoccludedMatchesRenderingEquation)
{
    const Vec3 surfacePos{0, 0, 0};
    const Vec3 surfaceNormal{0, 0, 1};
    const Vec4 reflectance{0.8f, 0.8f, 0.8f, 0};
    const Vec4 emission{1, 1, 1, 0};
    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};
    const ObjectsInfo info{.total_radiant_power = 1.0f};

    const auto result = evaluateDirectLighting(
        surfacePos, surfaceNormal, reflectance, light, emission,
        std::span<const TriangleMesh>{}, info);

    const float brdf = 0.8f / pi;
    const float geometric = 1.0f * 1.0f / 4.0f;
    const float expected = 1.0f * brdf * geometric / light.pdf;
    test::expectVec4Near(result, {expected, expected, expected, 0}, 1e-6f);
}

TEST(DirectLightingTest, BackFacingSurfaceContributesNothing)
{
    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};
    const ObjectsInfo info{.total_radiant_power = 1.0f};

    const auto result = evaluateDirectLighting(
        {0, 0, 0}, {0, 0, -1}, {0.8f, 0.8f, 0.8f, 0}, light, {1, 1, 1, 0},
        std::span<const TriangleMesh>{}, info);

    test::expectVec4Near(result, {0, 0, 0, 0});
}

TEST(DirectLightingTest, OccluderBlocksContribution)
{
    HostObject occluder = makeHostObject(test::flatTriangleAtZ(1.0f));
    const std::array<TriangleMesh, 1> objects{occluder.view()};

    const LightSample light{.p = {0, 0, 2}, .n = {0, 0, -1}, .mat = 0, .pdf = 0.5f};
    const ObjectsInfo info{.total_radiant_power = 1.0f};

    const auto result = evaluateDirectLighting(
        {0, 0, 0}, {0, 0, 1}, {0.8f, 0.8f, 0.8f, 0}, light, {1, 1, 1, 0},
        objects, info);

    test::expectVec4Near(result, {0, 0, 0, 0});
}

// --- samplePointOnLights ---

TEST(SamplePointOnLightsTest, SamplesLieOnEmitterWithCorrectPdf)
{
    constexpr float half = 1.0f;
    constexpr float totalArea = (2 * half) * (2 * half);

    HostObject light = makeHostObject(test::squareMeshXY(half), /*withTriangleSampler=*/true);
    const std::array<TriangleMesh, 1> objects{light.view()};

    const AliasTable objectTable = generateAliasTable(std::vector<float>{1.0f});
    const std::span<const AliasEntry> lightTable{objectTable.entries.hostPtr(), objectTable.entries.size()};

    DeterministicRng rng;
    for (int i = 0; i < 2000; ++i)
    {
        const auto sample = samplePointOnLights(std::span<const TriangleMesh>{objects}, lightTable, rng);

        EXPECT_NEAR(sample.p.z, 0.0f, 1e-5f);
        EXPECT_LE(std::abs(sample.p.x), half + 1e-5f);
        EXPECT_LE(std::abs(sample.p.y), half + 1e-5f);
        EXPECT_NEAR(std::abs(sample.n.z), 1.0f, 1e-5f);
        EXPECT_NEAR(sample.pdf, 1.0f / totalArea, 1e-5f);
        EXPECT_EQ(sample.mat, light.material);
    }
}
