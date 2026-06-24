// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/DistributionSamplers.hpp"

#include <gtest/gtest.h>

using test::DeterministicRng;

namespace
{
    constexpr int kSamples = 5000;
}

// --- uniformSphereSample ---

TEST(SphereSampleTest, AllSamplesLieOnUnitSphere)
{
    DeterministicRng rng;
    for (int i = 0; i < kSamples; ++i)
        EXPECT_NEAR(uniformSphereSample(rng).length(), 1.0f, 1e-5f);
}

TEST(SphereSampleTest, MeanIsApproximatelyOrigin)
{
    DeterministicRng rng;
    Vec3 sum{0, 0, 0};
    for (int i = 0; i < kSamples; ++i)
        sum += uniformSphereSample(rng);

    const Vec3 mean = sum / static_cast<float>(kSamples);
    EXPECT_NEAR(mean.length(), 0.0f, 0.05f);
}

// --- uniformHemisphereSample ---

TEST(HemisphereSampleTest, StaysInPositiveHemisphere)
{
    DeterministicRng rng;
    const Vec3 normal = Vec3{1, 2, 3}.normalized();
    for (int i = 0; i < kSamples; ++i)
    {
        const auto sample = uniformHemisphereSample(normal, rng);
        EXPECT_GE(sample.dot(normal), 0.0f);
        EXPECT_NEAR(sample.length(), 1.0f, 1e-5f);
    }
}

// --- cosineWeightedHemisphereSample ---

TEST(CosineSampleTest, IsUnitLengthAndInHemisphere)
{
    DeterministicRng rng;
    const Vec3 normal{0, 0, 1};
    for (int i = 0; i < kSamples; ++i)
    {
        const auto sample = cosineWeightedHemisphereSample(normal, rng);
        EXPECT_NEAR(sample.length(), 1.0f, 1e-5f);
        EXPECT_GE(sample.dot(normal), -1e-5f);
    }
}

TEST(CosineSampleTest, FavorsTheNormalDirection)
{
    DeterministicRng rng;
    const Vec3 normal{0, 0, 1};
    double meanCos = 0;
    for (int i = 0; i < kSamples; ++i)
        meanCos += cosineWeightedHemisphereSample(normal, rng).dot(normal);
    meanCos /= kSamples;

    // Cosine-weighted hemisphere has E[cos theta] = 2/3.
    EXPECT_NEAR(meanCos, 2.0 / 3.0, 0.03);
}

// --- uniformTriangleSample ---

TEST(TriangleSampleTest, PointsStayInsideTriangle)
{
    // Triangle A(0,0,0) B(1,0,0) C(0,1,0): sample = (u, v, 0) with u,v >= 0, u+v <= 1.
    DeterministicRng rng;
    for (int i = 0; i < kSamples; ++i)
    {
        const auto p = uniformTriangleSample({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, rng);
        EXPECT_GE(p.x, -1e-6f);
        EXPECT_GE(p.y, -1e-6f);
        EXPECT_LE(p.x + p.y, 1.0f + 1e-6f);
        EXPECT_NEAR(p.z, 0.0f, 1e-6f);
    }
}

TEST(TriangleSampleTest, MeanIsApproximatelyCentroid)
{
    DeterministicRng rng;
    Vec3 sum{0, 0, 0};
    for (int i = 0; i < kSamples; ++i)
        sum += uniformTriangleSample({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, rng);

    const Vec3 mean = sum / static_cast<float>(kSamples);
    test::expectVec3Near(mean, {1.0f / 3.0f, 1.0f / 3.0f, 0.0f}, 0.02f);
}
