// AI-generated tests (Claude), reviewed by hand before committing.

#include "Utils.hpp"
#include "VectorAssertions.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <sstream>
#include <thread>
#include <vector>

using test::expectVec3Near;

// --- Vec3 ---

TEST(Vec3Test, ArithmeticOperators)
{
    constexpr Vec3 a{1, 2, 3};
    constexpr Vec3 b{4, 5, 6};

    expectVec3Near(a + b, {5, 7, 9});
    expectVec3Near(b - a, {3, 3, 3});
    expectVec3Near(a * b, {4, 10, 18});
    expectVec3Near(a * 2.0f, {2, 4, 6});
    expectVec3Near(b / 2.0f, {2, 2.5f, 3});
}

TEST(Vec3Test, CompoundAddition)
{
    Vec3 a{1, 1, 1};
    a += Vec3{2, 3, 4};
    expectVec3Near(a, {3, 4, 5});
}

TEST(Vec3Test, DotProduct)
{
    constexpr Vec3 a{1, 2, 3};
    constexpr Vec3 b{4, 5, 6};

    EXPECT_FLOAT_EQ(a.dot(b), 32.0f);
    EXPECT_FLOAT_EQ((Vec3{1, 0, 0}.dot(Vec3{0, 1, 0})), 0.0f);
}

TEST(Vec3Test, CrossProductFollowsRightHandRule)
{
    expectVec3Near(Vec3{1, 0, 0}.cross(Vec3{0, 1, 0}), {0, 0, 1});
    expectVec3Near(Vec3{0, 1, 0}.cross(Vec3{1, 0, 0}), {0, 0, -1});
}

TEST(Vec3Test, CrossProductOfParallelVectorsIsZero)
{
    expectVec3Near(Vec3{2, 4, 6}.cross(Vec3{1, 2, 3}), {0, 0, 0});
}

TEST(Vec3Test, Length)
{
    EXPECT_FLOAT_EQ(Vec3(3, 4, 0).length(), 5.0f);
    EXPECT_FLOAT_EQ(Vec3(0, 0, 0).length(), 0.0f);
}

TEST(Vec3Test, Normalized)
{
    expectVec3Near(Vec3(0, 5, 0).normalized(), {0, 1, 0});
    const auto n = Vec3(1, 2, 2).normalized();
    EXPECT_NEAR(n.length(), 1.0f, 1e-6f);
}

TEST(Vec3Test, NormalizedZeroVectorProducesNan)
{
    EXPECT_TRUE(Vec3(0, 0, 0).normalized().anyNan());
}

TEST(Vec3Test, Inverse)
{
    expectVec3Near(Vec3(2, 4, 0.5f).inv(), {0.5f, 0.25f, 2.0f});
}

TEST(Vec3Test, AnyNan)
{
    EXPECT_FALSE(Vec3(1, 2, 3).anyNan());
    EXPECT_TRUE(Vec3(NAN, 0, 0).anyNan());
    EXPECT_TRUE(Vec3(0, NAN, 0).anyNan());
    EXPECT_TRUE(Vec3(0, 0, NAN).anyNan());
}

// --- Vec4 ---

TEST(Vec4Test, ArithmeticOperators)
{
    constexpr Vec4 a{1, 2, 3, 4};
    constexpr Vec4 b{5, 6, 7, 8};

    const auto sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 6);
    EXPECT_FLOAT_EQ(sum.w, 12);

    const auto prod = a * b;
    EXPECT_FLOAT_EQ(prod.x, 5);
    EXPECT_FLOAT_EQ(prod.w, 32);

    const auto scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.z, 6);

    const auto divided = b / 2.0f;
    EXPECT_FLOAT_EQ(divided.w, 4);
}

TEST(Vec4Test, MaxIgnoresWComponent)
{
    EXPECT_FLOAT_EQ((Vec4{1, 5, 3, 100}.max()), 5.0f);
    EXPECT_FLOAT_EQ((Vec4{-1, -5, -3, 0}.max()), -1.0f);
}

TEST(Vec4Test, To8bitColorPacksChannelsInRgbaByteOrder)
{
    EXPECT_EQ((Vec4{1, 0, 0, 0}.to8bitColor()), 0x000000FFu); // red   -> byte 0
    EXPECT_EQ((Vec4{0, 1, 0, 0}.to8bitColor()), 0x0000FF00u); // green -> byte 1
    EXPECT_EQ((Vec4{0, 0, 1, 0}.to8bitColor()), 0x00FF0000u); // blue  -> byte 2
    EXPECT_EQ((Vec4{0, 0, 0, 1}.to8bitColor()), 0xFF000000u); // alpha -> byte 3

    EXPECT_EQ((Vec4{0, 0, 0, 0}.to8bitColor()), 0x00000000u);
    EXPECT_EQ((Vec4{1, 1, 1, 1}.to8bitColor()), 0xFFFFFFFFu);
}

TEST(Vec4Test, To8bitColorTruncatesAndClampsOutOfRange)
{
    // Scaling truncates rather than rounds: 0.5 * 255 = 127.5 -> 127.
    EXPECT_EQ((Vec4{0.5f, 0, 0, 0}.to8bitColor()), 127u);

    // Values outside [0, 1] saturate instead of wrapping.
    EXPECT_EQ((Vec4{2.0f, -1.0f, 0, 0}.to8bitColor()), 0x000000FFu);
    EXPECT_EQ((Vec4{-5.0f, 0, 0, 9.0f}.to8bitColor()), 0xFF000000u);
}

TEST(Vec4Test, ConversionsToAndFromVec3)
{
    const auto fromVec3 = Vec4::from(Vec3{1, 2, 3}, 4.0f);
    test::expectVec4Near(fromVec3, {1, 2, 3, 4});
    expectVec3Near(fromVec3.xyz(), {1, 2, 3});
}

TEST(Vec4Test, WithAlphaReplacesOnlyW)
{
    test::expectVec4Near(Vec4{1, 2, 3, 4}.with_alpha(9.0f), {1, 2, 3, 9});
}

TEST(Vec4Test, AbsAppliesToEveryComponentIncludingW)
{
    test::expectVec4Near(Vec4{-1, 2, -3, -4}.abs(), {1, 2, 3, 4});
}

// --- Vec2 ---

TEST(Vec2Test, ArithmeticOperators)
{
    constexpr Vec2f a{1, 2};
    constexpr Vec2f b{4, 8};

    EXPECT_EQ((a + b).x, 5);
    EXPECT_EQ((a + b).y, 10);
    EXPECT_EQ((b - a).x, 3);
    EXPECT_EQ((b - a).y, 6);
    EXPECT_EQ((a * b).x, 4);
    EXPECT_EQ((a * b).y, 16);
    EXPECT_EQ((b / a).x, 4);
    EXPECT_EQ((b / a).y, 4);
    EXPECT_EQ((a * 3.0f).y, 6);
    EXPECT_EQ((b / 2.0f).x, 2);
}

TEST(Vec2Test, LengthAndSquaredLength)
{
    constexpr Vec2f threeFourFive{3, 4};

    EXPECT_FLOAT_EQ(threeFourFive.length(), 5.0f);
    EXPECT_FLOAT_EQ(threeFourFive.length_squared(), 25.0f);
}

TEST(Vec2Test, MapAppliesFunctionToBothComponentsAndMayChangeType)
{
    const auto floored = Vec2f{1.5f, -2.5f}.map([](float v) { return std::floor(v); });
    EXPECT_EQ(floored.x, 1.0f);
    EXPECT_EQ(floored.y, -3.0f);

    const Vec2i truncated = Vec2f{1.5f, -2.5f}.map([](float v) { return static_cast<int>(v); });
    EXPECT_EQ(truncated.x, 1);
    EXPECT_EQ(truncated.y, -2);
}

TEST(Vec2Test, AsConvertsComponentTypeByTruncation)
{
    const auto ints = Vec2f{2.9f, -2.9f}.as<Vec2i>();
    EXPECT_EQ(ints.x, 2);
    EXPECT_EQ(ints.y, -2);

    const auto floats = Vec2i{3, 4}.as<Vec2f>();
    EXPECT_EQ(floats.x, 3.0f);
    EXPECT_EQ(floats.y, 4.0f);
}

TEST(Vec2Test, ImplicitlyConvertsBetweenComponentTypes)
{
    const Vec2f floats = Vec2i{3, 4};
    EXPECT_EQ(floats.x, 3.0f);
    EXPECT_EQ(floats.y, 4.0f);
}

// --- Size2 ---

TEST(Size2Test, AreaAndVectorForm)
{
    constexpr Size2i size{4, 3};

    EXPECT_EQ(size.area(), 12);
    EXPECT_EQ(size.toVec().x, 4);
    EXPECT_EQ(size.toVec().y, 3);
}

TEST(Size2Test, ScalarMultiplicationAndDivisionScaleBothSides)
{
    constexpr Size2i size{8, 6};

    EXPECT_EQ((size * 2).width, 16);
    EXPECT_EQ((size * 2).height, 12);
    EXPECT_EQ((size / 2).width, 4);
    EXPECT_EQ((size / 2).height, 3);
}

TEST(Size2Test, ValidIndexCoversExactlyTheHalfOpenRectangle)
{
    constexpr Size2i size{4, 3};

    EXPECT_TRUE(size.valid_index(Vec2i{0, 0}));
    EXPECT_TRUE(size.valid_index(Vec2i{3, 2}));

    EXPECT_FALSE(size.valid_index(Vec2i{4, 0}));
    EXPECT_FALSE(size.valid_index(Vec2i{0, 3}));
    EXPECT_FALSE(size.valid_index(Vec2i{-1, 0}));
    EXPECT_FALSE(size.valid_index(Vec2i{0, -1}));
}

// --- Stream output ---

TEST(StreamOutputTest, VectorsAndSizesPrintTheirComponents)
{
    const auto printed = [](const auto &value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    };

    EXPECT_EQ(printed(Vec3{1, 2, 3}), "(1, 2, 3)");
    EXPECT_EQ(printed(Vec4{1, 2, 3, 4}), "(1, 2, 3, 4)");
    EXPECT_EQ(printed(Vec2i{5, 6}), "(5, 6)");
    EXPECT_EQ(printed(Size2i{7, 8}), "(7, 8)");
}

// --- Timer ---

TEST(TimerTest, ElapsedTimeCoversAtLeastTheSleptDuration)
{
    using namespace std::chrono_literals;
    constexpr auto slept = 5ms;

    const Timer timer;
    std::this_thread::sleep_for(slept);

    EXPECT_GE(timer.elapsedSeconds(), std::chrono::duration<float>(slept).count());
}

// --- parallel_for ---

namespace
{
    std::vector<int> visitCountsOfParallelFor(int range)
    {
        std::vector<std::atomic<int>> visits(range);
        parallel_for(range, [&](int i) { ++visits[i]; });

        return std::vector<int>(visits.begin(), visits.end());
    }
}

TEST(ParallelForTest, VisitsEveryIndexExactlyOnce)
{
    for (const int range : {1, 3, 1000})
    {
        const auto visits = visitCountsOfParallelFor(range);

        EXPECT_EQ(visits, std::vector<int>(range, 1)) << "range=" << range;
    }
}

TEST(ParallelForTest, EmptyRangeNeverCallsTheFunction)
{
    EXPECT_TRUE(visitCountsOfParallelFor(0).empty());
}

// --- AABB ---

TEST(AABBTest, EmptyBoxIsInverted)
{
    constexpr auto box = AABB::empty();
    EXPECT_GT(box.min.x, box.max.x);
    EXPECT_GT(box.min.y, box.max.y);
    EXPECT_GT(box.min.z, box.max.z);
}

TEST(AABBTest, ExtendingEmptyBoxYieldsSinglePoint)
{
    const auto box = AABB::empty().extend(Vec3{1, 2, 3});
    expectVec3Near(box.min, {1, 2, 3});
    expectVec3Near(box.max, {1, 2, 3});
}

TEST(AABBTest, ExtendGrowsToContainAllPoints)
{
    auto box = AABB::empty();
    box = box.extend(Vec3{-1, 0, 5});
    box = box.extend(Vec3{4, -3, 2});
    box = box.extend(Vec3{0, 0, 0});

    expectVec3Near(box.min, {-1, -3, 0});
    expectVec3Near(box.max, {4, 0, 5});
}

// --- Free functions ---

TEST(GeometryHelpers, TriangleArea)
{
    EXPECT_FLOAT_EQ(triangleArea({0, 0, 0}, {1, 0, 0}, {0, 1, 0}), 0.5f);
    EXPECT_FLOAT_EQ(triangleArea({0, 0, 0}, {2, 0, 0}, {0, 3, 0}), 3.0f);
}

TEST(GeometryHelpers, DegenerateTriangleHasZeroArea)
{
    EXPECT_FLOAT_EQ(triangleArea({0, 0, 0}, {1, 0, 0}, {2, 0, 0}), 0.0f);
}

TEST(GeometryHelpers, LuminanceWeightsSumToOne)
{
    EXPECT_NEAR(luminance(Vec4{1, 1, 1, 0}), 1.0f, 1e-6f);
    EXPECT_FLOAT_EQ(luminance(Vec4{1, 0, 0, 0}), 0.2126f);
    EXPECT_FLOAT_EQ(luminance(Vec4{0, 1, 0, 0}), 0.7152f);
    EXPECT_FLOAT_EQ(luminance(Vec4{0, 0, 1, 0}), 0.0722f);
}

TEST(GeometryHelpers, PowerHeuristic)
{
    EXPECT_FLOAT_EQ(powerHeuristic(1.0f, 1.0f), 0.5f);
    EXPECT_FLOAT_EQ(powerHeuristic(3.0f, 1.0f), 0.9f);
    EXPECT_FLOAT_EQ(powerHeuristic(0.0f, 5.0f), 0.0f);
}

TEST(GeometryHelpers, AreaToSolidAngle)
{
    // Facing patch at distance 2, normal pointing back along the ray.
    EXPECT_FLOAT_EQ(areaToSolidAngle({0, 0, 0}, {0, 0, 2}, {0, 0, -1}), 4.0f);

    // Grazing: patch normal perpendicular to the connecting ray -> degenerate.
    EXPECT_FLOAT_EQ(areaToSolidAngle({0, 0, 0}, {0, 0, 2}, {1, 0, 0}), 0.0f);
}

TEST(GeometryHelpers, CosineWeightedHemispherePdf)
{
    EXPECT_FLOAT_EQ(cosineWeightedHemispherePdf({0, 0, 0}, {0, 0, 1}, {0, 0, 1}), 1.0f / pi);

    const float pdf45 = cosineWeightedHemispherePdf({0, 0, 0}, {1, 0, 1}, {0, 0, 1});
    EXPECT_NEAR(pdf45, std::cos(pi / 4) / pi, 1e-6f);
}

TEST(GeometryHelpers, CosineWeightedHemispherePdfIsZeroBelowHorizon)
{
    EXPECT_FLOAT_EQ(cosineWeightedHemispherePdf({0, 0, 0}, {0, 0, -1}, {0, 0, 1}), 0.0f);
}

// --- Spherical angles ---

namespace
{
    Vec3 directionFromSphericalAngles(float azimuth, float elevation)
    {
        return Vec3{
            std::cos(elevation) * std::sin(azimuth),
            std::sin(elevation),
            std::cos(elevation) * std::cos(azimuth),
        };
    }
}

TEST(SphericalAnglesTest, AxisAndDiagonalDirectionsLandOnTheExpectedAngles)
{
    struct Case
    {
        Vec3 direction;
        float azimuth;
        float elevation;
    };

    const Case cases[] = {
        {{0, 0, 1}, 0.0f, 0.0f},
        {{1, 0, 0}, pi / 2, 0.0f},
        {{0, 0, -1}, pi, 0.0f},
        {{-1, 0, 0}, 1.5f * pi, 0.0f},
        {Vec3{1, 1, 0}.normalized(), pi / 2, pi / 4},
        {Vec3{0, -1, 1}.normalized(), 0.0f, -pi / 4},
    };

    for (const auto &[direction, azimuth, elevation] : cases)
    {
        const auto angles = directionToSphericalAngles(direction);

        SCOPED_TRACE(testing::Message() << "direction=" << direction);
        EXPECT_NEAR(angles.azimuth, azimuth, 1e-6f);
        EXPECT_NEAR(angles.elevation, elevation, 1e-6f);
    }
}

TEST(SphericalAnglesTest, PolesHaveAnExactElevationAndAFiniteAzimuth)
{
    for (const float zero : {0.0f, -0.0f})
    {
        const auto north = directionToSphericalAngles(Vec3{zero, 1, zero});
        const auto south = directionToSphericalAngles(Vec3{zero, -1, zero});

        EXPECT_FLOAT_EQ(north.elevation, pi / 2);
        EXPECT_FLOAT_EQ(south.elevation, -pi / 2);

        for (const auto &angles : {north, south})
        {
            EXPECT_TRUE(std::isfinite(angles.azimuth));
            EXPECT_GE(angles.azimuth, 0.0f);
            EXPECT_LT(angles.azimuth, 2 * pi);
        }
    }
}

TEST(SphericalAnglesTest, RoundTripsThroughTheInverseParameterization)
{
    constexpr int elevation_steps = 16;
    constexpr int azimuth_steps = 32;

    for (int i = 1; i < elevation_steps; ++i)
    {
        for (int j = 0; j < azimuth_steps; ++j)
        {
            const float elevation = pi * i / elevation_steps - pi / 2;
            const float azimuth = 2 * pi * j / azimuth_steps;

            const auto angles = directionToSphericalAngles(directionFromSphericalAngles(azimuth, elevation));

            SCOPED_TRACE(testing::Message() << "azimuth=" << azimuth << " elevation=" << elevation);
            EXPECT_NEAR(angles.azimuth, azimuth, 1e-5f);
            EXPECT_NEAR(angles.elevation, elevation, 1e-5f);
        }
    }
}

TEST(SphericalAnglesTest, AnglesStayInTheirRangesForAllSignAndZeroCombinations)
{
    const float values[] = {-1.0f, -0.5f, -0.1f, -0.0f, 0.0f, 0.1f, 0.5f, 1.0f};

    for (const float x : values)
    {
        for (const float y : values)
        {
            for (const float z : values)
            {
                if (x == 0.0f && y == 0.0f && z == 0.0f) continue;

                const Vec3 direction{x, y, z};
                const auto angles = directionToSphericalAngles(direction);

                SCOPED_TRACE(testing::Message() << "direction=" << direction);
                EXPECT_GE(angles.azimuth, 0.0f);
                EXPECT_LT(angles.azimuth, 2 * pi);
                EXPECT_GE(angles.elevation, -pi / 2);
                EXPECT_LE(angles.elevation, pi / 2);
            }
        }
    }
}

TEST(SphericalAnglesTest, TinyNegativeXFoldsToZeroAzimuthInsteadOfAFullTurn)
{
    const auto angles = directionToSphericalAngles(Vec3{-1e-9f, 0, 1});

    EXPECT_GE(angles.azimuth, 0.0f);
    EXPECT_LT(angles.azimuth, 2 * pi);
    EXPECT_NEAR(angles.azimuth, 0.0f, 1e-6f);
}

TEST(SphericalAnglesTest, ResolvesElevationsTooCloseToThePoleForAsin)
{
    const auto north = directionToSphericalAngles(Vec3{1e-4f, 1.0f, 0}.normalized());
    const auto south = directionToSphericalAngles(Vec3{1e-4f, -1.0f, 0}.normalized());

    EXPECT_NEAR(north.elevation, pi / 2 - 1e-4f, 5e-7f);
    EXPECT_NEAR(south.elevation, -pi / 2 + 1e-4f, 5e-7f);
}

TEST(SphericalAnglesTest, IsScaleInvariant)
{
    const Vec3 d{0.3f, -0.5f, 0.8f};

    const auto unit = directionToSphericalAngles(d.normalized());
    const auto scaled = directionToSphericalAngles(d * 7.0f);

    EXPECT_NEAR(scaled.azimuth, unit.azimuth, 1e-6f);
    EXPECT_NEAR(scaled.elevation, unit.elevation, 1e-6f);
}
