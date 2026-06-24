// AI-generated tests (Claude), reviewed by hand before committing.

#include "Utils.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{
    void expectVec3Near(const Vec3 &actual, const Vec3 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
    }
}

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
    EXPECT_FLOAT_EQ(dot(a, b), 32.0f);
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
