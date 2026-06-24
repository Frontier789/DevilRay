// AI-generated tests (Claude), reviewed by hand before committing.

#include "Transform.hpp"
#include "models/Matrix.hpp"

#include <gtest/gtest.h>

namespace
{
    void expectVec3Near(const Vec3 &actual, const Vec3 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
    }
}

// --- Transform ---

TEST(TransformTest, ApplyToPointScalesThenTranslates)
{
    constexpr Transform t{.s = {2, 3, 4}, .p = {10, 20, 30}};
    expectVec3Near(t.applyToPoint({1, 1, 1}), {12, 23, 34});
    expectVec3Near(t.applyToPoint({0, 0, 0}), {10, 20, 30});
}

TEST(TransformTest, ApplyToPointIdentityIsNoOp)
{
    constexpr Transform t{};
    expectVec3Near(t.applyToPoint({5, -2, 7}), {5, -2, 7});
}

TEST(TransformTest, ApplyInverseUndoesPointTransformForRayOrigin)
{
    constexpr Transform t{.s = {2, 4, 5}, .p = {1, 2, 3}};
    const Vec3 worldPoint{9, 10, 13};

    const Ray worldRay{.p = worldPoint, .v = {0, 0, 1}};
    const Ray modelRay = t.applyInverse(worldRay);

    // Inverse of origin must round-trip back through applyToPoint.
    expectVec3Near(t.applyToPoint(modelRay.p), worldPoint);
    expectVec3Near(modelRay.p, {4, 2, 2});
}

TEST(TransformTest, ApplyInverseScalesDirectionWithoutTranslation)
{
    constexpr Transform t{.s = {2, 4, 5}, .p = {100, 200, 300}};
    const Ray modelRay = t.applyInverse(Ray{.p = {0, 0, 0}, .v = {2, 4, 5}});

    // Direction ignores translation and is divided by the scale.
    expectVec3Near(modelRay.v, {1, 1, 1});
}

// --- Matrix4x4 pieces not covered by test_matrix ---

TEST(MatrixTest, IdentityIsDiagonal)
{
    const auto id = Matrix4x4f::identity();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(id.values[r][c], r == c ? 1.0f : 0.0f);
}

TEST(MatrixTest, ApplyToDirectionIgnoresTranslation)
{
    const auto m = Matrix4x4f::translation({10, 20, 30});
    expectVec3Near(m.applyToDirection({1, 2, 3}), {1, 2, 3});
}

TEST(MatrixTest, ApplyToDirectionRotates)
{
    const auto rot = Matrix4x4f::rotation({0, 0, 1}, pi / 2);
    expectVec3Near(rot.applyToDirection({1, 0, 0}), {0, 1, 0});
}

TEST(MatrixTest, GetOffsetReturnsTranslationColumn)
{
    expectVec3Near(Matrix4x4f::translation({7, -3, 11}).getOffset(), {7, -3, 11});
    expectVec3Near(Matrix4x4f::identity().getOffset(), {0, 0, 0});
}

TEST(MatrixTest, GetOffsetAfterComposition)
{
    const auto a = Matrix4x4f::translation({1, 2, 3});
    const auto b = Matrix4x4f::translation({4, 5, 6});
    expectVec3Near((a * b).getOffset(), {5, 7, 9});
}
