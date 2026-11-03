// AI-generated tests (Claude), reviewed by hand before committing.

#include "VectorAssertions.hpp"

#include "Transform.hpp"

#include <gtest/gtest.h>

using test::expectVec3Near;

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
