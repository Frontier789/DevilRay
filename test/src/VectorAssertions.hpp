// AI-generated test helpers (Claude), reviewed by hand before committing.

#pragma once

#include "Utils.hpp"

#include <gtest/gtest.h>

namespace test
{
    inline void expectVec3Near(const Vec3 &actual, const Vec3 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
    }

    inline void expectVec4Near(const Vec4 &actual, const Vec4 &expected, float tol = 1e-5f)
    {
        EXPECT_NEAR(actual.x, expected.x, tol);
        EXPECT_NEAR(actual.y, expected.y, tol);
        EXPECT_NEAR(actual.z, expected.z, tol);
        EXPECT_NEAR(actual.w, expected.w, tol);
    }
}
