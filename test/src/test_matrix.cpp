// Parts of this file are AI-generated tests (Claude), reviewed by hand before committing.

#include "Utils.hpp"
#include "models/Matrix.hpp"

#include <gtest/gtest.h>

// Helper to compare Vec3 components
void ExpectVec3Eq(const Vec3& actual, float x, float y, float z, float abs_error = 1e-5f) {
    EXPECT_NEAR(actual.x, x, abs_error);
    EXPECT_NEAR(actual.y, y, abs_error);
    EXPECT_NEAR(actual.z, z, abs_error);
}

// --- Multiplication Tests ---

TEST(MatrixTest, MultiplicationByIdentity) {
    Matrix4x4f m = Matrix4x4f::translation({1, 2, 3});
    Matrix4x4f result = m * Matrix4x4f::identity();

    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            EXPECT_EQ(result.values[i][j], m.values[i][j]);
}

TEST(MatrixTest, IdentityIsDiagonal) {
    const auto id = Matrix4x4f::identity();

    for(int r = 0; r < 4; ++r)
        for(int c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(id.values[r][c], r == c ? 1.0f : 0.0f);
}

// --- Translation Tests ---

TEST(MatrixTest, TranslationMatrixComposition) {
    Vec3 offset{5.0f, -2.0f, 10.0f};
    auto trans = Matrix4x4f::translation(offset);

    Vec3 pos{0.0f, 0.0f, 0.0f};
    Vec3 result = trans.applyToPosition(pos);

    ExpectVec3Eq(result, 5.0f, -2.0f, 10.0f);
}

TEST(MatrixTest, TranslationDoesNotAffectDirection) {
    auto trans = Matrix4x4f::translation({10, 10, 10});
    Vec3 dir{1.0f, 0.0f, 0.0f};
    Vec3 result = trans.applyToDirection(dir);

    // Directions (vectors) should ignore translation (w=0)
    ExpectVec3Eq(result, 1.0f, 0.0f, 0.0f);
}

// --- Rotation Tests ---

TEST(MatrixTest, Rotation90DegreesZ) {
    // Rotate 90 degrees (PI/2) around Z axis
    float angle = 1.57079632679f;
    auto rot = Matrix4x4f::rotation({0, 0, 1}, angle);

    Vec3 pos{1.0f, 0.0f, 0.0f};
    Vec3 result = rot.applyToPosition(pos);

    // (1,0,0) rotated 90 deg around Z becomes (0,1,0)
    ExpectVec3Eq(result, 0.0f, 1.0f, 0.0f);
}

TEST(MatrixTest, RotationZeroAngle) {
    auto rot = Matrix4x4f::rotation({1, 1, 1}, 0.0f);
    Vec3 pos{1, 2, 3};
    Vec3 result = rot.applyToPosition(pos);

    ExpectVec3Eq(result, 1.0f, 2.0f, 3.0f);
}

TEST(MatrixTest, ApplyToDirectionRotates) {
    auto rot = Matrix4x4f::rotation({0, 0, 1}, pi / 2);

    Vec3 result = rot.applyToDirection({1, 0, 0});

    ExpectVec3Eq(result, 0.0f, 1.0f, 0.0f);
}

// --- Offset Tests ---

TEST(MatrixTest, GetOffsetReturnsTranslationColumn) {
    Vec3 translation = Matrix4x4f::translation({7, -3, 11}).getOffset();
    Vec3 identity_offset = Matrix4x4f::identity().getOffset();

    ExpectVec3Eq(translation, 7.0f, -3.0f, 11.0f);
    ExpectVec3Eq(identity_offset, 0.0f, 0.0f, 0.0f);
}

TEST(MatrixTest, GetOffsetAfterComposition) {
    auto a = Matrix4x4f::translation({1, 2, 3});
    auto b = Matrix4x4f::translation({4, 5, 6});

    ExpectVec3Eq((a * b).getOffset(), 5.0f, 7.0f, 9.0f);
}

// --- Edge Cases & Transformation Application ---

TEST(MatrixTest, ApplyToPositionWithWComponent) {
    // Testing that applyToPosition handles the implicit w=1 correctly
    // Translation + Scale (if you had it) or just Translation
    auto mat = Matrix4x4f::translation({1, 1, 1});
    Vec3 p{2, 2, 2};
    Vec3 result = mat.applyToPosition(p);

    ExpectVec3Eq(result, 3, 3, 3);
}

TEST(MatrixTest, CombinedTransformations) {
    // Translate then Rotate
    auto T = Matrix4x4<float>::translation({1, 0, 0});
    auto R = Matrix4x4<float>::rotation({0, 0, 1}, 1.57079632679f);

    // Standard order: R * T (Translate first, then rotate)
    Matrix4x4<float> combined = R * T;

    Vec3 p = {0, 0, 0};
    Vec3 result = combined.applyToPosition(p);

    // (0,0,0) translated by (1,0,0) -> (1,0,0)
    // (1,0,0) rotated 90deg around Z -> (0,1,0)
    ExpectVec3Eq(result, 0.0f, 1.0f, 0.0f);
}
