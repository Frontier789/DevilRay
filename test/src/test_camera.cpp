// AI-generated tests (Claude), reviewed by hand before committing.

#include "TracingTestHelpers.hpp"

#include "tracing/Camera.hpp"
#include "tracing/CameraRay.hpp"
#include "tracing/PixelSampling.hpp"

#include <gtest/gtest.h>

using test::DeterministicRng;
using test::ScriptedRng;

namespace
{
    // Principal point placed so that pixel (50,50) sits exactly on the optical axis.
    Camera axisCamera(const Matrix4x4f &transform)
    {
        constexpr float pixelSize = 0.1f;
        return Camera{
            .transform = transform,
            .intrinsics = Intrinsics{.focal_length = 1.0f, .center = Vec2f{50.5f * pixelSize, 50.5f * pixelSize}},
            .resolution = Size2i{100, 100},
            .physical_pixel_size = Size2f{pixelSize, pixelSize},
        };
    }
}

TEST(CameraRayTest, CenterPixelLooksDownOpticalAxis)
{
    const Camera cam = axisCamera(Matrix4x4f::identity());
    DeterministicRng rng;

    const Ray ray = cameraRay(cam, Vec2f{50, 50}, PixelSampling::Center, 0, rng);
    test::expectVec3Near(ray.p, {0, 0, 0});
    test::expectVec3Near(ray.v, {0, 0, 1}, 1e-5f);
}

TEST(CameraRayTest, OriginComesFromTransformOffset)
{
    const Camera cam = axisCamera(Matrix4x4f::translation({1, 2, 3}));
    DeterministicRng rng;

    const Ray ray = cameraRay(cam, Vec2f{50, 50}, PixelSampling::Center, 0, rng);
    test::expectVec3Near(ray.p, {1, 2, 3});
    // Pure translation must not rotate the view direction.
    test::expectVec3Near(ray.v, {0, 0, 1}, 1e-5f);
}

TEST(CameraRayTest, OffAxisPixelsPointOutward)
{
    const Camera cam = axisCamera(Matrix4x4f::identity());
    DeterministicRng rng;

    const Ray right = cameraRay(cam, Vec2f{80, 50}, PixelSampling::Center, 0, rng);
    EXPECT_GT(right.v.x, 0.0f);
    EXPECT_NEAR(right.v.y, 0.0f, 1e-5f);

    const Ray up = cameraRay(cam, Vec2f{50, 20}, PixelSampling::Center, 0, rng);
    EXPECT_LT(up.v.y, 0.0f);
    EXPECT_NEAR(up.v.x, 0.0f, 1e-5f);
}

TEST(CameraRayTest, DirectionIsNormalized)
{
    const Camera cam = axisCamera(Matrix4x4f::identity());
    DeterministicRng rng;

    const Ray ray = cameraRay(cam, Vec2f{10, 90}, PixelSampling::Center, 0, rng);
    EXPECT_NEAR(ray.v.length(), 1.0f, 1e-5f);
}

TEST(CameraRayTest, TransformRotatesViewDirection)
{
    const auto rotation = Matrix4x4f::rotation({0, 1, 0}, pi / 2);
    const Camera cam = axisCamera(rotation);
    DeterministicRng rng;

    const Ray ray = cameraRay(cam, Vec2f{50, 50}, PixelSampling::Center, 0, rng);
    const Vec3 expected = rotation.applyToDirection(Vec3{0, 0, 1});
    test::expectVec3Near(ray.v, expected, 1e-5f);
}

TEST(CameraRayTest, UniformRandomAtHalfMatchesCenter)
{
    const Camera cam = axisCamera(Matrix4x4f::identity());

    DeterministicRng centerRng;
    const Ray center = cameraRay(cam, Vec2f{37, 64}, PixelSampling::Center, 0, centerRng);

    ScriptedRng halfRng{.values = {0.5f, 0.5f}};
    const Ray random = cameraRay(cam, Vec2f{37, 64}, PixelSampling::UniformRandom, 0, halfRng);

    test::expectVec3Near(random.v, center.v, 1e-6f);
}
