// AI-generated tests (Claude), reviewed by hand before committing.

#include "CameraController.hpp"
#include "Utils.hpp"
#include "VectorAssertions.hpp"
#include "models/Matrix.hpp"
#include "tracing/Camera.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using test::expectVec3Near;

namespace
{
    constexpr float focal_length = 0.05f;
    constexpr float pixel_size = 0.001f;
    constexpr Vec3 world_up{0, 1, 0};

    CameraController controllerLookingAt(Vec3 target, float distance, float pitch = 0.0f, float yaw = 0.0f)
    {
        return CameraController{
            .camera = Camera{
                .transform = Matrix4x4f::identity(),
                .intrinsics = Intrinsics{.focal_length = focal_length, .center = Vec2f{0, 0}},
                .resolution = Size2i{100, 100},
                .physical_pixel_size = Size2f{pixel_size, pixel_size},
            },
            .target = target,
            .pitch = pitch,
            .yaw = yaw,
            .distance = distance,
        };
    }

    struct Orientation
    {
        float pitch;
        float yaw;
    };

    const std::vector<Orientation> orientationsAroundTheSphere = {
        {0.0f, 0.0f},
        {0.3f, 1.0f},
        {-0.7f, 2.5f},
        {1.4f, -1.0f},
        {2.0f, 0.5f},   // upside down
        {-2.5f, -2.0f}, // upside down
    };
}

// --- Orientation vectors ---

TEST(CameraControllerTest, ForwardAtZeroAnglesLooksDownPlusZ)
{
    expectVec3Near(controllerLookingAt({0, 0, 0}, 1).forward(), {0, 0, 1});
}

TEST(CameraControllerTest, YawTurnsForwardAroundTheVerticalAxis)
{
    expectVec3Near(controllerLookingAt({0, 0, 0}, 1, 0.0f, pi / 2).forward(), {1, 0, 0});
}

TEST(CameraControllerTest, PitchTiltsForwardTowardPlusY)
{
    const auto forward = controllerLookingAt({0, 0, 0}, 1, pi / 6).forward();

    expectVec3Near(forward, {0, std::sin(pi / 6), std::cos(pi / 6)});
}

TEST(CameraControllerTest, ForwardAzimuthAndElevationAreYawAndPitch)
{
    for (const auto &[pitch, yaw] : orientationsAroundTheSphere)
    {
        const auto controller = controllerLookingAt({0, 0, 0}, 1, pitch, yaw);

        if (controller.isUpsideDown()) continue;

        const auto angles = directionToSphericalAngles(controller.forward());

        SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " yaw=" << yaw);
        EXPECT_NEAR(angles.azimuth, yaw < 0 ? yaw + 2 * pi : yaw, 1e-5f);
        EXPECT_NEAR(angles.elevation, pitch, 1e-5f);
    }
}

TEST(CameraControllerTest, RightUpForwardFormARightHandedOrthonormalBasis)
{
    for (const auto &[pitch, yaw] : orientationsAroundTheSphere)
    {
        const auto controller = controllerLookingAt({0, 0, 0}, 1, pitch, yaw);
        const auto right = controller.right();
        const auto up = controller.up();
        const auto forward = controller.forward();

        SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " yaw=" << yaw);
        EXPECT_NEAR(right.length(), 1.0f, 1e-5f);
        EXPECT_NEAR(up.length(), 1.0f, 1e-5f);
        EXPECT_NEAR(forward.length(), 1.0f, 1e-5f);
        EXPECT_NEAR(right.dot(up), 0.0f, 1e-5f);
        EXPECT_NEAR(right.dot(forward), 0.0f, 1e-5f);
        EXPECT_NEAR(up.dot(forward), 0.0f, 1e-5f);
        expectVec3Near(right.cross(up), forward);
    }
}

TEST(CameraControllerTest, UpHasPositiveWorldUpComponentUntilTheCameraFlipsOver)
{
    const auto upright = controllerLookingAt({0, 0, 0}, 1, 0.4f * pi, 0.7f);
    const auto flipped = controllerLookingAt({0, 0, 0}, 1, 0.6f * pi, 0.7f);

    EXPECT_GT(upright.up().dot(world_up), 0.0f);
    EXPECT_LT(flipped.up().dot(world_up), 0.0f);
}

TEST(CameraControllerTest, IsUpsideDownOnlyBeyondTheZenithInEitherDirection)
{
    EXPECT_FALSE(controllerLookingAt({0, 0, 0}, 1, 0.0f).isUpsideDown());
    EXPECT_FALSE(controllerLookingAt({0, 0, 0}, 1, 0.49f * pi).isUpsideDown());
    EXPECT_FALSE(controllerLookingAt({0, 0, 0}, 1, -0.49f * pi).isUpsideDown());

    EXPECT_TRUE(controllerLookingAt({0, 0, 0}, 1, 0.51f * pi).isUpsideDown());
    EXPECT_TRUE(controllerLookingAt({0, 0, 0}, 1, -0.51f * pi).isUpsideDown());
}

TEST(CameraControllerTest, RightVectorIsContinuousAcrossTheZenith)
{
    const auto just_before = controllerLookingAt({0, 0, 0}, 1, 0.49f * pi, 0.8f).right();
    const auto just_after = controllerLookingAt({0, 0, 0}, 1, 0.51f * pi, 0.8f).right();

    expectVec3Near(just_before, just_after, 0.05f);
}

// --- Position ---

TEST(CameraControllerTest, PositionIsTheDistanceBehindTheTargetAlongForward)
{
    for (const auto &[pitch, yaw] : orientationsAroundTheSphere)
    {
        const Vec3 target{1, -2, 3};
        constexpr float distance = 4.0f;
        const auto controller = controllerLookingAt(target, distance, pitch, yaw);

        SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " yaw=" << yaw);
        EXPECT_NEAR((target - controller.position()).length(), distance, 1e-5f);
        expectVec3Near((target - controller.position()).normalized(), controller.forward());
    }
}

// --- Matrices ---

TEST(CameraControllerTest, TransformHasTheBasisAsRotationAndThePositionAsOffset)
{
    for (const auto &[pitch, yaw] : orientationsAroundTheSphere)
    {
        const auto controller = controllerLookingAt({1, -2, 3}, 4.0f, pitch, yaw);
        const auto transform = controller.calculateTransform();

        SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " yaw=" << yaw);
        expectVec3Near(transform.getOffset(), controller.position());
        expectVec3Near(transform.applyToDirection({1, 0, 0}), controller.right());
        expectVec3Near(transform.applyToDirection({0, 1, 0}), controller.up());
        expectVec3Near(transform.applyToDirection({0, 0, 1}), controller.forward());
    }
}

TEST(CameraControllerTest, ViewMatrixSendsTheCameraPositionToTheOrigin)
{
    for (const auto &[pitch, yaw] : orientationsAroundTheSphere)
    {
        const auto controller = controllerLookingAt({1, -2, 3}, 4.0f, pitch, yaw);

        SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " yaw=" << yaw);
        expectVec3Near(controller.getViewMatrix().applyToPosition(controller.position()), {0, 0, 0}, 1e-4f);
    }
}

TEST(CameraControllerTest, ViewMatrixUndoesTransformExceptForOpenGlLookingDownMinusZ)
{
    const auto controller = controllerLookingAt({1, -2, 3}, 4.0f, 0.3f, 1.0f);
    const Vec3 point_in_camera_space{1, 2, 3};
    const Vec3 same_point_in_gl_view_space{1, 2, -3};

    const auto point_in_world = controller.calculateTransform().applyToPosition(point_in_camera_space);

    expectVec3Near(controller.getViewMatrix().applyToPosition(point_in_world), same_point_in_gl_view_space, 1e-4f);
}

TEST(CameraControllerTest, GetCameraCarriesTheCurrentTransform)
{
    const auto controller = controllerLookingAt({1, -2, 3}, 4.0f, 0.3f, 1.0f);

    const Camera camera = controller.getCamera();

    expectVec3Near(camera.transform.getOffset(), controller.position());
    EXPECT_EQ(camera.resolution.width, controller.camera.resolution.width);
    EXPECT_EQ(camera.intrinsics.focal_length, focal_length);
}

// --- Input handling ---

TEST(CameraControllerTest, ScrollingForwardZoomsInAndBackwardZoomsOut)
{
    auto controller = controllerLookingAt({0, 0, 0}, 2.0f);

    controller.handleScroll(1.0f);
    EXPECT_LT(controller.distance, 2.0f);

    controller.handleScroll(-2.0f);
    EXPECT_GT(controller.distance, 2.0f);
}

TEST(CameraControllerTest, ScrollingByTheSameAmountInOppositeDirectionsRestoresTheDistance)
{
    auto controller = controllerLookingAt({0, 0, 0}, 2.0f);

    controller.handleScroll(3.0f);
    controller.handleScroll(-3.0f);

    EXPECT_NEAR(controller.distance, 2.0f, 1e-5f);
}

TEST(CameraControllerTest, ScrollNotchesComposeMultiplicatively)
{
    auto twoSteps = controllerLookingAt({0, 0, 0}, 2.0f);
    twoSteps.handleScroll(1.0f);
    twoSteps.handleScroll(1.0f);

    auto oneDoubleStep = controllerLookingAt({0, 0, 0}, 2.0f);
    oneDoubleStep.handleScroll(2.0f);

    EXPECT_NEAR(twoSteps.distance, oneDoubleStep.distance, 1e-5f);
}

TEST(CameraControllerTest, DraggingRightTurnsYawAndDraggingDownLowersPitch)
{
    auto controller = controllerLookingAt({0, 0, 0}, 1.0f);
    constexpr Vec2f unit_sensitivity{1.0f, 1.0f};

    controller.handleRotate(Vec2f{0.1f, 0.0f}, unit_sensitivity);
    EXPECT_NEAR(controller.yaw, 0.1f, 1e-6f);
    EXPECT_NEAR(controller.pitch, 0.0f, 1e-6f);

    controller.handleRotate(Vec2f{0.0f, 0.2f}, unit_sensitivity);
    EXPECT_NEAR(controller.yaw, 0.1f, 1e-6f);
    EXPECT_NEAR(controller.pitch, -0.2f, 1e-6f);
}

TEST(CameraControllerTest, HorizontalDragTurnsTheOtherWayWhileUpsideDown)
{
    auto controller = controllerLookingAt({0, 0, 0}, 1.0f, 0.75f * pi);
    ASSERT_TRUE(controller.isUpsideDown());

    controller.handleRotate(Vec2f{0.1f, 0.0f}, Vec2f{1.0f, 1.0f});

    EXPECT_NEAR(controller.yaw, -0.1f, 1e-6f);
}

TEST(CameraControllerTest, PitchWrapsBackWhenItPassesOneAndAHalfPi)
{
    constexpr float margin = 0.1f;
    auto controller = controllerLookingAt({0, 0, 0}, 1.0f, 1.5f * pi - margin);

    controller.handleRotate(Vec2f{0.0f, -0.3f}, Vec2f{1.0f, 1.0f});

    EXPECT_NEAR(controller.pitch, 1.5f * pi - margin + 0.3f - 2 * pi, 1e-5f);
}

TEST(CameraControllerTest, DraggingRightMovesTheTargetLeftSoTheSceneFollowsTheCursor)
{
    auto controller = controllerLookingAt({0, 0, 0}, 2.0f, 0.2f, 0.5f);
    const auto original_target = controller.target;
    constexpr float dragged_pixels = 10.0f;

    controller.handleDrag(Vec2f{dragged_pixels, 0.0f});

    const auto moved = controller.target - original_target;
    const float expected_distance_on_target_plane = dragged_pixels * pixel_size * controller.distance / focal_length;
    expectVec3Near(moved, controller.right() * -expected_distance_on_target_plane);
}

TEST(CameraControllerTest, DraggingDownMovesTheTargetUp)
{
    auto controller = controllerLookingAt({0, 0, 0}, 2.0f, 0.2f, 0.5f);
    const auto original_target = controller.target;
    constexpr float dragged_pixels = 10.0f;

    controller.handleDrag(Vec2f{0.0f, dragged_pixels});

    const auto moved = controller.target - original_target;
    const float expected_distance_on_target_plane = dragged_pixels * pixel_size * controller.distance / focal_length;
    expectVec3Near(moved, controller.up() * expected_distance_on_target_plane);
}

TEST(CameraControllerTest, DraggingMovesTheTargetFartherWhenTheCameraIsFartherAway)
{
    auto near_camera = controllerLookingAt({0, 0, 0}, 1.0f);
    auto far_camera = controllerLookingAt({0, 0, 0}, 3.0f);

    near_camera.handleDrag(Vec2f{10.0f, 0.0f});
    far_camera.handleDrag(Vec2f{10.0f, 0.0f});

    EXPECT_NEAR(far_camera.target.length(), 3.0f * near_camera.target.length(), 1e-5f);
}
