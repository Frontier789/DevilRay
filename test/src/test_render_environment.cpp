// AI-generated tests (Claude), reviewed by hand before committing.

#include "Image.hpp"
#include "RenderTestHelpers.hpp"
#include "Renderer.hpp"
#include "Utils.hpp"
#include "tracing/Scene.hpp"

#include <gtest/gtest.h>

#include <utility>

using namespace render_test;

namespace
{
    constexpr Size2i tiny{4, 4};
    const Vec3 origin{0, 0, 0};

    Scene sceneWithEnvironmentOnly(Image4f panorama, float intensity_multiplier = 1.0f)
    {
        Scene scene;
        const int material = addDiffuse(scene, Vec4{0.5f, 0.5f, 0.5f, 0});
        addTinyQuadFarFromEverything(scene, material);
        setEnvironment(scene, std::move(panorama), intensity_multiplier);

        return scene;
    }

    Image4f sunnyPanorama()
    {
        constexpr Size2i size{64, 32};
        constexpr int first_sun_row = 5;
        constexpr int end_sun_row = 11;
        constexpr int first_sun_column = 8;
        constexpr int end_sun_column = 18;

        auto panorama = uniformPanorama(size, Vec3{0.1f, 0.15f, 0.3f});

        for (int row = first_sun_row; row < end_sun_row; ++row)
            for (int column = first_sun_column; column < end_sun_column; ++column)
                panorama[Vec2i{column, row}] = Vec4{8.0f, 7.0f, 5.0f, 1.0f};

        return panorama;
    }

    void expectGlassBallOnAFloorRendersFinitePixels(bool with_distant_emitter)
    {
        constexpr Size2i resolution{8, 8};
        constexpr int frames = 4;

        Renderer renderer(resolution);
        useDeterministicLinearOutput(renderer);
        renderer.setCamera(cameraFacing(resolution, Vec3{0, 1, 0}, 0.0f, -0.3f, 0.5f));

        Scene scene;
        const int floor = addDiffuse(scene, Vec4{0.5f, 0.5f, 0.5f, 0});
        const int glass = addGlass(scene, 1.5f);
        addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, floor);
        addBall(scene, Vec3{0, 0.5f, 3}, 0.5f, glass);

        if (with_distant_emitter)
        {
            const int dim_lamp = addDiffuse(scene, Vec4{0, 0, 0, 0}, Vec4{1e-3f, 1e-3f, 1e-3f, 0});
            addQuad(scene, Vec3{0, 40, 0}, Vec3{0, -1, 0}, Vec3{1, 0, 0}, 0.1f, dim_lamp);
        }

        setEnvironment(scene, gradientPanorama(Size2i{32, 16}));
        renderer.setScene(std::move(scene));

        renderFrames(renderer, frames);

        expectEveryPixelFiniteWithSampleCount(accumulatedPixels(renderer, resolution), frames);
    }
}

TEST(EnvironmentBackdropTest, UniformMapAppearsAtItsExactRadianceAndDoesNotLeakIntoTheSampleCount)
{
    Renderer renderer(tiny);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(tiny, origin, 0.0f, 0.0f));
    renderer.setScene(sceneWithEnvironmentOnly(uniformPanorama(Size2i{8, 4}, Vec3{0.5f, 0.25f, 2.0f})));

    renderFrames(renderer, 3);

    const auto pixels = accumulatedPixels(renderer, tiny);
    expectEveryPixelFiniteWithSampleCount(pixels, 3);
    expectEveryPixelHasMeanRadiance(pixels, Channels{0.5, 0.25, 2.0}, 1e-6);
}

TEST(EnvironmentBackdropTest, LooksUpThePanoramaAlongTheCardinalDirections)
{
    constexpr Size2i resolution{5, 5};
    constexpr int center_pixel = 2 + 2 * resolution.width;
    constexpr double tolerance = 0.01;

    struct Direction
    {
        const char *name;
        float yaw;
        float pitch;
        double expected_u;
        double expected_v;
    };

    const Direction directions[] = {
        {"+x", float(pi / 2), 0.0f, 0.25, 0.5},
        {"-z", float(pi), 0.0f, 0.5, 0.5},
        {"-x", float(3 * pi / 2), 0.0f, 0.75, 0.5},
        {"45 degrees up", float(pi), float(pi / 4), 0.5, 0.25},
        {"45 degrees down", float(pi), float(-pi / 4), 0.5, 0.75},
    };

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setScene(sceneWithEnvironmentOnly(gradientPanorama(Size2i{256, 128})));

    for (const auto &direction : directions)
    {
        SCOPED_TRACE(direction.name);

        renderer.setCamera(cameraFacing(resolution, origin, direction.yaw, direction.pitch));
        renderer.render();

        const auto looked_up = meanRadianceOf(accumulatedPixels(renderer, resolution)[center_pixel]);
        EXPECT_NEAR(looked_up[0], direction.expected_u, tolerance);
        EXPECT_NEAR(looked_up[1], direction.expected_v, tolerance);
    }
}

TEST(EnvironmentBackdropTest, IsNeitherMirroredNorFlipped)
{
    constexpr Size2i resolution{9, 9};
    constexpr float coarse_pixels = 2.0f;
    constexpr int middle = 4;
    constexpr int last = 8;

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, float(pi / 2), 0.0f, coarse_pixels));
    renderer.setScene(sceneWithEnvironmentOnly(gradientPanorama(Size2i{256, 128})));

    renderer.render();

    const auto pixels = accumulatedPixels(renderer, resolution);
    auto at = [&](int x, int y) { return meanRadianceOf(pixels[x + y * resolution.width]); };

    EXPECT_GT(at(last, middle)[0] - at(0, middle)[0], 0.03)
        << "turning towards larger pixel x must move towards larger panorama u";
    EXPECT_GT(at(middle, 0)[1] - at(middle, last)[1], 0.05)
        << "larger pixel y looks higher, towards smaller panorama v";
}

TEST(EnvironmentBackdropTest, IntensityScaleMultipliesTheBackdropAndRestartsAccumulation)
{
    const Channels radiance{0.5, 0.25, 0.125};

    Renderer renderer(tiny);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(tiny, origin, 0.0f, 0.0f));
    renderer.setScene(sceneWithEnvironmentOnly(uniformPanorama(Size2i{8, 4}, Vec3{0.5f, 0.25f, 0.125f})));

    auto expectAccumulated = [&](int frames, double scale)
    {
        const auto pixels = accumulatedPixels(renderer, tiny);
        expectEveryPixelFiniteWithSampleCount(pixels, frames);
        expectEveryPixelHasMeanRadiance(pixels, Channels{radiance[0] * scale, radiance[1] * scale, radiance[2] * scale}, 1e-6);
    };

    renderer.render();
    expectAccumulated(1, 1.0);

    renderer.setEnvironmentMapIntensityScale(2.0f);
    renderer.render();
    expectAccumulated(1, 2.0);

    renderer.render();
    expectAccumulated(2, 2.0);

    renderer.setEnvironmentMapIntensityScale(0.25f);
    renderer.render();
    expectAccumulated(1, 0.25);
}

TEST(EnvironmentBackdropTest, SceneIntensityMultiplierComposesWithTheRendererIntensityScale)
{
    Renderer renderer(tiny);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(tiny, origin, 0.0f, 0.0f));
    renderer.setScene(sceneWithEnvironmentOnly(uniformPanorama(Size2i{8, 4}, Vec3{0.5f, 0.25f, 0.125f}), 2.0f));
    renderer.setEnvironmentMapIntensityScale(3.0f);

    renderer.render();

    expectEveryPixelHasMeanRadiance(accumulatedPixels(renderer, tiny), Channels{3.0, 1.5, 0.75}, 1e-6);
}

TEST(EnvironmentBackdropTest, OnePixelPanoramaIsUniformInEveryDirection)
{
    Renderer renderer(tiny);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(tiny, origin, 2.1f, 0.6f));
    renderer.setScene(sceneWithEnvironmentOnly(uniformPanorama(Size2i{1, 1}, Vec3{0.75f, 0.5f, 0.25f})));

    renderer.render();

    expectEveryPixelHasMeanRadiance(accumulatedPixels(renderer, tiny), Channels{0.75, 0.5, 0.25}, 1e-6);
}

TEST(EnvironmentLightingTest, UniformEnvironmentOnAFloorGivesAlbedoTimesRadiance)
{
    constexpr Size2i resolution{8, 8};
    constexpr int frames = 16;

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, Vec3{0, 2, 0}, 0.0f, -1.0f));

    Scene scene;
    const int floor = addDiffuse(scene, Vec4{0.8f, 0.5f, 0.2f, 0});
    addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, floor);
    setEnvironment(scene, uniformPanorama(Size2i{8, 4}, Vec3{1.0f, 0.5f, 2.0f}));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, frames);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, frames);
    expectEstimateMatches(estimateMeanRadiance(pixels), Channels{0.8, 0.25, 0.4});
}

TEST(EnvironmentLightingTest, UniformEnvironmentOnAConvexBallGivesAlbedoTimesRadiance)
{
    constexpr Size2i resolution{8, 8};
    constexpr int frames = 16;
    constexpr double tolerance_for_smooth_shading_of_a_polygon_mesh = 0.03;

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

    Scene scene;
    const int grey = addDiffuse(scene, Vec4{0.5f, 0.5f, 0.5f, 0});
    addBall(scene, Vec3{0, 0, 3}, 1.0f, grey);
    setEnvironment(scene, uniformPanorama(Size2i{8, 4}, Vec3{1.0f, 1.0f, 1.0f}));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, frames);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, frames);
    expectEstimateMatches(estimateMeanRadiance(pixels), Channels{0.5, 0.5, 0.5}, 5.0, tolerance_for_smooth_shading_of_a_polygon_mesh);
}

TEST(EnvironmentLightingTest, FloorUnderASyntheticSunMatchesTheIntegratedPanorama)
{
    constexpr Size2i resolution{16, 16};
    constexpr int frames = 512;
    const Channels albedo{0.6, 0.5, 0.4};
    constexpr double relative_slack_for_float_accumulation = 0.01;

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, Vec3{0, 2, 0}, 0.0f, -1.0f));

    Scene scene;
    const int floor = addDiffuse(scene, Vec4{float(albedo[0]), float(albedo[1]), float(albedo[2]), 0});
    addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, floor);
    setEnvironment(scene, sunnyPanorama());
    renderer.setScene(std::move(scene));

    renderFrames(renderer, frames);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, frames);
    expectEstimateMatches(
        estimateMeanRadiance(pixels),
        lambertianFloorRadiance(sunnyPanorama(), albedo),
        5.0,
        relative_slack_for_float_accumulation);
}

TEST(EnvironmentLightingTest, PanoramaBelowTheHorizonDoesNotLightAFloor)
{
    constexpr Size2i resolution{8, 8};
    constexpr int frames = 64;
    constexpr int first_bright_row = 20;

    auto panorama = uniformPanorama(Size2i{32, 32}, Vec3{0, 0, 0});
    for (int row = first_bright_row; row < panorama.size.height; ++row)
        for (int column = 0; column < panorama.size.width; ++column)
            panorama[Vec2i{column, row}] = Vec4{10.0f, 10.0f, 10.0f, 1.0f};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, Vec3{0, 2, 0}, 0.0f, -1.0f));

    Scene scene;
    const int floor = addDiffuse(scene, Vec4{0.9f, 0.9f, 0.9f, 0});
    addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, floor);
    setEnvironment(scene, std::move(panorama));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, frames);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, frames);
    expectEveryPixelHasMeanRadiance(pixels, Channels{0, 0, 0}, 0.0);
}

TEST(EnvironmentGlassTest, GlassBallInAUniformEnvironmentConservesEnergy)
{
    constexpr Size2i resolution{8, 8};
    constexpr int frames = 8;
    constexpr double slack_for_paths_cut_off_by_the_depth_limit = 0.01;

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

    Scene scene;
    const int glass = addGlass(scene, 1.5f);
    addBall(scene, Vec3{0, 0, 3}, 1.0f, glass);
    setEnvironment(scene, uniformPanorama(Size2i{8, 4}, Vec3{1.0f, 1.0f, 1.0f}));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, frames);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, frames);
    expectEstimateMatches(estimateMeanRadiance(pixels), Channels{1.0, 1.0, 1.0}, 5.0, slack_for_paths_cut_off_by_the_depth_limit);
}

TEST(EnvironmentGlassTest, GlassBallOnAFloorWithADistantEmitterRendersFinitePixels)
{
    expectGlassBallOnAFloorRendersFinitePixels(true);
}

TEST(EnvironmentGlassTest, DISABLED_GlassBallOnAFloorLitOnlyByTheEnvironmentRendersFinitePixels)
{
    expectGlassBallOnAFloorRendersFinitePixels(false);
}
