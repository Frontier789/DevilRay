// AI-generated tests (Claude), reviewed by hand before committing.

#include "DebugOptions.hpp"
#include "Renderer.hpp"
#include "Utils.hpp"
#include "models/Matrix.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Material.hpp"
#include "tracing/OutputOptions.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Scene.hpp"
#include "tracing/TriangleMesh.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
    constexpr Size2i resolution{16, 16};
    constexpr float emission = 0.2f;
    constexpr float a_different_emission = 0.6f;

    Camera cameraLookingDownPlusZ(Vec3 position = Vec3{0, 0, 0})
    {
        constexpr float focal_length = 50.0f;
        constexpr float physical_pixel_size = 0.1f;

        return Camera{
            .transform = Matrix4x4f::translation(position),
            .intrinsics = Intrinsics{
                .focal_length = focal_length,
                .center = Vec2f{
                    resolution.width / 2.0f * physical_pixel_size,
                    resolution.height / 2.0f * physical_pixel_size,
                },
            },
            .resolution = resolution,
            .physical_pixel_size = Size2f{physical_pixel_size, physical_pixel_size},
        };
    }

    Scene sceneWithEmitterFillingTheView(float emitter_emission)
    {
        Scene scene;

        DiffuseMaterial emitter_that_reflects_nothing{};
        emitter_that_reflects_nothing.debug_color = Vec4{0, 0, 0, 0};
        emitter_that_reflects_nothing.emission = Vec4{emitter_emission, emitter_emission, emitter_emission, 0};
        emitter_that_reflects_nothing.diffuse_reflectance = Vec4{0, 0, 0, 0};

        const int emitter_material = static_cast<int>(scene.materials.size());
        scene.materials.push_back(emitter_that_reflects_nothing);

        constexpr float quad_size_far_larger_than_the_view = 1000.0f;
        scene.mesh_storage.push_back(createQuadMesh(
            Vec3{0, 0, 1}, Vec3{0, 0, -1}, Vec3{1, 0, 0}, quad_size_far_larger_than_the_view));
        auto quad = scene.mesh_storage.back().view();
        quad.material = emitter_material;
        scene.objects.push_back(std::move(quad));

        return scene;
    }

    void showEmitterFillingTheView(Renderer &renderer)
    {
        renderer.setCamera(cameraLookingDownPlusZ());
        renderer.setScene(sceneWithEmitterFillingTheView(emission));
        renderer.setDebug(DebugOptions::Off);
        renderer.setPixelSampling(PixelSampling::UniformRandom);
        renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::Linear});
    }

    void renderTimes(Renderer &renderer, int frame_count)
    {
        for (int i = 0; i < frame_count; ++i) renderer.render();
    }

    std::vector<Vec4> accumulatedPixels(Renderer &renderer)
    {
        const Vec4 *pixels = renderer.getRawPixels();

        return std::vector<Vec4>(pixels, pixels + resolution.area());
    }

    void expectEveryPixelHasSampleCount(Renderer &renderer, int expected_sample_count)
    {
        for (const auto &pixel : accumulatedPixels(renderer))
            ASSERT_EQ(pixel.w, static_cast<float>(expected_sample_count));
    }

    void expectEveryPixelHasMeanRadiance(Renderer &renderer, float expected_radiance)
    {
        for (const auto &pixel : accumulatedPixels(renderer))
        {
            EXPECT_FLOAT_EQ(pixel.x / pixel.w, expected_radiance);
            EXPECT_FLOAT_EQ(pixel.y / pixel.w, expected_radiance);
            EXPECT_FLOAT_EQ(pixel.z / pixel.w, expected_radiance);
        }
    }

    int redByteOf(uint32_t packed_pixel) { return (packed_pixel >> 0) & 0xff; }
    int greenByteOf(uint32_t packed_pixel) { return (packed_pixel >> 8) & 0xff; }
    int blueByteOf(uint32_t packed_pixel) { return (packed_pixel >> 16) & 0xff; }
    int alphaByteOf(uint32_t packed_pixel) { return (packed_pixel >> 24) & 0xff; }

    int byteFor(float encoded_channel) { return static_cast<int>(std::lround(encoded_channel * 255.0f)); }

    float srgbEncoded(float linear)
    {
        constexpr float linear_segment_end = 0.0031308f;
        if (linear <= linear_segment_end) return 12.92f * linear;

        return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    }

    float plainGamma22Encoded(float linear) { return std::pow(linear, 1.0f / 2.2f); }

    void expectEveryChannelIs(uint32_t packed_pixel, int expected_byte)
    {
        EXPECT_EQ(redByteOf(packed_pixel), expected_byte);
        EXPECT_EQ(greenByteOf(packed_pixel), expected_byte);
        EXPECT_EQ(blueByteOf(packed_pixel), expected_byte);
        EXPECT_EQ(alphaByteOf(packed_pixel), 255);
    }
}

TEST(RendererAccumulationTest, EachRenderAddsExactlyOneSamplePerPixel)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);

    renderTimes(renderer, 3);

    expectEveryPixelHasSampleCount(renderer, 3);
}

TEST(RendererAccumulationTest, DirectlyVisibleEmitterAccumulatesExactlyItsEmissionPerSample)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);

    renderTimes(renderer, 4);

    expectEveryPixelHasMeanRadiance(renderer, emission);
}

TEST(RendererAccumulationTest, ClearDiscardsTheAccumulatedSamples)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);
    renderTimes(renderer, 2);

    renderer.clear();
    renderTimes(renderer, 1);

    expectEveryPixelHasSampleCount(renderer, 1);
}

TEST(RendererAccumulationTest, ChangingAnInputThatAffectsTheImageRestartsAccumulation)
{
    struct InputChange
    {
        const char *input;
        void (*change)(Renderer &);
    };

    const std::vector<InputChange> input_changes = {
        {"camera", [](Renderer &r) { r.setCamera(cameraLookingDownPlusZ(Vec3{0.5f, 0, 0})); }},
        {"debug mode", [](Renderer &r) { r.setDebug(DebugOptions::UVChecker); }},
        {"pixel sampling", [](Renderer &r) { r.setPixelSampling(PixelSampling::Center); }},
    };

    for (const auto &[input, change] : input_changes)
    {
        Renderer renderer(resolution);
        showEmitterFillingTheView(renderer);
        renderTimes(renderer, 2);

        change(renderer);
        renderTimes(renderer, 1);

        SCOPED_TRACE(input);
        expectEveryPixelHasSampleCount(renderer, 1);
    }
}

TEST(RendererAccumulationTest, ChangingTheSceneRestartsAccumulationWithTheNewScenesRadiance)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);
    renderTimes(renderer, 2);

    renderer.setScene(sceneWithEmitterFillingTheView(a_different_emission));
    renderTimes(renderer, 1);

    expectEveryPixelHasSampleCount(renderer, 1);
    expectEveryPixelHasMeanRadiance(renderer, a_different_emission);
}

TEST(RendererAccumulationTest, ChangingOnlyTheOutputEncodingKeepsTheAccumulatedSamples)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);
    renderTimes(renderer, 2);

    renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::GammaCorrected});
    renderTimes(renderer, 1);

    expectEveryPixelHasSampleCount(renderer, 3);
}

TEST(RendererOutputTest, LinearOutputQuantizesTheMeanRadianceDirectly)
{
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);
    renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::Linear});

    renderTimes(renderer, 2);

    expectEveryChannelIs(renderer.getPixels()[0], byteFor(emission));
}

TEST(RendererOutputTest, GammaCorrectedOutputAppliesTheSrgbTransferCurve)
{
    const int expected_byte = byteFor(srgbEncoded(emission));
    ASSERT_NE(expected_byte, byteFor(plainGamma22Encoded(emission)))
        << "The emission has to tell the sRGB curve apart from a plain 2.2 gamma";
    Renderer renderer(resolution);
    showEmitterFillingTheView(renderer);
    renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::GammaCorrected});

    renderTimes(renderer, 2);

    expectEveryChannelIs(renderer.getPixels()[0], expected_byte);
}
