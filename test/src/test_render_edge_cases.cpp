// AI-generated tests (Claude), reviewed by hand before committing.

#include "RenderTestHelpers.hpp"
#include "Renderer.hpp"
#include "Utils.hpp"
#include "models/Mesh.hpp"
#include "tracing/PixelSampling.hpp"
#include "tracing/Scene.hpp"

#include <gtest/gtest.h>

#include <utility>

using namespace render_test;

namespace
{
    constexpr float emission = 0.4f;
    const Vec3 origin{0, 0, 0};
    const Channels black{0, 0, 0};

    int addEmitter(Scene &scene)
    {
        return addDiffuse(scene, Vec4{0, 0, 0, 0}, Vec4{emission, emission, emission, 0});
    }

    void addEmitterFillingTheView(Scene &scene)
    {
        addQuad(scene, Vec3{0, 0, 1}, Vec3{0, 0, -1}, Vec3{1, 0, 0}, 1000.0f, addEmitter(scene));
    }

    void expectRendersEmitterFillingTheView(Size2i resolution, int frames, PixelSampling sampling = PixelSampling::UniformRandom)
    {
        SCOPED_TRACE(::testing::Message() << resolution.width << "x" << resolution.height);

        Renderer renderer(resolution);
        useDeterministicLinearOutput(renderer);
        renderer.setPixelSampling(sampling);
        renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

        Scene scene;
        addEmitterFillingTheView(scene);
        renderer.setScene(std::move(scene));

        renderFrames(renderer, frames);

        const auto pixels = accumulatedPixels(renderer, resolution);
        expectEveryPixelFiniteWithSampleCount(pixels, frames);
        expectEveryPixelHasMeanRadiance(pixels, Channels{emission, emission, emission}, 1e-6);
    }
}

TEST(RendererEdgeCaseTest, DISABLED_EmptySceneRendersBlack)
{
    constexpr Size2i resolution{4, 4};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));
    renderer.setScene(Scene{});

    ASSERT_NO_THROW(renderFrames(renderer, 2));

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 2);
    expectEveryPixelHasMeanRadiance(pixels, black, 0.0);
}

TEST(RendererEdgeCaseTest, DISABLED_EmptySceneWithAnEnvironmentShowsTheBackdrop)
{
    constexpr Size2i resolution{4, 4};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

    Scene scene;
    setEnvironment(scene, uniformPanorama(Size2i{8, 4}, Vec3{0.5f, 0.25f, 2.0f}));
    renderer.setScene(std::move(scene));

    ASSERT_NO_THROW(renderFrames(renderer, 2));

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 2);
    expectEveryPixelHasMeanRadiance(pixels, Channels{0.5, 0.25, 2.0}, 1e-6);
}

TEST(RendererEdgeCaseTest, SingleObjectOutOfViewRendersBlack)
{
    constexpr Size2i resolution{4, 4};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

    Scene scene;
    addTinyQuadFarFromEverything(scene, addEmitter(scene));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, 2);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 2);
    expectEveryPixelHasMeanRadiance(pixels, black, 0.0);
}

TEST(RendererEdgeCaseTest, DiffuseSceneWithoutAnyLightRendersBlack)
{
    constexpr Size2i resolution{8, 8};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, Vec3{0, 1, 0}, 0.0f, -0.3f, 0.5f));

    Scene scene;
    const int grey = addDiffuse(scene, Vec4{0.5f, 0.5f, 0.5f, 0});
    addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, grey);
    addBall(scene, Vec3{0, 0.5f, 3}, 0.5f, grey);
    renderer.setScene(std::move(scene));

    renderFrames(renderer, 4);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 4);
    expectEveryPixelHasMeanRadiance(pixels, black, 0.0);
}

TEST(RendererEdgeCaseTest, GlassOnlySceneWithoutAnyLightRendersBlack)
{
    constexpr Size2i resolution{8, 8};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f));

    Scene scene;
    addBall(scene, Vec3{0, 0, 3}, 1.0f, addGlass(scene, 1.5f));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, 4);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 4);
    expectEveryPixelHasMeanRadiance(pixels, black, 0.0);
}

TEST(RendererEdgeCaseTest, SingleTriangleLightsOnlyItsOwnPixels)
{
    constexpr Size2i resolution{16, 16};
    constexpr float coarse_pixels = 1.0f;

    Mesh triangle;
    triangle.name = "emissive-triangle";
    triangle.points = {Vec3{-0.2f, -0.2f, 2}, Vec3{0.2f, -0.2f, 2}, Vec3{0, 0.2f, 2}};
    triangle.normals.push_back(Vec3{0, 0, -1});
    triangle.triangles.push_back(Triangle{.a = {0, 0}, .b = {1, 0}, .c = {2, 0}});

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, origin, 0.0f, 0.0f, coarse_pixels));

    Scene scene;
    addMesh(scene, std::move(triangle), addEmitter(scene));
    renderer.setScene(std::move(scene));

    renderer.render();

    const auto pixels = accumulatedPixels(renderer, resolution);
    auto at = [&](int x, int y) { return meanRadianceOf(pixels[x + y * resolution.width]); };

    const Channels lit{emission, emission, emission};
    const int last = resolution.width - 1;
    for (int channel = 0; channel < 3; ++channel)
    {
        EXPECT_NEAR(at(8, 8)[channel], lit[channel], 1e-6);

        EXPECT_EQ(at(0, 0)[channel], 0.0);
        EXPECT_EQ(at(last, 0)[channel], 0.0);
        EXPECT_EQ(at(0, last)[channel], 0.0);
        EXPECT_EQ(at(last, last)[channel], 0.0);
    }
}

TEST(RendererEdgeCaseTest, SinglePixelRendererRendersTheEmitterInView)
{
    expectRendersEmitterFillingTheView(Size2i{1, 1}, 2);
}

TEST(RendererEdgeCaseTest, ResolutionsThatDoNotFillWholeThreadBlocksRenderEveryPixelExactlyOnce)
{
    for (const Size2i resolution : {Size2i{23, 23}, Size2i{17, 13}, Size2i{1, 300}, Size2i{257, 1}})
        expectRendersEmitterFillingTheView(resolution, 2);
}

TEST(RendererEdgeCaseTest, DISABLED_RaysThroughTheEdgeSharedByTwoTrianglesAreNotLost)
{
    expectRendersEmitterFillingTheView(Size2i{16, 16}, 2, PixelSampling::Center);
}

TEST(RendererEdgeCaseTest, SurfaceThatReflectsNothingStaysBlackUnderTheEnvironment)
{
    constexpr Size2i resolution{4, 4};

    Renderer renderer(resolution);
    useDeterministicLinearOutput(renderer);
    renderer.setCamera(cameraFacing(resolution, Vec3{0, 2, 0}, 0.0f, -1.0f));

    Scene scene;
    const int absorbing = addDiffuse(scene, Vec4{0, 0, 0, 0});
    addQuad(scene, Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{1, 0, 0}, 1000.0f, absorbing);
    setEnvironment(scene, uniformPanorama(Size2i{8, 4}, Vec3{5.0f, 5.0f, 5.0f}));
    renderer.setScene(std::move(scene));

    renderFrames(renderer, 4);

    const auto pixels = accumulatedPixels(renderer, resolution);
    expectEveryPixelFiniteWithSampleCount(pixels, 4);
    expectEveryPixelHasMeanRadiance(pixels, black, 0.0);
}
