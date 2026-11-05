// AI-generated test helpers (Claude), reviewed by hand before committing.

#pragma once

#include "CameraController.hpp"
#include "Image.hpp"
#include "Renderer.hpp"
#include "models/Mesh.hpp"
#include "tracing/TriangleMesh.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace render_test
{
    using Channels = std::array<double, 3>;

    inline Camera cameraFacing(
        Size2i resolution,
        Vec3 position,
        float yaw,
        float pitch,
        float physical_pixel_size = 0.1f)
    {
        constexpr float focal_length = 50.0f;

        const Camera pinhole{
            .transform = Matrix4x4f::identity(),
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

        return CameraController{
            .camera = pinhole,
            .target = position,
            .pitch = pitch,
            .yaw = yaw,
            .distance = 0.0f,
        }.getCamera();
    }

    inline int addDiffuse(Scene &scene, Vec4 reflectance, Vec4 emission = Vec4{0, 0, 0, 0})
    {
        DiffuseMaterial material{};
        material.debug_color = Vec4{0, 0, 0, 0};
        material.emission = emission;
        material.diffuse_reflectance = reflectance;

        scene.materials.push_back(material);
        return static_cast<int>(scene.materials.size()) - 1;
    }

    inline int addGlass(Scene &scene, float index_of_refraction)
    {
        TransparentMaterial material{};
        material.debug_color = Vec4{0, 0, 0, 0};
        material.inside_medium.ior = index_of_refraction;

        scene.materials.push_back(material);
        return static_cast<int>(scene.materials.size()) - 1;
    }

    inline void addMesh(Scene &scene, Mesh mesh, int material)
    {
        scene.mesh_storage.push_back(convertMeshToTris(mesh));
        auto view = scene.mesh_storage.back().view();
        view.material = material;
        scene.objects.push_back(std::move(view));
    }

    inline void addQuad(Scene &scene, Vec3 center, Vec3 normal, Vec3 right, float size, int material)
    {
        scene.mesh_storage.push_back(createQuadMesh(center, normal, right, size));
        auto view = scene.mesh_storage.back().view();
        view.material = material;
        scene.objects.push_back(std::move(view));
    }

    inline void addBall(Scene &scene, Vec3 center, float radius, int material)
    {
        auto mesh = loadMesh("models/ball.obj");
        normalizeMeshSize(mesh);

        scene.mesh_storage.push_back(convertMeshToTris(mesh));
        auto view = scene.mesh_storage.back().view();
        view.material = material;
        view.setPosition(center);
        view.setScale(Vec3{radius, radius, radius});
        scene.objects.push_back(std::move(view));
    }

    inline void addTinyQuadFarFromEverything(Scene &scene, int material)
    {
        constexpr float far_away = 1e5f;
        addQuad(scene, Vec3{far_away, far_away, far_away}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 1.0f, material);
    }

    inline Image4f uniformPanorama(Size2i size, Vec3 radiance)
    {
        return Image4f::create(size, Vec4{radiance.x, radiance.y, radiance.z, 1.0f});
    }

    inline Image4f gradientPanorama(Size2i size)
    {
        auto panorama = Image4f::create_uninitialized(size);

        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x)
                panorama[Vec2i{x, y}] = Vec4{
                    (x + 0.5f) / size.width,
                    (y + 0.5f) / size.height,
                    0.0f,
                    1.0f,
                };

        return panorama;
    }

    inline void setEnvironment(Scene &scene, Image4f panorama, float intensity_multiplier = 1.0f)
    {
        scene.environment_map = EnvironmentMap{
            .intensity_multiplier = intensity_multiplier,
            .panorama_image = std::move(panorama),
        };
    }

    inline void useDeterministicLinearOutput(Renderer &renderer)
    {
        renderer.setDebug(DebugOptions::Off);
        renderer.setPixelSampling(PixelSampling::Center);
        renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::Linear});
    }

    inline void renderFrames(Renderer &renderer, int frame_count)
    {
        for (int i = 0; i < frame_count; ++i) renderer.render();
    }

    inline std::vector<Vec4> accumulatedPixels(Renderer &renderer, Size2i resolution)
    {
        const Vec4 *pixels = renderer.getRawPixels();

        return std::vector<Vec4>(pixels, pixels + resolution.area());
    }

    inline Channels meanRadianceOf(const Vec4 &accumulated)
    {
        return Channels{accumulated.x / accumulated.w, accumulated.y / accumulated.w, accumulated.z / accumulated.w};
    }

    inline bool isFinite(const Vec4 &pixel)
    {
        return std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z) && std::isfinite(pixel.w);
    }

    inline void expectEveryPixelFiniteWithSampleCount(const std::vector<Vec4> &pixels, int expected_sample_count)
    {
        for (size_t i = 0; i < pixels.size(); ++i)
        {
            ASSERT_TRUE(isFinite(pixels[i])) << "pixel " << i << " = (" << pixels[i].x << ", " << pixels[i].y << ", "
                                              << pixels[i].z << ", " << pixels[i].w << ")";
            ASSERT_EQ(pixels[i].w, static_cast<float>(expected_sample_count)) << "pixel " << i;
        }
    }

    inline void expectEveryPixelHasMeanRadiance(const std::vector<Vec4> &pixels, const Channels &expected, double tolerance)
    {
        for (size_t i = 0; i < pixels.size(); ++i)
        {
            const auto mean = meanRadianceOf(pixels[i]);

            for (int channel = 0; channel < 3; ++channel)
                ASSERT_NEAR(mean[channel], expected[channel], tolerance) << "pixel " << i << ", channel " << channel;
        }
    }

    struct Estimate
    {
        Channels mean;
        Channels standard_error;
    };

    inline Estimate estimateMeanRadiance(const std::vector<Vec4> &pixels)
    {
        Channels sum{};
        Channels sum_of_squares{};

        for (const Vec4 &pixel : pixels)
        {
            const auto radiance = meanRadianceOf(pixel);

            for (int channel = 0; channel < 3; ++channel)
            {
                sum[channel] += radiance[channel];
                sum_of_squares[channel] += radiance[channel] * radiance[channel];
            }
        }

        const double count = static_cast<double>(pixels.size());
        Estimate estimate{};

        for (int channel = 0; channel < 3; ++channel)
        {
            const double mean = sum[channel] / count;
            const double variance = count > 1
                ? std::max(0.0, sum_of_squares[channel] / count - mean * mean) * count / (count - 1)
                : 0.0;

            estimate.mean[channel] = mean;
            estimate.standard_error[channel] = std::sqrt(variance / count);
        }

        return estimate;
    }

    inline void expectEstimateMatches(
        const Estimate &estimate,
        const Channels &expected,
        double standard_errors = 5.0,
        double relative_slack = 1e-3)
    {
        constexpr double absolute_slack = 1e-6;

        for (int channel = 0; channel < 3; ++channel)
        {
            const double tolerance = standard_errors * estimate.standard_error[channel]
                + relative_slack * std::abs(expected[channel])
                + absolute_slack;

            EXPECT_NEAR(estimate.mean[channel], expected[channel], tolerance)
                << "channel " << channel << ", standard error " << estimate.standard_error[channel];
        }
    }

    inline Channels lambertianFloorRadiance(const Image4f &panorama, const Channels &albedo, double intensity = 1.0)
    {
        const double pi = std::numbers::pi;
        const int width = panorama.size.width;
        const int height = panorama.size.height;

        EXPECT_EQ(height % 2, 0) << "the horizon must fall on a pixel-row boundary";

        Channels irradiance{};

        for (int row = 0; row < height / 2; ++row)
        {
            const double sin_top = std::sin(row * pi / height);
            const double sin_bottom = std::sin((row + 1) * pi / height);
            const double projected_solid_angle = (2 * pi / width) * (sin_bottom * sin_bottom - sin_top * sin_top) / 2;

            for (int column = 0; column < width; ++column)
            {
                const Vec4 pixel = panorama[Vec2i{column, row}];

                irradiance[0] += pixel.x * projected_solid_angle;
                irradiance[1] += pixel.y * projected_solid_angle;
                irradiance[2] += pixel.z * projected_solid_angle;
            }
        }

        Channels radiance{};
        for (int channel = 0; channel < 3; ++channel)
            radiance[channel] = albedo[channel] * irradiance[channel] / pi * intensity;

        return radiance;
    }
}
