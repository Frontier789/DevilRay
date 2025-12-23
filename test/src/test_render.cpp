#include <Utils.hpp>
#include <Renderer.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <cmath>

//////////////////////////////////////////////////
///             DISCLAIMER                     ///
///                                            ///
/// Parts of these test were ai generated with ///
/// Google's Gemini 3, then reviewed and       ///
/// corrected here and there by hand.          ///
///                                            ///
//////////////////////////////////////////////////

namespace {
    Camera createCamera(Size2i resolution, const float focal_length, const float physical_pixel_size)
    {
        Camera cam{
            .intrinsics = Intrinsics{
                .focal_length = focal_length,
                .center = Vec2{
                    resolution.width/2.f * physical_pixel_size,
                    resolution.height/2.f * physical_pixel_size,
                }
            },
            .resolution = resolution,
            .physical_pixel_size = Size2f{physical_pixel_size, physical_pixel_size},
        };
    
        return cam;
    }
    
    std::filesystem::path ensureTestOutputFolder()
    {
        const auto path = std::filesystem::path("test_output");
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
        }

        return path;
    }
}

TEST(RendererTest, AnalyticalDiffuseReflection) {
    // 1. Setup Resolution and Camera
    const Size2i resolution{512, 512};
    Renderer renderer(resolution);
    
    // focal_length and pixel_size set to keep FOV simple
    Camera cam = createCamera(resolution, 50.0f, 0.1f); 
    renderer.setCamera(std::move(cam));

    // 2. Setup Scene
    Scene scene;

    const double emittance = 0.55;
    
    // Define Materials
    // White Diffuse Material
    const int white_mat_idx = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0, 0},
            .diffuse_reflectance = Vec4{1.0, 1.0, 1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.7, 0.5, 0.0};
        scene.materials.push_back(material);
    }

    // Light Source Material (Emissive)
    const int light_mat_idx = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{emittance, emittance, emittance, 0.0},
            .diffuse_reflectance = Vec4{0.0, 0.0, 0.0, 0.0},
        };
        scene.materials.push_back(material);
    }

    // 3. Add Objects
    // Receiver: A large white square in front of the camera (z = 1.0)
    {
        auto receiver = Square{
            .p = Vec3{0, 0, 1.0},
            .n = Vec3{0, 0, -1}, // Facing the camera
            .right = Vec3{1, 0, 0},
            .size = 1000.0, // Large size to simulate infinite plane
        };
        receiver.mat = white_mat_idx;
        scene.objects.push_back(std::move(receiver));
    }

    // Source: A large emissive square behind the camera (z = -1.0)
    // This will illuminate the 'receiver' square
    {
        auto source = Square{
            .p = Vec3{0, 0, -1.0},
            .n = Vec3{0, 0, 1}, // Facing the receiver
            .right = Vec3{1, 0, 0},
            .size = 1000.0,
        };
        source.mat = light_mat_idx;
        scene.objects.push_back(std::move(source));
    }

    renderer.setScene(std::move(scene));

    // 4. Render
    renderer.setOutputOptions(OutputOptions{.linearity=OutputLinearity::Linear});
    renderer.setDebug(false);
    renderer.useCudaDevice(true);
    for (int i=0;i<10;++i)
    renderer.render();
    renderer.createPixels();

    const auto outputPath = ensureTestOutputFolder();
    renderer.saveImage(outputPath / "analytic_diffuse_planes.png");

    // 5. Verification
    const uint32_t* pixels = renderer.getPixels();
    
    std::vector<float> intensities;
    intensities.reserve(resolution.area());

    for (int i=0;i<resolution.height;++i) {
        for (int j=0;j<resolution.width;++j) {
            const auto px = pixels[i * resolution.width + j];

            const auto r = static_cast<int>((px >>  0) & 0xff);
            const auto g = static_cast<int>((px >>  8) & 0xff);
            const auto b = static_cast<int>((px >> 16) & 0xff);

            EXPECT_EQ(r, g);
            EXPECT_EQ(r, b);

            intensities.push_back(static_cast<float>(r));
        }
    }

    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;

    for (const auto f : intensities) {
        const double d = f;

        sum1 += d;
        sum2 += d*d;
        sum3 += d*d*d;
    }

    const double n = intensities.size();
    const double mean = sum1 / n;
    const double stddev = std::sqrt(sum2 / n - mean*mean);
    const double skew = (sum3/n - 3*mean*stddev*stddev - mean*mean*mean) / (stddev*stddev*stddev);

    EXPECT_LE(std::abs(mean - 255.0 * emittance), 0.01);
    EXPECT_LE(stddev, 10.0);
    EXPECT_LE(skew, 0.01);
}


TEST(RendererTest, DebugRenderSquare) {
    // 1. Setup Resolution and Camera
    const Size2i resolution{512, 512};
    Renderer renderer(resolution);
    
    // focal_length and pixel_size set to keep FOV simple
    Camera cam = createCamera(resolution, 50.0f, 0.1f); 
    renderer.setCamera(std::move(cam));

    // 2. Setup Scene
    Scene scene;
    
    // Define Materials
    // White Diffuse Material
    const int white_mat_idx = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0, 0},
            .diffuse_reflectance = Vec4{1.0, 1.0, 1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.7, 0.5, 0.0};
        scene.materials.push_back(material);
    }

    // 3. Add Object
    {
        auto obj = Square{
            .p = Vec3{0, 0, 1.0},
            .n = Vec3{0, 0, -1}, // Facing the camera
            .right = Vec3{1, 0, 0},
            .size = 1.0,
        };
        obj.mat = white_mat_idx;
        scene.objects.push_back(std::move(obj));
    }

    renderer.setScene(std::move(scene));

    // 4. Render
    renderer.setOutputOptions(OutputOptions{.linearity=OutputLinearity::GammaCorrected});
    renderer.setDebug(true);
    renderer.useCudaDevice(true);
    renderer.render();
    renderer.createPixels();

    const auto outputPath = ensureTestOutputFolder();
    renderer.saveImage(outputPath / "debug_render_1_ray.png");
    
    // 5. Verification
    const uint32_t* pixels = renderer.getPixels();
    for (int i=0;i<resolution.height;++i) {
        for (int j=0;j<resolution.width;++j) {
            const auto px = pixels[i * resolution.width + j];

            const auto r = static_cast<float>((px >>  0) & 0xff) / 255.0f;
            const auto g = static_cast<float>((px >>  8) & 0xff) / 255.0f;
            const auto b = static_cast<float>((px >> 16) & 0xff) / 255.0f;
            
            const auto is_rim = i < 6 || j < 6 || i >= resolution.height - 6 || j >= resolution.width - 6;

            if (is_rim) {
                EXPECT_FLOAT_EQ(r, 0.0f);
                EXPECT_FLOAT_EQ(g, 0.0f);
                EXPECT_FLOAT_EQ(b, 0.0f);
            }
            else {
                EXPECT_GE(r, 0.65f);
                EXPECT_GE(g, 0.6f);
                EXPECT_GE(b, 0.5f);
            }
        }
    }
}