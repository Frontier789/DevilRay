#include <Utils.hpp>
#include <Renderer.hpp>

#include "tracing/GpuTris.hpp"
#include <gtest/gtest.h>
#include <stb_image.h>

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
            .transform = Matrix4x4f::identity(),
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
        scene.mesh_storage.push_back(createQuadMesh(
            Vec3{0, 0, 1.0}, Vec3{0, 0, -1}, Vec3{1, 0, 0}, 1000.0));
        auto receiver = viewGpuTris(scene.mesh_storage.back());
        receiver.material = white_mat_idx;
        scene.objects.push_back(std::move(receiver));
    }

    // Source: A large emissive square behind the camera (z = -1.0)
    // This will illuminate the 'receiver' square
    {
        scene.mesh_storage.push_back(createQuadMesh(
            Vec3{0, 0, -1.0}, Vec3{0, 0, 1}, Vec3{1, 0, 0}, 1000.0));
        auto source = viewGpuTris(scene.mesh_storage.back());
        source.material = light_mat_idx;
        scene.objects.push_back(std::move(source));
    }

    renderer.setScene(std::move(scene));

    // 4. Render
    renderer.setOutputOptions(OutputOptions{.linearity=OutputLinearity::Linear});
    renderer.setDebug(DebugOptions::Off);
    for (int i=0;i<100;++i)
    renderer.render();
    renderer.createPixels();

    const auto outputPath = ensureTestOutputFolder();
    renderer.saveImage(outputPath / "analytic_diffuse_planes.png");

    // 5. Verification
    const Vec4* pixels = renderer.getRawPixels();
    
    std::vector<float> intensities;
    intensities.reserve(resolution.area());

    const auto referenceSampleCount = pixels[0].w;

    for (int i=0;i<resolution.height;++i) {
        for (int j=0;j<resolution.width;++j) {
            const auto px = pixels[i * resolution.width + j];

            const auto sampleCount = px.w;
            const auto r = px.x / sampleCount;
            const auto g = px.y / sampleCount;
            const auto b = px.z / sampleCount;

            ASSERT_EQ(referenceSampleCount, sampleCount);
            ASSERT_EQ(r, g);
            ASSERT_EQ(g, b);
            ASSERT_EQ(r, b);

            const auto intensity = r;
            intensities.push_back(intensity);
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

    EXPECT_NEAR(mean, emittance, 0.01);
    EXPECT_LE(stddev, 0.04);
    EXPECT_LE(skew, 4e-5);
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
        scene.mesh_storage.push_back(createQuadMesh(
            Vec3{0, 0, 1.0}, Vec3{0, 0, -1}, Vec3{1, 0, 0}, 1.0));
        auto obj = viewGpuTris(scene.mesh_storage.back());
        obj.material = white_mat_idx;
        scene.objects.push_back(std::move(obj));
    }

    renderer.setScene(std::move(scene));

    // 4. Render
    renderer.setOutputOptions(OutputOptions{.linearity=OutputLinearity::GammaCorrected});
    renderer.setDebug(DebugOptions::UVChecker);
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
                ASSERT_GE(r, 0.65f);
                ASSERT_GE(g, 0.6f);
                ASSERT_GE(b, 0.5f);
            }
        }
    }
}


TEST(RendererTest, TriangleSceneRender) {
    // 1. Setup Resolution and Camera
    const Size2i resolution{512, 512};
    Renderer renderer(resolution);

    Camera cam = createCamera(resolution, 50.0f, 0.1f);
    renderer.setCamera(std::move(cam));

    // 2. Setup Scene
    Scene scene;

    // -- Materials --
    // Emissive (warm white light)
    const int emissive_mat = scene.materials.size();
    {
        auto m = DiffuseMaterial{
            .emission = Vec4{0.9f, 0.8f, 0.6f, 0.0f},
            .diffuse_reflectance = Vec4{0.0f, 0.0f, 0.0f, 0.0f},
        };
        scene.materials.push_back(m);
    }

    // Red diffuse
    const int red_mat = scene.materials.size();
    {
        auto m = DiffuseMaterial{
            .emission = Vec4{0, 0, 0, 0},
            .diffuse_reflectance = Vec4{0.9f, 0.15f, 0.15f, 0.0f},
        };
        scene.materials.push_back(m);
    }

    // Green diffuse
    const int green_mat = scene.materials.size();
    {
        auto m = DiffuseMaterial{
            .emission = Vec4{0, 0, 0, 0},
            .diffuse_reflectance = Vec4{0.15f, 0.9f, 0.15f, 0.0f},
        };
        scene.materials.push_back(m);
    }

    // Blue diffuse
    const int blue_mat = scene.materials.size();
    {
        auto m = DiffuseMaterial{
            .emission = Vec4{0, 0, 0, 0},
            .diffuse_reflectance = Vec4{0.15f, 0.15f, 0.9f, 0.0f},
        };
        scene.materials.push_back(m);
    }

    // 3. Build triangle meshes
    // GpuTris must stay alive so the device pointers remain valid.
    std::vector<GpuTris> trisStorage;

    auto addTriangle = [&](const Vec3 &a, const Vec3 &b, const Vec3 &c, int mat_idx) {
        Mesh mesh;
        const uint32_t ia = mesh.points.size(); mesh.points.push_back(a);
        const uint32_t ib = mesh.points.size(); mesh.points.push_back(b);
        const uint32_t ic = mesh.points.size(); mesh.points.push_back(c);

        const auto normal = (b - a).cross(c - a).normalized();
        mesh.normals.push_back(normal);
        mesh.normals.push_back(normal);
        mesh.normals.push_back(normal);

        mesh.triangles.push_back(Triangle{
            .a = Vertex{ia, ia},
            .b = Vertex{ib, ib},
            .c = Vertex{ic, ic},
        });

        trisStorage.push_back(convertMeshToTris(mesh));
        auto obj = viewGpuTris(trisStorage.back());
        obj.material = mat_idx;
        scene.objects.push_back(std::move(obj));
    };

    // Emissive triangle – large, behind the camera, facing forward to illuminate the scene
    addTriangle(
        Vec3{-50.0f, -50.0f, -1.0f},
        Vec3{  0.0f,   0.0f, -1.0f},
        Vec3{ 50.0f, -50.0f, -1.0f},
        emissive_mat
    );

    // Procedurally generate a half sphere with 20 triangles
    // 4 longitude segments × (1 cap band + 2 quad bands) = 4 + 8 + 8 = 20
    {
        constexpr int nLon = 6;
        constexpr int nLat = 5;
        constexpr float radius = 0.5f;
        const Vec3 center{0.0f, 0.0f, 1.2f};
        constexpr float pi = std::numbers::pi_v<float>;

        // Alternating colors per quadrant
        const int colors[3] = {red_mat, green_mat, blue_mat};

        auto spherePoint = [&](float theta, float phi) -> Vec3 {
            return Vec3{
                center.x + radius * std::sin(theta) * std::cos(phi),
                center.y + radius * std::sin(theta) * std::sin(phi),
                center.z - radius * std::cos(theta)
            };
        };

        for (int lat = 0; lat < nLat; ++lat) {
            const float theta0 = static_cast<float>(lat) / nLat * (pi / 2.0f);
            const float theta1 = static_cast<float>(lat + 1) / nLat * (pi / 2.0f);

            for (int lon = 0; lon < nLon; ++lon) {
                const float phi0 = static_cast<float>(lon) / nLon * 2.0f * pi;
                const float phi1 = static_cast<float>(lon + 1) / nLon * 2.0f * pi;
                const int mat = colors[lon % 3];

                if (lat == 0) {
                    // Cap triangle: pole to two points on next latitude ring
                    addTriangle(
                        spherePoint(theta0, phi0),
                        spherePoint(theta1, phi1),
                        spherePoint(theta1, phi0),
                        mat
                    );
                } else {
                    // Two triangles forming a quad strip
                    const Vec3 p00 = spherePoint(theta0, phi0);
                    const Vec3 p10 = spherePoint(theta1, phi0);
                    const Vec3 p01 = spherePoint(theta0, phi1);
                    const Vec3 p11 = spherePoint(theta1, phi1);

                    addTriangle(p00, p01, p10, mat);
                    addTriangle(p01, p11, p10, mat);
                }
            }
        }
    }

    renderer.setScene(std::move(scene));

    // 4. Render 100 iterations
    renderer.setOutputOptions(OutputOptions{.linearity = OutputLinearity::Linear});
    renderer.setDebug(DebugOptions::Off);
    for (int i = 0; i < 100; ++i)
        renderer.render();
    renderer.createPixels();

    // 5. Save image
    const auto outputPath = ensureTestOutputFolder();
    renderer.saveImage(outputPath / "triangle_umbrella.png");

    // 6. Compare against reference image
    const auto inputFolder = std::filesystem::path("test_input");

    const auto refPath = inputFolder / "triangle_umbrella.png";
    ASSERT_TRUE(std::filesystem::exists(refPath))
        << "Reference image '" << refPath << "' does not exist";

    int refW = 0, refH = 0, refChannels = 0;
    unsigned char *refData = stbi_load(refPath.string().c_str(), &refW, &refH, &refChannels, 4);
    ASSERT_NE(refData, nullptr) << "Failed to load reference image: " << refPath;
    ASSERT_EQ(refW, resolution.width);
    ASSERT_EQ(refH, resolution.height);

    const uint32_t *rendered = renderer.getPixels();

    // Compute per-channel mean absolute error (MAE) over all pixels,
    // tolerating Monte-Carlo noise.
    double totalError = 0.0;
    const int numPixels = resolution.width * resolution.height;

    for (int i = 0; i < numPixels; ++i) {
        const uint32_t rpx = rendered[i];
        const uint8_t rR = (rpx >>  0) & 0xff;
        const uint8_t rG = (rpx >>  8) & 0xff;
        const uint8_t rB = (rpx >> 16) & 0xff;

        const uint8_t eR = refData[i * 4 + 0];
        const uint8_t eG = refData[i * 4 + 1];
        const uint8_t eB = refData[i * 4 + 2];

        totalError += std::abs(static_cast<int>(rR) - static_cast<int>(eR));
        totalError += std::abs(static_cast<int>(rG) - static_cast<int>(eG));
        totalError += std::abs(static_cast<int>(rB) - static_cast<int>(eB));
    }

    stbi_image_free(refData);

    std::cout << "Total error: " << totalError << std::endl;

    const double mae = totalError / (numPixels * 3.0);
    // With 100 render iterations the noise should be low;
    EXPECT_LE(mae, 1.5)
        << "Rendered image differs too much from reference (MAE=" << mae << "/255)";
}