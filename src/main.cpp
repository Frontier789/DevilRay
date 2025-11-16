#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <filesystem>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <deque>

#include "Application.hpp"
#include "Shaders.hpp"
#include "Image.hpp"
#include "Utils.hpp"
#include "Renderer.hpp"
#include "tracing/Material.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"

/*
    PLAN
     - bidirectional path tracing
        - need to be able to sample light sources
     - importance sampling
        - report and use pdf
     - metropolis
     - keep track of variance
        - add option to show it
     - stratify sample pixels
*/

void onKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_Q)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

template<typename DrawCallback>
void runEventLoop(Application &app, DrawCallback drawCallback)
{
    using namespace std::chrono;
    auto lastFpsPrint = steady_clock::now();
    int framesSinceLastPrint = 0;
    std::string fpsString = "? fps";

    glfwSetKeyCallback(app.window, onKeyEvent);

    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ImGui demo window (for testing purposes)
        // ImGui::ShowDemoWindow();

        ImGui::Text("%s", fpsString.c_str());

        drawCallback();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(app.window);

        ++framesSinceLastPrint;
        auto t = steady_clock::now();
        if (duration_cast<milliseconds>(t - lastFpsPrint) > 1s)
        {
            fpsString = std::to_string(framesSinceLastPrint) + " fps";
            framesSinceLastPrint = 0;
            lastFpsPrint += 1s;
        }
    }
}

const char* vertexShaderSource = R"glsl(
#version 430 core

out vec2 tex_coord;

void main() {
    const vec2 vertices[4] = vec2[4](
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0)
    );

    tex_coord = vertices[gl_VertexID]*vec2(1,-1)/2.0 + 0.5;

    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 430 core

in vec2 tex_coord;

layout(location = 0)
uniform sampler2D tex;

out vec4 frag_color;

void main() {
    frag_color = texture(tex, tex_coord);
}
)glsl";


GLuint createTexture(Size2i resolution)
{
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);

    glTextureStorage2D(texture, 1, GL_RGBA8, resolution.width, resolution.height);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    return texture;
}

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

Scene createScene()
{
    Scene scene;

    const int  light3 = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{3,3,3},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0};
        scene.materials.push_back(material);
    }

    const int  light_mid = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{4,4,4},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0},
        scene.materials.push_back(material);
    }

    const int  light_bright = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{100,100,100},
            .diffuse_reflectance = Vec4{1.0,1.0,1.0, 0.0},
        };
        material.debug_color = Vec4{0.9, 0.3, 0.4, 0.0},
        scene.materials.push_back(material);
    }

    const int  red = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.9, 0.3, 0.3, 0.0},
        };
        material.debug_color = Vec4{0.8, 0.2, 0.2, 0.0},
        scene.materials.push_back(material);
    }

    const int  green = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.3, 0.9, 0.3, 0.0},
        };
        material.debug_color = Vec4{0.2, 0.8, 0.2, 0.0},
        scene.materials.push_back(material);
    }

    const int  blue = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.3, 0.3, 0.9, 0.0},
        };
        material.debug_color = Vec4{0.2, 0.2, 0.8, 0.0},
        scene.materials.push_back(material);
    }

    const int  white = scene.materials.size();
    {
        auto material = DiffuseMaterial{
            .emission = Vec4{0, 0, 0},
            .diffuse_reflectance = Vec4{0.9, 0.9, 0.9, 0.0},
        };
        material.debug_color = Vec4{0.7, 0.7, 0.7, 0.0},
        scene.materials.push_back(material);
    }

    const int  glass = scene.materials.size();
    {
        auto material = TransparentMaterial{
            .inside_medium = Medium{.ior = 1.5595f},
        };
        material.debug_color = Vec4{0.9, 0.9, 0.9, 0.0},
        scene.materials.push_back(material);
    }

    const int  air = scene.materials.size();
    {
        auto material = TransparentMaterial{
            .inside_medium = Medium{.ior = 1.0f},
        };
        material.debug_color = Vec4{0.9, 0.9, 0.9, 0.0},
        scene.materials.push_back(material);
    }

    // Avoid GCC false positive warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
/*
    scene.objects.push_back(Sphere{
        .center = Vec3{0, 0, 2},
        .radius = 300e-3,
        .mat = blue,
    });

    scene.objects.push_back(Sphere{
        .center = Vec3{0.3, 0.2, 2},
        .radius = 100e-3,
        .mat = blue,
    });

    scene.objects.push_back(Sphere{
        .center = Vec3{-.3, .2, 2},
        .radius = 100e-3,
        .mat = blue,
    });
*/
    {
        auto obj = Square{
            .p = Vec3{0,0,2.5},
            .n = Vec3{0,0,-1},
            .right = Vec3{1,0,0},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));    
    }

    {
        auto obj = Square{
            .p = Vec3{0.5,0,2},
            .n = Vec3{1,0,0},
            .right = Vec3{0,1,0},
            .size = 1,
        };
        obj.mat = red;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{-0.5,0,2},
            .n = Vec3{1,0,0},
            .right = Vec3{0,1,0},
            .size = 1,
        };
        obj.mat = green;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,0.5,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,-0.5,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 1,
        };
        obj.mat = white;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Square{
            .p = Vec3{0,0.499,2},
            .n = Vec3{0,1,0},
            .right = Vec3{0,0,1},
            .size = 0.5,
        };
        obj.mat = light_mid;
        scene.objects.push_back(std::move(obj));
    }

    {
        auto obj = Sphere{
            .center = Vec3{0.3, -0.4, 2},
            .radius = 100e-3,
        };
        obj.mat = blue;
        scene.objects.push_back(std::move(obj));
    }

    // for (int x=0;x<10;++x) {
    //     for (int y=0;y<10;++y) {
    //         {
    //             float h = std::sin(x*152.48548 + y*1867.8613385 + 5168.4185674);
    //             auto obj = Sphere{
    //                 .center = Vec3{(x - 9/2.f) / 10.0f *0.8f, 0.1f + h*0.1f, 1.6f + y / 10.0f *0.8f},
    //                 .radius = 40e-3,
    //             };
    //             obj.mat = glass;
    //             scene.objects.push_back(std::move(obj));
    //         }
    //     }
    // }

    {
        auto obj = Sphere{
            .center = Vec3{-0.25, -0.3, 1.8},
            .radius = 200e-3,
        };
        obj.mat = glass;
        scene.objects.push_back(std::move(obj));
    }


    #pragma GCC diagnostic pop

    return scene;
}

std::string counterToString(uint64_t cntr)
{
    const auto original = cntr;

    for (const auto ending : {"", "K", "M", "G", "T"}) {
        if (cntr < 1000) return std::to_string(cntr) + ending;

        cntr /= 1000;
    }

    return std::to_string(cntr);
}

struct RunningAverage
{
    RunningAverage(int sampleCount) : m_sampleCount(sampleCount) {}

    void add(double value) {
        m_samples.push_back(value);
        if (m_samples.size() > m_sampleCount) {
            m_samples.pop_front();
        }
    }

    double mean() const {
        if (m_samples.empty()) return 0;

        return std::accumulate(m_samples.begin(), m_samples.end(), 0.0) / m_samples.size();
    }

    void reset() {m_samples.clear();}

private:
    std::deque<double> m_samples;
    int m_sampleCount;
};

int main() {
    printCudaDeviceInfo();

    const auto resolution = Size2i{640, 640};
    const auto render_scale = 1;

    auto app = initApplication(resolution);

    try
    {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

        const auto shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        const auto texture = createTexture(resolution / render_scale);

        GLuint vao;
        glGenVertexArrays(1, &vao);

        Renderer renderer(resolution / render_scale);
        renderer.setPixelSampling(PixelSampling::UniformRandom);

        RunningAverage renderTimes(20);

        float focal_length_mm = 14.2f;
        bool debug = false;
        bool useCuda = true;
        int pixel_sampling = static_cast<int>(PixelSampling::UniformRandom);

        renderer.setScene(createScene());
        renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000, 3.72e-6 * 4 * render_scale));
        renderer.setDebug(debug);
        renderer.useCudaDevice(useCuda);

        runEventLoop(app, [&]{

            if (ImGui::SliderFloat("Focal length", &focal_length_mm, 3, 150, "%.1f mm")) {
                renderTimes.reset();
                renderer.clear();
                renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000, 3.72e-6 * 4 * render_scale));
            }

            if (ImGui::Checkbox("Debug", &debug))
            {
                renderTimes.reset();
                renderer.clear();
                renderer.setDebug(debug);
            }

            ImGui::SameLine();
            if (ImGui::Checkbox("Use CUDA", &useCuda))
            {
                renderTimes.reset();
                renderer.clear();
                renderer.useCudaDevice(useCuda);
            }

            constexpr const char* pixel_sampling_names[] = {"Center", "UniformRandom"};
            if (ImGui::Combo("Pixel Sampling", &pixel_sampling, pixel_sampling_names, IM_ARRAYSIZE(pixel_sampling_names)))
            {
                renderTimes.reset();
                renderer.clear();
                renderer.setPixelSampling(static_cast<PixelSampling>(pixel_sampling));
            }

            Timer t;
            renderer.render();
            const auto elapsed_ms = t.elapsed_seconds() * 1000;
            renderTimes.add(elapsed_ms);

            ImGui::Text("Render pass: %.1fms", renderTimes.mean());

            auto &buffers = renderer.getBuffers();
            if (useCuda) {
                buffers.casts.updateHostData();
            }
            ImGui::Text("Rays per pixel: %s", counterToString(buffers.totalCasts() / resolution.width / resolution.height).c_str());
            
            glTextureSubImage2D(texture, 0, 0, 0, resolution.width / render_scale, resolution.height / render_scale, GL_RGBA, GL_UNSIGNED_BYTE, renderer.getPixels());

            if (ImGui::Button("Capture snapshot"))
            {
                const auto imageFolder = std::filesystem::path{"captures"};
                (void)std::filesystem::create_directory(imageFolder);

                std::time_t time = std::time({});
                char timeString[100];
                std::strftime(timeString, 100, "%Y_%m_%d_%H_%M.png", std::gmtime(&time));
                std::string timePng = timeString;

                std::cout << "Saving to " << imageFolder / timePng << std::endl;

                renderer.saveImage(imageFolder / timePng);
            }

            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glBindVertexArray(vao);
            glUseProgram(shader);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        });

        glDeleteProgram(shader);
    }
    catch (std::exception &e)
    {
        std::cout << "Got exception, closing window: " << e.what() << std::endl;
        closeApplication(app);

        std::throw_with_nested(e);
    }
}
