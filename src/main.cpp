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

#include "Application.hpp"
#include "Shaders.hpp"
#include "Image.hpp"
#include "Utils.hpp"
#include "DeviceUtils.hpp"
#include "tracing/Material.hpp"
#include "tracing/Camera.hpp"
#include "tracing/Objects.hpp"
#include "tracing/Intersection.hpp"
#include "tracing/SampleScene.hpp"

/*
    PLAN
     - add cuda support
        - also keep supporting cpu
     - bidirectional path tracing
        - need to be able to sample light sources
     - importance sampling
        - report and use pdf
     - metropolis
     - keep track of variance
        - add option to show it
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

    return texture;
}

struct RenderStats
{
    std::atomic<int64_t> total_casts;
};

class Renderer
{
    DeviceVector<Vec4> accumulator;
    std::vector<uint32_t> pixels;
    Size2i resolution;
    GLuint texture;
    bool debug;

    RenderStats stats;

    Camera camera;
    std::vector<Object> objects;

    Timer timer;
public:

    Renderer(Size2i resolution)
        : accumulator(resolution.width * resolution.height, Vec4{0,0,0,0})
        , pixels(resolution.width * resolution.height)
        , resolution(resolution)
        , stats(RenderStats{0})
    {
        texture = createTexture(resolution);
    }

    const RenderStats &getStats() const {return stats;}

    Size2i getResolution() const {return resolution;}

    void setDebug(bool dbg)
    {
        debug = dbg;
    }

    void setCamera(Camera cam)
    {
        camera = std::move(cam);
    }

    void setObjects(std::vector<Object> objs)
    {
        objects = std::move(objs);
    }

    void render()
    {
        parallel_for(resolution.height, [&](int y){

            Random random = RandomPool::singleton().borrowRandom();
            //std::cout << "Got random id " << random.get_id() << std::endl;

            const int max_depth = debug ? 1 : 5;

            int ray_casts = 0;

            for (int x=0;x<resolution.width;++x)
            {
                const auto iterations = 10;
                
                auto &pix = accumulator.hostPtr()[x + y*resolution.width];
                pix.w += iterations;

                const auto ray = cameraRay(camera, Vec2{x, y});
                const auto sample = sampleColor(ray, max_depth, std::span{objects}, debug, iterations, random);

                pix = pix + sample.color;
                ray_casts += sample.casts;
            }
            RandomPool::singleton().returnRandom(std::move(random));
            stats.total_casts += ray_casts;

	    });
    }

    void clear()
    {
        std::fill_n(accumulator.hostPtr(), accumulator.size(), Vec4{0,0,0,0});
    }

    void createPixels()
    {
        auto toSRGB = [](float linear) -> float
        {
            const float r = std::clamp(linear, 0.f, 1.f);
            if (r < 0.0031308) return r * 12.92;
            return std::pow(r, 1/2.4f)*1.055 - 0.055f;
        };

        auto packPixel = [&](float r, float g, float b, float a) -> uint32_t {
            return (static_cast<uint32_t>(toSRGB(r) * 255)<< 0) |
                (static_cast<uint32_t>(toSRGB(g) * 255)<< 8) |
                (static_cast<uint32_t>(toSRGB(b) * 255)<<16) |
                (static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255)<<24);
        };

        auto setPixel = [&](int x, int y, float r, float g, float b, float a=1.0f) {
            const auto flipped_y = resolution.height - y - 1;
            pixels[x + flipped_y*resolution.width] = packPixel(r,g,b,a);
        };

        accumulator.updateHostData();

        for (int y=0;y<resolution.height;++y)
        for (int x=0;x<resolution.width;++x)
        {
            const auto pix = accumulator.hostPtr()[x + y*resolution.width];
            if (pix.w == 0)
                setPixel(x, y, 0,0,0);
            else
                setPixel(x, y, pix.x / pix.w, pix.y / pix.w, pix.z / pix.w);
        }
    }

    void saveImage(const std::string &fileName)
    {
        createPixels();

        savePNG(fileName, pixels, resolution);
    }

    void upload()
    {
        createPixels();

        glTextureSubImage2D(texture, 0, 0, 0, resolution.width, resolution.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    GLuint tex() const {return texture;}
};

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

std::vector<Object> createObjects()
{
    static Material light3{
        .emission = Vec4{3,3,3},
        .diffuse_reflectance = Vec4{1.0,1.0,1.0,0.0},
        .debug_color = Vec4{0.9,0.3,0.4,0.0},
    };
    static Material light_mid{
        .emission = Vec4{10,10,10},
        .diffuse_reflectance = Vec4{1.0,1.0,1.0,0.0},
        .debug_color = Vec4{0.9,0.3,0.4,0.0},
    };
    static Material light_bright{
        .emission = Vec4{100,100,100},
        .diffuse_reflectance = Vec4{1.0,1.0,1.0,0.0},
        .debug_color = Vec4{0.9,0.3,0.4,0.0},
    };

    static Material red{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.9, 0.3, 0.3, 0.0},
        .debug_color = Vec4{0.8, 0.2, 0.2, 0.0},
    };

    static Material green{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.3, 0.9, 0.3, 0.0},
        .debug_color = Vec4{0.2, 0.8, 0.2, 0.0},
    };

    static Material blue{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.3, 0.3, 0.9, 0.0},
        .debug_color = Vec4{0.2, 0.2, 0.8, 0.0},
    };

    static Material white{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.9, 0.9, 0.9, 0.0},
        .debug_color = Vec4{0.7, 0.7, 0.7, 0.0},
    };

    std::vector<Object> objects;

    // Avoid GCC false positive warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overflow"
/*
    objects.emplace_back(Sphere{
        .center = Vec3{0, 0, 2},
        .radius = 300e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{0.3, 0.2, 2},
        .radius = 100e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{-.3, .2, 2},
        .radius = 100e-3,
        .mat = &blue,
    });
*/
    objects.emplace_back(Square{
        .p = Vec3{0,0,2.5},
        .n = Vec3{0,0,-1},
        .right = Vec3{1,0,0},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0.5,0,2},
        .n = Vec3{1,0,0},
        .right = Vec3{0,1,0},
        .size = 1,
        .mat = &red,
    });

    objects.emplace_back(Square{
        .p = Vec3{-0.5,0,2},
        .n = Vec3{1,0,0},
        .right = Vec3{0,1,0},
        .size = 1,
        .mat = &green,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,0.5,2},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,-0.5,2},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,0.499,2},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 0.5,
        .mat = &light_mid,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{0.3, -0.4, 2},
        .radius = 100e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{-0.25, -0.3, 1.8},
        .radius = 200e-3,
        .mat = &red,
    });

    #pragma GCC diagnostic pop

    return objects;
}

void test_f();

int main() {
    test_f();

    const auto resolution = Size2i{640, 640};
    const auto render_scale = 1;

    auto app = initApplication(resolution);

    try
    {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

        const auto shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);

        GLuint vao;
        glGenVertexArrays(1, &vao);

        Renderer renderer(resolution / render_scale);

        float focal_length_mm = 14.2f;
        bool debug = false;

        renderer.setObjects(createObjects());
        renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000, 3.72e-6 * 4 * render_scale));
        renderer.setDebug(debug);

        runEventLoop(app, [&]{

            if (ImGui::SliderFloat("Focal length", &focal_length_mm, 3, 75, "%.1f mm")) {
                renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000, 3.72e-6 * 4 * render_scale));
                renderer.clear();
            }

            if (ImGui::Checkbox("Debug", &debug))
            {
                renderer.clear();
                renderer.setDebug(debug);
            }

            Timer t;
            renderer.render();
            const auto elapsed_ms = t.elapsed_seconds() * 1000;

            ImGui::Text("Render pass: %.0fms", elapsed_ms);
            ImGui::Text("Rays per pixel: %.0f", renderer.getStats().total_casts.load()*1.f / resolution.width / resolution.height);
            renderer.upload();

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
            glBindTexture(GL_TEXTURE_2D, renderer.tex());

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
