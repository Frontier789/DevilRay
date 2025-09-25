#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include "Application.hpp"
#include "Shaders.hpp"
#include <cmath>

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

    tex_coord = vertices[gl_VertexID]/2.0 + 0.5;
    
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

struct ColorRGBA8
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

static_assert(sizeof(ColorRGBA8) == sizeof(uint8_t)*4);


struct Vec4
{
    float x;
    float y;
    float z;
    float w;
    Vec4 operator+(const Vec4 &v) const {return Vec4{x+v.x, y+v.y, z+v.z, w+v.w};}
    Vec4 operator*(const Vec4 &v) const {return Vec4{x*v.x, y*v.y, z*v.z, w*v.w};}
};

struct Vec3
{
    float x;
    float y;
    float z;

    Vec3 operator-(const Vec3 &v) const {return Vec3{x-v.x, y-v.y, z-v.z};}
    Vec3 operator+(const Vec3 &v) const {return Vec3{x+v.x, y+v.y, z+v.z};}
    Vec3 operator*(const float f) const {return Vec3{x*f, y*f, z*f};}
};

struct Vec2
{
    float x;
    float y;
};

inline float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


template<typename T>
struct Size2
{
    T width;
    T height;
};

using Size2i = Size2<int>;
using Size2f = Size2<float>;

struct Ray
{
    Vec3 p;
    Vec3 v;
};


struct Intrinsics
{
    float focal_length_x;
    float focal_length_y;
    float center_x;
    float center_y;
};


GLuint createTexture(Size2i resolution)
{
    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);

    glTextureStorage2D(texture, 1, GL_RGBA8, resolution.width, resolution.height);

    return texture;
}


struct Camera
{
    Intrinsics intrinsics;
    Size2i resolution;
    Size2f physical_pixel_size;

    inline Ray sampleRay(float u, float v)
    {
        const float x = (u * physical_pixel_size.width - intrinsics.center_x) / intrinsics.focal_length_x;
        const float y = (v * physical_pixel_size.height - intrinsics.center_y) / intrinsics.focal_length_y;

        return Ray{
            .p = Vec3{0,0,0},
            .v = Vec3{x, y, 1}
        };
    }
};

struct Sphere
{
    Vec3 center;
    float radius;
    Vec4 color;
};


struct Intersection
{
    float t;
    Vec3 p;
    Vec2 uv;
};


std::optional<Intersection> intersect(const Ray &ray, const Sphere &sphere)
{
    // |p + v*t - o| = r
    // <po + v*t, po + v*t> = r^2
    // |po|^2 + 2<po, v> * t + |v|^2 * t^2 = r^2

    const auto o = ray.p - sphere.center;

    const auto a = dot(ray.v, ray.v);
    const auto b = 2*dot(o, ray.v);
    const auto c = dot(o,o) - sphere.radius * sphere.radius;

    const auto D = b*b - 4*a*c;
    if (D < 0) return std::nullopt;

    const auto d = std::sqrt(D);
    const auto t1 = (-b - d) / (2*a);
    const auto t2 = (-b + d) / (2*a);

    const auto tmin = std::min(t1, t2);
    const auto tmax = std::max(t1, t2);

    if (tmax < 0) return std::nullopt;
    
    const auto t = tmin < 0 ? tmax : tmin;
    const auto p = ray.p + ray.v * t;

    const auto pl = p - sphere.center;

    const auto uv = Vec2{
        .x = std::atan2(pl.z, pl.x) / std::numbers::pi_v<float> / 2 + 0.5f,
        .y = std::acos(pl.y / sphere.radius) / std::numbers::pi_v<float>,
    };

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
    };
}

Vec4 sphere_color(const Sphere &sphere, const Intersection &intersection)
{
    const int checker_count = 10;
    const int check_x = int(intersection.uv.x * checker_count) % 2;
    const int check_y = int(intersection.uv.y * checker_count) % 2;
    const int check = (check_y + check_x) % 2;

    const float intensity = check*0.7 + 0.3;

    return Vec4{
        .x = intensity,
        .y = intensity,
        .z = intensity,
        .w = 0,
    } * sphere.color;
}


class Timer
{
    std::chrono::steady_clock::time_point start_time;

public:
    Timer() : start_time(std::chrono::steady_clock::now()) {}

    float elapsed_seconds() const
    {
        const auto delta = std::chrono::steady_clock::now() - start_time;
        const auto mics = std::chrono::duration_cast<std::chrono::microseconds>(delta);
        return mics.count() / 1e6f;
    }
};


class Renderer
{
    std::vector<Vec4> accumulator;
    std::vector<uint32_t> pixels;
    Size2i resolution;
    GLuint texture;

    Camera camera;
    std::vector<Sphere> spheres;

    Timer timer;
public:

    Renderer(Size2i resolution) 
        : accumulator(resolution.width * resolution.height, Vec4{0,0,0,0})
        , pixels(resolution.width * resolution.height)
        , resolution(resolution)
    {
        texture = createTexture(resolution);
    }

    Size2i getResolution() const {return resolution;}

    void setCamera(Camera cam)
    {
        camera = std::move(cam);
    }

    void setSpheres(std::vector<Sphere> objs)
    {
        spheres = std::move(objs);
    }

    void render()
    {
        const auto sx = std::sin(timer.elapsed_seconds() / 2 * std::numbers::pi_v<float> * 2) * 0.03f;
        const auto sy = std::cos(timer.elapsed_seconds() / 2 * std::numbers::pi_v<float> * 2) * 0.03f;

        Sphere s{
            .center = Vec3{sx, sy, 4},
            .radius = 300e-3,
            .color = Vec4{0.8,0.9,1.0,0.0},
        };

        for (int y=0;y<resolution.height;++y)
        for (int x=0;x<resolution.width;++x)
        {
            const auto ray = camera.sampleRay(x, y);

            auto &pix = accumulator[x + y*resolution.width];
            pix.w += 1;

            const auto intersection = intersect(ray, s);
            if (intersection.has_value())
            {
                pix = pix + sphere_color(s, *intersection);
            }
        }
    }

    void clear()
    {
        std::fill_n(accumulator.begin(), accumulator.size(), Vec4{0,0,0,0});
    }

    void upload()
    {
        auto packPixel = [](float r, float g, float b, float a) -> uint32_t {
            return (static_cast<uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255)<< 0) | 
                (static_cast<uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255)<< 8) | 
                (static_cast<uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255)<<16) | 
                (static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255)<<24);
        };

        auto setPixel = [&](int x, int y, float r, float g, float b, float a=1.0f) {
            pixels[x + y*resolution.width] = packPixel(r,g,b,a);
        };

        for (int y=0;y<resolution.height;++y)
        for (int x=0;x<resolution.width;++x)
        {
            const auto pix = accumulator[x + y*resolution.width];
            setPixel(x, y, pix.x / pix.w, pix.y / pix.w, pix.z / pix.w);
        }
        
        glTextureSubImage2D(texture, 0, 0, 0, resolution.width, resolution.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    GLuint tex() const {return texture;}
};

Camera createCamera(Size2i resolution)
{
    const float physical_pixel_size = 3.72e-6 * 4;
    const float focal_length = 50e-3;
    Camera cam{
        .intrinsics = Intrinsics{
            .focal_length_x = focal_length,
            .focal_length_y = focal_length,
            .center_x = resolution.width/2.f * physical_pixel_size,
            .center_y = resolution.height/2.f * physical_pixel_size,
        },
        .resolution = resolution,
        .physical_pixel_size = Size2f{physical_pixel_size, physical_pixel_size},
    };

    return cam;
}

int main() {
    auto app = initApplication();
    
    try
    {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
        const auto shader = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    
        GLuint vao;
        glGenVertexArrays(1, &vao);
        
        Renderer renderer(Size2i{800, 600});

        renderer.setCamera(createCamera(renderer.getResolution()));
        renderer.render();
        renderer.upload();

        int cntr = 0;
        
        runEventLoop(app, [&]{

            renderer.render();
            
            if (cntr % 3 == 0) {
                renderer.upload();
                renderer.clear();
            }
            ++cntr;

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