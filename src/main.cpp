#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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
#include <filesystem>

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


struct Intrinsics
{
    float focal_length;
    Vec2 center;
};


struct Camera
{
    Intrinsics intrinsics;
    Size2i resolution;
    Size2f physical_pixel_size;
};

inline Ray cameraRay(const Camera &cam, Vec2 pixelCoord)
{
    const auto pixelCenter = pixelCoord + Vec2{0.5, 0.5};
    const auto physicalPixelCenter = pixelCenter * cam.physical_pixel_size - cam.intrinsics.center;

    const auto dir = physicalPixelCenter / cam.intrinsics.focal_length;

    return Ray{
        .p = Vec3{0,0,0},
        .v = Vec3{dir.x, dir.y, 1}
    };
}

struct Material
{
    Vec4 emission;
    Vec4 diffuse_reflectance;
    Vec4 debug_color;
};

struct Square
{
    Vec3 p;
    Vec3 n;
    Vec3 right;
    float size;
    Material *mat;
};

struct Sphere
{
    Vec3 center;
    float radius;
    Material *mat;
};

using Object = std::variant<Square, Sphere>;

struct Intersection
{
    float t;
    Vec3 p;
    Vec2 uv;
    Vec3 n;
    Material *mat;
};



Vec3 uniformHemisphereSample(const Vec3 &normal, Random &r)
{
    const float theta0 = 2 * std::numbers::pi_v<float> * r.rnd();
    const float theta1 = std::acos(1 - 2 * r.rnd());

    const float x = std::sin(theta1) * std::sin(theta0);
    const float y = std::sin(theta1) * std::cos(theta0);
    const float z = std::cos(theta1);

    const auto v = Vec3{x,y,z};

    if (dot(v, normal) < 0) return v*-1;

    return v;
}

std::optional<Intersection> getIntersection(const Ray &ray, const Square &square)
{
    // <p + v*t - o, n> = 0
    // <po, n> + <v,n>*t = 0


    const auto vn = dot(ray.v, square.n);
    if (std::abs(vn) < std::numeric_limits<float>::epsilon()) return std::nullopt;

    const auto t = -dot(ray.p - square.p, square.n) / vn;

    // std::cout << ray.p.x << "," << ray.p.y << "," << ray.p.z << " -> "  << ray.v.x << "," << ray.v.y << "," << ray.v.z << " -> t=" << t << std::endl;
    if (t < std::numeric_limits<float>::epsilon()) return std::nullopt;

    const auto p = ray.p + ray.v * t;
    const auto s = square.size / 2;

    const auto dx = dot(square.right, p - square.p);
    if (std::abs(dx) > s) return std::nullopt;

    const auto up = square.right.cross(square.n);
    const auto dy = dot(up, p - square.p);
    if (std::abs(dy) > s) return std::nullopt;

    const auto uv = Vec2{
        dx / s / 2 + 0.5f,
        dy / s / 2 + 0.5f,
    };

    const auto n = vn < 0 ? square.n : square.n * -1;

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
        .n = n,
        .mat = square.mat,
    };
}

std::optional<Intersection> getIntersection(const Ray &ray, const Sphere &sphere)
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

    const float q = -0.5f * ((b > 0) ? (b + d) : (b - d));
    const float t1 = q / a;
    const float t2 = c / q;

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

    auto n = (p - sphere.center).normalized();
    if (c < -1e-8) n = n * -1;

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
        .n = n,
        .mat = sphere.mat,
    };
}

struct RenderStats
{
    std::atomic<int64_t> total_casts;
};


class Renderer
{
    std::vector<Vec4> accumulator;
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

    std::optional<Intersection> cast(const Ray &ray)
    {
        std::optional<Intersection> best = std::nullopt;

        for (const auto &obj : objects)
        {
            const auto intersection = std::visit([&](auto&& o) {return getIntersection(ray, o);}, obj);

            if (!intersection.has_value()) continue;

            if (!best.has_value() || best->t > intersection->t)
            {
                best = intersection;
            }
        }

        return best;
    }

    void render()
    {
        parallel_for(resolution.height, [&](int y){

        Random random = RandomPool::singleton().borrowRandom();
        //std::cout << "Got random id " << random.get_id() << std::endl;

        const int max_depth = debug ? 1 : 5;

        int ray_casts = 0;

        for (int x=0;x<resolution.width;++x) for (int iterations=0;iterations<10;++iterations)
        {
            auto &pix = accumulator[x + y*resolution.width];
            pix.w += 1;


            auto ray = cameraRay(camera, Vec2{static_cast<float>(x), static_cast<float>(y)});
            Vec4 total_transmission{1,1,1,0};

            for (int depth=0;depth<max_depth;++depth)
            {
                const auto intersection = cast(ray);
                ++ray_casts;

                // if (x == resolution.width/3 && y == resolution.height/3) {
                //     std::cout << "intersected with object at " << intersection->p << " t=" << intersection->t << std::endl;
                // }
                if (!intersection.has_value()) break;

                if (dot(intersection->n, ray.v) > 1e-5) {
                    std::cout << "Hit something from the back?" << std::endl;
                    std::cout << "\tDistance to ball: " << std::sqrt(dot(intersection->p - Vec3(0,0,4.5), intersection->p - Vec3(0,0,4.5))) << std::endl;
                    std::cout << "\tDot is " << dot(intersection->n, ray.v) << std::endl;
                    std::cout << "\tpos=" << intersection->p << " t=" << intersection->t << std::endl;
                    // pix.y += 10000;
                }

                if (dot(intersection->p - ray.p, intersection->p - ray.p) < 1e-12) {
                    ray.p = ray.p + intersection->n * 1e-6;
                    continue;
                    // std::cout << "Self intersection!" << std::endl;
                    // std::cout << "\tpos=" << intersection->p << " t=" << intersection->t << std::endl;
                    // pix.x += 10000;
                }

                const auto &material = *intersection->mat;

                if (debug)
                {
                    pix = pix + material.debug_color * checkerPattern(intersection->uv, 8);
                }
                else
                {
                    pix = pix + material.emission * total_transmission;
                }

                const auto new_v = uniformHemisphereSample(intersection->n, random);

                const auto weakening_factor = dot(intersection->n, new_v);
                total_transmission = total_transmission * weakening_factor * material.diffuse_reflectance;

                ray = Ray{
                    .p = intersection->p,
                    .v = new_v
                };

                ray.p = ray.p + intersection->n * 1e-4;
            }
        }
	    RandomPool::singleton().returnRandom(std::move(random));
        stats.total_casts += ray_casts;

	    });
    }

    void clear()
    {
        std::fill_n(accumulator.begin(), accumulator.size(), Vec4{0,0,0,0});
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

        for (int y=0;y<resolution.height;++y)
        for (int x=0;x<resolution.width;++x)
        {
            const auto pix = accumulator[x + y*resolution.width];
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
        .diffuse_reflectance = Vec4{1.0,0.8,0.8,0.0},
        .debug_color = Vec4{1.0,0.8,0.7,0.0},
    };

    static Material green{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.8,1.0,0.8,0.0},
        .debug_color = Vec4{0.7,1.0,0.8,0.0},
    };

    static Material blue{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{0.9,0.9,1.0,0.0},
        .debug_color = Vec4{0.7,0.8,1.0,0.0},
    };

    static Material white{
        .emission = Vec4{0, 0, 0},
        .diffuse_reflectance = Vec4{1.0,0.97,0.92,0.0},
        .debug_color = Vec4{0.9,0.9,0.9,0.0},
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
