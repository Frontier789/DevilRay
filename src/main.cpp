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
    Vec4 operator*(const float f) const {return Vec4{x*f, y*f, z*f, w*f};}
};

struct Vec3
{
    float x;
    float y;
    float z;

    Vec3 operator-(const Vec3 &v) const {return Vec3{x-v.x, y-v.y, z-v.z};}
    Vec3 operator+(const Vec3 &v) const {return Vec3{x+v.x, y+v.y, z+v.z};}
    Vec3 operator*(const float f) const {return Vec3{x*f, y*f, z*f};}

    Vec3 normalized() const {
        const auto length = std::sqrt(x*x + y*y + z*z);
        return Vec3{x/length, y/length, z/length};
    }

    Vec3 cross(const Vec3 &v) const {
        return Vec3{
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x,
        };
    }
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

std::ostream &operator<<(std::ostream &os, const ColorRGBA8 &color) {
    os << "(" << static_cast<int>(color.r) << ", " 
       << static_cast<int>(color.g) << ", " 
       << static_cast<int>(color.b) << ", " 
       << static_cast<int>(color.a) << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Vec4 &vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Vec3 &vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Vec2 &vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Size2<T> &size) {
    os << "(" << size.width << ", " << size.height << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    os << "Ray(p=" << ray.p << ", v=" << ray.v << ")";
    return os;
}


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

static int id_counter = 0;

struct Random
{
    Random() : id{id_counter++}, generator(13), distribution_0_1(0, 1) {}

    float rnd()
    {
        return distribution_0_1(generator);
    }

    Random(const Random &) = delete;
    Random(Random &&) = default;
	
    int get_id() {return id;}
private:
    int id;
    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution_0_1;
};


struct RandomPool
{
	Random borrowRandom();
	void returnRandom(Random r);
private:
    std::vector<Random> m_randoms;
	std::mutex m_mutex;
};

Random RandomPool::borrowRandom()
{
	std::lock_guard<std::mutex> guard(m_mutex);

	if (m_randoms.empty()) m_randoms.push_back(Random{});
	Random r = std::move(m_randoms.back());
	m_randoms.pop_back();
	
	return r;
}

void RandomPool::returnRandom(Random r)
{
	std::lock_guard<std::mutex> guard(m_mutex);
	
	m_randoms.emplace_back(std::move(r));	
}

RandomPool randomPool;

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

    auto n = (p - sphere.center).normalized();
    if (tmin < 0) n = n * -1;

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
        .n = n,
        .mat = sphere.mat,
    };
}

Vec4 calculate_color(const Intersection &intersection)
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
    } * intersection.mat->debug_color;
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


template <typename F>
void parallel_for(int range, const F &func) {
    auto num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    
    int chunk_size = (range + num_threads - 1) / num_threads;

    std::vector<std::jthread> threads;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, range);
        
        threads.emplace_back([start, end, &func]{
            for (int i = start; i < end; ++i) {
                func(i);
            }
        });
    }
}


Vec4 checkerPattern(
    const Vec2 &uv, 
    const int checker_count, 
    const Vec4 dark=Vec4{0.5,0.5,0.5,0}, 
    const Vec4 bright=Vec4{0.8,0.8,0.8,0}
){
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;

    const float checker = checker_x ^ checker_y;

    return bright * checker + dark * (1-checker);
}


class Renderer
{
    std::vector<Vec4> accumulator;
    std::vector<uint32_t> pixels;
    Size2i resolution;
    GLuint texture;
    bool debug{true};

    Camera camera;
    std::vector<Object> objects;

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

    std::optional<Intersection> trace(const Ray &ray)
    {
        std::optional<Intersection> best = std::nullopt;

        for (const auto &obj : objects)
        {
            const auto intersection = std::visit([&](auto&& concrete) {return getIntersection(ray, concrete);}, obj);

            if (!intersection.has_value()) continue;

            // TRICK: avoid hitting the object the ray is leaving
            // if (intersection->t < 1e-6) continue;
            
            if (!best.has_value() || best->t > intersection->t)
            {
                best = intersection;
            }
        }

        return best;
    }

    void render()
    {
        // for (int y=0;y<resolution.height;++y) {
        parallel_for(resolution.height, [&](int y){
        
        Random random = randomPool.borrowRandom();
        //std::cout << "Got random id " << random.get_id() << std::endl;

        const int max_depth = debug ? 1 : 10;

        for (int x=0;x<resolution.width;++x) // for (int iterations=0;iterations<10;++iterations)
        {
            auto &pix = accumulator[x + y*resolution.width];
            pix.w += 1;
            

            auto ray = camera.sampleRay(x, y);
            Vec4 total_transmission{1,1,1,0};
            
            for (int depth=0;depth<max_depth;++depth)
            {
                const auto intersection = trace(ray);
                // if (x == resolution.width/3 && y == resolution.height/3) {
                //     std::cout << "intersected with object at " << intersection->p << " t=" << intersection->t << std::endl;
                // }
                if (!intersection.has_value()) break;

                // if (dot(intersection->n, ray.v) > 0) {
                //     // std::cout << "Hit something from the back?" << std::endl;
                //     pix.y += 10000;
                // }

                // if (dot(intersection->p - ray.p, intersection->p - ray.p) < 1e-12) {
                //     // std::cout << "Self intersection!" << std::endl;
                //     // std::cout << "\tpos=" << intersection->p << " t=" << intersection->t << std::endl;
                //     pix.x += 10000;
                // }

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
	randomPool.returnRandom(std::move(random));
	}
        );
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
            if (pix.w == 0)
                setPixel(x, y, 0,0,0);
            else 
                setPixel(x, y, pix.x / pix.w, pix.y / pix.w, pix.z / pix.w);
        }
        
        glTextureSubImage2D(texture, 0, 0, 0, resolution.width, resolution.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    GLuint tex() const {return texture;}
};

Camera createCamera(Size2i resolution, const float focal_length)
{
    const float physical_pixel_size = 3.72e-6 * 4;
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

std::vector<Object> createObjects()
{
    static Material light3{
        .emission = Vec4{3,3,3},
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
        .center = Vec3{0, 0, 4},
        .radius = 300e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{0.3, 0.2, 4},
        .radius = 100e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{-.3, .2, 4},
        .radius = 100e-3,
        .mat = &blue,
    });
*/
    objects.emplace_back(Square{
        .p = Vec3{0,0,5},
        .n = Vec3{0,0,-1},
        .right = Vec3{1,0,0},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0.5,0,4.5},
        .n = Vec3{1,0,0},
        .right = Vec3{0,1,0},
        .size = 1,
        .mat = &red,
    });

    objects.emplace_back(Square{
        .p = Vec3{-0.5,0,4.5},
        .n = Vec3{1,0,0},
        .right = Vec3{0,1,0},
        .size = 1,
        .mat = &green,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,0.5,4.5},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,-0.5,4.5},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 1,
        .mat = &white,
    });

    objects.emplace_back(Square{
        .p = Vec3{0,0.499,4.5},
        .n = Vec3{0,1,0},
        .right = Vec3{0,0,1},
        .size = 0.2,
        .mat = &light_bright,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{0.3, -0.4, 4.5},
        .radius = 100e-3,
        .mat = &blue,
    });

    objects.emplace_back(Sphere{
        .center = Vec3{-0.25, -0.3, 4.3},
        .radius = 200e-3,
        .mat = &red,
    });
    
    #pragma GCC diagnostic pop

    return objects;
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

        float focal_length_mm = 33;
        bool debug = true;

        renderer.setObjects(createObjects());
        renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000));
        
        runEventLoop(app, [&]{
            
            if (ImGui::SliderFloat("Focal length", &focal_length_mm, 3, 75, "%.0f mm")) {
                renderer.setCamera(createCamera(renderer.getResolution(), focal_length_mm / 1000));
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
            renderer.upload();

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
