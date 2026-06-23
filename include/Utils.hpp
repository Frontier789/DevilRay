#pragma once

#include <chrono>
#include <random>
#include <mutex>
#include <thread>

#ifndef __CUDACC__
    #define __global__
    #define __device__
    #define __host__
    #define HD
#else
    #define HD __host__ __device__
#endif

struct Timer
{
    Timer();

    float elapsed_seconds() const;

private:
    std::chrono::steady_clock::time_point start_time;
};

constexpr float pi = std::numbers::pi_v<float>;

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

    static int id_counter;
};


struct RandomPool
{
	Random borrowRandom();
	void returnRandom(Random r);

    static RandomPool &singleton();
private:
    std::vector<Random> m_randoms;
	std::mutex m_mutex;
};


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

    constexpr Vec4 operator+(const Vec4 &v) const {return Vec4{x+v.x, y+v.y, z+v.z, w+v.w};}
    constexpr Vec4 operator*(const Vec4 &v) const {return Vec4{x*v.x, y*v.y, z*v.z, w*v.w};}
    constexpr Vec4 operator*(const float f) const {return Vec4{x*f, y*f, z*f, w*f};}
    constexpr Vec4 operator/(const float f) const {return *this * (1.0f / f);}
    constexpr float max() const {return std::max(std::max(x,y),z);}
};

struct Vec3
{
    float x;
    float y;
    float z;

    constexpr Vec3 operator-(const Vec3 &v) const {return Vec3{x-v.x, y-v.y, z-v.z};}
    constexpr Vec3 operator+(const Vec3 &v) const {return Vec3{x+v.x, y+v.y, z+v.z};}
    constexpr Vec3 operator*(const Vec3 &v) const {return Vec3{x*v.x, y*v.y, z*v.z};}
    constexpr Vec3 operator*(const float f) const {return Vec3{x*f, y*f, z*f};}
    constexpr Vec3 operator/(const float f) const {return *this*(1.0f/f);}

    constexpr float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    constexpr Vec3 &operator+=(const Vec3 &v) { return *this = *this + v; }

    constexpr Vec3 normalized() const {
        const auto l = length();
        return Vec3{x/l, y/l, z/l};
    }

    constexpr Vec3 cross(const Vec3 &v) const {
        return Vec3{
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x,
        };
    }

    constexpr float dot(const Vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    constexpr bool anyNan() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    constexpr Vec3 inv() const {
        return Vec3{1/x, 1/y, 1/z};
    }
};

template<typename T>
struct Vec2
{
    T x;
    T y;

    constexpr Vec2 operator+(const Vec2 &v) const {return Vec2{x+v.x, y+v.y};}
    constexpr Vec2 operator-(const Vec2 &v) const {return Vec2{x-v.x, y-v.y};}
    constexpr Vec2 operator/(const Vec2 &v) const {return Vec2{x/v.x, y/v.y};}
    constexpr Vec2 operator*(const Vec2 &v) const {return Vec2{x*v.x, y*v.y};}
    constexpr Vec2 operator*(const T &f) const {return Vec2{x*f, y*f};}
    constexpr Vec2 operator/(const T &f) const {return Vec2{x/f, y/f};}

    template<typename U>
    constexpr operator Vec2<U>() const
    {
        return Vec2<U>(static_cast<U>(x), static_cast<U>(y));
    }
};

using Vec2i = Vec2<int>;
using Vec2f = Vec2<float>;

template<typename T>
struct Size2
{
    T width;
    T height;

    constexpr Size2 operator/(const T &v) const {return Size2{width / v, height / v};}

    constexpr T area() const {return width * height;}

    constexpr Vec2<T> toVec() const { return Vec2<T>{.x = width, .y = height}; }
};

using Size2i = Size2<int>;
using Size2f = Size2<float>;

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
struct AABB
{
    Vec3 min;
    Vec3 max;

    static constexpr AABB empty() {
        const auto float_max = std::numeric_limits<float>::max();
        return AABB{.min=Vec3{float_max, float_max, float_max}, .max=Vec3{-float_max, -float_max, -float_max}};
    }

    constexpr AABB extend(const Vec3 &p) const {
        AABB box = *this;

        box.min.x = std::min(box.min.x, p.x);
        box.min.y = std::min(box.min.y, p.y);
        box.min.z = std::min(box.min.z, p.z);

        box.max.x = std::max(box.max.x, p.x);
        box.max.y = std::max(box.max.y, p.y);
        box.max.z = std::max(box.max.z, p.z);

        return box;
    }
};

inline constexpr float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

namespace Resolutions
{
    inline Size2i vga() {return Size2i{640, 480};}
    inline Size2i hd() {return Size2i{1280, 720};}
    inline Size2i fhd() {return Size2i{1920, 1280};}
}

struct Ray
{
    Vec3 p;
    Vec3 v;
};

std::ostream &operator<<(std::ostream &os, const ColorRGBA8 &color);
std::ostream &operator<<(std::ostream &os, const Vec4 &vec);
std::ostream &operator<<(std::ostream &os, const Vec3 &vec);
std::ostream &operator<<(std::ostream &os, const Ray &ray);

template<typename T>
std::ostream &operator<<(std::ostream &os, const Size2<T> &size) {
    os << "(" << size.width << ", " << size.height << ")";
    return os;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Vec2<T> &vec) {
    os << "(" << vec.x << ", " << vec.y << ")";
    return os;
}

template<class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;


inline constexpr float triangleArea(const Vec3 &A, const Vec3 &B, const Vec3 &C)
{
    const auto u = A - B;
    const auto v = A - C;

    return u.cross(v).length() / 2;
}

inline constexpr float luminance(const Vec4 &linear_rgb)
{
    return 0.2126f * linear_rgb.x + 0.7152f * linear_rgb.y + 0.0722f * linear_rgb.z;
}

inline constexpr float powerHeuristic(const float pdf_0, const float pdf_1)
{
    return pdf_0*pdf_0 / (pdf_0*pdf_0 + pdf_1*pdf_1);
}

inline constexpr float areaToSolidAngle(Vec3 from, Vec3 to, Vec3 to_n)
{
    const auto diff = to - from;
    const auto distance_squared = diff.dot(diff);

    const auto dir = diff.normalized();

    const auto cos_theta = std::abs(dot(dir, to_n));
    if (cos_theta < 1e-6f) return 0.0f;

    return distance_squared / cos_theta;
};

inline constexpr float cosineWeightedHemispherePdf(Vec3 current_vertex_position, Vec3 next_vertex_position, Vec3 normal)
{
    const auto dir = (next_vertex_position - current_vertex_position).normalized();
    const auto cos_phi = dot(dir, normal);
    return cos_phi > 0 ? cos_phi / pi : 0.0f;
}
