#pragma once

#include <chrono>
#include <random>
#include <mutex>
#include <thread>


class Timer
{
    std::chrono::steady_clock::time_point start_time;

public:
    Timer();
    float elapsed_seconds() const;
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

std::ostream &operator<<(std::ostream &os, const ColorRGBA8 &color);
std::ostream &operator<<(std::ostream &os, const Vec4 &vec);
std::ostream &operator<<(std::ostream &os, const Vec3 &vec);
std::ostream &operator<<(std::ostream &os, const Vec2 &vec);
std::ostream &operator<<(std::ostream &os, const Ray &ray);

template<typename T>
std::ostream &operator<<(std::ostream &os, const Size2<T> &size) {
    os << "(" << size.width << ", " << size.height << ")";
    return os;
}


Vec4 checkerPattern(
    const Vec2 &uv, 
    const int checker_count, 
    const Vec4 dark=Vec4{0.5,0.5,0.5,0}, 
    const Vec4 bright=Vec4{0.8,0.8,0.8,0}
);
