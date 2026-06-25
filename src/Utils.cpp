#include "Utils.hpp"


Timer::Timer() : start_time(std::chrono::steady_clock::now()) {}

float Timer::elapsedSeconds() const
{
    const auto delta = std::chrono::steady_clock::now() - start_time;
    const auto mics = std::chrono::duration_cast<std::chrono::microseconds>(delta);
    return mics.count() / 1e6f;
}


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

std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    os << "Ray(p=" << ray.p << ", v=" << ray.v << ")";
    return os;
}